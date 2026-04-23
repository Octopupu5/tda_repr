from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from InquirerPy import inquirer
from tqdm import tqdm

from tda_repr.data import get_dataset, make_dataloaders
from tda_repr.models import SelectionValidationError, csv_to_list, list_module_names, select_names, set_trainable_by_name_selection
from tda_repr.training import BenchmarkSpec, ExperimentTracker, RepresentationMonitor, RepresentationMonitorConfig, RunStore, TrackerConfig
from tda_repr.viz.runlog import get_series, load_epoch_end_records
from tools.correlation_report import generate_correlation_report
from tools._shared import (
	build_cv_model,
	build_text_model,
	default_device_string,
	ensure_dir,
	infer_cv_input_flat_dim,
	infer_num_classes,
	is_text_batch,
	move_to_device,
	resolve_tokenizer_name,
	seed_everything,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Run one experiment and log artifacts to runs/.")
	ap.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--task", choices=["cv", "nlp"], default="cv")
	ap.add_argument("--dataset", default="cifar10")
	ap.add_argument("--model", default="resnet18")
	ap.add_argument("--device", default=default_device_string())
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--runs_base", default="runs/exp", help="Base path prefix for run directories.")
	ap.add_argument("--data_root", default="./data")
	ap.add_argument("--download", action="store_true")

	ap.add_argument("--epochs", type=int, default=5)
	ap.add_argument("--batch_size", type=int, default=64)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--weight_decay", type=float, default=1e-4)
	ap.add_argument("--pretrained", action="store_true")

	ap.add_argument("--max_train_batches", type=int, default=0, help="0 = no limit")
	ap.add_argument("--max_val_batches", type=int, default=0, help="0 = no limit")
	ap.add_argument("--nlp_max_train_examples", type=int, default=0, help="0 = no limit (NLP only)")
	ap.add_argument("--nlp_max_val_examples", type=int, default=0, help="0 = no limit (NLP only)")
	ap.add_argument("--nlp_max_test_examples", type=int, default=0, help="0 = no limit (NLP only)")
	ap.add_argument("--nlp_subset_seed", type=int, default=0, help="Seed for deterministic NLP subsetting")

	ap.add_argument("--finetune", choices=["full", "last_linear", "last_n", "selected_layers"], default="full")
	ap.add_argument("--last_n_params", type=int, default=200)
	ap.add_argument("--train_layers", default="", help="CSV module names to unfreeze (for --finetune selected_layers).")
	ap.add_argument("--train_strict", action=argparse.BooleanOptionalAction, default=True)

	ap.add_argument("--layers", default="", help="CSV module names to hook; empty = defaults for model.")
	ap.add_argument("--layer_include", default="", help="CSV patterns to select hook layers from model modules.")
	ap.add_argument("--layer_exclude", default="", help="CSV patterns to exclude hook layers.")
	ap.add_argument("--layer_regex", action="store_true")
	ap.add_argument("--list_layers_only", action="store_true")
	ap.add_argument("--list_layers_leaf_only", action="store_true")
	ap.add_argument("--strict_validation", action=argparse.BooleanOptionalAction, default=True)

	ap.add_argument("--compute_mtopdiv", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--compute_q1_spectra", action=argparse.BooleanOptionalAction, default=True)

	ap.add_argument("--early_stop", action="store_true")
	ap.add_argument("--early_stop_layer", default="")
	ap.add_argument("--early_stop_metric", default="")
	ap.add_argument("--early_stop_mode", choices=["min", "max"], default="max")
	ap.add_argument(
		"--early_stop_signals",
		default="",
		help="Semicolon-separated signals: layer:metric[:mode].",
	)
	ap.add_argument("--early_stop_aggregate", choices=["any", "all"], default="any")
	ap.add_argument("--early_stop_patience", type=int, default=5)
	ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
	ap.add_argument("--early_stop_start_epoch", type=int, default=0)

	ap.add_argument("--plot_every", type=int, default=1, help="Epoch interval for writing figures.")
	ap.add_argument(
		"--write_correlation_report",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Write correlations_report/* from metrics.jsonl at end of run.",
	)
	ap.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--bench_metric", default="f1_macro", help="Metric key used to select best_main checkpoint.")
	ap.add_argument(
		"--nlp_objective",
		choices=["classification", "generation"],
		default="classification",
		help="NLP objective; smol-summarize defaults to generation.",
	)
	ap.add_argument(
		"--bench_metrics",
		default="",
		help="CSV metrics for benchmark logging. Empty = defaults by objective.",
	)
	return ap.parse_args(argv)


def _tui_require_tty() -> None:
	if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
		raise RuntimeError("Interactive mode requires a TTY. Re-run with --no-interactive.")


def _tui_int(message: str, default: int, *, min_value: int = 1) -> int:
	def _validate(x: str) -> bool:
		try:
			v = int(str(x).strip())
		except Exception:
			return False
		return v >= int(min_value)

	raw = inquirer.text(message=message, default=str(int(default)), validate=_validate).execute()
	return int(str(raw).strip())


def _tui_select_or_custom(message: str, *, choices: list[str], default: str) -> str:
	items = list(choices)
	custom_key = "<custom>"
	if custom_key not in items:
		items.append(custom_key)
	ch = inquirer.select(message=message, choices=items, default=default if default in items else items[0]).execute()
	if str(ch) == custom_key:
		out = inquirer.text(message=f"{message} (custom)", default=str(default)).execute()
		return str(out).strip()
	return str(ch).strip()


def _interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
	_tui_require_tty()

	args.task = inquirer.select(message="Task", choices=["cv", "nlp"], default=str(args.task)).execute()

	cv_datasets = [
		"cifar10",
		"mnist",
		"imagenette",
		"bloodmnist",
		"pathmnist",
	]
	nlp_datasets = ["trec6", "sst2", "smol-summarize"]
	args.dataset = _tui_select_or_custom(
		"Dataset",
		choices=cv_datasets if str(args.task) == "cv" else nlp_datasets,
		default=str(args.dataset),
	)

	cv_models = ["resnet18", "efficientnet_b0", "convnext_tiny", "mlp"]
	nlp_models = ["distilbert-base-uncased", "smollm"]
	args.model = _tui_select_or_custom(
		"Model",
		choices=cv_models if str(args.task) == "cv" else nlp_models,
		default=str(args.model),
	)

	args.device = str(inquirer.text(message="Device", default=str(args.device)).execute()).strip()
	args.pretrained = bool(inquirer.confirm(message="Use pretrained weights?", default=bool(args.pretrained)).execute())
	args.download = bool(inquirer.confirm(message="Download datasets if missing?", default=bool(args.download)).execute())

	args.epochs = _tui_int("Epochs", int(args.epochs), min_value=1)
	args.batch_size = _tui_int("Batch size", int(args.batch_size), min_value=1)

	args.finetune = inquirer.select(
		message="Finetune mode",
		choices=["full", "last_linear", "last_n", "selected_layers"],
		default=str(args.finetune),
	).execute()
	if str(args.finetune).strip() == "last_n":
		args.last_n_params = _tui_int("last_n_params", int(args.last_n_params), min_value=1)

	args.compute_mtopdiv = bool(inquirer.confirm(message="Compute MTopDiv?", default=bool(args.compute_mtopdiv)).execute())
	args.compute_q1_spectra = bool(inquirer.confirm(message="Compute q1 spectra?", default=bool(args.compute_q1_spectra)).execute())

	args.early_stop = bool(inquirer.confirm(message="Enable early-stop?", default=bool(args.early_stop)).execute())
	if bool(args.early_stop):
		args.early_stop_signals = str(
			inquirer.text(
				message="Early-stop signals (layer:metric[:mode]; separate by ';')",
				default=str(args.early_stop_signals or "").strip(),
			).execute()
		).strip()
		if not str(args.early_stop_signals).strip():
			args.early_stop_layer = str(inquirer.text(message="Early-stop layer", default=str(args.early_stop_layer)).execute()).strip()
			args.early_stop_metric = str(inquirer.text(message="Early-stop metric", default=str(args.early_stop_metric)).execute()).strip()
			args.early_stop_mode = inquirer.select(message="Early-stop mode", choices=["min", "max"], default=str(args.early_stop_mode)).execute()
	return args


def _interactive_pick_hook_layers(model: nn.Module, args: argparse.Namespace, defaults: list[str]) -> None:
	_tui_require_tty()
	mode = inquirer.select(
		message="Hook layers selection",
		choices=[
			"Use defaults",
			"Pick from defaults",
			"Pick from all modules",
			"Use include/exclude patterns",
		],
		default="Use defaults",
	).execute()

	args.layers = ""
	args.layer_include = ""
	args.layer_exclude = ""
	args.layer_regex = False

	if str(mode) == "Use defaults":
		return

	if str(mode) == "Use include/exclude patterns":
		args.layer_include = str(inquirer.text(message="Include patterns (CSV, glob)", default=str(args.layer_include)).execute()).strip()
		args.layer_exclude = str(inquirer.text(message="Exclude patterns (CSV, glob)", default=str(args.layer_exclude)).execute()).strip()
		args.layer_regex = bool(inquirer.confirm(message="Treat patterns as regex?", default=bool(args.layer_regex)).execute())
		return

	if str(mode) == "Pick from defaults":
		choices = [{"name": n, "value": n, "enabled": True} for n in defaults]
		picked = inquirer.checkbox(message="Pick hook layers", choices=choices).execute()
		if not picked:
			raise SelectionValidationError("No hook layers selected.")
		args.layers = ",".join([str(x) for x in picked])
		return

	all_mods = list_module_names(model, leaf_only=True)
	sub = str(inquirer.text(message="Filter substring for module list (empty = all)", default="").execute()).strip()
	if sub:
		all_mods = [m for m in all_mods if sub.lower() in str(m).lower()]
	if not all_mods:
		raise SelectionValidationError("No modules matched the filter; cannot pick hook layers.")
	choices = [{"name": n, "value": n, "enabled": n in set(defaults)} for n in all_mods]
	picked = inquirer.checkbox(message="Pick hook layers", choices=choices).execute()
	if not picked:
		raise SelectionValidationError("No hook layers selected.")
	args.layers = ",".join([str(x) for x in picked])


def _unfreeze_last_linear(model: nn.Module) -> None:
	last: Optional[nn.Linear] = None
	for m in model.modules():
		if isinstance(m, nn.Linear):
			last = m
	if last is None:
		return
	for p in last.parameters():
		p.requires_grad = True


def _unfreeze_last_n_params(model: nn.Module, n: int) -> None:
	params = list(model.parameters())
	for p in params:
		p.requires_grad = False
	for p in params[-max(int(n), 1) :]:
		p.requires_grad = True


def _apply_finetune(model: nn.Module, args: argparse.Namespace) -> None:
	mode = str(args.finetune).lower().strip()
	if mode == "full":
		for p in model.parameters():
			p.requires_grad = True
		return
	for p in model.parameters():
		p.requires_grad = False
	if mode == "last_linear":
		_unfreeze_last_linear(model)
		return
	if mode == "last_n":
		_unfreeze_last_n_params(model, int(args.last_n_params))
		return
	if mode == "selected_layers":
		names = csv_to_list(str(args.train_layers))
		if not names:
			raise SelectionValidationError("finetune=selected_layers requires --train_layers.")
		rep = set_trainable_by_name_selection(model, include=names, exclude=(), use_regex=False, strict=bool(args.train_strict))
		if not rep.selected:
			raise SelectionValidationError("finetune=selected_layers selected no trainable parameters.")
		return
	raise ValueError(f"Unknown finetune mode: {mode}")


def _pick_hook_layers(model: nn.Module, defaults: list[str], args: argparse.Namespace) -> list[str]:
	explicit = csv_to_list(str(args.layers))
	if explicit:
		return explicit
	all_mods = list_module_names(model, leaf_only=bool(args.list_layers_leaf_only))
	inc = csv_to_list(str(args.layer_include))
	exc = csv_to_list(str(args.layer_exclude))
	if inc or exc:
		rep = select_names(all_mods, include=inc, exclude=exc, use_regex=bool(args.layer_regex))
		if bool(args.strict_validation) and rep.unmatched:
			raise SelectionValidationError(f"Unmatched layer patterns: {rep.unmatched}")
		return rep.selected
	return defaults


def _extract_scalar(d: Any) -> Optional[float]:
	try:
		x = float(d)
	except Exception:
		return None
	return x if np.isfinite(x) else None


def _repr_value(epoch_out: Mapping[str, Any], layer: str, metric: str) -> Optional[float]:
	rep = epoch_out.get("repr", {}) if isinstance(epoch_out, dict) else {}
	layers = rep.get("layers", {}) if isinstance(rep, dict) else {}
	blk = layers.get(str(layer), None) if isinstance(layers, dict) else None
	val = blk.get(str(metric), None) if isinstance(blk, dict) else None
	return _extract_scalar(val)


def _update_plateau(
	state: Dict[str, Any],
	*,
	value: float,
	mode: str,
	min_delta: float,
) -> None:
	best = state.get("best", None)
	if best is None:
		state["best"] = float(value)
		state["bad"] = 0
		return
	best_v = float(best)
	if str(mode) == "min":
		improved = float(value) < (best_v - float(min_delta))
	else:
		improved = float(value) > (best_v + float(min_delta))
	if improved:
		state["best"] = float(value)
		state["bad"] = 0
	else:
		state["bad"] = int(state.get("bad", 0)) + 1


def _parse_early_signals(args: argparse.Namespace) -> list[Dict[str, str]]:
	out: list[Dict[str, str]] = []
	if str(args.early_stop_signals).strip():
		raw = [x.strip() for x in str(args.early_stop_signals).split(";") if x.strip()]
		for part in raw:
			p = [x.strip() for x in part.split(":") if x.strip()]
			if len(p) < 2:
				raise ValueError(f"Bad early_stop_signals entry: {part!r}")
			lay, met = p[0], p[1]
			mode = p[2] if len(p) >= 3 else str(args.early_stop_mode)
			out.append({"layer": lay, "metric": met, "mode": mode})
	elif str(args.early_stop_layer).strip() and str(args.early_stop_metric).strip():
		out.append({"layer": str(args.early_stop_layer).strip(), "metric": str(args.early_stop_metric).strip(), "mode": str(args.early_stop_mode)})
	return out


def _load_stop_epoch_for_plots(run_dir: str) -> Optional[int]:
	analysis_dir = os.path.join(os.path.abspath(run_dir), "analysis")
	p_online = os.path.join(analysis_dir, "early_stop_online.json")
	if os.path.isfile(p_online):
		try:
			with open(p_online, "r", encoding="utf-8") as f:
				obj = json.load(f)
			ep = obj.get("trigger_epoch")
			return int(ep) if isinstance(ep, int) else None
		except Exception:
			return None

	p_offline = os.path.join(analysis_dir, "early_stop_offline.json")
	if os.path.isfile(p_offline):
		try:
			with open(p_offline, "r", encoding="utf-8") as f:
				obj = json.load(f)
			ep = obj.get("trigger_epoch")
			return int(ep) if isinstance(ep, int) else None
		except Exception:
			return None

	p_sweep = os.path.join(analysis_dir, "repr_early_stop_sweep.json")
	if os.path.isfile(p_sweep):
		try:
			with open(p_sweep, "r", encoding="utf-8") as f:
				obj = json.load(f)
			best = obj.get("best") if isinstance(obj, dict) else None
			if isinstance(best, dict):
				ep = best.get("effective_stop_epoch", None)
				return int(ep) if isinstance(ep, int) else None
		except Exception:
			return None
	return None


def _write_figures(run_dir: str, *, plot_every: int, dataset_key: str, hook_layers: list[str]) -> None:
	if int(plot_every) <= 0:
		return
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.isfile(metrics_path):
		return
	recs = load_epoch_end_records(metrics_path)
	if not recs:
		return

	fig_dir = ensure_dir(os.path.join(run_dir, "figures"))
	keep = {"fig_quality_progress.png", "fig_repr_progress.png"}
	for fn in os.listdir(fig_dir):
		if fn.endswith(".png") and fn not in keep:
			os.remove(os.path.join(fig_dir, fn))

	stop_epoch = _load_stop_epoch_for_plots(run_dir)

	bench_prefix = f"bench.{str(dataset_key).strip()}-val"
	series_specs = [
		("Validation accuracy", f"{bench_prefix}.accuracy"),
		("Validation F1 (macro)", f"{bench_prefix}.f1_macro"),
		("Validation loss", f"{bench_prefix}.loss"),
	]
	fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.6), dpi=100)
	for ax, (title, key) in zip(axs, series_specs, strict=True):
		ser = get_series(recs, key)
		xs = [int(e) for e, _v in ser]
		ys = [float(v) for _e, v in ser]
		ax.plot(xs, ys, marker="o", linewidth=1.4)
		ax.set_title(title)
		ax.set_xlabel("epoch")
		if stop_epoch is not None:
			ax.axvline(float(stop_epoch), color="red", linestyle="--", linewidth=1.2, alpha=0.9)
			for x, y in zip(xs, ys, strict=False):
				if int(x) == int(stop_epoch):
					ax.scatter([x], [y], s=60, facecolors="none", edgecolors="red", linewidths=2.0, zorder=5)
		ax.grid(True, alpha=0.2, linestyle="--")
	fig.tight_layout()
	fig.savefig(os.path.join(fig_dir, "fig_quality_progress.png"))
	plt.close(fig)

	metric = "mtopdiv_train_val"
	head_n, tail_n = 4, 5
	if len(hook_layers) <= 9:
		plot_layers = list(hook_layers)
	else:
		head = list(hook_layers[: int(head_n)])
		tail = list(hook_layers[-int(tail_n) :])
		plot_layers = tail + head

	fig, axs = plt.subplots(3, 3, figsize=(14.1, 8.69), dpi=100)
	axs_flat = [axs[r][c] for r in range(3) for c in range(3)]
	for ax in axs_flat:
		ax.axis("off")

	for i, layer in enumerate(plot_layers[:9]):
		ax = axs_flat[i]
		ax.axis("on")
		key = f"repr.layers.{layer}.{metric}"
		ser = get_series(recs, key)
		xs = [int(e) for e, _v in ser]
		ys = [float(v) for _e, v in ser]
		ax.plot(xs, ys, marker="o", linewidth=1.4)
		ax.set_title(f"{layer}: {metric}", fontsize=10)
		ax.set_xlabel("epoch")
		if stop_epoch is not None:
			ax.axvline(float(stop_epoch), color="red", linestyle="--", linewidth=1.2, alpha=0.9)
			for x, y in zip(xs, ys, strict=False):
				if int(x) == int(stop_epoch):
					ax.scatter([x], [y], s=60, facecolors="none", edgecolors="red", linewidths=2.0, zorder=5)
		ax.grid(True, alpha=0.2, linestyle="--")

	fig.tight_layout()
	fig.savefig(os.path.join(fig_dir, "fig_repr_progress.png"))
	plt.close(fig)


def _save_ckpt(path: str, model: nn.Module) -> None:
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	try:
		obj = model.state_dict()
	except Exception as e:
		raise RuntimeError("Failed to read model.state_dict().") from e
	try:
		torch.save(obj, path)
	except Exception as e:
		raise RuntimeError(f"Failed to torch.save() checkpoint: {path}") from e


def _infer_batch_size(batch: Any) -> Optional[int]:
	if isinstance(batch, (tuple, list)) and batch:
		x = batch[0]
		if hasattr(x, "shape") and getattr(x, "shape", None) is not None and len(getattr(x, "shape")) >= 1:
			return int(x.shape[0])
	if isinstance(batch, dict):
		for k in ("input_ids", "pixel_values", "x", "images", "image"):
			x = batch.get(k)
			if hasattr(x, "shape") and getattr(x, "shape", None) is not None and len(getattr(x, "shape")) >= 1:
				return int(x.shape[0])
		for x in batch.values():
			if hasattr(x, "shape") and getattr(x, "shape", None) is not None and len(getattr(x, "shape")) >= 1:
				return int(x.shape[0])
	return None


def _train_epoch_cv(
	model: nn.Module,
	monitor: RepresentationMonitor,
	loader: Any,
	*,
	device: torch.device,
	optimizer: torch.optim.Optimizer,
	loss_fn: nn.Module,
	preprocess: Optional[Any],
	max_batches: int,
	epoch: int,
) -> None:
	model.train()
	with monitor.attach(model):
		n_total = len(loader) if hasattr(loader, "__len__") else None
		total = None
		if n_total is not None:
			total = int(n_total) if int(max_batches) <= 0 else min(int(n_total), int(max_batches))
		pbar = tqdm(enumerate(loader), total=total, desc=f"train(e{int(epoch)})", leave=False)
		seen = 0
		loss_sum = 0.0
		loss_steps = 0
		for bi, batch in pbar:
			if int(max_batches) > 0 and bi >= int(max_batches):
				break
			x, y = batch
			bs = _infer_batch_size(batch)
			x = move_to_device(x, device)
			y = move_to_device(y, device)
			if isinstance(y, torch.Tensor) and y.dim() > 1:
				if y.shape[-1] == 1:
					y = y.view(-1)
				else:
					raise RuntimeError(
						f"Expected single-label targets for CrossEntropyLoss, got y.shape={tuple(y.shape)}. "
						"Multi-label datasets require a different loss/metrics setup."
					)
			if isinstance(y, torch.Tensor) and y.dtype != torch.long:
				y = y.long()
			if preprocess is not None:
				x = preprocess(x)
			optimizer.zero_grad(set_to_none=True)
			logits = model(x)
			loss = loss_fn(logits, y)
			loss.backward()
			optimizer.step()
			monitor.collect("train")
			lv = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
			loss_sum += lv
			loss_steps += 1
			if bs is not None:
				seen += int(bs)
			avg = float(loss_sum) / float(max(loss_steps, 1))
			pbar.set_postfix(loss=f"{lv:.4g}", avg=f"{avg:.4g}", seen=str(int(seen)))


def _train_epoch_nlp(
	model: nn.Module,
	monitor: RepresentationMonitor,
	loader: Any,
	*,
	device: torch.device,
	optimizer: torch.optim.Optimizer,
	max_batches: int,
	epoch: int,
) -> None:
	model.train()
	with monitor.attach(model):
		n_total = len(loader) if hasattr(loader, "__len__") else None
		total = None
		if n_total is not None:
			total = int(n_total) if int(max_batches) <= 0 else min(int(n_total), int(max_batches))
		pbar = tqdm(enumerate(loader), total=total, desc=f"train(e{int(epoch)})", leave=False)
		seen = 0
		loss_sum = 0.0
		loss_steps = 0
		for bi, batch in pbar:
			if int(max_batches) > 0 and bi >= int(max_batches):
				break
			bs = _infer_batch_size(batch)
			b = move_to_device(batch, device)
			optimizer.zero_grad(set_to_none=True)
			out = model(**b)
			loss = getattr(out, "loss", None)
			if loss is None:
				logits = getattr(out, "logits", None)
				labels = b.get("labels") if isinstance(b, dict) else None
				if logits is None or labels is None:
					raise RuntimeError("NLP batch must provide model.loss or logits+labels.")
				loss = nn.CrossEntropyLoss()(logits, labels)
			loss.backward()
			optimizer.step()
			monitor.collect("train")
			lv = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
			loss_sum += lv
			loss_steps += 1
			if bs is not None:
				seen += int(bs)
			avg = float(loss_sum) / float(max(loss_steps, 1))
			pbar.set_postfix(loss=f"{lv:.4g}", avg=f"{avg:.4g}", seen=str(int(seen)))


def _collect_val(
	model: nn.Module,
	monitor: RepresentationMonitor,
	loader: Any,
	*,
	device: torch.device,
	preprocess: Optional[Any],
	max_batches: int,
	epoch: int,
) -> None:
	model.eval()
	with torch.no_grad(), monitor.attach(model):
		n_total = len(loader) if hasattr(loader, "__len__") else None
		total = None
		if n_total is not None:
			total = int(n_total) if int(max_batches) <= 0 else min(int(n_total), int(max_batches))
		pbar = tqdm(enumerate(loader), total=total, desc=f"val(e{int(epoch)})", leave=False)
		seen = 0
		for bi, batch in pbar:
			if int(max_batches) > 0 and bi >= int(max_batches):
				break
			bs = _infer_batch_size(batch)
			if is_text_batch(batch):
				_ = model(**move_to_device(batch, device))
			else:
				x, _y = batch
				x = move_to_device(x, device)
				if preprocess is not None:
					x = preprocess(x)
				_ = model(x)
			monitor.collect("val")
			if bs is not None:
				seen += int(bs)
			pbar.set_postfix(seen=str(int(seen)))


def main(argv: Optional[list[str]] = None) -> str:
	args = _parse_args(argv)
	if bool(args.interactive):
		args = _interactive_fill(args)

	if str(args.task) == "nlp" and str(args.dataset).strip().lower() in {"smol-summarize", "smol_summarize", "smol_summarize_v0"}:
		args.nlp_objective = "generation"
		if str(args.bench_metric).strip().lower() == "f1_macro":
			args.bench_metric = "loss_assistant_only"

	seed_everything(int(args.seed))
	device = torch.device(str(args.device))

	tokenizer_name = resolve_tokenizer_name(str(args.model)) if str(args.task) == "nlp" else "distilbert-base-uncased"
	bundle = get_dataset(
		str(args.dataset),
		root=str(args.data_root),
		download=bool(args.download),
		tokenizer_name=str(tokenizer_name),
		nlp_max_train_examples=int(args.nlp_max_train_examples),
		nlp_max_val_examples=int(args.nlp_max_val_examples),
		nlp_max_test_examples=int(args.nlp_max_test_examples),
		nlp_subset_seed=int(args.nlp_subset_seed),
	)
	loaders = make_dataloaders(bundle, batch_size=int(args.batch_size), num_workers=0)
	train_loader = loaders.get("train")
	val_loader = loaders.get("val") or loaders.get("test")
	if train_loader is None or val_loader is None:
		raise RuntimeError(f"Dataset '{args.dataset}' did not provide required splits (train/val or train/test).")

	ds_for_nc = bundle.train or bundle.val or bundle.test
	num_classes = 0
	if not (str(args.task) == "nlp" and str(args.nlp_objective) == "generation"):
		num_classes = infer_num_classes(str(args.dataset), ds_for_nc)

	preprocess = None
	if str(args.task) == "cv":
		input_flat_dim = infer_cv_input_flat_dim(ds_for_nc)
		model, preprocess, default_layers = build_cv_model(
			str(args.model),
			num_classes=int(num_classes),
			device=device,
			pretrained=bool(args.pretrained),
			input_flat_dim=input_flat_dim,
		)
	else:
		model, default_layers = build_text_model(
			str(args.model),
			num_classes=int(num_classes),
			device=device,
			pretrained=bool(args.pretrained),
			objective=str(args.nlp_objective),
		)

	if bool(args.interactive):
		_interactive_pick_hook_layers(model, args, list(default_layers))

	if bool(args.list_layers_only):
		all_mods = list_module_names(model, leaf_only=bool(args.list_layers_leaf_only))
		for n in all_mods:
			print(n)
			return ""

	_apply_finetune(model, args)
	hook_layers = _pick_hook_layers(model, default_layers, args)
	if not hook_layers:
		raise SelectionValidationError("No hook layers selected.")

	monitor_cfg = RepresentationMonitorConfig(
		layer_names=hook_layers,
		max_samples_per_stage=1024,
		seq_pooling="first_token",
		max_points_for_graph=256,
		max_points_for_mtopdiv=768,
		max_eigs=10,
		knn_k_small=5,
		knn_k_large=15,
		graph_stage="train",
		fixed_graph_points=True,
		fixed_graph_seed=0,
		build_triangles=True,
		max_triangles=50000,
		zero_tol=1e-8,
		regularization=1e-10,
		compute_hodge=True,
		compute_persistent=True,
		compute_mtopdiv=bool(args.compute_mtopdiv),
		compute_q1_spectra=bool(args.compute_q1_spectra),
		mtopdiv_runs=2,
		mtopdiv_subset_size=512,
		mtopdiv_outer_runs=10,
		mtopdiv_pdist_device="cpu",
		mtopdiv_stage_a="train",
		mtopdiv_stage_b="val",
		fixed_mtopdiv_points=True,
		fixed_mtopdiv_seed=0,
		compute_gudhi=False,
		verbose=True,
	)
	monitor = RepresentationMonitor(monitor_cfg)

	store = RunStore(str(args.runs_base), suffix=f"{args.task}_{args.dataset}_{args.model}_ft-{args.finetune}", unique=True)
	bench_kind = "classification"
	bench_metrics = ("loss", "accuracy", "f1_macro")
	paper_bench_metric = str(args.bench_metric).strip().lower()
	if str(args.task) == "nlp" and str(args.nlp_objective) == "generation":
		bench_kind = "generation"
		bench_metrics = ("loss", "loss_assistant_only", "ppl")
		paper_bench_metric = "loss_assistant_only"
	if str(args.bench_metrics).strip():
		bench_metrics = tuple(x.strip() for x in csv_to_list(str(args.bench_metrics)) if str(x).strip())
		if not bench_metrics:
			raise ValueError("--bench_metrics must be a non-empty CSV when provided.")
	if paper_bench_metric not in set(bench_metrics):
		if str(args.task) == "nlp" and str(args.nlp_objective) == "generation":
			paper_bench_metric = "loss_assistant_only"
		else:
			paper_bench_metric = str(args.bench_metric).strip().lower() or "f1_macro"

	store.write_meta(
		{
			"name": "exp",
			"task": str(args.task),
			"dataset": str(args.dataset),
			"model": str(args.model),
			"finetune": str(args.finetune),
			"device": str(args.device),
			"num_classes": int(num_classes),
			"monitor": asdict(monitor_cfg),
			"_paper_bench_metric": str(paper_bench_metric),
			"args": vars(args),
		}
	)

	checkpoints_dir = ensure_dir(os.path.join(store.run_dir, "checkpoints"))
	best_ckpt_path = os.path.join(checkpoints_dir, "model_best_main.pt")
	last_ckpt_path = os.path.join(checkpoints_dir, "model_last.pt")
	early_ckpt_path = os.path.join(checkpoints_dir, "model_early_signal.pt")

	bench_specs = [
		BenchmarkSpec(name=f"{args.dataset}-val", dataloader_key="val", kind=bench_kind, metrics=bench_metrics),
		BenchmarkSpec(name=f"{args.dataset}-test", dataloader_key="test", kind=bench_kind, metrics=bench_metrics),
	]
	tracker = ExperimentTracker(
		monitor=monitor,
		benchmarks=bench_specs,
		store=store,
		cfg=TrackerConfig(run_dir=store.run_dir, eval_every=1, max_eval_batches=(int(args.max_val_batches) or None)),
	)

	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
	loss_fn = nn.CrossEntropyLoss()

	signals = _parse_early_signals(args) if bool(args.early_stop) else []
	states: list[Dict[str, Any]] = [
		{"layer": s["layer"], "metric": s["metric"], "mode": s["mode"], "best": None, "bad": 0} for s in signals
	]
	trigger_epoch: Optional[int] = None
	best_val: Optional[float] = None

	for epoch in tqdm(range(int(args.epochs)), desc="epochs"):
		monitor.reset_epoch()
		t_train0 = time.perf_counter()
		if str(args.task) == "cv":
			_train_epoch_cv(
				model,
				monitor,
				train_loader,
				device=device,
				optimizer=opt,
				loss_fn=loss_fn,
				preprocess=preprocess,
				max_batches=int(args.max_train_batches),
				epoch=int(epoch),
			)
		else:
			_train_epoch_nlp(
				model,
				monitor,
				train_loader,
				device=device,
				optimizer=opt,
				max_batches=int(args.max_train_batches),
				epoch=int(epoch),
			)
		train_s = float(time.perf_counter() - t_train0)

		t_val0 = time.perf_counter()
		_collect_val(
			model,
			monitor,
			val_loader,
			device=device,
			preprocess=preprocess,
			max_batches=int(args.max_val_batches),
			epoch=int(epoch),
		)
		val_s = float(time.perf_counter() - t_val0)

		out = tracker.on_epoch_end(
			int(epoch),
			model=model,
			dataloaders={"train": train_loader, "val": val_loader, "test": loaders.get("test")},
			loss_fn=loss_fn,
			preprocess=preprocess,
			extra={"train_s": train_s, "val_s": val_s},
		)

		bench_name = f"{args.dataset}-val"
		bench_row = (out.get("bench") or {}).get(bench_name) if isinstance(out, dict) else None
		val_metric = None
		if isinstance(bench_row, dict):
			val_metric = _extract_scalar(bench_row.get(str(paper_bench_metric)))
			if val_metric is None:
				val_metric = (
					_extract_scalar(bench_row.get("f1_macro"))
					or _extract_scalar(bench_row.get("accuracy"))
					or _extract_scalar(bench_row.get("loss_assistant_only"))
					or _extract_scalar(bench_row.get("loss"))
					or _extract_scalar(bench_row.get("ppl"))
				)
		if val_metric is not None:
			mode = "min" if str(paper_bench_metric) in {"loss", "ppl", "loss_assistant_only"} else "max"
			if best_val is None:
				best_val = float(val_metric)
				if bool(args.save_checkpoints):
					_save_ckpt(best_ckpt_path, model)
			else:
				is_better = float(val_metric) < float(best_val) if mode == "min" else float(val_metric) > float(best_val)
				if bool(is_better):
					best_val = float(val_metric)
					if bool(args.save_checkpoints):
						_save_ckpt(best_ckpt_path, model)

		if bool(args.save_checkpoints):
			_save_ckpt(last_ckpt_path, model)

		if states and trigger_epoch is None and int(epoch) >= int(args.early_stop_start_epoch):
			cur_vals: list[Optional[float]] = []
			for st in states:
				v = _repr_value(out, st["layer"], st["metric"])
				cur_vals.append(v)
				if v is not None:
					_update_plateau(st, value=float(v), mode=str(st["mode"]), min_delta=float(args.early_stop_min_delta))
			bads = [int(st.get("bad", 0)) >= int(args.early_stop_patience) for st in states]
			cond = any(bads) if str(args.early_stop_aggregate) == "any" else all(bads)
			if cond and any(v is not None for v in cur_vals):
				trigger_epoch = int(epoch)
				out_path = os.path.join(store.run_dir, "analysis", "early_stop_online.json")
				ensure_dir(os.path.dirname(out_path))
				with open(out_path, "w", encoding="utf-8") as f:
					json.dump(
						{
							"trigger_epoch": int(trigger_epoch),
							"signals": [{"layer": st["layer"], "metric": st["metric"], "mode": st["mode"]} for st in states],
						},
						f,
						indent=2,
					)
				if bool(args.save_checkpoints):
					_save_ckpt(early_ckpt_path, model)

		if int(args.plot_every) > 0 and (int(epoch) % int(args.plot_every) == 0):
			_write_figures(store.run_dir, plot_every=int(args.plot_every), dataset_key=str(args.dataset), hook_layers=list(hook_layers))

	if bool(args.write_correlation_report):
		out = generate_correlation_report(run_dir=str(store.run_dir))
		print(f"[CorrelationReport] saved to: {out['out_dir']} pairs={out['pairs']} top={out['top']}")

	print(f"[Done] run_dir={store.run_dir}")
	return str(store.run_dir)


if __name__ == "__main__":
	main()
