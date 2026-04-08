import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from tqdm import tqdm

# Allow running as a script: ensure project root (parent of /tools) is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from tda_repr.data import get_dataset, make_dataloaders
from tda_repr.models import (
	LayerTaps,
	SelectionValidationError,
	csv_to_list,
	get_model_info,
	list_module_names,
	select_names,
	set_trainable_by_name_selection,
)
from tda_repr.training import BenchmarkSpec, ExperimentTracker, RepresentationMonitor, RepresentationMonitorConfig, RunStore, TrackerConfig
from tda_repr.viz.runlog import get_series, list_scalar_series_keys, load_epoch_end_records


def _known_datasets_for_task(task: str) -> List[str]:
	task = str(task).lower().strip()
	if task == "cv":
		return [
			"mnist",
			"cifar10",
			"imagenette",
			"pathmnist",
			"chestmnist",
			"bloodmnist",
			"dermamnist",
			"retinamnist",
		]
	return ["sst2", "ag_news", "trec6", "yahoo_answers_topics"]


def _known_models_for_task(task: str) -> List[str]:
	task = str(task).lower().strip()
	if task == "cv":
		return ["resnet18", "convnext_tiny", "efficientnet_b0", "mlp"]
	return ["distilbert", "smollm2-135m", "smollm2-360m"]


def _require_transformers_runexp() -> object:
	"""
	Lazily import `transformers` for NLP experiments.
	Kept local to avoid making CV-only runs depend on optional NLP deps.
	"""
	try:
		import transformers  # type: ignore
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"Optional dependency `transformers` is required for task='nlp'. "
			"Install it with `pip install transformers` (or `pip install .[nlp]`)."
		) from e
	return transformers


def _pick_transformer_blocks_generic_for_model(model: nn.Module) -> List[str]:
	"""
	Best-effort taps for transformer models based on module names.
	Works for DistilBERT-like models and many decoder-only LMs.
	"""
	all_named = set(dict(model.named_modules()).keys())

	# Pick an embedding-like module if present.
	out: List[str] = []
	for emb_name in ("embeddings", "model.embed_tokens", "embed_tokens", "wte"):
		if emb_name in all_named:
			out.append(emb_name)
			break

	# Prefer the prefix that yields the most numeric block indices.
	prefixes = [
		"distilbert.transformer.layer",
		"transformer.layer",
		"model.layers",
		"encoder.layer",
		"layers",
	]
	best_prefix = ""
	best_ids: List[int] = []
	for p in prefixes:
		ids: List[int] = []
		for name in all_named:
			if not name.startswith(p + "."):
				continue
			rest = name[len(p) + 1 :]
			head = rest.split(".", 1)[0]
			if head.isdigit():
				ids.append(int(head))
		ids = sorted(set(ids))
		if len(ids) > len(best_ids):
			best_ids = ids
			best_prefix = p

	if best_prefix and best_ids:
		n = len(best_ids)
		picks = sorted({best_ids[0], best_ids[n // 3], best_ids[(2 * n) // 3], best_ids[-1]})
		out.extend([f"{best_prefix}.{i}" for i in picks])

	# Keep only existing module paths and preserve order.
	existing = set(dict(model.named_modules()).keys())
	out2: List[str] = []
	seen = set()
	for n in out:
		if n in existing and n not in seen:
			out2.append(n)
			seen.add(n)
	return out2


def _interactive_config_tui(args: argparse.Namespace) -> argparse.Namespace:
	"""
	Interactive wizard for main experiment configuration.
	Applied before model/dataloader build.
	"""
	BACK = "__back__"
	CUSTOM = "__custom__"

	def _back_choice(step: int) -> List[str]:
		return [BACK] if step > 0 else []

	def _ask_int(msg: str, default_val: int, step: int):
		raw = str(
			inquirer.text(
				message=f"{msg} (type 'b' to go back):",
				default=str(default_val),
			).execute()
		).strip()
		if raw.lower() in ("b", "back"):
			return BACK
		return int(raw)

	def _ask_float(msg: str, default_val: float, step: int):
		raw = str(
			inquirer.text(
				message=f"{msg} (type 'b' to go back):",
				default=str(default_val),
			).execute()
		).strip()
		if raw.lower() in ("b", "back"):
			return BACK
		return float(raw)

	step = 0
	while True:
		if step == 0:
			task = inquirer.select(
				message="Task:",
				choices=["cv", "nlp"],
				default=str(args.task),
			).execute()
			args.task = str(task)
			step += 1
			continue

		if step == 1:
			dataset_choices = _known_datasets_for_task(str(args.task)) + [CUSTOM] + _back_choice(step)
			ds = inquirer.select(
				message="Dataset:",
				choices=dataset_choices,
				default=(args.dataset if args.dataset in dataset_choices else dataset_choices[0]),
			).execute()
			if ds == BACK:
				step -= 1
				continue
			if ds == CUSTOM:
				raw = str(inquirer.text(message="Custom dataset key (or 'b' to go back):", default=str(args.dataset)).execute()).strip()
				if raw.lower() in ("b", "back"):
					continue
				args.dataset = raw
			else:
				args.dataset = str(ds)
			step += 1
			continue

		if step == 2:
			model_choices = _known_models_for_task(str(args.task)) + [CUSTOM] + _back_choice(step)
			m = inquirer.select(
				message="Model:",
				choices=model_choices,
				default=(args.model if args.model in model_choices else model_choices[0]),
			).execute()
			if m == BACK:
				step -= 1
				continue
			if m == CUSTOM:
				raw = str(inquirer.text(message="Custom model key (or 'b' to go back):", default=str(args.model)).execute()).strip()
				if raw.lower() in ("b", "back"):
					continue
				args.model = raw
			else:
				args.model = str(m)
			step += 1
			continue

		if step == 3:
			device_choices = ["cuda:0", "cpu", CUSTOM] + _back_choice(step)
			d = inquirer.select(
				message="Device:",
				choices=device_choices,
				default=(args.device if args.device in device_choices else ("cuda:0" if torch.cuda.is_available() else "cpu")),
			).execute()
			if d == BACK:
				step -= 1
				continue
			if d == CUSTOM:
				raw = str(inquirer.text(message="Custom device string (or 'b' to go back):", default=str(args.device)).execute()).strip()
				if raw.lower() in ("b", "back"):
					continue
				args.device = raw
			else:
				args.device = str(d)
			step += 1
			continue

		if step == 4:
			ft_choices = ["full", "linear_probe", "last_n_params", "named_prefixes", "named_patterns", "selected_layers", "tracked_layers"] + _back_choice(step)
			ft = inquirer.select(
				message="Fine-tune strategy:",
				choices=ft_choices,
				default=(str(args.finetune) if str(args.finetune) in ft_choices else ft_choices[0]),
			).execute()
			if ft == BACK:
				step -= 1
				continue
			args.finetune = str(ft)
			step += 1
			continue

		if step == 5:
			pre = inquirer.select(
				message="Use pretrained backbone?",
				choices=["Yes", "No"] + _back_choice(step),
				default=("Yes" if (str(args.task).lower().strip() == "nlp" or bool(args.pretrained)) else "No"),
			).execute()
			if pre == BACK:
				step -= 1
				continue
			args.pretrained = bool(pre == "Yes")
			step += 1
			continue

		if step == 6:
			dl = inquirer.select(
				message="Allow dataset download?",
				choices=["Yes", "No"] + _back_choice(step),
				default=("Yes" if bool(args.download) else "No"),
			).execute()
			if dl == BACK:
				step -= 1
				continue
			args.download = bool(dl == "Yes")
			step += 1
			continue

		if step == 7:
			v = _ask_int("Epochs", int(args.epochs), step)
			if v == BACK:
				step -= 1
				continue
			args.epochs = int(v)
			step += 1
			continue

		if step == 8:
			v = _ask_int("Batch size", int(args.batch_size), step)
			if v == BACK:
				step -= 1
				continue
			args.batch_size = int(v)
			step += 1
			continue

		if step == 9:
			v = _ask_float("Learning rate", float(args.lr), step)
			if v == BACK:
				step -= 1
				continue
			args.lr = float(v)
			step += 1
			continue

		if step == 10:
			v = _ask_float("Weight decay", float(args.weight_decay), step)
			if v == BACK:
				step -= 1
				continue
			args.weight_decay = float(v)
			step += 1
			continue

		if step == 11:
			v = _ask_int("Max train batches (0=no limit)", int(args.max_train_batches), step)
			if v == BACK:
				step -= 1
				continue
			args.max_train_batches = int(v)
			step += 1
			continue

		if step == 12:
			v = _ask_int("Max val batches (0=no limit)", int(args.max_val_batches), step)
			if v == BACK:
				step -= 1
				continue
			args.max_val_batches = int(v)
			step += 1
			continue

		if step == 13:
			# Optional NLP subsampling knobs (kept simple; shuffle+take-N in code).
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			v = _ask_int("NLP: max train examples (0 = no subsample)", int(getattr(args, "nlp_max_train_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_train_examples = int(v)
			step += 1
			continue

		if step == 14:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			v = _ask_int("NLP: max val examples (0 = no subsample)", int(getattr(args, "nlp_max_val_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_val_examples = int(v)
			step += 1
			continue

		if step == 15:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			v = _ask_int("NLP: max test examples (0 = no subsample)", int(getattr(args, "nlp_max_test_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_test_examples = int(v)
			step += 1
			continue

		if step == 16:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			v = _ask_int("NLP: subsample seed", int(getattr(args, "nlp_subset_seed", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_subset_seed = int(v)
			step += 1
			continue

		if step == 17:
			bt = inquirer.select(
				message="Build triangles (dim=2) for higher-order spectra?",
				choices=["Yes", "No"] + _back_choice(step),
				default=("Yes" if bool(args.build_triangles) else "No"),
			).execute()
			if bt == BACK:
				step -= 1
				continue
			args.build_triangles = bool(bt == "Yes")
			# Keep schedule options consistent.
			if not bool(args.build_triangles):
				args.dim2_every = 0
			step += 1
			continue

		if step == 18:
			# Scheduling knobs for triangle construction (only meaningful when enabled).
			if not bool(args.build_triangles):
				step += 1
				continue
			v = _ask_int("dim=2 every N epochs (0 = every epoch)", int(args.dim2_every), step)
			if v == BACK:
				step -= 1
				continue
			args.dim2_every = int(v)
			step += 1
			continue

		if step == 19:
			if not bool(args.build_triangles):
				step += 1
				continue
			v = _ask_int("Max triangles per layer (safety cap)", int(args.max_triangles), step)
			if v == BACK:
				step -= 1
				continue
			args.max_triangles = int(v)
			step += 1
			continue

		if step == 20:
			q1 = inquirer.select(
				message="Compute q=1 spectra (Δ₁) to analyze connectivity/cycles?",
				choices=["Yes", "No"] + _back_choice(step),
				default=("Yes" if bool(args.compute_q1_spectra) else "No"),
			).execute()
			if q1 == BACK:
				step -= 1
				continue
			args.compute_q1_spectra = bool(q1 == "Yes")
			if not bool(args.compute_q1_spectra):
				args.q1_every = 0
			step += 1
			continue

		if step == 21:
			if not bool(args.compute_q1_spectra):
				break
			v = _ask_int("q=1 spectra every N epochs (0 = every epoch)", int(args.q1_every), step)
			if v == BACK:
				step -= 1
				continue
			args.q1_every = int(v)
			break

	return args


def _is_topo_spectral_metric(metric_name: str) -> bool:
	m = str(metric_name).strip()
	if not m:
		return False
	if m.endswith("_error"):
		return False
	prefixes = ("beta", "hodge", "persistent", "mtopdiv", "gudhi", "graph_")
	return any(m.startswith(p) for p in prefixes)


def _get_repr_layer_scalar(out: Dict[str, Any], layer: str, metric: str) -> Optional[float]:
	try:
		v = (((out.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {}).get(metric, None)
	except Exception:
		return None
	if isinstance(v, (int, float)):
		return float(v)
	return None


def _parse_early_stop_signals(raw: str, default_mode: str = "max") -> List[Dict[str, str]]:
	"""
	Parse semicolon-separated early-stop rules:
	  layer:metric[:mode];layer:metric[:mode]
	mode is optional and defaults to default_mode.
	"""
	out: List[Dict[str, str]] = []
	for part in [x.strip() for x in str(raw).split(";") if x.strip()]:
		items = [x.strip() for x in part.split(":") if x.strip()]
		if len(items) < 2:
			continue
		layer = items[0]
		metric = items[1]
		mode = items[2] if len(items) >= 3 else str(default_mode)
		mode = mode.lower().strip()
		if mode not in ("min", "max"):
			mode = str(default_mode)
		out.append({"layer": layer, "metric": metric, "mode": mode})
	return out


def _format_early_stop_signal(layer: str, metric: str, mode: str) -> str:
	return f"{layer}:{metric}:{mode}"


def _is_repr_progress_metric(metric_name: str) -> bool:
	"""
	Metrics suitable for progress plotting of computed representation characteristics.
	Exclude bookkeeping/status fields.
	"""
	m = str(metric_name).strip()
	if not m:
		return False
	if m.endswith("_error") or m.endswith("_method"):
		return False
	if m in {
		"train_n",
		"val_n",
		"dim",
		"graph_n",
		"gudhi_n",
		"gudhi_on",
		"gudhi_max_dim",
		"gudhi_max_edge_length",
		"gudhi_grid_n",
		"graph_fixed",
		"graph_fixed_checksum",
		"mtopdiv_stage_a",
		"mtopdiv_stage_b",
		"mtopdiv_a_n",
		"mtopdiv_b_n",
		"mtopdiv_a_fixed_checksum",
		"mtopdiv_b_fixed_checksum",
	}:
		return False
	return m.startswith(("beta", "hodge", "persistent", "mtopdiv", "gudhi"))


def _series_is_near_constant(series: List[Tuple[int, float]], atol: float = 1e-8, rtol: float = 1e-4) -> bool:
	if len(series) < 2:
		return True
	vals = np.asarray([float(v) for _, v in series], dtype=np.float64)
	if vals.size < 2:
		return True
	vmin = float(np.min(vals))
	vmax = float(np.max(vals))
	scale = max(1.0, float(np.max(np.abs(vals))))
	return (vmax - vmin) <= (float(atol) + float(rtol) * scale)


def _series_value_at_epoch(series: List[Tuple[int, float]], epoch: int) -> Optional[float]:
	for e, v in series:
		if int(e) == int(epoch):
			return float(v)
	return None


def _rewrite_progress_figures(
	run_dir: str,
	dataset: str,
	early_stop_key: Optional[str] = None,
	early_stop_signal_epoch: Optional[int] = None,
) -> None:
	"""
	Rewrite simple training-progress figures after each epoch so users can open one file
	and always see the latest full history.
	"""
	import matplotlib.pyplot as plt

	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.exists(metrics_path):
		return
	recs = load_epoch_end_records(metrics_path)
	if not recs:
		return

	fig_dir = os.path.join(run_dir, "figures")
	os.makedirs(fig_dir, exist_ok=True)

	s_acc = get_series(recs, f"bench.{dataset}-val.accuracy")
	s_f1 = get_series(recs, f"bench.{dataset}-val.f1_macro")
	s_loss = get_series(recs, f"bench.{dataset}-val.loss")

	fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
	axes[0].set_title("Validation accuracy")
	axes[1].set_title("Validation F1 (macro)")
	axes[2].set_title("Validation loss")
	if s_acc:
		axes[0].plot([e for e, _ in s_acc], [v for _, v in s_acc], marker="o")
		if early_stop_signal_epoch is not None:
			v = _series_value_at_epoch(s_acc, int(early_stop_signal_epoch))
			if v is not None:
				axes[0].scatter([int(early_stop_signal_epoch)], [v], color="red", s=46, zorder=5)
	if s_f1:
		axes[1].plot([e for e, _ in s_f1], [v for _, v in s_f1], marker="o")
		if early_stop_signal_epoch is not None:
			v = _series_value_at_epoch(s_f1, int(early_stop_signal_epoch))
			if v is not None:
				axes[1].scatter([int(early_stop_signal_epoch)], [v], color="red", s=46, zorder=5)
	if s_loss:
		axes[2].plot([e for e, _ in s_loss], [v for _, v in s_loss], marker="o")
		if early_stop_signal_epoch is not None:
			v = _series_value_at_epoch(s_loss, int(early_stop_signal_epoch))
			if v is not None:
				axes[2].scatter([int(early_stop_signal_epoch)], [v], color="red", s=46, zorder=5)
	for ax in axes:
		ax.set_xlabel("epoch")
		ax.grid(True, alpha=0.25, linestyle="--")
	fig.tight_layout()
	fig.savefig(os.path.join(fig_dir, "fig_quality_progress.png"))
	plt.close(fig)

	# Representation metrics progress (computed topo/spectral characteristics).
	repr_keys_all = [k for k in list_scalar_series_keys(recs) if k.startswith("repr.layers.")]
	repr_items: List[Tuple[str, str, str, List[Tuple[int, float]]]] = []
	for k in repr_keys_all:
		rest = k[len("repr.layers.") :]
		if "." not in rest:
			continue
		layer, metric = rest.rsplit(".", 1)
		if not _is_repr_progress_metric(metric):
			continue
		ser = get_series(recs, k)
		if _series_is_near_constant(ser):
			continue
		repr_items.append((k, layer, metric, ser))

	# Prioritize common compact characteristics, then keep others.
	metric_priority = {
		"mtopdiv_train_val": 0,
		"beta1_L_est": 1,
		"beta1_persistent_est": 2,
		"beta0_L_est": 3,
		"beta0_persistent_est": 4,
	}
	repr_items.sort(key=lambda x: (metric_priority.get(x[2], 999), x[1], x[2]))
	max_repr_plots = 9
	repr_items = repr_items[:max_repr_plots]

	if repr_items:
		n = len(repr_items)
		ncols = 3 if n > 4 else 2
		nrows = int(np.ceil(n / float(ncols)))
		fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 2.9 * nrows))
		axes_arr = np.atleast_1d(axes).reshape(-1)
		for ax_i, (_key, layer, metric, ser) in enumerate(repr_items):
			ax = axes_arr[ax_i]
			ax.plot([e for e, _ in ser], [v for _, v in ser], marker="o")
			if early_stop_signal_epoch is not None:
				v = _series_value_at_epoch(ser, int(early_stop_signal_epoch))
				if v is not None:
					ax.scatter([int(early_stop_signal_epoch)], [v], color="red", s=38, zorder=5)
			layer_short = layer if len(layer) <= 28 else f"...{layer[-25:]}"
			ax.set_title(f"{layer_short}: {metric}", fontsize=9)
			ax.set_xlabel("epoch")
			ax.grid(True, alpha=0.25, linestyle="--")
		for ax in axes_arr[n:]:
			ax.axis("off")
		fig.tight_layout()
		fig.savefig(os.path.join(fig_dir, "fig_repr_progress.png"))
		plt.close(fig)

	if early_stop_key:
		s_es = get_series(recs, early_stop_key)
		if s_es:
			fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.6))
			ax.set_title(f"Early-stop metric: {early_stop_key}")
			ax.plot([e for e, _ in s_es], [v for _, v in s_es], marker="o")
			if early_stop_signal_epoch is not None:
				v = _series_value_at_epoch(s_es, int(early_stop_signal_epoch))
				if v is not None:
					ax.scatter([int(early_stop_signal_epoch)], [v], color="red", s=52, zorder=6)
			ax.set_xlabel("epoch")
			ax.grid(True, alpha=0.25, linestyle="--")
			fig.tight_layout()
			fig.savefig(os.path.join(fig_dir, "fig_early_stop_metric.png"))
			plt.close(fig)


def _save_model_checkpoint(path: str, model: nn.Module, epoch: int, payload: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	obj = {
		"epoch": int(epoch),
		"state_dict": model.state_dict(),
		"payload": dict(payload or {}),
	}
	torch.save(obj, path)


def _collect_cv_embeddings(
	model: nn.Module,
	loader: Any,
	layer_name: str,
	preprocess: Optional[Any],
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	embeds: List[torch.Tensor] = []
	labels: List[torch.Tensor] = []
	model.eval()
	with torch.no_grad(), LayerTaps(model, [layer_name]) as taps:
		for batch in loader:
			if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
				continue
			x = batch[0].to(device)
			y = batch[1]
			if not isinstance(y, torch.Tensor):
				y = torch.as_tensor(y)
			y = y.view(-1).long().detach().cpu()
			if preprocess is not None:
				x = preprocess(x)
			_ = model(x)
			z = _repr_from_activation(taps.outputs.get(layer_name, None))
			if z is None:
				continue
			embeds.append(z.detach().to("cpu"))
			labels.append(y)
	if not embeds:
		raise RuntimeError(f"Could not collect embeddings from layer '{layer_name}'.")
	return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def _evaluate_embedding_retrieval_cv(
	model: nn.Module,
	loader: Any,
	layers: List[str],
	preprocess: Optional[Any],
	device: torch.device,
	top_k: int,
	seed: int,
	preferred_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
	if not layers:
		return {"error": "no_layers"}
	if loader is None:
		return {"error": "no_loader"}

	first_ok_layer: Optional[str] = None
	first_emb: Optional[torch.Tensor] = None
	labels: Optional[torch.Tensor] = None
	initial_errors: List[Dict[str, str]] = []
	for layer in layers:
		try:
			emb0, lab0 = _collect_cv_embeddings(model, loader, layer_name=layer, preprocess=preprocess, device=device)
			first_ok_layer = str(layer)
			first_emb = emb0
			labels = lab0
			break
		except Exception as e:
			initial_errors.append({"layer": str(layer), "error": str(e)})
	if first_emb is None or labels is None or first_ok_layer is None:
		return {"error": "no_valid_embedding_layers", "rows": initial_errors}
	n = int(labels.numel())
	if n <= 1:
		return {"error": "not_enough_samples"}
	labels_np = labels.numpy()
	classes, counts = np.unique(labels_np, return_counts=True)
	valid_classes = [int(c) for c, cnt in zip(classes.tolist(), counts.tolist()) if int(cnt) >= 2]
	if not valid_classes:
		return {"error": "not_enough_class_samples", "n": n}
	rng = np.random.default_rng(int(seed))
	anchor_class = int(rng.choice(valid_classes))
	anchor_candidates = np.where(labels_np == anchor_class)[0]
	anchor_idx = int(rng.choice(anchor_candidates))

	def _score(emb: torch.Tensor) -> Dict[str, Any]:
		e = emb.float()
		k = max(1, min(int(top_k), int(e.shape[0]) - 1))
		d = torch.cdist(e[anchor_idx : anchor_idx + 1], e).view(-1)
		d[anchor_idx] = float("inf")
		nn_idx = torch.topk(d, k=k, largest=False).indices
		nn_labels = labels[nn_idx]
		hits = int((nn_labels == anchor_class).sum().item())
		return {"top_k": int(k), "same_class_count": hits, "same_class_ratio": (float(hits) / float(k))}

	rows: List[Dict[str, Any]] = list(initial_errors)
	rows.append({"layer": str(first_ok_layer), **_score(first_emb)})
	for layer in [x for x in layers if str(x) != str(first_ok_layer)]:
		try:
			emb, _labels = _collect_cv_embeddings(model, loader, layer_name=layer, preprocess=preprocess, device=device)
			rows.append({"layer": str(layer), **_score(emb)})
		except Exception as e:
			rows.append({"layer": str(layer), "error": str(e)})

	preferred = set([str(x) for x in (preferred_layers or [])])
	for r in rows:
		r["is_signal_layer"] = bool(str(r.get("layer", "")) in preferred)

	ok_rows = [r for r in rows if "same_class_ratio" in r]
	ok_rows.sort(key=lambda x: float(x["same_class_ratio"]), reverse=True)

	return {
		"n_samples": n,
		"anchor_class": int(anchor_class),
		"anchor_index": int(anchor_idx),
		"rows": rows,
		"top": ok_rows[: min(10, len(ok_rows))],
	}


def _parse_csv_int_ranges(s: str) -> List[int]:
	"""
	Parse strings like "0,2,4-6" into integer indices.
	"""
	out: List[int] = []
	for part in [x.strip() for x in s.split(",") if x.strip()]:
		if "-" in part:
			a, b = part.split("-", 1)
			if a.strip().isdigit() and b.strip().isdigit():
				ai, bi = int(a.strip()), int(b.strip())
				if ai <= bi:
					out.extend(list(range(ai, bi + 1)))
				else:
					out.extend(list(range(bi, ai + 1)))
		elif part.isdigit():
			out.append(int(part))
	# stable unique
	seen = set()
	res: List[int] = []
	for x in out:
		if x not in seen:
			seen.add(x)
			res.append(x)
	return res


def _prompt_yes_no(prompt: str, default: bool) -> bool:
	suffix = "[Y/n]" if default else "[y/N]"
	raw = input(f"{prompt} {suffix}: ").strip().lower()
	if not raw:
		return bool(default)
	if raw in ("y", "yes", "1", "true"):
		return True
	if raw in ("n", "no", "0", "false"):
		return False
	return bool(default)


def _module_display_name(model: nn.Module, name: str) -> str:
	mod = dict(model.named_modules()).get(name, None)
	if mod is None:
		return name
	cls = mod.__class__.__name__
	shape = ""
	if isinstance(mod, nn.Linear):
		shape = f" {mod.in_features}->{mod.out_features}"
	elif isinstance(mod, nn.Conv2d):
		shape = f" {mod.in_channels}->{mod.out_channels} k={tuple(mod.kernel_size)}"
	return f"{name} [{cls}{shape}]"


def _interactive_pick_layers(current_layers: List[str], all_modules: List[str], model: nn.Module) -> List[str]:
	print("\n[Interactive] Available modules (index -> name):")
	for i, name in enumerate(all_modules):
		print(f"  {i:3d}: {_module_display_name(model, name)}")
	print(f"\n[Interactive] Current selected layers ({len(current_layers)}): {current_layers}")
	print("[Interactive] Enter indices (e.g. 0,4,8-12) to override, or press Enter to keep current.")
	raw = input("layer indices: ").strip()
	if not raw:
		return current_layers
	ids = _parse_csv_int_ranges(raw)
	chosen = [all_modules[i] for i in ids if 0 <= i < len(all_modules)]
	if not chosen:
		print("[Interactive] No valid indices selected, keeping current layers.")
		return current_layers
	return chosen


def _interactive_pick_bench_metrics(current: Tuple[str, ...]) -> Tuple[str, ...]:
	available = ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
	print("\n[Interactive] Available benchmark metrics:")
	for i, m in enumerate(available):
		flag = "*" if m in set(current) else " "
		print(f"  {i:2d}: {m} {flag}")
	print("[Interactive] Enter metric indices (e.g. 0,1,4) or press Enter to keep current.")
	raw = input("metric indices: ").strip()
	if not raw:
		return tuple(current)
	ids = _parse_csv_int_ranges(raw)
	chosen = [available[i] for i in ids if 0 <= i < len(available)]
	if not chosen:
		print("[Interactive] No valid metrics selected, keeping current metrics.")
		return tuple(current)
	return tuple(chosen)


def _interactive_pick_layers_tui(current_layers: List[str], all_modules: List[str], model: nn.Module) -> List[str]:
	current = set(current_layers)
	choices = [
		Choice(value=name, name=f"{i:3d}: {_module_display_name(model, name)}", enabled=(name in current))
		for i, name in enumerate(all_modules)
	]
	selected = inquirer.checkbox(
		message="Select layers to track (Space toggle, Enter confirm):",
		choices=choices,
		instruction="↑/↓ navigate, Space select, Enter confirm",
	).execute()
	if not selected:
		print("[Interactive/TUI] Empty selection, keeping current layers.")
		return current_layers
	return list(selected)


def _interactive_pick_bench_metrics_tui(current: Tuple[str, ...]) -> Tuple[str, ...]:
	available = ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
	current_set = set(current)
	choices = [Choice(value=m, name=m, enabled=(m in current_set)) for m in available]
	selected = inquirer.checkbox(
		message="Select benchmark metrics:",
		choices=choices,
		instruction="↑/↓ navigate, Space select, Enter confirm",
	).execute()
	if not selected:
		print("[Interactive/TUI] Empty metric selection, keeping current metrics.")
		return tuple(current)
	return tuple(selected)


def _set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)


def _maybe_subsample_hf_dataset(ds: Any, max_examples: int, seed: int) -> Any:
	"""
	Subsample HuggingFace Dataset (map-style) by shuffling then taking first N.
	No-op for non-HF datasets or when max_examples <= 0.
	"""
	try:
		n = int(max_examples)
	except Exception:
		n = 0
	if n <= 0 or ds is None:
		return ds
	if not hasattr(ds, "select"):
		return ds
	try:
		if hasattr(ds, "shuffle"):
			ds = ds.shuffle(seed=int(seed))
		# select first N
		n_eff = min(int(n), int(len(ds)))
		return ds.select(range(n_eff))
	except Exception:
		return ds


def _infer_num_classes(ds: Any) -> int:
	# torchvision style
	if hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and len(ds.classes) > 0:
		return int(len(ds.classes))
	# MedMNIST style
	if hasattr(ds, "n_classes"):
		return int(getattr(ds, "n_classes"))
	if hasattr(ds, "info") and isinstance(getattr(ds, "info"), dict):
		info = getattr(ds, "info")
		lab = info.get("label", None)
		if isinstance(lab, dict) and len(lab) > 0:
			return int(len(lab))
	if hasattr(ds, "labels"):
		try:
			labs = np.asarray(getattr(ds, "labels"))
			if labs.size > 0:
				return int(np.max(labs)) + 1
		except Exception:
			pass
	# HF datasets style
	if hasattr(ds, "features") and isinstance(ds.features, dict) and "label" in ds.features:
		f = ds.features["label"]
		if hasattr(f, "num_classes"):
			return int(f.num_classes)
	# Fallback: assume 2-class if unknown
	return 2


def _infer_cv_input_flat_dim(ds: Any) -> Optional[int]:
	"""
	Best-effort inference of flattened input dimension from first sample.
	Used to adapt MLP to non-28x28 CV datasets.
	"""
	try:
		ex = ds[0]
	except Exception:
		return None
	x = None
	if isinstance(ex, (tuple, list)) and len(ex) >= 1:
		x = ex[0]
	elif isinstance(ex, dict):
		x = ex.get("image", None)
	if not isinstance(x, torch.Tensor):
		return None
	if x.numel() <= 0:
		return None
	return int(np.prod(tuple(x.shape)))


def _is_text_batch(batch: Any) -> bool:
	return isinstance(batch, dict) and ("input_ids" in batch or "attention_mask" in batch)


def _cv_noise_transform(x: torch.Tensor, sigma: float) -> torch.Tensor:
	# x in [0,1]
	if sigma <= 0:
		return x
	noise = torch.randn_like(x) * float(sigma)
	return torch.clamp(x + noise, 0.0, 1.0)


class _NoisyDataset(torch.utils.data.Dataset):
	def __init__(self, base: torch.utils.data.Dataset, sigma: float):
		self.base = base
		self.sigma = float(sigma)

	def __len__(self) -> int:
		return len(self.base)

	def __getitem__(self, idx: int):
		x, y = self.base[idx]
		if isinstance(x, torch.Tensor):
			x = _cv_noise_transform(x, self.sigma)
		return x, y


def _build_cv_model(
	kind: str,
	num_classes: int,
	device: torch.device,
	pretrained: bool,
	input_flat_dim: Optional[int] = None,
) -> Tuple[nn.Module, Optional[Any], List[str]]:
	mi = get_model_info(kind, device=device, pretrained=pretrained)
	model = mi.model
	preprocess = mi.preprocess
	layer_names = mi.layer_names

	# Adjust classifier head for CV models where needed
	if kind.lower() == "resnet18":
		in_f = model.fc.in_features
		model.fc = nn.Linear(in_f, num_classes).to(device)
	elif kind.lower() in ("convnext_tiny", "convnext"):
		# convnext_tiny.classifier = Sequential(LayerNorm2d, Flatten, Linear)
		if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
			last = model.classifier[-1]
			if isinstance(last, nn.Linear):
				model.classifier[-1] = nn.Linear(last.in_features, num_classes).to(device)
	elif kind.lower() in ("efficientnet_b0", "efficientnet"):
		if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
			last = model.classifier[-1]
			if isinstance(last, nn.Linear):
				model.classifier[-1] = nn.Linear(last.in_features, num_classes).to(device)
	elif kind.lower() == "mlp":
		# Make MLP input size match current CV dataset tensor shape.
		if isinstance(model, nn.Sequential) and len(model) >= 2 and isinstance(model[1], nn.Linear):
			if input_flat_dim is not None and int(input_flat_dim) > 0 and int(input_flat_dim) != int(model[1].in_features):
				model[1] = nn.Linear(int(input_flat_dim), 512).to(device)
		# MLP does not use pretrained weights in this project.
		if pretrained:
			print("[Model] 'mlp' has no pretrained backbone; proceeding with random initialization.")

	return model, preprocess, layer_names


def _build_text_model(kind: str, num_classes: int, device: torch.device, pretrained: bool):
	tf = _require_transformers_runexp()
	AutoModel = getattr(tf, "AutoModel")
	AutoConfig = getattr(tf, "AutoConfig")
	AutoModelForSequenceClassification = getattr(tf, "AutoModelForSequenceClassification")
	DistilBertConfig = getattr(tf, "DistilBertConfig")
	DistilBertForSequenceClassification = getattr(tf, "DistilBertForSequenceClassification")

	k = kind.lower()
	if k in ("distilbert", "distilbert-base-uncased"):
		if pretrained:
			model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
		else:
			model = DistilBertForSequenceClassification(DistilBertConfig(num_labels=num_classes))
		layer_names = _pick_transformer_blocks_generic_for_model(model)
		model.to(device)
		return model, layer_names

	SMOLLM_IDS = {
		"smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
		"smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
		# Backward-compatible aliases
		"smollm": "HuggingFaceTB/SmolLM2-135M",
		"smollm2": "HuggingFaceTB/SmolLM2-135M",
		"smollm-135m": "HuggingFaceTB/SmolLM2-135M",
		"smollm-360m": "HuggingFaceTB/SmolLM2-360M",
	}

	if k in SMOLLM_IDS:
		model_id = SMOLLM_IDS[k]
		# Prefer a pretrained transformer backbone, then adapt to classification.
		if pretrained:
			# Try direct seq-classification head (may be unsupported for some model cards).
			try:
				model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
				layer_names = _pick_transformer_blocks_generic_for_model(model)
				model.to(device)
				return model, layer_names
			except Exception:
				# Fall back to a base model + lightweight classification head.
				cfg = AutoConfig.from_pretrained(model_id)
				base = AutoModel.from_pretrained(model_id, config=cfg)

				class _WrappedLMForSequenceClassification(nn.Module):
					def __init__(self, base_model: nn.Module, n_labels: int):
						super().__init__()
						self.base = base_model
						hid = getattr(getattr(base_model, "config", None), "hidden_size", None)
						if hid is None:
							hid = getattr(getattr(base_model, "config", None), "n_embd", None)
						if hid is None:
							raise ValueError("Could not infer hidden size for classification head.")
						self.classifier = nn.Linear(int(hid), int(n_labels))

					def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
						out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
						# Prefer last_hidden_state; fall back to tuple[0]
						h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
						# Pool: first token if available, else mean pool with mask.
						if h.dim() == 3:
							if attention_mask is not None and attention_mask.dim() == 2:
								m = attention_mask.unsqueeze(-1).float()
								hp = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
							else:
								hp = h[:, 0, :]
						else:
							hp = h
						logits = self.classifier(hp)
						loss = None
						if labels is not None:
							loss = nn.CrossEntropyLoss()(logits, labels)
						# Mimic HF output object minimally.
						return type("Out", (), {"logits": logits, "loss": loss})()

				model = _WrappedLMForSequenceClassification(base, num_classes)
				layer_names = [f"base.{n}" for n in _pick_transformer_blocks_generic_for_model(base)]
				model.to(device)
				return model, layer_names
		else:
			# Offline fallback: use a small distilbert-sized classifier.
			model = DistilBertForSequenceClassification(DistilBertConfig(num_labels=num_classes))
			layer_names = _pick_transformer_blocks_generic_for_model(model)
			model.to(device)
			return model, layer_names

	raise ValueError(f"Unknown text model kind: {kind}")


def _freeze_all(model: nn.Module) -> None:
	for p in model.parameters():
		p.requires_grad = False


def _unfreeze_named_prefixes(model: nn.Module, prefixes: Iterable[str]) -> None:
	pfx = tuple(prefixes)
	for name, p in model.named_parameters():
		if name.startswith(pfx):
			p.requires_grad = True


def _unfreeze_last_n_params(model: nn.Module, n: int) -> None:
	# unfreeze by parameter order (simple, model-agnostic)
	params = list(model.parameters())
	for p in params:
		p.requires_grad = False
	for p in params[-max(int(n), 1) :]:
		p.requires_grad = True


def _unfreeze_last_linear(model: nn.Module) -> None:
	"""
	Best-effort fallback for linear probing: unfreeze the last nn.Linear module parameters.
	Works for nn.Sequential MLPs and many CNN heads.
	"""
	last: Optional[nn.Linear] = None
	for m in model.modules():
		if isinstance(m, nn.Linear):
			last = m
	if last is None:
		return
	for p in last.parameters():
		p.requires_grad = True


def _graph_smoothness_penalty(z: torch.Tensor, k: int = 8, max_points: int = 96) -> torch.Tensor:
	"""
	Differentiable spectral-ish regularizer: sum over kNN edges of ||z_i - z_j||^2.
	z: (N,D) float tensor.
	"""
	N = z.shape[0]
	if N <= 2:
		return z.sum() * 0.0
	if N > max_points:
		z = z[:max_points]
		N = z.shape[0]
	k = min(int(k), N - 1)
	# pairwise distances
	D = torch.cdist(z, z)
	D.fill_diagonal_(float("inf"))
	# kNN indices per row
	_, nn_idx = torch.topk(D, k=k, dim=1, largest=False)
	# gather neighbor vectors
	zz = z.unsqueeze(1).expand(-1, k, -1)
	zn = z[nn_idx]
	return ((zz - zn) ** 2).sum(dim=-1).mean()


def _repr_from_activation(act: Any) -> Optional[torch.Tensor]:
	# Mirror logic of _repr_from_tensor but keep torch.Tensor for differentiable penalties
	if not isinstance(act, torch.Tensor):
		return None
	x = act
	if not x.is_floating_point():
		x = x.float()
	if x.dim() == 4:
		# (N,C,H,W) -> GAP -> (N,C)
		x = x.mean(dim=(2, 3))
	elif x.dim() == 3:
		# (N,T,D) -> first token -> (N,D)
		x = x[:, 0, :]
	elif x.dim() == 2:
		pass
	else:
		x = x.view(x.shape[0], -1)
	return x


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--task", type=str, choices=["cv", "nlp"], default="cv")
	ap.add_argument("--dataset", type=str, default="cifar10")
	ap.add_argument("--model", type=str, default="resnet18")
	ap.add_argument("--data_root", type=str, default="./data")
	ap.add_argument("--download", action="store_true")
	ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument(
		"--interactive",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Interactive mode (enabled by default). Use --no-interactive for fully non-interactive CLI.",
	)
	ap.add_argument(
		"--interactive_ui",
		type=str,
		default="tui",
		choices=["text", "tui"],
		help="Interactive UI type: plain text prompts or TUI with arrows/space (default: tui).",
	)

	# training
	ap.add_argument("--epochs", type=int, default=5)
	ap.add_argument("--batch_size", type=int, default=64)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--weight_decay", type=float, default=1e-4)
	ap.add_argument("--pretrained", action="store_true")
	ap.add_argument("--max_train_batches", type=int, default=0, help="0 = no limit")
	ap.add_argument("--max_val_batches", type=int, default=0, help="0 = no limit")
	# NLP subsampling (HF datasets)
	ap.add_argument("--nlp_max_train_examples", type=int, default=0, help="If >0, subsample first N train examples for NLP datasets.")
	ap.add_argument("--nlp_max_val_examples", type=int, default=0, help="If >0, subsample first N val examples for NLP datasets.")
	ap.add_argument("--nlp_max_test_examples", type=int, default=0, help="If >0, subsample first N test examples for NLP datasets.")
	ap.add_argument("--nlp_subset_seed", type=int, default=0, help="Seed used when subsampling NLP datasets (shuffle before selecting).")

	# fine-tune strategies
	ap.add_argument(
		"--finetune",
		type=str,
		choices=["full", "linear_probe", "last_n_params", "named_prefixes", "named_patterns", "selected_layers", "tracked_layers"],
		default="full",
	)
	ap.add_argument(
		"--sweep_finetune",
		type=str,
		default="",
		help="Comma-separated finetune modes to run sequentially (e.g. 'full,linear_probe,last_n_params'). If set, overrides --finetune.",
	)
	ap.add_argument("--last_n_params", type=int, default=200)
	ap.add_argument("--train_prefixes", type=str, default="", help="comma-separated parameter name prefixes to unfreeze")
	ap.add_argument("--train_patterns", type=str, default="", help="CSV include patterns over parameter names (glob by default).")
	ap.add_argument("--train_exclude_patterns", type=str, default="", help="CSV exclude patterns over parameter names.")
	ap.add_argument("--train_regex", action="store_true", help="Treat train pattern lists as regex.")
	ap.add_argument("--train_layers", type=str, default="", help="CSV module names to unfreeze (for --finetune selected_layers).")

	# monitor / topo
	ap.add_argument("--layers", type=str, default="", help="comma-separated module names to hook; empty = defaults for model")
	ap.add_argument("--layer_include", type=str, default="", help="CSV include patterns to select hook layers from model modules.")
	ap.add_argument("--layer_exclude", type=str, default="", help="CSV exclude patterns for hook layer selection.")
	ap.add_argument("--layer_regex", action="store_true", help="Treat layer include/exclude patterns as regex.")
	ap.add_argument("--list_layers_only", action="store_true", help="Print model layers/params and exit (no training).")
	ap.add_argument("--list_layers_leaf_only", action="store_true", help="When listing/selecting layers, use only leaf modules.")
	ap.add_argument("--strict_validation", action=argparse.BooleanOptionalAction, default=True, help="Fail on unmatched layer/parameter patterns or invalid explicit layer names.")
	ap.add_argument("--max_samples_per_stage", type=int, default=1024)
	ap.add_argument("--max_points_for_graph", type=int, default=256)
	ap.add_argument("--max_eigs", type=int, default=10)
	ap.add_argument("--knn_small", type=int, default=5)
	ap.add_argument("--knn_large", type=int, default=15)
	ap.add_argument("--graph_stage", type=str, default="train", help="stage used for graph/topo computations: train|val")
	ap.add_argument("--fixed_graph_points", action="store_true", help="use fixed subset of points for graph across epochs (per layer)")
	ap.add_argument("--fixed_graph_seed", type=int, default=0)
	ap.add_argument("--compute_gudhi", action="store_true", help="compute PH via GUDHI (Vietoris–Rips) on graph point set")
	ap.add_argument("--gudhi_max_points", type=int, default=128)
	ap.add_argument("--gudhi_max_dim", type=int, default=1)
	ap.add_argument("--gudhi_max_edge_length", type=float, default=2.0)
	ap.add_argument("--gudhi_every", type=int, default=0, help="if >0, compute gudhi only every N epochs (epoch % N == 0)")
	ap.add_argument("--gudhi_grid_n", type=int, default=64)
	ap.add_argument("--gudhi_compute_lifetime_stats", action="store_true")
	ap.add_argument("--gudhi_compute_betti_curve", action="store_true")
	ap.add_argument("--gudhi_compute_landscape", action="store_true")
	ap.add_argument("--gudhi_landscape_k", type=int, default=3)
	ap.add_argument("--gudhi_compute_silhouette", action="store_true")
	ap.add_argument("--gudhi_silhouette_q", type=float, default=1.0)
	ap.add_argument("--gudhi_compute_persistence_image", action="store_true")
	ap.add_argument("--gudhi_pi_size", type=int, default=20)
	ap.add_argument("--gudhi_pi_sigma", type=float, default=0.15)
	ap.add_argument("--gudhi_pi_tau", type=float, default=0.5)
	ap.add_argument("--build_triangles", action="store_true")
	ap.add_argument(
		"--dim2_every",
		type=int,
		default=0,
		help="If >0, enable dim=2 (triangles) only every N epochs (epoch %% N == 0). 0 means: follow --build_triangles for all epochs.",
	)
	ap.add_argument("--max_triangles", type=int, default=50000)
	ap.add_argument("--compute_q1_spectra", action="store_true")
	ap.add_argument(
		"--q1_every",
		type=int,
		default=0,
		help="If >0, enable q=1 spectra only every N epochs (epoch %% N == 0). 0 means: follow --compute_q1_spectra for all epochs.",
	)
	ap.add_argument("--compute_mtopdiv", action="store_true")
	ap.add_argument("--mtopdiv_runs", type=int, default=2)
	ap.add_argument("--mtopdiv_stage_a", type=str, default="train")
	ap.add_argument("--mtopdiv_stage_b", type=str, default="val")
	ap.add_argument("--fixed_mtopdiv_points", action="store_true", help="use fixed subsets for MTopDiv across epochs (per layer, per stage)")
	ap.add_argument("--fixed_mtopdiv_seed", type=int, default=0)
	ap.add_argument("--regularization", type=float, default=1e-10)

	# optional differentiable regularizer
	ap.add_argument("--reg_kind", type=str, choices=["none", "graph_smoothness"], default="none")
	ap.add_argument("--reg_weight", type=float, default=0.0)
	ap.add_argument("--reg_layer", type=str, default="", help="module name used for reg; empty=first monitor layer")
	ap.add_argument("--reg_knn_k", type=int, default=8)
	ap.add_argument("--reg_max_points", type=int, default=96)

	# early stopping (repr/topo/spectral only)
	ap.add_argument("--early_stop", action="store_true", help="Enable repr-based early-stop signal using repr.layers.<layer>.<metric> (does not stop training).")
	ap.add_argument("--early_stop_layer", type=str, default="", help="Layer name for early stop metric (must exist in monitor layers).")
	ap.add_argument("--early_stop_metric", type=str, default="", help="Topo/spectral metric key inside repr.layers.<layer>.*")
	ap.add_argument("--early_stop_mode", type=str, choices=["min", "max"], default="max")
	ap.add_argument(
		"--early_stop_signals",
		type=str,
		default="",
		help="Semicolon-separated rules: layer:metric[:mode];layer:metric[:mode]. If set, overrides single --early_stop_layer/--early_stop_metric.",
	)
	ap.add_argument(
		"--early_stop_aggregate",
		type=str,
		choices=["all", "any"],
		default="all",
		help="How to combine multiple early-stop signals: all (default) or any.",
	)
	ap.add_argument("--early_stop_patience", type=int, default=5)
	ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
	ap.add_argument("--early_stop_start_epoch", type=int, default=0)

	# evaluation options
	ap.add_argument("--cv_noise_sigma", type=float, default=0.0, help="if >0, also eval on noisy val with this sigma")
	ap.add_argument(
		"--bench_metrics",
		type=str,
		default="loss,accuracy,precision_macro,recall_macro,f1_macro",
		help="CSV benchmark metrics to log for val/test (classification): loss,accuracy,precision_macro,recall_macro,f1_macro",
	)
	ap.add_argument(
		"--live_plots",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Rewrite progress figures in run_dir/figures after each epoch.",
	)
	ap.add_argument(
		"--save_models",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Save best-main and (if triggered) early-signal model checkpoints.",
	)

	args = ap.parse_args()
	if bool(args.interactive) and str(args.interactive_ui) == "tui" and not bool(args.list_layers_only):
		args = _interactive_config_tui(args)
	def _schedule(epoch: int, base: bool, every: int) -> bool:
		if every and every > 0:
			return (epoch % every) == 0
		return bool(base)

	def run_one(finetune_mode: str) -> str:
		_set_seed(args.seed)
		device = torch.device(args.device)

		bundle = get_dataset(args.dataset, root=args.data_root, download=args.download, tokenizer_name="distilbert-base-uncased")
		# Optional NLP dataset subsampling (useful for large HF datasets like yahoo_answers_topics).
		if str(args.task).lower().strip() == "nlp":
			bundle.train = _maybe_subsample_hf_dataset(bundle.train, int(args.nlp_max_train_examples), int(args.nlp_subset_seed))
			bundle.val = _maybe_subsample_hf_dataset(bundle.val, int(args.nlp_max_val_examples), int(args.nlp_subset_seed))
			bundle.test = _maybe_subsample_hf_dataset(bundle.test, int(args.nlp_max_test_examples), int(args.nlp_subset_seed))
		loaders = make_dataloaders(bundle, batch_size=args.batch_size, num_workers=0)
		train_loader = loaders["train"]
		val_loader = loaders["val"] or loaders["test"]
		if train_loader is None or val_loader is None:
			raise RuntimeError("Dataset did not provide required splits.")

		num_classes = _infer_num_classes(bundle.train) if bundle.train is not None else _infer_num_classes(bundle.val or bundle.test)

		# Build model
		layer_names: List[str] = []
		bench_metrics = tuple([x.strip() for x in args.bench_metrics.split(",") if x.strip()]) or ("loss", "accuracy")
		preprocess = None
		if args.task == "cv":
			input_flat_dim = _infer_cv_input_flat_dim(bundle.train) if str(args.model).lower() == "mlp" and bundle.train is not None else None
			model, preprocess, layer_names = _build_cv_model(
				args.model,
				num_classes=num_classes,
				device=device,
				pretrained=args.pretrained,
				input_flat_dim=input_flat_dim,
			)
			loss_fn = nn.CrossEntropyLoss()
		elif args.task == "nlp":
			model, layer_names = _build_text_model(args.model, num_classes=num_classes, device=device, pretrained=args.pretrained)
			loss_fn = nn.CrossEntropyLoss()
		else:
			raise ValueError("Unknown task")

		# Override layers if provided
		all_modules = list_module_names(model, leaf_only=bool(args.list_layers_leaf_only))
		if args.layers.strip():
			layer_names = [x.strip() for x in args.layers.split(",") if x.strip()]
			missing_layers = [n for n in layer_names if n not in set(all_modules)]
			if missing_layers:
				msg = f"Invalid layer names (not found in model.named_modules): {missing_layers}"
				if args.strict_validation:
					raise SelectionValidationError(msg)
				print("[Layer selection]", msg)
				layer_names = [n for n in layer_names if n in set(all_modules)]
				if not layer_names:
					raise ValueError("No valid layers left after filtering invalid --layers values.")
		else:
			# Optional pattern-based selection over model.named_modules() for monitor layers.
			inc = csv_to_list(args.layer_include)
			exc = csv_to_list(args.layer_exclude)
			if inc or exc:
				rep = select_names(all_modules, include=inc, exclude=exc, use_regex=bool(args.layer_regex))
				if rep.unmatched:
					msg = f"Unmatched layer selection patterns: {rep.unmatched}"
					if args.strict_validation:
						raise SelectionValidationError(msg)
					print("[Layer selection]", msg)
				layer_names = rep.selected
				if not layer_names:
					raise ValueError("Layer selection produced no layers. Check --layer_include/--layer_exclude.")

		if args.interactive and not args.list_layers_only:
			if str(args.interactive_ui) == "tui":
				layer_names = _interactive_pick_layers_tui(layer_names, all_modules, model)
				bench_metrics = _interactive_pick_bench_metrics_tui(tuple(bench_metrics))
				args.compute_mtopdiv = bool(
					inquirer.confirm(
						message="Enable MTopDiv?",
						default=bool(args.compute_mtopdiv),
					).execute()
				)
				args.compute_gudhi = bool(
					inquirer.confirm(
						message="Enable GUDHI PH?",
						default=bool(args.compute_gudhi),
					).execute()
				)
			else:
				layer_names = _interactive_pick_layers(layer_names, all_modules, model)
				bench_metrics = _interactive_pick_bench_metrics(tuple(bench_metrics))
				# Optional toggles for expensive metrics.
				args.compute_mtopdiv = _prompt_yes_no("Enable MTopDiv?", bool(args.compute_mtopdiv))
				args.compute_gudhi = _prompt_yes_no("Enable GUDHI PH?", bool(args.compute_gudhi))
			print(f"[Interactive] Final layer count: {len(layer_names)}")
			print(f"[Interactive] Final benchmark metrics: {bench_metrics}")

			if str(args.finetune) == "selected_layers":
				current_train_layers = [x for x in csv_to_list(str(args.train_layers)) if x in set(all_modules)]
				if not current_train_layers:
					current_train_layers = [x for x in layer_names if x in set(all_modules)]
				if str(args.interactive_ui) == "tui":
					chosen_train_layers = _interactive_pick_layers_tui(current_train_layers, all_modules, model)
				else:
					chosen_train_layers = _interactive_pick_layers(current_train_layers, all_modules, model)
				args.train_layers = ",".join(chosen_train_layers)
				print(f"[Interactive] Trainable layers ({len(chosen_train_layers)}): {chosen_train_layers}")

			enable_es = bool(
				inquirer.confirm(
					message="Enable repr-based early-stop signal (no hard stop)?",
					default=bool(args.early_stop),
				).execute()
			)
			args.early_stop = enable_es
			if enable_es:
				common_metrics = ["beta0_L_est", "beta1_L_est", "beta0_persistent_est", "beta1_persistent_est", "mtopdiv_train_val", "__custom__"]
				existing_signals = _parse_early_stop_signals(str(args.early_stop_signals), default_mode=str(args.early_stop_mode))
				use_multi = bool(
					inquirer.confirm(
						message="Use multiple early-stop signals?",
						default=bool(len(existing_signals) > 1),
					).execute()
				)
				signals: List[Dict[str, str]] = []
				if use_multi:
					add_more = True
					while add_more:
						lay = str(
							inquirer.select(
								message="Early-stop layer:",
								choices=list(layer_names) if layer_names else [str(args.early_stop_layer or "")],
								default=(args.early_stop_layer if args.early_stop_layer in layer_names else (layer_names[0] if layer_names else "")),
							).execute()
						)
						mm = inquirer.select(
							message=f"Early-stop metric for {lay}:",
							choices=common_metrics,
							default=(args.early_stop_metric if args.early_stop_metric in common_metrics else "beta1_persistent_est"),
						).execute()
						if mm == "__custom__":
							met = str(
								inquirer.text(
									message="Custom metric key (inside repr.layers.<layer>.*):",
									default=str(args.early_stop_metric or "beta1_persistent_est"),
								).execute()
							).strip()
						else:
							met = str(mm)
						md = str(
							inquirer.select(
								message=f"Mode for {lay}:{met}:",
								choices=["max", "min"],
								default=str(args.early_stop_mode),
							).execute()
						)
						signals.append({"layer": lay, "metric": met, "mode": md})
						add_more = bool(
							inquirer.confirm(
								message="Add another early-stop signal?",
								default=False,
							).execute()
						)
					args.early_stop_aggregate = str(
						inquirer.select(
							message="Aggregate condition for stopping:",
							choices=["all", "any"],
							default=str(args.early_stop_aggregate),
						).execute()
					)
				else:
					if layer_names:
						args.early_stop_layer = str(
							inquirer.select(
								message="Early-stop layer:",
								choices=list(layer_names),
								default=(args.early_stop_layer if args.early_stop_layer in layer_names else layer_names[0]),
							).execute()
						)
					mm = inquirer.select(
						message="Early-stop metric:",
						choices=common_metrics,
						default=(args.early_stop_metric if args.early_stop_metric in common_metrics else "beta1_persistent_est"),
					).execute()
					if mm == "__custom__":
						args.early_stop_metric = str(
							inquirer.text(
								message="Custom metric key (inside repr.layers.<layer>.*):",
								default=str(args.early_stop_metric or "beta1_persistent_est"),
							).execute()
						).strip()
					else:
						args.early_stop_metric = str(mm)
					args.early_stop_mode = str(
						inquirer.select(
							message="Early-stop mode:",
							choices=["max", "min"],
							default=str(args.early_stop_mode),
						).execute()
					)
					signals = [{"layer": str(args.early_stop_layer), "metric": str(args.early_stop_metric), "mode": str(args.early_stop_mode)}]
				args.early_stop_signals = ";".join(
					[_format_early_stop_signal(s["layer"], s["metric"], s["mode"]) for s in signals if s.get("layer") and s.get("metric")]
				)
				args.early_stop_patience = int(
					inquirer.text(message="Early-stop patience (epochs):", default=str(int(args.early_stop_patience))).execute()
				)
				args.early_stop_min_delta = float(
					inquirer.text(message="Early-stop min_delta:", default=str(float(args.early_stop_min_delta))).execute()
				)
				args.early_stop_start_epoch = int(
					inquirer.text(message="Early-stop start epoch:", default=str(int(args.early_stop_start_epoch))).execute()
				)

		if args.list_layers_only:
			print(f"[Model] {args.model}")
			print(f"[Selected layer_names] {len(layer_names)}")
			for x in layer_names:
				print("  ", x)
			print(f"[All modules] count={len(all_modules)}")
			for x in all_modules:
				print("  ", x)
			print("[Parameters]")
			for n, p in model.named_parameters():
				print(f"  {n}\tshape={tuple(p.shape)}")
			return ""

		monitor_cfg = RepresentationMonitorConfig(
			layer_names=layer_names,
			max_samples_per_stage=args.max_samples_per_stage,
			max_points_for_graph=args.max_points_for_graph,
			max_points_for_mtopdiv=max(100, min(800, args.max_points_for_graph * 3)),
			max_eigs=args.max_eigs,
			knn_k_small=args.knn_small,
			knn_k_large=args.knn_large,
			graph_stage=str(args.graph_stage),
			fixed_graph_points=bool(args.fixed_graph_points),
			fixed_graph_seed=int(args.fixed_graph_seed),
			compute_gudhi=bool(args.compute_gudhi),
			gudhi_max_points=int(args.gudhi_max_points),
			gudhi_max_dim=int(args.gudhi_max_dim),
			gudhi_max_edge_length=float(args.gudhi_max_edge_length),
			gudhi_every=int(args.gudhi_every),
			gudhi_grid_n=int(args.gudhi_grid_n),
			gudhi_compute_lifetime_stats=bool(args.gudhi_compute_lifetime_stats),
			gudhi_compute_betti_curve=bool(args.gudhi_compute_betti_curve),
			gudhi_compute_landscape=bool(args.gudhi_compute_landscape),
			gudhi_landscape_k=int(args.gudhi_landscape_k),
			gudhi_compute_silhouette=bool(args.gudhi_compute_silhouette),
			gudhi_silhouette_q=float(args.gudhi_silhouette_q),
			gudhi_compute_persistence_image=bool(args.gudhi_compute_persistence_image),
			gudhi_pi_size=int(args.gudhi_pi_size),
			gudhi_pi_sigma=float(args.gudhi_pi_sigma),
			gudhi_pi_tau=float(args.gudhi_pi_tau),
			compute_hodge=True,
			compute_persistent=True,
			compute_mtopdiv=bool(args.compute_mtopdiv),
			compute_q1_spectra=bool(args.compute_q1_spectra),
			mtopdiv_runs=args.mtopdiv_runs,
			mtopdiv_pdist_device=args.device if str(args.device).startswith("cuda") else "cpu",
			mtopdiv_stage_a=str(args.mtopdiv_stage_a),
			mtopdiv_stage_b=str(args.mtopdiv_stage_b),
			fixed_mtopdiv_points=bool(args.fixed_mtopdiv_points or args.fixed_graph_points),
			fixed_mtopdiv_seed=int(args.fixed_mtopdiv_seed),
			regularization=args.regularization,
			build_triangles=bool(args.build_triangles),
			max_triangles=args.max_triangles,
			verbose=True,
		)
		monitor = RepresentationMonitor(monitor_cfg)

		early_stop_key: Optional[str] = None
		early_stop_signals = _parse_early_stop_signals(str(args.early_stop_signals), default_mode=str(args.early_stop_mode))
		if bool(args.early_stop) and not early_stop_signals:
			# Backward-compatible single-signal config.
			early_stop_signals = [
				{
					"layer": str(args.early_stop_layer),
					"metric": str(args.early_stop_metric),
					"mode": str(args.early_stop_mode),
				}
			]
		early_states: List[Dict[str, Any]] = []
		if bool(args.early_stop):
			if not early_stop_signals:
				raise SelectionValidationError("Early stopping requires at least one signal (use --early_stop_signals or --early_stop_layer/--early_stop_metric).")
			for s in early_stop_signals:
				lay = str(s.get("layer", "")).strip()
				met = str(s.get("metric", "")).strip()
				mod = str(s.get("mode", "max")).strip().lower()
				if not lay:
					raise SelectionValidationError("Early stopping signal has empty layer.")
				if not met:
					raise SelectionValidationError(f"Early stopping signal for layer '{lay}' has empty metric.")
				if lay not in set(layer_names):
					raise SelectionValidationError(f"Early-stop layer '{lay}' is not in selected monitor layers: {layer_names}")
				if not _is_topo_spectral_metric(met):
					raise SelectionValidationError(
						f"Early-stop metric '{met}' must be topology/spectral (prefix one of: beta, hodge, persistent, mtopdiv, gudhi, graph_)."
					)
				if mod not in ("min", "max"):
					raise SelectionValidationError(f"Invalid early-stop mode '{mod}' for signal {lay}:{met}.")
				early_states.append(
					{
						"layer": lay,
						"metric": met,
						"mode": mod,
						"key": f"repr.layers.{lay}.{met}",
						"best": None,
						"bad": 0,
					}
				)
			early_stop_key = str(early_states[0]["key"])

		# Optional noisy val loader (CV only)
		noisy_val_loader = None
		if args.task == "cv" and args.cv_noise_sigma > 0 and bundle.val is not None:
			nd = _NoisyDataset(bundle.val, sigma=args.cv_noise_sigma)
			noisy_val_loader = torch.utils.data.DataLoader(nd, batch_size=args.batch_size, shuffle=False, num_workers=0)

		# Run store + tracker
		store = RunStore("runs/exp", suffix=f"{args.task}_{args.dataset}_{args.model}_ft-{finetune_mode}", unique=True)
		store.write_meta(
			{
				"name": "exp",
				"task": args.task,
				"dataset": args.dataset,
				"model": args.model,
				"finetune": finetune_mode,
				"device": str(device),
				"num_classes": int(num_classes),
				"monitor": asdict(monitor_cfg),
				"args": vars(args),
			}
		)
		checkpoints_dir = os.path.join(store.run_dir, "checkpoints")
		best_ckpt_path = os.path.join(checkpoints_dir, "model_best_main.pt")
		early_ckpt_path = os.path.join(checkpoints_dir, "model_early_signal.pt")
		tracker = ExperimentTracker(
			monitor=monitor,
			benchmarks=[
				BenchmarkSpec(name=f"{args.dataset}-val", dataloader_key="val", metrics=tuple(bench_metrics)),
				BenchmarkSpec(name=f"{args.dataset}-test", dataloader_key="test", metrics=tuple(bench_metrics)),
			],
			store=store,
			cfg=TrackerConfig(run_dir=store.run_dir, eval_every=1, max_eval_batches=(args.max_val_batches or None)),
		)

		# Fine-tune setup
		if finetune_mode == "full":
			pass
		elif finetune_mode == "linear_probe":
			_freeze_all(model)
			_unfreeze_named_prefixes(model, prefixes=("classifier", "fc", "head", "score"))
			# Fallback for models without conventional head names (e.g. nn.Sequential MLP)
			if not any(p.requires_grad for p in model.parameters()):
				_unfreeze_last_linear(model)
		elif finetune_mode == "last_n_params":
			_unfreeze_last_n_params(model, n=args.last_n_params)
		elif finetune_mode == "named_prefixes":
			_freeze_all(model)
			pfx = [p.strip() for p in args.train_prefixes.split(",") if p.strip()]
			if not pfx and args.strict_validation:
				raise SelectionValidationError("No prefixes were provided for --finetune named_prefixes (use --train_prefixes).")
			_unfreeze_named_prefixes(model, prefixes=pfx)
		elif finetune_mode == "named_patterns":
			inc = csv_to_list(args.train_patterns)
			exc = csv_to_list(args.train_exclude_patterns)
			rep = set_trainable_by_name_selection(
				model,
				include=inc,
				exclude=exc,
				use_regex=bool(args.train_regex),
				strict=bool(args.strict_validation),
			)
			if rep.unmatched:
				print("[Trainable selection] unmatched patterns:", rep.unmatched)
		elif finetune_mode == "selected_layers":
			selected_layers = [x for x in csv_to_list(args.train_layers) if x in set(all_modules)]
			if not selected_layers and args.strict_validation:
				raise SelectionValidationError(
					"No trainable layers were provided for --finetune selected_layers (use --train_layers or TUI selection)."
				)
			if not selected_layers:
				# Fallback to monitor layers if user didn't specify explicit trainable layers.
				selected_layers = [x for x in layer_names if x in set(all_modules)]
			# Always keep classifier/head trainable when present.
			head_prefixes = ("classifier", "score", "head", "fc")
			include = [f"{m}.*" for m in selected_layers] + [f"{pfx}.*" for pfx in head_prefixes]
			rep = set_trainable_by_name_selection(
				model,
				include=include,
				exclude=[],
				use_regex=False,
				strict=bool(args.strict_validation),
			)
			if rep.unmatched:
				print("[Trainable selection] unmatched module patterns:", rep.unmatched)
			print(f"[Trainable selection] selected_layers={selected_layers}")
		elif finetune_mode == "tracked_layers":
			# Train only parameters under currently tracked monitor layers (+ always keep classifier/head trainable).
			selected_layers = [x for x in layer_names if x in set(all_modules)]
			if not selected_layers and args.strict_validation:
				raise SelectionValidationError("No tracked layers available to fine-tune (monitor layer list is empty).")
			head_prefixes = ("classifier", "score", "head", "fc")
			include = [f"{m}.*" for m in selected_layers] + [f"{pfx}.*" for pfx in head_prefixes]
			rep = set_trainable_by_name_selection(
				model,
				include=include,
				exclude=[],
				use_regex=False,
				strict=bool(args.strict_validation),
			)
			if rep.unmatched:
				print("[Trainable selection] unmatched module patterns:", rep.unmatched)
			print(f"[Trainable selection] tracked_layers={selected_layers}")
		else:
			raise ValueError(f"Unknown finetune mode: {finetune_mode}")

		params = [p for p in model.parameters() if p.requires_grad]
		if not params:
			raise ValueError(f"No trainable parameters for finetune mode '{finetune_mode}'.")
		opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

		reg_layer = args.reg_layer.strip() or (layer_names[0] if layer_names else "")
		early_signal_logged = False
		early_signal_epoch: Optional[int] = None
		early_signal_main_metric: Optional[float] = None

		best = {"metric": -1.0, "epoch": -1, "bench": "", "metric_name": ""}
		t0_total = time.perf_counter()
		for epoch in range(int(args.epochs)):
			t_epoch0 = time.perf_counter()
			monitor.reset_epoch()

			# Schedule heavy dim=2 / q=1 computations (affects end_epoch only)
			dim2_on = _schedule(epoch, base=args.build_triangles, every=args.dim2_every)
			q1_on = _schedule(epoch, base=args.compute_q1_spectra, every=args.q1_every)
			monitor.cfg.build_triangles = bool(dim2_on)
			monitor.cfg.compute_q1_spectra = bool(q1_on)

			# Train loop
			t0 = time.perf_counter()
			model.train()
			with monitor.attach(model), LayerTaps(model, [reg_layer] if (args.reg_kind != "none" and reg_layer) else []) as taps:
				try:
					train_total = len(train_loader)
				except Exception:
					train_total = None
				if args.max_train_batches:
					train_total = int(min(int(args.max_train_batches), train_total)) if train_total is not None else int(args.max_train_batches)
				train_it = tqdm(
					enumerate(train_loader),
					total=train_total,
					desc=f"train e{epoch}",
					leave=False,
					dynamic_ncols=True,
				)
				for bi, batch in train_it:
					if args.max_train_batches and bi >= args.max_train_batches:
						break
					opt.zero_grad()

					if isinstance(batch, (tuple, list)) and len(batch) >= 2:
						x, y = batch[0].to(device), batch[1].to(device)
						if isinstance(y, torch.Tensor) and y.dim() > 1 and y.shape[-1] == 1:
							y = y.view(-1)
						if isinstance(y, torch.Tensor):
							y = y.long()
						if preprocess is not None:
							x = preprocess(x)
						logits = model(x)
						loss = loss_fn(logits, y)
					elif _is_text_batch(batch):
						batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
						out = model(**batch)
						logits = out.logits if hasattr(out, "logits") else out[0]
						y = batch.get("labels", None)
						loss = out.loss if hasattr(out, "loss") and out.loss is not None else loss_fn(logits, y)
					else:
						continue

					# Optional differentiable regularizer on a chosen layer
					if args.reg_kind == "graph_smoothness" and args.reg_weight > 0 and reg_layer:
						act = taps.outputs.get(reg_layer, None)
						z = _repr_from_activation(act)
						if z is not None:
							pen = _graph_smoothness_penalty(z, k=args.reg_knn_k, max_points=args.reg_max_points)
							loss = loss + float(args.reg_weight) * pen

					loss.backward()
					opt.step()
					monitor.collect("train")
					train_it.set_postfix(loss=float(loss.detach().item()))
			train_s = time.perf_counter() - t0

			# Val pass for representation collection (no grad)
			t0 = time.perf_counter()
			model.eval()
			with torch.no_grad(), monitor.attach(model):
				try:
					val_total = len(val_loader)
				except Exception:
					val_total = None
				if args.max_val_batches:
					val_total = int(min(int(args.max_val_batches), val_total)) if val_total is not None else int(args.max_val_batches)
				val_it = tqdm(
					enumerate(val_loader),
					total=val_total,
					desc=f"val e{epoch}",
					leave=False,
					dynamic_ncols=True,
				)
				for bi, batch in val_it:
					if args.max_val_batches and bi >= args.max_val_batches:
						break
					if isinstance(batch, (tuple, list)) and len(batch) >= 2:
						x = batch[0].to(device)
						if preprocess is not None:
							x = preprocess(x)
						_ = model(x)
					elif _is_text_batch(batch):
						batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
						_ = model(**batch)
					monitor.collect("val")
			val_s = time.perf_counter() - t0

			extra = {"train_s": float(train_s), "val_s": float(val_s), "dim2_on": bool(dim2_on), "q1_on": bool(q1_on)}
			out = tracker.on_epoch_end(
				epoch,
				model=model,
				dataloaders={"train": train_loader, "val": val_loader, "test": loaders.get("test"), "noisy_val": noisy_val_loader},
				loss_fn=loss_fn,
				extra=extra,
			)
			# Short one-line summary for terminal visibility.
			bench = out.get("bench", {}) or {}
			main_key = f"{args.dataset}-val"
			mm = (bench.get(main_key, {}) or {})
			acc = mm.get("accuracy", None)
			f1 = mm.get("f1_macro", None)
			los = mm.get("loss", None)
			rep_s = float(((out.get("timing_s", {}) or {}).get("repr_end_epoch", 0.0) or 0.0))
			bench_s = float(((out.get("timing_s", {}) or {}).get("bench_total", 0.0) or 0.0))
			epoch_s = float(time.perf_counter() - t_epoch0)
			print(
				f"[Epoch {epoch}] val_loss={los} val_acc={acc} val_f1_macro={f1} "
				f"train_s={train_s:.2f} val_s={val_s:.2f} repr_s={rep_s:.2f} bench_s={bench_s:.2f} epoch_s={epoch_s:.2f}"
			)

			# Repr-based early-stop signal (no hard stop).
			if bool(args.early_stop) and epoch >= int(args.early_stop_start_epoch):
				missing = []
				for st in early_states:
					cur = _get_repr_layer_scalar(out, layer=str(st["layer"]), metric=str(st["metric"]))
					if cur is None:
						missing.append(str(st["key"]))
						continue
					if st["best"] is None:
						st["best"] = float(cur)
						st["bad"] = 0
						continue
					if str(st["mode"]) == "max":
						improved = float(cur) > float(st["best"]) + float(args.early_stop_min_delta)
					else:
						improved = float(cur) < float(st["best"]) - float(args.early_stop_min_delta)
					if improved:
						st["best"] = float(cur)
						st["bad"] = 0
					else:
						st["bad"] = int(st["bad"]) + 1
					st["value"] = float(cur)
				if missing:
					msg = f"Early-stop metrics not found or not scalar: {missing}"
					if args.strict_validation:
						raise SelectionValidationError(msg)
					print("[EarlyStop]", msg)
				else:
					pat = int(args.early_stop_patience)
					if str(args.early_stop_aggregate).lower() == "any":
						should_stop = any(int(st.get("bad", 0)) >= pat for st in early_states)
					else:
						should_stop = all(int(st.get("bad", 0)) >= pat for st in early_states)
					if should_stop and not early_signal_logged:
						early_signal_logged = True
						early_signal_epoch = int(epoch)
						early_signal_main_metric = float(mm.get("f1_macro", mm.get("accuracy", -1.0)) or -1.0)
						store.log(
							"early_stop_signal",
							{
								"epoch": int(epoch),
								"aggregate": str(args.early_stop_aggregate),
								"patience": pat,
								"min_delta": float(args.early_stop_min_delta),
								"signals": [
									{
										"key": str(st["key"]),
										"mode": str(st["mode"]),
										"value": float(st.get("value", st.get("best", 0.0) or 0.0)),
										"best": float(st.get("best", 0.0) or 0.0),
										"bad_epochs": int(st.get("bad", 0)),
									}
									for st in early_states
								],
							},
						)
						print(f"[EarlyStopSignal] epoch={epoch} aggregate={args.early_stop_aggregate} patience={pat}")
						for st in early_states:
							print(
								"  "
								f"{st['key']} mode={st['mode']} value={st.get('value', st.get('best'))} "
								f"best={st.get('best')} bad={st.get('bad')}"
							)
						if bool(args.save_models):
							_save_model_checkpoint(
								early_ckpt_path,
								model,
								epoch=int(epoch),
								payload={
									"kind": "early_signal",
									"main_metric": float(early_signal_main_metric),
									"main_metric_name": ("f1_macro" if "f1_macro" in mm else "accuracy"),
									"task": str(args.task),
									"dataset": str(args.dataset),
									"model": str(args.model),
									"num_classes": int(num_classes),
									"finetune": str(finetune_mode),
									"monitor_layers": list(layer_names),
								},
							)
							store.log(
								"checkpoint_saved",
								{
									"epoch": int(epoch),
									"path": early_ckpt_path,
									"kind": "early_signal",
								},
							)
						if bool(args.live_plots):
							_rewrite_progress_figures(
								store.run_dir,
								dataset=str(args.dataset),
								early_stop_key=early_stop_key,
								early_stop_signal_epoch=early_signal_epoch,
			)

			# Extra eval on noisy val (CV)
			if noisy_val_loader is not None:
				from tda_repr.training.benchmarks import evaluate_classification

				m = evaluate_classification(model, noisy_val_loader, loss_fn=loss_fn, max_batches=(args.max_val_batches or None))
				store.log("noisy_val", {"epoch": epoch, "sigma": float(args.cv_noise_sigma), "metrics": m})

			# Track best by f1_macro if available, else accuracy.
			if "f1_macro" in mm:
				score = float(mm.get("f1_macro", -1.0) or -1.0)
				main_metric_name = "f1_macro"
			elif "accuracy" in mm:
				score = float(mm.get("accuracy", -1.0) or -1.0)
				main_metric_name = "accuracy"
			else:
				score = -float(mm.get("loss", 1e9) or 1e9)
				main_metric_name = "loss_neg"
			if score > best["metric"]:
				best = {"metric": score, "epoch": int(epoch), "bench": main_key, "metric_name": main_metric_name}
				store.log("best_so_far", {"epoch": int(epoch), "metric": float(score), "bench": main_key})
				if bool(args.save_models):
					_save_model_checkpoint(
						best_ckpt_path,
						model,
						epoch=int(epoch),
						payload={
							"kind": "best_main",
							"main_metric": float(score),
							"main_metric_name": str(main_metric_name),
							"task": str(args.task),
							"dataset": str(args.dataset),
							"model": str(args.model),
							"num_classes": int(num_classes),
							"finetune": str(finetune_mode),
							"monitor_layers": list(layer_names),
						},
					)
					store.log(
						"checkpoint_saved",
						{
							"epoch": int(epoch),
							"path": best_ckpt_path,
							"kind": "best_main",
							"main_metric_name": str(main_metric_name),
							"main_metric": float(score),
						},
					)

			# Always rewrite progress figures to reflect all epochs so far.
			if bool(args.live_plots):
				_rewrite_progress_figures(
					store.run_dir,
					dataset=str(args.dataset),
					early_stop_key=early_stop_key,
					early_stop_signal_epoch=early_signal_epoch,
				)

		store.log(
			"train_done",
			{
				"total_s": float(time.perf_counter() - t0_total),
				"best": best,
				"early_stop_signal_epoch": (int(early_signal_epoch) if early_signal_epoch is not None else None),
				"checkpoints": {
					"best_main": (best_ckpt_path if os.path.exists(best_ckpt_path) else None),
					"early_signal": (early_ckpt_path if os.path.exists(early_ckpt_path) else None),
				},
			},
		)

		# Auto correlation report (replaces lightweight auto-correlation summary).
		from tools.correlation_report import generate_correlation_report

		corr_report = generate_correlation_report(
			run_dir=store.run_dir,
			out_dir=os.path.join(store.run_dir, "correlations_report"),
			min_common_epochs=3,
			top_k=100,
		)
		store.log("correlation_report", corr_report)
		print(f"[Done] run_dir={store.run_dir}")
		print(f"[Best] {best}")
		print(
			f"[CorrelationReport] out_dir={corr_report['out_dir']} "
			f"pairs={corr_report['pairs']} top={corr_report['top']}"
		)
		return store.run_dir

	# Sweep logic
	if args.sweep_finetune.strip():
		modes = [m.strip() for m in args.sweep_finetune.split(",") if m.strip()]
		run_dirs = []
		for m in modes:
			run_dirs.append(run_one(m))
		print("[Sweep done] run_dirs:")
		for rd in run_dirs:
			print(" ", rd)
	else:
		run_one(args.finetune)


if __name__ == "__main__":
	main()
