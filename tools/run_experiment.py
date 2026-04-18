import argparse
import ctypes
import gc
import os
import platform
import sys
import time
from collections.abc import Mapping
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

# Ensure matplotlib (used indirectly by progress plotting utilities) has a writable cache.
_MPLCFG = os.path.join(_ROOT, ".mplconfig")
try:
	os.makedirs(_MPLCFG, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", _MPLCFG)
except Exception:
	# If we cannot set it here, matplotlib will handle its own fallback.
	pass

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


def _cleanup_memory(*, device: str = "") -> None:
	"""
	Best-effort memory cleanup between epochs.

	This helps long runs on constrained environments (e.g., WSL) by releasing Python cycles,
	clearing CUDA caching allocator, and (on Linux/glibc) trimming the malloc arena.
	"""
	gc.collect()
	try:
		if torch.cuda.is_available() and str(device).startswith("cuda"):
			torch.cuda.empty_cache()
	except Exception as e:
		raise RuntimeError("Failed to run torch.cuda.empty_cache() during cleanup.") from e

	if platform.system() == "Linux":
		try:
			libc = ctypes.CDLL("libc.so.6")
			trim = getattr(libc, "malloc_trim", None)
			if trim is not None:
				trim(0)
		except Exception as e:
			# Non-fatal: trimming is best-effort and can be unavailable on some systems.
			global _CLEANUP_TRIM_WARNED
			if not _CLEANUP_TRIM_WARNED:
				_CLEANUP_TRIM_WARNED = True
				print("[Cleanup] malloc_trim failed:", str(e))


_CLEANUP_TRIM_WARNED = False


def _known_datasets_for_task(task: str, nlp_objective: str = "") -> List[str]:
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
	obj = str(nlp_objective).lower().strip()
	if obj == "generation":
		return ["smol-summarize"]
	# classification
	return ["sst2", "trec6"]


def _known_models_for_task(task: str, nlp_objective: str = "") -> List[str]:
	task = str(task).lower().strip()
	if task == "cv":
		return ["resnet18", "convnext_tiny", "efficientnet_b0", "mlp"]
	obj = str(nlp_objective).lower().strip()
	if obj == "generation":
		return ["smollm2-135m", "smollm2-360m"]
	# classification
	return ["distilbert"]


def _is_smollm_model_key(model_key: str) -> bool:
	k = str(model_key).lower().strip()
	return "smollm" in k


def _default_device_string() -> str:
	# Prefer CUDA, then Apple MPS, then CPU.
	if torch.cuda.is_available():
		return "cuda:0"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def _nlp_tokenizer_name(model_kind: str) -> str:
	k = str(model_kind).lower().strip()
	if k in ("smollm2-360m", "smollm-360m"):
		return "HuggingFaceTB/SmolLM2-360M"
	if k in ("smollm2-135m", "smollm2", "smollm", "smollm-135m"):
		return "HuggingFaceTB/SmolLM2-135M"
	return "distilbert-base-uncased"


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
	for emb_name in ("distilbert.embeddings", "embeddings", "model.embed_tokens", "embed_tokens", "wte"):
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
		# DistilBERT: track more than just a few blocks by default.
		# This improves layer-wise monitoring without requiring manual selection in TUI.
		if best_prefix.startswith("distilbert.transformer.layer") and len(best_ids) <= 24:
			# all transformer blocks
			out.extend([f"{best_prefix}.{i}" for i in best_ids])
			# FFN modules per block (stable tensor outputs)
			out.extend([f"{best_prefix}.{i}.ffn" for i in best_ids])
		else:
			# generic transformers: pick several layers spread across depth (smollm etc.)
			n = len(best_ids)
			k = int(min(12, n))  # cap to keep monitoring overhead bounded
			if k <= 1:
				picks = [best_ids[0]]
			else:
				picks = []
				for qi in range(k):
					j = int(round(qi * (n - 1) / (k - 1)))
					picks.append(best_ids[j])
				picks = sorted(set(picks))
			out.extend([f"{best_prefix}.{i}" for i in picks])

	# DistilBERT seq-classification head (when present)
	for head_name in ("pre_classifier", "classifier"):
		if head_name in all_named:
			out.append(head_name)
	# Common decoder-only / generic heads (if present)
	for head_name in ("model.norm", "norm", "ln_f", "lm_head", "score", "head", "classifier"):
		if head_name in all_named:
			out.append(head_name)

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
			# Task-specific sensible defaults.
			# CV defaults (lr=1e-3) are too aggressive for transformer fine-tuning.
			if str(args.task).lower().strip() == "nlp":
				# Common stable defaults for DistilBERT-like fine-tuning.
				args.lr = 2e-5
				args.weight_decay = 1e-2
				# Keep memory-friendly but not too small.
				args.batch_size = 32
			step += 1
			continue

		if step == 1:
			# NLP: objective first (then model, then dataset).
			if str(args.task).lower().strip() == "nlp":
				obj_choices = ["classification", "generation"] + _back_choice(step)
				obj = inquirer.select(
					message="NLP objective:",
					choices=obj_choices,
					default=(str(args.nlp_objective) if str(args.nlp_objective) in obj_choices else "classification"),
				).execute()
				if obj == BACK:
					step -= 1
					continue
				args.nlp_objective = str(obj)
				# Default benchmark metrics per objective.
				if str(args.nlp_objective) == "generation":
					args.bench_metrics = "loss,loss_assistant_only,ppl"
				step += 1
				continue

			# CV: dataset first (then model).
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
			# For NLP, available models depend on objective.
			model_choices = _known_models_for_task(str(args.task), nlp_objective=str(getattr(args, "nlp_objective", ""))) + [CUSTOM] + _back_choice(step)
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
			# SmolLM is generation-only here and uses only SmolTalk summarize.
			if str(args.task).lower().strip() == "nlp" and _is_smollm_model_key(str(args.model)):
				args.dataset = "smol-summarize"
				args.nlp_objective = "generation"
				args.bench_metrics = "loss,loss_assistant_only,ppl"
			step += 1
			continue

		if step == 3 and str(args.task).lower().strip() != "nlp":
			# CV: objective doesn't exist; advance to the shared next step.
			step += 1
			continue

		if step == 3 and str(args.task).lower().strip() == "nlp":
			# NLP: dataset after objective+model. For generation it's fixed to smol-summarize.
			if str(getattr(args, "nlp_objective", "")).lower().strip() == "generation":
				args.dataset = "smol-summarize"
				step += 1
				continue
			dataset_choices = _known_datasets_for_task(str(args.task), nlp_objective=str(args.nlp_objective)) + [CUSTOM] + _back_choice(step)
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

		if step == 4:
			device_choices = ["cuda:0", "cpu"]
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				device_choices.insert(1, "mps")
			device_choices = device_choices + [CUSTOM] + _back_choice(step)
			d = inquirer.select(
				message="Device:",
				choices=device_choices,
				default=(args.device if args.device in device_choices else _default_device_string()),
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

		if step == 5:
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

		if step == 6:
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

		if step == 7:
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

		if step == 8:
			v = _ask_int("Epochs", int(args.epochs), step)
			if v == BACK:
				step -= 1
				continue
			args.epochs = int(v)
			step += 1
			continue

		if step == 9:
			v = _ask_int("Batch size", int(args.batch_size), step)
			if v == BACK:
				step -= 1
				continue
			args.batch_size = int(v)
			step += 1
			continue

		if step == 10:
			v = _ask_float("Learning rate", float(args.lr), step)
			if v == BACK:
				step -= 1
				continue
			args.lr = float(v)
			step += 1
			continue

		if step == 11:
			v = _ask_float("Weight decay", float(args.weight_decay), step)
			if v == BACK:
				step -= 1
				continue
			args.weight_decay = float(v)
			step += 1
			continue

		if step == 12:
			v = _ask_int("Max train batches (0=no limit)", int(args.max_train_batches), step)
			if v == BACK:
				step -= 1
				continue
			args.max_train_batches = int(v)
			step += 1
			continue

		if step == 13:
			v = _ask_int("Max val batches (0=no limit)", int(args.max_val_batches), step)
			if v == BACK:
				step -= 1
				continue
			args.max_val_batches = int(v)
			step += 1
			continue

		if step == 14:
			# Optional NLP subsampling knobs (kept simple; shuffle+take-N in code).
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			if str(args.dataset).lower().strip() == "smol-summarize":
				msg = "NLP: total examples before train/val split (0 = cap at 20k)"
			else:
				msg = "NLP: max train examples (0 = no subsample)"
			v = _ask_int(msg, int(getattr(args, "nlp_max_train_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_train_examples = int(v)
			step += 1
			continue

		if step == 15:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			# smol-summarize creates its own train/val split after subsampling; skip split-specific knobs here.
			if str(args.dataset).lower().strip() == "smol-summarize":
				args.nlp_max_val_examples = 0
				step += 1
				continue
			v = _ask_int("NLP: max val examples (0 = no subsample)", int(getattr(args, "nlp_max_val_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_val_examples = int(v)
			step += 1
			continue

		if step == 16:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			if str(args.dataset).lower().strip() == "smol-summarize":
				args.nlp_max_test_examples = 0
				step += 1
				continue
			v = _ask_int("NLP: max test examples (0 = no subsample)", int(getattr(args, "nlp_max_test_examples", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_max_test_examples = int(v)
			step += 1
			continue

		if step == 17:
			if str(args.task).lower().strip() != "nlp":
				step += 1
				continue
			v = _ask_int("NLP: subsample seed", int(getattr(args, "nlp_subset_seed", 0)), step)
			if v == BACK:
				step -= 1
				continue
			args.nlp_subset_seed = int(v)
			# Generation-specific knobs (kept minimal; only asked for generation objective).
			if str(getattr(args, "nlp_objective", "")).lower().strip() == "generation":
				v2 = _ask_int("NLP generation: max_len (tokens)", int(getattr(args, "nlp_gen_max_len", 512)), step)
				if v2 == BACK:
					step -= 1
					continue
				args.nlp_gen_max_len = int(v2)
				v3 = _ask_int("NLP generation: max_target_len (tokens)", int(getattr(args, "nlp_gen_max_target_len", 96)), step)
				if v3 == BACK:
					step -= 1
					continue
				args.nlp_gen_max_target_len = int(v3)
			step += 1
			continue

		if step == 18:
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

		if step == 19:
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

		if step == 20:
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

		if step == 21:
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

		if step == 22:
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

	# Quality metrics progress (benchmark metrics). For generation, we plot the chosen
	# generation metrics (e.g., ppl/loss_assistant_only) instead of accuracy/f1.
	bench_prefix = f"bench.{dataset}-val."
	all_scalar_keys = list_scalar_series_keys(recs)
	val_metric_keys = [k for k in all_scalar_keys if k.startswith(bench_prefix)]
	available_metrics = [k[len(bench_prefix) :] for k in val_metric_keys]

	def _metric_title(m: str) -> str:
		m = str(m)
		if m == "f1_macro":
			return "Validation F1 (macro)"
		if m == "precision_macro":
			return "Validation precision (macro)"
		if m == "recall_macro":
			return "Validation recall (macro)"
		if m == "loss_assistant_only":
			return "Validation loss (assistant-only)"
		if m == "ppl":
			return "Validation perplexity"
		if m == "bleu":
			return "Validation BLEU"
		return f"Validation {m}"

	is_generation_like = any(m in set(available_metrics) for m in ("ppl", "bleu", "loss_assistant_only"))
	if is_generation_like:
		priority = ["ppl", "loss_assistant_only", "loss", "bleu", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
	else:
		priority = ["accuracy", "f1_macro", "loss", "precision_macro", "recall_macro", "bleu", "ppl", "loss_assistant_only"]

	selected: List[str] = []
	for m in priority:
		if m in set(available_metrics) and m not in selected:
			selected.append(m)
		if len(selected) >= 3:
			break
	if not selected:
		# Nothing to plot.
		return

	n = len(selected)
	fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6))
	axes_arr = np.atleast_1d(axes).reshape(-1)
	for ax_i, m in enumerate(selected):
		ser = get_series(recs, f"{bench_prefix}{m}")
		ax = axes_arr[ax_i]
		ax.set_title(_metric_title(m))
		if ser:
			ax.plot([e for e, _ in ser], [v for _, v in ser], marker="o")
			if early_stop_signal_epoch is not None:
				v = _series_value_at_epoch(ser, int(early_stop_signal_epoch))
				if v is not None:
					ax.scatter([int(early_stop_signal_epoch)], [v], color="red", s=46, zorder=5)
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


def _save_train_state_checkpoint(
	path: str,
	*,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	lr_scheduler: Optional[Any],
	epoch: int,
	global_step: int,
	payload: Dict[str, Any],
) -> None:
	"""
	Save a resumable training checkpoint (model + optimizer + scheduler).
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	obj: Dict[str, Any] = {
		"epoch": int(epoch),
		"global_step": int(global_step),
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"payload": dict(payload or {}),
	}
	if lr_scheduler is not None:
		try:
			obj["lr_scheduler"] = lr_scheduler.state_dict()
		except Exception as e:
			raise RuntimeError("Failed to serialize lr_scheduler.state_dict() for checkpoint.") from e
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


def _interactive_pick_bench_metrics(current: Tuple[str, ...], *, nlp_objective: str = "") -> Tuple[str, ...]:
	obj = str(nlp_objective).lower().strip()
	if obj == "generation":
		available = ["loss_assistant_only", "ppl", "loss", "bleu"]
	else:
		available = ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "bleu"]
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


def _interactive_pick_bench_metrics_tui(current: Tuple[str, ...], *, nlp_objective: str = "") -> Tuple[str, ...]:
	obj = str(nlp_objective).lower().strip()
	if obj == "generation":
		available = ["loss_assistant_only", "ppl", "loss", "bleu"]
	else:
		available = ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "bleu"]
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
	feats = getattr(ds, "features", None)
	if feats is not None and hasattr(feats, "keys"):
		try:
			keys = list(feats.keys())
		except Exception:
			keys = []
		preferred = ["label", "labels", "coarse_label", "fine_label", "target", "topic", "category", "class"]
		candidates = [k for k in preferred if k in keys] + [k for k in keys if "label" in str(k).lower()]
		# Some datasets use 'topic'/'category' instead of 'label'
		candidates += [k for k in keys if any(s in str(k).lower() for s in ("topic", "category", "class"))]
		# de-dup while preserving order
		seen_c = set()
		candidates = [k for k in candidates if not (k in seen_c or seen_c.add(k))]
		for k in candidates:
			try:
				f = feats[k]  # datasets.Features supports __getitem__
			except Exception:
				try:
					f = feats.get(k, None)  # type: ignore[union-attr]
				except Exception:
					f = None
			if f is not None and hasattr(f, "num_classes"):
				try:
					return int(getattr(f, "num_classes"))
				except Exception:
					continue

	# Last-resort: probe a prefix of labels
	if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
		try:
			n = int(len(ds))
		except Exception:
			n = 0
		if n > 0:
			try:
				ex0 = ds[0]
			except Exception:
				ex0 = None
			if isinstance(ex0, dict):
				ex_keys = list(ex0.keys())
				preferred = ["label", "labels", "coarse_label", "fine_label", "target", "topic", "category", "class"]
				candidates = [k for k in preferred if k in ex_keys] + [k for k in ex_keys if "label" in str(k).lower()]
				candidates += [k for k in ex_keys if any(s in str(k).lower() for s in ("topic", "category", "class"))]
				seen_c = set()
				candidates = [k for k in candidates if not (k in seen_c or seen_c.add(k))]
				label_key = candidates[0] if candidates else None
				if label_key is not None:
					vals: list[int] = []
					n_probe = min(512, n)
					for i in range(n_probe):
						try:
							ex = ds[i]
						except Exception:
							continue
						if not isinstance(ex, dict) or label_key not in ex:
							continue
						v = ex.get(label_key, None)
						try:
							vi = int(v)
						except Exception:
							continue
						if vi >= 0:
							vals.append(vi)
					if vals:
						vmin, vmax = int(min(vals)), int(max(vals))
						# handle 1-based labels (common in some datasets)
						if vmin == 1 and 0 not in set(vals):
							return int(vmax)
						return int(vmax) + 1
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
	# HF tokenizer collate returns `BatchEncoding` (Mapping, not dict).
	return isinstance(batch, Mapping) and ("input_ids" in batch or "attention_mask" in batch or "labels" in batch)


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
	ap.add_argument("--device", type=str, default=_default_device_string())
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
	ap.add_argument(
		"--nlp_scheduler",
		type=str,
		choices=["none", "linear"],
		default="linear",
		help="NLP LR scheduler. 'linear' = linear warmup then linear decay to 0.",
	)
	ap.add_argument(
		"--nlp_warmup_ratio",
		type=float,
		default=0.1,
		help="Warmup fraction of total train steps for NLP (used when --nlp_scheduler=linear).",
	)
	ap.add_argument(
		"--nlp_warmup_steps",
		type=int,
		default=0,
		help="If >0, overrides warmup steps (used when --nlp_scheduler=linear).",
	)
	ap.add_argument(
		"--nlp_objective",
		type=str,
		choices=["classification", "generation"],
		default="classification",
		help="NLP objective: classification (default) or causal-LM generation (prompt->target).",
	)
	ap.add_argument("--max_train_batches", type=int, default=0, help="0 = no limit")
	ap.add_argument("--max_val_batches", type=int, default=0, help="0 = no limit")
	# NLP subsampling (HF datasets)
	ap.add_argument("--nlp_max_train_examples", type=int, default=0, help="If >0, subsample first N train examples for NLP datasets.")
	ap.add_argument("--nlp_max_val_examples", type=int, default=0, help="If >0, subsample first N val examples for NLP datasets.")
	ap.add_argument("--nlp_max_test_examples", type=int, default=0, help="If >0, subsample first N test examples for NLP datasets.")
	ap.add_argument("--nlp_subset_seed", type=int, default=0, help="Seed used when subsampling NLP datasets (shuffle before selecting).")
	# NLP generation batching (SmolLM)
	ap.add_argument("--nlp_gen_max_len", type=int, default=512, help="Max total tokens (prompt+target+eos) for generation batching.")
	ap.add_argument("--nlp_gen_max_target_len", type=int, default=96, help="Max target tokens for generation batching (assistant).")
	ap.add_argument("--nlp_gen_log_token_lens", action="store_true", help="Log token length stats (max/p95/p99) for prompt/target/total.")
	ap.add_argument("--nlp_gen_scan_token_lens", action="store_true", help="Scan the selected dataset subset and print token length stats before training.")
	ap.add_argument("--nlp_gen_scan_token_lens_only", action="store_true", help="If set, run the token length scan and exit before training.")

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

		tok_name = _nlp_tokenizer_name(str(args.model)) if str(args.task).lower().strip() == "nlp" else "distilbert-base-uncased"
		nlp_objective = str(getattr(args, "nlp_objective", "classification")).lower().strip()

		# Enforce SmolLM protocol: generation-only on SmolTalk summarize.
		if str(args.task).lower().strip() == "nlp" and _is_smollm_model_key(str(args.model)):
			if str(nlp_objective) != "generation":
				raise ValueError("SmolLM experiments in this repo are generation-only (nlp_objective='generation').")
			if str(args.dataset) != "smol-summarize":
				raise ValueError("SmolLM experiments in this repo use only dataset='smol-summarize'.")

		# If user didn't override benchmark metrics for generation runs, switch to the
		# paper's metrics: assistant-only validation loss + perplexity.
		if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
			default_cli = "loss,accuracy,precision_macro,recall_macro,f1_macro"
			if str(getattr(args, "bench_metrics", "")).strip() == default_cli:
				args.bench_metrics = "loss,loss_assistant_only,ppl"

		nlp_tokenizer = None
		if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
			tf = _require_transformers_runexp()
			AutoTokenizer = getattr(tf, "AutoTokenizer")
			nlp_tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
			# Ensure padding token exists (Llama-like tokenizers often lack it).
			if getattr(nlp_tokenizer, "pad_token", None) is None:
				eos = getattr(nlp_tokenizer, "eos_token", None)
				if eos is not None:
					nlp_tokenizer.pad_token = eos
				else:
					nlp_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
			if getattr(nlp_tokenizer, "pad_token_id", None) is None:
				raise ValueError("Tokenizer is missing pad_token_id after pad token setup; cannot build padded batches.")
			# Decoder-only generation is more reliable with left padding.
			try:
				nlp_tokenizer.padding_side = "left"
			except Exception as e:
				raise RuntimeError("Failed to set tokenizer.padding_side='left' for generation objective.") from e
		bundle = get_dataset(args.dataset, root=args.data_root, download=args.download, tokenizer_name=tok_name)
		smol_kept_pos: Optional[List[int]] = None
		# SmolTalk summarize: choose subset size (cap 20k), then create train/val split.
		# This uses args.nlp_max_train_examples as the TOTAL number of examples used before splitting.
		if str(args.task).lower().strip() == "nlp" and str(args.dataset) == "smol-summarize":
			if bundle.train is None:
				raise RuntimeError("Dataset 'smol-summarize' failed to load (train split is None).")
			try:
				n_total = int(len(bundle.train))
			except Exception as e:
				raise RuntimeError("Dataset 'smol-summarize' does not support len().") from e
			cap = 20000
			want = int(args.nlp_max_train_examples) if int(args.nlp_max_train_examples) > 0 else cap
			want = max(1, min(int(want), int(cap), int(n_total)))
			base = bundle.train.shuffle(seed=int(args.nlp_subset_seed))
			# Do NOT pre-truncate to `want` here. We will filter out too-long examples (no truncation),
			# scanning the shuffled stream until we accumulate exactly `want` valid samples.
			max_len_cfg = int(getattr(args, "nlp_gen_max_len", 512))
			max_target_cfg = int(getattr(args, "nlp_gen_max_target_len", 96))
			max_len_cfg = max(8, int(max_len_cfg))
			max_target_cfg = max(1, min(int(max_target_cfg), int(max_len_cfg) - 1))
			# Optional: scan token lengths (prompt / target / total) before training.
			if bool(getattr(args, "nlp_gen_scan_token_lens", False)):
				if nlp_tokenizer is None:
					raise RuntimeError("Internal error: tokenizer is required for token length scan.")
				eos_id = getattr(nlp_tokenizer, "eos_token_id", None)
				if eos_id is None:
					raise ValueError("Tokenizer is missing eos_token_id (required for causal-LM batching).")

				def _prompt_target_from_messages(ex: dict) -> Optional[Tuple[str, str]]:
					msgs = ex.get("messages", None)
					if not isinstance(msgs, list) or not msgs:
						return None
					if not isinstance(msgs[0], dict):
						return None
					if not ("role" in msgs[0] and "content" in msgs[0]):
						return None
					last_assistant = None
					for i in range(len(msgs) - 1, -1, -1):
						role = str((msgs[i] or {}).get("role", "")).lower().strip()
						if role == "assistant":
							last_assistant = i
							break
					if last_assistant is None:
						return None
					target = str((msgs[last_assistant] or {}).get("content", "")).strip()
					ctx = msgs[:last_assistant]
					parts: List[str] = []
					for m in ctx:
						if not isinstance(m, dict):
							continue
						role = str(m.get("role", "")).strip().lower()
						content = str(m.get("content", "")).strip()
						if not content:
							continue
						if role == "system":
							parts.append(f"System:\n{content}")
						elif role == "user":
							parts.append(f"User:\n{content}")
						elif role == "assistant":
							parts.append(f"Assistant:\n{content}")
						else:
							parts.append(f"{role.capitalize()}:\n{content}")
					prompt = "\n\n".join(parts).strip()
					prompt = (prompt + "\n\nAssistant:\n") if prompt else "Assistant:\n"
					return prompt, target

				def _lens_for_texts(texts: List[str]) -> List[int]:
					# Use batched tokenization. Prefer 'length' if provided; else fall back to len(input_ids).
					out = nlp_tokenizer(
						texts,
						add_special_tokens=False,
						padding=False,
						truncation=False,
						return_length=True,
					)
					# transformers tokenizers return BatchEncoding (Mapping), not a plain dict.
					if isinstance(out, Mapping) and "length" in out:
						return [int(x) for x in out["length"]]
					ids = out.get("input_ids", None) if isinstance(out, Mapping) else None
					if ids is None:
						raise ValueError("Tokenizer output did not include 'length' or 'input_ids'.")
					return [int(len(x)) for x in ids]

				p_lens: List[int] = []
				t_lens: List[int] = []
				tot_lens: List[int] = []
				bs = 64
				buf_p: List[str] = []
				buf_t: List[str] = []
				# Scan only the intended subset size (<=20k) for speed.
				scan_n = int(min(int(want), int(len(base))))
				scan_ds = base.select(range(scan_n)) if hasattr(base, "select") else base
				for ex in scan_ds:
					if not isinstance(ex, dict):
						continue
					pt = _prompt_target_from_messages(ex)
					if pt is None:
						continue
					prompt, target = pt
					if not target.strip():
						continue
					buf_p.append(prompt)
					buf_t.append(target)
					if len(buf_p) >= bs:
						lp = _lens_for_texts(buf_p)
						lt = _lens_for_texts(buf_t)
						p_lens.extend(lp)
						t_lens.extend(lt)
						tot_lens.extend([int(a) + int(b) + 1 for a, b in zip(lp, lt)])
						buf_p, buf_t = [], []
				if buf_p:
					lp = _lens_for_texts(buf_p)
					lt = _lens_for_texts(buf_t)
					p_lens.extend(lp)
					t_lens.extend(lt)
					tot_lens.extend([int(a) + int(b) + 1 for a, b in zip(lp, lt)])

				def _pct(xs: List[int], q: float) -> int:
					if not xs:
						return 0
					s = sorted(int(x) for x in xs)
					i = int(max(0, min(len(s) - 1, round(q * (len(s) - 1)))))
					return int(s[i])

				max_len_cfg = int(getattr(args, "nlp_gen_max_len", 512))
				max_target_cfg = int(getattr(args, "nlp_gen_max_target_len", 96))
				max_target_cfg = max(1, min(int(max_target_cfg), int(max_len_cfg) - 2))
				max_prompt_cfg = max(8, int(max_len_cfg) - int(max_target_cfg) - 1)
				stats = {
					"subset_n": int(len(p_lens)),
					"prompt": {"max": max(p_lens) if p_lens else 0, "p95": _pct(p_lens, 0.95), "p99": _pct(p_lens, 0.99)},
					"target": {"max": max(t_lens) if t_lens else 0, "p95": _pct(t_lens, 0.95), "p99": _pct(t_lens, 0.99)},
					"total": {"max": max(tot_lens) if tot_lens else 0, "p95": _pct(tot_lens, 0.95), "p99": _pct(tot_lens, 0.99)},
					"max_len_cfg": max_len_cfg,
					"max_target_len_cfg": int(max_target_cfg),
					"max_prompt_len_cfg": int(max_prompt_cfg),
					"frac_total_gt_max_len_cfg": (sum(1 for x in tot_lens if int(x) > max_len_cfg) / float(len(tot_lens))) if tot_lens else 0.0,
					"frac_prompt_gt_max_prompt_len_cfg": (sum(1 for x in p_lens if int(x) > max_prompt_cfg) / float(len(p_lens))) if p_lens else 0.0,
					"frac_target_gt_max_target_len_cfg": (sum(1 for x in t_lens if int(x) > max_target_cfg) / float(len(t_lens))) if t_lens else 0.0,
				}
				print("[TokenLengthScan]", stats)
				if bool(getattr(args, "nlp_gen_scan_token_lens_only", False)):
					raise SystemExit(0)

			# Filter out all examples that would be truncated under (max_len_cfg, max_target_cfg).
			# We then take the first `want` valid samples in shuffled order.
			if nlp_tokenizer is None:
				raise RuntimeError("Internal error: tokenizer is required for smol-summarize filtering.")
			eos = str(getattr(nlp_tokenizer, "eos_token", "") or "")
			kept_pos: List[int] = []
			n_seen = 0
			n_bad_parse = 0
			n_empty_target = 0
			n_too_long_total = 0
			n_too_long_target = 0

			def _prompt_target_from_messages(msgs: Any) -> Optional[Tuple[str, str]]:
				if not isinstance(msgs, list) or not msgs:
					return None
				if not isinstance(msgs[0], dict) or not ("role" in msgs[0] and "content" in msgs[0]):
					return None
				last_assistant = None
				for i in range(len(msgs) - 1, -1, -1):
					role = str((msgs[i] or {}).get("role", "")).lower().strip()
					if role == "assistant":
						last_assistant = i
						break
				if last_assistant is None:
					return None
				target = str((msgs[last_assistant] or {}).get("content", "")).strip()
				ctx = msgs[:last_assistant]
				parts: List[str] = []
				for m in ctx:
					if not isinstance(m, dict):
						continue
					role = str(m.get("role", "")).strip().lower()
					content = str(m.get("content", "")).strip()
					if not content:
						continue
					if role == "system":
						parts.append(f"System:\n{content}")
					elif role == "user":
						parts.append(f"User:\n{content}")
					elif role == "assistant":
						parts.append(f"Assistant:\n{content}")
					else:
						parts.append(f"{role.capitalize()}:\n{content}")
				prompt = "\n\n".join(parts).strip()
				prompt = (prompt + "\n\nAssistant:\n") if prompt else "Assistant:\n"
				return prompt, target

			bs = 64
			for start in range(0, int(len(base)), bs):
				if len(kept_pos) >= int(want):
					break
				chunk = base[start : start + bs]
				msg_list = chunk.get("messages", None) if isinstance(chunk, dict) else None
				if not isinstance(msg_list, list):
					continue
				prompts: List[str] = []
				targets: List[str] = []
				positions: List[int] = []
				for j, msgs in enumerate(msg_list):
					n_seen += 1
					pt = _prompt_target_from_messages(msgs)
					if pt is None:
						n_bad_parse += 1
						continue
					p, t = pt
					if not t.strip():
						n_empty_target += 1
						continue
					prompts.append(p)
					targets.append(t)
					positions.append(int(start + j))
				if not prompts:
					continue

				full_texts = [(p + t + eos) for p, t in zip(prompts, targets)]
				target_texts = [(t + eos) for t in targets]
				enc_full = nlp_tokenizer(full_texts, add_special_tokens=False, padding=False, truncation=False)
				enc_t = nlp_tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=False)
				full_ids = enc_full.get("input_ids", None) if isinstance(enc_full, Mapping) else None
				t_ids = enc_t.get("input_ids", None) if isinstance(enc_t, Mapping) else None
				if not isinstance(full_ids, list) or not isinstance(t_ids, list):
					raise RuntimeError("Tokenizer did not return list input_ids for filtering.")

				for pos, f_ids, tt_ids in zip(positions, full_ids, t_ids):
					if len(kept_pos) >= int(want):
						break
					total_len = int(len(f_ids))
					tlen = int(len(tt_ids))
					if tlen > int(max_target_cfg):
						n_too_long_target += 1
						continue
					if total_len > int(max_len_cfg):
						n_too_long_total += 1
						continue
					kept_pos.append(int(pos))

			if len(kept_pos) < int(want):
				raise RuntimeError(
					"Could not collect enough non-truncated smol-summarize examples under the requested limits: "
					f"want={int(want)} got={int(len(kept_pos))}. "
					f"Try increasing --nlp_gen_max_len/--nlp_gen_max_target_len or reducing the subset size. "
					f"seen={int(n_seen)} bad_parse={int(n_bad_parse)} empty_target={int(n_empty_target)} "
					f"too_long_total={int(n_too_long_total)} too_long_target={int(n_too_long_target)}."
				)

			base = base.select(kept_pos)
			smol_kept_pos = list(int(x) for x in kept_pos)
			print(
				"[SmolSummarize] selected",
				f"n={len(base)}",
				f"seen={n_seen}",
				f"dropped_total={n_too_long_total}",
				f"dropped_target={n_too_long_target}",
				f"bad_parse={n_bad_parse}",
				f"empty_target={n_empty_target}",
				f"max_len={max_len_cfg}",
				f"max_target={max_target_cfg}",
			)
			split = base.train_test_split(test_size=0.1, seed=int(args.nlp_subset_seed))
			bundle.train = split["train"]
			bundle.val = split["test"]
			bundle.test = None
		# Optional NLP dataset subsampling (useful for large HF datasets like yahoo_answers_topics).
		if str(args.task).lower().strip() == "nlp":
			# For smol-summarize we already subsampled + split above.
			if str(args.dataset) != "smol-summarize":
				bundle.train = _maybe_subsample_hf_dataset(bundle.train, int(args.nlp_max_train_examples), int(args.nlp_subset_seed))
				bundle.val = _maybe_subsample_hf_dataset(bundle.val, int(args.nlp_max_val_examples), int(args.nlp_subset_seed))
				bundle.test = _maybe_subsample_hf_dataset(bundle.test, int(args.nlp_max_test_examples), int(args.nlp_subset_seed))

		# Override collate for generation objective: prompt -> target, with token-level labels.
		if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
			if nlp_tokenizer is None:
				raise RuntimeError("Internal error: tokenizer is required for generation objective.")
			# Track token length stats for the actually used subset (post-subsample, pre-split).
			# We keep the lists small (<=20k) so it's safe to store per-example lengths.
			token_len_prompt: List[int] = []
			token_len_target: List[int] = []
			token_len_total: List[int] = []

			def _qa_gen_collate(batch):
				if not batch:
					raise ValueError("Empty batch for generation collate.")
				ex0 = batch[0]
				if not isinstance(ex0, dict):
					raise ValueError("Expected HF dataset items to be dicts.")

				def _get_text(ex: dict, keys: list[str]) -> str:
					parts = []
					for k in keys:
						v = ex.get(k, "")
						s = str(v).strip()
						if s:
							parts.append(s)
					return "\n\n".join(parts)

				def _prompt_target_from_messages(ex: dict) -> Optional[Tuple[str, str]]:
					# SmolTalk-like format: `messages` = list[{role, content}]
					for k in ("messages", "conversation", "conversations"):
						msgs = ex.get(k, None)
						if not isinstance(msgs, list) or not msgs:
							continue
						if not isinstance(msgs[0], dict):
							continue
						if not ("role" in msgs[0] and "content" in msgs[0]):
							continue

						last_assistant = None
						for i in range(len(msgs) - 1, -1, -1):
							role = str((msgs[i] or {}).get("role", "")).lower().strip()
							if role in ("assistant", "bot", "gpt"):
								last_assistant = i
								break
						if last_assistant is None:
							return None

						target = str((msgs[last_assistant] or {}).get("content", "")).strip()
						ctx = msgs[:last_assistant]
						parts: List[str] = []
						for m in ctx:
							if not isinstance(m, dict):
								continue
							role = str(m.get("role", "")).strip().lower()
							content = str(m.get("content", "")).strip()
							if not content:
								continue
							if role == "system":
								parts.append(f"System:\n{content}")
							elif role == "user":
								parts.append(f"User:\n{content}")
							elif role == "assistant":
								parts.append(f"Assistant:\n{content}")
							else:
								parts.append(f"{role.capitalize()}:\n{content}")
						prompt = "\n\n".join(parts).strip()
						if prompt:
							prompt = prompt + "\n\nAssistant:\n"
						else:
							prompt = "Assistant:\n"
						return prompt, target
					return None

				prompts = []
				targets = []
				for ex in batch:
					if not isinstance(ex, dict):
						continue
					mt = _prompt_target_from_messages(ex)
					if mt is not None:
						prompt, target = mt
						prompts.append(prompt)
						targets.append(target)
						continue

					# Yahoo Answers Topics fields (preferred)
					q = _get_text(ex, ["question_title", "question_content"])
					a = _get_text(ex, ["best_answer"])
					# Fallbacks for other datasets
					if not q:
						q = _get_text(ex, ["question", "prompt", "text"])
					if not a:
						a = _get_text(ex, ["answer", "target", "text"])
					if not q or not a:
						raise ValueError(f"Could not infer prompt/target fields from keys={list(ex.keys())}")
					prompt = f"Question:\n{q}\n\nAnswer:\n"
					prompts.append(prompt)
					targets.append(a)

				import torch

				# We intentionally avoid tokenizer.pad(...) here to suppress the fast-tokenizer warning
				# and use the tokenizer __call__ API with padding=True.
				eos = str(getattr(nlp_tokenizer, "eos_token", "") or "")

				full_texts: List[str] = []
				prompt_texts: List[str] = []
				target_texts: List[str] = []
				for p, t in zip(prompts, targets):
					if not str(t).strip():
						continue
					prompt_texts.append(str(p))
					target_texts.append(str(t) + eos)
					full_texts.append(str(p) + str(t) + eos)
				if not full_texts:
					raise ValueError("All samples in batch had empty targets; cannot build generation batch.")

				prompt_enc = nlp_tokenizer(
					prompt_texts,
					add_special_tokens=False,
					padding=True,
					truncation=False,
					return_tensors="pt",
				)
				full_enc = nlp_tokenizer(
					full_texts,
					add_special_tokens=False,
					padding=True,
					truncation=False,
					return_tensors="pt",
				)
				target_enc = nlp_tokenizer(
					target_texts,
					add_special_tokens=False,
					padding=True,
					truncation=False,
					return_tensors="pt",
				)

				# Labels: full input ids with prompt masked out (-100), plus pad masked out.
				labels = full_enc["input_ids"].clone()
				labels[full_enc["attention_mask"] == 0] = -100
				prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()
				for i, pl in enumerate(prompt_lens):
					pl = int(pl)
					if pl > 0:
						labels[i, :pl] = -100

				# Optional: track length stats.
				if bool(getattr(args, "nlp_gen_log_token_lens", False)):
					try:
						token_len_prompt.extend([int(x) for x in prompt_enc["attention_mask"].sum(dim=1).tolist()])
						token_len_target.extend([int(x) for x in target_enc["attention_mask"].sum(dim=1).tolist()])
						token_len_total.extend([int(x) for x in full_enc["attention_mask"].sum(dim=1).tolist()])
					except Exception:
						pass

				out = dict(full_enc)
				out["labels"] = labels
				out["gen_input_ids"] = prompt_enc["input_ids"]
				out["gen_attention_mask"] = prompt_enc["attention_mask"]
				ref_labels = target_enc["input_ids"].clone()
				ref_labels[target_enc["attention_mask"] == 0] = -100
				out["gen_ref_labels"] = ref_labels
				return out

			bundle.collate_fn = _qa_gen_collate
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
			if nlp_objective == "generation":
				tf = _require_transformers_runexp()
				AutoConfig = getattr(tf, "AutoConfig")
				AutoModelForCausalLM = getattr(tf, "AutoModelForCausalLM")
				model_id = _nlp_tokenizer_name(str(args.model))
				if bool(args.pretrained):
					model = AutoModelForCausalLM.from_pretrained(model_id)
				else:
					cfg = AutoConfig.from_pretrained(model_id)
					model = AutoModelForCausalLM.from_config(cfg)
				model.to(device)
				# Training does not benefit from KV caching; disabling it prevents extra allocations
				# and makes layer hooks cheaper/stable.
				try:
					if hasattr(model, "config") and getattr(model, "config", None) is not None:
						model.config.use_cache = False
				except Exception as e:
					raise RuntimeError("Failed to set model.config.use_cache=False for generation training.") from e
				# Avoid repeated transformers warnings during open-end generation:
				# set pad_token_id once on the model/generation config.
				if nlp_tokenizer is not None:
					pad_id = getattr(nlp_tokenizer, "pad_token_id", None)
					if pad_id is None:
						eos_id = getattr(nlp_tokenizer, "eos_token_id", None)
						if isinstance(eos_id, (tuple, list)):
							eos_id = eos_id[0] if eos_id else None
						if eos_id is None:
							raise ValueError("Tokenizer is missing eos_token_id; required for causal-LM generation.")
						pad_id = int(eos_id)
					try:
						model.config.pad_token_id = int(pad_id)
					except Exception as e:
						raise RuntimeError("Failed to set model.config.pad_token_id for generation objective.") from e
					if hasattr(model, "generation_config") and getattr(model, "generation_config", None) is not None:
						try:
							model.generation_config.pad_token_id = int(pad_id)
						except Exception as e:
							raise RuntimeError("Failed to set model.generation_config.pad_token_id for generation objective.") from e
					# If a new PAD token was added to the tokenizer, align model embeddings.
					try:
						tok_len = int(len(nlp_tokenizer))
						emb = model.get_input_embeddings()
						if emb is not None and hasattr(emb, "weight") and hasattr(emb.weight, "shape"):
							emb_n = int(emb.weight.shape[0])
							if tok_len != emb_n:
								model.resize_token_embeddings(tok_len)
								model.to(device)
					except Exception as e:
						raise RuntimeError("Failed to resize model token embeddings after tokenizer special-tokens update.") from e
				layer_names = _pick_transformer_blocks_generic_for_model(model)
				loss_fn = None
			else:
				model, layer_names = _build_text_model(args.model, num_classes=num_classes, device=device, pretrained=args.pretrained)
				loss_fn = nn.CrossEntropyLoss()
		else:
			raise ValueError("Unknown task")

		# Helpful runtime provenance: confirm which model initialization path was used.
		model_name_or_path = getattr(model, "name_or_path", None)
		if not model_name_or_path:
			model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
		print(
			f"[Model] class={model.__class__.__name__} pretrained_flag={bool(args.pretrained)} "
			f"nlp_objective={nlp_objective if str(args.task).lower().strip() == 'nlp' else ''} "
			f"name_or_path={model_name_or_path!r}"
		)

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
				bench_metrics = _interactive_pick_bench_metrics_tui(tuple(bench_metrics), nlp_objective=str(nlp_objective))
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
				bench_metrics = _interactive_pick_bench_metrics(tuple(bench_metrics), nlp_objective=str(nlp_objective))
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
			seq_pooling=("mean_masked" if (str(args.task).lower().strip() == "nlp" and nlp_objective == "generation") else "first_token"),
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
				"model_name_or_path": model_name_or_path,
				"monitor": asdict(monitor_cfg),
				"args": vars(args),
				"smol_summarize_kept_pos": smol_kept_pos,
			}
		)
		checkpoints_dir = os.path.join(store.run_dir, "checkpoints")
		best_ckpt_path = os.path.join(checkpoints_dir, "model_best_main.pt")
		early_ckpt_path = os.path.join(checkpoints_dir, "model_early_signal.pt")
		last_ckpt_path = os.path.join(checkpoints_dir, "model_last.pt")
		bench_specs: List[BenchmarkSpec] = []
		if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
			bench_specs = [
				BenchmarkSpec(
					name=f"{args.dataset}-val",
					dataloader_key="val",
					kind="generation",
					metrics=tuple(bench_metrics),
					tokenizer=nlp_tokenizer,
				),
				BenchmarkSpec(
					name=f"{args.dataset}-test",
					dataloader_key="test",
					kind="generation",
					metrics=tuple(bench_metrics),
					tokenizer=nlp_tokenizer,
				),
			]
		else:
			bench_specs = [
				BenchmarkSpec(name=f"{args.dataset}-val", dataloader_key="val", metrics=tuple(bench_metrics)),
				BenchmarkSpec(name=f"{args.dataset}-test", dataloader_key="test", metrics=tuple(bench_metrics)),
			]

		tracker = ExperimentTracker(
			monitor=monitor,
			benchmarks=bench_specs,
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
		lr_sched = None
		sched_info: Dict[str, Any] = {"kind": "none"}
		if str(args.task).lower().strip() == "nlp" and str(getattr(args, "nlp_scheduler", "none")).lower().strip() == "linear":
			# Linear warmup then linear decay to 0 (common HF fine-tune recipe).
			try:
				epochs = int(args.epochs)
				steps_per_epoch = None
				try:
					steps_per_epoch = int(len(train_loader))
				except Exception:
					steps_per_epoch = None
				if args.max_train_batches and int(args.max_train_batches) > 0:
					steps_per_epoch = (
						int(min(int(args.max_train_batches), steps_per_epoch)) if steps_per_epoch is not None else int(args.max_train_batches)
					)
				if steps_per_epoch is None or steps_per_epoch <= 0:
					steps_per_epoch = 1
				total_steps = max(1, int(epochs) * int(steps_per_epoch))
				warmup_steps = int(getattr(args, "nlp_warmup_steps", 0) or 0)
				if warmup_steps <= 0:
					warmup_steps = int(float(getattr(args, "nlp_warmup_ratio", 0.1)) * float(total_steps))
				warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
				sched_info = {"kind": "linear", "warmup_steps": int(warmup_steps), "total_steps": int(total_steps)}

				def _lr_lambda(step: int) -> float:
					if warmup_steps > 0 and step < warmup_steps:
						return float(step) / float(max(1, warmup_steps))
					den = float(max(1, total_steps - warmup_steps))
					return max(0.0, float(total_steps - step) / den)

				lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
				print(f"[NLP] lr_scheduler=linear warmup_steps={warmup_steps} total_steps={total_steps}")
			except Exception as e:
				print("[NLP] Failed to set LR scheduler:", e)

		reg_layer = args.reg_layer.strip() or (layer_names[0] if layer_names else "")
		early_signal_logged = False
		early_signal_epoch: Optional[int] = None
		early_signal_main_metric: Optional[float] = None

		# Best checkpoint tracking. We always keep "score higher is better":
		# - for metrics to minimize (loss/ppl) we use negative values
		# - for metrics to maximize (accuracy/f1/bleu) we use the raw value
		best = {"metric": -float("inf"), "epoch": -1, "bench": "", "metric_name": ""}
		global_step = 0
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
					mininterval=1.0,
					miniters=10,
					smoothing=0.05,
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
						if nlp_objective == "generation":
							# CausalLM training: use token-level labels with prompt masked by -100.
							fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
							out = model(**fwd)
							if hasattr(out, "loss") and out.loss is not None:
								loss = out.loss
							else:
								raise ValueError("CausalLM forward did not return .loss; ensure batch contains token-level 'labels'.")
						else:
							y = batch.get("labels", None)
							# Avoid letting HF models compute loss internally before we validate labels.
							fwd = {k: v for k, v in batch.items() if k != "labels"}
							out = model(**fwd)
							logits = out.logits if hasattr(out, "logits") else out[0]
							if y is None:
								raise ValueError("Text batch is missing 'labels' field.")
							try:
								y_cpu = y.detach().to("cpu")
								ymin = int(y_cpu.min().item())
								ymax = int(y_cpu.max().item())
								nc = int(logits.shape[-1])
								if ymin < 0 or ymax >= nc:
									raise ValueError(
										f"Label values out of range for classification head: "
										f"min={ymin}, max={ymax}, num_classes={nc}. "
										f"(Hint: dataset label field may be wrong or needs remapping to 0..C-1.)"
									)
							except Exception:
								# If anything goes wrong in validation, proceed; the loss will surface the error.
								pass
							loss = loss_fn(logits, y)
					else:
						continue

					# Fail fast on NaNs/Infs (common when a batch has zero supervised tokens).
					try:
						if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
							raise ValueError("Non-finite loss (NaN/Inf) encountered. Likely cause: batch has no assistant tokens after truncation/masking.")
					except Exception:
						# If isfinite check fails for any reason, let the loss value surface naturally.
						pass

					# Optional differentiable regularizer on a chosen layer
					if args.reg_kind == "graph_smoothness" and args.reg_weight > 0 and reg_layer:
						act = taps.outputs.get(reg_layer, None)
						z = _repr_from_activation(act)
						if z is not None:
							pen = _graph_smoothness_penalty(z, k=args.reg_knn_k, max_points=args.reg_max_points)
							loss = loss + float(args.reg_weight) * pen

					loss.backward()
					# Transformers fine-tuning is often sensitive; clip to keep it stable.
					if str(args.task).lower().strip() == "nlp":
						try:
							torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
						except Exception:
							pass
					opt.step()
					global_step += 1
					if lr_sched is not None:
						try:
							lr_sched.step()
						except Exception:
							pass
					attn = None
					if _is_text_batch(batch):
						try:
							attn = batch.get("attention_mask", None) if hasattr(batch, "get") else None
						except Exception:
							attn = None
					monitor.collect("train", attention_mask=attn)
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
					mininterval=1.0,
					miniters=10,
					smoothing=0.05,
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
						if nlp_objective == "generation":
							fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
							_ = model(**fwd)
						else:
							_ = model(**batch)
					attn = None
					if _is_text_batch(batch):
						try:
							attn = batch.get("attention_mask", None) if hasattr(batch, "get") else None
						except Exception:
							attn = None
					monitor.collect("val", attention_mask=attn)
			val_s = time.perf_counter() - t0

			extra = {"train_s": float(train_s), "val_s": float(val_s), "dim2_on": bool(dim2_on), "q1_on": bool(q1_on)}
			out = tracker.on_epoch_end(
				epoch,
				model=model,
				dataloaders={"train": train_loader, "val": val_loader, "test": loaders.get("test"), "noisy_val": noisy_val_loader},
				loss_fn=loss_fn,
				extra=extra,
			)
			# Save resumable state checkpoint for manual per-epoch continuation.
			if bool(args.save_models):
				_save_train_state_checkpoint(
					last_ckpt_path,
					model=model,
					optimizer=opt,
					lr_scheduler=lr_sched,
					epoch=int(epoch),
					global_step=int(global_step),
					payload={
						"kind": "last",
						"task": str(args.task),
						"dataset": str(args.dataset),
						"model": str(args.model),
						"model_name_or_path": str(model_name_or_path),
						"nlp_objective": str(nlp_objective),
						"finetune": str(finetune_mode),
						"device": str(device),
						"args": vars(args),
						"monitor": asdict(monitor_cfg),
						"monitor_layers": list(layer_names),
						"bench_metrics": list(bench_metrics),
						"scheduler": dict(sched_info),
						"smol_summarize_kept_pos": smol_kept_pos,
					},
				)
			# Release representation buffers early to keep memory stable across epochs.
			monitor.reset_epoch()
			_cleanup_memory(device=str(args.device))
			# Short one-line summary for terminal visibility.
			bench = out.get("bench", {}) or {}
			main_key = f"{args.dataset}-val"
			mm = (bench.get(main_key, {}) or {})
			acc = mm.get("accuracy", None)
			f1 = mm.get("f1_macro", None)
			bleu = mm.get("bleu", None)
			los = mm.get("loss_assistant_only", mm.get("loss", None))
			ppl = mm.get("ppl", None)
			err = mm.get("error", None)
			rep_s = float(((out.get("timing_s", {}) or {}).get("repr_end_epoch", 0.0) or 0.0))
			bench_s = float(((out.get("timing_s", {}) or {}).get("bench_total", 0.0) or 0.0))
			epoch_s = float(time.perf_counter() - t_epoch0)
			err_str = f" val_error={err}" if err else ""
			if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
				print(
					f"[Epoch {epoch}] val_loss_assistant_only={los} val_ppl={ppl} val_bleu={bleu} "
					f"train_s={train_s:.2f} val_s={val_s:.2f} repr_s={rep_s:.2f} bench_s={bench_s:.2f} epoch_s={epoch_s:.2f}"
					f"{err_str}"
				)
			else:
				print(
					f"[Epoch {epoch}] val_loss={los} val_acc={acc} val_f1_macro={f1} "
					f"train_s={train_s:.2f} val_s={val_s:.2f} repr_s={rep_s:.2f} bench_s={bench_s:.2f} epoch_s={epoch_s:.2f}"
					f"{err_str}"
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

			# Track best checkpoint.
			# Generation runs should minimize ppl/loss, while classification runs maximize f1/accuracy.
			if str(args.task).lower().strip() == "nlp" and nlp_objective == "generation":
				ppl_v = mm.get("ppl", None)
				loss_a = mm.get("loss_assistant_only", None)
				loss_v = mm.get("loss", None)
				bleu_v = mm.get("bleu", None)
				if ppl_v is not None:
					score = -float(ppl_v)
					main_metric_name = "ppl_min"
				elif loss_a is not None:
					score = -float(loss_a)
					main_metric_name = "loss_assistant_only_min"
				elif loss_v is not None:
					score = -float(loss_v)
					main_metric_name = "loss_min"
				elif bleu_v is not None:
					score = float(bleu_v)
					main_metric_name = "bleu"
				else:
					score = -float("inf")
					main_metric_name = "none"
			else:
				# Classification: maximize f1_macro if available, else accuracy; fallback to minimizing loss.
				if "f1_macro" in mm:
					score = float(mm.get("f1_macro", -1.0) or -1.0)
					main_metric_name = "f1_macro"
				elif "accuracy" in mm:
					score = float(mm.get("accuracy", -1.0) or -1.0)
					main_metric_name = "accuracy"
				else:
					score = -float(mm.get("loss", 1e9) or 1e9)
					main_metric_name = "loss_min"
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
					"last": (last_ckpt_path if os.path.exists(last_ckpt_path) else None),
				},
			},
		)
		# Token length stats for SmolLM generation (computed on-the-fly inside collate).
		if str(args.task).lower().strip() == "nlp" and str(nlp_objective) == "generation" and bool(getattr(args, "nlp_gen_log_token_lens", False)):
			def _pct(xs: List[int], q: float) -> int:
				if not xs:
					return 0
				s = sorted(int(x) for x in xs)
				i = int(max(0, min(len(s) - 1, round(q * (len(s) - 1)))))
				return int(s[i])
			stats = {
				"max_len_cfg": int(getattr(args, "nlp_gen_max_len", 0) or 0),
				"max_target_len_cfg": int(getattr(args, "nlp_gen_max_target_len", 0) or 0),
				"n_samples_seen": int(len(token_len_total)),
				"prompt": {"max": max(token_len_prompt) if token_len_prompt else 0, "p95": _pct(token_len_prompt, 0.95), "p99": _pct(token_len_prompt, 0.99)},
				"target": {"max": max(token_len_target) if token_len_target else 0, "p95": _pct(token_len_target, 0.95), "p99": _pct(token_len_target, 0.99)},
				"total": {"max": max(token_len_total) if token_len_total else 0, "p95": _pct(token_len_total, 0.95), "p99": _pct(token_len_total, 0.99)},
			}
			store.log("token_length_stats", stats)
			print("[TokenLengthStats]", stats)

		# Auto correlation report (replaces lightweight auto-correlation summary).
		from tools.correlation_report import generate_correlation_report

		corr_report = generate_correlation_report(
			run_dir=store.run_dir,
			out_dir=os.path.join(store.run_dir, "correlations_report"),
			min_common_epochs=3,
			top_k=100,
			negate_bench_loss=(str(args.task).lower().strip() == "nlp" and str(nlp_objective) == "generation"),
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
