import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure matplotlib cache dir is writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections.abc import Mapping
import textwrap
from matplotlib.gridspec import GridSpec

# Allow running as a script: ensure project root (parent of /tools) is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from tda_repr.data import get_dataset, make_dataloaders
from tda_repr.models import LayerTaps, csv_to_list, list_module_names
from tools.run_experiment import (
	_build_cv_model,
	_build_text_model,
	_infer_num_classes,
	_repr_from_activation,
)

_ILLUSTRATION_VERSION = 3


def _load_meta(run_dir: str) -> Dict[str, Any]:
	p = os.path.join(run_dir, "meta.json")
	if not os.path.exists(p):
		return {}
	with open(p, "r", encoding="utf-8") as f:
		return json.load(f)


def _resolve_checkpoint_path(run_dir: str, checkpoint: str) -> str:
	ck = str(checkpoint).strip()
	if os.path.exists(ck):
		return ck
	if ck in {"best_main", "best"}:
		return os.path.join(run_dir, "checkpoints", "model_best_main.pt")
	if ck in {"early_signal", "early"}:
		return os.path.join(run_dir, "checkpoints", "model_early_signal.pt")
	return os.path.join(run_dir, "checkpoints", ck)


def _read_signal_layers(run_dir: str) -> List[str]:
	out: List[str] = []
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.exists(metrics_path):
		return out
	try:
		with open(metrics_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				obj = json.loads(line)
				if obj.get("event") != "early_stop_signal":
					continue
				for s in obj.get("signals", []):
					key = str(s.get("key", ""))
					pref = "repr.layers."
					if key.startswith(pref):
						rest = key[len(pref) :]
						if "." in rest:
							layer = rest.rsplit(".", 1)[0]
							if layer and layer not in out:
								out.append(layer)
	except Exception:
		return out
	return out


def _safe_name(s: str) -> str:
	return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))


def _auto_device_string() -> str:
	# Prefer MPS on macOS when available; otherwise CUDA; otherwise CPU.
	try:
		if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return "mps"
	except Exception:
		pass
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_device(preferred: str) -> str:
	"""
	Resolve device string safely on heterogeneous machines.
	- If user explicitly passes --device, torch will validate it via torch.device().
	- If device comes from meta.json and is invalid (e.g., 'cuda:0' on macOS), fall back to mps/cpu.
	"""
	dev = str(preferred).strip()
	if not dev:
		return _auto_device_string()
	low = dev.lower()
	if low.startswith("cuda"):
		if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)) and torch.cuda.is_available():
			return dev
		try:
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				return "mps"
		except Exception:
			pass
		return "cpu"
	if low == "mps":
		try:
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				return "mps"
		except Exception:
			pass
		raise ValueError("Requested device 'mps', but torch.backends.mps.is_available() is False on this machine.")
	if low == "cpu":
		return "cpu"
	return dev


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
	# torch tensor
	if isinstance(x, torch.Tensor):
		return x
	# tuple/list
	if isinstance(x, (list, tuple)):
		for it in x:
			t = _first_tensor(it)
			if t is not None:
				return t
		return None
	# mapping / dict-like
	if isinstance(x, (dict, Mapping)) or (hasattr(x, "keys") and hasattr(x, "__getitem__")):
		try:
			keys = list(x.keys())  # type: ignore[attr-defined]
		except Exception:
			keys = []
		for key in ("last_hidden_state", "hidden_states", "logits"):
			if key in keys:
				try:
					v = x[key]  # type: ignore[index]
				except Exception:
					continue
				t = _first_tensor(v)
				if t is not None:
					return t
		try:
			vals = list(x.values())  # type: ignore[attr-defined]
		except Exception:
			vals = []
		for v in vals:
			t = _first_tensor(v)
			if t is not None:
				return t
		return None
	# HF ModelOutput often provides .to_tuple()
	if hasattr(x, "to_tuple") and callable(getattr(x, "to_tuple")):
		try:
			return _first_tensor(x.to_tuple())
		except Exception:
			return None
	return None


def _repr_from_hook_output(act: Any) -> Optional[torch.Tensor]:
	# Try existing fast path first (tensor only).
	z = _repr_from_activation(act)
	if isinstance(z, torch.Tensor):
		return z
	t = _first_tensor(act)
	if t is None:
		return None
	x = t
	if not x.is_floating_point():
		x = x.float()
	if x.dim() == 4:
		x = x.mean(dim=(2, 3))
	elif x.dim() == 3:
		x = x[:, 0, :]
	elif x.dim() == 2:
		pass
	else:
		x = x.view(x.shape[0], -1)
	return x


def _metric_group_from_repr_key(repr_key: str) -> str:
	"""
	Classify repr metric key for layer auto-selection.
	Returns one of: spectral, topo_mtopdiv, other.
	"""
	k = str(repr_key)
	if ".hodge_L_q0_lambda" in k or ".hodge_L_q1_lambda" in k or ".persistent_q0_lambda" in k or ".persistent_q1_lambda" in k:
		return "spectral"
	if any(x in k for x in (".beta", ".mtopdiv", ".gudhi_", ".graph_")):
		return "topo_mtopdiv"
	return "other"


def _extract_layer_from_repr_key(repr_key: str) -> Optional[str]:
	pref = "repr.layers."
	k = str(repr_key)
	if not k.startswith(pref):
		return None
	rest = k[len(pref) :]
	if "." not in rest:
		return None
	return rest.rsplit(".", 1)[0]


def _read_top_layers_from_correlations(
	csv_path: str,
	selection_group: str,
	top_n_layers: int,
	min_abs_rho: float,
	max_p: float,
) -> List[str]:
	"""
	Select unique layers by descending |rho| from correlation report.
	selection_group: spectral | topo_mtopdiv
	"""
	if not os.path.exists(csv_path):
		return []
	rows: List[Tuple[float, str]] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			repr_key = str(r.get("repr_key", "") or "")
			if _metric_group_from_repr_key(repr_key) != str(selection_group):
				continue
			try:
				abs_rho = float(r.get("abs_rho", "nan"))
				p_val = float(r.get("p", "nan"))
			except Exception:
				continue
			if not np.isfinite(abs_rho) or not np.isfinite(p_val):
				continue
			if abs_rho < float(min_abs_rho) or p_val > float(max_p):
				continue
			layer = _extract_layer_from_repr_key(repr_key)
			if not layer:
				continue
			rows.append((float(abs_rho), str(layer)))
	rows.sort(key=lambda x: x[0], reverse=True)
	seen = set()
	out: List[str] = []
	for _rho, layer in rows:
		if layer in seen:
			continue
		seen.add(layer)
		out.append(layer)
		if len(out) >= int(top_n_layers):
			break
	return out


def _can_reuse_layer_report(
	layer_json_path: str,
	run_dir: str,
	ckpt_path: str,
	split: str,
	top_k: int,
	anchors_per_class: int,
	seed: int,
	max_batches: int,
	max_samples: int,
	require_illustration: bool,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
	if not os.path.exists(layer_json_path):
		return False, None
	try:
		obj = json.load(open(layer_json_path, "r", encoding="utf-8"))
	except Exception:
		return False, None
	# Be path-robust: artifacts may be moved across machines/workspaces.
	# Prefer strict equality; fallback to basename check.
	obj_run_dir = str(obj.get("run_dir", "") or "")
	obj_ckpt = str(obj.get("checkpoint", "") or "")
	if obj_run_dir and str(obj_run_dir) != str(run_dir):
		if os.path.basename(os.path.normpath(obj_run_dir)) != os.path.basename(os.path.normpath(run_dir)):
			return False, None
	if obj_ckpt and str(obj_ckpt) != str(ckpt_path):
		if os.path.basename(obj_ckpt) != os.path.basename(str(ckpt_path)):
			return False, None
	if str(obj.get("split", "")) != str(split):
		return False, None
	if int(obj.get("top_k", -1)) != int(top_k):
		return False, None
	if int(obj.get("anchors_per_class", -1)) != int(anchors_per_class):
		return False, None
	if int(obj.get("seed", -1)) != int(seed):
		return False, None
	if int(obj.get("max_batches", 0)) != int(max_batches):
		return False, None
	if int(obj.get("max_samples", 0)) != int(max_samples):
		return False, None
	if bool(require_illustration):
		ill = obj.get("illustration_path", None)
		# If illustration is missing or the file is absent, force recompute to generate PNG.
		if not ill:
			return False, None
		if int(obj.get("illustration_version", 0) or 0) != int(_ILLUSTRATION_VERSION):
			return False, None
		try:
			if not os.path.exists(str(ill)):
				return False, None
		except Exception:
			return False, None
	return True, obj


def _collect_embeddings_and_labels(
	model: torch.nn.Module,
	loader: Any,
	layer_name: str,
	preprocess: Optional[Any],
	device: torch.device,
	max_batches: int,
	max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
	embeds: List[torch.Tensor] = []
	labels: List[torch.Tensor] = []
	model.eval()
	with torch.no_grad(), LayerTaps(model, [layer_name]) as taps:
		seen_batches = 0
		seen_samples = 0
		for batch in loader:
			if int(max_batches) > 0 and int(seen_batches) >= int(max_batches):
				break
			# CV: batch = (x, y). NLP: batch is mapping with input_ids/... and labels.
			if isinstance(batch, (tuple, list)) and len(batch) >= 2:
				x = batch[0].to(device)
				y = batch[1]
				if not isinstance(y, torch.Tensor):
					y = torch.as_tensor(y)
				y = y.view(-1).long().detach().cpu()
				if preprocess is not None:
					x = preprocess(x)
				_ = model(x)
			elif isinstance(batch, Mapping) or (hasattr(batch, "keys") and hasattr(batch, "__getitem__")):
				try:
					y = batch.get("labels", None) if hasattr(batch, "get") else batch["labels"]  # type: ignore[index]
				except Exception:
					y = None
				if y is None:
					try:
						y = batch.get("label", None) if hasattr(batch, "get") else batch["label"]  # type: ignore[index]
					except Exception:
						y = None
				if y is None:
					continue
				if not isinstance(y, torch.Tensor):
					y = torch.as_tensor(y)
				y = y.view(-1).long().detach().cpu()

				inputs: Dict[str, Any] = {}
				for k in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
					try:
						v = batch.get(k, None) if hasattr(batch, "get") else batch[k]  # type: ignore[index]
					except Exception:
						v = None
					if isinstance(v, torch.Tensor):
						inputs[k] = v.to(device)
				if not inputs:
					continue
				_ = model(**inputs)
			else:
				continue

			z = _repr_from_hook_output(taps.outputs.get(layer_name, None))
			if z is None:
				continue
			z = z.detach().to("cpu")
			# Ignore unknown labels (e.g., -1 in some test splits).
			if isinstance(y, torch.Tensor) and y.numel() == int(z.shape[0]):
				mask = y >= 0
				if bool(mask.any().item()) is False:
					continue
				z = z[mask]
				y = y[mask]
			embeds.append(z)
			labels.append(y)
			seen_batches += 1
			seen_samples += int(y.numel())
			if int(max_samples) > 0 and int(seen_samples) >= int(max_samples):
				break
	if not embeds:
		raise RuntimeError(f"Could not collect embeddings from layer '{layer_name}'.")
	return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def _collect_embeddings_and_labels_many(
	model: torch.nn.Module,
	loader: Any,
	layer_names: List[str],
	preprocess: Optional[Any],
	device: torch.device,
	max_batches: int,
	max_samples: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
	"""
	Collect embeddings for multiple layers in a single forward pass per batch.
	This is substantially faster than running one forward per layer.
	Returns: layer -> (embeddings[N,D] on CPU, labels[N] on CPU)
	"""
	embeds_by_layer: Dict[str, List[torch.Tensor]] = {str(k): [] for k in layer_names}
	labels_by_layer: Dict[str, List[torch.Tensor]] = {str(k): [] for k in layer_names}
	model.eval()
	with torch.no_grad(), LayerTaps(model, layer_names) as taps:
		seen_batches = 0
		seen_samples = 0
		for batch in loader:
			if int(max_batches) > 0 and int(seen_batches) >= int(max_batches):
				break

			# Prepare inputs + labels for CV/NLP
			y: Optional[torch.Tensor] = None
			if isinstance(batch, (tuple, list)) and len(batch) >= 2:
				x = batch[0].to(device)
				yy = batch[1]
				if not isinstance(yy, torch.Tensor):
					yy = torch.as_tensor(yy)
				y = yy.view(-1).long().detach().cpu()
				if preprocess is not None:
					x = preprocess(x)
				_ = model(x)
			elif isinstance(batch, Mapping) or (hasattr(batch, "keys") and hasattr(batch, "__getitem__")):
				try:
					yy = batch.get("labels", None) if hasattr(batch, "get") else batch["labels"]  # type: ignore[index]
				except Exception:
					yy = None
				if yy is None:
					try:
						yy = batch.get("label", None) if hasattr(batch, "get") else batch["label"]  # type: ignore[index]
					except Exception:
						yy = None
				if yy is None:
					continue
				if not isinstance(yy, torch.Tensor):
					yy = torch.as_tensor(yy)
				y = yy.view(-1).long().detach().cpu()

				inputs: Dict[str, Any] = {}
				for k in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
					try:
						v = batch.get(k, None) if hasattr(batch, "get") else batch[k]  # type: ignore[index]
					except Exception:
						v = None
					if isinstance(v, torch.Tensor):
						inputs[k] = v.to(device)
				if not inputs:
					continue
				_ = model(**inputs)
			else:
				continue

			if y is None:
				continue
			# Valid label mask (ignore -1)
			mask = y >= 0
			if bool(mask.any().item()) is False:
				continue

			for layer in layer_names:
				act = taps.outputs.get(layer, None)
				z = _repr_from_hook_output(act)
				if z is None:
					continue
				z = z.detach().to("cpu")
				if int(z.shape[0]) != int(y.numel()):
					continue
				zz = z[mask]
				yy2 = y[mask]
				if int(zz.shape[0]) == 0:
					continue
				embeds_by_layer[str(layer)].append(zz)
				labels_by_layer[str(layer)].append(yy2)

			seen_batches += 1
			seen_samples += int(mask.sum().item())
			if int(max_samples) > 0 and int(seen_samples) >= int(max_samples):
				break

	out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
	for layer in layer_names:
		ee = embeds_by_layer.get(str(layer), [])
		ll = labels_by_layer.get(str(layer), [])
		if not ee or not ll:
			continue
		out[str(layer)] = (torch.cat(ee, dim=0), torch.cat(ll, dim=0))
	if not out:
		raise RuntimeError("Could not collect embeddings for any requested layer.")
	return out


def _choose_per_class_anchors(labels_np: np.ndarray, seed: int, anchors_per_class: int) -> Dict[int, List[int]]:
	classes, counts = np.unique(labels_np, return_counts=True)
	valid_classes = [int(c) for c, cnt in zip(classes.tolist(), counts.tolist()) if int(cnt) >= 2]
	rng = np.random.default_rng(int(seed))
	out: Dict[int, List[int]] = {}
	k_req = max(1, int(anchors_per_class))
	for cls in valid_classes:
		ids = np.where(labels_np == cls)[0]
		k = min(k_req, int(ids.shape[0]))
		chosen = rng.choice(ids, size=k, replace=False)
		out[int(cls)] = [int(x) for x in np.asarray(chosen).tolist()]
	return out


def _score_anchor(emb: torch.Tensor, labels: torch.Tensor, anchor_idx: int, anchor_class: int, top_k: int) -> Dict[str, Any]:
	e = emb.float()
	n = int(e.shape[0])
	k = max(1, min(int(top_k), n - 1))
	d = torch.cdist(e[anchor_idx : anchor_idx + 1], e).view(-1)
	d[int(anchor_idx)] = float("inf")
	nn_idx = torch.topk(d, k=k, largest=False).indices
	nn_labels = labels[nn_idx]
	hits = int((nn_labels == int(anchor_class)).sum().item())
	return {
		"top_k": int(k),
		"same_class_count": int(hits),
		"same_class_ratio": (float(hits) / float(k)),
		"neighbor_indices": [int(i) for i in nn_idx.detach().cpu().tolist()],
	}


def _to_hwc_image(x: torch.Tensor) -> np.ndarray:
	img = x.detach().cpu()
	if img.dim() == 2:
		arr = img.numpy()
		arr = np.clip(arr, 0.0, 1.0)
		return arr
	if img.dim() == 3:
		if img.shape[0] in (1, 3):
			arr = img.permute(1, 2, 0).numpy()
		else:
			arr = img.numpy()
		arr = np.clip(arr, 0.0, 1.0)
		if arr.shape[-1] == 1:
			arr = arr[:, :, 0]
		return arr
	arr = img.view(-1).numpy()
	s = int(np.sqrt(arr.size))
	arr = arr[: s * s].reshape(s, s)
	return np.clip(arr, 0.0, 1.0)


def _fetch_dataset_image(dataset: Any, idx: int) -> torch.Tensor:
	item = dataset[int(idx)]
	if isinstance(item, (tuple, list)) and len(item) >= 1:
		x = item[0]
	else:
		x = item
	if not isinstance(x, torch.Tensor):
		x = torch.as_tensor(x)
	if x.dim() == 2:
		x = x.unsqueeze(0)
	return x.float()


def _fetch_dataset_text(dataset: Any, idx: int) -> str:
	"""
	Best-effort extraction of a human-readable text field from a dataset item.
	Works for common HF text datasets (SST-2, AG News, Yahoo Answers Topics, etc.).
	"""
	ex = dataset[int(idx)]
	# Some wrappers may return (text, label) or similar.
	if isinstance(ex, (tuple, list)) and ex:
		for it in ex:
			if isinstance(it, str) and it.strip():
				return str(it)
		# fallback to first element
		ex = ex[0]
	if isinstance(ex, str):
		return ex
	if not isinstance(ex, dict):
		return str(ex)

	def _get(k: str) -> str:
		v = ex.get(k, "")
		s = str(v).strip()
		return s

	# Yahoo Answers Topics: join fields (question + answer)
	if any(k in ex for k in ("question_title", "question_content", "best_answer")):
		parts = []
		for k in ("question_title", "question_content", "best_answer"):
			s = _get(k)
			if s:
				parts.append(s)
		if parts:
			return "\n\n".join(parts)

	# Common single-field datasets
	for k in ("text", "sentence", "content", "question", "review", "prompt"):
		s = _get(k)
		if s:
			return s

	# Fallback: first non-empty string-ish field
	for k, v in ex.items():
		if isinstance(v, str) and v.strip():
			return v.strip()
	return str(ex)


def _make_layer_illustration_text(
	out_path: str,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	anchor_class: int,
	neighbor_indices: List[int],
) -> None:
	"""
	Render a compact, readable layout:
	- Anchor text (yellow) spans full width at the top and is centered.
	- Up to 20 neighbors below in 2 columns (green same class, red different class).
	"""
	col_anchor = "#C9A227"  # golden/yellow
	col_ok = "#228B22"      # forest green
	col_bad = "#B22222"     # firebrick

	def _clean(s: str) -> str:
		s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
		paras = []
		for p in s.split("\n"):
			p2 = " ".join(p.split()).strip()
			if p2:
				paras.append(p2)
		return "\n\n".join(paras)

	anchor_id = int(anchor_idx)
	neighbor_ids = [int(i) for i in neighbor_indices[:20]]
	nn = len(neighbor_ids)

	# More compact layout so anchor is visually closer to neighbors.
	n_rows = 1 + int(np.ceil(max(1, nn) / 2.0))
	fig_h = 0.72 + 0.48 * (n_rows - 1)
	fig = plt.figure(figsize=(11.2, fig_h))
	gs = GridSpec(
		nrows=n_rows,
		ncols=2,
		figure=fig,
		height_ratios=[0.68] + [0.48] * (n_rows - 1),
		wspace=0.01,
		hspace=0.03,
	)

	def _draw_cell(ax, sid: int, color: str, is_anchor: bool) -> None:
		ax.axis("off")
		lab = int(labels[sid].item())
		txt = _clean(_fetch_dataset_text(dataset, sid))
		if not txt:
			txt = "[empty]"

		if is_anchor:
			max_chars = 420
			wrap_w = 95
			fs = 8.6
		else:
			max_chars = 150
			wrap_w = 54
			fs = 7.6

		if len(txt) > max_chars:
			txt = txt[: max_chars - 1] + "…"

		prefix = f"id={sid} y={lab} — "
		wrap = textwrap.TextWrapper(width=wrap_w, break_long_words=False, break_on_hyphens=False)
		wrapped = wrap.fill(prefix + txt)

		if is_anchor:
			x_pos, y_pos = 0.5, 0.45
			ha, va = "center", "center"
		else:
			x_pos, y_pos = 0.01, 0.98
			ha, va = "left", "top"

		ax.text(
			x_pos,
			y_pos,
			wrapped,
			va=va,
			ha=ha,
			transform=ax.transAxes,
			fontsize=fs,
			color=color,
			wrap=True,
			bbox=dict(
				boxstyle="round,pad=0.12",
				facecolor="white",
				edgecolor="#DDDDDD",
				linewidth=0.7,
			),
		)

	# Anchor spans both columns.
	ax0 = fig.add_subplot(gs[0, :])
	_draw_cell(ax0, anchor_id, col_anchor, is_anchor=True)

	# Neighbors in 2 columns.
	for r in range(1, n_rows):
		for c in range(2):
			j = (r - 1) + (n_rows - 1) * c
			ax = fig.add_subplot(gs[r, c])
			ax.axis("off")
			if j >= nn:
				continue
			sid = neighbor_ids[j]
			lab = int(labels[sid].item())
			color = col_ok if lab == int(anchor_class) else col_bad
			_draw_cell(ax, sid, color, is_anchor=False)

	fig.suptitle("Anchor (yellow) and nearest neighbors (green/red)", fontsize=9.5)
	fig.tight_layout(pad=0.08, rect=(0, 0, 1, 0.96))
	fig.savefig(out_path)
	plt.close(fig)


def _make_layer_illustration(
	out_path: str,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	anchor_class: int,
	neighbor_indices: List[int],
) -> None:
	fig, axes = plt.subplots(3, 7, figsize=(14.0, 6.2))
	axes_arr = axes.reshape(-1)
	all_ids = [int(anchor_idx)] + [int(i) for i in neighbor_indices[:20]]
	for i in range(21):
		ax = axes_arr[i]
		if i < len(all_ids):
			sid = int(all_ids[i])
			img = _fetch_dataset_image(dataset, sid)
			arr = _to_hwc_image(img)
			if arr.ndim == 2:
				ax.imshow(arr, cmap="gray")
			else:
				ax.imshow(arr)
			lab = int(labels[sid].item())
			ax.set_title(f"id={sid}, y={lab}", fontsize=8)
			if i == 0:
				col = "yellow"
				lw = 3.0
			else:
				col = "green" if lab == int(anchor_class) else "red"
				lw = 2.5
			for spine in ax.spines.values():
				spine.set_edgecolor(col)
				spine.set_linewidth(lw)
		ax.set_xticks([])
		ax.set_yticks([])
	for ax in axes_arr[21:]:
		ax.axis("off")
	fig.suptitle("Anchor (yellow) and nearest neighbors (green same class, red different class)", fontsize=11)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def _make_neighbor_pair_illustrations(
	out_dir: str,
	file_prefix: str,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	anchor_class: int,
	neighbor_indices: List[int],
	top_k_pairs: int,
) -> List[str]:
	"""
	Save one PNG per neighbor: (anchor, neighbor_i) side-by-side.
	Returns list of written file paths.
	"""
	os.makedirs(out_dir, exist_ok=True)
	k = int(min(int(top_k_pairs), len(neighbor_indices)))
	written: List[str] = []
	for i in range(k):
		nid = int(neighbor_indices[i])
		nlab = int(labels[int(nid)].item())
		ok = (int(nlab) == int(anchor_class))

		xa = _fetch_dataset_image(dataset, int(anchor_idx))
		xn = _fetch_dataset_image(dataset, int(nid))
		ima = _to_hwc_image(xa)
		imn = _to_hwc_image(xn)

		fig = plt.figure(figsize=(6.0, 3.2), dpi=200)
		ax1 = fig.add_subplot(1, 2, 1)
		ax2 = fig.add_subplot(1, 2, 2)
		ax1.imshow(ima)
		ax2.imshow(imn)
		for ax in (ax1, ax2):
			ax.set_xticks([])
			ax.set_yticks([])
		ax1.set_title(f"Anchor (id={int(anchor_idx)})", fontsize=10)
		ax2.set_title(f"#{i+1} (id={int(nid)})", fontsize=10)
		col = "green" if ok else "red"
		for ax in (ax1, ax2):
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(3.0)
				spine.set_edgecolor(col)
		fig.suptitle(
			f"anchor y={int(anchor_class)} vs neighbor y={int(nlab)} ({'match' if ok else 'mismatch'})",
			fontsize=10,
			y=0.98,
		)
		fig.tight_layout(rect=(0, 0, 1, 0.92))

		out_path = os.path.join(out_dir, f"{file_prefix}__pair_{i+1:02d}.png")
		fig.savefig(out_path)
		plt.close(fig)
		written.append(out_path)
	return written


def _make_two_layer_topk_comparison(
	out_path: str,
	dataset: Any,
	labels: torch.Tensor,
	*,
	anchor_idx: int,
	anchor_class: int,
	layer_a: str,
	neighbors_a: List[int],
	layer_b: str,
	neighbors_b: List[int],
	top_k: int,
) -> None:
	"""
	Render one figure that shows the same anchor with top-k neighbors for two layers.
	Layout: two blocks (A then B), each a 2x6 grid (12 cells) with 11 used (anchor + k neighbors).
	"""
	# 2x6 grid: fits anchor + 11 neighbors perfectly.
	k = int(max(1, min(int(top_k), 11)))
	na = [int(x) for x in neighbors_a[:k]]
	nb = [int(x) for x in neighbors_b[:k]]

	fig = plt.figure(figsize=(12.0, 8.2), dpi=220)

	# Reserve a spacer row between blocks so the second label is never clipped.
	# CHANGED: Reduced image row heights from 1.0 to 0.9 to create more vertical space for labels.
	gs = GridSpec(
		nrows=5,
		ncols=6,
		figure=fig,
		height_ratios=[0.9, 0.9, 0.30, 0.9, 0.9],
		# Slightly larger vertical spacing so rows are not visually glued.
		hspace=0.10,
		wspace=0.03,
	)

	def _draw_block(row0: int, layer_name: str, neigh: List[int]) -> List[plt.Axes]:
		all_ids = [int(anchor_idx)] + [int(i) for i in neigh]
		axes: List[plt.Axes] = []
		for pos in range(12):
			r = row0 + (pos // 6)
			c = pos % 6
			ax = fig.add_subplot(gs[r, c])
			axes.append(ax)
			ax.set_xticks([])
			ax.set_yticks([])
			if pos >= len(all_ids):
				ax.axis("off")
				continue
			sid = int(all_ids[pos])
			img = _fetch_dataset_image(dataset, sid)
			arr = _to_hwc_image(img)
			if arr.ndim == 2:
				ax.imshow(arr, cmap="gray")
			else:
				ax.imshow(arr)
			if pos == 0:
				col = "yellow"
				lw = 3.0
			else:
				lab = int(labels[sid].item())
				col = "green" if lab == int(anchor_class) else "red"
				lw = 2.5
			# No per-tile titles (ids / classes / indices) to keep the figure clean.
			for spine in ax.spines.values():
				spine.set_edgecolor(col)
				spine.set_linewidth(lw)
		return axes

	axes_a = _draw_block(0, str(layer_a), na)
	axes_b = _draw_block(3, str(layer_b), nb)

	def _place_label(layer_name: str, axes: List[plt.Axes], is_top: bool = False) -> None:
		bb = axes[0].get_position(fig)
		# Верхнюю надпись поднимаем сильнее, нижнюю — чуть-чуть.
		y = float(bb.y1) + (0.08 if is_top else 0.020)
		fig.text(
			0.5,
			y,
			f"Layer: {layer_name}",
			ha="center",
			va="bottom",
			fontsize=13,
			weight="bold",
		)

	_place_label(str(layer_a), axes_a, is_top=True)
	_place_label(str(layer_b), axes_b, is_top=False)


	# Avoid tight_layout squeezing rows too much; keep GridSpec spacing.
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)

	fig.savefig(out_path)
	plt.close(fig)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--run_dir", type=str, required=True, help="Run directory with meta.json and checkpoints/")
	ap.add_argument(
		"--checkpoint",
		type=str,
		default="best_main",
		help="Checkpoint selector: best_main, early_signal, or explicit .pt path.",
	)
	ap.add_argument("--split", type=str, choices=["val", "test"], default="val")
	ap.add_argument("--layers", type=str, default="", help="CSV embedding layers. Empty uses monitor layers from checkpoint/meta.")
	ap.add_argument(
		"--layers_from_correlations",
		type=str,
		default="",
		choices=["", "spectral", "topo_mtopdiv"],
		help="Auto-select layers from correlation report by metric family.",
	)
	ap.add_argument(
		"--correlation_csv",
		type=str,
		default="",
		help="Path to all_pairs.csv. Default: <run_dir>/correlations_report/all_pairs.csv",
	)
	ap.add_argument("--top_corr_layers", type=int, default=8, help="How many unique layers to take from correlations.")
	ap.add_argument("--corr_min_abs_rho", type=float, default=0.0)
	ap.add_argument("--corr_max_p", type=float, default=1.0)
	ap.add_argument("--top_k", type=int, default=20)
	ap.add_argument("--anchors_per_class", type=int, default=1, help="Number of random anchors per class for averaging.")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument(
		"--illustration_anchor_class",
		type=int,
		default=-1,
		help="If set >=0, force the class label used for the illustration anchor across all layers (CV: 0..C-1).",
	)
	ap.add_argument(
		"--illustration_anchor_index",
		type=int,
		default=-1,
		help="If set >=0, force the dataset index used for the illustration anchor across all layers.",
	)
	ap.add_argument("--batch_size", type=int, default=0, help="0 means use training batch size from meta.")
	ap.add_argument("--max_batches", type=int, default=0, help="If >0, limit embedding collection to this many batches.")
	ap.add_argument("--max_samples", type=int, default=0, help="If >0, limit embedding collection to this many samples.")
	ap.add_argument("--device", type=str, default="", help="Override device, for example mps, cuda:0 or cpu.")
	ap.add_argument(
		"--neighbor_pairs_top_k",
		type=int,
		default=0,
		help="If >0, save one 'anchor vs neighbor' PNG per top-k neighbor (per layer).",
	)
	ap.add_argument(
		"--compare_two_layers_top_k",
		type=int,
		default=0,
		help="If >0 and exactly 2 layers are requested, also save one PNG that shows top-k neighbors for both layers.",
	)
	ap.add_argument("--download", action="store_true", help="Allow dataset download if missing locally.")
	ap.add_argument(
		"--build_pretrained",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="If true, build model from pretrained weights/config before loading checkpoint state_dict. "
		"Default false to avoid Hub downloads (checkpoint contains weights already).",
	)
	ap.add_argument(
		"--all_main_layers",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Use all main model layers from model registry (ignores --layers).",
	)
	ap.add_argument(
		"--skip_existing",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Reuse existing layer report json/png when settings match.",
	)
	args = ap.parse_args()

	# When not explicitly downloading, prefer strict offline mode for HF stacks.
	# This prevents HEAD/resolve calls to the Hub on machines without network access.
	if not bool(args.download):
		os.environ.setdefault("HF_HUB_OFFLINE", "1")
		os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
		os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

	run_dir = os.path.abspath(str(args.run_dir))
	meta = _load_meta(run_dir)
	meta_args = dict(meta.get("args", {}) or {})
	ckpt_path = _resolve_checkpoint_path(run_dir, str(args.checkpoint))
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location="cpu")
	payload = dict(ckpt.get("payload", {}) or {})

	task = str(payload.get("task", meta.get("task", "cv"))).lower().strip()
	if task not in {"cv", "nlp"}:
		raise ValueError(f"Unknown task '{task}'.")
	nlp_objective = str(payload.get("nlp_objective", meta_args.get("nlp_objective", "classification"))).lower().strip()
	if task == "nlp" and nlp_objective != "classification":
		raise ValueError("Embedding evaluation currently supports only NLP classification runs (needs class labels).")
	dataset = str(payload.get("dataset", meta.get("dataset", meta_args.get("dataset", "")))).strip()
	model_name = str(payload.get("model", meta.get("model", meta_args.get("model", "")))).strip()
	if not dataset or not model_name:
		raise ValueError("Could not infer dataset/model from checkpoint payload or run meta.")

	raw_device = str(args.device).strip() if str(args.device).strip() else str(meta_args.get("device", "")).strip()
	device_name = _resolve_device(raw_device)
	device = torch.device(str(device_name))

	batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(meta_args.get("batch_size", 64))
	data_root = str(meta_args.get("data_root", "./data"))
	tokenizer_name = str(meta_args.get("tokenizer_name", "") or "").strip()
	if not tokenizer_name and task == "nlp":
		tokenizer_name = "distilbert-base-uncased"
	bundle = get_dataset(dataset, root=data_root, download=bool(args.download), tokenizer_name=tokenizer_name)
	loaders = make_dataloaders(bundle, batch_size=batch_size, num_workers=0)
	requested_split = str(args.split)
	loader = loaders.get(requested_split)
	# Fallback for datasets where test split is unlabeled (common for GLUE/SST-2).
	# We'll detect this later too (negative-label error or empty valid labels) and switch to val.
	if loader is None and requested_split == "test":
		loader = loaders.get("val")
	if loader is None:
		raise RuntimeError(f"No dataloader for split '{args.split}'.")
	effective_split = requested_split if loaders.get(requested_split) is not None else "val"

	num_classes = int(payload.get("num_classes", meta.get("num_classes", 0)) or 0)
	if num_classes <= 0:
		num_classes = _infer_num_classes(bundle.train) if bundle.train is not None else _infer_num_classes(bundle.val or bundle.test)
	preprocess = None
	if task == "cv":
		model, preprocess, model_layer_names = _build_cv_model(
			kind=model_name,
			num_classes=int(num_classes),
			device=device,
			pretrained=bool(args.build_pretrained) and bool(meta_args.get("pretrained", False)),
			input_flat_dim=None,
		)
	else:
		model, model_layer_names = _build_text_model(
			kind=model_name,
			num_classes=int(num_classes),
			device=device,
			pretrained=bool(args.build_pretrained) and bool(meta_args.get("pretrained", False)),
		)
	model.load_state_dict(ckpt["state_dict"], strict=True)
	model.eval()

	if bool(args.all_main_layers):
		req_layers = [str(x) for x in model_layer_names]
	else:
		req_layers = csv_to_list(str(args.layers))
		if not req_layers and str(args.layers_from_correlations).strip():
			corr_csv = str(args.correlation_csv).strip() or os.path.join(run_dir, "correlations_report", "all_pairs.csv")
			req_layers = _read_top_layers_from_correlations(
				csv_path=corr_csv,
				selection_group=str(args.layers_from_correlations).strip(),
				top_n_layers=int(args.top_corr_layers),
				min_abs_rho=float(args.corr_min_abs_rho),
				max_p=float(args.corr_max_p),
			)
		if not req_layers:
			req_layers = [str(x) for x in payload.get("monitor_layers", meta.get("monitor", {}).get("layer_names", []))]
	all_modules = set(list_module_names(model))
	layers = [x for x in req_layers if x in all_modules]
	if not layers:
		raise ValueError("No valid embedding layers were provided/found in the model.")
	if int(args.compare_two_layers_top_k) > 0 and len(layers) != 2:
		raise ValueError("--compare_two_layers_top_k requires exactly 2 layers (use --layers \"layerA,layerB\").")

	signal_layers = _read_signal_layers(run_dir)

	analysis_dir = os.path.join(run_dir, "analysis")
	os.makedirs(analysis_dir, exist_ok=True)
	tag = os.path.splitext(os.path.basename(ckpt_path))[0]
	layer_reports: List[Dict[str, Any]] = []
	anchor_class_for_illustration: Optional[int] = int(args.illustration_anchor_class) if int(args.illustration_anchor_class) >= 0 else None
	anchor_idx_for_illustration: Optional[int] = int(args.illustration_anchor_index) if int(args.illustration_anchor_index) >= 0 else None
	valid_layers: List[str] = []
	reused_count = 0

	# First, reuse existing per-layer reports when possible.
	layers_to_compute: List[str] = []
	layer_paths: Dict[str, Tuple[str, str]] = {}
	for layer in layers:
		layer_safe = _safe_name(layer)
		layer_json = os.path.join(analysis_dir, f"embedding_retrieval_{tag}__layer_{layer_safe}.json")
		layer_png = os.path.join(analysis_dir, f"embedding_retrieval_{tag}__layer_{layer_safe}.png")
		layer_paths[str(layer)] = (layer_json, layer_png)
		try:
			if bool(args.skip_existing):
				reuse_ok, reuse_obj = _can_reuse_layer_report(
					layer_json_path=layer_json,
					run_dir=run_dir,
					ckpt_path=ckpt_path,
					split=str(args.split),
					top_k=int(args.top_k),
					anchors_per_class=int(args.anchors_per_class),
					seed=int(args.seed),
					max_batches=int(args.max_batches),
					max_samples=int(args.max_samples),
					require_illustration=True,
				)
				if reuse_ok and isinstance(reuse_obj, dict):
					reuse_obj["layer_report_path"] = layer_json
					reuse_obj["reused_existing"] = True
					layer_reports.append(reuse_obj)
					valid_layers.append(str(layer))
					reused_count += 1
					continue
		except Exception:
			pass
		layers_to_compute.append(str(layer))

	# Compute the remaining layers in a single pass over the loader.
	emb_by_layer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
	if layers_to_compute:
		def _try_collect(_loader) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
			return _collect_embeddings_and_labels_many(
				model=model,
				loader=_loader,
				layer_names=layers_to_compute,
				preprocess=preprocess,
				device=device,
				max_batches=int(args.max_batches),
				max_samples=int(args.max_samples),
			)

		try:
			emb_by_layer = _try_collect(loader)
		except Exception as e:
			msg = str(e)
			# If NLP test split has unknown labels (-1) and collate is strict, fall back to val.
			if task == "nlp" and requested_split == "test":
				val_loader = loaders.get("val")
				if val_loader is not None and (
					("Found negative labels" in msg)
					or ("negative labels" in msg.lower())
					or ("no valid labeled" in msg.lower())
					or ("Could not collect embeddings" in msg)
				):
					print("[EmbeddingEval] test split appears unlabeled (e.g. label=-1); using val split instead.")
					loader = val_loader
					effective_split = "val"
					try:
						emb_by_layer = _try_collect(loader)
					except Exception as e2:
						for layer in layers_to_compute:
							layer_reports.append({"layer": str(layer), "error": str(e2)})
						emb_by_layer = {}
				else:
					for layer in layers_to_compute:
						layer_reports.append({"layer": str(layer), "error": msg})
					emb_by_layer = {}
			else:
				for layer in layers_to_compute:
					layer_reports.append({"layer": str(layer), "error": msg})
				emb_by_layer = {}

	# Choose anchors once (using any successfully collected layer).
	labels_np = None
	if emb_by_layer:
		_any_layer = next(iter(emb_by_layer.keys()))
		labels_np = emb_by_layer[_any_layer][1].numpy()
		anchors_by_class = _choose_per_class_anchors(
			labels_np,
			seed=int(args.seed),
			anchors_per_class=int(args.anchors_per_class),
		)
		# Only auto-pick when neither is forced by args.
		if anchors_by_class and (anchor_class_for_illustration is None and anchor_idx_for_illustration is None):
			rng = np.random.default_rng(int(args.seed))
			class_choices = sorted(list(anchors_by_class.keys()))
			anchor_class_for_illustration = int(rng.choice(class_choices))
			anchor_idx_for_illustration = int(rng.choice(anchors_by_class[anchor_class_for_illustration]))
	else:
		anchors_by_class = {}

	# If the user forces anchor index, infer/validate its class once (before per-layer loop),
	# so we don't silently write per-layer error rows.
	if anchor_idx_for_illustration is not None:
		if labels_np is None:
			raise RuntimeError("Could not infer labels to validate --illustration_anchor_index.")
		n_lab = int(labels_np.shape[0])
		if not (0 <= int(anchor_idx_for_illustration) < n_lab):
			raise ValueError(f"--illustration_anchor_index={anchor_idx_for_illustration} out of range (n={n_lab}).")
		inferred_cls = int(labels_np[int(anchor_idx_for_illustration)])
		if anchor_class_for_illustration is None:
			anchor_class_for_illustration = inferred_cls
		elif int(anchor_class_for_illustration) != int(inferred_cls):
			raise ValueError(
				f"Illustration anchor label mismatch: index={anchor_idx_for_illustration} has label={inferred_cls}, "
				f"but --illustration_anchor_class={anchor_class_for_illustration}. "
				"Did you swap class and index?"
			)
		if task == "cv" and int(num_classes) > 0 and int(anchor_class_for_illustration) >= int(num_classes):
			raise ValueError(
				f"--illustration_anchor_class={anchor_class_for_illustration} is out of range for num_classes={num_classes}. "
				"Did you swap class and index?"
			)

	for layer in layers_to_compute:
		try:
			if layer not in emb_by_layer:
				layer_reports.append({"layer": str(layer), "error": "no_embeddings_collected"})
				continue
			emb, labels = emb_by_layer[layer]
			if not anchors_by_class:
				layer_reports.append({"layer": str(layer), "error": "not_enough_class_samples"})
				continue

			per_class: List[Dict[str, Any]] = []
			for cls in sorted(anchors_by_class.keys()):
				anchor_ids = [int(x) for x in anchors_by_class[cls]]
				class_scores = [
					_score_anchor(emb=emb, labels=labels, anchor_idx=int(aid), anchor_class=int(cls), top_k=int(args.top_k))
					for aid in anchor_ids
				]
				ratios = [float(sc["same_class_ratio"]) for sc in class_scores]
				counts = [int(sc["same_class_count"]) for sc in class_scores]
				per_class.append(
					{
						"class": int(cls),
						"anchor_indices": anchor_ids,
						"n_anchors": int(len(anchor_ids)),
						"same_class_count_mean": float(np.mean(counts)) if counts else 0.0,
						"same_class_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
						"same_class_ratio_std": float(np.std(ratios)) if ratios else 0.0,
						"top_k": int(class_scores[0]["top_k"]) if class_scores else int(args.top_k),
					}
				)
			class_means = [float(x["same_class_ratio_mean"]) for x in per_class]
			macro_ratio = float(np.mean(class_means)) if class_means else 0.0
			macro_ratio_std = float(np.std(class_means)) if class_means else 0.0

			layer_json, layer_png = layer_paths[layer]
			if anchor_idx_for_illustration is None or anchor_class_for_illustration is None:
				raise RuntimeError("Internal error: illustration anchor was not selected.")
			if not (0 <= int(anchor_idx_for_illustration) < int(labels.shape[0])):
				raise ValueError(
					f"Illustration anchor index out of range: {anchor_idx_for_illustration} (n={int(labels.shape[0])})"
				)
			lbl = int(labels[int(anchor_idx_for_illustration)].item())
			if lbl != int(anchor_class_for_illustration):
				raise ValueError(
					f"Illustration anchor label mismatch: idx={anchor_idx_for_illustration} has label={lbl}, "
					f"but --illustration_anchor_class={anchor_class_for_illustration}."
				)
			illustr = _score_anchor(
				emb=emb,
				labels=labels,
				anchor_idx=int(anchor_idx_for_illustration),
				anchor_class=int(anchor_class_for_illustration),
				top_k=max(20, int(args.top_k)),
			)
			illustr_path = None
			if task == "cv":
				_make_layer_illustration(
					out_path=layer_png,
					dataset=loader.dataset,
					labels=labels,
					anchor_idx=int(anchor_idx_for_illustration),
					anchor_class=int(anchor_class_for_illustration),
					neighbor_indices=list(illustr["neighbor_indices"]),
				)
				if int(args.neighbor_pairs_top_k) > 0:
					_safe_layer = _safe_name(str(layer))
					prefix = f"embedding_retrieval_{tag}__layer_{_safe_layer}"
					_make_neighbor_pair_illustrations(
						out_dir=analysis_dir,
						file_prefix=prefix,
						dataset=loader.dataset,
						labels=labels,
						anchor_idx=int(anchor_idx_for_illustration),
						anchor_class=int(anchor_class_for_illustration),
						neighbor_indices=list(illustr["neighbor_indices"]),
						top_k_pairs=int(args.neighbor_pairs_top_k),
					)
				illustr_path = layer_png
			elif task == "nlp":
				_make_layer_illustration_text(
					out_path=layer_png,
					dataset=loader.dataset,
					labels=labels,
					anchor_idx=int(anchor_idx_for_illustration),
					anchor_class=int(anchor_class_for_illustration),
					neighbor_indices=list(illustr["neighbor_indices"]),
				)
				illustr_path = layer_png

			row = {
				"layer": str(layer),
				"is_signal_layer": bool(str(layer) in set(signal_layers)),
				"n_samples": int(labels.numel()),
				"macro_avg_same_class_ratio": float(macro_ratio),
				"macro_avg_same_class_ratio_std": float(macro_ratio_std),
				"anchors_per_class": int(args.anchors_per_class),
				"per_class": per_class,
				"illustration_anchor": {
					"class": int(anchor_class_for_illustration),
					"index": int(anchor_idx_for_illustration),
					"same_class_count": int(illustr["same_class_count"]),
					"same_class_ratio": float(illustr["same_class_ratio"]),
					"top_k": int(illustr["top_k"]),
					"neighbor_indices": [int(x) for x in illustr["neighbor_indices"]],
				},
				"illustration_path": illustr_path,
				"illustration_version": int(_ILLUSTRATION_VERSION) if illustr_path else None,
				"split": str(requested_split),
				"effective_split": str(effective_split),
				"top_k": int(args.top_k),
				"seed": int(args.seed),
				"max_batches": int(args.max_batches),
				"max_samples": int(args.max_samples),
				"checkpoint": ckpt_path,
				"run_dir": run_dir,
				"reused_existing": False,
			}
			with open(layer_json, "w", encoding="utf-8") as f:
				json.dump(row, f, ensure_ascii=False, indent=2)
			row["layer_report_path"] = layer_json
			layer_reports.append(row)
			valid_layers.append(str(layer))
		except Exception as e:
			layer_reports.append({"layer": str(layer), "error": str(e)})

	summary = {
		"run_dir": run_dir,
		"checkpoint": ckpt_path,
		"split": str(requested_split),
		"effective_split": str(effective_split),
		"top_k": int(args.top_k),
		"anchors_per_class": int(args.anchors_per_class),
		"seed": int(args.seed),
		"max_batches": int(args.max_batches),
		"max_samples": int(args.max_samples),
		"layers": valid_layers,
		"signal_layers": signal_layers,
		"illustration_anchor_class": anchor_class_for_illustration,
		"illustration_anchor_index": anchor_idx_for_illustration,
		"rows": layer_reports,
		"reused_layers": int(reused_count),
	}
	summary_ok = [r for r in layer_reports if "macro_avg_same_class_ratio" in r]
	summary_ok.sort(key=lambda r: float(r["macro_avg_same_class_ratio"]), reverse=True)
	summary["top_by_macro_ratio"] = summary_ok[: min(10, len(summary_ok))]
	out_path = os.path.join(analysis_dir, f"embedding_retrieval_{tag}__summary.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	# Optional: single PNG comparing top-k neighbors for two layers.
	if task == "cv" and int(args.compare_two_layers_top_k) > 0:
		# Prefer a "good" row that has illustration info; don't accidentally pick an error row.
		by_layer: Dict[str, Dict[str, Any]] = {}
		last_err: Dict[str, str] = {}
		for rr in layer_reports:
			if not isinstance(rr, dict):
				continue
			ln = str(rr.get("layer", "") or "")
			if not ln:
				continue
			if "error" in rr and isinstance(rr.get("error", None), str):
				last_err[ln] = str(rr.get("error", "") or "")
				continue
			if isinstance(rr.get("illustration_anchor", None), dict):
				by_layer[ln] = rr
		la = str(layers[0])
		lb = str(layers[1])
		ra = by_layer.get(la, None)
		rb = by_layer.get(lb, None)
		if not (isinstance(ra, dict) and isinstance(rb, dict)):
			ea = last_err.get(la, "")
			eb = last_err.get(lb, "")
			raise RuntimeError(
				"Could not find both layer reports for comparison rendering. "
				f"layer_a={la!r} error={ea!r}; layer_b={lb!r} error={eb!r}. "
				"Tip: ensure --illustration_anchor_index/--illustration_anchor_class are correct and rerun with --no-skip_existing."
			)
		ia = ra.get("illustration_anchor", None)
		ib = rb.get("illustration_anchor", None)
		if not (isinstance(ia, dict) and isinstance(ib, dict)):
			raise RuntimeError(
				"Missing illustration_anchor in one of the layer reports. "
				"Tip: rerun with --no-skip_existing and a valid illustration anchor."
			)
		na = ia.get("neighbor_indices", None)
		nb = ib.get("neighbor_indices", None)
		if not (isinstance(na, list) and isinstance(nb, list)):
			raise RuntimeError("Missing neighbor_indices in one of the layer reports.")
		out_cmp = os.path.join(
			analysis_dir,
			f"embedding_retrieval_{tag}__compare_{_safe_name(la)}_vs_{_safe_name(lb)}.png",
		)
		if anchor_idx_for_illustration is None or anchor_class_for_illustration is None:
			raise RuntimeError("Internal error: illustration anchor not set.")
		_make_two_layer_topk_comparison(
			out_path=out_cmp,
			dataset=loader.dataset,
			labels=labels,
			anchor_idx=int(anchor_idx_for_illustration),
			anchor_class=int(anchor_class_for_illustration),
			layer_a=la,
			neighbors_a=[int(x) for x in na],
			layer_b=lb,
			neighbors_b=[int(x) for x in nb],
			top_k=int(args.compare_two_layers_top_k),
		)
		print("[EmbeddingEval] saved comparison:", out_cmp)

	print(f"[EmbeddingEval] saved summary: {out_path}")
	for r in summary_ok[:5]:
		print(
			f"  layer={r.get('layer')} macro_ratio={r.get('macro_avg_same_class_ratio'):.4f} "
			f"signal_layer={r.get('is_signal_layer')} file={r.get('layer_report_path')}"
		)


if __name__ == "__main__":
	main()
