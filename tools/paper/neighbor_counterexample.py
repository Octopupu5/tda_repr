from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running as a script: ensure project root (parent of /tools) is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from tda_repr.data import get_dataset, make_dataloaders
from tda_repr.models import LayerTaps, list_module_names
from tools.run_experiment import _build_cv_model, _build_text_model, _infer_num_classes, _repr_from_activation


def _load_meta(run_dir: str) -> Dict[str, Any]:
	p = os.path.join(run_dir, "meta.json")
	if not os.path.exists(p):
		raise FileNotFoundError(f"Missing meta.json: {p}")
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


def _auto_device_string() -> str:
	try:
		if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return "mps"
	except Exception:
		pass
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_device(preferred: str) -> str:
	"""
	Resolve device string safely on heterogeneous machines.
	- If user explicitly passes --device, we validate and either use it or error.
	- If device comes from meta.json and is invalid (e.g., 'cuda:0' on macOS), we fall back to mps/cpu.
	"""
	dev = str(preferred).strip()
	if not dev:
		return _auto_device_string()

	low = dev.lower()
	if low.startswith("cuda"):
		# Meta can store cuda:0 even when the current machine doesn't have CUDA.
		if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)) and torch.cuda.is_available():
			return dev
		# Prefer MPS on macOS, otherwise CPU.
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

	# Other device strings: trust torch to validate.
	return dev


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
	if isinstance(x, torch.Tensor):
		return x
	if isinstance(x, (list, tuple)):
		for it in x:
			t = _first_tensor(it)
			if t is not None:
				return t
		return None
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
	if hasattr(x, "to_tuple") and callable(getattr(x, "to_tuple")):
		try:
			return _first_tensor(x.to_tuple())
		except Exception:
			return None
	return None


def _repr_from_hook_output(act: Any) -> Optional[torch.Tensor]:
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


def _collect_embeddings_and_labels_many(
	model: torch.nn.Module,
	loader: Any,
	layer_names: List[str],
	preprocess: Optional[Any],
	device: torch.device,
	max_batches: int,
	max_samples: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
	embeds_by_layer: Dict[str, List[torch.Tensor]] = {str(k): [] for k in layer_names}
	labels_by_layer: Dict[str, List[torch.Tensor]] = {str(k): [] for k in layer_names}
	model.eval()
	with torch.no_grad(), LayerTaps(model, layer_names) as taps:
		seen_batches = 0
		seen_samples = 0
		for batch in loader:
			if int(max_batches) > 0 and int(seen_batches) >= int(max_batches):
				break

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
					raise RuntimeError("Batch does not contain labels (expected 'labels' or 'label').")
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
					raise RuntimeError("Batch does not contain model inputs (input_ids/attention_mask...).")
				_ = model(**inputs)
			else:
				raise RuntimeError(f"Unsupported batch type: {type(batch)}")

			if y is None:
				raise RuntimeError("Internal error: labels are missing.")
			mask = y >= 0
			if bool(mask.any().item()) is False:
				continue

			for layer in layer_names:
				act = taps.outputs.get(layer, None)
				z = _repr_from_hook_output(act)
				if z is None:
					raise RuntimeError(f"Could not extract representation tensor for layer '{layer}'.")
				z = z.detach().to("cpu")
				if int(z.shape[0]) != int(y.numel()):
					raise RuntimeError(f"Batch size mismatch for layer '{layer}': z={tuple(z.shape)} y={tuple(y.shape)}")
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
			raise RuntimeError(f"No embeddings collected for layer '{layer}'.")
		out[str(layer)] = (torch.cat(ee, dim=0), torch.cat(ll, dim=0))
	return out


def _score_anchor(
	emb: torch.Tensor,
	labels: torch.Tensor,
	anchor_idx: int,
	top_k: int,
) -> Dict[str, Any]:
	e = emb.float()
	n = int(e.shape[0])
	if not (0 <= int(anchor_idx) < n):
		raise ValueError(f"anchor_idx out of range: {anchor_idx} (n={n})")
	k = max(1, min(int(top_k), n - 1))
	d = torch.cdist(e[anchor_idx : anchor_idx + 1], e).view(-1)
	d[int(anchor_idx)] = float("inf")
	nn_idx = torch.topk(d, k=k, largest=False).indices
	nn_labels = labels[nn_idx]
	anchor_class = int(labels[int(anchor_idx)].item())
	hits = int((nn_labels == int(anchor_class)).sum().item())
	return {
		"top_k": int(k),
		"anchor_class": int(anchor_class),
		"same_class_count": int(hits),
		"same_class_ratio": (float(hits) / float(k)),
		"neighbor_indices": [int(i) for i in nn_idx.detach().cpu().tolist()],
		"neighbor_labels": [int(i) for i in nn_labels.detach().cpu().tolist()],
	}


def main() -> None:
	ap = argparse.ArgumentParser(
		description="Find an EfficientNet/ImageNette anchor where proposed layer improves top-k nearest-neighbor purity vs default layer."
	)
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--checkpoint", type=str, default="best_main")
	ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
	ap.add_argument("--baseline_layer", type=str, default="classifier")
	ap.add_argument("--proposed_layer", type=str, default="features.8.1")
	ap.add_argument("--top_k", type=int, default=10)
	ap.add_argument("--anchors_per_class", type=int, default=100)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--max_batches", type=int, default=0)
	ap.add_argument("--max_samples", type=int, default=0)
	ap.add_argument("--device", type=str, default="")
	ap.add_argument("--out_json", type=str, default="", help="If set, write the best found counterexample as JSON.")
	args = ap.parse_args()

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
	if task == "nlp":
		raise ValueError("This helper is intended for CV runs (ImageNette).")

	dataset = str(payload.get("dataset", meta.get("dataset", meta_args.get("dataset", "")))).strip()
	model_name = str(payload.get("model", meta.get("model", meta_args.get("model", "")))).strip()
	if not dataset or not model_name:
		raise ValueError("Could not infer dataset/model from checkpoint payload or run meta.")

	# If user passed --device, honor it strictly; otherwise tolerate invalid meta device (e.g., cuda on macOS).
	raw_device = str(args.device).strip() if str(args.device).strip() else str(meta_args.get("device", "")).strip()
	device_name = _resolve_device(raw_device)
	device = torch.device(str(device_name))

	batch_size = int(meta_args.get("batch_size", 64))
	data_root = str(meta_args.get("data_root", "./data"))
	bundle = get_dataset(dataset, root=data_root, download=False, tokenizer_name=str(meta_args.get("tokenizer_name", "") or "").strip())
	loaders = make_dataloaders(bundle, batch_size=int(batch_size), num_workers=0)
	loader = loaders.get(str(args.split))
	if loader is None:
		raise RuntimeError(f"No dataloader for split '{args.split}'.")

	num_classes = int(payload.get("num_classes", meta.get("num_classes", 0)) or 0)
	if num_classes <= 0:
		num_classes = _infer_num_classes(bundle.train) if bundle.train is not None else _infer_num_classes(bundle.val or bundle.test)

	model, preprocess, _layer_names = _build_cv_model(
		kind=model_name,
		num_classes=int(num_classes),
		device=device,
		pretrained=False,
	)
	if "state_dict" not in ckpt or not isinstance(ckpt.get("state_dict", None), dict):
		raise ValueError(
			"Checkpoint does not contain a 'state_dict' dict. "
			"Expected the same format as tools/run_experiment.py (_save_model_checkpoint)."
		)
	model.load_state_dict(ckpt["state_dict"], strict=True)
	model.eval()

	all_modules = set(list_module_names(model))
	base = str(args.baseline_layer).strip()
	prop = str(args.proposed_layer).strip()
	if base not in all_modules:
		raise ValueError(f"Baseline layer not found in model: {base}")
	if prop not in all_modules:
		raise ValueError(f"Proposed layer not found in model: {prop}")

	embs = _collect_embeddings_and_labels_many(
		model=model,
		loader=loader,
		layer_names=[base, prop],
		preprocess=preprocess,
		device=device,
		max_batches=int(args.max_batches),
		max_samples=int(args.max_samples),
	)
	emb_base, labels = embs[base]
	emb_prop, labels2 = embs[prop]
	if int(labels.shape[0]) != int(labels2.shape[0]):
		raise RuntimeError("Label tensor mismatch across layers.")

	labels_np = labels.numpy()
	classes, counts = np.unique(labels_np, return_counts=True)
	valid_classes = [int(c) for c, cnt in zip(classes.tolist(), counts.tolist()) if int(cnt) >= 2]
	if not valid_classes:
		raise RuntimeError("No valid classes found (need >=2 samples per class).")

	rng = np.random.default_rng(int(args.seed))
	best: Optional[Dict[str, Any]] = None
	best_gain = None
	k = int(args.top_k)
	for cls in valid_classes:
		ids = np.where(labels_np == int(cls))[0]
		if ids.size == 0:
			continue
		m = min(int(args.anchors_per_class), int(ids.size))
		chosen = rng.choice(ids, size=m, replace=False)
		for anchor_idx in chosen.tolist():
			sb = _score_anchor(emb=emb_base, labels=labels, anchor_idx=int(anchor_idx), top_k=k)
			sp = _score_anchor(emb=emb_prop, labels=labels, anchor_idx=int(anchor_idx), top_k=k)
			if int(sb["anchor_class"]) != int(sp["anchor_class"]):
				raise RuntimeError("Anchor class mismatch across layers.")
			gain = int(sp["same_class_count"]) - int(sb["same_class_count"])
			if gain <= 0:
				continue
			if best_gain is None or gain > int(best_gain):
				best_gain = int(gain)
				best = {
					"run_dir": run_dir,
					"checkpoint": ckpt_path,
					"split": str(args.split),
					"top_k": k,
					"anchor_index": int(anchor_idx),
					"anchor_class": int(sb["anchor_class"]),
					"baseline_layer": base,
					"proposed_layer": prop,
					"baseline": sb,
					"proposed": sp,
					"gain_same_class_count": int(gain),
				}

	if best is None:
		raise SystemExit(
			"No counterexample found with proposed > baseline under current sampling. "
			"Try increasing --anchors_per_class or changing --seed."
		)

	print("[OK] Found counterexample")
	print(" anchor_index:", best["anchor_index"])
	print(" anchor_class:", best["anchor_class"])
	print(" baseline_layer:", best["baseline_layer"])
	print(" proposed_layer:", best["proposed_layer"])
	print(" top_k:", best["top_k"])
	print(" baseline hits:", best["baseline"]["same_class_count"], "neighbors(labels):", best["baseline"]["neighbor_labels"][:k])
	print(" proposed hits:", best["proposed"]["same_class_count"], "neighbors(labels):", best["proposed"]["neighbor_labels"][:k])
	print(" gain:", best["gain_same_class_count"])

	if str(args.out_json).strip():
		out_p = os.path.abspath(str(args.out_json))
		os.makedirs(os.path.dirname(out_p), exist_ok=True)
		with open(out_p, "w", encoding="utf-8") as f:
			json.dump(best, f, ensure_ascii=False, indent=2)
		print("[OK] wrote:", out_p)


if __name__ == "__main__":
	main()

