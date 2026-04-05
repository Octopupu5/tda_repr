from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.figures.i18n import I18N
from tda_repr.data import get_dataset
from tda_repr.models import LayerTaps, list_module_names
from tools.run_experiment import _build_text_model, _infer_num_classes, _repr_from_activation


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	return float(v) if math.isfinite(float(v)) else None


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
	if isinstance(x, torch.Tensor):
		return x
	if isinstance(x, (list, tuple)):
		for it in x:
			t = _first_tensor(it)
			if t is not None:
				return t
		return None
	if isinstance(x, dict) or (hasattr(x, "keys") and hasattr(x, "__getitem__")):
		try:
			keys = list(x.keys())
		except Exception:
			keys = []
		for key in ("last_hidden_state", "hidden_states", "logits"):
			if key in keys:
				try:
					v = x[key]
				except Exception:
					continue
				t = _first_tensor(v)
				if t is not None:
					return t
		try:
			vals = list(x.values())
		except Exception:
			vals = []
		for v in vals:
			t = _first_tensor(v)
			if t is not None:
				return t
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
	if x.dim() == 3:
		x = x[:, 0, :]
	elif x.dim() == 4:
		x = x.mean(dim=(2, 3))
	elif x.dim() == 2:
		pass
	else:
		x = x.view(x.shape[0], -1)
	return x


def _pick_text(example: Dict[str, Any]) -> str:
	for k in ("text", "question", "sentence", "content", "query"):
		v = example.get(k, None)
		if isinstance(v, str) and v.strip():
			return v.strip()
	for _k, v in example.items():
		if isinstance(v, str) and v.strip():
			return v.strip()
	return str(example)


def _pick_label(example: Dict[str, Any]) -> int:
	for k in ("label", "labels", "coarse_label", "target", "class"):
		v = example.get(k, None)
		if isinstance(v, (int, float)) and math.isfinite(float(v)):
			return int(v)
	for _k, v in example.items():
		if isinstance(v, (int, float)) and math.isfinite(float(v)):
			return int(v)
	raise ValueError(f"Could not infer label from keys={list(example.keys())}")


@dataclass(frozen=True)
class Neighbor:
	idx: int
	y: int
	text: str
	is_same: bool


def _cosine_topk(emb: np.ndarray, anchor_i: int, top_k: int) -> List[int]:
	x = emb.astype(np.float64, copy=False)
	n = int(x.shape[0])
	if not (0 <= int(anchor_i) < n):
		raise ValueError(f"anchor_idx out of range: {anchor_i} (n={n})")
	norm = np.linalg.norm(x, axis=1, keepdims=True)
	norm = np.clip(norm, 1e-12, None)
	xn = x / norm
	sim = xn @ xn[int(anchor_i)].reshape(-1, 1)
	sim = sim.reshape(-1)
	sim[int(anchor_i)] = -1e9
	k = int(max(1, min(int(top_k), n - 1)))
	idx = np.argpartition(-sim, kth=k - 1)[:k]
	idx = idx[np.argsort(-sim[idx])]
	return [int(i) for i in idx.tolist()]


def main() -> None:
	ap = argparse.ArgumentParser(description="Render TREC-6 nearest-neighbor text example for a DistilBERT run.")
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--checkpoint", type=str, default="checkpoints/model_best_main.pt")
	ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
	ap.add_argument("--layer", type=str, default="distilbert.transformer.layer.5.ffn")
	ap.add_argument("--anchor_idx", type=int, default=364)
	ap.add_argument("--top_k", type=int, default=20)
	ap.add_argument("--batch_size", type=int, default=64)
	ap.add_argument("--data_root", type=str, default="./data")
	ap.add_argument("--download", action="store_true")
	ap.add_argument("--device", type=str, default="cpu")
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--lang", type=str, default="en", choices=["en", "ru"])
	args = ap.parse_args()

	i18n = I18N(lang=str(args.lang))
	run_dir = os.path.abspath(str(args.run_dir))
	meta_path = os.path.join(run_dir, "meta.json")
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"Missing meta.json: {meta_path}")
	meta = json.load(open(meta_path, "r", encoding="utf-8"))
	meta_args = dict(meta.get("args", {}) or {})

	ckpt_path = str(args.checkpoint).strip()
	if not os.path.isabs(ckpt_path):
		ckpt_path = os.path.join(run_dir, ckpt_path)
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

	payload = {}
	ckpt = torch.load(ckpt_path, map_location="cpu")
	if isinstance(ckpt, dict):
		payload = dict(ckpt.get("payload", {}) or {})

	dataset = str(payload.get("dataset", meta.get("dataset", meta_args.get("dataset", ""))) or "").strip()
	model_kind = str(payload.get("model", meta.get("model", meta_args.get("model", ""))) or "").strip()
	if not dataset or not model_kind:
		raise ValueError("Could not infer dataset/model from meta.json or checkpoint payload.")

	if str(dataset).lower().strip() not in {"trec6", "trec-6", "trec"}:
		raise ValueError(f"This helper expects TREC-6. Got dataset={dataset!r}.")

	bundle = get_dataset(
		"trec6",
		root=str(args.data_root),
		download=bool(args.download),
		tokenizer_name=str(meta_args.get("tokenizer_name", "distilbert-base-uncased")),
	)
	ds = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[str(args.split)]
	if ds is None:
		raise RuntimeError(f"No split '{args.split}' available for dataset '{bundle.name}'.")

	num_classes = int(payload.get("num_classes", meta.get("num_classes", 0)) or 0)
	if num_classes <= 0:
		num_classes = _infer_num_classes(bundle.train) if bundle.train is not None else _infer_num_classes(ds)

	device = torch.device(str(args.device))
	model, _layer_names = _build_text_model(kind=str(model_kind), num_classes=int(num_classes), device=device, pretrained=False)

	state_dict = ckpt.get("state_dict", None) if isinstance(ckpt, dict) else None
	if not isinstance(state_dict, dict):
		raise ValueError("Checkpoint does not contain a 'state_dict' dict (expected tools/run_experiment.py format).")
	model.load_state_dict(state_dict, strict=True)
	model.eval()

	all_mods = set(list_module_names(model))
	layer = str(args.layer).strip()
	if layer not in all_mods:
		raise ValueError(f"Layer not found in model modules: {layer}")

	class _Indexed(torch.utils.data.Dataset):
		def __init__(self, base: Any):
			self.base = base

		def __len__(self) -> int:
			return int(len(self.base))

		def __getitem__(self, idx: int) -> Dict[str, Any]:
			ex = dict(self.base[int(idx)])
			ex["__idx"] = int(idx)
			return ex

	def _collate_with_idx(batch: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, Any]]:
		idxs = [int(x["__idx"]) for x in batch]
		clean = []
		for x in batch:
			y = dict(x)
			y.pop("__idx", None)
			clean.append(y)
		if bundle.collate_fn is None:
			raise RuntimeError("Missing collate_fn for NLP dataset bundle.")
		tok = bundle.collate_fn(clean)
		return idxs, tok

	loader = torch.utils.data.DataLoader(_Indexed(ds), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate_with_idx)

	emb_by_idx: Dict[int, np.ndarray] = {}
	label_by_idx: Dict[int, int] = {}
	text_by_idx: Dict[int, str] = {}

	with torch.no_grad(), LayerTaps(model, [layer]) as taps:
		for idxs, batch in loader:
			batch_map: Dict[str, Any] = {}
			if isinstance(batch, Mapping):
				batch_map = dict(batch)
			elif isinstance(batch, (list, tuple)) and len(batch) == 2:
				a, b = batch
				if isinstance(a, Mapping):
					batch_map = dict(a)
					if isinstance(b, torch.Tensor):
						batch_map.setdefault("labels", b)
				elif isinstance(b, Mapping):
					batch_map = dict(b)
					if isinstance(a, torch.Tensor):
						batch_map.setdefault("labels", a)
			if not batch_map:
				raise RuntimeError(f"Unexpected collate output for NLP batch: type={type(batch)!r}")

			inputs: Dict[str, Any] = {}
			for k in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
				v = batch_map.get(k, None)
				if isinstance(v, torch.Tensor):
					inputs[k] = v.to(device)
			if not inputs:
				raise RuntimeError(f"Batch does not contain model inputs (input_ids/attention_mask...). keys={sorted(batch_map.keys())}")

			_ = model(**inputs)
			act = taps.outputs.get(layer, None)
			z = _repr_from_hook_output(act)
			if z is None:
				raise RuntimeError(f"Could not extract representation tensor for layer '{layer}'.")
			z = z.detach().to("cpu")

			if int(z.shape[0]) != int(len(idxs)):
				raise RuntimeError(f"Batch size mismatch: z={tuple(z.shape)} idxs={len(idxs)}")

			for i, idx in enumerate(idxs):
				ex = dict(ds[int(idx)])
				text_by_idx[int(idx)] = _pick_text(ex)
				label_by_idx[int(idx)] = _pick_label(ex)
				emb_by_idx[int(idx)] = z[i].float().numpy()

	n = int(len(ds))
	emb = np.stack([emb_by_idx[i] for i in range(n)], axis=0)
	ys = [int(label_by_idx[i]) for i in range(n)]

	anchor_i = int(args.anchor_idx)
	anchor_y = int(ys[anchor_i])
	knn = _cosine_topk(emb, anchor_i=anchor_i, top_k=int(args.top_k))
	neigh = [
		Neighbor(idx=j, y=int(ys[j]), text=str(text_by_idx[j]), is_same=(int(ys[j]) == int(anchor_y))) for j in knn
	]

	title = (
		"Anchor (yellow) and nearest neighbors (green/red)"
		if i18n.lang == "en"
		else "Якорь (желтый) и ближайшие соседи (зеленый/красный)"
	)
	anchor_line = f"id={anchor_i} y={anchor_y} - {text_by_idx[anchor_i]}"

	fig = plt.figure(figsize=(14.0, 7.6), dpi=150)
	ax = fig.add_subplot(1, 1, 1)
	ax.set_axis_off()
	ax.set_title(title, fontsize=14, pad=14)

	def _box(text: str, xy: Tuple[float, float], face: str, edge: str) -> None:
		ax.text(
			xy[0],
			xy[1],
			text,
			transform=ax.transAxes,
			ha="left",
			va="top",
			fontsize=9.5,
			color=edge,
			bbox={"boxstyle": "round,pad=0.25", "facecolor": face, "edgecolor": edge, "linewidth": 1.2, "alpha": 0.22},
		)

	_box(anchor_line, (0.12, 0.92), face="#ffd54f", edge="#a26a00")

	left = neigh[: (len(neigh) + 1) // 2]
	right = neigh[(len(neigh) + 1) // 2 :]
	y0 = 0.78
	dy = 0.085

	for col_i, col in enumerate((left, right)):
		x0 = 0.06 if col_i == 0 else 0.52
		for r_i, nbh in enumerate(col):
			y = y0 - float(r_i) * dy
			edge = "#1b7f2a" if nbh.is_same else "#b71c1c"
			face = "#b7f7c2" if nbh.is_same else "#ffcdd2"
			line = f"id={nbh.idx} y={nbh.y} - {nbh.text}"
			_box(line, (x0, y), face=face, edge=edge)

	out_png = os.path.abspath(str(args.out_png))
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	fig.savefig(out_png, bbox_inches="tight")
	plt.close(fig)
	print("[OK] wrote", out_png)


if __name__ == "__main__":
	main()

