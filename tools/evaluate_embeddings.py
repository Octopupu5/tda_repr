from __future__ import annotations

import argparse
import json
import os
import textwrap
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tda_repr.data import get_dataset, make_dataloaders
from tda_repr.models import LayerTaps, csv_to_list, list_module_names
from tools._shared import (
	build_cv_model,
	build_text_model,
	ensure_dir,
	infer_num_classes,
	move_to_device,
	repr_from_activation,
	resolve_tokenizer_name,
)
from tools.aggregate.embedding_selection import build_selection_rows


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Evaluate embedding retrieval quality for selected layers.")
	ap.add_argument("--run_dir", required=True)
	ap.add_argument("--checkpoint", default="best_main", help="best_main|last|early_signal|<path to .pt>")
	ap.add_argument("--split", default="val", choices=["train", "val", "test"])
	ap.add_argument("--layers", default="", help="CSV layer module names. Empty = layers from run meta.")
	ap.add_argument("--all_main_layers", action="store_true", help="Use model registry default layers for this architecture.")
	ap.add_argument("--top_k", type=int, default=20)
	ap.add_argument("--anchors_per_class", type=int, default=100)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--device", default="cpu")
	ap.add_argument("--download", action="store_true")
	ap.add_argument("--batch_size", type=int, default=0, help="0 = use run meta batch_size")
	ap.add_argument("--max_batches", type=int, default=0, help="0 = no limit")
	ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
	ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
	ap.add_argument("--write_bundle", action=argparse.BooleanOptionalAction, default=True)

	ap.add_argument("--bench_metric", default="f1_macro")
	ap.add_argument("--selection_strict_min_abs_rho", type=float, default=0.6)
	ap.add_argument("--selection_strict_max_p", type=float, default=0.05)

	ap.add_argument("--compare_two_layers_top_k", type=int, default=0)
	ap.add_argument("--compare_out_png", default="", help="Optional explicit path for the 2-layer compare PNG.")
	ap.add_argument("--neighbors_out_png", default="", help="Optional explicit path for a single-layer neighbors PNG.")
	ap.add_argument("--neighbors_layer", default="", help="Layer name for neighbors_out_png (default: first layer).")
	ap.add_argument(
		"--neighbors_style",
		choices=["paper", "compact"],
		default="paper",
		help="Neighbor panel style for neighbors_out_png (paper=1400x620, compact=1225x620).",
	)
	ap.add_argument(
		"--write_layer_pngs",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Write analysis/embedding_retrieval_<tag>__layer_<layer>.png for each evaluated layer.",
	)
	ap.add_argument("--anchor_idx", type=int, default=-1)
	ap.add_argument("--search_best_illustration_anchor", action="store_true")
	ap.add_argument("--search_min_delta_neighbors", type=int, default=0)
	return ap.parse_args(argv)


def _read_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _load_meta(run_dir: str) -> Dict[str, Any]:
	mp = os.path.join(os.path.abspath(run_dir), "meta.json")
	if not os.path.isfile(mp):
		raise FileNotFoundError(mp)
	obj = _read_json(mp)
	return obj if isinstance(obj, dict) else {}


def _resolve_checkpoint_path(run_dir: str, ckpt: str) -> str:
	c = str(ckpt).strip()
	if os.path.isabs(c) and os.path.isfile(c):
		return c
	if os.path.isfile(os.path.join(os.path.abspath(run_dir), c)):
		return os.path.join(os.path.abspath(run_dir), c)
	if c == "best_main":
		return os.path.join(os.path.abspath(run_dir), "checkpoints", "model_best_main.pt")
	if c == "last":
		return os.path.join(os.path.abspath(run_dir), "checkpoints", "model_last.pt")
	if c == "early_signal":
		return os.path.join(os.path.abspath(run_dir), "checkpoints", "model_early_signal.pt")
	raise FileNotFoundError(f"Unknown checkpoint spec: {ckpt}")


def _safe_name(s: str) -> str:
	out = []
	for ch in str(s):
		if ch.isalnum() or ch in ("_", ".", "-"):
			out.append(ch)
		else:
			out.append("_")
	return "".join(out)


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
	if isinstance(x, torch.Tensor):
		return x
	if isinstance(x, (tuple, list)):
		for v in x:
			t = _first_tensor(v)
			if t is not None:
				return t
		return None
	if isinstance(x, Mapping):
		for v in x.values():
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
	z = repr_from_activation(act)
	if isinstance(z, torch.Tensor):
		return z
	t = _first_tensor(act)
	return repr_from_activation(t)


def _collect_embeddings(
	model: Any,
	loader: Any,
	*,
	layers: list[str],
	device: torch.device,
	preprocess: Optional[Any],
	max_batches: int,
	max_samples: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
	emb_chunks: Dict[str, List[torch.Tensor]] = {k: [] for k in layers}
	lab_chunks: List[torch.Tensor] = []
	n_total = 0

	model.eval()
	with torch.no_grad(), LayerTaps(model, layers) as taps:
		for bi, batch in enumerate(loader):
			if int(max_batches) > 0 and bi >= int(max_batches):
				break
			if isinstance(batch, Mapping):
				b = move_to_device(batch, device)
				out = model(**b)
				labels = b.get("labels", None)
				if labels is None:
					raise RuntimeError("Text batch must contain labels.")
				_ = out
			else:
				x, y = batch
				x = move_to_device(x, device)
				if preprocess is not None:
					x = preprocess(x)
				_ = model(x)
				labels = move_to_device(y, device)

			lab_chunks.append(labels.detach().view(-1).cpu())
			bs = int(labels.shape[0])
			n_total += bs
			for ln in layers:
				act = taps.outputs.get(ln)
				z = _repr_from_hook_output(act)
				if z is None:
					raise RuntimeError(f"Could not extract tensor representation from layer={ln!r}.")
				emb_chunks[ln].append(z.detach().cpu())

			if int(max_samples) > 0 and n_total >= int(max_samples):
				break

	labels_all = torch.cat(lab_chunks, dim=0)
	out_emb: Dict[str, torch.Tensor] = {}
	for ln in layers:
		out_emb[ln] = torch.cat(emb_chunks[ln], dim=0)
	return out_emb, labels_all


def _neighbors_topk(emb: torch.Tensor, anchor_idx: int, top_k: int) -> torch.Tensor:
	e = emb.float()
	v = e[int(anchor_idx)]
	d2 = torch.sum((e - v) ** 2, dim=1)
	d2[int(anchor_idx)] = float("inf")
	k = max(1, min(int(top_k), int(e.shape[0]) - 1))
	return torch.topk(d2, k=k, largest=False).indices


def _sample_candidate_anchors(labels: torch.Tensor, *, anchors_per_class: int, seed: int) -> list[int]:
	labs = labels.view(-1).long()
	classes = sorted(set(int(x) for x in labs.tolist()))
	rng = np.random.default_rng(int(seed))
	candidates: list[int] = []
	seen: set[int] = set()
	for c in classes:
		idx = torch.nonzero(labs == int(c)).view(-1)
		if idx.numel() < 2:
			continue
		choices = idx.detach().cpu().numpy()
		n_a = min(int(anchors_per_class), int(choices.shape[0]))
		anchors = rng.choice(choices, size=n_a, replace=False).tolist()
		for x in anchors:
			ai = int(x)
			if ai not in seen:
				seen.add(ai)
				candidates.append(ai)
	return candidates


def _search_best_anchor_for_two_layers(
	emb_a: torch.Tensor,
	emb_b: torch.Tensor,
	labels: torch.Tensor,
	*,
	top_k: int,
	anchors_per_class: int,
	seed: int,
	min_delta_neighbors: int,
) -> int:
	cands = _sample_candidate_anchors(labels, anchors_per_class=int(anchors_per_class), seed=int(seed))
	if not cands:
		return 0
	best_anchor = int(cands[0])
	best_score: tuple[int, int, int] = (-1, -1, -1)
	labs = labels.view(-1).long()
	for ai in cands:
		anchor_y = int(labs[int(ai)].item())
		na = _neighbors_topk(emb_a, int(ai), int(top_k)).detach().cpu().tolist()
		nb = _neighbors_topk(emb_b, int(ai), int(top_k)).detach().cpu().tolist()
		na_i = [int(x) for x in na]
		nb_i = [int(x) for x in nb]
		hit_a = int((labs[torch.tensor(na_i, dtype=torch.long)] == int(anchor_y)).sum().item())
		hit_b = int((labs[torch.tensor(nb_i, dtype=torch.long)] == int(anchor_y)).sum().item())
		d = int(abs(int(hit_a) - int(hit_b)))
		if int(d) < int(min_delta_neighbors):
			continue
		score = (int(d), int(max(hit_a, hit_b)), int(hit_a + hit_b))
		if score > best_score:
			best_score = score
			best_anchor = int(ai)
	return int(best_anchor)


def _layer_retrieval_report(
	emb: torch.Tensor,
	labels: torch.Tensor,
	*,
	top_k: int,
	anchors_per_class: int,
	seed: int,
) -> Dict[str, Any]:
	labs = labels.view(-1).long()
	classes = sorted(set(int(x) for x in labs.tolist()))
	rng = np.random.default_rng(int(seed))
	per_class = []
	all_anchor_ratios = []
	for c in classes:
		idx = torch.nonzero(labs == int(c)).view(-1)
		if idx.numel() < 2:
			continue
		choices = idx.detach().cpu().numpy()
		n_a = min(int(anchors_per_class), int(choices.shape[0]))
		anchors = rng.choice(choices, size=n_a, replace=False).tolist()
		ratios = []
		for ai in anchors:
			nn = _neighbors_topk(emb, int(ai), int(top_k))
			hits = int((labs[nn] == int(c)).sum().item())
			ratios.append(float(hits) / float(max(int(top_k), 1)))
		mu = float(np.mean(ratios)) if ratios else 0.0
		sd = float(np.std(ratios, ddof=1)) if len(ratios) >= 2 else 0.0
		per_class.append({"class": int(c), "same_class_ratio_mean": mu, "same_class_ratio_std": sd})
		all_anchor_ratios.append(mu)
	macro = float(np.mean(all_anchor_ratios)) if all_anchor_ratios else 0.0
	macro_sd = float(np.std(all_anchor_ratios, ddof=1)) if len(all_anchor_ratios) >= 2 else 0.0
	return {
		"n_samples": int(labs.shape[0]),
		"macro_same_class_ratio": macro,
		"macro_same_class_ratio_std": macro_sd,
		"per_class": per_class,
	}


def _write_two_layer_compare_png(
	out_path: str,
	*,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	anchor_class: int,
	layer_a: str,
	neighbors_a: list[int],
	layer_b: str,
	neighbors_b: list[int],
	top_k: int,
) -> None:
	k = int(max(1, min(int(top_k), 11)))
	idxs_a = [int(anchor_idx)] + [int(x) for x in neighbors_a[:k]]
	idxs_b = [int(anchor_idx)] + [int(x) for x in neighbors_b[:k]]
	if len(idxs_a) < 12 or len(idxs_b) < 12:
		raise RuntimeError("Not enough neighbors to render the 2x6 grid.")

	def _img(i: int):
		ex = dataset[int(i)]
		x = ex[0] if isinstance(ex, (tuple, list)) else ex.get("image")
		if hasattr(x, "permute") and hasattr(x, "numpy"):
			t = x
			if t.dim() == 3 and int(t.shape[0]) in (1, 3):
				t = t.permute(1, 2, 0)
			return np.asarray(t)
		return np.asarray(x)

	def _edge(i: int) -> str:
		if int(i) == int(anchor_idx):
			return "yellow"
		return "green" if int(labels[int(i)]) == int(anchor_class) else "red"

	dpi = 100
	fig = plt.figure(figsize=(10.22, 7.04), dpi=dpi)
	gs = fig.add_gridspec(
		nrows=5,
		ncols=6,
		height_ratios=[1.0, 1.0, 0.18, 1.0, 1.0],
		left=0.02,
		right=0.98,
		bottom=0.03,
		top=0.92,
		wspace=0.05,
		hspace=0.10,
	)
	fig.text(0.5, 0.975, f"Layer: {layer_a}", ha="center", va="top", fontsize=14, fontweight="bold")
	ax_mid = fig.add_subplot(gs[2, :])
	ax_mid.set_axis_off()
	ax_mid.text(0.5, 0.5, f"Layer: {layer_b}", ha="center", va="center", fontsize=14, fontweight="bold", transform=ax_mid.transAxes)

	def _draw(idxs: list[int], row0: int) -> None:
		for pos in range(12):
			r = pos // 6
			c = pos % 6
			ax = fig.add_subplot(gs[row0 + r, c])
			ax.set_xticks([])
			ax.set_yticks([])
			i = int(idxs[pos])
			ax.imshow(_img(i))
			for sp in ax.spines.values():
				sp.set_linewidth(2.4)
				sp.set_edgecolor(_edge(i))

	_draw(idxs_a, 0)
	_draw(idxs_b, 3)

	ensure_dir(os.path.dirname(os.path.abspath(out_path)))
	fig.savefig(out_path, dpi=dpi)
	plt.close(fig)


def _write_single_layer_neighbors_png(
	out_path: str,
	*,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	neighbors: list[int],
	top_k: int,
	style: str = "paper",
) -> None:
	labs = labels.view(-1).long()
	cols, rows = 7, 3
	max_cells = int(cols * rows)
	max_neighbors = int(max_cells - 1)
	k = int(max(1, min(int(top_k), int(len(neighbors)), max_neighbors)))
	idxs: list[int] = [int(anchor_idx)] + [int(x) for x in neighbors[:k]]

	def _img(i: int):
		ex = dataset[int(i)]
		x = ex[0] if isinstance(ex, (tuple, list)) else ex.get("image")
		if hasattr(x, "permute") and hasattr(x, "numpy"):
			t = x
			if t.dim() == 3 and int(t.shape[0]) in (1, 3):
				t = t.permute(1, 2, 0)
			return np.asarray(t)
		return np.asarray(x)

	def _edge(i: int) -> str:
		if int(i) == int(anchor_idx):
			return "yellow"
		return "green" if int(labs[int(i)].item()) == int(labs[int(anchor_idx)].item()) else "red"

	dpi = 100
	mode = str(style).strip().lower()
	if mode not in {"paper", "compact"}:
		raise ValueError(f"Unknown neighbors style: {style!r}")
	fig_w = 14.0 if mode == "paper" else 14.0 * (7.0 / 8.0)
	fig_h = 6.2
	fig = plt.figure(figsize=(float(fig_w), float(fig_h)), dpi=int(dpi))
	gs = fig.add_gridspec(
		nrows=rows,
		ncols=cols,
		left=0.02,
		right=0.98,
		bottom=0.04,
		top=0.90,
		wspace=0.06,
		hspace=0.28,
	)
	fig.text(
		0.5,
		0.965,
		"Anchor (yellow) and nearest neighbors (green same class, red different class)",
		ha="center",
		va="top",
		fontsize=11,
	)
	lw_anchor = 3.0
	lw_other = 2.4
	for pos in range(min(len(idxs), max_cells)):
		r = int(pos // cols)
		c = int(pos % cols)
		ax = fig.add_subplot(gs[r, c])
		ax.set_xticks([])
		ax.set_yticks([])
		i = int(idxs[pos])
		ax.set_title(f"id={int(i)}, y={int(labs[int(i)].item())}", fontsize=8.0, pad=2.0)
		ax.imshow(_img(i))
		lw = float(lw_anchor) if int(i) == int(anchor_idx) else float(lw_other)
		for sp in ax.spines.values():
			sp.set_linewidth(lw)
			sp.set_edgecolor(_edge(i))
	ensure_dir(os.path.dirname(os.path.abspath(out_path)))
	fig.savefig(out_path, dpi=dpi)
	plt.close(fig)


def _example_to_text(ex: Any) -> str:
	if isinstance(ex, dict):
		for k in ("text", "sentence", "question", "content", "title"):
			if k in ex:
				return str(ex.get(k, "") or "")
		return str(ex)
	if isinstance(ex, (tuple, list)) and ex:
		return str(ex[0])
	return str(ex)


def _write_single_layer_neighbors_text_png(
	out_path: str,
	*,
	dataset: Any,
	labels: torch.Tensor,
	anchor_idx: int,
	neighbors: list[int],
	top_k: int,
) -> None:
	labs = labels.view(-1).long()
	max_k = int(min(int(top_k), int(len(neighbors)), 20))
	idxs = [int(anchor_idx)] + [int(x) for x in neighbors[:max_k]]

	def _edge(i: int) -> str:
		if int(i) == int(anchor_idx):
			return "gold"
		return "green" if int(labs[int(i)].item()) == int(labs[int(anchor_idx)].item()) else "red"

	def _label(i: int) -> str:
		ex = dataset[int(i)]
		txt = _example_to_text(ex).strip()
		txt = textwrap.fill(txt, width=44)
		return f"id={int(i)}, y={int(labs[int(i)].item())} \u2014 {txt}"

	dpi = 100
	fig = plt.figure(figsize=(11.2, 5.52), dpi=dpi)
	ax = fig.add_subplot(111)
	ax.set_axis_off()

	fig.text(0.5, 0.97, "Anchor (yellow) and nearest neighbors (green/red)", ha="center", va="top", fontsize=12)

	anchor_txt = _label(int(anchor_idx))
	fig.text(
		0.5,
		0.88,
		anchor_txt,
		ha="center",
		va="top",
		fontsize=9,
		color="goldenrod",
		bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": _edge(int(anchor_idx)), "linewidth": 1.6},
	)

	left_x, right_x = 0.25, 0.75
	rows = 10
	ys = np.linspace(0.78, 0.12, num=rows)
	for j in range(min(max_k, rows)):
		i = int(idxs[1 + j])
		fig.text(
			left_x,
			float(ys[j]),
			_label(i),
			ha="center",
			va="center",
			fontsize=8.5,
			color=_edge(i),
			bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": _edge(i), "linewidth": 1.0, "alpha": 0.22},
		)
	for j in range(min(max_k - rows, rows)):
		i = int(idxs[1 + rows + j])
		fig.text(
			right_x,
			float(ys[j]),
			_label(i),
			ha="center",
			va="center",
			fontsize=8.5,
			color=_edge(i),
			bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": _edge(i), "linewidth": 1.0, "alpha": 0.22},
		)

	ensure_dir(os.path.dirname(os.path.abspath(out_path)))
	fig.savefig(out_path, dpi=dpi)
	plt.close(fig)


def _bundle_matches_request(obj: Any, *, layers: list[str], top_k: int, anchors_per_class: int, seed: int) -> bool:
	if not isinstance(obj, dict):
		return False
	ev = obj.get("eval", None)
	if not isinstance(ev, dict):
		return False
	if ev.get("top_k") != int(top_k) or ev.get("anchors_per_class") != int(anchors_per_class) or ev.get("seed") != int(seed):
		return False
	blk = obj.get("layers", None)
	if not isinstance(blk, dict):
		return False
	return all(str(l) in blk for l in layers)


def _expected_layer_png_paths(analysis_dir: str, *, tag: str, layers: list[str]) -> list[str]:
	out = []
	for ln in layers:
		fn = f"embedding_retrieval_{str(tag)}__layer_{_safe_name(str(ln))}.png"
		out.append(os.path.join(str(analysis_dir), fn))
	return out


def main(argv: Optional[list[str]] = None) -> None:
	args = _parse_args(argv)
	run_dir = os.path.abspath(str(args.run_dir))
	meta = _load_meta(run_dir)
	task = str(meta.get("task", (meta.get("args") or {}).get("task", "cv"))).strip().lower()
	dataset = str(meta.get("dataset", (meta.get("args") or {}).get("dataset", ""))).strip()
	model_name = str(meta.get("model", (meta.get("args") or {}).get("model", ""))).strip()
	if not dataset or not model_name:
		raise ValueError("Could not infer dataset/model from run meta.")

	meta_args = meta.get("args") or {}
	device = torch.device(str(args.device).strip() or str(meta_args.get("device", "cpu")))
	batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(meta_args.get("batch_size", 64))
	data_root = str(meta_args.get("data_root", "./data"))
	tokenizer_name = resolve_tokenizer_name(str(meta_args.get("tokenizer_name", "") or "").strip() or str(model_name))

	bundle = get_dataset(dataset, root=data_root, download=bool(args.download), tokenizer_name=tokenizer_name)
	loaders = make_dataloaders(bundle, batch_size=batch_size, num_workers=0)
	requested_split = str(args.split)
	loader = loaders.get(requested_split)
	if loader is None and requested_split == "test":
		loader = loaders.get("val")
	if loader is None:
		raise RuntimeError(f"No dataloader for split '{args.split}'.")
	effective_split = requested_split if loaders.get(requested_split) is not None else "val"

	ckpt_path = _resolve_checkpoint_path(run_dir, str(args.checkpoint))
	ckpt = torch.load(ckpt_path, map_location="cpu")
	state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
	if not isinstance(state, dict):
		raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

	ds_nc = bundle.train if bundle.train is not None else (bundle.val or bundle.test)
	num_classes = int(meta.get("num_classes", 0) or 0)
	if num_classes <= 0:
		num_classes = infer_num_classes(dataset, ds_nc)

	preprocess = None
	if task == "cv":
		model, preprocess, model_layer_names = build_cv_model(
			model_name,
			num_classes=int(num_classes),
			device=device,
			pretrained=False,
			input_flat_dim=None,
		)
	else:
		obj = str(meta_args.get("nlp_objective", "classification") or "classification").strip().lower()
		model, model_layer_names = build_text_model(
			model_name,
			num_classes=int(num_classes),
			device=device,
			pretrained=False,
			objective=str(obj),
		)

	try:
		model.load_state_dict(state, strict=True)
	except Exception as e:
		raise RuntimeError(f"Failed to load checkpoint into model: {ckpt_path}") from e
	model.eval()

	if bool(args.all_main_layers):
		req_layers = [str(x) for x in model_layer_names]
	else:
		req_layers = csv_to_list(str(args.layers))
		if not req_layers:
			req_layers = [str(x) for x in (meta.get("monitor", {}) or {}).get("layer_names", [])]
	all_modules = set(list_module_names(model))
	layers = [x for x in req_layers if x in all_modules]
	if not layers:
		raise ValueError("No valid embedding layers were provided/found in the model.")
	if int(args.compare_two_layers_top_k) > 0 and len(layers) != 2:
		raise ValueError("--compare_two_layers_top_k requires exactly 2 layers.")

	analysis_dir = ensure_dir(os.path.join(run_dir, "analysis"))
	tag = os.path.splitext(os.path.basename(str(ckpt_path)))[0]
	bundle_path = os.path.join(analysis_dir, f"embedding_retrieval_{tag}.json")

	if bool(args.skip_existing) and os.path.isfile(bundle_path):
		prev = _read_json(bundle_path)
		if _bundle_matches_request(
			prev,
			layers=list(layers),
			top_k=int(args.top_k),
			anchors_per_class=int(args.anchors_per_class),
			seed=int(args.seed),
		) and int(args.compare_two_layers_top_k) <= 0:
			if bool(args.write_bundle) and bool(args.write_layer_pngs):
				exp_pngs = _expected_layer_png_paths(analysis_dir, tag=str(tag), layers=list(layers))
				if all(os.path.isfile(p) for p in exp_pngs):
					print("[EmbeddingEval] skip_existing:", bundle_path)
					return
			else:
				print("[EmbeddingEval] skip_existing:", bundle_path)
				return

	emb_by_layer, labels = _collect_embeddings(
		model,
		loader,
		layers=list(layers),
		device=device,
		preprocess=preprocess,
		max_batches=int(args.max_batches),
		max_samples=int(args.max_samples),
	)

	layer_reports = []
	r_by_layer: Dict[str, float] = {}
	labs = labels.view(-1).long()
	anchor_idx = int(args.anchor_idx) if int(args.anchor_idx) >= 0 else 0
	if int(anchor_idx) < 0 or int(anchor_idx) >= int(labs.shape[0]):
		raise ValueError(f"anchor_idx out of range: {anchor_idx} (n={int(labs.shape[0])})")

	if int(args.anchor_idx) < 0 and bool(args.search_best_illustration_anchor) and len(layers) >= 2 and (
		int(args.compare_two_layers_top_k) > 0 or str(args.neighbors_out_png).strip() or (bool(args.write_bundle) and bool(args.write_layer_pngs))
	):
		la, lb = str(layers[0]), str(layers[1])
		anchor_idx = _search_best_anchor_for_two_layers(
			emb_by_layer[str(la)],
			emb_by_layer[str(lb)],
			labels,
			top_k=int(args.compare_two_layers_top_k) if int(args.compare_two_layers_top_k) > 0 else int(args.top_k),
			anchors_per_class=int(args.anchors_per_class),
			seed=int(args.seed),
			min_delta_neighbors=int(args.search_min_delta_neighbors),
		)

	anchor_class = int(labs[int(anchor_idx)].item())

	if bool(args.write_bundle):
		for ln in layers:
			rep = _layer_retrieval_report(
				emb_by_layer[str(ln)],
				labels,
				top_k=int(args.top_k),
				anchors_per_class=int(args.anchors_per_class),
				seed=int(args.seed),
			)
			layer_reports.append({"layer": str(ln), **rep})
			r_by_layer[str(ln)] = float(rep["macro_same_class_ratio"])

		corr_csv = os.path.join(run_dir, "correlations_report", "all_pairs.csv")
		selection_rows, selection_meta = build_selection_rows(
			r_by_layer,
			model_name=str(model_name),
			dataset_slug=str(dataset),
			corr_csv_path=str(corr_csv),
			strict_min_abs_rho=float(args.selection_strict_min_abs_rho),
			strict_max_p=float(args.selection_strict_max_p),
			bench_metric=str(args.bench_metric),
		)

		out_obj: Dict[str, Any] = {
			"run_dir": run_dir,
			"task": str(task),
			"dataset": str(dataset),
			"model": str(model_name),
			"checkpoint_file": os.path.basename(str(ckpt_path)),
			"illustration_anchor": {"index": int(anchor_idx), "class": int(anchor_class)},
			"split": {"requested": str(requested_split), "effective": str(effective_split)},
			"eval": {
				"top_k": int(args.top_k),
				"anchors_per_class": int(args.anchors_per_class),
				"seed": int(args.seed),
				"bench_metric": str(args.bench_metric).strip().lower(),
				"strict_min_abs_rho": float(args.selection_strict_min_abs_rho),
				"strict_max_p": float(args.selection_strict_max_p),
			},
			"layers_order": [str(x) for x in layers],
			"layers": {str(r["layer"]): {k: r[k] for k in ("n_samples", "macro_same_class_ratio", "macro_same_class_ratio_std", "per_class")} for r in layer_reports},
			"r_by_layer": {str(k): float(v) for k, v in r_by_layer.items()},
			"top_by_macro_same_class_ratio": [
				{"layer": str(k), "macro_same_class_ratio": float(v), "is_signal_layer": False}
				for k, v in sorted(r_by_layer.items(), key=lambda kv: float(kv[1]), reverse=True)
			],
			"selection": {"meta": selection_meta, "rows": selection_rows},
		}
		with open(bundle_path, "w", encoding="utf-8") as f:
			json.dump(out_obj, f, ensure_ascii=False, indent=2)

	if bool(args.write_bundle) and bool(args.write_layer_pngs):
		for ln in layers:
			nn = _neighbors_topk(emb_by_layer[str(ln)], int(anchor_idx), int(args.top_k)).detach().cpu().tolist()
			out_png = os.path.join(analysis_dir, f"embedding_retrieval_{tag}__layer_{_safe_name(str(ln))}.png")
			if task == "cv":
				_write_single_layer_neighbors_png(
					out_png,
					dataset=loader.dataset,
					labels=labs.cpu(),
					anchor_idx=int(anchor_idx),
					neighbors=[int(x) for x in nn],
					top_k=int(args.top_k),
					style="paper",
				)
			else:
				_write_single_layer_neighbors_text_png(
					out_png,
					dataset=loader.dataset,
					labels=labs.cpu(),
					anchor_idx=int(anchor_idx),
					neighbors=[int(x) for x in nn],
					top_k=int(args.top_k),
				)

	if task == "cv" and int(args.compare_two_layers_top_k) > 0:
			la, lb = str(layers[0]), str(layers[1])
			na = _neighbors_topk(emb_by_layer[la], anchor_idx, int(args.compare_two_layers_top_k)).detach().cpu().tolist()
			nb = _neighbors_topk(emb_by_layer[lb], anchor_idx, int(args.compare_two_layers_top_k)).detach().cpu().tolist()
			out_cmp = str(args.compare_out_png).strip() or os.path.join(
				analysis_dir, f"embedding_retrieval_{tag}__compare_{_safe_name(la)}_vs_{_safe_name(lb)}.png"
			)
			_write_two_layer_compare_png(
				out_cmp,
				dataset=loader.dataset,
				labels=labs.cpu(),
				anchor_idx=int(anchor_idx),
				anchor_class=int(anchor_class),
				layer_a=la,
				neighbors_a=[int(x) for x in na],
				layer_b=lb,
				neighbors_b=[int(x) for x in nb],
				top_k=int(args.compare_two_layers_top_k),
			)

	if str(args.neighbors_out_png).strip():
		lay = str(args.neighbors_layer).strip()
		if not lay:
			lay = str(layers[0])
		if lay not in emb_by_layer:
			raise ValueError(f"neighbors_layer is not in evaluated layers: {lay}")
		nn = _neighbors_topk(emb_by_layer[str(lay)], int(anchor_idx), int(args.top_k)).detach().cpu().tolist()
		if task == "cv":
			_write_single_layer_neighbors_png(
				str(args.neighbors_out_png).strip(),
				dataset=loader.dataset,
				labels=labs.cpu(),
				anchor_idx=int(anchor_idx),
				neighbors=[int(x) for x in nn],
				top_k=int(args.top_k),
				style=str(args.neighbors_style),
			)
		else:
			_write_single_layer_neighbors_text_png(
				str(args.neighbors_out_png).strip(),
				dataset=loader.dataset,
				labels=labs.cpu(),
				anchor_idx=int(anchor_idx),
				neighbors=[int(x) for x in nn],
				top_k=int(args.top_k),
			)

	if bool(args.write_bundle):
		print("[EmbeddingEval] wrote:", bundle_path)


if __name__ == "__main__":
	main()
