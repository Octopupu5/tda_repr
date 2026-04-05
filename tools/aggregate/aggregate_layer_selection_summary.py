from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.aggregate.embedding_selection import _layer_metric_from_repr_key


def _metric_group_from_repr_key(repr_key: str) -> str:
	k = str(repr_key)
	if ".hodge_L_q0_lambda" in k or ".hodge_L_q1_lambda" in k or ".persistent_q0_lambda" in k or ".persistent_q1_lambda" in k:
		return "spectral"
	if ".mtopdiv" in k:
		return "mtopdiv"
	if any(x in k for x in (".beta", ".gudhi_", ".graph_")):
		return "topo"
	return "other"


def _extract_layer_from_repr_key(repr_key: str) -> Optional[str]:
	return _layer_metric_from_repr_key(str(repr_key))[0]


def _read_top_layers_from_correlations(
	csv_path: str,
	selection_group: str,
	top_n_layers: int,
	min_abs_rho: float,
	max_p: float,
) -> List[str]:
	if not os.path.exists(csv_path):
		return []
	rows: List[Tuple[float, str]] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			repr_key = str(r.get("repr_key", "") or "")
			group = str(selection_group)
			if group == "topo_mtopdiv":
				group = "topo"
			if _metric_group_from_repr_key(repr_key) != group:
				continue
			try:
				abs_rho = float(r.get("abs_rho", "nan"))
				p_val = float(r.get("p", "nan"))
			except Exception:
				continue
			if not math.isfinite(abs_rho) or not math.isfinite(p_val):
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


def _layer_selection_config() -> Tuple[str, List[str], Dict[str, Tuple[str, str]], Dict[str, str]]:
	header = (
		r"\textbf{Architecture} & \textbf{Default Layer} & "
		r"\textbf{Avg. Relative Deviation in $R$ (Default)} & "
		r"\textbf{Avg. Relative Deviation in $R$ (Proposed)} \\"
	)
	arch_order = ["mlp", "resnet18", "efficientnet_b0", "convnext_tiny", "distilbert"]
	arch_tex = {
		"mlp": ("MLP", "b13"),
		"resnet18": ("ResNet18", "b14"),
		"efficientnet_b0": ("EfficientNet-B0", "b15"),
		"convnext_tiny": ("ConvNeXt-Tiny", "b16"),
		"distilbert": ("DistilBERT", "b17"),
	}
	default_layer = {
		"mlp": "1",
		"resnet18": "avgpool",
		"efficientnet_b0": "classifier.0",
		"convnext_tiny": "avgpool",
		"distilbert": "pre_classifier",
	}
	return header, arch_order, arch_tex, default_layer


def _canonical_model_key(raw: str) -> str:
	return str(raw or "").strip().lower()


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
	xs = [float(x) for x in vals if math.isfinite(float(x))]
	if not xs:
		return float("nan"), float("nan")
	mu = sum(xs) / len(xs)
	if len(xs) < 2:
		return mu, 0.0
	var = sum((float(x) - mu) ** 2 for x in xs) / float(len(xs) - 1)
	return mu, math.sqrt(max(0.0, var))


def _rel_dev_pct(r_star: float, r_x: float) -> float:
	rs = max(float(r_star), 1e-12)
	return max(0.0, (float(r_star) - float(r_x)) / rs * 100.0)


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as rf:
		return json.load(rf)


def _is_embedding_bundle_json(filename: str) -> bool:
	"""True for consolidated `embedding_retrieval_<tag>.json`, excluding legacy artifacts."""
	fn = str(filename)
	if not (fn.startswith("embedding_retrieval_") and fn.endswith(".json")):
		return False
	if "__layer_" in fn:
		return False
	if fn.endswith("_selection.json"):
		return False
	if "__summary.json" in fn:
		return False
	return True


def _discover_summaries(roots: Sequence[str]) -> List[str]:
	out_bundles: List[str] = []
	out_legacy: List[str] = []
	for rt in roots:
		r_abs = os.path.abspath(rt)
		if not os.path.isdir(r_abs):
			print("[warn] missing root:", r_abs, file=sys.stderr)
			continue
		for dp, _, fns in os.walk(r_abs):
			for fn in fns:
				if _is_embedding_bundle_json(fn):
					out_bundles.append(os.path.join(dp, fn))
				elif fn.startswith("embedding_retrieval_") and fn.endswith("__summary.json"):
					out_legacy.append(os.path.join(dp, fn))
	bundle_keys = set()
	for p in out_bundles:
		dp = os.path.dirname(p)
		bn = os.path.basename(p)
		tag = bn[len("embedding_retrieval_") : -len(".json")]
		bundle_keys.add((dp, tag))
	filtered_legacy: List[str] = []
	for p in out_legacy:
		dp = os.path.dirname(p)
		bn = os.path.basename(p)
		suf = "__summary.json"
		pre = "embedding_retrieval_"
		if bn.startswith(pre) and bn.endswith(suf):
			tag = bn[len(pre) : -len(suf)]
			if (dp, tag) in bundle_keys:
				continue
		filtered_legacy.append(p)
	return sorted(set(out_bundles + filtered_legacy))


def _ratio_by_layer(summary: Mapping[str, Any], summary_path: str = "") -> Dict[str, float]:
	rb = summary.get("r_by_layer")
	if isinstance(rb, dict) and rb:
		out: Dict[str, float] = {}
		for k, v in rb.items():
			try:
				out[str(k)] = float(v)
			except Exception:
				continue
		if out:
			return out
	if str(summary_path):
		alt_sel = str(summary_path).replace("__summary.json", "_selection.json")
		if os.path.isfile(alt_sel):
			try:
				sel = _load_json(alt_sel)
				rb2 = sel.get("r_by_layer")
				if isinstance(rb2, dict) and rb2:
					out2: Dict[str, float] = {}
					for k, v in rb2.items():
						try:
							out2[str(k)] = float(v)
						except Exception:
							continue
					if out2:
						return out2
			except Exception:
				pass
	layers_blk = summary.get("layers")
	if isinstance(layers_blk, dict) and layers_blk:
		out_l: Dict[str, float] = {}
		for k, row in layers_blk.items():
			if not isinstance(row, dict) or "macro_same_class_ratio" not in row:
				continue
			try:
				out_l[str(k)] = float(row["macro_same_class_ratio"])
			except Exception:
				continue
		if out_l:
			return out_l
	metrics = summary.get("metrics")
	if isinstance(metrics, dict) and metrics:
		out_m: Dict[str, float] = {}
		for k, row in metrics.items():
			if not isinstance(row, dict) or "R" not in row:
				continue
			try:
				out_m[str(k)] = float(row["R"])
			except Exception:
				continue
		if out_m:
			return out_m
	out = {}
	rows = summary.get("rows")
	if not isinstance(rows, list):
		return out
	for r in rows:
		if not isinstance(r, dict):
			continue
		if "macro_avg_same_class_ratio" not in r or "layer" not in r:
			continue
		out[str(r["layer"])] = float(r["macro_avg_same_class_ratio"])
	return out


def _r_star(by_layer: Mapping[str, float]) -> Optional[float]:
	if not by_layer:
		return None
	return max(float(v) for v in by_layer.values())


def _proposed_candidate_layers(
	run_dir: str,
	*,
	top_n_layers: int,
	min_abs_rho: float,
	max_p: float,
) -> List[str]:
	csv_path = os.path.join(run_dir, "correlations_report", "all_pairs.csv")
	if not os.path.isfile(csv_path):
		return []

	best_abs_by_layer: Dict[str, float] = {}
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			repr_key = str(r.get("repr_key", "") or "")
			g = _metric_group_from_repr_key(repr_key)
			if g not in {"topo", "spectral", "mtopdiv"}:
				continue
			try:
				abs_rho = float(r.get("abs_rho", "nan"))
				p_val = float(r.get("p", "nan"))
			except Exception:
				continue
			if not math.isfinite(abs_rho) or not math.isfinite(p_val):
				continue
			if abs_rho < float(min_abs_rho) or p_val > float(max_p):
				continue
			layer = _extract_layer_from_repr_key(repr_key)
			if not layer:
				continue
			prev = best_abs_by_layer.get(str(layer))
			if prev is None or float(abs_rho) > float(prev):
				best_abs_by_layer[str(layer)] = float(abs_rho)

	ranked = sorted(best_abs_by_layer.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
	return [k for k, _v in ranked[: int(top_n_layers)]]


def _tex_tt(s: str) -> str:
	return str(s).replace("_", r"\_")


def _write_tabular_body_tex(tex_path: str, body_lines: Sequence[str]) -> None:
	path_out = os.path.abspath(tex_path)
	os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
	with open(path_out, "w", encoding="utf-8") as wf:
		for ln in body_lines:
			wf.write(ln + "\n")


def write_layer_selection_summary_tex(
	*,
	roots: Sequence[str],
	update_tex: str,
	exclude_models: str = "",
	defaults_json: str = "",
	top_corr_layers: int = 8,
	corr_min_abs_rho: float = 0.0,
	corr_max_p: float = 1.0,
) -> None:
	"""
	Write `table_layer_selection_summary.tex` as a booktabs fragment (no outer table).

	Used by the reproduction pipeline (`tools/reproduce_tables.py`).
	"""
	excl = {x.strip().lower() for x in str(exclude_models).split(",") if x.strip()}
	header, arch_order, arch_tex, default_layer_base = _layer_selection_config()
	default_layer = dict(default_layer_base)
	if str(defaults_json).strip():
		ov = _load_json(str(defaults_json).strip())
		if isinstance(ov, dict):
			default_layer.update({str(k).strip().lower(): str(v) for k, v in ov.items()})

	by_model: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
	for sp in _discover_summaries(list(roots)):
		summary = _load_json(sp)
		run_dir = str(summary.get("run_dir") or "").strip()
		if not run_dir or not os.path.isdir(run_dir):
			run_dir = os.path.abspath(os.path.join(os.path.dirname(sp), ".."))
		meta_path = os.path.join(run_dir, "meta.json")
		if not os.path.isfile(meta_path):
			continue
		meta = _load_json(meta_path)
		model = str(meta.get("model", "") or "").strip()
		if not model:
			model = str((meta.get("args") or {}).get("model", "") or "").strip()
		model_lc = _canonical_model_key(model)
		if not model_lc or model_lc in excl:
			continue
		by_layer = _ratio_by_layer(summary, sp)
		rbest = _r_star(by_layer)
		if rbest is None:
			continue
		dl = default_layer.get(model_lc)
		r_def = by_layer.get(str(dl)) if dl else None
		cand = _proposed_candidate_layers(
			run_dir,
			top_n_layers=int(top_corr_layers),
			min_abs_rho=float(corr_min_abs_rho),
			max_p=float(corr_max_p),
		)
		r_props = [by_layer[str(x)] for x in cand if str(x) in by_layer]
		r_prop = max(r_props) if r_props else None
		row = {
			"summary_path": sp,
			"run_dir": run_dir,
			"model": model,
			"dataset": str(meta.get("dataset", "") or ""),
			"r_star": float(rbest),
			"default_layer": dl,
			"r_default": float(r_def) if r_def is not None else None,
			"rel_dev_default_pct": _rel_dev_pct(rbest, float(r_def)) if r_def is not None else None,
			"proposed_candidate_n": len(cand),
			"r_proposed": float(r_prop) if r_prop is not None else None,
			"rel_dev_proposed_pct": _rel_dev_pct(rbest, float(r_prop)) if r_prop is not None else None,
		}
		by_model[model_lc].append(row)

	body_lines: List[str] = []
	for model in arch_order:
		if model not in arch_tex:
			continue
		lab, _ck = arch_tex[model]
		items = by_model.get(model, [])
		rd = [float(x["rel_dev_default_pct"]) for x in items if x.get("rel_dev_default_pct") is not None]
		rp = [float(x["rel_dev_proposed_pct"]) for x in items if x.get("rel_dev_proposed_pct") is not None]
		m_d, s_d = _mean_std(rd)
		m_p, s_p = _mean_std(rp)
		disp_layer = _tex_tt(str(default_layer.get(model, "")))
		if not items:
			line = f"{lab} & \\texttt{{{disp_layer}}} & --- & --- \\\\"
		else:
			fd = "---" if not rd else f"{m_d:.2f}\\% $\\pm$ {s_d:.2f}\\%"
			fp = "---" if not rp else f"{m_p:.2f}\\% $\\pm$ {s_p:.2f}\\%"
			line = f"{lab} & \\texttt{{{disp_layer}}} & {fd} & {fp} \\\\"
		body_lines.append(line)

	full_lines = [r"\toprule", header, r"\midrule"] + body_lines + [r"\bottomrule"]
	_write_tabular_body_tex(str(update_tex).strip(), full_lines)


def main() -> None:
	ap = argparse.ArgumentParser(description="Aggregate embedding retrieval summaries for layer-selection table.")
	ap.add_argument("--roots", type=str, nargs="+", required=True)
	ap.add_argument(
		"--exclude_models",
		type=str,
		default="",
		help="Comma-separated meta.model substrings to skip when collecting layer markers; empty includes all compatible runs.",
	)
	ap.add_argument("--defaults_json", type=str, default="", help="Optional JSON: {model: default_layer_module_name}")
	ap.add_argument("--top_corr_layers", type=int, default=8)
	ap.add_argument("--corr_min_abs_rho", type=float, default=0.0)
	ap.add_argument("--corr_max_p", type=float, default=1.0)
	ap.add_argument("--out_json", type=str, default="")
	ap.add_argument(
		"--update_tex",
		type=str,
		default="",
		help="If set: overwrite path with booktabs fragment (toprule/header/midrule/rows/bottomrule), no outer table.",
	)
	args = ap.parse_args()

	_header, arch_order, arch_tex, default_layer_base = _layer_selection_config()
	excl = {x.strip().lower() for x in str(args.exclude_models).split(",") if x.strip()}
	default_layer = dict(default_layer_base)
	if str(args.defaults_json).strip():
		ov = _load_json(str(args.defaults_json).strip())
		if isinstance(ov, dict):
			default_layer.update({str(k).strip().lower(): str(v) for k, v in ov.items()})

	by_model: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

	for sp in _discover_summaries(args.roots):
		try:
			summary = _load_json(sp)
			run_dir = str(summary.get("run_dir") or "").strip()
			if not run_dir or not os.path.isdir(run_dir):
				run_dir = os.path.abspath(os.path.join(os.path.dirname(sp), ".."))
			meta_path = os.path.join(run_dir, "meta.json")
			if not os.path.isfile(meta_path):
				continue
			meta = _load_json(meta_path)
			model = str(meta.get("model", "") or "").strip()
			if not model:
				model = str((meta.get("args") or {}).get("model", "") or "").strip()
			model_lc = _canonical_model_key(model)
			if not model_lc or model_lc in excl:
				continue
			by_layer = _ratio_by_layer(summary, sp)
			rbest = _r_star(by_layer)
			if rbest is None:
				continue
			dl = default_layer.get(model_lc)
			r_def = by_layer.get(str(dl)) if dl else None
			cand = _proposed_candidate_layers(
				run_dir,
				top_n_layers=int(args.top_corr_layers),
				min_abs_rho=float(args.corr_min_abs_rho),
				max_p=float(args.corr_max_p),
			)
			r_props = [by_layer[str(x)] for x in cand if str(x) in by_layer]
			r_prop = max(r_props) if r_props else None
			row = {
				"summary_path": sp,
				"run_dir": run_dir,
				"model": model,
				"dataset": str(meta.get("dataset", "") or ""),
				"r_star": float(rbest),
				"default_layer": dl,
				"r_default": float(r_def) if r_def is not None else None,
				"rel_dev_default_pct": _rel_dev_pct(rbest, float(r_def)) if r_def is not None else None,
				"proposed_candidate_n": len(cand),
				"r_proposed": float(r_prop) if r_prop is not None else None,
				"rel_dev_proposed_pct": _rel_dev_pct(rbest, float(r_prop)) if r_prop is not None else None,
			}
			by_model[model_lc].append(row)
		except Exception as e:
			print(f"[fail] {sp}: {e}", file=sys.stderr)

	agg: Dict[str, Any] = {"models": {}, "rows": []}
	body_lines: List[str] = []

	for model in arch_order:
		if model not in arch_tex:
			continue
		lab, _ck = arch_tex[model]
		items = by_model.get(model, [])
		rd = [float(x["rel_dev_default_pct"]) for x in items if x.get("rel_dev_default_pct") is not None]
		rp = [float(x["rel_dev_proposed_pct"]) for x in items if x.get("rel_dev_proposed_pct") is not None]
		m_d, s_d = _mean_std(rd)
		m_p, s_p = _mean_std(rp)
		disp_layer = _tex_tt(str(default_layer.get(model, "")))
		agg["models"][model] = {
			"n": len(items),
			"rel_dev_default_pct_mean": m_d,
			"rel_dev_default_pct_std": s_d,
			"rel_dev_proposed_pct_mean": m_p,
			"rel_dev_proposed_pct_std": s_p,
		}
		if not items:
			line = f"{lab} & \\texttt{{{disp_layer}}} & --- & --- \\\\"
		else:
			if not rd:
				fd = "---"
			else:
				fd = f"{m_d:.2f}\\% $\\pm$ {s_d:.2f}\\%"
			if not rp:
				fp = "---"
			else:
				fp = f"{m_p:.2f}\\% $\\pm$ {s_p:.2f}\\%"
			line = f"{lab} & \\texttt{{{disp_layer}}} & {fd} & {fp} \\\\"
		body_lines.append(line)
		agg["rows"].append(line)

	if str(args.out_json).strip():
		p = os.path.abspath(str(args.out_json).strip())
		os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
		with open(p, "w", encoding="utf-8") as wf:
			json.dump(agg, wf, indent=2, ensure_ascii=False)
		print("[ok] wrote", p, file=sys.stderr)

	if str(args.update_tex).strip():
		write_layer_selection_summary_tex(
			roots=list(args.roots),
			update_tex=str(args.update_tex).strip(),
			exclude_models=str(args.exclude_models),
			defaults_json=str(args.defaults_json),
			top_corr_layers=int(args.top_corr_layers),
			corr_min_abs_rho=float(args.corr_min_abs_rho),
			corr_max_p=float(args.corr_max_p),
		)
		print("[ok] updated", args.update_tex, file=sys.stderr)

	for ln in body_lines:
		print(ln)


if __name__ == "__main__":
	main()
