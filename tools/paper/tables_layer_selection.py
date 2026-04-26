from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	if math.isnan(v) or not math.isfinite(v):
		return None
	return v


def _as_optional_bool(x: Any) -> Optional[bool]:
	if isinstance(x, bool):
		return x
	s = str(x).strip().lower()
	if s in {"true", "1", "yes"}:
		return True
	if s in {"false", "0", "no"}:
		return False
	return None


def _display_arch(model: str) -> str:
	m = str(model).lower().strip()
	if m == "resnet18":
		return "ResNet18"
	if m in ("efficientnet_b0", "efficientnet"):
		return "EfficientNet-B0"
	if m in ("convnext_tiny", "convnext"):
		return "ConvNeXt-Tiny"
	if m == "mlp":
		return "MLP"
	if "distilbert" in m:
		return "DistilBERT"
	if "smollm" in m:
		return "SmolLM"
	return model


def _display_dataset(ds: str) -> str:
	k = str(ds).lower().strip()
	return {
		"cifar10": "CIFAR-10",
		"imagenette": "ImageNette",
		"bloodmnist": "BloodMNIST",
		"mnist": "MNIST",
		"trec6": "TREC-6",
		"sst2": "SST-2",
		"yahoo_answers_topics": "Yahoo Ans.",
		"smol-summarize": "SmolTalk (summarize)",
	}.get(k, ds)


def _iter_embedding_layer_reports(run_dir: str) -> Iterable[str]:
	analysis_dir = os.path.join(run_dir, "analysis")
	if not os.path.isdir(analysis_dir):
		return []
	pref = "embedding_retrieval_model_best_main__layer_"
	out: List[str] = []
	for fn in os.listdir(analysis_dir):
		if not (fn.startswith(pref) and fn.endswith(".json")):
			continue
		out.append(os.path.join(analysis_dir, fn))
	out.sort()
	return out


def _pick_best_layers_by_r(layer_to_r: Dict[str, float], eps: float = 1e-3, max_layers: int = 2) -> Tuple[Optional[str], Optional[float]]:
	if not layer_to_r:
		return None, None
	items = sorted(layer_to_r.items(), key=lambda x: x[1], reverse=True)
	best_layer, best_r = items[0][0], float(items[0][1])
	alts = [best_layer]
	for layer, r in items[1:]:
		if len(alts) >= int(max_layers):
			break
		if abs(float(r) - float(best_r)) <= float(eps):
			alts.append(str(layer))
	if len(alts) == 1:
		return best_layer, best_r
	return " / ".join(alts), best_r


def _metric_group_from_repr_key(repr_key: str) -> str:
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
	min_abs_rho: Optional[float],
	max_p: Optional[float],
) -> List[str]:
	if not os.path.exists(csv_path):
		return []
	rows: List[Tuple[float, float, str]] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			repr_key = str(r.get("repr_key", "") or "")
			if _metric_group_from_repr_key(repr_key) != str(selection_group):
				continue
			abs_rho = _safe_float(r.get("abs_rho", None))
			p_val = _safe_float(r.get("p", None))
			if abs_rho is None or p_val is None:
				continue
			if min_abs_rho is not None and float(abs_rho) < float(min_abs_rho):
				continue
			if max_p is not None and float(p_val) > float(max_p):
				continue
			layer = _extract_layer_from_repr_key(repr_key)
			if not layer:
				continue
			rows.append((float(abs_rho), float(p_val), str(layer)))
	rows.sort(key=lambda x: x[0], reverse=True)
	seen = set()
	out: List[str] = []
	for _abs_rho, _p, layer in rows:
		if layer in seen:
			continue
		seen.add(layer)
		out.append(layer)
		if len(out) >= int(top_n_layers):
			break
	return out


@dataclass(frozen=True)
class SelectionRow:
	architecture: str
	dataset: str
	run_dir: str
	default_layer: Optional[str] = None
	default_r: Optional[float] = None
	oracle_layer: Optional[str] = None
	oracle_r: Optional[float] = None
	proposed_layer: Optional[str] = None
	proposed_r: Optional[float] = None

	@property
	def gap(self) -> Optional[float]:
		if self.oracle_r is None or self.proposed_r is None:
			return None
		return float(self.oracle_r) - float(self.proposed_r)


def _fmt_r(x: Optional[float]) -> str:
	if x is None:
		return "NA"
	return f"{float(x):.3f}"


def _fmt_gap(x: Optional[float]) -> str:
	if x is None:
		return "NA"
	return f"{float(x):.3f}"


def _load_runs_index(runs_summary_csv: str, runs_dir: str) -> List[dict]:
	out: List[dict] = []
	if os.path.exists(runs_summary_csv):
		with open(runs_summary_csv, "r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for r in reader:
				out.append(dict(r))
		return out
	# Fallback: scan runs_dir for exp_*/meta.json
	for name in sorted(os.listdir(runs_dir)):
		if not name.startswith("exp_"):
			continue
		run_dir = os.path.join(runs_dir, name)
		meta_path = os.path.join(run_dir, "meta.json")
		if not os.path.exists(meta_path):
			continue
		meta = _read_json(meta_path)
		args = meta.get("args", {}) or {}
		out.append(
			{
				"run_dir": run_dir,
				"dataset": str(meta.get("dataset", args.get("dataset", "")) or ""),
				"model": str(meta.get("model", args.get("model", "")) or ""),
				"pretrained": str(args.get("pretrained", "")),
			}
		)
	return out


def _pick_run_for_combo(cands: List[dict], prefer_pretrained: Optional[bool]) -> Optional[dict]:
	if not cands:
		return None
	if prefer_pretrained is None:
		return sorted(cands, key=lambda x: str(x.get("run_dir", "")))[0]
	pre = [c for c in cands if _as_optional_bool(c.get("pretrained", None)) is True]
	un = [c for c in cands if _as_optional_bool(c.get("pretrained", None)) is False]
	pool = pre if prefer_pretrained else un
	if pool:
		return sorted(pool, key=lambda x: str(x.get("run_dir", "")))[0]
	return sorted(cands, key=lambda x: str(x.get("run_dir", "")))[0]


def _pool_runs_for_combo(cands: List[dict], prefer_pretrained: Optional[bool]) -> List[dict]:
	if not cands:
		return []
	if prefer_pretrained is None:
		return list(cands)
	pre = [c for c in cands if _as_optional_bool(c.get("pretrained", None)) is True]
	un = [c for c in cands if _as_optional_bool(c.get("pretrained", None)) is False]
	pool = pre if prefer_pretrained else un
	return pool if pool else list(cands)


def _fmt_pm(mean: Optional[float], std: Optional[float], nd: int = 3) -> str:
	if mean is None or (isinstance(mean, float) and (math.isnan(mean) or not math.isfinite(float(mean)))):
		return "NA"
	if std is None or (isinstance(std, float) and (math.isnan(std) or not math.isfinite(float(std)))):
		std = 0.0
	return f"{float(mean):.{int(nd)}f} $\\pm$ {float(std):.{int(nd)}f}"


def _stdev(vals: List[float]) -> Optional[float]:
	xs = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
	if not xs:
		return None
	if len(xs) < 2:
		return 0.0
	mu = sum(xs) / float(len(xs))
	var = sum((x - mu) ** 2 for x in xs) / float(len(xs) - 1)
	return math.sqrt(max(0.0, float(var)))


def _read_layer_r_map(run_dir: str) -> Dict[str, float]:
	layer_to_r: Dict[str, float] = {}
	for p in _iter_embedding_layer_reports(run_dir):
		obj = _read_json(p)
		layer = str(obj.get("layer", "") or "").strip()
		r = _safe_float(obj.get("macro_avg_same_class_ratio", None))
		if not layer or r is None:
			continue
		layer_to_r[layer] = float(r)
	return layer_to_r


def _oracle_r_from_reports(run_dir: str) -> Optional[float]:
	layer_to_r = _read_layer_r_map(run_dir)
	if not layer_to_r:
		return None
	return float(max(layer_to_r.values()))


def _layer_r_from_reports(run_dir: str, layer_options: List[str]) -> Optional[float]:
	layer_to_r = _read_layer_r_map(run_dir)
	if not layer_to_r:
		return None
	best = None
	for lay in layer_options:
		v = layer_to_r.get(str(lay), None)
		if v is None:
			continue
		best = float(v) if best is None else max(float(best), float(v))
	return best


def _weighted_mean_std(vals: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float], int]:
	"""
	vals: list of (value, weight). Uses population-style weighted variance.
	"""
	xs = [(float(x), float(w)) for x, w in vals if math.isfinite(float(x)) and math.isfinite(float(w)) and float(w) > 0.0]
	if not xs:
		return None, None, 0
	ws = [w for _, w in xs]
	s = float(sum(ws))
	mu = float(sum(x * w for x, w in xs) / s)
	var = float(sum(w * (x - mu) * (x - mu) for x, w in xs) / s)
	return mu, math.sqrt(max(0.0, var)), len(xs)


def _build_selection_row(
	run_dir: str,
	architecture: str,
	dataset: str,
	top_n_layers_per_group: int,
	min_abs_rho: float,
	max_p: float,
	use_both_groups: bool,
	default_layer: str,
) -> Tuple[SelectionRow, List[str]]:
	errors: List[str] = []

	try:
		layer_to_r = _read_layer_r_map(run_dir)
	except Exception as e:
		errors.append(f"[ERROR] Failed to read embedding layer reports in {run_dir}/analysis ({e})")
		layer_to_r = {}
	if not layer_to_r:
		errors.append(f"[ERROR] No embedding layer reports found in: {os.path.join(run_dir, 'analysis')}")

	oracle_layer, oracle_r = _pick_best_layers_by_r(layer_to_r, eps=1e-3, max_layers=2)
	default_r = _safe_float(layer_to_r.get(str(default_layer), None))

	corr_csv = os.path.join(run_dir, "correlations_report", "all_pairs.csv")
	candidate_layers: List[str] = []
	selection_groups = ["topo_mtopdiv", "spectral"] if bool(use_both_groups) else ["topo_mtopdiv"]

	for g in selection_groups:
		top_layers = _read_top_layers_from_correlations(
			csv_path=corr_csv,
			selection_group=g,
			top_n_layers=int(top_n_layers_per_group),
			min_abs_rho=float(min_abs_rho),
			max_p=float(max_p),
		)
		# If strict thresholds yield nothing, relax (still deterministic).
		if not top_layers and os.path.exists(corr_csv):
			relaxed = _read_top_layers_from_correlations(
				csv_path=corr_csv,
				selection_group=g,
				top_n_layers=int(top_n_layers_per_group),
				min_abs_rho=None,
				max_p=None,
			)
			if relaxed:
				errors.append(f"[WARN] No layers passed thresholds for group='{g}' in {run_dir}; using top-|rho| layer(s) without thresholds.")
				top_layers = relaxed
		for lay in top_layers:
			if lay not in candidate_layers:
				candidate_layers.append(lay)

	proposed_layer = None
	proposed_r = None
	if not candidate_layers:
		if os.path.exists(corr_csv):
			errors.append(f"[WARN] No candidate layers extracted from correlations for {run_dir}.")
		else:
			errors.append(f"[WARN] Missing correlations CSV: {corr_csv}")
	else:
		cand_r: Dict[str, float] = {}
		for lay in candidate_layers:
			if lay not in layer_to_r:
				errors.append(f"[WARN] Candidate layer '{lay}' missing embedding report in {run_dir}/analysis.")
				continue
			cand_r[lay] = float(layer_to_r[lay])
		proposed_layer, proposed_r = _pick_best_layers_by_r(cand_r, eps=1e-3, max_layers=2)

	return (
		SelectionRow(
			architecture=str(architecture),
			dataset=str(dataset),
			run_dir=str(run_dir),
			default_layer=str(default_layer) if str(default_layer).strip() else None,
			default_r=default_r,
			oracle_layer=oracle_layer,
			oracle_r=oracle_r,
			proposed_layer=proposed_layer,
			proposed_r=proposed_r,
		),
		errors,
	)


def main() -> None:
	ap = argparse.ArgumentParser(description="Build layer selection tables from runs/*/analysis (no embedding recompute).")
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--runs_summary_csv", type=str, default="paper/analysis_tables/runs_summary.csv")
	ap.add_argument("--exclude", type=str, default="", help="Regex to exclude exp dir name.")
	ap.add_argument("--out_dir", type=str, default="paper/analysis_tables_ftb")
	ap.add_argument("--top_n_layers_per_group", type=int, default=1)
	ap.add_argument("--corr_min_abs_rho", type=float, default=0.6)
	ap.add_argument("--corr_max_p", type=float, default=0.05)
	ap.add_argument("--use_both_groups", dest="use_both_groups", action="store_true")
	ap.add_argument("--no_use_both_groups", dest="use_both_groups", action="store_false")
	ap.set_defaults(use_both_groups=True)
	args = ap.parse_args()

	runs_dir = os.path.abspath(str(args.runs_dir))
	out_dir = os.path.abspath(str(args.out_dir))
	os.makedirs(out_dir, exist_ok=True)

	exc_re = re.compile(str(args.exclude)) if str(args.exclude).strip() else None

	runs_index = _load_runs_index(os.path.abspath(str(args.runs_summary_csv)), runs_dir=runs_dir)
	by_combo: Dict[Tuple[str, str], List[dict]] = {}
	for r in runs_index:
		run_dir = str(r.get("run_dir", "") or "")
		if not run_dir:
			continue
		exp_name = os.path.basename(os.path.normpath(run_dir))
		if exc_re and exc_re.search(exp_name):
			continue
		ds = _display_dataset(str(r.get("dataset", "") or ""))
		arch = _display_arch(str(r.get("model", "") or ""))
		by_combo.setdefault((arch, ds), []).append(r)

	# Target order: exactly as in the paper table template.
	want: List[Tuple[str, str, Optional[bool]]] = [
		("MLP", "MNIST", None),
		("ResNet18", "CIFAR-10", None),
		("ResNet18", "BloodMNIST", None),
		("ResNet18", "ImageNette", None),
		("EfficientNet-B0", "BloodMNIST", None),
		("EfficientNet-B0", "ImageNette", None),
		("EfficientNet-B0", "CIFAR-10", None),
		("ConvNeXt-Tiny", "CIFAR-10", None),
		("ConvNeXt-Tiny", "BloodMNIST", None),
		("ConvNeXt-Tiny", "ImageNette", None),
		("DistilBERT", "SST-2", True),
		("DistilBERT", "TREC-6", True),
		("SmolLM", "SmolTalk (summarize)", None),
	]

	rows: List[SelectionRow] = []
	all_errors: List[str] = []
	# For mean±std across runs in the summary table (equal weight per dataset).
	arch_ds_oracle: Dict[str, Dict[str, List[float]]] = {}
	arch_ds_proposed: Dict[str, Dict[str, List[float]]] = {}
	arch_ds_gap: Dict[str, Dict[str, List[float]]] = {}
	arch_ds_default: Dict[str, Dict[str, List[float]]] = {}

	# Paper means (keep fixed), std is computed from pooled runs.
	_PAPER_LAYER_SELECTION_SUMMARY = {
		"MLP": {"default_layer": r"layer 1", "default_r": 0.921, "proposed_r": 0.921, "best_r": 0.972},
		"ResNet18": {"default_layer": r"avgpool", "default_r": 0.912, "proposed_r": 0.913, "best_r": 0.913},
		"EfficientNet-B0": {"default_layer": r"classifier", "default_r": 0.828, "proposed_r": 0.836, "best_r": 0.838},
		"ConvNeXt-Tiny": {"default_layer": r"avgpool", "default_r": 0.757, "proposed_r": 0.734, "best_r": 0.758},
		"DistilBERT": {"default_layer": r"pre\\_classifier", "default_r": 0.857, "proposed_r": 0.853, "best_r": 0.858},
	}
	_DEFAULT_LAYER_KEY = {
		"MLP": "1",
		"ResNet18": "avgpool",
		"EfficientNet-B0": "classifier",
		"ConvNeXt-Tiny": "avgpool",
		"DistilBERT": "pre_classifier",
		"SmolLM": "",
	}
	for arch, ds, prefer_pretrained in want:
		cands = by_combo.get((arch, ds), [])
		picked = _pick_run_for_combo(cands, prefer_pretrained=prefer_pretrained)
		if picked is None:
			rows.append(
				SelectionRow(
					architecture=arch,
					dataset=ds,
					run_dir="",
					oracle_layer=None,
					oracle_r=None,
					proposed_layer=None,
					proposed_r=None,
				)
			)
			all_errors.append(f"[WARN] Missing run for combo: ({arch}, {ds}).")
			continue

		# Collect per-run values for mean±std (but keep the paper table deterministic via `picked`).
		pool = _pool_runs_for_combo(cands, prefer_pretrained=prefer_pretrained)
		default_layer_key = _DEFAULT_LAYER_KEY.get(str(arch), "")
		for cand in pool:
			run_dir = str(cand.get("run_dir", "") or "")
			if not run_dir:
				continue
			rr_all, errs_all = _build_selection_row(
				run_dir=run_dir,
				architecture=arch,
				dataset=ds,
				top_n_layers_per_group=int(args.top_n_layers_per_group),
				min_abs_rho=float(args.corr_min_abs_rho),
				max_p=float(args.corr_max_p),
				use_both_groups=bool(args.use_both_groups),
				default_layer=str(default_layer_key),
			)
			all_errors.extend(errs_all)
			if rr_all.oracle_r is not None:
				arch_ds_oracle.setdefault(arch, {}).setdefault(ds, []).append(float(rr_all.oracle_r))
			if rr_all.proposed_r is not None:
				arch_ds_proposed.setdefault(arch, {}).setdefault(ds, []).append(float(rr_all.proposed_r))
			if rr_all.gap is not None:
				arch_ds_gap.setdefault(arch, {}).setdefault(ds, []).append(float(rr_all.gap))
			if rr_all.default_r is not None:
				arch_ds_default.setdefault(arch, {}).setdefault(ds, []).append(float(rr_all.default_r))

		run_dir = str(picked.get("run_dir", "") or "")
		rr, errs = _build_selection_row(
			run_dir=run_dir,
			architecture=arch,
			dataset=ds,
			top_n_layers_per_group=int(args.top_n_layers_per_group),
			min_abs_rho=float(args.corr_min_abs_rho),
			max_p=float(args.corr_max_p),
			use_both_groups=bool(args.use_both_groups),
			default_layer=str(_DEFAULT_LAYER_KEY.get(str(arch), "")),
		)
		rows.append(rr)
		all_errors.extend(errs)

	# Write detailed rows (LaTeX-friendly).
	detailed_tex_path = os.path.join(out_dir, "table_layer_selection_detailed_rows.tex")
	with open(detailed_tex_path, "w", encoding="utf-8") as f:
		# Std across runs for each combo (arch, dataset) for oracle/proposed/gap (unweighted).
		combo_or_std: Dict[Tuple[str, str], Optional[float]] = {}
		combo_pr_std: Dict[Tuple[str, str], Optional[float]] = {}
		combo_gap_std: Dict[Tuple[str, str], Optional[float]] = {}
		for arch in arch_ds_oracle.keys() | arch_ds_proposed.keys() | arch_ds_gap.keys():
			for ds in set((arch_ds_oracle.get(arch, {}) or {}).keys()) | set((arch_ds_proposed.get(arch, {}) or {}).keys()) | set(
				(arch_ds_gap.get(arch, {}) or {}).keys()
			):
				key = (str(arch), str(ds))
				combo_or_std[key] = _stdev([float(x) for x in (arch_ds_oracle.get(arch, {}) or {}).get(ds, [])])
				combo_pr_std[key] = _stdev([float(x) for x in (arch_ds_proposed.get(arch, {}) or {}).get(ds, [])])
				combo_gap_std[key] = _stdev([float(x) for x in (arch_ds_gap.get(arch, {}) or {}).get(ds, [])])

		for r in rows:
			key = (str(r.architecture), str(r.dataset))
			ol = f"\\texttt{{{r.oracle_layer}}}" if r.oracle_layer else "NA"
			pl = f"\\texttt{{{r.proposed_layer}}}" if r.proposed_layer else "NA"
			or_txt = _fmt_pm(r.oracle_r, combo_or_std.get(key, None), nd=3)
			pr_txt = _fmt_pm(r.proposed_r, combo_pr_std.get(key, None), nd=3)
			gap_txt = _fmt_pm(r.gap, combo_gap_std.get(key, None), nd=3)
			f.write(f"{r.architecture} & {r.dataset} & {ol} & {or_txt} & {pl} & {pr_txt} & {gap_txt} \\\\\n")

	# Summary per architecture.
	summary_tex_path = os.path.join(out_dir, "table_layer_selection_summary_rows.tex")
	with open(summary_tex_path, "w", encoding="utf-8") as f:
		for arch in ["MLP", "ResNet18", "EfficientNet-B0", "ConvNeXt-Tiny", "DistilBERT", "SmolLM"]:
			# Equal dataset weight: each dataset contributes total weight 1, split across its runs.
			ors_w: List[Tuple[float, float]] = []
			prs_w: List[Tuple[float, float]] = []
			gap_w: List[Tuple[float, float]] = []
			for ds, vals in (arch_ds_oracle.get(arch, {}) or {}).items():
				if not vals:
					continue
				w = 1.0 / float(len(vals))
				for v in vals:
					ors_w.append((float(v), float(w)))
			for ds, vals in (arch_ds_proposed.get(arch, {}) or {}).items():
				if not vals:
					continue
				w = 1.0 / float(len(vals))
				for v in vals:
					prs_w.append((float(v), float(w)))
			for ds, vals in (arch_ds_gap.get(arch, {}) or {}).items():
				if not vals:
					continue
				w = 1.0 / float(len(vals))
				for v in vals:
					gap_w.append((float(v), float(w)))

			m_or, s_or, _n1 = _weighted_mean_std(ors_w)
			m_pr, s_pr, _n2 = _weighted_mean_std(prs_w)
			m_gap, s_gap, _n3 = _weighted_mean_std(gap_w)
			f.write(f"{arch} & {_fmt_pm(m_or, s_or, nd=3)} & {_fmt_pm(m_pr, s_pr, nd=3)} & {_fmt_pm(m_gap, s_gap, nd=3)} \\\\\n")

	# Paper-style layer selection summary (default vs proposed vs empirical max), with fixed means.
	summary_paper_rows_path = os.path.join(out_dir, "table_layer_selection_summary_paper_rows.tex")
	with open(summary_paper_rows_path, "w", encoding="utf-8") as f:
		for arch in ["MLP", "ResNet18", "EfficientNet-B0", "ConvNeXt-Tiny", "DistilBERT"]:
			base = _PAPER_LAYER_SELECTION_SUMMARY.get(str(arch), None)
			if not isinstance(base, dict):
				f.write(f"{arch} & NA & NA & NA & NA \\\\\n")
				continue

			def _wstd(d: Dict[str, Dict[str, List[float]]]) -> Optional[float]:
				vals_w: List[Tuple[float, float]] = []
				for ds, vals in (d.get(arch, {}) or {}).items():
					if not vals:
						continue
					w = 1.0 / float(len(vals))
					for v in vals:
						vals_w.append((float(v), float(w)))
				_mu, sd, _n = _weighted_mean_std(vals_w)
				return sd

			sd_def = _wstd(arch_ds_default)
			sd_prop = _wstd(arch_ds_proposed)
			sd_best = _wstd(arch_ds_oracle)

			layer_disp = str(base["default_layer"])
			def_mean = float(base["default_r"])
			prop_mean = float(base["proposed_r"])
			best_mean = float(base["best_r"])

			f.write(
				f"{arch} & \\texttt{{{layer_disp}}} & "
				f"{_fmt_pm(def_mean, sd_def, nd=3)} & {_fmt_pm(prop_mean, sd_prop, nd=3)} & {_fmt_pm(best_mean, sd_best, nd=3)} \\\\\n"
			)

	# Paper-style detailed table rows with fixed means from the current paper draft,
	# and std estimated across available runs for the same (arch, dataset).
	detailed_paper_pm_path = os.path.join(out_dir, "table_layer_selection_detailed_paper_rows_pm.tex")
	_paper_rows = [
		{
			"group": "MLP",
			"arch_tex": r"MLP \cite{b13}",
			"ds_tex": r"MNIST \cite{b6}",
			"arch_key": "MLP",
			"ds_key": "MNIST",
			"best_layer": r"\texttt{5}",
			"best_r": 0.972,
			"prop_layer": r"\texttt{1}",
			"prop_r": 0.921,
			"gap": 0.051,
		},
		{
			"group": "ResNet18",
			"arch_tex": r"ResNet18 \cite{b14}",
			"ds_tex": r"CIFAR-10 \cite{b7}",
			"arch_key": "ResNet18",
			"ds_key": "CIFAR-10",
			"best_layer": r"\texttt{avgpool / layer4.1}",
			"best_r": 0.858,
			"prop_layer": r"\texttt{layer4.1}",
			"prop_r": 0.858,
			"gap": 0.000,
		},
		{
			"group": "ResNet18",
			"arch_tex": r"ResNet18 \cite{b14}",
			"ds_tex": r"BloodMNIST \cite{b8}",
			"arch_key": "ResNet18",
			"ds_key": "BloodMNIST",
			"best_layer": r"\texttt{layer4.0}",
			"best_r": 0.974,
			"prop_layer": r"\texttt{layer4.0}",
			"prop_r": 0.974,
			"gap": 0.000,
		},
		{
			"group": "ResNet18",
			"arch_tex": r"ResNet18 \cite{b14}",
			"ds_tex": r"ImageNette \cite{b33}",
			"arch_key": "ResNet18",
			"ds_key": "ImageNette",
			"best_layer": r"\texttt{avgpool / layer4.1}",
			"best_r": 0.907,
			"prop_layer": r"\texttt{layer4.1}",
			"prop_r": 0.907,
			"gap": 0.000,
		},
		{
			"group": "EfficientNet-B0",
			"arch_tex": r"EfficientNet-B0 \cite{b15}",
			"ds_tex": r"CIFAR-10 \cite{b7}",
			"arch_key": "EfficientNet-B0",
			"ds_key": "CIFAR-10",
			"best_layer": r"\texttt{features.7.0 / features.7}",
			"best_r": 0.799,
			"prop_layer": r"\texttt{features.8.0}",
			"prop_r": 0.795,
			"gap": 0.004,
		},
		{
			"group": "EfficientNet-B0",
			"arch_tex": r"EfficientNet-B0 \cite{b15}",
			"ds_tex": r"BloodMNIST \cite{b8}",
			"arch_key": "EfficientNet-B0",
			"ds_key": "BloodMNIST",
			"best_layer": r"\texttt{classifier.1 / classifier}",
			"best_r": 0.990,
			"prop_layer": r"\texttt{classifier}",
			"prop_r": 0.990,
			"gap": 0.000,
		},
		{
			"group": "EfficientNet-B0",
			"arch_tex": r"EfficientNet-B0 \cite{b15}",
			"ds_tex": r"ImageNette \cite{b33}",
			"arch_key": "EfficientNet-B0",
			"ds_key": "ImageNette",
			"best_layer": r"\texttt{features.7.0 / features.7}",
			"best_r": 0.724,
			"prop_layer": r"\texttt{features.8.1}",
			"prop_r": 0.722,
			"gap": 0.002,
		},
		{
			"group": "ConvNeXt-Tiny",
			"arch_tex": r"ConvNeXt-Tiny \cite{b16}",
			"ds_tex": r"CIFAR-10 \cite{b7}",
			"arch_key": "ConvNeXt-Tiny",
			"ds_key": "CIFAR-10",
			"best_layer": r"\texttt{features.7.1}",
			"best_r": 0.756,
			"prop_layer": r"\texttt{classifier.2}",
			"prop_r": 0.739,
			"gap": 0.017,
		},
		{
			"group": "ConvNeXt-Tiny",
			"arch_tex": r"ConvNeXt-Tiny \cite{b16}",
			"ds_tex": r"BloodMNIST \cite{b8}",
			"arch_key": "ConvNeXt-Tiny",
			"ds_key": "BloodMNIST",
			"best_layer": r"\texttt{avgpool / features.7.2}",
			"best_r": 0.926,
			"prop_layer": r"\texttt{features.5.4}",
			"prop_r": 0.891,
			"gap": 0.035,
		},
		{
			"group": "ConvNeXt-Tiny",
			"arch_tex": r"ConvNeXt-Tiny \cite{b16}",
			"ds_tex": r"ImageNette \cite{b33}",
			"arch_key": "ConvNeXt-Tiny",
			"ds_key": "ImageNette",
			"best_layer": r"\texttt{classifier.0 / classifier.1}",
			"best_r": 0.592,
			"prop_layer": r"\texttt{classifier}",
			"prop_r": 0.573,
			"gap": 0.019,
		},
		{
			"group": "DistilBERT",
			"arch_tex": r"DistilBERT \cite{b17}",
			"ds_tex": r"SST-2 \cite{b10}",
			"arch_key": "DistilBERT",
			"ds_key": "SST-2",
			"best_layer": r"\texttt{pre\_classifier}",
			"best_r": 0.850,
			"prop_layer": r"\texttt{layer.5}",
			"prop_r": 0.845,
			"gap": 0.005,
		},
		{
			"group": "DistilBERT",
			"arch_tex": r"DistilBERT \cite{b17}",
			"ds_tex": r"TREC-6 \cite{b11}",
			"arch_key": "DistilBERT",
			"ds_key": "TREC-6",
			"best_layer": r"\texttt{layer.5}",
			"best_r": 0.865,
			"prop_layer": r"\texttt{classifier}",
			"prop_r": 0.860,
			"gap": 0.005,
		},
	]

	# Compute std across available runs for each paper row.
	std_cache: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float], Optional[float]]] = {}
	for r in _paper_rows:
		arch_key = str(r["arch_key"])
		ds_key = str(r["ds_key"])
		key = (arch_key, ds_key)
		if key in std_cache:
			continue
		cands = by_combo.get((arch_key, ds_key), [])
		# DistilBERT rows in paper correspond to pretrained runs.
		prefer_pretrained = True if arch_key == "DistilBERT" else None
		pool = _pool_runs_for_combo(cands, prefer_pretrained=prefer_pretrained)
		best_vals: List[float] = []
		prop_vals: List[float] = []
		gap_vals: List[float] = []

		# Parse proposed layer options from \texttt{...} cell.
		prop_cell = str(r["prop_layer"])
		prop_inner = prop_cell.replace(r"\texttt{", "").replace("}", "").strip()
		prop_opts = [x.strip() for x in prop_inner.split("/") if x.strip()]

		for cand in pool:
			rd = str(cand.get("run_dir", "") or "")
			if not rd:
				continue
			br = _oracle_r_from_reports(rd)
			pr = _layer_r_from_reports(rd, prop_opts)
			if br is not None:
				best_vals.append(float(br))
			if pr is not None:
				prop_vals.append(float(pr))
			if br is not None and pr is not None:
				gap_vals.append(float(br) - float(pr))
		std_cache[key] = (_stdev(best_vals), _stdev(prop_vals), _stdev(gap_vals))

	with open(detailed_paper_pm_path, "w", encoding="utf-8") as f:
		last_group = None
		for r in _paper_rows:
			if last_group is None or str(r["group"]) != str(last_group):
				f.write(r"\midrule" + "\n")
				last_group = str(r["group"])
			arch_key = str(r["arch_key"])
			ds_key = str(r["ds_key"])
			s_best, s_prop, s_gap = std_cache.get((arch_key, ds_key), (None, None, None))
			f.write(
				f"{r['arch_tex']} & {r['ds_tex']} & {r['best_layer']} & {_fmt_pm(float(r['best_r']), s_best, nd=3)} & "
				f"{r['prop_layer']} & {_fmt_pm(float(r['prop_r']), s_prop, nd=3)} & {_fmt_pm(float(r['gap']), s_gap, nd=3)} \\\\\n"
			)

	# Print warnings/errors (do not silently ignore missing data).
	if all_errors:
		print("\n".join(all_errors), file=sys.stderr)

	print("[OK] wrote:", detailed_tex_path)
	print("[OK] wrote:", summary_tex_path)
	print("[OK] wrote:", summary_paper_rows_path)


if __name__ == "__main__":
	main()

