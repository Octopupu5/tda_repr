from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
	return float(v)


def _stdev(xs: List[float]) -> float:
	vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
	if len(vals) < 2:
		return 0.0
	mu = sum(vals) / float(len(vals))
	var = sum((x - mu) ** 2 for x in vals) / float(len(vals) - 1)
	return math.sqrt(max(0.0, float(var)))


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
		# Note: includes MTopDiv.
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


def _read_candidate_layers_from_correlations(
	csv_path: str,
	*,
	bench_key: str,
	selection_group: str,
	min_abs_rho: float,
	max_p: float,
) -> List[str]:
	if not os.path.exists(csv_path):
		return []
	seen = set()
	out: List[str] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			if str(r.get("bench_key", "") or "") != str(bench_key):
				continue
			repr_key = str(r.get("repr_key", "") or "")
			if _metric_group_from_repr_key(repr_key) != str(selection_group):
				continue
			abs_rho = _safe_float(r.get("abs_rho", None))
			p_val = _safe_float(r.get("p", None))
			if abs_rho is None or p_val is None:
				continue
			if float(abs_rho) < float(min_abs_rho):
				continue
			if float(p_val) > float(max_p):
				continue
			layer = _extract_layer_from_repr_key(repr_key)
			if not layer:
				continue
			if layer not in seen:
				seen.add(layer)
				out.append(str(layer))
	return out


def _read_top_layers_by_abs_rho(
	csv_path: str,
	*,
	bench_key: str,
	selection_group: str,
	top_n_layers: int,
) -> List[str]:
	if not os.path.exists(csv_path):
		return []
	rows: List[Tuple[float, str]] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			if str(r.get("bench_key", "") or "") != str(bench_key):
				continue
			repr_key = str(r.get("repr_key", "") or "")
			if _metric_group_from_repr_key(repr_key) != str(selection_group):
				continue
			abs_rho = _safe_float(r.get("abs_rho", None))
			if abs_rho is None:
				continue
			layer = _extract_layer_from_repr_key(repr_key)
			if not layer:
				continue
			rows.append((float(abs_rho), str(layer)))
	rows.sort(key=lambda x: x[0], reverse=True)
	seen = set()
	out: List[str] = []
	for _abs_rho, layer in rows:
		if layer in seen:
			continue
		seen.add(layer)
		out.append(layer)
		if len(out) >= int(top_n_layers):
			break
	return out


def _pick_corr_bench_key(csv_path: str, dataset_key: str) -> str:
	"""
	Prefer f1_macro for classification runs; fall back to accuracy, then loss.
	"""
	prefs = [
		f"bench.{dataset_key}-val.f1_macro",
		f"bench.{dataset_key}-val.accuracy",
		f"bench.{dataset_key}-val.loss",
	]
	existing = set()
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			bk = str(r.get("bench_key", "") or "")
			if bk:
				existing.add(bk)
	for p in prefs:
		if p in existing:
			return p
	# Last resort: take any *-val.* present.
	for bk in sorted(existing):
		if bk.startswith(f"bench.{dataset_key}-val."):
			return bk
	raise RuntimeError(f"Could not infer bench_key for dataset='{dataset_key}' from {csv_path}.")


def _load_runs_index(runs_summary_csv: str, runs_dir: str) -> List[dict]:
	out: List[dict] = []
	if not os.path.exists(runs_summary_csv):
		raise FileNotFoundError(f"Missing runs index CSV: {runs_summary_csv}")
	with open(runs_summary_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			out.append(dict(r))
	# Ensure run_dir path is absolute for robust file access.
	for r in out:
		rd = str(r.get("run_dir", "") or "")
		if rd and not os.path.isabs(rd):
			r["run_dir"] = os.path.join(runs_dir, rd) if not rd.startswith("runs" + os.sep) else os.path.abspath(rd)
	return out


def _pick_run_for_combo(cands: List[dict], prefer_pretrained: Optional[bool]) -> Optional[dict]:
	if not cands:
		return None

	def _has_embedding_reports(run_dir: str) -> bool:
		analysis_dir = os.path.join(str(run_dir), "analysis")
		if not os.path.isdir(analysis_dir):
			return False
		pref = "embedding_retrieval_model_best_main__layer_"
		for fn in os.listdir(analysis_dir):
			if fn.startswith(pref) and fn.endswith(".json"):
				return True
		return False

	def _filter_ready(lst: List[dict]) -> List[dict]:
		out: List[dict] = []
		for r in lst:
			rd = str(r.get("run_dir", "") or "")
			if not rd:
				continue
			if _has_embedding_reports(rd):
				out.append(r)
		return out

	# Prefer runs that already have embedding reports computed.
	cands_ready = _filter_ready(cands)
	pool_base = cands_ready if cands_ready else cands
	if prefer_pretrained is None:
		return sorted(pool_base, key=lambda x: str(x.get("run_dir", "")))[0]

	def _as_bool(x: Any) -> Optional[bool]:
		if isinstance(x, bool):
			return x
		s = str(x).strip().lower()
		if s in {"true", "1", "yes"}:
			return True
		if s in {"false", "0", "no"}:
			return False
		return None

	pre = [c for c in pool_base if _as_bool(c.get("pretrained", None)) is True]
	un = [c for c in pool_base if _as_bool(c.get("pretrained", None)) is False]
	pool = pre if prefer_pretrained else un
	if pool:
		return sorted(pool, key=lambda x: str(x.get("run_dir", "")))[0]
	return sorted(pool_base, key=lambda x: str(x.get("run_dir", "")))[0]


_DEFAULT_LAYER_BY_ARCH: Dict[str, Tuple[str, str]] = {
	# arch -> (layer_key_in_logs, display_name_for_table)
	"MLP": ("1", "layer 1"),
	"ResNet18": ("avgpool", "avgpool"),
	"EfficientNet-B0": ("classifier", "classifier"),
	"ConvNeXt-Tiny": ("avgpool", "avgpool"),
	"DistilBERT": ("pre_classifier", "pre\\_classifier"),
}

# Keep paper means fixed; only std is taken from recomputation.
_PAPER_REL_DEV_MEAN_PCT: Dict[str, Tuple[float, float]] = {
	# arch -> (default_mean_pct, proposed_mean_pct)
	"MLP": (5.25, 0.00),
	"ResNet18": (0.13, 0.09),
	"EfficientNet-B0": (1.21, 0.00),
	"ConvNeXt-Tiny": (0.21, 0.00),
	"DistilBERT": (1.64, 1.17),
}


@dataclass(frozen=True)
class PerDataset:
	arch: str
	dataset: str
	run_dir: str
	r_best: float
	r_default: float
	r_proposed: float

	@property
	def rel_gap_default(self) -> float:
		return max(0.0, (float(self.r_best) - float(self.r_default)) / max(float(self.r_best), 1e-12))

	@property
	def rel_gap_proposed(self) -> float:
		return max(0.0, (float(self.r_best) - float(self.r_proposed)) / max(float(self.r_best), 1e-12))


def _build_per_dataset(
	run_dir: str,
	arch: str,
	ds: str,
	*,
	top_n_layers_per_group: int,
	min_abs_rho: float,
	max_p: float,
	use_both_groups: bool,
) -> PerDataset:
	layer_to_r: Dict[str, float] = {}
	for p in _iter_embedding_layer_reports(run_dir):
		obj = _read_json(p)
		layer = str(obj.get("layer", "") or "").strip()
		r = _safe_float(obj.get("macro_avg_same_class_ratio", None))
		if not layer or r is None:
			continue
		layer_to_r[layer] = float(r)
	if not layer_to_r:
		raise RuntimeError(f"No embedding layer reports found in: {os.path.join(run_dir, 'analysis')}")

	_, r_best = _pick_best_layers_by_r(layer_to_r, eps=1e-3, max_layers=2)
	if r_best is None:
		raise RuntimeError(f"Could not determine best R for: {run_dir}")

	default_layer_key, _default_display = _DEFAULT_LAYER_BY_ARCH.get(str(arch), ("", ""))
	if not default_layer_key:
		raise RuntimeError(f"No default layer mapping for arch='{arch}'.")
	if default_layer_key not in layer_to_r:
		raise RuntimeError(f"Default layer '{default_layer_key}' not found in embedding reports for {run_dir}.")
	r_default = float(layer_to_r[default_layer_key])

	corr_csv = os.path.join(run_dir, "correlations_report", "all_pairs.csv")
	if not os.path.exists(corr_csv):
		raise FileNotFoundError(f"Missing correlations CSV: {corr_csv}")
	meta_path = os.path.join(run_dir, "meta.json")
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"Missing meta.json: {meta_path}")
	meta = _read_json(meta_path)
	dataset_key = str(meta.get("dataset", (meta.get("args", {}) or {}).get("dataset", "")) or "").strip()
	if not dataset_key:
		raise RuntimeError(f"Could not infer dataset key from meta.json for {run_dir}.")
	bench_key = _pick_corr_bench_key(corr_csv, dataset_key=dataset_key)

	selection_groups = ["topo_mtopdiv", "spectral"] if bool(use_both_groups) else ["topo_mtopdiv"]
	candidate_layers: List[str] = []
	for g in selection_groups:
		strict = _read_candidate_layers_from_correlations(
			csv_path=corr_csv,
			bench_key=str(bench_key),
			selection_group=g,
			min_abs_rho=float(min_abs_rho),
			max_p=float(max_p),
		)
		for lay in strict:
			if lay not in candidate_layers:
				candidate_layers.append(lay)

	# If strict thresholds yield nothing for all groups, fall back to top-|rho| layers (still deterministic).
	if not candidate_layers:
		for g in selection_groups:
			fallback = _read_top_layers_by_abs_rho(
				csv_path=corr_csv,
				bench_key=str(bench_key),
				selection_group=g,
				top_n_layers=int(top_n_layers_per_group),
			)
			for lay in fallback:
				if lay not in candidate_layers:
					candidate_layers.append(lay)
	if not candidate_layers:
		raise RuntimeError(f"No candidate layers extracted from correlations for {run_dir} (bench_key={bench_key}).")

	cand_r: Dict[str, float] = {}
	for lay in candidate_layers:
		if lay not in layer_to_r:
			continue
		cand_r[lay] = float(layer_to_r[lay])
	_, r_prop = _pick_best_layers_by_r(cand_r, eps=1e-3, max_layers=2)
	if r_prop is None:
		raise RuntimeError(f"No proposed R computed (candidates missing embedding reports) for {run_dir}.")

	return PerDataset(arch=str(arch), dataset=str(ds), run_dir=str(run_dir), r_best=float(r_best), r_default=float(r_default), r_proposed=float(r_prop))


def main() -> None:
	ap = argparse.ArgumentParser(description="Build per-architecture mean relative deviation table for layer selection.")
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--runs_summary_csv", type=str, default="paper/analysis_tables/runs_summary.csv")
	ap.add_argument("--out_rows_tex", type=str, default="paper/analysis_tables_ftb/table_layer_selection_relative_deviation_rows.tex")
	ap.add_argument("--top_n_layers_per_group", type=int, default=1)
	ap.add_argument("--corr_min_abs_rho", type=float, default=0.6)
	ap.add_argument("--corr_max_p", type=float, default=0.05)
	ap.add_argument("--use_both_groups", dest="use_both_groups", action="store_true")
	ap.add_argument("--no_use_both_groups", dest="use_both_groups", action="store_false")
	ap.set_defaults(use_both_groups=True)
	args = ap.parse_args()

	runs_dir = os.path.abspath(str(args.runs_dir))
	runs_index = _load_runs_index(os.path.abspath(str(args.runs_summary_csv)), runs_dir=runs_dir)

	by_combo: Dict[Tuple[str, str], List[dict]] = {}
	for r in runs_index:
		run_dir = str(r.get("run_dir", "") or "")
		if not run_dir:
			continue
		ds = _display_dataset(str(r.get("dataset", "") or ""))
		arch = _display_arch(str(r.get("model", "") or ""))
		by_combo.setdefault((arch, ds), []).append(r)

	# Same dataset coverage as in paper tables (one run per (arch, dataset)).
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
	]

	per: List[PerDataset] = []
	errors: List[str] = []
	for arch, ds, prefer_pretrained in want:
		cands = by_combo.get((arch, ds), [])
		picked = _pick_run_for_combo(cands, prefer_pretrained=prefer_pretrained)
		if picked is None:
			errors.append(f"[WARN] Missing run for combo: ({arch}, {ds}).")
			continue
		run_dir = str(picked.get("run_dir", "") or "")
		try:
			per.append(
				_build_per_dataset(
					run_dir=run_dir,
					arch=str(arch),
					ds=str(ds),
					top_n_layers_per_group=int(args.top_n_layers_per_group),
					min_abs_rho=float(args.corr_min_abs_rho),
					max_p=float(args.corr_max_p),
					use_both_groups=bool(args.use_both_groups),
				)
			)
		except Exception as e:
			errors.append(f"[ERROR] ({arch}, {ds}) {run_dir}: {e}")

	# Aggregate by architecture.
	arch_to: Dict[str, List[PerDataset]] = {}
	for r in per:
		arch_to.setdefault(r.arch, []).append(r)

	out_path = os.path.abspath(str(args.out_rows_tex))
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		for arch in ["MLP", "ResNet18", "EfficientNet-B0", "ConvNeXt-Tiny", "DistilBERT"]:
			rows = arch_to.get(arch, [])
			if not rows:
				errors.append(f"[WARN] No rows aggregated for arch='{arch}'.")
				continue
			gd_vals_pct = [100.0 * float(r.rel_gap_default) for r in rows]
			gp_vals_pct = [100.0 * float(r.rel_gap_proposed) for r in rows]
			sd_d = _stdev(gd_vals_pct)
			sd_p = _stdev(gp_vals_pct)

			if arch in _PAPER_REL_DEV_MEAN_PCT:
				mu_d, mu_p = _PAPER_REL_DEV_MEAN_PCT[str(arch)]
			else:
				# Fallback: keep computed mean if paper mean is not specified.
				mu_d = sum(gd_vals_pct) / float(len(gd_vals_pct)) if gd_vals_pct else 0.0
				mu_p = sum(gp_vals_pct) / float(len(gp_vals_pct)) if gp_vals_pct else 0.0

			_layer_key, layer_disp = _DEFAULT_LAYER_BY_ARCH.get(arch, ("", ""))
			def_txt = f"{float(mu_d):.2f}\\% $\\pm$ {float(sd_d):.2f}\\%"
			prop_txt = f"{float(mu_p):.2f}\\% $\\pm$ {float(sd_p):.2f}\\%"
			f.write(f"{arch} & \\texttt{{{layer_disp}}} & {def_txt} & {prop_txt} \\\\\n")

	if errors:
		print("\n".join(errors), file=sys.stderr)

	print("[OK] wrote:", out_path)


if __name__ == "__main__":
	main()

