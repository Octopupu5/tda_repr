from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.aggregate.tables_corr_summary import (
	DESCRIPTORS as CORR_TABLE_DESCRIPTORS,
	_compute_spearman,
	_load_epoch_series,
	_mean_std,
	_median,
)
from tools.aggregate.tables_depth_arch import (
	_aggregate_depth_dynamics,
	_display_arch,
	_display_dataset,
	_fmt_pm,
	_load_run,
	RunSummary,
)


def _collect_run_dirs_flat(runs_root: str) -> List[str]:
	from tools.aggregate.reproduction_tables import _experiment_run_dirs_flat_and_nested

	return _experiment_run_dirs_flat_and_nested(str(runs_root))


def load_run_summaries_for_paper_tables(runs_root: str) -> List[RunSummary]:
	"""Collect ``RunSummary`` for each runnable experiment dir (includes MLP for arch row)."""

	out: List[RunSummary] = []
	for rd in _collect_run_dirs_flat(runs_root):
		rs = _load_run(str(rd))
		if rs is not None:
			out.append(rs)
	return out


def _spec_to_display_bucket(spec: Any) -> Tuple[str, str, str]:
	from tools.aggregate.reproduction_tables import PaperSpec

	if not isinstance(spec, PaperSpec):
		raise TypeError("PaperSpec expected")
	mod = "NLP" if str(spec.task).lower().strip() == "nlp" else "CV"
	ds = _display_dataset(str(spec.dataset))
	m = str(spec.model).lower().strip()
	if m == "distilbert":
		arch = _display_arch(spec.model, True)
	elif m in ("smollm2-135m", "smollm"):
		arch = _display_arch("smollm2-135m", None)
	else:
		arch = _display_arch(spec.model, None)
	return arch, ds, mod


def arch_rows_to_stats_by_key(rows: Sequence[Mapping[str, Any]], specs: Sequence[Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Any]]:
	from tools.aggregate.reproduction_tables import PaperSpec

	by_triple: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
	for r in rows:
		k = (str(r["architecture"]), str(r["dataset"]), str(r["modality"]))
		by_triple[k] = r
	stats_by_key: Dict[str, Dict[str, Any]] = {}
	specs_kept: List[Any] = []
	for spec in specs:
		if not isinstance(spec, PaperSpec):
			continue
		arch, ds, mod = _spec_to_display_bucket(spec)
		row = by_triple.get((arch, ds, mod))
		mdl = str(spec.model).lower().strip()
		if row is None and "distilbert" in mdl:
			for pa in (True, False, None):
				a2 = _display_arch(spec.model, pa)
				row = by_triple.get((a2, ds, mod))
				if row is not None:
					break
		if row is None:
			continue
		stats_by_key[spec.key] = {
			"mean_task": row.get("d_task_metric_mean"),
			"std_task": row.get("d_task_metric_std") if row.get("d_task_metric_std") is not None else 0.0,
			"mean_b1kl": row.get("d_beta1_persistent_mean"),
			"std_b1kl": row.get("d_beta1_persistent_std") if row.get("d_beta1_persistent_std") is not None else 0.0,
			"mean_mtop": row.get("d_mtopdiv_mean"),
			"std_mtop": row.get("d_mtopdiv_std") if row.get("d_mtopdiv_std") is not None else 0.0,
			"mean_l0": row.get("d_lambda2_q0_mean"),
			"std_l0": row.get("d_lambda2_q0_std") if row.get("d_lambda2_q0_std") is not None else 0.0,
			"mean_l1": row.get("d_lambda1_q1_persistent_mean"),
			"std_l1": row.get("d_lambda1_q1_persistent_std") if row.get("d_lambda1_q1_persistent_std") is not None else 0.0,
		}
		specs_kept.append(spec)
	return stats_by_key, specs_kept


def _mean_run_frac_mtopdiv_tex(depth_row: Mapping[str, Any]) -> str:
	mu = depth_row.get("frac_mtopdiv_pos", {}).get("mean")
	if mu is None:
		return r"$\mathrm{n/a}$"
	try:
		fv = float(mu)
	except (TypeError, ValueError):
		return r"$\mathrm{n/a}$"
	if not math.isfinite(fv) or fv < 0.0:
		return r"$\mathrm{n/a}$"
	clamped = min(1.0, max(0.0, fv))
	return f"{100.0 * clamped:.1f}\\%"


def write_depth_dynamics_table_tex(
	path: str,
	run_dirs: Sequence[str],
	*,
	modality: Optional[str] = None,
) -> None:
	_mod = str(modality or "").strip().lower()
	assert _mod in {"", "cv", "nlp"}
	runs = [_load_run(str(rd)) for rd in run_dirs]
	runs = [r for r in runs if r is not None]
	depth = _aggregate_depth_dynamics(runs)
	buckets = ("early", "intermediate", "deep", "head")
	labels = {
		"early": r"Early (Stem/Embeddings)",
		"intermediate": r"Intermediate (Mid Blocks)",
		"deep": r"Deep (Final Backbone/Pool)",
		"head": r"Head (Classifier / pre-head)",
	}

	def row(g: str) -> str:
		db1 = _fmt_pm(depth[g]["d_beta1_L"].get("mean"), depth[g]["d_beta1_L"].get("std"), nd=1, signed=True)
		db1p = _fmt_pm(depth[g]["d_beta1_pers"].get("mean"), depth[g]["d_beta1_pers"].get("std"), nd=1, signed=True)
		dl2 = _fmt_pm(depth[g]["d_lambda2_q0"].get("mean"), depth[g]["d_lambda2_q0"].get("std"), nd=3, signed=True)
		dl1 = _fmt_pm(depth[g]["d_l1_q1_pers"].get("mean"), depth[g]["d_l1_q1_pers"].get("std"), nd=3, signed=True)
		dl2q1 = _fmt_pm(depth[g]["d_l2_q1_pers"].get("mean"), depth[g]["d_l2_q1_pers"].get("std"), nd=3, signed=True)
		fr = _mean_run_frac_mtopdiv_tex(depth[g])
		return f"{labels[g]:<30} & {db1} & {db1p} & {dl2} & {dl1} & {dl2q1} & {fr} \\\\"

	body_lines = [row(g) for g in buckets]
	depth_dynamics_header = (
		r"\textbf{Layer Depth} & $\mathbf{\Delta \beta_1(L)}$ & $\mathbf{\Delta \beta_1^{K,L}}$ & "
		r"$\mathbf{\Delta \lambda_2(\Delta_0(L))}$ & $\mathbf{\Delta \lambda_1(\Delta_1^{K,L})}$ & "
		r"$\mathbf{\Delta \lambda_2(\Delta_1^{K,L})}$ & "
		r"\textbf{Fraction with $\mathbf{\Delta \text{MTopDiv} > 0}$} \\"
	)
	lines = [r"\toprule", depth_dynamics_header, r"\midrule"] + body_lines + [r"\bottomrule"]
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))
		f.write("\n")


def write_correlation_summary_tex(
	path: str,
	run_dirs: Sequence[str],
	*,
	abs_rho_threshold: float = 0.6,
	modality: Optional[str] = None,
) -> None:
	_modality = str(modality or "").strip().lower()
	assert _modality in {"", "cv", "nlp"}
	run_dirs = [str(p) for p in run_dirs if os.path.isfile(os.path.join(str(p), "metrics.jsonl"))]
	per_run_med_abs: Dict[str, List[float]] = {d.key: [] for d in CORR_TABLE_DESCRIPTORS}
	per_run_frac_strong: Dict[str, List[float]] = {d.key: [] for d in CORR_TABLE_DESCRIPTORS}
	th = float(abs_rho_threshold)

	for run_dir in run_dirs:
		try:
			_run_info, monitor, zero_tol, series = _load_epoch_series(str(run_dir))
		except (OSError, ValueError, json.JSONDecodeError, TypeError):
			continue
		layer_names = monitor.get("layer_names", None)
		if not isinstance(layer_names, list) or not layer_names:
			continue
		run_abs_by_desc: Dict[str, List[float]] = {d.key: [] for d in CORR_TABLE_DESCRIPTORS}
		for layer in layer_names:
			layer = str(layer)
			for d in CORR_TABLE_DESCRIPTORS:
				xs: List[float] = []
				ys: List[float] = []
				for _, repr_layers, y in series:
					layer_out = repr_layers.get(layer, None)
					if not isinstance(layer_out, dict):
						continue
					x = d.extract(layer_out, zero_tol)
					if x is None:
						continue
					xs.append(float(x))
					ys.append(float(y))
				rho = _compute_spearman(xs, ys)
				if rho is None:
					continue
				run_abs_by_desc[d.key].append(abs(float(rho)))
		for d in CORR_TABLE_DESCRIPTORS:
			vals = run_abs_by_desc[d.key]
			med = _median(vals)
			if med is not None:
				per_run_med_abs[d.key].append(float(med))
			if vals:
				frac = float(sum(1 for v in vals if float(v) >= th)) / float(len(vals))
				per_run_frac_strong[d.key].append(float(frac))

	out_rows: List[Tuple[str, str, str]] = []
	for d in CORR_TABLE_DESCRIPTORS:
		mu_m, sd_m, _n1 = _mean_std(per_run_med_abs[d.key])
		mu_f, _sd_f, _n2 = _mean_std(per_run_frac_strong[d.key])
		med_txt = "NA" if mu_m is None else f"{mu_m:.2f} $\\pm$ {0.0 if sd_m is None else sd_m:.2f}"
		frac_txt = "NA" if mu_f is None else f"{mu_f:.2f}"
		out_rows.append((d.display_latex, med_txt, frac_txt))

	body_rows = [f"{name} & {med_s} & {frac_s} \\\\" for name, med_s, frac_s in out_rows]
	corr_summary_header = (
		r"\textbf{Descriptor} & \textbf{Mean median $|\rho_S|$} & "
		r"\textbf{Frac. with $|\rho_S| \ge 0.6$} \\"
	)
	lines = [r"\toprule", corr_summary_header, r"\midrule"] + body_rows + [r"\bottomrule"]
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))
		f.write("\n")
