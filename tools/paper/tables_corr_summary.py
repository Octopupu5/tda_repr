from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _iter_jsonl(path: str) -> Iterable[dict]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _pick_primary_metric(bench: dict) -> Tuple[str, float]:
	"""
	Return (metric_name, value). Prefers:
	- ppl for generation-style runs
	- f1_macro for classification-style runs
	- then accuracy / bleu / loss-like metrics
	"""
	for k in ("ppl",):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("f1_macro", "accuracy", "bleu"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("loss_assistant_only", "loss", "ppl"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	return "unknown", float("nan")


def _first_two_positive(eigs: Any, zero_tol: float) -> Tuple[Optional[float], Optional[float]]:
	if not isinstance(eigs, (list, tuple)) or not eigs:
		return None, None
	vals: List[float] = []
	for v in eigs:
		fv = float(v)
		if math.isfinite(fv) and fv > float(zero_tol):
			vals.append(fv)
	vals.sort()
	if not vals:
		return None, None
	if len(vals) == 1:
		return vals[0], None
	return vals[0], vals[1]


def _first_positive(eigs: Any, zero_tol: float) -> Optional[float]:
	x, _ = _first_two_positive(eigs, zero_tol=zero_tol)
	return x


def _median(xs: List[float]) -> Optional[float]:
	if not xs:
		return None
	return float(np.median(np.asarray(xs, dtype=np.float64)))


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float], int]:
	vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
	if not vals:
		return None, None, 0
	mu = float(np.mean(np.asarray(vals, dtype=np.float64)))
	if len(vals) >= 2:
		sd = float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))
	else:
		sd = 0.0
	return mu, sd, int(len(vals))


@dataclass(frozen=True)
class DescriptorSpec:
	key: str
	display_latex: str
	extract: Callable[[dict, float], Optional[float]]


def _extract_scalar(k: str) -> Callable[[dict, float], Optional[float]]:
	def _f(layer_out: dict, zero_tol: float) -> Optional[float]:
		v = layer_out.get(k, None)
		if isinstance(v, (int, float)):
			fv = float(v)
			return fv if math.isfinite(fv) else None
		return None

	return _f


def _extract_l2_q0(layer_out: dict, zero_tol: float) -> Optional[float]:
	return _first_positive(layer_out.get("hodge_L_q0_smallest", None), zero_tol=zero_tol)


def _extract_l2_q0_persistent(layer_out: dict, zero_tol: float) -> Optional[float]:
	return _first_positive(layer_out.get("persistent_q0_smallest", None), zero_tol=zero_tol)


def _extract_l1_q1(layer_out: dict, zero_tol: float) -> Optional[float]:
	return _first_positive(layer_out.get("hodge_L_q1_smallest", None), zero_tol=zero_tol)


def _extract_l2_q1(layer_out: dict, zero_tol: float) -> Optional[float]:
	_, l2 = _first_two_positive(layer_out.get("hodge_L_q1_smallest", None), zero_tol=zero_tol)
	return l2


def _extract_l1_q1_persistent(layer_out: dict, zero_tol: float) -> Optional[float]:
	return _first_positive(layer_out.get("persistent_q1_smallest", None), zero_tol=zero_tol)


def _extract_l2_q1_persistent(layer_out: dict, zero_tol: float) -> Optional[float]:
	_, l2 = _first_two_positive(layer_out.get("persistent_q1_smallest", None), zero_tol=zero_tol)
	return l2


DESCRIPTORS: List[DescriptorSpec] = [
	DescriptorSpec(key="beta0_L_est", display_latex=r"$\beta_0(L)$", extract=_extract_scalar("beta0_L_est")),
	DescriptorSpec(key="beta1_L_est", display_latex=r"$\beta_1(L)$", extract=_extract_scalar("beta1_L_est")),
	DescriptorSpec(key="beta0_persistent_est", display_latex=r"$\beta_0^{K,L}$", extract=_extract_scalar("beta0_persistent_est")),
	DescriptorSpec(key="beta1_persistent_est", display_latex=r"$\beta_1^{K,L}$", extract=_extract_scalar("beta1_persistent_est")),
	DescriptorSpec(key="lambda2_q0", display_latex=r"$\lambda_2(\Delta_0)$", extract=_extract_l2_q0),
	DescriptorSpec(key="lambda2_q0_persistent", display_latex=r"$\lambda_2(\Delta_0^{K,L})$", extract=_extract_l2_q0_persistent),
	DescriptorSpec(key="lambda1_q1", display_latex=r"$\lambda_1(\Delta_1)$", extract=_extract_l1_q1),
	DescriptorSpec(key="lambda2_q1", display_latex=r"$\lambda_2(\Delta_1)$", extract=_extract_l2_q1),
	DescriptorSpec(key="lambda1_q1_persistent", display_latex=r"$\lambda_1(\Delta_1^{K,L})$", extract=_extract_l1_q1_persistent),
	DescriptorSpec(key="lambda2_q1_persistent", display_latex=r"$\lambda_2(\Delta_1^{K,L})$", extract=_extract_l2_q1_persistent),
	DescriptorSpec(key="mtopdiv_train_val", display_latex=r"$\mathrm{MTopDiv}$", extract=_extract_scalar("mtopdiv_train_val")),
]


def _compute_spearman(xs: List[float], ys: List[float]) -> Optional[float]:
	if len(xs) < 3 or len(ys) < 3:
		return None
	if len(set(xs)) < 2 or len(set(ys)) < 2:
		return None
	x = np.asarray(xs, dtype=np.float64)
	y = np.asarray(ys, dtype=np.float64)
	if x.shape != y.shape:
		return None

	def _rankdata(a: np.ndarray) -> np.ndarray:
		# Average ranks for ties, ranks start at 1.
		n = int(a.shape[0])
		order = np.argsort(a, kind="mergesort")
		ranks = np.empty(n, dtype=np.float64)
		i = 0
		while i < n:
			j = i
			av = a[order[i]]
			while j + 1 < n and a[order[j + 1]] == av:
				j += 1
			r = 0.5 * (float(i) + float(j)) + 1.0
			ranks[order[i : j + 1]] = r
			i = j + 1
		return ranks

	rx = _rankdata(x)
	ry = _rankdata(y)
	rx = rx - float(np.mean(rx))
	ry = ry - float(np.mean(ry))
	den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
	if den <= 0.0 or not math.isfinite(den):
		return None
	rho = float(np.sum(rx * ry) / den)
	return rho if math.isfinite(rho) else None


def _load_epoch_series(run_dir: str) -> Tuple[dict, dict, float, List[Tuple[int, Dict[str, dict], float]]]:
	meta_path = os.path.join(run_dir, "meta.json")
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	meta = _read_json(meta_path)
	args = meta.get("args", {}) or {}
	task = str(meta.get("task", args.get("task", "")) or "")
	dataset = str(meta.get("dataset", args.get("dataset", "")) or "")

	monitor = meta.get("monitor", {}) or {}
	zero_tol = float(monitor.get("zero_tol", 1e-8) or 1e-8)

	epochs: List[Tuple[int, Dict[str, dict], dict]] = []
	for ev in _iter_jsonl(metrics_path):
		if str(ev.get("event", "")) != "epoch_end":
			continue
		ep = ev.get("epoch", None)
		if not isinstance(ep, int):
			continue
		repr_layers = (ev.get("repr", {}) or {}).get("layers", None)
		bench = ev.get("bench", {}) or {}
		if not isinstance(repr_layers, dict):
			raise ValueError(f"Bad repr.layers at epoch={ep} in {metrics_path}")
		epochs.append((ep, repr_layers, bench))

	if not epochs:
		raise ValueError(f"No epoch_end records in {metrics_path}")
	epochs.sort(key=lambda x: x[0])

	val_key = f"{dataset}-val"
	metric_name = "unknown"
	for _, _, b in epochs:
		bx = (b or {}).get(val_key, {}) or {}
		mn, _ = _pick_primary_metric(bx)
		if mn != "unknown":
			metric_name = mn
			break
	if metric_name == "unknown":
		raise ValueError(f"Could not infer primary val metric for run={run_dir} val_key={val_key!r}")

	series: List[Tuple[int, Dict[str, dict], float]] = []
	for ep, repr_layers, b in epochs:
		v = ((b or {}).get(val_key, {}) or {}).get(metric_name, None)
		if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
			continue
		series.append((ep, repr_layers, float(v)))
	if len(series) < 3:
		raise ValueError(f"Not enough (epoch, val_metric) points for run={run_dir} metric={metric_name}")

	return {"task": task, "dataset": dataset, "val_key": val_key, "metric": metric_name}, monitor, zero_tol, series


def main() -> None:
	ap = argparse.ArgumentParser(
		description="Build LaTeX table summarizing Spearman alignment between descriptors and validation metric."
	)
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--out_tex", type=str, default="paper/analysis_tables_ftb/table_correlation_summary.tex")
	ap.add_argument("--out_rows_tex", type=str, default="paper/analysis_tables_ftb/table_correlation_summary_rows.tex")
	ap.add_argument("--abs_rho_threshold", type=float, default=0.6)
	args = ap.parse_args()

	runs_root = Path(str(args.runs_dir))
	if not runs_root.exists():
		raise FileNotFoundError(f"Missing runs_dir: {runs_root}")

	# Each run contributes one summary statistic per descriptor, then we aggregate across runs.
	per_run_med_abs: Dict[str, List[float]] = {d.key: [] for d in DESCRIPTORS}
	per_run_frac_strong: Dict[str, List[float]] = {d.key: [] for d in DESCRIPTORS}

	run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
	if not run_dirs:
		raise ValueError(f"No runs found under: {runs_root}")

	for run_dir in run_dirs:
		meta_path = run_dir / "meta.json"
		metrics_path = run_dir / "metrics.jsonl"
		if not (meta_path.exists() and metrics_path.exists()):
			continue

		_run_info, monitor, zero_tol, series = _load_epoch_series(str(run_dir))
		layer_names = monitor.get("layer_names", None)
		if not isinstance(layer_names, list) or not layer_names:
			raise ValueError(f"Missing monitor.layer_names in {meta_path}")

		run_abs_by_desc: Dict[str, List[float]] = {d.key: [] for d in DESCRIPTORS}
		for layer in layer_names:
			layer = str(layer)
			for d in DESCRIPTORS:
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

		th = float(args.abs_rho_threshold)
		for d in DESCRIPTORS:
			vals = run_abs_by_desc[d.key]
			med = _median(vals)
			if med is not None:
				per_run_med_abs[d.key].append(float(med))
			if vals:
				frac = float(sum(1 for v in vals if float(v) >= th)) / float(len(vals))
				per_run_frac_strong[d.key].append(float(frac))

	out_rows: List[Tuple[str, str, str]] = []
	th = float(args.abs_rho_threshold)
	for d in DESCRIPTORS:
		mu_m, sd_m, _n1 = _mean_std(per_run_med_abs[d.key])
		mu_f, sd_f, _n2 = _mean_std(per_run_frac_strong[d.key])
		med_txt = "NA" if mu_m is None else f"{mu_m:.2f} $\\pm$ {0.0 if sd_m is None else sd_m:.2f}"
		# Keep as a fraction in [0, 1] to match the paper tables (not percentages).
		frac_txt = "NA" if mu_f is None else f"{mu_f:.2f} $\\pm$ {0.0 if sd_f is None else sd_f:.2f}"
		out_rows.append((d.display_latex, med_txt, frac_txt))

	os.makedirs(os.path.dirname(str(args.out_tex)), exist_ok=True)
	os.makedirs(os.path.dirname(str(args.out_rows_tex)), exist_ok=True)

	rows_tex_lines = []
	for name, med_s, frac_s in out_rows:
		rows_tex_lines.append(f"{name} & {med_s} & {frac_s} \\\\")
	rows_tex = "\n".join(rows_tex_lines) + "\n"
	Path(str(args.out_rows_tex)).write_text(rows_tex, encoding="utf-8")

	full_tex = (
		"\\begin{table}[htbp]\n"
		"\\caption{Summary of alignment between representation descriptors and validation metrics across monitored layers. "
		"For each run and descriptor, we compute (i) the median absolute Spearman correlation with the validation metric across monitored layers, "
		f"and (ii) the fraction of layers with strong monotonic association ($|\\rho_S| \\ge {th:.1f}$). "
		"We then report the mean $\\pm$ standard deviation of these quantities across runs.}\n"
		"\\label{tab:correlation_summary}\n"
		"\\centering\n"
		"\\renewcommand{\\arraystretch}{1.2}\n"
		"\\begin{tabular}{lcc}\n"
		"\\toprule\n"
		"\\textbf{Descriptor} & \\textbf{Median $|\\rho_S|$} & \\textbf{Frac. with $|\\rho_S| \\ge 0.6$} \\\\\n"
		"\\midrule\n"
		f"{rows_tex}"
		"\\bottomrule\n"
		"\\end{tabular}\n"
		"\\end{table}\n"
	)
	Path(str(args.out_tex)).write_text(full_tex, encoding="utf-8")

	print(f"[OK] wrote {args.out_tex}")
	print(f"[OK] wrote {args.out_rows_tex}")
	for d in DESCRIPTORS:
		print(f"[runs] {d.key}: n_med={len(per_run_med_abs[d.key])} n_frac={len(per_run_frac_strong[d.key])}")


if __name__ == "__main__":
	main()

