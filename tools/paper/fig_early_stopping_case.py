from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Use a project-local Matplotlib config dir to avoid slow/locked global caches.
_MPLCONFIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplconfig_render"))
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
_XDG_CACHE_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache_render"))
os.makedirs(_XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_HOME)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


@dataclass(frozen=True)
class Rule:
	layer: str
	metric: str
	mode: str  # "min" | "max"
	patience: int
	start_epoch: int
	min_delta: float


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	if not math.isfinite(float(v)):
		return None
	return float(v)


def _first_positive(vals: Any, zero_tol: float) -> Optional[float]:
	if not isinstance(vals, (list, tuple)):
		return None
	best: Optional[float] = None
	for v in vals:
		fv = _safe_float(v)
		if fv is None:
			continue
		if float(fv) > float(zero_tol):
			best = float(fv) if best is None else min(float(best), float(fv))
	return best


def _extract_signal_value(rec: dict, layer: str, metric: str, zero_tol: float) -> Optional[float]:
	layer_obj = (((rec.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {})
	if not isinstance(layer_obj, dict):
		return None
	if metric in {"beta1_L_est", "beta1_persistent_est", "mtopdiv_train_val"}:
		return _safe_float(layer_obj.get(metric, None))
	if metric == "persistent_q1_lambda1":
		return _first_positive(layer_obj.get("persistent_q1_smallest", None), zero_tol=zero_tol)
	if metric == "hodge_L_q0_lambda2":
		return _first_positive(layer_obj.get("hodge_L_q0_smallest", None), zero_tol=zero_tol)
	raise ValueError(f"Unknown metric: {metric}")


def _load_epoch_end_records(metrics_path: str) -> List[Dict[str, Any]]:
	recs: List[Dict[str, Any]] = []
	with open(metrics_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if obj.get("event") == "epoch_end":
				recs.append(obj)
	recs.sort(key=lambda r: int(r.get("epoch", -1)))
	return recs


def _series_bench(recs: Sequence[Dict[str, Any]], bench_key: str, metric: str) -> List[Tuple[int, float]]:
	out: List[Tuple[int, float]] = []
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		bench = (r.get("bench", {}) or {}).get(bench_key, {}) or {}
		if not isinstance(bench, dict):
			continue
		v = _safe_float(bench.get(metric, None))
		if v is None:
			continue
		out.append((int(ep), float(v)))
	out.sort(key=lambda x: x[0])
	return out


def _series_signal(recs: Sequence[Dict[str, Any]], rule: Rule, zero_tol: float) -> List[Tuple[int, float]]:
	out: List[Tuple[int, float]] = []
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		v = _extract_signal_value(r, layer=rule.layer, metric=rule.metric, zero_tol=zero_tol)
		if v is None:
			continue
		out.append((int(ep), float(v)))
	out.sort(key=lambda x: x[0])
	return out


def _simulate_stop_epoch(sig: Sequence[Tuple[int, float]], rule: Rule) -> Optional[int]:
	best: Optional[float] = None
	bad = 0
	for ep, v in sig:
		if int(ep) < int(rule.start_epoch):
			continue
		if best is None:
			best = float(v)
			bad = 0
			continue
		if rule.mode == "max":
			improved = float(v) > float(best) + float(rule.min_delta)
		else:
			improved = float(v) < float(best) - float(rule.min_delta)
		if improved:
			best = float(v)
			bad = 0
		else:
			bad += 1
		if int(bad) >= int(rule.patience):
			return int(ep)
	return None


def _value_at(series: Sequence[Tuple[int, float]], epoch: int) -> Optional[float]:
	for ep, v in series:
		if int(ep) == int(epoch):
			return float(v)
	return None


def _best_epoch(series: Sequence[Tuple[int, float]], mode: str = "max") -> Optional[int]:
	if not series:
		return None
	if mode == "min":
		return int(min(series, key=lambda x: x[1])[0])
	return int(max(series, key=lambda x: x[1])[0])


def _epoch_time_s(rec: Dict[str, Any]) -> Optional[float]:
	# Prefer timing_s.known_total if present (includes repr+bench overhead).
	timing = rec.get("timing_s", {}) or {}
	kt = _safe_float(timing.get("known_total", None))
	if kt is not None:
		return kt
	ex = rec.get("extra", {}) or {}
	tr = _safe_float(ex.get("train_s", None))
	va = _safe_float(ex.get("val_s", None))
	if tr is None or va is None:
		return None
	return float(tr + va)


def _pretty_metric(metric: str) -> str:
	m = str(metric)
	if m == "beta1_L_est":
		return r"$\beta_1(L)$"
	if m == "beta1_persistent_est":
		return r"$\beta_1^{K,L}$"
	if m == "hodge_L_q0_lambda2":
		return r"$\lambda_2(\Delta_0(L))$"
	if m == "persistent_q1_lambda1":
		return r"$\lambda_1(\Delta_1^{K,L})$"
	if m == "mtopdiv_train_val":
		return r"$\mathrm{MTopDiv}$"
	return m


def _pretty_bench_metric(metric: str) -> str:
	m = str(metric).lower().strip()
	if m == "accuracy":
		return "Доля правильных ответов"
	if m == "f1_macro":
		return "Макро-F1"
	if m == "bleu":
		return "BLEU"
	if m == "ppl":
		return "Perplexity"
	if m == "loss_assistant_only":
		return "Loss (assistant-only)"
	if m == "loss":
		return "Loss"
	return metric


def _bench_mode(metric: str) -> str:
	m = str(metric).lower().strip()
	if m in {"loss", "loss_assistant_only", "ppl"} or m.endswith("loss"):
		return "min"
	return "max"


def main() -> None:
	ap = argparse.ArgumentParser(description="Render a paper-style early-stopping case-study figure (2-panel).")
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--layer", type=str, required=True)
	ap.add_argument(
		"--metric",
		type=str,
		required=True,
		choices=["beta1_L_est", "beta1_persistent_est", "mtopdiv_train_val", "hodge_L_q0_lambda2", "persistent_q1_lambda1"],
	)
	ap.add_argument("--mode", type=str, required=True, choices=["min", "max"])
	ap.add_argument("--patience", type=int, required=True)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument(
		"--bench_metric",
		type=str,
		default="",
		help="Optional override for the validation metric plotted on the top panel (e.g., ppl, loss, loss_assistant_only). "
		"If omitted, uses Macro-F1 if present, otherwise Accuracy.",
	)
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--out_pdf", type=str, default="")
	ap.add_argument("--title", type=str, default="")
	args = ap.parse_args()

	run_dir = os.path.abspath(str(args.run_dir))
	meta_path = os.path.join(run_dir, "meta.json")
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"Missing meta.json: {meta_path}")
	if not os.path.exists(metrics_path):
		raise FileNotFoundError(f"Missing metrics.jsonl: {metrics_path}")

	meta = json.load(open(meta_path, "r", encoding="utf-8"))
	dataset = str(meta.get("dataset", ""))
	bench_key = f"{dataset}-val"
	zero_tol = float((meta.get("monitor", {}) or {}).get("zero_tol", 1e-8) or 1e-8)

	recs = _load_epoch_end_records(metrics_path)
	if not recs:
		raise RuntimeError("No epoch_end records found.")

	rule = Rule(
		layer=str(args.layer),
		metric=str(args.metric),
		mode=str(args.mode),
		patience=int(args.patience),
		start_epoch=int(args.start_epoch),
		min_delta=float(args.min_delta),
	)

	bench_metric = str(args.bench_metric).strip()
	if bench_metric:
		main_series = _series_bench(recs, bench_key=bench_key, metric=bench_metric)
		main_name = _pretty_bench_metric(bench_metric)
		bench_mode = _bench_mode(bench_metric)
		s_acc = []
		s_f1 = []
	else:
		s_acc = _series_bench(recs, bench_key=bench_key, metric="accuracy")
		s_f1 = _series_bench(recs, bench_key=bench_key, metric="f1_macro")
		main_series = s_f1 if s_f1 else s_acc
		main_name = "Макро-F1" if s_f1 else "Доля правильных ответов"
		bench_mode = "max"
	if not main_series:
		raise RuntimeError(f"No validation series found in bench '{bench_key}'.")

	s_sig = _series_signal(recs, rule=rule, zero_tol=zero_tol)
	if not s_sig:
		raise RuntimeError(f"No signal series for layer={rule.layer}, metric={rule.metric}.")

	stop_ep = _simulate_stop_epoch(s_sig, rule=rule)
	last_ep = int(main_series[-1][0])
	stop_eff = int(stop_ep) if stop_ep is not None else last_ep

	best_ep = _best_epoch(main_series, mode=bench_mode)
	best_val = _value_at(main_series, int(best_ep)) if best_ep is not None else None
	stop_val = _value_at(main_series, int(stop_eff))

	# Compute time saved (single-run) using recorded per-epoch timing.
	saved_s = 0.0
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		if int(ep) <= int(stop_eff):
			continue
		et = _epoch_time_s(r)
		if et is not None:
			saved_s += float(et)

	saved_epochs = max(0, last_ep - stop_eff)
	saved_pct = (saved_epochs / max(last_ep + 1, 1)) * 100.0

	os.makedirs(os.path.dirname(os.path.abspath(str(args.out_png))), exist_ok=True)
	if args.out_pdf:
		os.makedirs(os.path.dirname(os.path.abspath(str(args.out_pdf))), exist_ok=True)

	# Paper-like styling (inspired by strong CV papers).
	plt.rcParams.update(
		{
			# Match common arXiv figure style: bold labels, thick spines, strong grid.
			"font.family": "DejaVu Sans",
			"text.usetex": False,
			"mathtext.fontset": "dejavusans",
			"font.size": 14.0,
			"axes.labelsize": 22.0,
			"axes.titlesize": 16.0,
			"xtick.labelsize": 18.0,
			"ytick.labelsize": 18.0,
			"legend.fontsize": 14.0,
			"axes.linewidth": 1.8,
			"figure.dpi": 180,
			"savefig.dpi": 300,
		}
	)

	fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]})

	def _beautify(ax):
		# Strong major grid like in many arXiv plots.
		ax.grid(True, which="major", color="#AFAFAF", alpha=0.90, linestyle="-", linewidth=1.2)
		ax.set_axisbelow(True)
		# Keep full box (top/right visible) similar to reference style.
		for sp in ax.spines.values():
			sp.set_linewidth(1.8)
		ax.tick_params(direction="out", length=7, width=1.8)
		ax.xaxis.set_major_locator(MultipleLocator(2))

	# Top: validation metrics
	ax = axes[0]
	_beautify(ax)
	xm = [e for e, _ in main_series]
	if bench_mode == "max" and not bench_metric:
		# Backward-compatible: accuracy/F1 are fractions in logs.
		ym = [100.0 * v for _, v in main_series]
		y_label = "Валидация (%)"
	else:
		ym = [v for _, v in main_series]
		y_label = "Validation value"
	ax.plot(
		xm,
		ym,
		color="#d62728",
		linewidth=3.2,
		marker="o",
		markersize=6.5,
		label=f"{main_name}",
	)
	if s_acc and s_f1:
		xa = [e for e, _ in s_acc]
		ya = [100.0 * v for _, v in s_acc]
		ax.plot(
			xa,
			ya,
			color="#1f77b4",
			linewidth=2.8,
			linestyle=":",
			marker="x",
			markersize=7.0,
			markeredgewidth=2.0,
			alpha=0.95,
			label="Доля правильных ответов",
		)

	if best_ep is not None:
		vb = _value_at(main_series, int(best_ep))
		if vb is not None:
			ax.scatter([best_ep], [100.0 * vb if (bench_mode == "max" and not bench_metric) else vb], s=120, color="#111111", zorder=6, label="Эмпирически лучший")

	if stop_eff is not None:
		vs = _value_at(main_series, int(stop_eff))
		if vs is not None:
			ax.axvline(stop_eff, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85)
			ax.scatter(
				[stop_eff],
				[100.0 * vs if (bench_mode == "max" and not bench_metric) else vs],
				s=140,
				color="#d62728",
				edgecolors="white",
				linewidths=1.2,
				zorder=7,
				label="Ранняя остановка",
			)

	ax.set_ylabel(y_label)
	ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax.legend(loc="lower right", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	# Bottom: signal curve
	ax = axes[1]
	_beautify(ax)
	xs = [e for e, _ in s_sig]
	ys = [v for _, v in s_sig]
	ax.plot(
		xs,
		ys,
		color="#6a3d9a",
		linewidth=3.2,
		marker="o",
		markersize=6.0,
		label=f"{_pretty_metric(rule.metric)} @ {rule.layer}",
	)
	if stop_eff is not None:
		vss = _value_at(s_sig, int(stop_eff))
		if vss is not None:
			ax.axvline(stop_eff, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85)
			ax.scatter([stop_eff], [vss], s=95, color="#d62728", edgecolors="white", linewidths=1.0, zorder=7)

	ax.set_xlabel("Эпоха", fontweight="bold")
	ax.set_ylabel(r"$\lambda_1(\Delta_1^{K,L})$", )#fontweight="bold")

	drop_txt = ""
	if best_val is not None and stop_val is not None:
		if bench_mode == "min":
			gap = float(stop_val) - float(best_val)
		else:
			gap = float(best_val) - float(stop_val)
		drop_txt = f"Δ={gap:.3f}"

	title_raw = str(args.title).strip()
	suppress_title = title_raw.lower() in {"__none__", "none", "off"}
	title = "" if suppress_title else title_raw
	if (not suppress_title) and (not title):
		title = f"{os.path.basename(run_dir)} | {dataset} | saved={saved_epochs} epochs ({saved_pct:.1f}%), {saved_s/60.0:.1f} min | {drop_txt}"
	if title:
		fig.suptitle(title, fontsize=13.5)

	fig.align_ylabels(axes)
	fig.subplots_adjust(left=0.16, right=0.95, top=0.86 if title else 0.96, bottom=0.15, hspace=0.15)
	fig.savefig(str(args.out_png), bbox_inches="tight")
	if args.out_pdf:
		fig.savefig(str(args.out_pdf), bbox_inches="tight")
	plt.close(fig)

	print("[OK] stop_epoch_trigger =", stop_ep)
	print("[OK] stop_epoch_effective =", stop_eff)
	print("[OK] last_epoch =", last_ep)
	print("[OK] saved_epochs =", saved_epochs)
	print("[OK] saved_pct =", saved_pct)
	print("[OK] saved_s =", saved_s)
	print("[OK] best_epoch =", best_ep)
	print("[OK] best_value =", best_val)
	print("[OK] stop_value =", stop_val)


if __name__ == "__main__":
	main()
