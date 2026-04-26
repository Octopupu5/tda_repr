from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.stats import spearmanr


# Use a project-local Matplotlib config dir to avoid slow/locked global caches.
_MPLCONFIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplconfig_render"))
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


def _iter_jsonl(path: str) -> Iterable[dict]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	if not math.isfinite(float(v)):
		return None
	return float(v)


def _is_near_constant(values: Sequence[float], atol: float = 1e-8, rtol: float = 1e-4) -> bool:
	if len(values) < 2:
		return True
	a = np.asarray(list(values), dtype=np.float64)
	vmin = float(np.min(a))
	vmax = float(np.max(a))
	scale = max(1.0, float(np.max(np.abs(a))))
	return (vmax - vmin) <= (float(atol) + float(rtol) * scale)


def _pick_primary_metric(bench: dict) -> Tuple[str, str]:
	"""
	Return (metric_name, display_name). Prefer acc/f1 for classification, bleu for generation, then ppl/loss.
	"""
	if isinstance(bench, dict):
		if isinstance(bench.get("accuracy"), (int, float)):
			return "accuracy", "Доля правильных ответов"
		if isinstance(bench.get("f1_macro"), (int, float)):
			return "f1_macro", "Макро-F1"
		if isinstance(bench.get("bleu"), (int, float)):
			return "bleu", "BLEU"
		if isinstance(bench.get("ppl"), (int, float)):
			return "ppl", "Перплексия"
		if isinstance(bench.get("loss_assistant_only"), (int, float)):
			return "loss_assistant_only", "Потери (assistant-only)"
		if isinstance(bench.get("loss"), (int, float)):
			return "loss", "Потери"
	return "unknown", "Metric"


def _pretty_metric_label(metric: str) -> str:
	m = str(metric).lower().strip()
	if m == "accuracy":
		return "Доля правильных ответов"
	if m == "ppl":
		return "Перплексия"
	if m == "f1_macro":
		return "Макро-F1"
	if m == "bleu":
		return "BLEU"
	if m == "loss_assistant_only":
		return "Потери (assistant-only)"
	if m == "loss":
		return "Потери"
	return str(metric)


def _is_minimization_metric(metric: str) -> bool:
	m = str(metric).lower().strip()
	if m in {"loss", "loss_assistant_only", "ppl"}:
		return True
	return m.endswith("loss")


def _load_epoch_end(metrics_path: str) -> List[dict]:
	recs = [ev for ev in _iter_jsonl(metrics_path) if ev.get("event") == "epoch_end"]
	recs.sort(key=lambda r: int(r.get("epoch", -1)))
	return recs


def _series_bench(recs: Sequence[dict], bench_key: str, metric: str) -> Dict[int, float]:
	out: Dict[int, float] = {}
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
		out[int(ep)] = float(v)
	return out


def _series_layer_mtopdiv(recs: Sequence[dict], layer: str) -> Dict[int, float]:
	out: Dict[int, float] = {}
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		layer_obj = (((r.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {})
		if not isinstance(layer_obj, dict):
			continue
		v = _safe_float(layer_obj.get("mtopdiv_train_val", None))
		if v is None:
			continue
		out[int(ep)] = float(v)
	return out


def _layer_names_with_mtopdiv(recs: Sequence[dict]) -> List[str]:
	names = set()
	for r in recs:
		layers = ((r.get("repr", {}) or {}).get("layers", {}) or {})
		if not isinstance(layers, dict):
			continue
		for k, v in layers.items():
			if not isinstance(v, dict):
				continue
			if isinstance(v.get("mtopdiv_train_val", None), (int, float)):
				names.add(str(k))
	return sorted(names)


@dataclass(frozen=True)
class BestLayer:
	layer: str
	rho: float
	abs_rho: float
	p_value: float
	n: int
	bench_metric: str
	bench_key: str


def _best_mtopdiv_layer(
	recs: Sequence[dict],
	*,
	bench_key: str,
	bench_metric: str,
	min_common_epochs: int = 3,
) -> BestLayer:
	bench = _series_bench(recs, bench_key=bench_key, metric=bench_metric)
	if len(bench) < int(min_common_epochs):
		raise RuntimeError(f"Not enough bench points for '{bench_key}.{bench_metric}': n={len(bench)}.")

	# If an init epoch exists (epoch < 0), exclude it from correlation ranking to avoid
	# init-vs-trained discontinuity dominating Spearman rho. We will still plot it later.
	if bench and min(bench.keys()) < 0:
		bench = {e: v for e, v in bench.items() if int(e) >= 0}

	bench_for_corr = dict(bench)
	if _is_minimization_metric(bench_metric):
		bench_for_corr = {e: -float(v) for e, v in bench_for_corr.items()}

	best: Optional[BestLayer] = None
	for layer in _layer_names_with_mtopdiv(recs):
		s = _series_layer_mtopdiv(recs, layer=layer)
		if s and min(s.keys()) < 0:
			s = {e: v for e, v in s.items() if int(e) >= 0}
		common = sorted(set(bench_for_corr.keys()) & set(s.keys()))
		if len(common) < int(min_common_epochs):
			continue
		x = [bench_for_corr[e] for e in common]
		y = [s[e] for e in common]
		if _is_near_constant(x) or _is_near_constant(y):
			continue
		rho, p = spearmanr(x, y)
		if not np.isfinite(rho):
			continue
		cand = BestLayer(
			layer=str(layer),
			rho=float(rho),
			abs_rho=float(abs(rho)),
			p_value=float(p),
			n=int(len(common)),
			bench_metric=str(bench_metric),
			bench_key=str(bench_key),
		)
		if best is None or cand.abs_rho > best.abs_rho:
			best = cand

	if best is None:
		raise RuntimeError("No valid layer correlations found for mtopdiv_train_val.")
	return best


def _pretty_run_title(meta: dict) -> str:
	model = str(meta.get("model", "") or "")
	dataset = str(meta.get("dataset", "") or "")
	return f"{model} | {dataset}"


def _beautify(ax) -> None:
	ax.grid(True, which="major", color="#AFAFAF", alpha=0.90, linestyle="-", linewidth=1.2)
	ax.set_axisbelow(True)
	for sp in ax.spines.values():
		sp.set_linewidth(1.8)
	ax.tick_params(direction="out", length=7, width=1.8)
	ax.xaxis.set_major_locator(MultipleLocator(2))


def _plot_one(
	axes: Tuple[Any, Any],
	*,
	recs: Sequence[dict],
	meta: dict,
	best: BestLayer,
	bench_metric_label: str,
	show_axis_labels: bool,
	show_legends: bool,
) -> None:
	dataset = str(meta.get("dataset", "") or "")
	bench_key = f"{dataset}-val"
	bench = _series_bench(recs, bench_key=bench_key, metric=best.bench_metric)
	s = _series_layer_mtopdiv(recs, layer=best.layer)
	common = sorted(set(bench.keys()) & set(s.keys()))
	if not common:
		raise RuntimeError("No common epochs between bench and mtopdiv series.")

	# If init epoch is logged as a negative epoch (epoch=-1), shift the x-axis by +1
	# so that init appears at epoch 0 on the plot.
	shift = 1 if (common and min(common) < 0) else 0
	x = [int(e) + int(shift) for e in common]
	y_bench = [bench[e] for e in common]
	y_sig = [s[e] for e in common]

	ax_top, ax_bot = axes
	_beautify(ax_top)
	_beautify(ax_bot)

	is_pct = best.bench_metric in {"accuracy", "f1_macro"}
	if is_pct:
		ax_top.plot(x, [100.0 * v for v in y_bench], color="#1f77b4", linewidth=3.2, marker="o", markersize=6.5, label=bench_metric_label)
	else:
		ax_top.plot(x, y_bench, color="#1f77b4", linewidth=3.2, marker="o", markersize=6.5, label=bench_metric_label)
	ax_top.yaxis.set_major_locator(MaxNLocator(nbins=6))
	if not bool(show_axis_labels):
		ax_top.set_ylabel("")
	if bool(show_legends):
		ax_top.legend(loc="best", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	ax_bot.plot(
		x,
		y_sig,
		color="#6a3d9a",
		linewidth=3.2,
		marker="o",
		markersize=6.0,
		label=r"$\mathrm{MTopDiv}$",
	)
	ax_bot.set_xlabel("Эпоха", fontweight="bold")
	if not bool(show_axis_labels):
		ax_bot.set_ylabel("")
	if bool(show_legends):
		ax_bot.legend(loc="best", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	title = _pretty_run_title(meta)
	ax_top.set_title(f"{title}", fontsize=13.5)


def main() -> None:
	ap = argparse.ArgumentParser(description="Render dynamics for the most mtopdiv-correlated layer (two runs).")
	ap.add_argument("--run_a", type=str, required=True, help="Run directory (e.g., runs/exp_...).")
	ap.add_argument("--run_b", type=str, required=True, help="Run directory (e.g., runs/exp_...).")
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--min_common_epochs", type=int, default=3)
	ap.add_argument(
		"--style",
		type=str,
		default="paper",
		choices=["paper", "debug"],
		help="paper: minimal in-plot labels; debug: include axis labels/legends.",
	)
	ap.add_argument(
		"--bench_metric_a",
		type=str,
		default="",
		help="Override validation metric for run A (e.g., accuracy, f1_macro, ppl, loss). If empty, auto-pick.",
	)
	ap.add_argument(
		"--bench_metric_b",
		type=str,
		default="",
		help="Override validation metric for run B (e.g., accuracy, f1_macro, ppl, loss). If empty, auto-pick.",
	)
	args = ap.parse_args()

	def load(run_dir: str) -> Tuple[List[dict], dict]:
		run_dir = os.path.abspath(str(run_dir))
		meta_path = os.path.join(run_dir, "meta.json")
		metrics_path = os.path.join(run_dir, "metrics.jsonl")
		if not os.path.exists(meta_path):
			raise FileNotFoundError(f"Missing meta.json: {meta_path}")
		if not os.path.exists(metrics_path):
			raise FileNotFoundError(f"Missing metrics.jsonl: {metrics_path}")
		meta = json.load(open(meta_path, "r", encoding="utf-8"))
		recs = _load_epoch_end(metrics_path)
		if not recs:
			raise RuntimeError(f"No epoch_end records: {metrics_path}")
		return recs, meta

	recs_a, meta_a = load(str(args.run_a))
	recs_b, meta_b = load(str(args.run_b))

	def auto_metric(recs: Sequence[dict], meta: dict, override: str) -> Tuple[str, str, str]:
		dataset = str(meta.get("dataset", "") or "")
		bench_key = f"{dataset}-val"
		first_bench = (recs[0].get("bench", {}) or {}).get(bench_key, {}) or {}
		if not isinstance(first_bench, dict):
			first_bench = {}
		if str(override).strip():
			m = str(override).strip()
			return bench_key, m, _pretty_metric_label(m)
		m, disp = _pick_primary_metric(first_bench)
		if m == "unknown":
			raise RuntimeError(f"Could not auto-pick bench metric for '{bench_key}'. Provide --bench_metric_*.")
		return bench_key, m, disp

	bench_key_a, metric_a, disp_a = auto_metric(recs_a, meta_a, str(args.bench_metric_a))
	bench_key_b, metric_b, disp_b = auto_metric(recs_b, meta_b, str(args.bench_metric_b))

	best_a = _best_mtopdiv_layer(recs_a, bench_key=bench_key_a, bench_metric=metric_a, min_common_epochs=int(args.min_common_epochs))
	best_b = _best_mtopdiv_layer(recs_b, bench_key=bench_key_b, bench_metric=metric_b, min_common_epochs=int(args.min_common_epochs))

	os.makedirs(os.path.dirname(os.path.abspath(str(args.out_png))), exist_ok=True)

	is_paper = str(args.style).lower().strip() == "paper"
	show_axis_labels = not bool(is_paper)
	show_legends = True

	plt.rcParams.update(
		{
			"font.family": "DejaVu Sans",
			"text.usetex": False,
			"mathtext.fontset": "dejavusans",
			"font.size": 14.0,
			"axes.labelsize": 20.0,
			"axes.titlesize": 15.0,
			"xtick.labelsize": 16.0,
			"ytick.labelsize": 16.0,
			"legend.fontsize": 12.5,
			"axes.linewidth": 1.8,
			"figure.dpi": 180,
			"savefig.dpi": 300,
		}
	)

	fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.4), sharex="col", gridspec_kw={"height_ratios": [1.0, 1.0]})

	_plot_one(
		(axes[0, 0], axes[1, 0]),
		recs=recs_a,
		meta=meta_a,
		best=best_a,
		bench_metric_label=disp_a,
		show_axis_labels=show_axis_labels,
		show_legends=show_legends,
	)
	_plot_one(
		(axes[0, 1], axes[1, 1]),
		recs=recs_b,
		meta=meta_b,
		best=best_b,
		bench_metric_label=disp_b,
		show_axis_labels=show_axis_labels,
		show_legends=show_legends,
	)

	fig.align_ylabels([axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]])
	fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.18, hspace=0.18)
	fig.savefig(str(args.out_png), bbox_inches="tight")
	plt.close(fig)

	print("[OK] wrote", str(args.out_png))
	print("[Run A] best_layer", best_a.layer, "rho", f"{best_a.rho:.4f}", "n", best_a.n, "bench", f"{best_a.bench_key}.{best_a.bench_metric}")
	print("[Run B] best_layer", best_b.layer, "rho", f"{best_b.rho:.4f}", "n", best_b.n, "bench", f"{best_b.bench_key}.{best_b.bench_metric}")


if __name__ == "__main__":
	main()
