from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
# Panels 2..4: descriptors.

# Avoid slow/locked global Matplotlib caches.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MPLCONFIGDIR = os.path.join(_ROOT, ".mplconfig_render")
_XDG_CACHE_HOME = os.path.join(_ROOT, ".cache_render")
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.makedirs(_XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_HOME)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


@dataclass(frozen=True)
class LayerTriplet:
	early: str
	mid: str
	deep: str

	@property
	def as_list(self) -> List[str]:
		return [str(self.early), str(self.mid), str(self.deep)]


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	return float(v) if math.isfinite(float(v)) else None


def _first_positive(vals: Any, *, zero_tol: float) -> Optional[float]:
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


def _pretty_quality(metric: str) -> str:
	m = str(metric).lower().strip()
	if m == "f1_macro":
		return "Макро-F1"
	if m == "accuracy":
		return "Доля правильных ответов"
	if m == "ppl":
		return "Перплексия"
	if m == "loss_assistant_only":
		return "Лосс (assistant-only)"
	if m == "loss":
		return "Лосс"
	return metric


def _pretty_desc(metric: str) -> str:
	m = str(metric)
	if m == "beta1_persistent_est":
		return r"$\beta_1^{K,L}$"
	if m == "hodge_L_q0_lambda2":
		return r"$\lambda_2(\Delta_0(L))$"
	if m == "persistent_q1_lambda1":
		return r"$\lambda_1(\Delta_1^{K,L})$"
	return m


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


def _bench_series(recs: Sequence[Dict[str, Any]], bench_key: str, metric: str) -> List[Tuple[int, float]]:
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


def _extract_layer_metric(layer_obj: Dict[str, Any], metric: str, *, zero_tol: float) -> Optional[float]:
	m = str(metric)
	if m in {"beta1_persistent_est", "beta1_L_est", "mtopdiv_train_val"}:
		return _safe_float(layer_obj.get(m, None))
	if m == "hodge_L_q0_lambda2":
		return _first_positive(layer_obj.get("hodge_L_q0_smallest", None), zero_tol=float(zero_tol))
	if m == "persistent_q1_lambda1":
		return _first_positive(layer_obj.get("persistent_q1_smallest", None), zero_tol=float(zero_tol))
	raise ValueError(f"Unsupported descriptor metric: {metric}")


def _layer_series(recs: Sequence[Dict[str, Any]], *, layer: str, metric: str, zero_tol: float) -> List[Tuple[int, float]]:
	out: List[Tuple[int, float]] = []
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		lo = (((r.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, None))
		if not isinstance(lo, dict):
			continue
		v = _extract_layer_metric(lo, metric=str(metric), zero_tol=float(zero_tol))
		if v is None:
			continue
		out.append((int(ep), float(v)))
	out.sort(key=lambda x: x[0])
	return out


def _pick_quality_metric(recs: Sequence[Dict[str, Any]], bench_key: str) -> Tuple[str, List[Tuple[int, float]]]:
	# Prefer Macro-F1, then Accuracy.
	f1 = _bench_series(recs, bench_key=bench_key, metric="f1_macro")
	acc = _bench_series(recs, bench_key=bench_key, metric="accuracy")
	if f1:
		return "f1_macro", f1
	if acc:
		return "accuracy", acc
	raise RuntimeError(f"No classification validation series found in bench '{bench_key}'.")


def render_figure(
	*,
	run_dir: str,
	layers: LayerTriplet,
	out_png: str,
	quality_metric_override: str = "",
	descriptors: Tuple[str, str, str] = ("beta1_persistent_est", "hodge_L_q0_lambda2", "persistent_q1_lambda1"),
) -> None:
	run_dir = os.path.abspath(str(run_dir))
	meta = json.load(open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8"))
	dataset = str(meta.get("dataset", "")).strip()
	bench_key = f"{dataset}-val"
	zero_tol = float((meta.get("monitor", {}) or {}).get("zero_tol", 1e-8) or 1e-8)
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	recs = _load_epoch_end_records(metrics_path)
	if not recs:
		raise RuntimeError(f"No epoch_end records in {metrics_path}.")

	# Validation series.
	if str(quality_metric_override).strip():
		qm = str(quality_metric_override).strip()
		qs = _bench_series(recs, bench_key=bench_key, metric=qm)
		if not qs:
			raise RuntimeError(f"No bench series for '{bench_key}.{qm}'.")
		quality_metric = qm
		quality_series = qs
	else:
		quality_metric, quality_series = _pick_quality_metric(recs, bench_key=bench_key)
	# Also plot the other common metric if present (F1+Acc).
	acc_series = _bench_series(recs, bench_key=bench_key, metric="accuracy")
	f1_series = _bench_series(recs, bench_key=bench_key, metric="f1_macro")

	# Descriptor series per layer.
	desc_to_series: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
	for desc in descriptors:
		desc_to_series[str(desc)] = {lay: _layer_series(recs, layer=lay, metric=str(desc), zero_tol=float(zero_tol)) for lay in layers.as_list}

	# Styling close to the paper case-study figures.
	plt.rcParams.update(
		{
			"font.family": "DejaVu Sans",
			"text.usetex": False,
			"mathtext.fontset": "dejavusans",
			"font.size": 14.0,
			"axes.labelsize": 18.0,
			"axes.titlesize": 14.0,
			"xtick.labelsize": 14.0,
			"ytick.labelsize": 14.0,
			"legend.fontsize": 12.5,
			"axes.linewidth": 1.6,
			"figure.dpi": 180,
			"savefig.dpi": 300,
		}
	)

	fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
	ax = axes.reshape(-1)

	def _beautify(a):
		a.grid(True, which="major", color="#AFAFAF", alpha=0.90, linestyle="-", linewidth=1.1)
		a.set_axisbelow(True)
		for sp in a.spines.values():
			sp.set_linewidth(1.6)
		a.tick_params(direction="out", length=6, width=1.6)
		a.xaxis.set_major_locator(MultipleLocator(2))

	# Panel 1: validation.
	_beautify(ax[0])
	if f1_series:
		x = [e for e, _ in f1_series]
		y = [100.0 * v for _, v in f1_series]
		ax[0].plot(x, y, color="#d62728", linewidth=3.0, marker="o", markersize=5.5, label="Макро-F1")
	if acc_series:
		x = [e for e, _ in acc_series]
		y = [100.0 * v for _, v in acc_series]
		ax[0].plot(
			x,
			y,
			color="#1f77b4",
			linewidth=2.6,
			linestyle=":",
			marker="x",
			markersize=6.5,
			markeredgewidth=2.0,
			alpha=0.95,
			label="Доля правильных ответов",
		)
	# If only one metric exists, still label it.
	if not f1_series and not acc_series:
		x = [e for e, _ in quality_series]
		y = [v for _, v in quality_series]
		ax[0].plot(x, y, color="#d62728", linewidth=3.0, marker="o", markersize=5.5, label=_pretty_quality(quality_metric))
	ax[0].set_ylabel("Валидация (%)" if (f1_series or acc_series) else "Валидация")
	ax[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax[0].legend(loc="lower right", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	# Panels 2..4: descriptors.
	colors = ["#2ca02c", "#ff7f0e", "#9467bd"]  # early/mid/deep
	labels_ru = ["ранний", "промежуточный", "глубокий"]
	for j, desc in enumerate(descriptors):
		a = ax[j + 1]
		_beautify(a)
		for c, lay, tag in zip(colors, layers.as_list, labels_ru):
			ser = desc_to_series[str(desc)].get(str(lay), [])
			if not ser:
				continue
			x = [e for e, _ in ser]
			y = [v for _, v in ser]
			a.plot(x, y, color=c, linewidth=2.7, marker="o", markersize=4.8, alpha=0.95, label=f"{tag}: {lay.replace("distilbert.transformer.", "")}")
		a.set_ylabel(_pretty_desc(desc))
		a.yaxis.set_major_locator(MaxNLocator(nbins=6))
		if j == 2:  # показать легенду слоёв только один раз
			a.legend(loc="best", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")


	ax[2].set_xlabel("Эпоха", fontweight="bold")
	ax[3].set_xlabel("Эпоха", fontweight="bold")

	fig.align_ylabels(ax)
	fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.10, wspace=0.22, hspace=0.22)

	os.makedirs(os.path.dirname(os.path.abspath(str(out_png))), exist_ok=True)
	fig.savefig(str(out_png), bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	ap = argparse.ArgumentParser(description="Render 2x2 layerwise descriptor dynamics figure.")
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--layers", type=str, required=True, help="CSV: early,mid,deep layer names.")
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--quality_metric", type=str, default="", help="Optional override (e.g., f1_macro or accuracy).")
	args = ap.parse_args()

	parts = [x.strip() for x in str(args.layers).split(",") if x.strip()]
	if len(parts) != 3:
		raise ValueError("--layers must contain exactly 3 comma-separated layer names.")
	layers = LayerTriplet(early=parts[0], mid=parts[1], deep=parts[2])

	render_figure(
		run_dir=str(args.run_dir),
		layers=layers,
		out_png=str(args.out_png),
		quality_metric_override=str(args.quality_metric),
	)


if __name__ == "__main__":
	main()
