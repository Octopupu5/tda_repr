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
class Signal:
	metric: str  # beta1_L_est | hodge_L_q0_lambda2 | persistent_q1_lambda1 | mtopdiv_train_val
	mode: str  # min | max
	layer: str


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


def _pretty_metric(metric: str) -> str:
	m = str(metric)
	if m == "beta1_L_est":
		return r"$\beta_1(L)$"
	if m == "hodge_L_q0_lambda2":
		return r"$\lambda_2(\Delta_0(L))$"
	if m == "persistent_q1_lambda1":
		return r"$\lambda_1(\Delta_1^{K,L})$"
	if m == "mtopdiv_train_val":
		return r"$\mathrm{MTopDiv}$"
	return m

def _pretty_layer_name(layer: str) -> str:
	s = str(layer).strip()
	if s.startswith("model."):
		s = s[len("model.") :]
	return s

def _pretty_plateau_mode(mode: str) -> str:
	m = str(mode).lower().strip()
	if m == "min":
		return "плато минимума"
	if m == "max":
		return "плато максимума"
	return m

def _pretty_bench_metric(metric: str) -> str:
	m = str(metric).lower().strip()
	if m == "ppl":
		return "Перплексия"
	if m == "loss_assistant_only":
		return "Лосс (assistant-only)"
	if m == "loss":
		return "Лосс"
	return metric


def _bench_mode(metric: str) -> str:
	m = str(metric).lower().strip()
	if m in {"loss", "loss_assistant_only", "ppl"} or m.endswith("loss"):
		return "min"
	return "max"


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


def _extract_signal_value(rec: dict, *, layer: str, metric: str, zero_tol: float) -> Optional[float]:
	layer_obj = (((rec.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {})
	if not isinstance(layer_obj, dict):
		return None
	m = str(metric)
	if m == "beta1_L_est":
		return _safe_float(layer_obj.get("beta1_L_est", None))
	if m == "hodge_L_q0_lambda2":
		return _first_positive(layer_obj.get("hodge_L_q0_smallest", None), zero_tol=float(zero_tol))
	if m == "persistent_q1_lambda1":
		return _first_positive(layer_obj.get("persistent_q1_smallest", None), zero_tol=float(zero_tol))
	if m == "mtopdiv_train_val":
		return _safe_float(layer_obj.get("mtopdiv_train_val", None))
	raise ValueError(f"Unknown signal metric: {metric}")


def _series_signal(recs: Sequence[Dict[str, Any]], *, sig: Signal, zero_tol: float) -> List[Tuple[int, float]]:
	out: List[Tuple[int, float]] = []
	for r in recs:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		v = _extract_signal_value(r, layer=str(sig.layer), metric=str(sig.metric), zero_tol=float(zero_tol))
		if v is None:
			continue
		out.append((int(ep), float(v)))
	out.sort(key=lambda x: x[0])
	return out


def _value_at(series: Sequence[Tuple[int, float]], epoch: int) -> Optional[float]:
	for ep, v in series:
		if int(ep) == int(epoch):
			return float(v)
	return None


def _best_epoch(series: Sequence[Tuple[int, float]], *, mode: str) -> Optional[int]:
	if not series:
		return None
	if str(mode) == "min":
		return int(min(series, key=lambda x: float(x[1]))[0])
	return int(max(series, key=lambda x: float(x[1]))[0])


def _simulate_single_plateau(
	series: Sequence[Tuple[int, float]],
	*,
	mode: str,
	patience: int,
	start_epoch: int,
	min_delta: float,
) -> Optional[int]:
	best: Optional[float] = None
	bad = 0
	for ep, v in series:
		if int(ep) < int(start_epoch):
			continue
		if best is None:
			best = float(v)
			bad = 0
			continue
		if str(mode) == "max":
			improved = float(v) > float(best) + float(min_delta)
		else:
			improved = float(v) < float(best) - float(min_delta)
		if improved:
			best = float(v)
			bad = 0
		else:
			bad += 1
		if int(bad) >= int(patience):
			return int(ep)
	return None


def _simulate_multi_plateau(
	series_by_signal: Sequence[Sequence[Tuple[int, float]]],
	*,
	modes: Sequence[str],
	aggregate: str,
	patience: int,
	start_epoch: int,
	min_delta: float,
) -> Optional[int]:
	maps: List[Dict[int, float]] = [{int(ep): float(v) for ep, v in ser} for ser in series_by_signal]
	common = sorted(set.intersection(*[set(m.keys()) for m in maps])) if maps else []
	if not common:
		return None
	state = [{"best": None, "bad": 0} for _ in maps]
	for ep in common:
		if int(ep) < int(start_epoch):
			continue
		for i, m in enumerate(maps):
			v = m.get(int(ep))
			if v is None or not math.isfinite(float(v)):
				break
			if state[i]["best"] is None:
				state[i]["best"] = float(v)
				state[i]["bad"] = 0
				continue
			best = float(state[i]["best"])
			mode = str(modes[i])
			if mode == "max":
				improved = float(v) > best + float(min_delta)
			else:
				improved = float(v) < best - float(min_delta)
			if improved:
				state[i]["best"] = float(v)
				state[i]["bad"] = 0
			else:
				state[i]["bad"] = int(state[i]["bad"]) + 1

		if str(aggregate).lower() == "any":
			triggered = any(int(st["bad"]) >= int(patience) for st in state)
		else:
			triggered = all(int(st["bad"]) >= int(patience) for st in state)
		if triggered:
			return int(ep)
	return None


def _choose_oracle_layer_for_metric(
	recs: Sequence[Dict[str, Any]],
	*,
	layers: Sequence[str],
	bench_series: Sequence[Tuple[int, float]],
	bench_mode: str,
	metric: str,
	signal_mode: str,
	patience: int,
	start_epoch: int,
	min_delta: float,
	zero_tol: float,
) -> str:
	if not bench_series:
		raise RuntimeError("Empty bench series.")
	last_ep = int(bench_series[-1][0])
	best_ep = _best_epoch(bench_series, mode=str(bench_mode))
	if best_ep is None:
		raise RuntimeError("Could not pick best epoch for bench series.")
	best_val = _value_at(bench_series, int(best_ep))
	if best_val is None:
		raise RuntimeError("Could not read best value for bench series.")

	best_obj = None
	best_layer = None
	for lay in layers:
		sig_ser = _series_signal(recs, sig=Signal(metric=str(metric), mode=str(signal_mode), layer=str(lay)), zero_tol=float(zero_tol))
		if not sig_ser:
			continue
		stop = _simulate_single_plateau(
			sig_ser,
			mode=str(signal_mode),
			patience=int(patience),
			start_epoch=int(start_epoch),
			min_delta=float(min_delta),
		)
		stop_eff = int(stop) if stop is not None else last_ep
		triggered = stop is not None and int(stop_eff) < int(last_ep)
		saved_ep = max(0, int(last_ep) - int(stop_eff))
		q_stop = _value_at(bench_series, int(stop_eff))
		if q_stop is None:
			raise RuntimeError(f"Missing bench value at stop_eff={stop_eff}.")

		if str(bench_mode) == "min":
			gap = max(0.0, float(q_stop) - float(best_val))
		else:
			gap = max(0.0, float(best_val) - float(q_stop))
		drop_rel = max(0.0, float(gap) / max(abs(float(best_val)), 1e-12) * 100.0)
		obj = (1 if triggered else 0, -float(drop_rel), float(saved_ep))
		if best_obj is None or obj > best_obj:
			best_obj = obj
			best_layer = str(lay)

	if best_layer is None:
		raise RuntimeError(f"No valid layers found for metric='{metric}'.")
	return best_layer


def main() -> None:
	ap = argparse.ArgumentParser(description="Render a paper-style early-stopping triplet case-study figure (quality + 3 signals).")
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument(
		"--metrics",
		type=str,
		default="beta1_L_est,hodge_L_q0_lambda2,mtopdiv_train_val",
		help="CSV of 3 signal metrics (supported: beta1_L_est,hodge_L_q0_lambda2,persistent_q1_lambda1,mtopdiv_train_val).",
	)
	ap.add_argument("--modes", type=str, default="min,min,min", help="CSV of 3 modes (min/max) aligned with --metrics.")
	ap.add_argument("--patience", type=int, default=4)
	ap.add_argument("--aggregate", type=str, default="all", choices=["any", "all"])
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument(
		"--layer_policy",
		type=str,
		default="oracle",
		choices=["oracle", "canonical", "manual"],
		help="oracle: pick best layer per signal (single-metric objective); canonical: use canonical pre-head layer for all; manual: pass --layers.",
	)
	ap.add_argument("--layers", type=str, default="", help="CSV of 3 layer names (used when --layer_policy=manual).")
	ap.add_argument("--bench_metric", type=str, default="ppl", help="Validation metric on the top panel (e.g., ppl, loss_assistant_only, loss).")
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--out_pdf", type=str, default="")
	args = ap.parse_args()

	run_dir = os.path.abspath(str(args.run_dir))
	meta_path = os.path.join(run_dir, "meta.json")
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"Missing meta.json: {meta_path}")
	if not os.path.exists(metrics_path):
		raise FileNotFoundError(f"Missing metrics.jsonl: {metrics_path}")

	meta = json.load(open(meta_path, "r", encoding="utf-8"))
	dataset = str(meta.get("dataset", "")).strip()
	bench_key = f"{dataset}-val"
	zero_tol = float((meta.get("monitor", {}) or {}).get("zero_tol", 1e-8) or 1e-8)
	layer_names = (meta.get("monitor", {}) or {}).get("layer_names", []) or []
	if not isinstance(layer_names, list) or not layer_names:
		raise RuntimeError("No monitor.layer_names found in meta.json.")

	recs = _load_epoch_end_records(metrics_path)
	if not recs:
		raise RuntimeError("No epoch_end records found.")

	first_layers = (((recs[0].get("repr", {}) or {}).get("layers", {}) or {}))
	layers_from_log = sorted(first_layers.keys()) if isinstance(first_layers, dict) else []
	log_set = set(layers_from_log)
	mon_layers = [str(x) for x in layer_names if str(x) in log_set]
	if not mon_layers:
		mon_layers = [str(x) for x in layers_from_log]
	if not mon_layers:
		raise RuntimeError("No monitored layers found in epoch_end records.")

	bench_metric = str(args.bench_metric).strip()
	bench_mode = _bench_mode(bench_metric)
	main_series = _series_bench(recs, bench_key=bench_key, metric=bench_metric)
	if not main_series:
		raise RuntimeError(f"No bench series for '{bench_key}.{bench_metric}'.")

	metrics = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
	modes = [x.strip().lower() for x in str(args.modes).split(",") if x.strip()]
	if len(metrics) != 3 or len(modes) != 3:
		raise ValueError("--metrics and --modes must each contain exactly 3 comma-separated items.")
	if any(m not in {"min", "max"} for m in modes):
		raise ValueError(f"Invalid --modes: {modes}. Expected only min/max.")

	lp = str(args.layer_policy).lower().strip()
	if lp == "manual":
		layers = [x.strip() for x in str(args.layers).split(",") if x.strip()]
		if len(layers) != 3:
			raise ValueError("--layers must contain exactly 3 comma-separated layer names when --layer_policy=manual.")
	elif lp == "canonical":
		preferred = ["avgpool", "pre_classifier"]
		canon = None
		for p in preferred:
			if p in mon_layers:
				canon = p
				break
		if canon is None:
			canon = mon_layers[-1]
		layers = [str(canon)] * 3
	else:
		layers = []
		for met, mode in zip(metrics, modes):
			layers.append(
				_choose_oracle_layer_for_metric(
					recs,
					layers=mon_layers,
					bench_series=main_series,
					bench_mode=str(bench_mode),
					metric=str(met),
					signal_mode=str(mode),
					patience=int(args.patience),
					start_epoch=int(args.start_epoch),
					min_delta=float(args.min_delta),
					zero_tol=float(zero_tol),
				)
			)

	signals = [Signal(metric=str(m), mode=str(md), layer=str(lay)) for m, md, lay in zip(metrics, modes, layers)]
	signal_series = [_series_signal(recs, sig=s, zero_tol=float(zero_tol)) for s in signals]
	if any(not s for s in signal_series):
		raise RuntimeError("One of the signal series is empty; cannot render figure.")

	trig_single = [
		_simulate_single_plateau(
			ser,
			mode=str(sig.mode),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
		)
		for sig, ser in zip(signals, signal_series)
	]
	stop_ep = _simulate_multi_plateau(
		signal_series,
		modes=[str(s.mode) for s in signals],
		aggregate=str(args.aggregate),
		patience=int(args.patience),
		start_epoch=int(args.start_epoch),
		min_delta=float(args.min_delta),
	)

	last_ep = int(main_series[-1][0])
	stop_eff = int(stop_ep) if stop_ep is not None else last_ep
	best_ep = _best_epoch(main_series, mode=str(bench_mode))

	plt.rcParams.update(
		{
			"font.family": "DejaVu Sans",
			"text.usetex": False,
			"mathtext.fontset": "dejavusans",
			"font.size": 14.0,
			"axes.labelsize": 22.0,
			"axes.titlesize": 16.0,
			"xtick.labelsize": 18.0,
			"ytick.labelsize": 18.0,
			"legend.fontsize": 13.0,
			"axes.linewidth": 1.8,
			"figure.dpi": 180,
			"savefig.dpi": 300,
		}
	)

	# 2x2 layout: quality + 3 signals.
	fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
	ax_grid = axes.reshape(-1)

	def _beautify(ax):
		ax.grid(True, which="major", color="#AFAFAF", alpha=0.90, linestyle="-", linewidth=1.2)
		ax.set_axisbelow(True)
		for sp in ax.spines.values():
			sp.set_linewidth(1.8)
		ax.tick_params(direction="out", length=7, width=1.8)
		ax.xaxis.set_major_locator(MultipleLocator(2))

	# Top-left: validation metric
	ax0 = ax_grid[0]
	_beautify(ax0)
	xm = [e for e, _ in main_series]
	ym = [v for _, v in main_series]
	ax0.plot(
		xm,
		ym,
		color="#d62728",
		linewidth=3.2,
		marker="o",
		markersize=6.0,
		label=_pretty_bench_metric(bench_metric),
	)

	vs = _value_at(main_series, int(stop_eff))
	if vs is not None:
		ax0.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85)
		ax0.scatter(
			[int(stop_eff)],
			[float(vs)],
			s=190,
			color="#1f9d8a",
			edgecolors="white",
			linewidths=1.6,
			zorder=8,
			label="Ранняя остановка",
		)

	if best_ep is not None:
		vb = _value_at(main_series, int(best_ep))
		if vb is not None:
			best_x = float(best_ep)
			best_y = float(vb)

			# If best epoch coincides with early-stop epoch, shift slightly left
			# so the black marker remains visible.
			if int(best_ep) == int(stop_eff):
				best_x = float(best_ep)

			ax0.scatter(
				[best_x],
				[best_y],
				s=320,
				facecolors="none",
				edgecolors="#111111",
				linewidths=3.0,
				zorder=10,
				label="Эмпирически лучший",
			)


	ax0.set_ylabel("Валидация")
	ax0.yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax0.legend(loc="upper right", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	# Remaining 3 panels: signals.
	for i in range(3):
		ax = ax_grid[i + 1]
		_beautify(ax)
		sig = signals[i]
		ser = signal_series[i]
		xs = [e for e, _ in ser]
		ys = [v for _, v in ser]
		label = f"{_pretty_layer_name(sig.layer)} ({_pretty_plateau_mode(sig.mode)}, p={int(args.patience)})"
		ax.plot(xs, ys, color="#6a3d9a", linewidth=3.0, marker="o", markersize=5.6, label=label)

		tr = trig_single[i]
		if tr is not None:
			vt = _value_at(ser, int(tr))
			if vt is not None:
				ax.scatter([int(tr)], [float(vt)], s=95, color="#ff7f0e", edgecolors="white", linewidths=1.0, zorder=7)

		vsig = _value_at(ser, int(stop_eff))
		if vsig is not None:
			ax.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85)
			ax.scatter(
				[int(stop_eff)],
				[float(vsig)],
				s=120,
				color="#1f9d8a",
				edgecolors="white",
				linewidths=1.2,
				zorder=8,
			)

		ax.set_ylabel(_pretty_metric(sig.metric))
		ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.legend(loc="upper left", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	# X label only on bottom row.
	ax_grid[2].set_xlabel("Эпоха", fontweight="bold")
	ax_grid[3].set_xlabel("Эпоха", fontweight="bold")

	fig.align_ylabels(ax_grid)
	fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.10, wspace=0.22, hspace=0.22)
	os.makedirs(os.path.dirname(os.path.abspath(str(args.out_png))), exist_ok=True)
	fig.savefig(str(args.out_png), bbox_inches="tight")
	if str(args.out_pdf).strip():
		os.makedirs(os.path.dirname(os.path.abspath(str(args.out_pdf))), exist_ok=True)
		fig.savefig(str(args.out_pdf), bbox_inches="tight")
	plt.close(fig)

	print("[OK] bench_key =", bench_key)
	print("[OK] bench_metric =", bench_metric, "mode=", bench_mode)
	print("[OK] best_epoch =", best_ep)
	print("[OK] stop_epoch_trigger =", stop_ep)
	print("[OK] stop_epoch_effective =", stop_eff)
	for s, tr in zip(signals, trig_single):
		print(f"[OK] signal {s.metric} layer={s.layer} mode={s.mode} single_trigger={tr}")


if __name__ == "__main__":
	main()
