from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tools.figures.i18n import I18N
from tools.repr_early_stop_sweep import (
	Signal,
	_best_epoch,
	_choose_oracle_layer_for_metric,
	_load_epoch_end_records,
	_series_bench,
	_series_signal,
	_value_at,
	evaluate_ensemble3_rule_on_run,
)


def _bench_mode(metric: str) -> str:
	m = str(metric).lower().strip()
	return "min" if m in {"loss", "loss_assistant_only", "ppl"} or m.endswith("loss") else "max"


def _pretty_metric(metric: str) -> str:
	m = str(metric)
	return {
		"beta1_L_est": r"$\beta_1(L)$",
		"beta1_persistent_est": r"$\beta_1^{K,L}$",
		"hodge_L_q0_lambda2": r"$\lambda_2(\Delta_0(L))$",
		"persistent_q1_lambda1": r"$\lambda_1(\Delta_1^{K,L})$",
		"mtopdiv_train_val": r"$\mathrm{MTopDiv}$",
	}.get(m, m)


def _csv3(s: str) -> list[str]:
	out = [x.strip() for x in str(s).split(",") if x.strip()]
	if len(out) != 3:
		raise ValueError("Expected exactly 3 comma-separated values.")
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description="Render an early-stopping triplet figure (validation + 3 layer signals).")
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--metrics", type=str, required=True, help="CSV: m1,m2,m3")
	ap.add_argument("--modes", type=str, required=True, help="CSV: max or min per metric")
	ap.add_argument("--aggregate", type=str, default="all", choices=["all", "any"])
	ap.add_argument("--patience", type=int, required=True)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument("--bench_metric", type=str, default="ppl")
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--out_pdf", type=str, default="")
	ap.add_argument("--lang", type=str, default="en", choices=["en", "ru"])
	args = ap.parse_args()

	i18n = I18N(lang=str(args.lang))
	run_dir = os.path.abspath(str(args.run_dir))
	meta_path = os.path.join(run_dir, "meta.json")
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not os.path.isfile(meta_path):
		raise FileNotFoundError(meta_path)
	if not os.path.isfile(metrics_path):
		raise FileNotFoundError(metrics_path)

	with open(meta_path, "r", encoding="utf-8") as f:
		meta = json.load(f)
	dataset = str(meta.get("dataset", "") or "")
	bench_key = f"{dataset}-val"
	zero_tol = float((meta.get("monitor", {}) or {}).get("zero_tol", 1e-8) or 1e-8)

	recs = _load_epoch_end_records(metrics_path)
	if not recs:
		raise RuntimeError("No epoch_end records found.")

	bench_metric = str(args.bench_metric).strip() or "ppl"
	bench_mode = _bench_mode(bench_metric)
	main_series = _series_bench(recs, bench_key=bench_key, metric=bench_metric)
	if not main_series:
		raise RuntimeError(f"No validation series found in bench '{bench_key}' for metric '{bench_metric}'.")

	metrics = _csv3(args.metrics)
	modes = _csv3(args.modes)

	layer_names = (meta.get("monitor", {}) or {}).get("layer_names", None)
	if not isinstance(layer_names, list) or not layer_names:
		layer_names = []
		for r in recs:
			lb = (((r.get("repr", {}) or {}).get("layers", {})) or {})
			if isinstance(lb, dict) and lb:
				layer_names = list(lb.keys())
				break
	if not layer_names:
		raise RuntimeError("Could not infer monitor layer_names for oracle selection.")

	layers = [
		_choose_oracle_layer_for_metric(
			recs,
			layers=[str(x) for x in layer_names],
			bench_series=main_series,
			bench_mode=str(bench_mode),
			metric=str(m),
			signal_mode=str(mo),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
			zero_tol=float(zero_tol),
		)
		for m, mo in zip(metrics, modes)
	]

	signals = [Signal(metric=str(m), mode=str(mo), layer=str(l)) for m, mo, l in zip(metrics, modes, layers)]
	signal_series = [_series_signal(recs, sig=s, zero_tol=float(zero_tol)) for s in signals]
	if any(not s for s in signal_series):
		raise RuntimeError("Empty signal series for at least one selected signal.")

	row = evaluate_ensemble3_rule_on_run(
		run_dir,
		metrics=list(metrics),
		modes=list(modes),
		layers=list(layers),
		aggregate=str(args.aggregate),
		patience=int(args.patience),
		start_epoch=int(args.start_epoch),
		min_delta=float(args.min_delta),
		bench_metric_override=str(bench_metric),
	)

	last_ep = int(main_series[-1][0])
	stop_eff = int(row.get("effective_stop_epoch", last_ep))
	best_ep = _best_epoch(main_series, mode=str(bench_mode))

	os.makedirs(os.path.dirname(os.path.abspath(str(args.out_png))), exist_ok=True)
	if str(args.out_pdf).strip():
		os.makedirs(os.path.dirname(os.path.abspath(str(args.out_pdf))), exist_ok=True)

	fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True)
	ax_grid = axes.reshape(-1)

	ax0 = ax_grid[0]
	xm = [e for e, _ in main_series]
	ym = [v for _, v in main_series]
	ax0.plot(xm, ym, color="#d62728", linewidth=2.2, label=i18n.bench_metric_label(bench_metric))

	vs = _value_at(main_series, int(stop_eff))
	if vs is not None:
		ax0.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.4)
		ax0.scatter([int(stop_eff)], [float(vs)], s=80, color="#1f9d8a", zorder=5, label=i18n.early_stop())

	if best_ep is not None:
		vb = _value_at(main_series, int(best_ep))
		if vb is not None:
			ax0.scatter([int(best_ep)], [float(vb)], s=140, facecolors="none", edgecolors="black", linewidths=2.0, zorder=6, label=i18n.empirical_best())

	ax0.set_ylabel(i18n.validation_value())
	ax0.yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax0.legend(loc="best")

	for i, (sig, ser) in enumerate(zip(signals, signal_series)):
		ax = ax_grid[i + 1]
		xs = [e for e, _ in ser]
		ys = [v for _, v in ser]
		ax.plot(xs, ys, color="#6a3d9a", linewidth=2.2, label=f"{sig.layer}")
		vsig = _value_at(ser, int(stop_eff))
		if vsig is not None:
			ax.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.2)
			ax.scatter([int(stop_eff)], [float(vsig)], s=60, color="#1f9d8a", zorder=5)
		ax.set_ylabel(_pretty_metric(sig.metric))
		ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.legend(loc="best")

	ax_grid[2].set_xlabel(i18n.epoch())
	ax_grid[3].set_xlabel(i18n.epoch())
	fig.tight_layout()
	fig.savefig(str(args.out_png), bbox_inches="tight")
	if str(args.out_pdf).strip():
		fig.savefig(str(args.out_pdf), bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	main()
