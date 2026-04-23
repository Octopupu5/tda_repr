from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from tools.figures.i18n import I18N
from tools.repr_early_stop_sweep import Signal, _best_epoch, _load_epoch_end_records, _series_bench, _series_signal, _value_at
from tools.repr_early_stop_sweep import evaluate_single_signal_rule_on_run


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


def _apply_paper_style() -> None:
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


def _beautify(ax) -> None:
	ax.grid(True, which="major", color="#AFAFAF", alpha=0.90, linestyle="-", linewidth=1.1)
	ax.set_axisbelow(True)
	for sp in ax.spines.values():
		sp.set_linewidth(1.6)
	ax.tick_params(direction="out", length=6, width=1.6)
	ax.xaxis.set_major_locator(MultipleLocator(2))


def main() -> None:
	ap = argparse.ArgumentParser(description="Render an early-stopping case-study figure (validation + signal).")
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
		help="Optional override for the validation metric in the top panel (e.g., ppl, loss).",
	)
	ap.add_argument("--out_png", type=str, required=True)
	ap.add_argument("--out_pdf", type=str, default="")
	ap.add_argument("--title", type=str, default="")
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

	bench_metric = str(args.bench_metric).strip()
	if bench_metric:
		main_series = _series_bench(recs, bench_key=bench_key, metric=bench_metric)
		main_name = i18n.bench_metric_label(bench_metric)
		bench_mode = _bench_mode(bench_metric)
		s_acc = []
		s_f1 = []
	else:
		s_f1 = _series_bench(recs, bench_key=bench_key, metric="f1_macro")
		s_acc = _series_bench(recs, bench_key=bench_key, metric="accuracy")
		main_series = s_f1 if s_f1 else s_acc
		main_name = i18n.macro_f1() if s_f1 else i18n.accuracy()
		bench_metric = "f1_macro" if s_f1 else "accuracy"
		bench_mode = "max"
	if not main_series:
		raise RuntimeError(f"No validation series found in bench '{bench_key}'.")

	sig = Signal(metric=str(args.metric), mode=str(args.mode), layer=str(args.layer))
	s_sig = _series_signal(recs, sig=sig, zero_tol=zero_tol)
	if not s_sig:
		raise RuntimeError(f"No signal series for layer={sig.layer}, metric={sig.metric}.")

	row = evaluate_single_signal_rule_on_run(
		run_dir,
		metric=str(args.metric),
		mode=str(args.mode),
		layer=str(args.layer),
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

	_apply_paper_style()
	fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
	_beautify(ax0)
	_beautify(ax1)

	xm = [e for e, _ in main_series]
	if bench_mode == "max" and bench_metric in {"accuracy", "f1_macro"}:
		ym = [100.0 * v for _, v in main_series]
		ax0.set_ylabel(i18n.validation_pct())
	else:
		ym = [v for _, v in main_series]
		ax0.set_ylabel(i18n.validation_value())
	ax0.plot(xm, ym, color="#d62728", linewidth=3.0, marker="o", markersize=5.5, label=main_name)
	if (not str(args.bench_metric).strip()) and s_f1 and s_acc:
		xa = [e for e, _ in s_acc]
		ya = [100.0 * v for _, v in s_acc]
		ax0.plot(
			xa,
			ya,
			color="#1f77b4",
			linewidth=2.6,
			linestyle=":",
			marker="x",
			markersize=6.5,
			markeredgewidth=2.0,
			alpha=0.95,
			label=i18n.accuracy(),
		)

	vs = _value_at(main_series, int(stop_eff))
	if vs is not None:
		yv = float(vs) * (100.0 if bench_mode == "max" and bench_metric in {"accuracy", "f1_macro"} else 1.0)
		ax0.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.4)
		ax0.scatter([int(stop_eff)], [float(yv)], s=80, color="#1f9d8a", zorder=5, label=i18n.early_stop())

	if best_ep is not None:
		vb = _value_at(main_series, int(best_ep))
		if vb is not None:
			yb = float(vb) * (100.0 if bench_mode == "max" and bench_metric in {"accuracy", "f1_macro"} else 1.0)
			ax0.scatter([int(best_ep)], [float(yb)], s=140, facecolors="none", edgecolors="black", linewidths=2.0, zorder=6, label=i18n.empirical_best())

	ax0.yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax0.legend(loc="lower right", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	xs = [e for e, _ in s_sig]
	ys = [v for _, v in s_sig]
	ax1.plot(xs, ys, color="#6a3d9a", linewidth=3.0, marker="o", markersize=5.0, label=_pretty_metric(str(args.metric)))
	vsig = _value_at(s_sig, int(stop_eff))
	if vsig is not None:
		ax1.axvline(int(stop_eff), color="#d62728", linestyle="--", linewidth=1.4)
		ax1.scatter([int(stop_eff)], [float(vsig)], s=60, color="#1f9d8a", zorder=5)

	ax1.set_ylabel(_pretty_metric(str(args.metric)))
	ax1.set_xlabel(i18n.epoch(), fontweight="bold")
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
	ax1.legend(loc="best", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#BBBBBB")

	fig.tight_layout()
	fig.savefig(str(args.out_png), bbox_inches="tight")
	if str(args.out_pdf).strip():
		fig.savefig(str(args.out_pdf), bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	main()
