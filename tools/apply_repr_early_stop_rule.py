from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.repr_early_stop_sweep import (
	Signal,
	_build_row_from_stop,
	_series_signal,
	_simulate_multi_plateau,
	_simulate_single_plateau,
	load_early_stop_run_context,
)
from tools.run_experiment import _write_figures


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Apply offline repr early-stop rule to a finished run and redraw figures.")
	ap.add_argument("--run_dir", required=True)
	ap.add_argument(
		"--signals",
		default="",
		help="Comma-separated signals: metric:mode:layer (e.g. mtopdiv_train_val:min:features.7.0).",
	)
	ap.add_argument("--aggregate", default="any", choices=["any", "all"])
	ap.add_argument("--patience", type=int, default=5)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument("--bench_metric_override", default="")
	ap.add_argument(
		"--out_json",
		default="analysis/early_stop_offline.json",
		help="Path relative to run_dir for the written rule result JSON.",
	)
	ap.add_argument("--redraw_figures", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--plot_every", type=int, default=1)
	return ap.parse_args(argv)


def _parse_signals(spec: str) -> List[Tuple[str, str, str]]:
	raw = [x.strip() for x in str(spec).split(",") if x.strip()]
	if not raw:
		raise ValueError("--signals is required (comma-separated metric:mode:layer entries).")
	out: List[Tuple[str, str, str]] = []
	for s in raw:
		parts = s.split(":")
		if len(parts) < 3:
			raise ValueError(f"Bad signal {s!r}. Expected metric:mode:layer.")
		metric = str(parts[0]).strip()
		mode = str(parts[1]).strip()
		layer = ":".join(parts[2:]).strip()
		if not metric or not mode or not layer:
			raise ValueError(f"Bad signal {s!r}. Expected metric:mode:layer.")
		out.append((metric, mode, layer))
	return out


def main(argv: Optional[List[str]] = None) -> None:
	args = _parse_args(argv)
	run_dir = os.path.abspath(str(args.run_dir))
	ctx = load_early_stop_run_context(run_dir, str(args.bench_metric_override).strip())

	signals = _parse_signals(str(args.signals))
	series_by_signal: List[List[Tuple[int, float]]] = []
	modes: List[str] = []
	metrics: List[str] = []
	layers: List[str] = []
	for metric, mode, layer in signals:
		metrics.append(str(metric))
		modes.append(str(mode))
		layers.append(str(layer))
		ser = _series_signal(ctx.recs, sig=Signal(metric=str(metric), mode=str(mode), layer=str(layer)), zero_tol=float(ctx.zero_tol))
		if not ser:
			raise RuntimeError(f"Empty signal series for {metric} @ {layer}")
		series_by_signal.append(ser)

	if len(series_by_signal) == 1:
		stop = _simulate_single_plateau(
			series_by_signal[0],
			mode=str(modes[0]),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
		)
		extra = {"kind": "single", "metric": str(metrics[0]), "mode": str(modes[0]), "layer": str(layers[0])}
	else:
		stop = _simulate_multi_plateau(
			series_by_signal,
			modes=modes,
			aggregate=str(args.aggregate),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
		)
		extra = {
			"kind": "multi",
			"metrics": list(metrics),
			"modes": list(modes),
			"layers": list(layers),
			"aggregate": str(args.aggregate),
		}

	row = _build_row_from_stop(
		ctx,
		stop=stop,
		start_epoch=int(args.start_epoch),
		min_delta=float(args.min_delta),
		patience=int(args.patience),
		extra=extra,
	)

	out_rel = str(args.out_json).strip() or "analysis/early_stop_offline.json"
	out_path = os.path.join(run_dir, out_rel)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"trigger_epoch": int(row["effective_stop_epoch"]),
				"triggered": bool(row["triggered"]),
				"epochs_saved": int(row["epochs_saved"]),
				"gap_rel_pct": float(row["gap_rel_pct"]),
				"gap_abs": float(row["gap_abs"]),
				"bench_metric": str(ctx.pick_bench_metric),
				"rule": extra,
				"row": row,
			},
			f,
			ensure_ascii=False,
			indent=2,
		)

	print(
		f"[OfflineEarlyStop] effective_stop_epoch={int(row['effective_stop_epoch'])} "
		f"triggered={bool(row['triggered'])} epochs_saved={int(row['epochs_saved'])} "
		f"gap_rel_pct={float(row['gap_rel_pct']):.2f}"
	)

	if bool(args.redraw_figures):
		meta_path = os.path.join(run_dir, "meta.json")
		meta: Dict[str, Any] = json.load(open(meta_path, "r", encoding="utf-8"))
		ds = str(meta.get("dataset", "") or "").strip()
		hook_layers = [str(x) for x in ((meta.get("monitor", {}) or {}).get("layer_names", []) or [])]
		_write_figures(run_dir, plot_every=int(args.plot_every), dataset_key=ds, hook_layers=hook_layers)
		print("[OfflineEarlyStop] figures updated under:", os.path.join(run_dir, "figures"))


if __name__ == "__main__":
	main()

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.repr_early_stop_sweep import (
	Signal,
	_build_row_from_stop,
	_series_signal,
	_simulate_multi_plateau,
	_simulate_single_plateau,
	load_early_stop_run_context,
)
from tools.run_experiment import _write_figures


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Apply offline repr early-stop rule to a finished run and redraw figures.")
	ap.add_argument("--run_dir", required=True)
	ap.add_argument(
		"--signals",
		default="",
		help="Comma-separated signals: metric:mode:layer (e.g. mtopdiv_train_val:min:features.7.0).",
	)
	ap.add_argument("--aggregate", default="any", choices=["any", "all"])
	ap.add_argument("--patience", type=int, default=5)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument("--bench_metric_override", default="")
	ap.add_argument(
		"--out_json",
		default="analysis/early_stop_offline.json",
		help="Path relative to run_dir for the written rule result JSON.",
	)
	ap.add_argument("--redraw_figures", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--plot_every", type=int, default=1)
	return ap.parse_args(argv)


def _parse_signals(spec: str) -> List[Tuple[str, str, str]]:
	raw = [x.strip() for x in str(spec).split(",") if x.strip()]
	if not raw:
		raise ValueError("--signals is required (comma-separated metric:mode:layer entries).")
	out: List[Tuple[str, str, str]] = []
	for s in raw:
		parts = s.split(":")
		if len(parts) < 3:
			raise ValueError(f"Bad signal {s!r}. Expected metric:mode:layer.")
		metric = str(parts[0]).strip()
		mode = str(parts[1]).strip()
		layer = ":".join(parts[2:]).strip()
		if not metric or not mode or not layer:
			raise ValueError(f"Bad signal {s!r}. Expected metric:mode:layer.")
		out.append((metric, mode, layer))
	return out


def main(argv: Optional[List[str]] = None) -> None:
	args = _parse_args(argv)
	run_dir = os.path.abspath(str(args.run_dir))
	ctx = load_early_stop_run_context(run_dir, str(args.bench_metric_override).strip())

	signals = _parse_signals(str(args.signals))
	series_by_signal: List[List[Tuple[int, float]]] = []
	modes: List[str] = []
	metrics: List[str] = []
	layers: List[str] = []
	for metric, mode, layer in signals:
		metrics.append(str(metric))
		modes.append(str(mode))
		layers.append(str(layer))
		ser = _series_signal(ctx.recs, sig=Signal(metric=str(metric), mode=str(mode), layer=str(layer)), zero_tol=float(ctx.zero_tol))
		if not ser:
			raise RuntimeError(f"Empty signal series for {metric} @ {layer}")
		series_by_signal.append(ser)

	if len(series_by_signal) == 1:
		stop = _simulate_single_plateau(
			series_by_signal[0],
			mode=str(modes[0]),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
		)
		extra = {"kind": "single", "metric": str(metrics[0]), "mode": str(modes[0]), "layer": str(layers[0])}
	else:
		stop = _simulate_multi_plateau(
			series_by_signal,
			modes=modes,
			aggregate=str(args.aggregate),
			patience=int(args.patience),
			start_epoch=int(args.start_epoch),
			min_delta=float(args.min_delta),
		)
		extra = {"kind": "multi", "metrics": list(metrics), "modes": list(modes), "layers": list(layers), "aggregate": str(args.aggregate)}

	row = _build_row_from_stop(
		ctx,
		stop=stop,
		start_epoch=int(args.start_epoch),
		min_delta=float(args.min_delta),
		patience=int(args.patience),
		extra=extra,
	)

	out_rel = str(args.out_json).strip() or "analysis/early_stop_offline.json"
	out_path = os.path.join(run_dir, out_rel)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"trigger_epoch": int(row["effective_stop_epoch"]),
				"triggered": bool(row["triggered"]),
				"epochs_saved": int(row["epochs_saved"]),
				"gap_rel_pct": float(row["gap_rel_pct"]),
				"gap_abs": float(row["gap_abs"]),
				"bench_metric": str(ctx.pick_bench_metric),
				"rule": extra,
				"row": row,
			},
			f,
			ensure_ascii=False,
			indent=2,
		)

	print(
		f"[OfflineEarlyStop] effective_stop_epoch={int(row['effective_stop_epoch'])} "
		f"triggered={bool(row['triggered'])} epochs_saved={int(row['epochs_saved'])} "
		f"gap_rel_pct={float(row['gap_rel_pct']):.2f}"
	)

	if bool(args.redraw_figures):
		meta_path = os.path.join(run_dir, "meta.json")
		meta: Dict[str, Any] = json.load(open(meta_path, "r", encoding="utf-8"))
		ds = str(meta.get("dataset", "") or "").strip()
		hook_layers = [str(x) for x in ((meta.get("monitor", {}) or {}).get("layer_names", []) or [])]
		_write_figures(run_dir, plot_every=int(args.plot_every), dataset_key=ds, hook_layers=hook_layers)
		print("[OfflineEarlyStop] figures updated under:", os.path.join(run_dir, "figures"))


if __name__ == "__main__":
	main()

