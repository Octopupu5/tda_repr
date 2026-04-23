from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class Signal:
	metric: str
	mode: str
	layer: str


def _extract_signal_value(rec: dict, *, layer: str, metric: str, zero_tol: float) -> Optional[float]:
	layer_obj = (((rec.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {})
	if not isinstance(layer_obj, dict):
		return None
	m = str(metric)
	if m == "beta1_L_est":
		return _safe_float(layer_obj.get("beta1_L_est", None))
	if m == "beta1_persistent_est":
		return _safe_float(layer_obj.get("beta1_persistent_est", None))
	if m == "hodge_L_q0_lambda2":
		return _first_positive(layer_obj.get("hodge_L_q0_smallest", None), zero_tol=float(zero_tol))
	if m == "persistent_q1_lambda1":
		return _first_positive(layer_obj.get("persistent_q1_smallest", None), zero_tol=float(zero_tol))
	if m == "mtopdiv_train_val":
		return _safe_float(layer_obj.get("mtopdiv_train_val", None))
	raise ValueError(f"Unsupported metric: {metric}")


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
	if not series_by_signal:
		return None
	maps: List[Dict[int, float]] = [{int(ep): float(v) for ep, v in ser} for ser in series_by_signal]
	common = sorted(set.intersection(*[set(m.keys()) for m in maps]))
	if not common:
		return None
	state = [{"best": None, "bad": 0} for _ in maps]
	for ep in common:
		if int(ep) < int(start_epoch):
			continue
		did_break = False
		for i, m in enumerate(maps):
			v = m.get(int(ep))
			if v is None or not math.isfinite(float(v)):
				did_break = True
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
		if did_break:
			continue

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
) -> Optional[str]:
	if not bench_series:
		return None
	last_ep = int(bench_series[-1][0])
	best_ep = _best_epoch(bench_series, mode=str(bench_mode))
	if best_ep is None:
		return None
	best_val = _value_at(bench_series, int(best_ep))
	if best_val is None:
		return None

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
			continue

		if str(bench_mode) == "min":
			gap = max(0.0, float(q_stop) - float(best_val))
		else:
			gap = max(0.0, float(best_val) - float(q_stop))
		drop_rel = max(0.0, float(gap) / max(abs(float(best_val)), 1e-12) * 100.0)
		obj = (1 if triggered else 0, -float(drop_rel), float(saved_ep))
		if best_obj is None or obj > best_obj:
			best_obj = obj
			best_layer = str(lay)
	return best_layer


def _bench_mode_from_metric(metric: str) -> str:
	m = str(metric).lower().strip()
	if m in {"loss", "loss_assistant_only", "ppl"} or m.endswith("loss"):
		return "min"
	return "max"


@dataclass(frozen=True)
class EarlyStopRunContext:
	run_dir_abs: str
	recs: List[Dict[str, Any]]
	bench_key: str
	pick_bench_metric: str
	bench_mode: str
	bench_series: List[Tuple[int, float]]
	last_ep: int
	best_ep: int
	best_val: float
	zero_tol: float
	mon_layers: List[str]


def load_early_stop_run_context(run_dir: str, bench_metric_override: str = "") -> EarlyStopRunContext:
	run_dir_abs = os.path.abspath(run_dir)
	meta_path = os.path.join(run_dir_abs, "meta.json")
	mp = os.path.join(run_dir_abs, "metrics.jsonl")
	if not os.path.isfile(meta_path) or not os.path.isfile(mp):
		raise FileNotFoundError(f"Missing meta or metrics.jsonl under {run_dir_abs}")
	meta = json.load(open(meta_path, "r", encoding="utf-8"))
	ds = str(meta.get("dataset", "") or "").strip()
	bench_key = f"{ds}-val"
	zero_tol = float((meta.get("monitor", {}) or {}).get("zero_tol", 1e-8) or 1e-8)
	layer_names = (meta.get("monitor", {}) or {}).get("layer_names", []) or []
	recs = _load_epoch_end_records(mp)
	if not recs:
		raise RuntimeError("No epoch_end records.")
	first_layers = (((recs[0].get("repr", {}) or {}).get("layers", {}) or {}))
	layers_log = sorted(first_layers.keys()) if isinstance(first_layers, dict) else []
	log_set = set(layers_log)
	mon_layers = [str(x) for x in layer_names if str(x) in log_set]
	if not mon_layers:
		mon_layers = [str(x) for x in layers_log]
	if not mon_layers:
		raise RuntimeError("No usable layers.")

	pick_bench_metric = bench_metric_override or str(meta.get("_paper_bench_metric", "") or "").strip()
	if not pick_bench_metric:
		b0 = (recs[0].get("bench", {}) or {}).get(bench_key, {}) or {}
		if isinstance(b0, dict):
			for cand in ("f1_macro", "accuracy"):
				if cand in b0:
					pick_bench_metric = cand
					break
		if not pick_bench_metric:
			if isinstance(b0, dict):
				for cand in ("ppl", "loss", "loss_assistant_only"):
					if cand in b0:
						pick_bench_metric = cand
						break
	if not pick_bench_metric:
		raise RuntimeError("Could not infer bench metric.")

	bench_mode = _bench_mode_from_metric(pick_bench_metric)
	bench_series = _series_bench(recs, bench_key=bench_key, metric=str(pick_bench_metric))
	if not bench_series:
		raise RuntimeError(f"No bench series for {bench_key}.{pick_bench_metric}")

	last_ep = int(bench_series[-1][0])
	best_ep = _best_epoch(bench_series, mode=str(bench_mode))
	if best_ep is None:
		raise RuntimeError("Could not determine best validation epoch.")
	best_val = float(_value_at(bench_series, int(best_ep)) or 0.0)

	return EarlyStopRunContext(
		run_dir_abs=run_dir_abs,
		recs=recs,
		bench_key=bench_key,
		pick_bench_metric=str(pick_bench_metric),
		bench_mode=str(bench_mode),
		bench_series=bench_series,
		last_ep=last_ep,
		best_ep=int(best_ep),
		best_val=best_val,
		zero_tol=float(zero_tol),
		mon_layers=mon_layers,
	)


def _attach_stop_summary(row: Dict[str, Any], *, last_ep: int, stop: Optional[int]) -> None:
	"""Annotate a rule row: whether the plateau rule fired, and epochs not trained past the stop."""
	row["triggered"] = stop is not None
	row["epochs_saved"] = max(0, int(last_ep) - int(row["effective_stop_epoch"]))


def sweep_run_directory(
	run_dir: str,
	*,
	start_epoch: int,
	min_delta: float,
	patiences: Iterable[int],
	bench_metric_override: str = "",
	emit_full_grid: bool = False,
	include_ensembles: bool = True,
) -> Dict[str, Any]:
	ctx = load_early_stop_run_context(run_dir, bench_metric_override)
	run_dir_abs = ctx.run_dir_abs
	recs = ctx.recs
	bench_key = ctx.bench_key
	pick_bench_metric = ctx.pick_bench_metric
	bench_mode = ctx.bench_mode
	bench_series = ctx.bench_series
	last_ep = ctx.last_ep
	best_ep = ctx.best_ep
	best_val = ctx.best_val
	zero_tol = ctx.zero_tol
	mon_layers = ctx.mon_layers

	signal_metrics = [
		"beta1_L_est",
		"beta1_persistent_est",
		"hodge_L_q0_lambda2",
		"persistent_q1_lambda1",
		"mtopdiv_train_val",
	]

	grid_rows: List[Dict[str, Any]] = []
	ranked_candidates: List[Dict[str, Any]] = []

	for patience in sorted(set(int(x) for x in patiences)):
		for metric in signal_metrics:
			for mode in ("min", "max"):
				for layer in mon_layers:
					try:
						ser = _series_signal(
							recs, sig=Signal(metric=str(metric), mode=str(mode), layer=str(layer)), zero_tol=float(zero_tol)
						)
					except ValueError:
						continue
					if not ser:
						continue
					stop = _simulate_single_plateau(ser, mode=str(mode), patience=patience, start_epoch=start_epoch, min_delta=min_delta)
					stop_eff = int(stop) if stop is not None else last_ep
					q_stop = _value_at(bench_series, int(stop_eff))
					if q_stop is None:
						continue
					if str(bench_mode) == "min":
						gap = float(q_stop) - float(best_val)
					else:
						gap = float(best_val) - float(q_stop)
					gap_abs = max(0.0, float(gap))
					rel_pct = gap_abs / max(abs(float(best_val)), 1e-12) * 100.0
					row = {
						"kind": "single",
						"metric": metric,
						"mode": mode,
						"layer": layer,
						"patience": patience,
						"start_epoch": start_epoch,
						"min_delta": float(min_delta),
						"stop_epoch": stop,
						"effective_stop_epoch": stop_eff,
						"best_val_epoch": int(best_ep),
						"bench_val_at_stop": float(q_stop),
						"bench_best": float(best_val),
						"bench_delta_at_stop": float(q_stop) - float(best_val),
						"gap_abs": float(gap_abs),
						"gap_rel_pct": float(rel_pct),
					}
					_attach_stop_summary(row, last_ep=last_ep, stop=stop)
					if emit_full_grid:
						grid_rows.append(row)
					ranked_candidates.append(dict(row))

		if not include_ensembles:
			continue

		for agg in ("all", "any"):
			for triplet in itertools.combinations(signal_metrics, 3):
				modelist = itertools.product(("min", "max"), repeat=3)
				for combo_modes in modelist:
					layers_three: List[str] = []
					series_three: List[List[Tuple[int, float]]] = []
					ok = True
					for mi, mdi in zip(triplet, combo_modes):
						lay = _choose_oracle_layer_for_metric(
							recs,
							layers=mon_layers,
							bench_series=bench_series,
							bench_mode=str(bench_mode),
							metric=str(mi),
							signal_mode=str(mdi),
							patience=patience,
							start_epoch=start_epoch,
							min_delta=min_delta,
							zero_tol=float(zero_tol),
						)
						if lay is None:
							ok = False
							break
						try:
							ser = _series_signal(
								recs, sig=Signal(metric=str(mi), mode=str(mdi), layer=str(lay)), zero_tol=float(zero_tol)
							)
						except ValueError:
							ok = False
							break
						if not ser:
							ok = False
							break
						layers_three.append(str(lay))
						series_three.append(ser)
					if not ok:
						continue
					stop = _simulate_multi_plateau(
						series_three,
						modes=[str(x) for x in combo_modes],
						aggregate=str(agg),
						patience=int(patience),
						start_epoch=int(start_epoch),
						min_delta=float(min_delta),
					)
					stop_eff = int(stop) if stop is not None else last_ep
					q_stop = _value_at(bench_series, int(stop_eff))
					if q_stop is None:
						continue
					if str(bench_mode) == "min":
						gap = float(q_stop) - float(best_val)
					else:
						gap = float(best_val) - float(q_stop)
					gap_abs = max(0.0, float(gap))
					rel_pct = gap_abs / max(abs(float(best_val)), 1e-12) * 100.0
					row = {
						"kind": "ensemble3",
						"metrics": list(triplet),
						"modes": list(combo_modes),
						"layers": layers_three,
						"aggregate": str(agg),
						"patience": patience,
						"start_epoch": start_epoch,
						"min_delta": float(min_delta),
						"stop_epoch": stop,
						"effective_stop_epoch": stop_eff,
						"best_val_epoch": int(best_ep),
						"bench_val_at_stop": float(q_stop),
						"bench_best": float(best_val),
						"bench_delta_at_stop": float(q_stop) - float(best_val),
						"gap_abs": float(gap_abs),
						"gap_rel_pct": float(rel_pct),
					}
					_attach_stop_summary(row, last_ep=last_ep, stop=stop)
					if emit_full_grid:
						grid_rows.append(row)
					ranked_candidates.append(dict(row))

	ranked_candidates.sort(key=lambda r: (float(r["gap_rel_pct"]), float(r.get("effective_stop_epoch", 0))))
	best_rule = ranked_candidates[0] if ranked_candidates else None
	total_rules = len(ranked_candidates)
	n_single = sum(1 for r in ranked_candidates if str(r.get("kind")) == "single")
	n_ensemble = sum(1 for r in ranked_candidates if str(r.get("kind")) == "ensemble3")
	ranked_export = ranked_candidates[:200]
	ensemble_oracle_rows = [
		{
			"kind": "ensemble3",
			"metrics": list(r.get("metrics") or []),
			"modes": list(r.get("modes") or []),
			"layers": list(r.get("layers") or []),
			"aggregate": str(r.get("aggregate", "")),
			"patience": int(r.get("patience", 0)),
			"start_epoch": int(r.get("start_epoch", start_epoch)),
			"min_delta": float(r.get("min_delta", min_delta)),
			"stop_epoch": r.get("stop_epoch", None),
			"effective_stop_epoch": int(r.get("effective_stop_epoch", last_ep)),
			"triggered": bool(r.get("triggered", False)),
			"epochs_saved": int(r.get("epochs_saved", 0)),
			"gap_rel_pct": float(r.get("gap_rel_pct", 0.0)),
		}
		for r in ranked_candidates
		if str(r.get("kind")) == "ensemble3"
	]

	out_obj: Dict[str, Any] = {
		"run_dir": run_dir_abs,
		"bench_key": bench_key,
		"bench_metric": str(pick_bench_metric),
		"bench_mode": str(bench_mode),
		"last_training_epoch": int(last_ep),
		"best_val_epoch": int(best_ep),
		"benchmark_best_value": float(best_val),
		"oracle_best_on_run": {"epoch": int(best_ep), "value": float(best_val)},
		"sweep_spec": {
			"signal_metrics": list(signal_metrics),
			"patience_values": sorted(set(int(x) for x in patiences)),
			"start_epoch": int(start_epoch),
			"min_delta": float(min_delta),
			"include_ensembles": bool(include_ensembles),
			"counts": {
				"single_rules": int(n_single),
				"ensemble3_rules": int(n_ensemble),
				"total_rules_ranked": int(total_rules),
				"ranked_entries_exported": min(200, int(total_rules)),
			},
		},
		"settings": {"start_epoch": int(start_epoch), "min_delta": float(min_delta), "patiences": sorted(set(int(x) for x in patiences))},
		"ranked_by_gap_rel_pct": ranked_export,
		"ensemble3_oracle_rows": ensemble_oracle_rows,
		"best": best_rule,
	}
	if emit_full_grid:
		out_obj["full_grid_rows"] = grid_rows
	return out_obj


def _build_row_from_stop(
	ctx: EarlyStopRunContext,
	*,
	stop: Optional[int],
	start_epoch: int,
	min_delta: float,
	patience: int,
	extra: Dict[str, Any],
) -> Dict[str, Any]:
	last_ep = ctx.last_ep
	best_ep = ctx.best_ep
	best_val = ctx.best_val
	bench_mode = ctx.bench_mode
	bench_series = ctx.bench_series
	stop_eff = int(stop) if stop is not None else last_ep
	q_stop = _value_at(bench_series, int(stop_eff))
	if q_stop is None:
		raise RuntimeError("missing bench value at effective stop epoch")
	if str(bench_mode) == "min":
		gap = float(q_stop) - float(best_val)
	else:
		gap = float(best_val) - float(q_stop)
	gap_abs = max(0.0, float(gap))
	rel_pct = gap_abs / max(abs(float(best_val)), 1e-12) * 100.0
	row: Dict[str, Any] = {
		**extra,
		"patience": int(patience),
		"start_epoch": int(start_epoch),
		"min_delta": float(min_delta),
		"stop_epoch": stop,
		"effective_stop_epoch": stop_eff,
		"best_val_epoch": int(best_ep),
		"bench_val_at_stop": float(q_stop),
		"bench_best": float(best_val),
		"bench_delta_at_stop": float(q_stop) - float(best_val),
		"gap_abs": float(gap_abs),
		"gap_rel_pct": float(rel_pct),
	}
	_attach_stop_summary(row, last_ep=last_ep, stop=stop)
	return row


def evaluate_single_signal_rule_on_run(
	run_dir: str,
	*,
	metric: str,
	mode: str,
	layer: str,
	patience: int,
	start_epoch: int = 3,
	min_delta: float = 0.0,
	bench_metric_override: str = "",
) -> Dict[str, Any]:
	ctx = load_early_stop_run_context(run_dir, bench_metric_override)
	try:
		ser = _series_signal(ctx.recs, sig=Signal(metric=str(metric), mode=str(mode), layer=str(layer)), zero_tol=float(ctx.zero_tol))
	except ValueError as e:
		raise ValueError(str(e)) from e
	if not ser:
		raise RuntimeError(f"empty signal series for {metric} @ {layer}")
	stop = _simulate_single_plateau(
		ser, mode=str(mode), patience=int(patience), start_epoch=int(start_epoch), min_delta=float(min_delta)
	)
	ex = {"kind": "single", "metric": str(metric), "mode": str(mode), "layer": str(layer)}
	return _build_row_from_stop(
		ctx,
		stop=stop,
		start_epoch=int(start_epoch),
		min_delta=float(min_delta),
		patience=int(patience),
		extra=ex,
	)


def evaluate_ensemble3_rule_on_run(
	run_dir: str,
	*,
	metrics: Sequence[str],
	modes: Sequence[str],
	layers: Sequence[str],
	aggregate: str,
	patience: int,
	start_epoch: int = 3,
	min_delta: float = 0.0,
	bench_metric_override: str = "",
) -> Dict[str, Any]:
	triple = list(zip(metrics, modes, layers))
	if len(triple) != 3:
		raise ValueError("ensemble3 requires exactly three (metric, mode, layer) entries")
	ctx = load_early_stop_run_context(run_dir, bench_metric_override)
	series_three: List[List[Tuple[int, float]]] = []
	modes_l: List[str] = []
	ms: List[str] = []
	ls: List[str] = []
	for metric, mode, layer in triple:
		ms.append(str(metric))
		modes_l.append(str(mode))
		ls.append(str(layer))
		try:
			ser = _series_signal(ctx.recs, sig=Signal(metric=str(metric), mode=str(mode), layer=str(layer)), zero_tol=float(ctx.zero_tol))
		except ValueError as e:
			raise ValueError(str(e)) from e
		if not ser:
			raise RuntimeError(f"empty signal series for {metric} @ {layer}")
		series_three.append(ser)
	stop = _simulate_multi_plateau(
		series_three,
		modes=modes_l,
		aggregate=str(aggregate),
		patience=int(patience),
		start_epoch=int(start_epoch),
		min_delta=float(min_delta),
	)
	ex = {"kind": "ensemble3", "metrics": ms, "modes": modes_l, "layers": ls, "aggregate": str(aggregate)}
	return _build_row_from_stop(
		ctx,
		stop=stop,
		start_epoch=int(start_epoch),
		min_delta=float(min_delta),
		patience=int(patience),
		extra=ex,
	)


def main() -> None:
	ap = argparse.ArgumentParser(description="Sweep repr-based plateau early-stop rules over finished runs (offline).")
	ap.add_argument("--run_dir", type=str, nargs="*", default=[], help="One or more run directories.")
	ap.add_argument("--run_dirs_file", type=str, default="", help="UTF-8 text file: one absolute run dir per line.")
	ap.add_argument(
		"--roots",
		type=str,
		nargs="*",
		default=[],
		help="Directories to scan recursively for exp_* with meta.json+metrics.jsonl.",
	)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument("--patience", type=str, default="3,4,5,6,7,8,9")
	ap.add_argument("--bench_metric", type=str, default="", help="Force bench metric key (otherwise infer from epoch 0).")
	ap.add_argument("--full_grid", action="store_true", help="Also store exhaustive grid_rows (large).")
	ap.add_argument("--out_suffix", type=str, default="repr_early_stop_sweep.json")
	ap.add_argument("--skip_existing", action="store_true", help="Skip if output JSON exists and is non-empty.")
	ap.add_argument("--json_pretty", action="store_true", help="Ignored (output is always compact JSON).")
	ap.add_argument("--single_only", action="store_true", help="Skip 3-signal ensemble rules; only evaluate single-signal rules (faster, smaller output).")
	args = ap.parse_args()

	patiences = [int(x.strip()) for x in str(args.patience).split(",") if x.strip().isdigit()]

	dir_list: List[str] = []
	dir_list.extend(str(x).strip() for x in args.run_dir if str(x).strip())
	if str(args.run_dirs_file).strip():
		with open(str(args.run_dirs_file).strip(), "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					dir_list.append(line)
	for rt in args.roots:
		root_abs = os.path.abspath(str(rt))
		if not os.path.isdir(root_abs):
			print("[skip] missing root:", root_abs, file=sys.stderr)
			continue
		for dp, dirs, fnames in os.walk(root_abs):
			if fnames == [] or "meta.json" not in fnames:
				continue
			if os.path.basename(dp).startswith("exp_"):
				dir_list.append(dp)

	dir_list_u: List[str] = []
	for p in sorted(set(os.path.abspath(x) for x in dir_list if x)):
		if p not in dir_list_u:
			dir_list_u.append(p)

	if not dir_list_u:
		raise SystemExit("No directories to sweep.")

	failures = 0
	for rd in dir_list_u:
		out_json = os.path.join(rd, "analysis", str(args.out_suffix))
		if args.skip_existing and os.path.isfile(out_json) and os.path.getsize(out_json) > 10:
			print("[skip existing]", rd)
			continue
		try:
			obj = sweep_run_directory(
				rd,
				start_epoch=int(args.start_epoch),
				min_delta=float(args.min_delta),
				patiences=patiences,
				bench_metric_override=str(args.bench_metric or "").strip(),
				emit_full_grid=bool(args.full_grid),
				include_ensembles=not bool(args.single_only),
			)
		except Exception as e:
			print(f"[fail] {rd}: {e}", file=sys.stderr)
			failures += 1
			continue
		os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
		with open(out_json, "w", encoding="utf-8") as wf:
			json.dump(obj, wf, separators=(",", ":"), ensure_ascii=False)
			wf.write("\n")
		print("[ok]", rd, "->", os.path.basename(out_json), "gap_rel(best)=", obj.get("best", {}).get("gap_rel_pct") if isinstance(obj.get("best"), dict) else None)

	if failures > 0:
		raise SystemExit(f"Sweep finished with failures={failures}")


if __name__ == "__main__":
	main()
