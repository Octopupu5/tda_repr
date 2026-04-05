from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from tools.aggregate.tables_depth_arch import (
	_iter_jsonl,
	_pick_primary_metric,
	_read_json,
)


def _kth_sorted_finite(xs: Any, k: int) -> Optional[float]:
	if not isinstance(xs, (list, tuple)) or len(xs) < 1:
		return None
	arr = np.asarray(xs, dtype=np.float64)
	arr = arr[np.isfinite(arr)]
	arr = np.sort(arr)
	if arr.size < int(k):
		return None
	return float(arr[int(k) - 1])


@dataclass(frozen=True)
class Descriptor:
	key: str
	display_latex: str
	extract: Callable[[Dict[str, Any], float], Optional[float]]


def _ex_b0(_layer: Dict[str, Any], _zt: float) -> Optional[float]:
	v = _layer.get("beta0_L_est")
	return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _ex_b1(_layer: Dict[str, Any], _zt: float) -> Optional[float]:
	v = _layer.get("beta1_L_est")
	return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _ex_b0p(_layer: Dict[str, Any], _zt: float) -> Optional[float]:
	v = _layer.get("beta0_persistent_est")
	return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _ex_b1p(_layer: Dict[str, Any], _zt: float) -> Optional[float]:
	v = _layer.get("beta1_persistent_est")
	return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _ex_hodge_q0_l2(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("hodge_L_q0_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 2)


def _ex_pers_q0_l2(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("persistent_q0_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 2)


def _ex_hodge_q1_l1(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("hodge_L_q1_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 1)


def _ex_hodge_q1_l2(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("hodge_L_q1_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 2)


def _ex_pers_q1_l1(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("persistent_q1_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 1)


def _ex_pers_q1_l2(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	xs = layer.get("persistent_q1_smallest")
	if not isinstance(xs, (list, tuple)):
		return None
	return _kth_sorted_finite(xs, 2)


def _ex_mtop(layer: Dict[str, Any], _zt: float) -> Optional[float]:
	v = layer.get("mtopdiv_train_val")
	return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


DESCRIPTORS: Tuple[Descriptor, ...] = (
	Descriptor("beta0_L", r"$\beta_0(L)$", _ex_b0),
	Descriptor("beta1_L", r"$\beta_1(L)$", _ex_b1),
	Descriptor("beta0_KL", r"$\beta_0^{K,L}$", _ex_b0p),
	Descriptor("beta1_KL", r"$\beta_1^{K,L}$", _ex_b1p),
	Descriptor("lam2_hodge_q0", r"$\lambda_2(\Delta_0(L))$", _ex_hodge_q0_l2),
	Descriptor("lam2_pers_q0", r"$\lambda_2(\Delta_0^{K,L})$", _ex_pers_q0_l2),
	Descriptor("lam1_hodge_q1", r"$\lambda_1(\Delta_1(L))$", _ex_hodge_q1_l1),
	Descriptor("lam2_hodge_q1", r"$\lambda_2(\Delta_1(L))$", _ex_hodge_q1_l2),
	Descriptor("lam1_pers_q1", r"$\lambda_1(\Delta_1^{K,L})$", _ex_pers_q1_l1),
	Descriptor("lam2_pers_q1", r"$\lambda_2(\Delta_1^{K,L})$", _ex_pers_q1_l2),
	Descriptor("mtopdiv", r"$\mathrm{MTopDiv}$", _ex_mtop),
)


def _bench_scalar_for_epoch(meta: Dict[str, Any], bench: Dict[str, Any]) -> Optional[float]:
	ds = str(meta.get("dataset", "")).strip()
	if not ds:
		return None
	for key in (f"{ds}-val", f"{ds}-test"):
		block = bench.get(key)
		if isinstance(block, dict):
			_name, y = _pick_primary_metric(block)
			if isinstance(y, (int, float)) and math.isfinite(float(y)):
				return float(y)
	for _k, block in bench.items():
		if not isinstance(block, dict):
			continue
		if "-val" not in str(_k) and "-test" not in str(_k):
			continue
		_name, y = _pick_primary_metric(block)
		if isinstance(y, (int, float)) and math.isfinite(float(y)):
			return float(y)
	return None


def _load_epoch_series(
	run_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[Tuple[int, Dict[str, Any], float]]]:
	meta_path = os.path.join(run_dir, "meta.json")
	meta = _read_json(meta_path)
	monitor = meta.get("monitor") if isinstance(meta.get("monitor"), dict) else {}
	try:
		zero_tol = float(monitor.get("zero_tol", 1e-8))
	except (TypeError, ValueError):
		zero_tol = 1e-8

	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	series: List[Tuple[int, Dict[str, Any], float]] = []
	for rec in _iter_jsonl(metrics_path):
		if str(rec.get("event", "")) != "epoch_end":
			continue
		ep_raw = rec.get("epoch", None)
		if not isinstance(ep_raw, int):
			try:
				ep = int(ep_raw)
			except (TypeError, ValueError):
				continue
		else:
			ep = ep_raw
		repr_part = rec.get("repr") or {}
		layers = repr_part.get("layers") if isinstance(repr_part, dict) else None
		if not isinstance(layers, dict):
			continue
		bench = rec.get("bench") or {}
		if not isinstance(bench, dict):
			continue
		y = _bench_scalar_for_epoch(meta, bench)
		if y is None or not math.isfinite(y):
			continue
		series.append((ep, layers, float(y)))

	series.sort(key=lambda t: t[0])
	return meta, monitor, zero_tol, series


def _compute_spearman(xs: List[float], ys: List[float]) -> Optional[float]:
	if len(xs) < 3 or len(xs) != len(ys):
		return None
	a = np.asarray(xs, dtype=np.float64)
	b = np.asarray(ys, dtype=np.float64)
	if float(np.std(a)) < 1e-14 or float(np.std(b)) < 1e-14:
		return None
	rho, _p = spearmanr(a, b)
	if not math.isfinite(float(rho)):
		return None
	return float(rho)
