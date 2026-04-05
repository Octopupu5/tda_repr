from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as rf:
		return json.load(rf)


def _discover_sweep_files(roots: Sequence[str]) -> List[str]:
	out: List[str] = []
	for rt in roots:
		root_abs = os.path.abspath(str(rt))
		if not os.path.isdir(root_abs):
			raise FileNotFoundError(root_abs)
		for dp, _dirs, fnames in os.walk(root_abs):
			if "repr_early_stop_sweep.json" in fnames:
				out.append(os.path.join(dp, "repr_early_stop_sweep.json"))
	return sorted(set(out))


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	return float(v) if math.isfinite(float(v)) else None


def _group_key(meta: Mapping[str, Any]) -> str:
	model = str(meta.get("model", "") or "").strip()
	if not model:
		model = str((meta.get("args") or {}).get("model", "") or "").strip()
	ds = str(meta.get("dataset", "") or "").strip()
	if not model or not ds:
		raise ValueError("meta.json missing model/dataset.")
	return f"{model}::{ds}"


def _best_row_of_kind(obj: Mapping[str, Any], kind: str) -> Optional[Dict[str, Any]]:
	rows = obj.get("ranked_by_gap_rel_pct")
	if not isinstance(rows, list):
		return None
	for r in rows:
		if not isinstance(r, dict):
			continue
		if str(r.get("kind", "")).lower().strip() == str(kind).lower().strip():
			return dict(r)
	return None


def _signature(row: Mapping[str, Any]) -> Tuple[Any, ...]:
	k = str(row.get("kind", "")).lower().strip()
	if k == "single":
		return (
			"single",
			str(row.get("metric", "")),
			str(row.get("mode", "")),
			str(row.get("layer", "")),
			int(row.get("patience", 0)),
			int(row.get("start_epoch", 0)),
			float(row.get("min_delta", 0.0)),
		)
	if k == "ensemble3":
		return (
			"ensemble3",
			tuple(str(x) for x in (row.get("metrics") or [])),
			tuple(str(x) for x in (row.get("modes") or [])),
			tuple(str(x) for x in (row.get("layers") or [])),
			str(row.get("aggregate", "")),
			int(row.get("patience", 0)),
			int(row.get("start_epoch", 0)),
			float(row.get("min_delta", 0.0)),
		)
	raise ValueError(f"Unknown kind: {k}")


def _rule_spec_from_row(row: Mapping[str, Any], *, bench_metric: str = "") -> Dict[str, Any]:
	k = str(row.get("kind", "")).lower().strip()
	out: Dict[str, Any] = {
		"type": k,
		"patience": int(row.get("patience", 0)),
		"start_epoch": int(row.get("start_epoch", 3)),
		"min_delta": float(row.get("min_delta", 0.0)),
	}
	if bench_metric:
		out["bench_metric"] = str(bench_metric)
	if k == "single":
		out.update(
			{
				"metric": str(row.get("metric", "")),
				"mode": str(row.get("mode", "")),
				"layer": str(row.get("layer", "")),
			}
		)
		return out
	if k == "ensemble3":
		out.update(
			{
				"metrics": [str(x) for x in (row.get("metrics") or [])],
				"modes": [str(x) for x in (row.get("modes") or [])],
				"layers": [str(x) for x in (row.get("layers") or [])],
				"aggregate": str(row.get("aggregate", "all")),
			}
		)
		return out
	raise ValueError(f"Unknown type: {k}")


def _mean(vals: Iterable[float]) -> float:
	xs = [float(x) for x in vals if math.isfinite(float(x))]
	if not xs:
		return float("inf")
	return sum(xs) / float(len(xs))


def main(argv: Sequence[str] | None = None) -> None:
	if argv is None:
		argv = sys.argv[1:]
	ap = argparse.ArgumentParser(description="Select early-stop rules from repr_early_stop_sweep.json files.")
	ap.add_argument("--roots", type=str, nargs="+", required=True)
	ap.add_argument("--top_k_ensembles", type=int, default=6)
	ap.add_argument("--out_json", type=str, required=True)
	args = ap.parse_args(list(argv))

	paths = _discover_sweep_files(args.roots)
	if not paths:
		raise SystemExit("No repr_early_stop_sweep.json files found under roots.")

	gaps: DefaultDict[str, DefaultDict[str, DefaultDict[Tuple[Any, ...], List[float]]]] = defaultdict(
		lambda: defaultdict(lambda: defaultdict(list))
	)
	rep_row: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
	bench_metric_by_group: Dict[str, str] = {}

	for sp in paths:
		run_dir = os.path.abspath(os.path.join(os.path.dirname(sp), ".."))
		meta_path = os.path.join(run_dir, "meta.json")
		if not os.path.isfile(meta_path):
			raise FileNotFoundError(meta_path)
		meta = _load_json(meta_path)
		gk = _group_key(meta)
		obj = _load_json(sp)
		bench_metric_by_group.setdefault(gk, str(obj.get("bench_metric", "") or "").strip())
		for kind in ("single", "ensemble3"):
			row = _best_row_of_kind(obj, kind)
			if not row:
				continue
			gap = _safe_float(row.get("gap_rel_pct"))
			if gap is None:
				continue
			sig = _signature(row)
			gaps[gk][kind][sig].append(float(gap))
			rep_row.setdefault((gk, sig), dict(row))

	out: Dict[str, Any] = {"version": 1, "group_by": "model_dataset", "primary": {}, "ensembles_top_k": {}}

	for gk, by_kind in gaps.items():
		best_single = None
		best_ens = None
		if by_kind.get("single"):
			best_single = min(by_kind["single"].items(), key=lambda kv: (_mean(kv[1]), str(kv[0])))[0]
		if by_kind.get("ensemble3"):
			best_ens = min(by_kind["ensemble3"].items(), key=lambda kv: (_mean(kv[1]), str(kv[0])))[0]

		primary_sig = None
		if best_single is None and best_ens is None:
			continue
		if best_single is None:
			primary_sig = best_ens
		elif best_ens is None:
			primary_sig = best_single
		else:
			ms = _mean(by_kind["single"][best_single])
			me = _mean(by_kind["ensemble3"][best_ens])
			primary_sig = best_single if ms <= me else best_ens

		row_primary = rep_row[(gk, primary_sig)]
		out["primary"][gk] = _rule_spec_from_row(row_primary, bench_metric=bench_metric_by_group.get(gk, ""))

		ens_items = list(by_kind.get("ensemble3", {}).items())
		ens_items.sort(key=lambda kv: (_mean(kv[1]), str(kv[0])))
		top = []
		for sig, _vals in ens_items[: int(args.top_k_ensembles)]:
			top.append(_rule_spec_from_row(rep_row[(gk, sig)], bench_metric=bench_metric_by_group.get(gk, "")))
		out["ensembles_top_k"][gk] = top

	path_out = os.path.abspath(str(args.out_json))
	os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
	with open(path_out, "w", encoding="utf-8") as wf:
		json.dump(out, wf, indent=2, ensure_ascii=False)
	print("[ok] wrote", path_out, file=sys.stderr)


if __name__ == "__main__":
	main()

