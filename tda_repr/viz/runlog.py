from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def list_run_dirs(runs_root: str = "runs") -> List[str]:
	"""
	Return run directories containing metrics.jsonl under runs_root.
	"""
	if not os.path.isdir(runs_root):
		return []
	out: List[str] = []
	for name in sorted(os.listdir(runs_root)):
		p = os.path.join(runs_root, name)
		if os.path.isdir(p) and os.path.exists(os.path.join(p, "metrics.jsonl")):
			out.append(p)
	return out


def load_epoch_end_records(metrics_jsonl_path: str) -> List[Dict[str, Any]]:
	"""
	Load all records with event='epoch_end' from a metrics.jsonl.
	"""
	recs: List[Dict[str, Any]] = []
	with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if obj.get("event") == "epoch_end":
				recs.append(obj)
	return recs


def load_meta(run_dir: str) -> Dict[str, Any]:
	"""
	Load run_dir/meta.json if present.
	"""
	p = os.path.join(run_dir, "meta.json")
	if not os.path.exists(p):
		return {}
	try:
		with open(p, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def run_label(run_dir: str, prefer_meta: bool = True) -> str:
	"""
	Human-readable label for a run. Uses meta.json if available; otherwise basename.
	"""
	base = os.path.basename(run_dir.rstrip(os.sep))
	if not prefer_meta:
		return base
	meta = load_meta(run_dir)
	if not meta:
		return base
	model = meta.get("model") or meta.get("extra", {}).get("model")
	dataset = meta.get("dataset") or meta.get("extra", {}).get("dataset")
	name = meta.get("name")
	parts = []
	if name:
		parts.append(str(name))
	if model:
		parts.append(str(model))
	if dataset:
		parts.append(str(dataset))
	return " | ".join(parts) if parts else base


def _is_scalar(x: Any) -> bool:
	return isinstance(x, (int, float)) and not isinstance(x, bool)


def _flatten_scalars(prefix: str, obj: Any, out: Dict[str, float]) -> None:
	"""
	Flatten nested dicts into { 'a.b.c': scalar } for scalars only.
	Ignore lists/arrays/strings by default to keep time series simple.
	"""
	if obj is None:
		return
	if _is_scalar(obj):
		out[prefix] = float(obj)
		return
	if isinstance(obj, dict):
		for k, v in obj.items():
			p = f"{prefix}.{k}" if prefix else str(k)
			_flatten_scalars(p, v, out)


def _epoch_records_to_scalar_maps(records: Iterable[Dict[str, Any]]) -> List[Tuple[int, Dict[str, float]]]:
	out: List[Tuple[int, Dict[str, float]]] = []
	for r in records:
		epoch = int(r.get("epoch", r.get("repr", {}).get("epoch", -1)))
		m: Dict[str, float] = {}
		# Prefix benchmark metrics with "bench." for stable, self-describing keys.
		_flatten_scalars("bench", r.get("bench", {}), m)
		_flatten_scalars("repr", r.get("repr", {}), m)
		out.append((epoch, m))
	return out


def list_benchmarks(records: List[Dict[str, Any]]) -> List[str]:
	bench = set()
	for r in records:
		b = r.get("bench", {}) or {}
		if isinstance(b, dict):
			for k in b.keys():
				bench.add(str(k))
	return sorted(bench)


def list_repr_layers(records: List[Dict[str, Any]]) -> List[str]:
	ls = set()
	for r in records:
		layers = ((r.get("repr", {}) or {}).get("layers", {}) or {})
		if isinstance(layers, dict):
			for k in layers.keys():
				ls.add(str(k))
	return sorted(ls)


def list_scalar_series_keys(records: List[Dict[str, Any]]) -> List[str]:
	"""
	Return all scalar keys available across epochs.
	Keys include:
	- bench.<bench_name>.<metric>
	- repr.layers.<layer>.<metric>
	- repr.timing_s.<layer>.<metric> etc
	"""
	keys = set()
	for _epoch, m in _epoch_records_to_scalar_maps(records):
		for k in m.keys():
			keys.add(k)
	return sorted(keys)


def get_series(records: List[Dict[str, Any]], key: str) -> List[Tuple[int, float]]:
	"""
	Return (epoch, value) list for a flattened scalar key.
	"""
	series: List[Tuple[int, float]] = []
	for epoch, m in _epoch_records_to_scalar_maps(records):
		if key in m:
			series.append((epoch, float(m[key])))
	return sorted(series, key=lambda x: x[0])


def find_metrics_file(run_dir: str) -> Optional[str]:
	p = os.path.join(run_dir, "metrics.jsonl")
	return p if os.path.exists(p) else None
