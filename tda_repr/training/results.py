from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def _json_default(obj: Any):
	# Best-effort conversion for common types
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, (np.ndarray,)):
		return obj.tolist()
	return str(obj)


@dataclass
class JSONLWriter:
	"""
	Append-only writer: one JSON object per line (great for long training runs).
	"""

	path: str
	flush: bool = True

	def __post_init__(self) -> None:
		os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

	def write(self, obj: Dict[str, Any]) -> None:
		with open(self.path, "a", encoding="utf-8") as f:
			f.write(json.dumps(obj, ensure_ascii=False, default=_json_default))
			f.write("\n")
			if self.flush:
				f.flush()


@dataclass
class RunStore:
	"""
	Small helper for run folders: run_dir/metrics.jsonl + run_dir/meta.json
	"""

	run_dir: str
	metrics_file: str = "metrics.jsonl"
	meta_file: str = "meta.json"
	# If True, avoid overwriting previous experiments by creating a unique run_dir if needed
	unique: bool = True
	# Optional prefix/suffix additions (e.g., model/dataset tags)
	prefix: str = ""
	suffix: str = ""

	def __post_init__(self) -> None:
		base_dir = self.run_dir
		if self.prefix:
			# prefix is applied as parent folder: prefix/base_dir_name
			base_dir = os.path.join(self.prefix, base_dir)
		self.run_dir = base_dir

		if self.unique:
			# Always create a new run directory with a timestamp suffix.
			ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
			base = f"{self.run_dir}_{ts}"
			if self.suffix:
				base = f"{base}_{self.suffix}"
			cand = base
			i = 1
			while os.path.exists(cand):
				i += 1
				cand = f"{base}_{i}"
			self.run_dir = cand
		elif self.suffix:
			# If unique is off but suffix requested, still add it deterministically
			self.run_dir = f"{self.run_dir}_{self.suffix}"

		os.makedirs(self.run_dir, exist_ok=True)
		self.metrics = JSONLWriter(os.path.join(self.run_dir, self.metrics_file))

	def write_meta(self, meta: Dict[str, Any]) -> None:
		meta = dict(meta)
		meta.setdefault("created_at", time.time())
		with open(os.path.join(self.run_dir, self.meta_file), "w", encoding="utf-8") as f:
			json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

	def log(self, event: str, payload: Dict[str, Any]) -> None:
		rec = {"event": event, "ts": time.time()}
		rec.update(payload)
		self.metrics.write(rec)
