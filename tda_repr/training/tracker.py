from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

from .benchmarks import BenchmarkSpec, evaluate_classification, evaluate_generation
from .monitor import RepresentationMonitor
from .results import RunStore


@dataclass
class TrackerConfig:
	run_dir: str = "runs/run1"

	eval_every: int = 1

	max_eval_batches: Optional[int] = None


class ExperimentTracker:
	def __init__(
		self,
		monitor: RepresentationMonitor,
		benchmarks: List[BenchmarkSpec],
		store: RunStore,
		cfg: Optional[TrackerConfig] = None,
	):
		self.monitor = monitor
		self.benchmarks = benchmarks
		self.store = store
		self.cfg = cfg or TrackerConfig(run_dir=store.run_dir)

	def on_epoch_end(
		self,
		epoch: int,
		model: Any,
		dataloaders: Dict[str, Any],
		loss_fn: Optional[Any] = None,
		preprocess: Optional[Any] = None,
		extra: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""
		Call after training/val loops finished and monitor collected activations.
		Returns merged dict and appends to JSONL.
		"""
		t_total0 = time.perf_counter()
		out: Dict[str, Any] = {"epoch": epoch}
		if extra:
			out["extra"] = extra


		t_rep0 = time.perf_counter()
		rep = self.monitor.end_epoch(epoch)
		rep_s = float(time.perf_counter() - t_rep0)
		out["repr"] = rep


		bench_out: Dict[str, Any] = {}
		bench_timing: Dict[str, float] = {}
		t_bench0 = time.perf_counter()
		if self.cfg.eval_every > 0 and (epoch % self.cfg.eval_every == 0):
			for spec in self.benchmarks:
				loader = dataloaders.get(spec.dataloader_key, None)
				if loader is None:
					continue
				t_spec0 = time.perf_counter()
				try:
					kind = str(getattr(spec, "kind", "classification")).lower().strip()
					if kind == "classification":
						metrics = evaluate_classification(
							model,
							loader,
							loss_fn=loss_fn,
							preprocess=preprocess,
							max_batches=self.cfg.max_eval_batches,
						)
					elif kind == "generation":
						metrics = evaluate_generation(
							model,
							loader,
							max_batches=self.cfg.max_eval_batches,
						)
					else:
						raise ValueError(f"Unknown BenchmarkSpec.kind: {spec.kind!r}")

					if spec.metrics:
						bench_out[spec.name] = {k: metrics[k] for k in spec.metrics if k in metrics}
					else:
						bench_out[spec.name] = metrics
				except Exception as e:
					bench_out[spec.name] = {"error": str(e)}
				finally:
					bench_timing[spec.name] = float(time.perf_counter() - t_spec0)
		out["bench"] = bench_out
		bench_total_s = float(time.perf_counter() - t_bench0)
		train_s = float((extra or {}).get("train_s", 0.0) or 0.0)
		val_s = float((extra or {}).get("val_s", 0.0) or 0.0)
		known_total_s = train_s + val_s + rep_s + bench_total_s

		out["timing_s"] = {
			"train_loop": train_s,
			"val_loop": val_s,
			"repr_end_epoch": rep_s,
			"bench_total": bench_total_s,
			"known_total": known_total_s,
			"bench": bench_timing,
			"tracker_total": float(time.perf_counter() - t_total0),
		}

		self.store.log("epoch_end", out)
		return out
