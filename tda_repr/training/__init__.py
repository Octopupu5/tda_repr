from importlib import import_module
from typing import Dict, Tuple


_SYMBOLS: Dict[str, Tuple[str, str]] = {
	"RepresentationMonitor": ("tda_repr.training.monitor", "RepresentationMonitor"),
	"RepresentationMonitorConfig": ("tda_repr.training.monitor", "RepresentationMonitorConfig"),
	"BenchmarkSpec": ("tda_repr.training.benchmarks", "BenchmarkSpec"),
	"evaluate_classification": ("tda_repr.training.benchmarks", "evaluate_classification"),
	"evaluate_generation": ("tda_repr.training.benchmarks", "evaluate_generation"),
	"RunStore": ("tda_repr.training.results", "RunStore"),
	"JSONLWriter": ("tda_repr.training.results", "JSONLWriter"),
	"ExperimentTracker": ("tda_repr.training.tracker", "ExperimentTracker"),
	"TrackerConfig": ("tda_repr.training.tracker", "TrackerConfig"),
}

__all__ = sorted(_SYMBOLS.keys())


def __getattr__(name: str):
	if name not in _SYMBOLS:
		raise AttributeError(f"module 'tda_repr.training' has no attribute {name!r}")
	module_name, attr_name = _SYMBOLS[name]
	module = import_module(module_name)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value
