"""
tda_repr public API with lazy symbol loading.

Lazy imports keep lightweight commands (e.g. layer inspection) fast and robust
when optional dependencies are not installed.
"""

from importlib import import_module
from typing import Dict, Tuple


_SYMBOLS: Dict[str, Tuple[str, str]] = {
	# spectral
	"SimplicialComplex": ("tda_repr.spectral", "SimplicialComplex"),
	"boundary_matrix": ("tda_repr.spectral", "boundary_matrix"),
	"up_laplacian": ("tda_repr.spectral", "up_laplacian"),
	"down_laplacian": ("tda_repr.spectral", "down_laplacian"),
	"hodge_laplacian": ("tda_repr.spectral", "hodge_laplacian"),
	"persistent_up_laplacian_operator": ("tda_repr.spectral", "persistent_up_laplacian_operator"),
	"eigs_persistent_up": ("tda_repr.spectral", "eigs_persistent_up"),
	"eigs_hodge": ("tda_repr.spectral", "eigs_hodge"),
	# models
	"get_model_info": ("tda_repr.models", "get_model_info"),
	"ModelInfo": ("tda_repr.models", "ModelInfo"),
	"LayerTaps": ("tda_repr.models", "LayerTaps"),
	"get_modules_by_names": ("tda_repr.models", "get_modules_by_names"),
	"list_module_names": ("tda_repr.models", "list_module_names"),
	"list_parameter_names": ("tda_repr.models", "list_parameter_names"),
	"select_names": ("tda_repr.models", "select_names"),
	"set_trainable_by_name_selection": ("tda_repr.models", "set_trainable_by_name_selection"),
	# data
	"DataBundle": ("tda_repr.data", "DataBundle"),
	"get_dataset": ("tda_repr.data", "get_dataset"),
	"make_dataloaders": ("tda_repr.data", "make_dataloaders"),
	# training
	"RepresentationMonitor": ("tda_repr.training", "RepresentationMonitor"),
	"RepresentationMonitorConfig": ("tda_repr.training", "RepresentationMonitorConfig"),
	"BenchmarkSpec": ("tda_repr.training", "BenchmarkSpec"),
	"evaluate_classification": ("tda_repr.training", "evaluate_classification"),
	"evaluate_generation_bleu": ("tda_repr.training", "evaluate_generation_bleu"),
	"RunStore": ("tda_repr.training", "RunStore"),
	"JSONLWriter": ("tda_repr.training", "JSONLWriter"),
	"ExperimentTracker": ("tda_repr.training", "ExperimentTracker"),
	"TrackerConfig": ("tda_repr.training", "TrackerConfig"),
}

__all__ = sorted(_SYMBOLS.keys())


def __getattr__(name: str):
	if name not in _SYMBOLS:
		raise AttributeError(f"module 'tda_repr' has no attribute {name!r}")
	module_name, attr_name = _SYMBOLS[name]
	module = import_module(module_name)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value
