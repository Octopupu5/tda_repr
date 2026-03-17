from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def get_modules_by_names(model: nn.Module, names: Iterable[str]) -> Dict[str, nn.Module]:
	"""
	Return a dict name->module for given qualified names from model.named_modules().
	Raises KeyError if any name is missing.
	"""
	all_named = dict(model.named_modules())
	result: Dict[str, nn.Module] = {}
	for n in names:
		if n not in all_named:
			raise KeyError(f"Module name not found: {n}")
		result[n] = all_named[n]
	return result


class LayerTaps:
	"""
	Attach forward hooks to selected modules to collect their outputs during forward passes.
	Usage:
		taps = LayerTaps(model, ["layer1", "layer2.0.conv1"])
		out = model(x); feats = taps.outputs
		taps.close()
	"""

	def __init__(self, model: nn.Module, layer_names: Iterable[str]) -> None:
		self.model = model
		self.layer_names: List[str] = list(layer_names)
		self.modules = get_modules_by_names(model, self.layer_names)
		self.outputs: Dict[str, Any] = {}
		self._handles: List[Any] = []
		self._register()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()
		return False

	def _register(self) -> None:
		def make_hook(k: str):
			def hook(_module: nn.Module, _inp: Tuple[torch.Tensor, ...], out: Any):
				self.outputs[k] = out
			return hook
		for name, mod in self.modules.items():
			self._handles.append(mod.register_forward_hook(make_hook(name)))

	def clear(self) -> None:
		self.outputs.clear()

	def close(self) -> None:
		for h in self._handles:
			h.remove()
		self._handles.clear()


