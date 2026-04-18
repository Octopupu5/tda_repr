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
		def _reduce_hook_value(mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
			"""
			Keep hook payload small/stable.

			Some transformer blocks return tuples that may include caches (e.g. KV), which can be very large.
			We keep only the primary tensor (hidden states / logits). For very wide Linear heads (e.g. LM vocab
			projection), we prefer the input activations to avoid exploding the monitored dimensionality.
			"""
			# Prefer the input activations for extremely wide linear projections (e.g. lm_head).
			if isinstance(mod, nn.Linear) and isinstance(out, torch.Tensor):
				try:
					if int(out.shape[-1]) >= 8192 and isinstance(inp, tuple) and inp and isinstance(inp[0], torch.Tensor):
						return inp[0]
				except Exception:
					pass

			# Common: module returns (hidden_states, *extras)
			if isinstance(out, (tuple, list)) and out:
				return out[0]

			# HF ModelOutput-like or mapping-like payloads
			if hasattr(out, "to_tuple") and callable(getattr(out, "to_tuple")):
				try:
					tup = out.to_tuple()
					if isinstance(tup, (tuple, list)) and tup:
						return tup[0]
				except Exception:
					pass
			if isinstance(out, dict) or (hasattr(out, "keys") and hasattr(out, "__getitem__")):
				try:
					keys = list(out.keys())  # type: ignore[attr-defined]
				except Exception:
					keys = []
				for key in ("last_hidden_state", "logits", "hidden_states"):
					if key in keys:
						try:
							return out[key]  # type: ignore[index]
						except Exception:
							break

			return out

		def make_hook(k: str):
			def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any):
				self.outputs[k] = _reduce_hook_value(module, inp, out)
			return hook
		for name, mod in self.modules.items():
			self._handles.append(mod.register_forward_hook(make_hook(name)))

	def clear(self) -> None:
		self.outputs.clear()

	def close(self) -> None:
		for h in self._handles:
			h.remove()
		self._handles.clear()
		# Ensure we drop any references to large tensors immediately.
		self.outputs.clear()


