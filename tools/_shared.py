from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from medmnist import INFO as MEDMNIST_INFO

from tda_repr.models import get_model_info


def default_device_string() -> str:
	if torch.cuda.is_available():
		return "cuda:0"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def seed_everything(seed: int) -> None:
	s = int(seed)
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(s)


def infer_num_classes(dataset_key: str, ds: Any) -> int:
	k = str(dataset_key or "").strip().lower()
	fixed = {
		"mnist": 10,
		"cifar10": 10,
		"imagenette": 10,
		"sst2": 2,
		"trec6": 6,
	}
	if k in fixed:
		return int(fixed[k])
	if k.startswith("medmnist:"):
		k = k.split("medmnist:", 1)[1]
	if k in MEDMNIST_INFO:
		labels = MEDMNIST_INFO[k].get("label", None)
		if isinstance(labels, dict) and len(labels) > 0:
			return int(len(labels))
	if hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and len(ds.classes) > 0:
		return int(len(ds.classes))
	if hasattr(ds, "n_classes"):
		try:
			return int(getattr(ds, "n_classes"))
		except Exception as e:
			raise RuntimeError("Failed to read ds.n_classes.") from e
	return 2


def infer_cv_input_flat_dim(ds: Any) -> Optional[int]:
	try:
		ex = ds[0]
	except Exception:
		return None
	x = None
	if isinstance(ex, (tuple, list)) and ex:
		x = ex[0]
	elif isinstance(ex, dict):
		x = ex.get("image", None)
	if not isinstance(x, torch.Tensor) or x.numel() <= 0:
		return None
	return int(np.prod(tuple(x.shape)))


def build_cv_model(
	kind: str,
	*,
	num_classes: int,
	device: torch.device,
	pretrained: bool,
	input_flat_dim: Optional[int] = None,
) -> Tuple[nn.Module, Optional[Any], list[str]]:
	mi = get_model_info(kind, device=device, pretrained=pretrained)
	model = mi.model
	preprocess = mi.preprocess
	layer_names = list(mi.layer_names)

	k = str(kind).lower().strip()
	if k == "resnet18":
		in_f = model.fc.in_features
		model.fc = nn.Linear(in_f, int(num_classes)).to(device)
	elif k in ("convnext_tiny", "convnext"):
		if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
			last = model.classifier[-1]
			if isinstance(last, nn.Linear):
				model.classifier[-1] = nn.Linear(last.in_features, int(num_classes)).to(device)
	elif k in ("efficientnet_b0", "efficientnet"):
		if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
			last = model.classifier[-1]
			if isinstance(last, nn.Linear):
				model.classifier[-1] = nn.Linear(last.in_features, int(num_classes)).to(device)
	elif k == "mlp":
		if isinstance(model, nn.Sequential) and len(model) >= 2 and isinstance(model[1], nn.Linear):
			if input_flat_dim is not None and int(input_flat_dim) > 0 and int(input_flat_dim) != int(model[1].in_features):
				model[1] = nn.Linear(int(input_flat_dim), 512).to(device)
	return model, preprocess, layer_names


def _pick_transformer_blocks_generic(model: nn.Module) -> list[str]:
	all_named = set(dict(model.named_modules()).keys())
	out: list[str] = []
	for emb_name in ("distilbert.embeddings", "embeddings", "model.embed_tokens", "embed_tokens", "wte"):
		if emb_name in all_named:
			out.append(emb_name)
			break

	for prefix in ("distilbert.transformer.layer", "transformer.layer", "model.layers", "gpt_neox.layers"):
		cands = [n for n in all_named if n.startswith(prefix)]
		if cands:
			out.extend(sorted(set(cands)))
			break
	return out


def build_text_model(
	kind: str,
	*,
	num_classes: int,
	device: torch.device,
	pretrained: bool,
	objective: str = "classification",
) -> Tuple[nn.Module, list[str]]:
	k = str(kind).lower().strip()
	obj = str(objective).strip().lower()
	if obj not in {"classification", "generation"}:
		raise ValueError(f"Unknown NLP objective: {objective!r}")
	if k == "distilbert":
		k = "distilbert-base-uncased"
	if k in ("distilbert", "distilbert-base-uncased"):
		if obj != "classification":
			raise ValueError("distilbert only supports objective=classification in this project.")
		if pretrained:
			try:
				model = transformers.AutoModelForSequenceClassification.from_pretrained(
					"distilbert-base-uncased", num_labels=int(num_classes), token=False
				)
			except TypeError:
				model = transformers.AutoModelForSequenceClassification.from_pretrained(
					"distilbert-base-uncased", num_labels=int(num_classes)
				)
		else:
			model = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(num_labels=int(num_classes)))
		layer_names = _pick_transformer_blocks_generic(model)
		model.to(device)
		return model, layer_names

	smollm_ids = {
		"smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
		"smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
		"smollm": "HuggingFaceTB/SmolLM2-135M",
		"smollm2": "HuggingFaceTB/SmolLM2-135M",
	}
	if k in smollm_ids:
		model_id = smollm_ids[k]
		if obj == "generation":
			if not pretrained:
				try:
					cfg = transformers.AutoConfig.from_pretrained(model_id, token=False)
				except TypeError:
					cfg = transformers.AutoConfig.from_pretrained(model_id)
				model = transformers.AutoModelForCausalLM.from_config(cfg).to(device)
			else:
				try:
					model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=False).to(device)
				except TypeError:
					model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to(device)
			layer_names = _pick_transformer_blocks_generic(model)
			return model, layer_names

		if not pretrained:
			try:
				cfg = transformers.AutoConfig.from_pretrained(model_id, token=False)
			except TypeError:
				cfg = transformers.AutoConfig.from_pretrained(model_id)
			base = transformers.AutoModel.from_config(cfg)
		else:
			try:
				base = transformers.AutoModel.from_pretrained(model_id, token=False)
			except TypeError:
				base = transformers.AutoModel.from_pretrained(model_id)

		class _Wrapped(nn.Module):
			def __init__(self, base_model: nn.Module, n_labels: int):
				super().__init__()
				self.base = base_model
				hid = getattr(getattr(base_model, "config", None), "hidden_size", None) or getattr(getattr(base_model, "config", None), "n_embd", None)
				if hid is None:
					raise ValueError("Could not infer hidden size for classification head.")
				self.classifier = nn.Linear(int(hid), int(n_labels))

			def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
				out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
				h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
				if isinstance(h, torch.Tensor) and h.dim() == 3:
					if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
						m = attention_mask.unsqueeze(-1).float()
						hp = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
					else:
						hp = h[:, 0, :]
				else:
					hp = h
				logits = self.classifier(hp)
				loss = None
				if labels is not None:
					loss = nn.CrossEntropyLoss()(logits, labels)
				return type("Out", (), {"logits": logits, "loss": loss})()

		model = _Wrapped(base, int(num_classes)).to(device)
		layer_names = [f"base.{n}" for n in _pick_transformer_blocks_generic(base)]
		return model, layer_names

	raise ValueError(f"Unknown text model kind: {kind}")


def resolve_tokenizer_name(model_kind_or_id: str) -> str:
	k = str(model_kind_or_id).strip()
	kl = k.lower()
	if kl in {"distilbert", "distilbert-base-uncased"}:
		return "distilbert-base-uncased"
	smollm_ids = {
		"smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
		"smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
		"smollm": "HuggingFaceTB/SmolLM2-135M",
		"smollm2": "HuggingFaceTB/SmolLM2-135M",
	}
	if kl in smollm_ids:
		return str(smollm_ids[kl])
	return k


def repr_from_activation(act: Any) -> Optional[torch.Tensor]:
	if not isinstance(act, torch.Tensor):
		return None
	x = act.float() if not act.is_floating_point() else act
	if x.dim() == 4:
		x = x.mean(dim=(2, 3))
	elif x.dim() == 3:
		x = x[:, 0, :]
	elif x.dim() == 2:
		pass
	else:
		x = x.view(x.shape[0], -1)
	return x


def is_text_batch(batch: Any) -> bool:
	return isinstance(batch, Mapping) and ("input_ids" in batch or "attention_mask" in batch or "labels" in batch)


def move_to_device(x: Any, device: torch.device) -> Any:
	if isinstance(x, torch.Tensor):
		return x.to(device)
	if isinstance(x, Mapping):
		return {k: move_to_device(v, device) for k, v in x.items()}
	if isinstance(x, (tuple, list)):
		return type(x)(move_to_device(v, device) for v in x)
	return x


def ensure_dir(p: str) -> str:
	path = os.path.abspath(str(p))
	os.makedirs(path, exist_ok=True)
	return path

