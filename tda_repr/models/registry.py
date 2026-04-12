from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

def _require_transformers() -> tuple[object, object, object, object]:
	"""
	Import HuggingFace Transformers lazily.

	CI for this repository does not install `transformers` by default, and most
	CV experiments/tests do not need it. We only require it when constructing
	transformer-based model entries.
	"""
	try:
		from transformers import AutoConfig, AutoModel, DistilBertConfig, DistilBertModel  # type: ignore
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"Optional dependency `transformers` is required for transformer models "
			"(e.g. kind='distilbert' or 'smollm'). Install it with `pip install transformers`."
		) from e
	return AutoConfig, AutoModel, DistilBertConfig, DistilBertModel


PreprocessFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ModelInfo:
	name: str
	family: str  # "mlp" | "cnn" | "transformer"
	model: nn.Module
	preprocess: Optional[PreprocessFn]
	layer_names: List[str]


def _make_mlp(in_ch: int = 1, num_classes: int = 10) -> nn.Module:
	return nn.Sequential(
		nn.Flatten(),
		nn.Linear(in_ch * 28 * 28, 512),
		nn.ReLU(),
		nn.Linear(512, 256),
		nn.ReLU(),
		nn.Linear(256, num_classes),
	)


def _cv_preprocess_224(x: torch.Tensor) -> torch.Tensor:
	# x: float in [0,1], shape (N,C,H,W). Resize to 224 if needed.
	if x.dim() != 4:
		raise ValueError("Expected image tensor (N,C,H,W)")
	n, c, h, w = x.shape
	# Many torchvision backbones expect RGB. If input is grayscale, replicate channel.
	if c == 1:
		x = x.repeat(1, 3, 1, 1)
		c = 3
	if h == 224 and w == 224:
		return x
	return torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


def _pick_sequential_layers(model: nn.Sequential) -> List[str]:
	# For plain nn.Sequential models, named modules are "", "0", "1", ...
	return [k for k in dict(model.named_modules()).keys() if k]


def _pick_resnet18_layers(model: nn.Module) -> List[str]:
	# A compact but wide set of stable module paths for torchvision ResNet18.
	return [
		"conv1",
		"bn1",
		"relu",
		"maxpool",
		"layer1.0",
		"layer1.1",
		"layer2.0",
		"layer2.1",
		"layer3.0",
		"layer3.1",
		"layer4.0",
		"layer4.1",
		"avgpool",
		"fc",
	]


def _pick_features_blocks(model: nn.Module, max_per_stage: int = 3) -> List[str]:
	"""
	Pick a wider set of feature extraction taps for torchvision-style CNNs with:
	- model.features: nn.Sequential
	- optional model.avgpool / model.classifier

	We include each top-level features.<i> and (when features.<i> is a Sequential) a few block indices.
	"""
	out: List[str] = []
	features = getattr(model, "features", None)
	if not isinstance(features, nn.Sequential):
		return out

	for i, stage in enumerate(features):
		stage_name = f"features.{i}"
		out.append(stage_name)
		if isinstance(stage, nn.Sequential) and len(stage) > 0 and max_per_stage > 0:
			# pick first / middle / last blocks
			ids = sorted({0, len(stage) // 2, len(stage) - 1})
			ids = ids[: max_per_stage * 2]  # keep bounded even if len(stage) is tiny
			for j in ids[:max_per_stage]:
				if 0 <= j < len(stage):
					out.append(f"{stage_name}.{j}")

	# common heads (if present)
	if hasattr(model, "avgpool"):
		out.append("avgpool")
	classifier = getattr(model, "classifier", None)
	if isinstance(classifier, nn.Sequential):
		out.append("classifier")
		for k in dict(classifier.named_modules()).keys():
			if k:
				out.append(f"classifier.{k}")
	elif classifier is not None:
		out.append("classifier")
	return out


def _filter_existing(model: nn.Module, names: List[str]) -> List[str]:
	# Keep only module paths that actually exist in model.named_modules(), preserving order.
	# This makes the registry robust across torchvision/transformers minor version differences.
	all_named = set(dict(model.named_modules()).keys())
	out: List[str] = []
	seen = set()
	for n in names:
		if n in all_named and n not in seen:
			out.append(n)
			seen.add(n)
	return out


def _pick_distilbert_layers(model: nn.Module) -> List[str]:
	# Wider set for DistilBERT-like models: embeddings + all transformer blocks + a couple of FFN linears.
	out = ["embeddings"]
	for i in range(6):
		out.append(f"transformer.layer.{i}")
	# Add a few inner FFN projections (useful, always tensors)
	out += [
		"transformer.layer.0.ffn.lin1",
		"transformer.layer.0.ffn.lin2",
		"transformer.layer.5.ffn.lin1",
		"transformer.layer.5.ffn.lin2",
	]
	return _filter_existing(model, out)


def _pick_transformer_blocks_generic(model: nn.Module) -> List[str]:
	"""
	Best-effort taps for generic HF transformer models (encoder/decoder/causal), based on module names.
	This is primarily used for the 'smollm' registry entry, which may map to different architectures.
	"""
	all_named = set(dict(model.named_modules()).keys())

	# Prefer the prefix that yields the most block indices.
	prefixes = [
		"transformer.layer",
		"model.layers",
		"encoder.layer",
		"layers",
	]
	best_prefix = ""
	best_ids: List[int] = []
	for p in prefixes:
		ids = []
		for name in all_named:
			if not name.startswith(p + "."):
				continue
			rest = name[len(p) + 1 :]
			head = rest.split(".", 1)[0]
			if head.isdigit():
				ids.append(int(head))
		ids = sorted(set(ids))
		if len(ids) > len(best_ids):
			best_ids = ids
			best_prefix = p

	out: List[str] = []
	# Common embedding modules (if present)
	for emb_name in ("embeddings", "model.embed_tokens", "embed_tokens", "wte"):
		if emb_name in all_named:
			out.append(emb_name)
			break

	if best_prefix and best_ids:
		# pick several layers spread across depth (bounded)
		n = len(best_ids)
		k = int(min(12, n))
		if k <= 1:
			picks = [best_ids[0]]
		else:
			picks = []
			for qi in range(k):
				j = int(round(qi * (n - 1) / (k - 1)))
				picks.append(best_ids[j])
			picks = sorted(set(picks))
		for i in picks:
			out.append(f"{best_prefix}.{i}")
	# common norms/heads (if present)
	for head_name in ("model.norm", "norm", "ln_f", "lm_head", "score", "head", "classifier"):
		if head_name in all_named:
			out.append(head_name)
	return _filter_existing(model, out)


def get_model_info(kind: str, device: Optional[torch.device] = None, pretrained: bool = False) -> ModelInfo:
	"""
	Construct a model by key and return its layer names to tap.
	Models are created untrained by default to avoid network I/O.
	"""
	kind = kind.lower()
	device = device or torch.device("cpu")

	if kind == "mlp":
		model = _make_mlp()
		model.to(device)
		# layer_names are paths inside nn.Sequential; expose a wide set.
		layer_names = _pick_sequential_layers(model)
		return ModelInfo(name="mlp", family="mlp", model=model, preprocess=None, layer_names=layer_names)

	if kind == "resnet18":
		model = tvm.resnet18(weights=None if not pretrained else tvm.ResNet18_Weights.DEFAULT)
		model.to(device)
		layer_names = _filter_existing(model, _pick_resnet18_layers(model))
		return ModelInfo(name="resnet18", family="cnn", model=model, preprocess=_cv_preprocess_224, layer_names=layer_names)

	if kind in ("convnext_tiny", "convnext"):
		model = tvm.convnext_tiny(weights=None)
		model.to(device)
		layer_names = _filter_existing(model, _pick_features_blocks(model, max_per_stage=3))
		return ModelInfo(name="convnext_tiny", family="cnn", model=model, preprocess=_cv_preprocess_224, layer_names=layer_names)

	if kind in ("efficientnet_b0", "efficientnet"):
		model = tvm.efficientnet_b0(weights=None)
		model.to(device)
		layer_names = _filter_existing(model, _pick_features_blocks(model, max_per_stage=3))
		return ModelInfo(name="efficientnet_b0", family="cnn", model=model, preprocess=_cv_preprocess_224, layer_names=layer_names)

	if kind in ("distilbert", "distilbert-base-uncased"):
		AutoConfig, AutoModel, DistilBertConfig, DistilBertModel = _require_transformers()
		cfg = DistilBertConfig()
		model = DistilBertModel(cfg)
		model.to(device)
		layer_names = _pick_distilbert_layers(model)
		return ModelInfo(name="distilbert-base-uncased", family="transformer", model=model, preprocess=None, layer_names=layer_names)

	if kind in ("smollm", "smollm2", "smollm-135m"):
		AutoConfig, AutoModel, DistilBertConfig, DistilBertModel = _require_transformers()
		# Default to a local model to avoid network I/O unless pretrained was explicitly requested.
		if pretrained:
			try:
				cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")  # may fail offline
				model = AutoModel.from_config(cfg)
			except Exception:
				model = DistilBertModel(DistilBertConfig())
		else:
			model = DistilBertModel(DistilBertConfig())
		model.to(device)
		# Wide taps based on whatever architecture we ended up with.
		layer_names = _pick_transformer_blocks_generic(model)
		return ModelInfo(name="smollm", family="transformer", model=model, preprocess=None, layer_names=layer_names)

	raise ValueError(f"Unknown model kind: {kind}")
