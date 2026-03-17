from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from collections.abc import Mapping

import sacrebleu
import torch


@dataclass
class BenchmarkSpec:
	"""
	Describe what to evaluate.
	- name: a label for logging (e.g., "mnist-val")
	- dataloader_key: which loader to use from make_dataloaders(...) output: "train"/"val"/"test"
	- kind:
	  - "classification": returns loss/accuracy and macro precision/recall/f1 (if labels available)
	  - "generation": returns bleu (requires tokenizer + model.generate)
	- metrics: list of metric keys to keep in logs
	"""

	name: str
	dataloader_key: str = "val"
	kind: str = "classification"
	metrics: Tuple[str, ...] = ("loss", "accuracy")
	tokenizer: Optional[Any] = None
	generate_kwargs: Optional[Dict[str, Any]] = None


def evaluate_classification(
	model: Any,
	dataloader: Any,
	loss_fn: Optional[Callable[[Any, Any], Any]] = None,
	device: Optional[Any] = None,
	max_batches: Optional[int] = None,
) -> Dict[str, float]:
	"""
	Evaluate a classifier on a dataloader yielding either:
	- (x, y) tuples, or
	- dicts with 'input_ids'/'attention_mask'/'labels' for transformers.
	Returns dict with loss/accuracy (if possible).
	"""
	device = device or next(model.parameters()).device
	model.eval()

	total = 0
	correct = 0
	loss_sum = 0.0
	loss_n = 0
	# confusion matrix (lazy init when we know num_classes)
	cm = None

	with torch.no_grad():
		for bi, batch in enumerate(dataloader):
			if max_batches is not None and bi >= max_batches:
				break

			out = None
			if isinstance(batch, (tuple, list)) and len(batch) >= 2:
				x, y = batch[0].to(device), batch[1].to(device)
				if isinstance(y, torch.Tensor) and y.dim() > 1 and y.shape[-1] == 1:
					y = y.view(-1)
				if isinstance(y, torch.Tensor):
					y = y.long()
				logits = model(x)
			elif isinstance(batch, Mapping):
				batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
				# Support both 'labels' and legacy 'label'
				y = batch.get("labels", None)
				if y is None and "label" in batch:
					y = batch["label"]
					batch["labels"] = y

				# Run model (if labels present, HF models will compute loss)
				out = model(**batch)
				# HF models often return objects with .logits/.loss
				logits = out.logits if hasattr(out, "logits") else out[0]
			else:
				continue

			# Mask out unknown labels (-1), common for HF test splits
			if y is not None:
				# Ensure tensor
				if not isinstance(y, torch.Tensor):
					try:
						y = torch.as_tensor(y, dtype=torch.long, device=device)
					except Exception:
						y = None

			if y is not None and isinstance(y, torch.Tensor):
				if y.dim() > 1 and y.shape[-1] == 1:
					y = y.view(-1)
				if y.dtype != torch.long:
					y = y.long()
				mask = (y >= 0)
				y_eval = y[mask]
				logits_eval = logits[mask]

				if y_eval.numel() > 0:
					pred = torch.argmax(logits_eval, dim=-1)
					total += int(y_eval.numel())
					correct += int((pred == y_eval).sum().item())

					# Confusion matrix
					if cm is None:
						# derive num_classes from logits
						num_classes = int(logits_eval.shape[-1])
						cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device="cpu")
					# move to cpu for accumulation
					y_cpu = y_eval.detach().to("cpu")
					p_cpu = pred.detach().to("cpu")
					# bincount over pairs (y, pred)
					k = cm.shape[0]
					flat = y_cpu * k + p_cpu
					counts = torch.bincount(flat, minlength=k * k).reshape(k, k)
					cm += counts

					# Loss: prefer model-provided loss if available, else compute from logits
					if out is not None and hasattr(out, "loss") and out.loss is not None:
						loss_sum += float(out.loss.item())
						loss_n += 1
					elif loss_fn is not None:
						l = loss_fn(logits_eval, y_eval)
						loss_sum += float(l.item())
						loss_n += 1

	out: Dict[str, float] = {}
	if loss_n > 0:
		out["loss"] = loss_sum / loss_n
	if total > 0:
		out["accuracy"] = correct / total
	if cm is not None:
		# per-class precision/recall/f1
		cm_f = cm.to(dtype=torch.float64)
		tp = torch.diag(cm_f)
		fp = cm_f.sum(dim=0) - tp
		fn = cm_f.sum(dim=1) - tp
		precision = tp / torch.clamp(tp + fp, min=1.0)
		recall = tp / torch.clamp(tp + fn, min=1.0)
		f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)

		out["precision_macro"] = float(precision.mean().item())
		out["recall_macro"] = float(recall.mean().item())
		out["f1_macro"] = float(f1.mean().item())
	return out


def evaluate_generation_bleu(
	model: Any,
	dataloader: Any,
	tokenizer: Any,
	device: Optional[Any] = None,
	max_batches: Optional[int] = None,
	generate_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
	"""
	Evaluate a generative model with corpus BLEU (sacrebleu).

	Expected dataloader batches: mappings that include:
	- input_ids: tokenized source/prompt
	- attention_mask: optional
	- labels: tokenized reference target (may contain -100 for ignored tokens)
	"""
	if tokenizer is None:
		raise ValueError("tokenizer is required for BLEU evaluation")
	if not hasattr(model, "generate"):
		raise AttributeError("Model must implement .generate(...) for BLEU evaluation")

	device = device or next(model.parameters()).device
	model.eval()

	hyp: list[str] = []
	ref: list[str] = []
	gen_kwargs = dict(generate_kwargs or {})

	pad_id = getattr(tokenizer, "pad_token_id", None)
	if pad_id is None:
		pad_id = 0

	with torch.no_grad():
		for bi, batch in enumerate(dataloader):
			if max_batches is not None and bi >= max_batches:
				break
			if not isinstance(batch, Mapping):
				raise TypeError("Expected batch to be a mapping with input_ids/labels")

			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
			input_ids = batch.get("input_ids", None)
			attention_mask = batch.get("attention_mask", None)
			labels = batch.get("labels", None)
			if labels is None and "label" in batch:
				labels = batch["label"]

			if input_ids is None or labels is None:
				raise KeyError("Batch must include 'input_ids' and 'labels' (or legacy 'label').")
			if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
				raise TypeError("'input_ids' and 'labels' must be torch.Tensors")

			gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
			if not isinstance(gen_ids, torch.Tensor):
				raise TypeError("model.generate(...) must return torch.Tensor token ids")

			lab = labels.detach().to("cpu")
			lab = lab.clone()
			lab[lab == -100] = int(pad_id)
			pred = gen_ids.detach().to("cpu")

			ref_texts = tokenizer.batch_decode(lab, skip_special_tokens=True)
			hyp_texts = tokenizer.batch_decode(pred, skip_special_tokens=True)

			ref.extend([t.strip() for t in ref_texts])
			hyp.extend([t.strip() for t in hyp_texts])

	if not hyp:
		return {"bleu": 0.0}
	bleu = sacrebleu.corpus_bleu(hyp, [ref])
	return {"bleu": float(bleu.score)}
