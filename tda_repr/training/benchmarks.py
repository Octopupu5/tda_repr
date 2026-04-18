from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from collections.abc import Mapping

import math
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

				# Always run model without labels to avoid crashes on splits with unknown labels (-1)
				# and keep evaluation loss computation consistent across HF/non-HF models.
				fwd = {k: v for k, v in batch.items() if k not in ("labels", "label")}
				out = model(**fwd)
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
				# keep only valid labels (HF datasets sometimes use -1; also guard against
				# accidental num_classes mismatch to avoid shape errors in cm building)
				mask = (y >= 0) & (y < int(logits.shape[-1]))
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
					if loss_fn is not None:
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

	Expected dataloader batches: mappings that include either:
	- input_ids/attention_mask/labels (prompt + reference), OR
	- gen_input_ids/gen_attention_mask/gen_ref_labels for generation,
	  plus input_ids/attention_mask/labels for optional loss computation.
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

	loss_sum = 0.0
	loss_n = 0

	with torch.no_grad():
		for bi, batch in enumerate(dataloader):
			if max_batches is not None and bi >= max_batches:
				break
			if not isinstance(batch, Mapping):
				raise TypeError("Expected batch to be a mapping with input_ids/labels")

			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

			# Prefer prompt-only fields when provided by a collate_fn.
			# NOTE: do NOT use `or` here: torch.Tensor truthiness is ambiguous.
			input_ids = batch.get("gen_input_ids", None)
			if input_ids is None:
				input_ids = batch.get("input_ids", None)
			attention_mask = batch.get("gen_attention_mask", None)
			if attention_mask is None:
				attention_mask = batch.get("attention_mask", None)
			labels = batch.get("gen_ref_labels", None)
			if labels is None:
				labels = batch.get("labels", None)
			if labels is None and "label" in batch:
				labels = batch["label"]

			if input_ids is None or labels is None:
				raise KeyError("Batch must include 'input_ids' and 'labels' (or legacy 'label').")
			if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
				raise TypeError("'input_ids' and 'labels' must be torch.Tensors")

			# Optional loss on full input/labels (if present).
			if ("input_ids" in batch) and ("labels" in batch):
				fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
				out = model(**fwd)
				if not (hasattr(out, "loss") and out.loss is not None):
					raise ValueError("Generation benchmark expected model forward to return .loss when 'labels' are provided.")
				loss_sum += float(out.loss.item())
				loss_n += 1

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
	out = {"bleu": float(bleu.score)}
	if loss_n > 0:
		# For our generation objective, the collate_fn masks the prompt tokens with -100,
		# so this loss reflects assistant-only tokens.
		loss = float(loss_sum / loss_n)
		out["loss"] = loss
		out["loss_assistant_only"] = loss
		try:
			out["ppl"] = float(math.exp(loss))
		except OverflowError:
			out["ppl"] = float("inf")
	return out
