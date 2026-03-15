from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from collections.abc import Mapping

import math
import torch


@dataclass
class BenchmarkSpec:
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
	preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
	device: Optional[Any] = None,
	max_batches: Optional[int] = None,
) -> Dict[str, float]:
	device = device or next(model.parameters()).device
	model.eval()

	total = 0
	correct = 0
	loss_sum = 0.0
	loss_n = 0

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
				if preprocess is not None:
					x = preprocess(x)
				logits = model(x)
			elif isinstance(batch, Mapping):
				batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

				y = batch.get("labels", None)
				if y is None and "label" in batch:
					y = batch["label"]
					batch["labels"] = y



				fwd = {k: v for k, v in batch.items() if k not in ("labels", "label")}
				out = model(**fwd)

				logits = out.logits if hasattr(out, "logits") else out[0]
			else:
				continue


			if y is not None:

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


				mask = (y >= 0) & (y < int(logits.shape[-1]))
				y_eval = y[mask]
				logits_eval = logits[mask]

				if y_eval.numel() > 0:
					pred = torch.argmax(logits_eval, dim=-1)
					total += int(y_eval.numel())
					correct += int((pred == y_eval).sum().item())


					if cm is None:

						num_classes = int(logits_eval.shape[-1])
						cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device="cpu")

					y_cpu = y_eval.detach().to("cpu")
					p_cpu = pred.detach().to("cpu")

					k = cm.shape[0]
					flat = y_cpu * k + p_cpu
					counts = torch.bincount(flat, minlength=k * k).reshape(k, k)
					cm += counts


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


def evaluate_generation(
	model: Any,
	dataloader: Any,
	device: Optional[Any] = None,
	max_batches: Optional[int] = None,
) -> Dict[str, float]:
	"""
	Expected dataloader batches: mappings that include either:
	- labels (prompt + reference), OR
	- gen_ref_labels for generation plus labels for optional loss computation.
	"""
	device = device or next(model.parameters()).device
	model.eval()

	loss_sum = 0.0
	loss_n = 0

	with torch.no_grad():
		for bi, batch in enumerate(dataloader):
			if max_batches is not None and bi >= max_batches:
				break
			if not isinstance(batch, Mapping):
				raise TypeError("Expected batch to be a mapping with input_ids/labels")

			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}



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
				raise KeyError("Batch must include 'input_ids' and 'labels'.")
			if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
				raise TypeError("'input_ids' and 'labels' must be torch.Tensors")


			if ("input_ids" in batch) and ("labels" in batch):
				fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
				out = model(**fwd)
				if not (hasattr(out, "loss") and out.loss is not None):
					raise ValueError("Generation benchmark expected model forward to return .loss when 'labels' are provided.")
				loss_sum += float(out.loss.item())
				loss_n += 1

	if loss_n <= 0:
		raise RuntimeError("Generation benchmark did not compute loss on any batch (expected forward pass with labels).")

	out: Dict[str, float] = {}
	if loss_n > 0:
		loss = float(loss_sum / loss_n)
		out["loss"] = loss
		out["loss_assistant_only"] = loss
		try:
			out["ppl"] = float(math.exp(loss))
		except OverflowError:
			out["ppl"] = float("inf")
	return out
