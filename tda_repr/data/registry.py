import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import medmnist
import torch
import torchvision.datasets as tvd
import torchvision.transforms as T
from datasets import IterableDataset, load_dataset
from medmnist import INFO
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import numpy as np


CollateFn = Callable[[list], dict]


@dataclass
class DataBundle:
	name: str
	train: Optional[torch.utils.data.Dataset]
	val: Optional[torch.utils.data.Dataset]
	test: Optional[torch.utils.data.Dataset]
	collate_fn: Optional[CollateFn]


def _cv_transforms(image_size: int = 224):
	return T.Compose([
		T.Resize((image_size, image_size)),
		T.ToTensor(),
	])


def _get_mnist(root: str, download: bool = True) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
	tx = _cv_transforms(28)
	tr = tvd.MNIST(root=root, train=True, transform=tx, download=download)
	va = tvd.MNIST(root=root, train=False, transform=tx, download=download)
	return tr, va


def _get_cifar10(root: str, download: bool = True) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
	tx = _cv_transforms(224)
	warnings.filterwarnings(
		"ignore",
		message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
		category=Warning,
	)
	tr = tvd.CIFAR10(root=root, train=True, transform=tx, download=download)
	va = tvd.CIFAR10(root=root, train=False, transform=tx, download=download)
	return tr, va


class _HFImageClassificationDataset(torch.utils.data.Dataset):
	def __init__(self, ds: Any, transform: Callable[[Any], torch.Tensor], image_key: str = "image", label_key: str = "label"):
		self.ds = ds
		self.transform = transform
		self.image_key = image_key
		self.label_key = label_key

	def __len__(self) -> int:
		return int(len(self.ds))

	def __getitem__(self, idx: int):
		ex = self.ds[idx]
		x = ex[self.image_key]
		y = ex[self.label_key]
		if hasattr(y, "item"):
			y = y.item()
		return self.transform(x), int(y)


def _get_imagenette(
	root: str,
	download: bool = True,
) -> Tuple[Optional[torch.utils.data.Dataset], Optional[torch.utils.data.Dataset], Optional[torch.utils.data.Dataset]]:
	"""Loads Imagenette: HF ``frgfm/imagenette`` first, else ``torchvision.datasets.Imagenette``."""
	tx = _cv_transforms(224)

	if download:
		try:
			tr = load_dataset("frgfm/imagenette", split="train")
			va = load_dataset("frgfm/imagenette", split="validation")
			return _HFImageClassificationDataset(tr, tx), _HFImageClassificationDataset(va, tx), None
		except Exception as e:
			print("[Imagenette] Failed to load HF dataset 'frgfm/imagenette':", e)

	try:
		DS = getattr(tvd, "Imagenette", None)
		if DS is None:
			raise RuntimeError("torchvision.datasets.Imagenette is unavailable in this torchvision version.")
		tr = DS(root=root, split="train", size="full", download=download, transform=tx)
		va = DS(root=root, split="val", size="full", download=download, transform=tx)
		return tr, va, None
	except Exception as e:
		print(f"[Imagenette] Failed to load torchvision Imagenette from root='{root}': {e}")
		traceback.print_exc()
		return None, None, None


def _get_medmnist(
	root: str,
	data_flag: str = "pathmnist",
	size: int = 224,
	download: bool = True,
):
	"""MedMNIST subset at ``size``; uses dataset ``size=`` when available else ``Resize``."""
	try:
		data_flag = data_flag.lower()
		if data_flag not in INFO:
			raise ValueError(f"Unknown MedMNIST subset: {data_flag}")
		ds_info = INFO[data_flag]
		DS = getattr(medmnist, ds_info["python_class"])
		tx = T.ToTensor()

		try:
			tr = DS(split="train", root=root, transform=tx, download=download, size=size)
			val = DS(split="val", root=root, transform=tx, download=download, size=size)
			te = DS(split="test", root=root, transform=tx, download=download, size=size)
			return tr, val, te
		except TypeError:
			tx = T.Compose([T.Resize((size, size)), T.ToTensor()])
			tr = DS(split="train", root=root, transform=tx, download=download)
			val = DS(split="val", root=root, transform=tx, download=download)
			te = DS(split="test", root=root, transform=tx, download=download)
			return tr, val, te
	except Exception as e:
		print(f"[MedMNIST] Failed to load subset '{data_flag}' with size={size}: {e}")
		traceback.print_exc()
		return None, None, None


def _hf_text(path: str, split: str, config: Optional[str] = None):
	if config is None:
		try:
			return load_dataset(path, split=split, token=False)
		except TypeError:
			return load_dataset(path, split=split)
	try:
		return load_dataset(path, config, split=split, token=False)
	except TypeError:
		return load_dataset(path, config, split=split)


def _subset_hf(ds: Any, *, max_examples: int, seed: int, name: str) -> Any:
	m = int(max_examples)
	if m <= 0:
		return ds
	if isinstance(ds, IterableDataset):
		raise RuntimeError(f"Cannot subset streaming IterableDataset for {name}. Disable streaming or set max_examples=0.")
	if not hasattr(ds, "shuffle") or not hasattr(ds, "select"):
		raise RuntimeError(f"Dataset object for {name} does not support shuffle/select.")
	n = len(ds) if hasattr(ds, "__len__") else 0
	if n <= 0:
		return ds
	k = min(int(m), int(n))
	sh = ds.shuffle(seed=int(seed))
	return sh.select(range(k))


def _build_text_collate(tokenizer_name: str = "distilbert-base-uncased", max_length: int = 128) -> CollateFn:
	tn = str(tokenizer_name).strip()
	if tn.lower() == "distilbert":
		tn = "distilbert-base-uncased"
	if tn.lower() in {"smollm", "smollm2"}:
		tn = "HuggingFaceTB/SmolLM2-135M"
	try:
		tok = AutoTokenizer.from_pretrained(tn, use_fast=True, token=False)
	except TypeError:
		tok = AutoTokenizer.from_pretrained(tn, use_fast=True)
	try:
		if getattr(tok, "pad_token", None) is None:
			eos = getattr(tok, "eos_token", None)
			if eos is not None:
				tok.pad_token = eos
			else:
				tok.add_special_tokens({"pad_token": "[PAD]"})
	except Exception:
		pass
	text_key: Optional[str] = None
	text_keys_multi: Optional[tuple[str, ...]] = None
	label_key: Optional[str] = None
	label_map: dict[str, int] = {}

	def _pick_key(example: dict, preferred: list[str], contains: str) -> Optional[str]:
		for k in preferred:
			if k in example:
				return k
		for k in example.keys():
			if contains in str(k).lower():
				return str(k)
		return None

	def _collate(batch):
		nonlocal text_key, text_keys_multi, label_key, label_map
		if not batch:
			raise ValueError("Empty batch for text collate.")
		ex0 = batch[0]
		if not isinstance(ex0, dict):
			raise ValueError("Expected HF text dataset items to be dicts.")

		if text_key is None and text_keys_multi is None:
			text_key = _pick_key(
				ex0,
				[
					"text",
					"sentence",
					"question",
					"content",
					"review",
					"question_content",
					"question_title",
					"best_answer",
					"answer",
					"title",
				],
				contains="text",
			)
			if all(k in ex0 for k in ("question_title", "question_content", "best_answer")):
				text_keys_multi = ("question_title", "question_content", "best_answer")
				text_key = None
			elif all(k in ex0 for k in ("question_title", "question_content")):
				text_keys_multi = ("question_title", "question_content")
				text_key = None
			elif ("question" in ex0) and ("answer" in ex0):
				text_keys_multi = ("question", "answer")
				text_key = None
			if text_key is None and text_keys_multi is None:
				for needle in ("question", "answer", "content", "title", "review"):
					k = _pick_key(ex0, [], contains=needle)
					if k is not None:
						text_key = k
						break
		if label_key is None:
			label_key = _pick_key(
				ex0,
				["label", "labels", "coarse_label", "fine_label", "target", "topic", "category", "class"],
				contains="label",
			)
			if label_key is None:
				for needle in ("topic", "category", "class"):
					k = _pick_key(ex0, [], contains=needle)
					if k is not None:
						label_key = k
						break

		if text_key is None and text_keys_multi is None:
			raise ValueError(f"Could not infer text field from keys={list(ex0.keys())}")
		if label_key is None:
			raise ValueError(f"Could not infer label field from keys={list(ex0.keys())}")

		def _join_text(ex: dict) -> str:
			if text_keys_multi is not None:
				parts: list[str] = []
				for k in text_keys_multi:
					v = ex.get(k, "")
					s = str(v).strip()
					if s:
						parts.append(s)
				return "\n\n".join(parts)
			return str(ex.get(text_key or "", ""))

		texts = [_join_text(x) for x in batch]
		labels_raw = [x.get(label_key, None) for x in batch]
		labels: list[int] = []
		for v in labels_raw:
			if v is None:
				raise ValueError(f"Missing label key '{label_key}' in batch item.")
			if hasattr(v, "item"):
				try:
					v = v.item()
				except Exception:
					pass
			if isinstance(v, bool):
				labels.append(int(v))
			elif isinstance(v, int):
				labels.append(int(v))
			elif isinstance(v, str):
				if v not in label_map:
					label_map[v] = len(label_map)
				labels.append(int(label_map[v]))
			else:
				try:
					labels.append(int(v))
				except Exception as e:
					raise ValueError(f"Unsupported label type: {type(v)} value={v}") from e

		if any(l < 0 for l in labels):
			raise ValueError(f"Found negative labels (min={min(labels)}) for label_key='{label_key}'.")
		if labels and (0 not in set(labels)):
			mn = min(labels)
			if mn > 0:
				labels = [int(x - mn) for x in labels]
		tok_out = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
		tok_out["labels"] = torch.tensor(labels, dtype=torch.long)
		return tok_out
	return _collate


def _build_smol_summarize_generation_collate(
	tokenizer_name: str,
	*,
	max_length: int = 512,
	max_target_len: int = 96,
) -> CollateFn:
	tn = str(tokenizer_name).strip()
	tn_l = tn.lower()
	smollm_ids = {
		"smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
		"smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
		"smollm": "HuggingFaceTB/SmolLM2-135M",
		"smollm2": "HuggingFaceTB/SmolLM2-135M",
	}
	if tn_l in smollm_ids:
		tn = str(smollm_ids[tn_l])
	try:
		tok = AutoTokenizer.from_pretrained(tn, use_fast=True, token=False)
	except TypeError:
		tok = AutoTokenizer.from_pretrained(tn, use_fast=True)

	try:
		if getattr(tok, "pad_token", None) is None:
			eos = getattr(tok, "eos_token", None)
			if eos is not None:
				tok.pad_token = eos
			else:
				tok.add_special_tokens({"pad_token": "[PAD]"})
	except Exception:
		pass
	try:
		tok.padding_side = "right"
	except Exception:
		pass

	def _as_messages(ex: Any) -> list[dict]:
		if isinstance(ex, dict) and isinstance(ex.get("messages", None), list):
			return ex["messages"]
		raise ValueError("Expected example to contain a 'messages' list.")

	def _prompt_and_full_ids(messages: list[dict]) -> tuple[list[int], list[int]]:
		last_assistant = None
		for i in range(len(messages) - 1, -1, -1):
			m = messages[i]
			if isinstance(m, dict) and str(m.get("role", "")).strip().lower() == "assistant":
				last_assistant = i
				break
		if last_assistant is None:
			last_assistant = len(messages) - 1
		prompt_msgs = list(messages[:last_assistant])

		def _fmt(ms: list[dict]) -> str:
			parts = []
			for m in ms:
				role = str(m.get("role", "")).strip().lower()
				content = str(m.get("content", ""))
				parts.append(f"{role}: {content}")
			return "\n".join(parts)

		asst_msg = messages[last_assistant]
		asst_content = str(asst_msg.get("content", "") if isinstance(asst_msg, dict) else "").lstrip()
		prompt_txt = _fmt(prompt_msgs) + "\nassistant: "
		prompt_ids = tok(prompt_txt, add_special_tokens=False, truncation=False)["input_ids"]
		asst_ids = tok(asst_content, add_special_tokens=False, truncation=False)["input_ids"]
		bos = getattr(tok, "bos_token_id", None)
		if isinstance(bos, int):
			prompt_ids = [int(bos)] + [int(x) for x in prompt_ids]
		full_ids = [int(x) for x in prompt_ids] + [int(x) for x in asst_ids]
		return [int(x) for x in prompt_ids], [int(x) for x in full_ids]

	def _collate(batch: list) -> dict:
		input_ids_l: list[list[int]] = []
		labels_l: list[list[int]] = []
		for ex in batch:
			msgs = _as_messages(ex)
			prompt_ids, full_ids = _prompt_and_full_ids(msgs)
			prompt_len = int(len(prompt_ids))
			if int(max_target_len) > 0 and int(max_target_len) >= 0:
				full_ids = list(full_ids[: prompt_len + int(max_target_len)])
			labels = list(full_ids)

			if int(max_length) > 0 and len(full_ids) > int(max_length):
				drop = int(len(full_ids) - int(max_length))
				full_ids = list(full_ids[drop:])
				labels = list(labels[drop:])
				prompt_len = max(0, int(prompt_len) - int(drop))

			for i in range(min(prompt_len, len(labels))):
				labels[i] = -100

			input_ids_l.append([int(x) for x in full_ids])
			labels_l.append([int(x) for x in labels])

		features = [{"input_ids": ids} for ids in input_ids_l]
		padded = tok.pad(features, padding=True, return_tensors="pt")
		input_ids = padded["input_ids"]
		attention_mask = padded.get("attention_mask", (input_ids != int(getattr(tok, "pad_token_id", 0))).long())

		labels_out = torch.full_like(input_ids, fill_value=-100)
		for i, lab in enumerate(labels_l):
			n = min(int(len(lab)), int(labels_out.shape[1]))
			labels_out[i, :n] = torch.tensor(lab[:n], dtype=torch.long)

		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_out}

	return _collate


def get_dataset(
	name: str,
	root: str = "./data",
	download: bool = True,
	tokenizer_name: str = "distilbert-base-uncased",
	medmnist_size: int = 224,
	imagenet_train_split: str = "train",
	imagenet_val_split: str = "validation",
	imagenet_test_split: str = "test",
	imagenet_hf_streaming: bool = False,
	imagenet_max_train_examples: Optional[int] = None,
	imagenet_max_val_examples: Optional[int] = None,
	imagenet_max_test_examples: Optional[int] = None,
	nlp_max_train_examples: int = 0,
	nlp_max_val_examples: int = 0,
	nlp_max_test_examples: int = 0,
	nlp_subset_seed: int = 0,
) -> DataBundle:
	"""CV (MNIST, CIFAR-10, ImageNette, MedMNIST) and NLP (HF datasets); missing loads yield ``None`` splits."""
	key = name.lower().replace(" ", "")

	if key == "mnist":
		tr, va = _get_mnist(root, download)
		return DataBundle(name="mnist", train=tr, val=va, test=None, collate_fn=None)
	if key == "cifar10":
		tr, va = _get_cifar10(root, download)
		return DataBundle(name="cifar10", train=tr, val=va, test=None, collate_fn=None)
	if key in ("imagenette", "imagenet", "ilsvrc"):
		if key != "imagenette":
			print("[Data] Dataset key 'imagenet' is treated as 'imagenette'.")
		tr, va, te = _get_imagenette(root, download=download)
		return DataBundle(name="imagenette", train=tr, val=va, test=te, collate_fn=None)

	medmnist_flags = {
		"pathmnist",
		"bloodmnist",
	}
	if key.startswith("medmnist:"):
		flag = key.split("medmnist:", 1)[1]
	elif key == "medmnist":
		flag = "pathmnist"
	elif key in medmnist_flags:
		flag = key
	else:
		flag = None
	if flag is not None:
		tr, va, te = _get_medmnist(root, data_flag=flag, size=medmnist_size, download=download)
		return DataBundle(name=f"medmnist:{flag}", train=tr, val=va, test=te, collate_fn=None)

	if key in ("sst2", "glue/sst2", "sst-2"):
		try:
			tr = _hf_text("glue", "train", config="sst2")
			va = _hf_text("glue", "validation", config="sst2")
			te = _hf_text("glue", "test", config="sst2")
			tr = _subset_hf(tr, max_examples=int(nlp_max_train_examples), seed=int(nlp_subset_seed), name="sst2/train")
			va = _subset_hf(va, max_examples=int(nlp_max_val_examples), seed=int(nlp_subset_seed) + 1, name="sst2/val")
			te = _subset_hf(te, max_examples=int(nlp_max_test_examples), seed=int(nlp_subset_seed) + 2, name="sst2/test")
			return DataBundle(
				name="sst2",
				train=tr,
				val=va,
				test=te,
				collate_fn=_build_text_collate(tokenizer_name),
			)
		except Exception as e:
			print("[HF] Failed to load dataset 'glue/sst2' (config='sst2'):", e)
			traceback.print_exc()
			try:
				tr = _hf_text("stanfordnlp/sst2", "train", config="default")
				va = _hf_text("stanfordnlp/sst2", "validation", config="default")
				te = _hf_text("stanfordnlp/sst2", "test", config="default")
				tr = _subset_hf(tr, max_examples=int(nlp_max_train_examples), seed=int(nlp_subset_seed), name="sst2/train")
				va = _subset_hf(va, max_examples=int(nlp_max_val_examples), seed=int(nlp_subset_seed) + 1, name="sst2/val")
				te = _subset_hf(te, max_examples=int(nlp_max_test_examples), seed=int(nlp_subset_seed) + 2, name="sst2/test")
				return DataBundle(
					name="sst2",
					train=tr,
					val=va,
					test=te,
					collate_fn=_build_text_collate(tokenizer_name),
				)
			except Exception as e2:
				print("[HF] Failed to load fallback dataset 'stanfordnlp/sst2' (config='default'):", e2)
				traceback.print_exc()
				return DataBundle(name="sst2", train=None, val=None, test=None, collate_fn=None)
	if key in ("trec", "trec6", "trec-6"):
		try:
			tr = _hf_text("rungalileo/trec6", "train")
			va = _hf_text("rungalileo/trec6", "validation")
			te = _hf_text("rungalileo/trec6", "test")
			tr = _subset_hf(tr, max_examples=int(nlp_max_train_examples), seed=int(nlp_subset_seed), name="trec6/train")
			va = _subset_hf(va, max_examples=int(nlp_max_val_examples), seed=int(nlp_subset_seed) + 1, name="trec6/val")
			te = _subset_hf(te, max_examples=int(nlp_max_test_examples), seed=int(nlp_subset_seed) + 2, name="trec6/test")
			return DataBundle(
				name="trec6",
				train=tr,
				val=va,
				test=te,
				collate_fn=_build_text_collate(tokenizer_name),
			)
		except Exception as e:
			print("[HF] Failed to load dataset 'trec':", e)
			traceback.print_exc()
			return DataBundle(name="trec6", train=None, val=None, test=None, collate_fn=None)

	if key in ("smol-summarize", "smol_summarize", "smol_summarize_v0"):
		try:
			try:
				dd = load_dataset("HuggingFaceTB/smoltalk", "smol-summarize", token=False)
			except TypeError:
				dd = load_dataset("HuggingFaceTB/smoltalk", "smol-summarize")
			base = None
			if isinstance(dd, dict) and "train" in dd:
				base = dd["train"]
			elif isinstance(dd, dict) and dd:
				base = dd[sorted(dd.keys())[0]]
			else:
				raise ValueError("Unexpected dataset object from load_dataset('HuggingFaceTB/smoltalk', 'smol-summarize').")
			if base is None:
				raise RuntimeError("smol-summarize dataset returned an empty 'train' split.")

			n = int(len(base)) if hasattr(base, "__len__") else 0
			if n < 100:
				raise RuntimeError(f"smol-summarize is too small for splitting (n={n}).")
			rs = np.random.RandomState(int(nlp_subset_seed))
			idx = rs.permutation(n)

			want_test = int(nlp_max_test_examples)
			want_val = int(nlp_max_val_examples)
			want_train = int(nlp_max_train_examples)
			if want_test > 0 or want_val > 0:
				n_test = max(1, want_test) if want_test > 0 else max(1, int(0.02 * n))
				n_val = max(1, want_val) if want_val > 0 else max(1, int(0.02 * n))
			else:
				n_test = max(1, int(0.02 * n))
				n_val = max(1, int(0.02 * n))
			n_test = min(int(n_test), int(n))
			n_val = min(int(n_val), max(0, int(n) - int(n_test)))

			start = 0
			test_idx = idx[start : start + n_test].tolist()
			start += int(n_test)
			val_idx = idx[start : start + n_val].tolist()
			start += int(n_val)
			train_idx_all = idx[start:].tolist()
			if want_train > 0:
				train_idx_all = train_idx_all[: min(int(want_train), int(len(train_idx_all)))]

			te = base.select(test_idx)
			va = base.select(val_idx)
			tr = base.select(train_idx_all)

			collate = _build_smol_summarize_generation_collate(tokenizer_name, max_length=512, max_target_len=96)
			return DataBundle(name="smol-summarize", train=tr, val=va, test=te, collate_fn=collate)
		except Exception as e:
			print("[HF] Failed to load dataset 'HuggingFaceTB/smoltalk' (config='smol-summarize'):", e)
			traceback.print_exc()
			return DataBundle(name="smol-summarize", train=None, val=None, test=None, collate_fn=None)

	return DataBundle(name=key, train=None, val=None, test=None, collate_fn=None)


def make_dataloaders(bundle: DataBundle, batch_size: int = 32, num_workers: int = 2):
	"""Build train/val/test ``DataLoader`` instances from ``DataBundle`` fields."""
	def _loader(ds, shuffle: bool):
		if ds is None:
			return None
		if isinstance(ds, IterableDataset):
			shuffle_flag = False
			if hasattr(ds, "with_format"):
				try:
					ds = ds.with_format("torch")
				except Exception:
					pass
		else:
			shuffle_flag = shuffle
		return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag, num_workers=num_workers, collate_fn=bundle.collate_fn)

	return {
		"train": _loader(bundle.train, shuffle=True),
		"val": _loader(bundle.val, shuffle=False),
		"test": _loader(bundle.test, shuffle=False),
	}


def preview_cv_samples(
	name: str,
	root: str = "./data",
	download: bool = False,
	n: int = 16,
	cols: int = 4,
	split: str = "train",
):
	"""Plots up to ``n`` CV samples (MNIST/CIFAR/MedMNIST/ImageNet-class) as a matplotlib grid."""
	import matplotlib.pyplot as plt

	bundle = get_dataset(name, root=root, download=download)
	ds = None
	if split == "train":
		ds = bundle.train
	elif split == "val":
		ds = bundle.val
	elif split == "test":
		ds = bundle.test
	if ds is None:
		raise ValueError(f"No split '{split}' available for dataset '{name}'.")

	loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
	batch = next(iter(loader))

	if isinstance(batch, (list, tuple)):
		imgs = batch[0]
		labels = batch[1] if len(batch) > 1 else None
	else:
		raise ValueError("preview_cv_samples supports only CV datasets (torchvision / medmnist).")

	imgs = imgs[:n]
	if labels is not None:
		labels = labels[:n]

	cols = max(1, min(cols, n))
	rows = (n + cols - 1) // cols

	plt.figure(figsize=(cols * 2.2, rows * 2.2))
	for i in range(min(n, imgs.shape[0])):
		ax = plt.subplot(rows, cols, i + 1)
		img = imgs[i]
		if img.dim() == 3 and img.shape[0] in (1, 3):
			arr = img.permute(1, 2, 0).numpy()
			if arr.shape[2] == 1:
				ax.imshow(arr[:, :, 0], cmap="gray")
			else:
				ax.imshow(arr)
		else:
			ax.imshow(img.numpy(), cmap="gray")
		ax.axis("off")
		if labels is not None:
			lab = labels[i]
			if hasattr(lab, "item"):
				lab = lab.item()
			ax.set_title(str(lab), fontsize=8)

	plt.tight_layout()
	plt.show()
