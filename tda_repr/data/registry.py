from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import medmnist
import warnings
import torch
import torchvision.datasets as tvd
import torchvision.transforms as T
import traceback
from datasets import IterableDataset, load_dataset
from medmnist import INFO
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


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
	# Silence a known torchvision+NumPy 2.4 warning emitted inside torchvision's CIFAR loader.
	# This does not affect correctness and only reduces log noise.
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
	"""
	Load Imagenette. Priority:
	1) HuggingFace frgfm/imagenette (train/validation).
	2) torchvision.datasets.Imagenette (train/val).
	"""
	tx = _cv_transforms(224)

	# 1) HF variant
	if download:
		try:
			tr = load_dataset("frgfm/imagenette", split="train")
			va = load_dataset("frgfm/imagenette", split="validation")
			return _HFImageClassificationDataset(tr, tx), _HFImageClassificationDataset(va, tx), None
		except Exception as e:
			print("[Imagenette] Failed to load HF dataset 'frgfm/imagenette':", e)

	# 2) torchvision Imagenette
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
	"""
	Load a specific MedMNIST subset (e.g. 'pathmnist', 'chestmnist') at a given target size.

	We first try to use MedMNIST+'s native size support (size argument) as described in
	the MedMNIST+ docs ([medmnist_plus](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/on_medmnist_plus.md)).
	If the installed medmnist version does not support this, we fall back to classic
	MedMNIST with a torchvision.Resize to the target size.
	"""
	try:
		data_flag = data_flag.lower()
		if data_flag not in INFO:
			raise ValueError(f"Unknown MedMNIST subset: {data_flag}")
		ds_info = INFO[data_flag]
		DS = getattr(medmnist, ds_info["python_class"])
		tx = T.ToTensor()

		# Try MedMNIST+ style: built-in resizing via size argument
		try:
			tr = DS(split="train", root=root, transform=tx, download=download, size=size)
			val = DS(split="val", root=root, transform=tx, download=download, size=size)
			te = DS(split="test", root=root, transform=tx, download=download, size=size)
			return tr, val, te
		except TypeError:
			# Older medmnist without size; fall back to manual resizing
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
		return load_dataset(path, split=split)
	return load_dataset(path, config, split=split)


def _build_text_collate(tokenizer_name: str = "distilbert-base-uncased", max_length: int = 128) -> CollateFn:
	tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
	def _collate(batch):
		texts = [x["text"] if "text" in x else x.get("sentence", "") for x in batch]
		labels = [x.get("label", -1) for x in batch]
		tok_out = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
		tok_out["labels"] = torch.tensor(labels, dtype=torch.long)
		return tok_out
	return _collate


def get_dataset(
	name: str,
	root: str = "./data",
	download: bool = True,
	tokenizer_name: str = "distilbert-base-uncased",
	medmnist_size: int = 224,
	# Kept for backward compatibility; ignored for Imagenette.
	imagenet_train_split: str = "train",
	imagenet_val_split: str = "validation",
	imagenet_test_split: str = "test",
	imagenet_hf_streaming: bool = False,
	imagenet_max_train_examples: Optional[int] = None,
	imagenet_max_val_examples: Optional[int] = None,
	imagenet_max_test_examples: Optional[int] = None,
) -> DataBundle:
	"""
	Unified dataset accessor. If named dataset unavailable locally, returns None datasets to allow user-supplied fallback.
	Supported:
	- CV: mnist, cifar10, imagenette
	  MedMNIST: medmnist (pathmnist default) or any subset flag (pathmnist, chestmnist, bloodmnist, ...)
	- NLP: ag_news, yahoo_answers_topics, sst2, trec
	"""
	key = name.lower().replace(" ", "")

	# CV
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
	# MedMNIST: either generic "medmnist" (defaults to pathmnist) or specific subset
	medmnist_flags = {
		"pathmnist",
		"chestmnist",
		"dermamnist",
		"octmnist",
		"pneumoniamnist",
		"retinamnist",
		"breastmnist",
		"bloodmnist",
		"tissuemnist",
		"organamnist",
		"organcmnist",
		"organsmnist",
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

	# NLP via HF datasets
	if key in ("ag_news", "ag news", "ag-news"):
		try:
			tr = _hf_text("ag_news", "train")
			te = _hf_text("ag_news", "test")
			return DataBundle(name="ag_news", train=tr, val=None, test=te, collate_fn=_build_text_collate(tokenizer_name))
		except Exception as e:
			print("[HF] Failed to load dataset 'ag_news':", e)
			traceback.print_exc()
			return DataBundle(name="ag_news", train=None, val=None, test=None, collate_fn=None)
	if key in ("yahoo_answers", "yahoo answers", "yahoo_answers_topics"):
		try:
			ds = _hf_text("yahoo_answers_topics", "train")
			te = _hf_text("yahoo_answers_topics", "test")
			return DataBundle(name="yahoo_answers_topics", train=ds, val=None, test=te, collate_fn=_build_text_collate(tokenizer_name))
		except Exception as e:
			print("[HF] Failed to load dataset 'yahoo_answers_topics':", e)
			traceback.print_exc()
			return DataBundle(name="yahoo_answers_topics", train=None, val=None, test=None, collate_fn=None)
	if key in ("sst2", "glue/sst2", "sst-2"):
		try:
			# Most stable option: GLUE with an explicit "sst2" config
			tr = _hf_text("glue", "train", config="sst2")
			va = _hf_text("glue", "validation", config="sst2")
			te = _hf_text("glue", "test", config="sst2")
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
			# Fallback: direct HF dataset (if available) https://huggingface.co/datasets/stanfordnlp/sst2
			try:
				tr = _hf_text("stanfordnlp/sst2", "train", config="default")
				va = _hf_text("stanfordnlp/sst2", "validation", config="default")
				te = _hf_text("stanfordnlp/sst2", "test", config="default")
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

	# Unknown dataset; user can pass their own later
	return DataBundle(name=key, train=None, val=None, test=None, collate_fn=None)


def make_dataloaders(bundle: DataBundle, batch_size: int = 32, num_workers: int = 2):
	"""
	Create PyTorch DataLoaders from a DataBundle. Works for torchvision datasets and HF datasets.
	"""
	def _loader(ds, shuffle: bool):
		if ds is None:
			return None
		# HF datasets (including streaming IterableDataset) yield dicts;
		# DataLoader can handle them directly. For IterableDataset, disable shuffle.
		if isinstance(ds, IterableDataset):
			shuffle_flag = False
			# Convert output to torch.Tensor when supported.
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
	"""
	Quick visualization of a few images from a CV dataset (MNIST, CIFAR-10, MedMNIST, ImageNet).
	Shows up to n images in a grid using matplotlib.
	"""
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

	# torchvision / medmnist typically return (img, label)
	if isinstance(batch, (list, tuple)):
		imgs = batch[0]
		labels = batch[1] if len(batch) > 1 else None
	else:
		# HF datasets and text tasks are not supported here
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
