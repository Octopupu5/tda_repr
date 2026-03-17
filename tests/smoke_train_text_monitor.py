import os
import time

import torch

from tda_repr import (
	BenchmarkSpec,
	ExperimentTracker,
	RepresentationMonitor,
	RepresentationMonitorConfig,
	RunStore,
	TrackerConfig,
	get_dataset,
	make_dataloaders,
)


def _maybe_login_hf() -> None:
	# Optional login for gated datasets/models. If token file is missing, proceed unauthenticated.
	try:
		from huggingface_hub import login

		token_path = os.path.expanduser("~/.hf_token")
		if os.path.exists(token_path):
			with open(token_path, "r", encoding="utf-8") as f:
				token = f.read().strip()
			if token:
				login(token)
	except Exception:
		pass


def _pick_layer_names(model) -> list[str]:
	"""
	Pick some transformer block module names that exist for hooking.
	Supports DistilBERT classification model where blocks are under 'distilbert.transformer.layer.*'.
	"""
	all_names = set(dict(model.named_modules()).keys())
	cands = [
		"distilbert.transformer.layer.0",
		"distilbert.transformer.layer.2",
		"distilbert.transformer.layer.4",
	]
	ok = [x for x in cands if x in all_names]
	if ok:
		return ok
	# Fallback for bare DistilBertModel.
	cands2 = ["transformer.layer.0", "transformer.layer.2", "transformer.layer.4"]
	return [x for x in cands2 if x in all_names]


def main() -> None:
	_maybe_login_hf()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset_key = "sst2"
	print(f"[Setup] device={device}, dataset={dataset_key}")

	bundle = get_dataset(dataset_key, root="./data", download=True, tokenizer_name="distilbert-base-uncased")
	loaders = make_dataloaders(bundle, batch_size=16, num_workers=0)
	train_loader = loaders["train"]
	val_loader = loaders["val"] or loaders["test"]

	if train_loader is None or val_loader is None:
		raise RuntimeError(f"Dataset '{dataset_key}' did not provide required splits (train/val).")

	print("[Setup] building model...")
	try:
		from transformers import AutoModelForSequenceClassification

		model = AutoModelForSequenceClassification.from_pretrained(
			"distilbert-base-uncased",
			num_labels=2,
		)
	except Exception:
		from transformers import DistilBertConfig, DistilBertForSequenceClassification

		model = DistilBertForSequenceClassification(DistilBertConfig(num_labels=2))

	model.to(device)
	print("[Setup] model ready")

	layer_names = _pick_layer_names(model)
	if not layer_names:
		raise RuntimeError("Could not find transformer layers to hook in the model.")

	monitor_cfg = RepresentationMonitorConfig(
		layer_names=layer_names,
		max_samples_per_stage=512,
		max_points_for_graph=256,
		max_points_for_mtopdiv=400,
		max_eigs=10,
		knn_k_small=5,
		knn_k_large=15,
		compute_hodge=True,
		compute_persistent=True,
		compute_mtopdiv=True,
		compute_q1_spectra=False,
		mtopdiv_runs=2,
		mtopdiv_pdist_device="cuda:0" if torch.cuda.is_available() else "cpu",
		verbose=True,
	)
	monitor = RepresentationMonitor(monitor_cfg)

	store = RunStore("runs/smoke_train_text_monitor", suffix=f"{dataset_key}_distilbert", unique=True)
	store.write_meta(
		{
			"name": "smoke_train_text_monitor",
			"dataset": dataset_key,
			"model": "distilbert-base-uncased",
			"device": str(device),
			"layer_names": layer_names,
			"monitor": monitor_cfg.__dict__,
		}
	)

	tracker = ExperimentTracker(
		monitor=monitor,
		benchmarks=[BenchmarkSpec(name=f"{dataset_key}-val", dataloader_key="val", metrics=("loss", "accuracy"))],
		store=store,
		cfg=TrackerConfig(run_dir=store.run_dir, eval_every=1, max_eval_batches=20),
	)

	opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
	loss_fn = torch.nn.CrossEntropyLoss()

	max_train_batches = 30
	max_val_batches = 10

	for epoch in range(2):
		monitor.reset_epoch()

		t_train0 = time.perf_counter()
		model.train()
		with monitor.attach(model):
			for bi, batch in enumerate(train_loader):
				if bi >= max_train_batches:
					break
				batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
				opt.zero_grad()
				out = model(**batch)
				loss = out.loss if hasattr(out, "loss") and out.loss is not None else loss_fn(out.logits, batch["labels"])
				loss.backward()
				opt.step()
				monitor.collect("train")
		train_s = time.perf_counter() - t_train0

		t_val0 = time.perf_counter()
		model.eval()
		with torch.no_grad(), monitor.attach(model):
			for bi, batch in enumerate(val_loader):
				if bi >= max_val_batches:
					break
				batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
				_ = model(**batch)
				monitor.collect("val")
		val_s = time.perf_counter() - t_val0

		out = tracker.on_epoch_end(
			epoch,
			model=model,
			dataloaders={"train": train_loader, "val": val_loader, "test": loaders.get("test")},
			loss_fn=loss_fn,
			extra={"train_s": train_s, "val_s": val_s},
		)
		print(f"[Epoch {epoch}] saved -> {store.run_dir}")
		print(f"[Bench] {out.get('bench')}")


if __name__ == "__main__":
	main()
