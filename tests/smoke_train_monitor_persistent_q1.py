import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tda_repr import BenchmarkSpec, ExperimentTracker, RepresentationMonitor, RepresentationMonitorConfig, RunStore, TrackerConfig


class SimpleMLP(nn.Module):
	def __init__(self, in_dim: int = 2, hidden: int = 32, num_classes: int = 2):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, hidden)
		self.act1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden, hidden)
		self.act2 = nn.ReLU()
		self.head = nn.Linear(hidden, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.act1(self.fc1(x))
		x = self.act2(self.fc2(x))
		return self.head(x)


def make_toy_dataset(n: int = 1200, seed: int = 0):
	rng = np.random.default_rng(seed)
	n1 = n // 2
	n2 = n - n1
	x1 = rng.normal(loc=(-1.0, 0.0), scale=0.8, size=(n1, 2))
	x2 = rng.normal(loc=(+1.0, 0.2), scale=0.8, size=(n2, 2))
	X = np.concatenate([x1, x2], axis=0).astype(np.float32)
	y = np.concatenate([np.zeros(n1), np.ones(n2)], axis=0).astype(np.int64)
	perm = rng.permutation(n)
	return X[perm], y[perm]


def main() -> None:
	device = torch.device("cpu")
	X, y = make_toy_dataset(n=1600, seed=0)
	Xtr, ytr = X[:1200], y[:1200]
	Xva, yva = X[1200:], y[1200:]

	train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=128, shuffle=True)
	val_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=256, shuffle=False)

	model = SimpleMLP().to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
	crit = nn.CrossEntropyLoss()

	cfg = RepresentationMonitorConfig(
		layer_names=["fc1", "fc2"],
		max_samples_per_stage=768,
		max_points_for_graph=120,
		max_points_for_mtopdiv=300,
		max_eigs=6,
		knn_k_small=6,
		knn_k_large=12,
		compute_hodge=True,
		compute_persistent=True,
		compute_mtopdiv=False,
		compute_q1_spectra=True,
		build_triangles=True,
		max_triangles=20000,
		regularization=1e-10,
		verbose=True,
	)

	monitor = RepresentationMonitor(cfg)
	store = RunStore("runs/smoke_train_monitor_persistent_q1")
	tracker = ExperimentTracker(
		monitor=monitor,
		benchmarks=[BenchmarkSpec(name="toy-val", dataloader_key="val", metrics=("loss", "accuracy"))],
		store=store,
		cfg=TrackerConfig(run_dir=store.run_dir, eval_every=1, max_eval_batches=10),
	)

	for epoch in range(1):
		monitor.reset_epoch()
		model.train()
		with monitor.attach(model):
			for xb, yb in train_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				opt.zero_grad()
				logits = model(xb)
				loss = crit(logits, yb)
				loss.backward()
				opt.step()
				monitor.collect("train")

		model.eval()
		with torch.no_grad(), monitor.attach(model):
			for xb, _yb in val_loader:
				xb = xb.to(device)
				_ = model(xb)
				monitor.collect("val")

		out = tracker.on_epoch_end(epoch, model=model, dataloaders={"train": train_loader, "val": val_loader, "test": None}, loss_fn=crit, extra={})
		print(f"[RunDir] {store.run_dir}; keys={list(out['repr']['layers'].keys())}")


if __name__ == "__main__":
	main()
