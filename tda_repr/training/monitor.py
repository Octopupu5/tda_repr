from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import time

from tda_repr.models import LayerTaps
from tda_repr.mtd import mtopdiv
from tda_repr.spectral import SimplicialComplex, eigs_hodge, eigs_persistent


@dataclass
class RepresentationMonitorConfig:
	# what to tap
	layer_names: List[str]

	# sampling limits (per epoch, per stage, across all batches)
	max_samples_per_stage: int = 2048

	# downsample before heavy computations
	max_points_for_graph: int = 256
	max_points_for_mtopdiv: int = 600
	# number of smallest eigenvalues to request (ARPACK may not converge for large k)
	max_eigs: int = 12

	# graph construction (kNN)
	knn_k_small: int = 5
	knn_k_large: int = 15
	# which stage to use as the point set for graph/Hodge/persistent computations ("train" or "val")
	graph_stage: str = "train"
	# if True, use a fixed subset of rows for graph computations across epochs (per layer, per stage)
	fixed_graph_points: bool = False
	fixed_graph_seed: int = 0
	# if True, build 2-simplices (triangles) from the kNN graph (clique complex up to dim=2)
	build_triangles: bool = False
	# cap to avoid worst-case blow-ups
	max_triangles: int = 50000

	# numerical knobs
	zero_tol: float = 1e-8
	regularization: float = 1e-10  # for persistent Laplacian (ridge)

	# which metrics to compute
	compute_hodge: bool = True
	compute_persistent: bool = True
	compute_mtopdiv: bool = True
	# In graph-only complexes (no 2-simplices), q=1 spectral computations are very expensive and
	# often not informative. By default we use fast graph Betti formulas and skip q=1 eigensolves.
	compute_q1_spectra: bool = False

	# MTopDiv params
	mtopdiv_runs: int = 3
	mtopdiv_pdist_device: str = "cpu"  # 'cpu' or 'cuda:0' etc
	# which stages to compare for MTopDiv
	mtopdiv_stage_a: str = "train"
	mtopdiv_stage_b: str = "val"
	# if True, use a fixed subset of rows for MTopDiv across epochs (per layer, per stage)
	fixed_mtopdiv_points: bool = False
	fixed_mtopdiv_seed: int = 0
	# optional PH via GUDHI (Vietoris–Rips)
	compute_gudhi: bool = False
	gudhi_max_points: int = 128
	gudhi_max_dim: int = 1  # compute H0..H_max_dim
	gudhi_max_edge_length: float = 2.0
	# if >0, compute gudhi only every N epochs (epoch % N == 0)
	gudhi_every: int = 0
	# additional stable features from persistence diagrams (computed from intervals)
	gudhi_grid_n: int = 64  # grid size for curves/landscapes/silhouettes/images
	gudhi_compute_lifetime_stats: bool = True
	gudhi_compute_betti_curve: bool = True
	gudhi_compute_landscape: bool = False
	gudhi_landscape_k: int = 3
	gudhi_compute_silhouette: bool = False
	gudhi_silhouette_q: float = 1.0
	gudhi_compute_persistence_image: bool = False
	gudhi_pi_size: int = 20  # image is pi_size x pi_size
	gudhi_pi_sigma: float = 0.15
	gudhi_pi_tau: float = 0.5

	# logging
	verbose: bool = True


def _first_tensor(x: Any) -> Optional["np.ndarray"]:
	"""
	Try to extract a torch.Tensor-like object from a hook output and return it as numpy later.
	"""
	# tensor
	if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
		return x
	# tuple/list
	if isinstance(x, (list, tuple)):
		for it in x:
			t = _first_tensor(it)
			if t is not None:
				return t
		return None
	# dict-like
	if isinstance(x, dict):
		for key in ("last_hidden_state", "hidden_states", "logits"):
			if key in x:
				t = _first_tensor(x[key])
				if t is not None:
					return t
		for v in x.values():
			t = _first_tensor(v)
			if t is not None:
				return t
		return None
	return None


def _repr_from_tensor(t: Any) -> np.ndarray:
	"""
	Convert common activation shapes into (N, D) point cloud.
	- (N, C, H, W) -> global avg pool -> (N, C)
	- (N, T, D) -> take first token -> (N, D)
	- (N, D) -> (N, D)
	- otherwise -> flatten to (N, -1)
	"""
	# torch tensor -> numpy float32
	if not isinstance(t, torch.Tensor):
		raise TypeError("Expected torch.Tensor")

	with torch.no_grad():
		x = t.detach()
		if x.is_floating_point() is False:
			x = x.float()
		# (N,C,H,W)
		if x.dim() == 4:
			x = x.mean(dim=(2, 3))
		# (N,T,D)
		elif x.dim() == 3:
			x = x[:, 0, :]
		# (N, D) ok
		elif x.dim() == 2:
			pass
		else:
			x = x.view(x.shape[0], -1)

		return x.cpu().numpy().astype(np.float32, copy=False)


def _subsample_rows(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
	if X.shape[0] <= n:
		return X
	ids = rng.choice(X.shape[0], size=n, replace=False)
	return X[ids]


def _checksum_indices(ids: np.ndarray) -> int:
	"""
	Stable small checksum for debugging/reproducibility checks (not cryptographic).
	"""
	ids = np.asarray(ids, dtype=np.int64).reshape(-1)
	if ids.size == 0:
		return 0
	mod = np.int64(2**32)
	return int(np.sum((ids + 1) * np.int64(1315423911)) % mod)


def _persistence_entropy(lifetimes: np.ndarray) -> float:
	l = np.asarray(lifetimes, dtype=np.float64).reshape(-1)
	l = l[l > 0]
	if l.size == 0:
		return 0.0
	p = l / float(np.sum(l))
	return float(-np.sum(p * np.log(p + 1e-12)))

def _finite_intervals(intervals: np.ndarray, max_edge_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Return (birth, death, lifetime) for finite intervals. Infinite deaths are clamped to max_edge_length.
	"""
	intervals = np.asarray(intervals, dtype=np.float64)
	if intervals.size == 0:
		return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
	b = intervals[:, 0]
	d = intervals[:, 1].copy()
	# clamp inf to max_edge_length for curve-based features
	d[~np.isfinite(d)] = float(max_edge_length)
	life = d - b
	mask = np.isfinite(life) & (life > 0)
	return b[mask], d[mask], life[mask]


def _betti_curve_from_intervals(b: np.ndarray, d: np.ndarray, grid: np.ndarray) -> np.ndarray:
	"""
	Betti curve over grid: count active intervals at each epsilon, birth <= eps < death.
	"""
	if b.size == 0:
		return np.zeros_like(grid, dtype=np.float64)
	# Broadcasting: (n,1) vs (1,m)
	active = (b[:, None] <= grid[None, :]) & (grid[None, :] < d[:, None])
	return active.sum(axis=0).astype(np.float64)


def _tent_values(b: np.ndarray, d: np.ndarray, grid: np.ndarray) -> np.ndarray:
	"""
	Tent function values for each interval across grid.
	Returns array (n_intervals, n_grid).
	"""
	if b.size == 0:
		return np.zeros((0, grid.size), dtype=np.float64)
	left = grid[None, :] - b[:, None]
	right = d[:, None] - grid[None, :]
	return np.maximum(0.0, np.minimum(left, right))


def _landscape_summaries(b: np.ndarray, d: np.ndarray, grid: np.ndarray, k: int) -> Dict[str, float]:
	tv = _tent_values(b, d, grid)
	if tv.size == 0:
		return {"landscape_l1": 0.0, "landscape_l2": 0.0, "landscape_max": 0.0}
	# sort descending per grid point and take first k layers
	k = max(1, int(k))
	# partial selection: get top-k values without full sort when possible
	# for simplicity and stability: full sort on small matrices
	top = np.sort(tv, axis=0)[::-1, :]
	L = top[:k, :]
	step = float(grid[1] - grid[0]) if grid.size >= 2 else 1.0
	l1 = float(np.sum(L) * step)
	l2 = float(np.sqrt(np.sum(L * L) * step))
	mx = float(np.max(L)) if L.size else 0.0
	return {"landscape_l1": l1, "landscape_l2": l2, "landscape_max": mx}


def _silhouette_summaries(b: np.ndarray, d: np.ndarray, life: np.ndarray, grid: np.ndarray, q: float) -> Dict[str, float]:
	tv = _tent_values(b, d, grid)
	if tv.size == 0:
		return {"silhouette_l1": 0.0, "silhouette_l2": 0.0, "silhouette_max": 0.0}
	q = float(q)
	w = np.power(life, q) if life.size else np.zeros((0,), dtype=np.float64)
	ws = float(np.sum(w))
	if ws <= 0:
		return {"silhouette_l1": 0.0, "silhouette_l2": 0.0, "silhouette_max": 0.0}
	s = (w[:, None] * tv).sum(axis=0) / ws
	step = float(grid[1] - grid[0]) if grid.size >= 2 else 1.0
	return {
		"silhouette_l1": float(np.sum(s) * step),
		"silhouette_l2": float(np.sqrt(np.sum(s * s) * step)),
		"silhouette_max": float(np.max(s)) if s.size else 0.0,
	}


def _persistence_image_summaries(
	b: np.ndarray,
	life: np.ndarray,
	max_edge_length: float,
	size: int,
	sigma: float,
	tau: float,
) -> Dict[str, float]:
	"""
	Simple persistence image over (birth, lifetime) with Gaussian kernels.
	Log only scalar summaries to keep logs small.
	"""
	if b.size == 0:
		return {"pi_l1": 0.0, "pi_l2": 0.0, "pi_max": 0.0}
	size = max(8, int(size))
	sigma = float(sigma)
	tau = float(tau)
	# grid in birth and lifetime
	x = np.linspace(0.0, float(max_edge_length), size, dtype=np.float64)
	y = np.linspace(0.0, float(max_edge_length), size, dtype=np.float64)
	xx, yy = np.meshgrid(x, y, indexing="xy")
	# weights: lifetime / (lifetime + tau)
	w = life / (life + max(tau, 1e-12))
	img = np.zeros_like(xx, dtype=np.float64)
	den = 2.0 * max(sigma, 1e-12) * max(sigma, 1e-12)
	for bi, li, wi in zip(b, life, w):
		img += wi * np.exp(-((xx - bi) ** 2 + (yy - li) ** 2) / den)
	# approximate integrals via sum * cell area
	dx = float(x[1] - x[0]) if x.size >= 2 else 1.0
	dy = float(y[1] - y[0]) if y.size >= 2 else 1.0
	area = dx * dy
	l1 = float(np.sum(np.abs(img)) * area)
	l2 = float(np.sqrt(np.sum(img * img) * area))
	mx = float(np.max(img)) if img.size else 0.0
	return {"pi_l1": l1, "pi_l2": l2, "pi_max": mx}


def _gudhi_rips_summaries(X: np.ndarray, max_dim: int, max_edge_length: float) -> Dict[str, float]:
	"""
	Compute Vietoris–Rips persistence via GUDHI and return scalar summaries per homology dimension.
	"""
	import gudhi

	X = np.asarray(X, dtype=np.float64)
	rc = gudhi.RipsComplex(points=X, max_edge_length=float(max_edge_length))
	st = rc.create_simplex_tree(max_dimension=int(max_dim) + 1)
	st.compute_persistence()

	out: Dict[str, float] = {}
	for d in range(int(max_dim) + 1):
		intervals = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64)
		b, de, life = _finite_intervals(intervals, max_edge_length=float(max_edge_length))

		out[f"gudhi_h{d}_n_finite"] = float(life.size)
		out[f"gudhi_h{d}_total_persistence"] = float(np.sum(life)) if life.size else 0.0
		out[f"gudhi_h{d}_max_persistence"] = float(np.max(life)) if life.size else 0.0
		out[f"gudhi_h{d}_entropy"] = _persistence_entropy(life)
	return out


def _knn_edges(X: np.ndarray, k: int, pdist_device: str = "cpu") -> List[Tuple[int, int]]:
	"""
	Build undirected kNN graph edges from point cloud X (N,D).
	Optional GPU distance via torch.cdist when pdist_device starts with 'cuda'.
	"""
	N = X.shape[0]
	if N <= 1 or k <= 0:
		return []
	k = min(k, N - 1)

	# distances
	use_cuda = bool(isinstance(pdist_device, str) and pdist_device.startswith("cuda") and torch.cuda.is_available())

	if use_cuda:
		with torch.no_grad():
			t = torch.from_numpy(X.astype(np.float32, copy=False)).to(pdist_device)
			D = torch.cdist(t, t).cpu().numpy()
	else:
		# NumPy distances (O(N^2) but ok for small N)
		a_sq = np.sum(X * X, axis=1, keepdims=True)
		D2 = np.maximum(a_sq + a_sq.T - 2.0 * (X @ X.T), 0.0)
		D = np.sqrt(D2)

	# build edges
	edges: set[Tuple[int, int]] = set()
	for i in range(N):
		# Exclude self by setting to +inf
		row = D[i].copy()
		row[i] = np.inf
		# partial select k nearest
		neigh = np.argpartition(row, kth=k - 1)[:k]
		for j in neigh:
			a, b = (i, int(j))
			if a == b:
				continue
			if a > b:
				a, b = b, a
			edges.add((a, b))
	return sorted(edges)


def _build_knn_complex(X: np.ndarray, k: int, pdist_device: str = "cpu"):
	N = X.shape[0]
	verts = [(i,) for i in range(N)]
	edges = _knn_edges(X, k=k, pdist_device=pdist_device)
	simplices: List[Tuple[int, ...]] = verts + edges
	return SimplicialComplex(simplices, closure=True)


def _build_knn_clique_complex_2(
	X: np.ndarray,
	k: int,
	pdist_device: str = "cpu",
	max_triangles: int = 50000,
):
	"""
	Build a clique complex up to dimension 2 from an undirected kNN edge set:
	- 0-simplices: vertices
	- 1-simplices: edges
	- 2-simplices: triangles (i,j,k) where all 3 edges are present
	"""
	N = X.shape[0]
	verts = [(i,) for i in range(N)]
	edges = _knn_edges(X, k=k, pdist_device=pdist_device)
	edge_set = set(edges)

	# adjacency
	adj: List[List[int]] = [[] for _ in range(N)]
	for a, b in edges:
		adj[a].append(b)
		adj[b].append(a)
	for i in range(N):
		adj[i].sort()

	triangles: List[Tuple[int, int, int]] = []
	# Enumerate triangles with a fixed smallest vertex i to avoid duplicates.
	for i in range(N):
		ni = adj[i]
		ln = len(ni)
		if ln < 2:
			continue
		for u in range(ln):
			j = ni[u]
			if j <= i:
				continue
			for v in range(u + 1, ln):
				k2 = ni[v]
				if k2 <= i:
					continue
				a, b = (j, k2) if j < k2 else (k2, j)
				if (a, b) in edge_set:
					triangles.append((i, a, b))
					if len(triangles) >= max_triangles:
						break
			if len(triangles) >= max_triangles:
				break
		if len(triangles) >= max_triangles:
			break

	simplices: List[Tuple[int, ...]] = verts + edges + triangles
	return SimplicialComplex(simplices, closure=True)


def _count_zeros(eigs: np.ndarray, tol: float) -> int:
	return int(np.sum(np.asarray(eigs) < tol))


def _graph_betti_from_edges(n_vertices: int, edges: List[Tuple[int, int]]) -> Tuple[int, int]:
	"""
	For an undirected graph with n vertices and edge list:
	beta0 is the number of connected components.
	beta1 is m - n + beta0.
	"""
	parent = list(range(n_vertices))
	rank = [0] * n_vertices

	def find(x: int) -> int:
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x

	def union(a: int, b: int) -> None:
		ra, rb = find(a), find(b)
		if ra == rb:
			return
		if rank[ra] < rank[rb]:
			parent[ra] = rb
		elif rank[ra] > rank[rb]:
			parent[rb] = ra
		else:
			parent[rb] = ra
			rank[ra] += 1

	for a, b in edges:
		union(a, b)

	comps = len({find(i) for i in range(n_vertices)})
	m = len(edges)
	beta0 = comps
	beta1 = m - n_vertices + comps
	return beta0, beta1


class RepresentationMonitor:
	"""
	Minimal training hook/monitor:
	- attaches forward hooks to chosen layers (via LayerTaps)
	- collects a limited number of representations per stage (train/val/test)
	- computes lightweight topo/spectral summaries at epoch end

	CPU-first, but can compute kNN distances on GPU via torch.cdist when available.
	"""

	def __init__(self, config: RepresentationMonitorConfig):
		self.cfg = config
		self._taps = None
		self._rng = np.random.default_rng(0)
		self._fixed_rng = np.random.default_rng(int(getattr(config, "fixed_graph_seed", 0)))
		self._fixed_mtopdiv_rng = np.random.default_rng(int(getattr(config, "fixed_mtopdiv_seed", 0)))
		# stage -> layer -> indices
		self._fixed_graph_idx: Dict[str, Dict[str, np.ndarray]] = {}
		self._fixed_mtopdiv_idx: Dict[str, Dict[str, np.ndarray]] = {}
		self._buf: Dict[str, Dict[str, List[np.ndarray]]] = {}
		# stage -> layer -> collected rows
		self._n: Dict[str, Dict[str, int]] = {}

	def reset_epoch(self) -> None:
		self._buf = {"train": {}, "val": {}, "test": {}}
		self._n = {"train": {}, "val": {}, "test": {}}
		# Do NOT reset fixed indices here; they must persist across epochs.

	def _get_graph_rows(self, X: np.ndarray, stage: str, layer: str) -> Tuple[np.ndarray, Dict[str, Any]]:
		"""
		Return a (possibly) downsampled point cloud for graph computations.
		If cfg.fixed_graph_points is True, reuse the same row indices across epochs.
		Also returns a small dict of debug fields to log.
		"""
		info: Dict[str, Any] = {"graph_stage": stage}
		if X.shape[0] <= 0:
			return X, info
		n = min(int(self.cfg.max_points_for_graph), int(X.shape[0]))
		if not self.cfg.fixed_graph_points:
			Xg = _subsample_rows(X, n, self._rng)
			info["graph_n"] = int(Xg.shape[0])
			return Xg, info

		per_stage = self._fixed_graph_idx.setdefault(stage, {})
		if layer not in per_stage or per_stage[layer].size == 0:
			if X.shape[0] <= n:
				ids = np.arange(X.shape[0], dtype=np.int64)
			else:
				ids = self._fixed_rng.choice(X.shape[0], size=n, replace=False).astype(np.int64)
				ids.sort()
			per_stage[layer] = ids
		ids = per_stage[layer]
		ids = ids[ids < X.shape[0]]
		Xg = X[ids]
		info["graph_n"] = int(Xg.shape[0])
		info["graph_fixed"] = True
		info["graph_fixed_checksum"] = _checksum_indices(ids)
		return Xg, info

	def _get_mtopdiv_rows(self, X: np.ndarray, stage: str, layer: str) -> Tuple[np.ndarray, Dict[str, Any]]:
		info: Dict[str, Any] = {"mtopdiv_stage": stage}
		if X.shape[0] <= 0:
			return X, info
		n = min(int(self.cfg.max_points_for_mtopdiv), int(X.shape[0]))
		if not self.cfg.fixed_mtopdiv_points:
			Xm = _subsample_rows(X, n, self._rng)
			info["mtopdiv_n"] = int(Xm.shape[0])
			return Xm, info

		per_stage = self._fixed_mtopdiv_idx.setdefault(stage, {})
		if layer not in per_stage or per_stage[layer].size == 0:
			if X.shape[0] <= n:
				ids = np.arange(X.shape[0], dtype=np.int64)
			else:
				ids = self._fixed_mtopdiv_rng.choice(X.shape[0], size=n, replace=False).astype(np.int64)
				ids.sort()
			per_stage[layer] = ids
		ids = per_stage[layer]
		ids = ids[ids < X.shape[0]]
		Xm = X[ids]
		info["mtopdiv_n"] = int(Xm.shape[0])
		info["mtopdiv_fixed"] = True
		info["mtopdiv_fixed_checksum"] = _checksum_indices(ids)
		return Xm, info

	def attach(self, model: Any):
		"""
		Context manager: attach LayerTaps to a model for the duration of the training/eval loop.
		"""

		@contextmanager
		def _ctx():
			with LayerTaps(model, self.cfg.layer_names) as taps:
				self._taps = taps
				try:
					yield self
				finally:
					self._taps = None
		return _ctx()

	def collect(self, stage: str) -> None:
		"""
		Collect current activations from attached taps.
		Call this after a forward pass.
		"""
		if self._taps is None:
			raise RuntimeError("Monitor is not attached. Use `with monitor.attach(model):`.")
		if stage not in self._buf:
			self._buf[stage] = {}
			self._n[stage] = {}

		for name in self.cfg.layer_names:
			n_layer = int(self._n[stage].get(name, 0))
			if n_layer >= int(self.cfg.max_samples_per_stage):
				continue
			out = self._taps.outputs.get(name, None)
			if out is None:
				continue
			t = _first_tensor(out)
			if t is None:
				continue
			X = _repr_from_tensor(t)
			remain = int(self.cfg.max_samples_per_stage) - n_layer
			if remain <= 0:
				continue
			X = X[:remain]
			self._buf[stage].setdefault(name, []).append(X)
			self._n[stage][name] = n_layer + int(X.shape[0])

	def _stack(self, stage: str, layer: str) -> Optional[np.ndarray]:
		chunks = self._buf.get(stage, {}).get(layer, [])
		if not chunks:
			return None
		return np.concatenate(chunks, axis=0)

	def end_epoch(self, epoch: int) -> Dict[str, Any]:
		"""
		Compute metrics for collected representations and return a nested dict.
		"""
		t_epoch0 = time.perf_counter()
		res: Dict[str, Any] = {"epoch": epoch, "layers": {}, "timing_s": {}}
		for layer in self.cfg.layer_names:
			layer_out: Dict[str, Any] = {}
			layer_time: Dict[str, float] = {}
			Xtr = self._stack("train", layer)
			Xva = self._stack("val", layer)

			# Choose which stage provides the point cloud for graph/topo computations.
			stage_key = str(self.cfg.graph_stage).lower().strip() or "train"
			if stage_key not in ("train", "val", "test"):
				stage_key = "train"
			Xgraph = self._stack(stage_key, layer)
			if Xgraph is None:
				Xgraph = Xtr

			if Xtr is not None:
				layer_out["train_n"] = int(Xtr.shape[0])
				layer_out["dim"] = int(Xtr.shape[1])
			if Xva is not None:
				layer_out["val_n"] = int(Xva.shape[0])

			# Graph/Hodge/Persistent on chosen stage point cloud
			if self.cfg.compute_hodge and Xgraph is not None and Xgraph.shape[0] >= 8:
				t0 = time.perf_counter()
				Xg, ginfo = self._get_graph_rows(Xgraph, stage=stage_key, layer=layer)
				layer_out.update(ginfo)
				# Optional PH via GUDHI (Vietoris–Rips) on the same fixed point set (or its prefix).
				gudhi_on = bool(self.cfg.compute_gudhi) and (self.cfg.gudhi_every <= 0 or (epoch % int(self.cfg.gudhi_every) == 0))
				layer_out["gudhi_on"] = bool(gudhi_on)
				if gudhi_on:
					ng = int(min(max(8, int(self.cfg.gudhi_max_points)), int(Xg.shape[0])))
					Xgh = Xg[:ng]
					layer_out["gudhi_n"] = int(Xgh.shape[0])
					layer_out["gudhi_max_dim"] = int(self.cfg.gudhi_max_dim)
					layer_out["gudhi_max_edge_length"] = float(self.cfg.gudhi_max_edge_length)
					layer_out["gudhi_grid_n"] = int(self.cfg.gudhi_grid_n)
					layer_out["gudhi_checksum"] = _checksum_indices(np.arange(int(Xgh.shape[0]), dtype=np.int64))
					try:
						tg0 = time.perf_counter()
						sums = _gudhi_rips_summaries(Xgh, max_dim=int(self.cfg.gudhi_max_dim), max_edge_length=float(self.cfg.gudhi_max_edge_length))
						layer_time["gudhi_ph"] = time.perf_counter() - tg0
						layer_out.update({k: float(v) for k, v in sums.items()})

						# Extra stable features from intervals (no extra deps beyond GUDHI itself)
						grid_n = max(16, int(self.cfg.gudhi_grid_n))
						grid = np.linspace(0.0, float(self.cfg.gudhi_max_edge_length), grid_n, dtype=np.float64)
						import gudhi
						rc = gudhi.RipsComplex(points=np.asarray(Xgh, dtype=np.float64), max_edge_length=float(self.cfg.gudhi_max_edge_length))
						st = rc.create_simplex_tree(max_dimension=int(self.cfg.gudhi_max_dim) + 1)
						st.compute_persistence()

						for d in range(int(self.cfg.gudhi_max_dim) + 1):
							intervals = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64)
							b, de, life = _finite_intervals(intervals, max_edge_length=float(self.cfg.gudhi_max_edge_length))

							if self.cfg.gudhi_compute_lifetime_stats:
								layer_out[f"gudhi_h{d}_life_mean"] = float(np.mean(life)) if life.size else 0.0
								layer_out[f"gudhi_h{d}_life_median"] = float(np.median(life)) if life.size else 0.0
								layer_out[f"gudhi_h{d}_life_std"] = float(np.std(life)) if life.size else 0.0
								mid = 0.5 * (b + de) if life.size else np.zeros((0,), dtype=np.float64)
								layer_out[f"gudhi_h{d}_midlife_mean"] = float(np.mean(mid)) if mid.size else 0.0
								layer_out[f"gudhi_h{d}_midlife_std"] = float(np.std(mid)) if mid.size else 0.0

							if self.cfg.gudhi_compute_betti_curve:
								bc = _betti_curve_from_intervals(b, de, grid)
								step = float(grid[1] - grid[0]) if grid.size >= 2 else 1.0
								layer_out[f"gudhi_h{d}_betti_auc"] = float(np.sum(bc) * step)
								layer_out[f"gudhi_h{d}_betti_max"] = float(np.max(bc)) if bc.size else 0.0

							if self.cfg.gudhi_compute_landscape:
								ls = _landscape_summaries(b, de, grid, k=int(self.cfg.gudhi_landscape_k))
								layer_out.update({f"gudhi_h{d}_{k}": float(v) for k, v in ls.items()})

							if self.cfg.gudhi_compute_silhouette:
								si = _silhouette_summaries(b, de, life, grid, q=float(self.cfg.gudhi_silhouette_q))
								layer_out.update({f"gudhi_h{d}_{k}": float(v) for k, v in si.items()})

							if self.cfg.gudhi_compute_persistence_image:
								pi = _persistence_image_summaries(
									b,
									life,
									max_edge_length=float(self.cfg.gudhi_max_edge_length),
									size=int(self.cfg.gudhi_pi_size),
									sigma=float(self.cfg.gudhi_pi_sigma),
									tau=float(self.cfg.gudhi_pi_tau),
								)
								layer_out.update({f"gudhi_h{d}_{k}": float(v) for k, v in pi.items()})
					except Exception as e:
						layer_out["gudhi_error"] = str(e)
				edges_K = _knn_edges(Xg, k=self.cfg.knn_k_small, pdist_device=self.cfg.mtopdiv_pdist_device)
				edges_L = _knn_edges(Xg, k=self.cfg.knn_k_large, pdist_device=self.cfg.mtopdiv_pdist_device)
				if self.cfg.build_triangles:
					K = _build_knn_clique_complex_2(
						Xg,
						k=self.cfg.knn_k_small,
						pdist_device=self.cfg.mtopdiv_pdist_device,
						max_triangles=self.cfg.max_triangles,
					)
					L = _build_knn_clique_complex_2(
						Xg,
						k=self.cfg.knn_k_large,
						pdist_device=self.cfg.mtopdiv_pdist_device,
						max_triangles=self.cfg.max_triangles,
					)
				else:
					K = _build_knn_complex(Xg, k=self.cfg.knn_k_small, pdist_device=self.cfg.mtopdiv_pdist_device)
					L = _build_knn_complex(Xg, k=self.cfg.knn_k_large, pdist_device=self.cfg.mtopdiv_pdist_device)
				layer_time["build_knn_complex"] = time.perf_counter() - t0

				# Hodge on L (graph at larger scale)
				n0 = L.num_simplices(0)
				k0 = min(self.cfg.max_eigs, max(n0 - 1, 1))
				t0 = time.perf_counter()
				w0 = eigs_hodge(L, q=0, k=k0, which="SA")
				layer_time["hodge_q0"] = time.perf_counter() - t0
				layer_out["hodge_L_q0_smallest"] = np.sort(w0).tolist()
				layer_out["beta0_L_est"] = _count_zeros(w0, self.cfg.zero_tol)

				n1 = L.num_simplices(1)
				if n1 >= 2:
					if self.cfg.compute_q1_spectra and L.num_simplices(2) > 0:
						k1 = min(self.cfg.max_eigs, max(n1 - 1, 1))
						t0 = time.perf_counter()
						w1 = eigs_hodge(L, q=1, k=k1, which="SA")
						layer_time["hodge_q1"] = time.perf_counter() - t0
						layer_out["hodge_L_q1_smallest"] = np.sort(w1).tolist()
						layer_out["beta1_L_est"] = _count_zeros(w1, self.cfg.zero_tol)
						layer_out["beta1_L_method"] = "spectral_zero_count(lower_bound)"
					else:
						# Fast graph Betti (no eigensolve)
						b0, b1 = _graph_betti_from_edges(n_vertices=n0, edges=edges_L)
						layer_out["beta0_L_graph"] = int(b0)
						layer_out["beta1_L_graph"] = int(b1)
						layer_out["beta1_L_est"] = int(b1)
						layer_out["beta1_L_method"] = "graph_cycle_rank(exact)"
						layer_out["graph_L_edges"] = int(len(edges_L))

				# Persistent bettis between K -> L (same points, two scales)
				if self.cfg.compute_persistent:
					# Fast path for graphs (no 2-simplices): persistent H0 rank equals beta0(L),
					# and persistent H1 rank equals beta1(K) (adding edges keeps cycle-space map injective).
					if L.num_simplices(2) == 0 and K.num_simplices(2) == 0:
						b0L, _ = _graph_betti_from_edges(n_vertices=n0, edges=edges_L)
						b0K, b1K = _graph_betti_from_edges(n_vertices=n0, edges=edges_K)
						layer_out["beta0_persistent_est"] = int(b0L)
						layer_out["beta1_persistent_est"] = int(b1K)
						layer_out["beta1_persistent_method"] = "graph_inclusion_cycle_rank(beta1(K))"
						layer_out["graph_K_edges"] = int(len(edges_K))
					else:
						# Spectral persistent Laplacian (expensive)
						t0 = time.perf_counter()
						w0p = eigs_persistent(
							L,
							K,
							q=0,
							k=min(self.cfg.max_eigs, max(K.num_simplices(0) - 1, 1)),
							which="SA",
							regularization=self.cfg.regularization,
						)
						layer_time["persistent_q0"] = time.perf_counter() - t0
						layer_out["persistent_q0_smallest"] = np.sort(w0p).tolist()
						layer_out["beta0_persistent_est"] = _count_zeros(w0p, self.cfg.zero_tol)
						layer_out["beta0_persistent_method"] = "spectral_zero_count(lower_bound)"

						if self.cfg.compute_q1_spectra and K.num_simplices(1) >= 2:
							t0 = time.perf_counter()
							w1p = eigs_persistent(
								L,
								K,
								q=1,
								k=min(self.cfg.max_eigs, max(K.num_simplices(1) - 1, 1)),
								which="SA",
								regularization=self.cfg.regularization,
							)
							layer_time["persistent_q1"] = time.perf_counter() - t0
							layer_out["persistent_q1_smallest"] = np.sort(w1p).tolist()
							layer_out["beta1_persistent_est"] = _count_zeros(w1p, self.cfg.zero_tol)
							layer_out["beta1_persistent_method"] = "spectral_zero_count(lower_bound)"

			# MTopDiv between train and val clouds at this layer
			if self.cfg.compute_mtopdiv:
				stage_a = str(self.cfg.mtopdiv_stage_a).lower().strip() or "train"
				stage_b = str(self.cfg.mtopdiv_stage_b).lower().strip() or "val"
				if stage_a not in ("train", "val", "test"):
					stage_a = "train"
				if stage_b not in ("train", "val", "test"):
					stage_b = "val"

				Xa = self._stack(stage_a, layer)
				Xb = self._stack(stage_b, layer)
				if Xa is not None and Xb is not None:
					P, pinfo = self._get_mtopdiv_rows(Xa, stage=stage_a, layer=layer)
					Q, qinfo = self._get_mtopdiv_rows(Xb, stage=stage_b, layer=layer)
					layer_out["mtopdiv_stage_a"] = stage_a
					layer_out["mtopdiv_stage_b"] = stage_b
					layer_out["mtopdiv_a_n"] = int(P.shape[0])
					layer_out["mtopdiv_b_n"] = int(Q.shape[0])
					if "mtopdiv_fixed_checksum" in pinfo:
						layer_out["mtopdiv_a_fixed_checksum"] = int(pinfo["mtopdiv_fixed_checksum"])
					if "mtopdiv_fixed_checksum" in qinfo:
						layer_out["mtopdiv_b_fixed_checksum"] = int(qinfo["mtopdiv_fixed_checksum"])

					try:
						t0 = time.perf_counter()
						score = mtopdiv(
							P,
							Q,
							batch_size1=min(400, P.shape[0]),
							batch_size2=min(400, Q.shape[0]),
							n=self.cfg.mtopdiv_runs,
							pdist_device=self.cfg.mtopdiv_pdist_device,
							is_plot=False,
							random_state=epoch,
						)
						layer_time["mtopdiv"] = time.perf_counter() - t0
						layer_out["mtopdiv_train_val"] = float(score)
					except Exception as e:
						layer_out["mtopdiv_train_val_error"] = str(e)

			res["layers"][layer] = layer_out
			if layer_time:
				res["timing_s"][layer] = layer_time

		if self.cfg.verbose:
			for layer, info in res["layers"].items():
				keys = [k for k in ("train_n", "val_n", "beta0_L_est", "beta1_L_est", "beta0_persistent_est", "beta1_persistent_est", "mtopdiv_train_val") if k in info]
				if keys:
					msg = ", ".join([f"{k}={info[k]}" for k in keys])
					print(f"[RepMonitor] epoch={epoch} layer={layer}: {msg}")

		res["timing_s"]["end_epoch_total"] = time.perf_counter() - t_epoch0
		return res
