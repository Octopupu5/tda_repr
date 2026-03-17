from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.metrics.pairwise import pairwise_distances


def _numpy_pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
	"""
	Compute Euclidean pairwise distances between rows of A and B using only NumPy.
	Returns matrix of shape (A.shape[0], B.shape[0]).
	"""
	a_sq = np.sum(A * A, axis=1, keepdims=True)
	b_sq = np.sum(B * B, axis=1, keepdims=True).T
	cross = A @ B.T
	d2 = np.maximum(a_sq + b_sq - 2.0 * cross, 0.0)
	return np.sqrt(d2)


def _cpu_pairwise_distances(B: np.ndarray, A: np.ndarray) -> np.ndarray:
	"""
	CPU pairwise distance computation: prefer sklearn, then scipy, then NumPy fallback.
	Returns distances of shape (B.shape[0], A.shape[0]).
	"""
	try:
		return pairwise_distances(B, A, n_jobs=-1)
	except Exception:
		pass
	try:
		return scipy_cdist(B, A)
	except Exception:
		pass
	return _numpy_pairwise_distances(B, A)


def pdist_gpu(a: np.ndarray, b: np.ndarray, device: str = "cuda:0") -> np.ndarray:
	"""
	Chunked GPU computation of pairwise distances between rows of a and b using torch.cdist.
	"""
	A = torch.tensor(a, dtype=torch.float64)
	B = torch.tensor(b, dtype=torch.float64)

	size_gb = (A.shape[0] + B.shape[0]) * max(1, A.shape[1]) / 1e9
	max_size = 0.2
	parts = int(size_gb / max_size) + 1 if size_gb > max_size else 1

	pdist = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
	At = A.to(device)
	try:
		for p in range(parts):
			i1 = int(p * B.shape[0] / parts)
			i2 = int((p + 1) * B.shape[0] / parts)
			i2 = min(i2, B.shape[0])

			Bt = B[i1:i2].to(device)
			pt = torch.cdist(At, Bt)
			pdist[:, i1:i2] = pt.detach().cpu().numpy()

			del Bt, pt
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
	finally:
		del At
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	return pdist


def sep_dist(a: np.ndarray, b: np.ndarray, pdist_device: str = "cpu") -> np.ndarray:
	"""
	Build the block distance matrix used for cross barcodes.
	Top-left (a,a) block is zeroed.
	Cross (b,a) block is actual distances; (b,b) block is distances within b.
	"""
	if pdist_device == "cpu":
		d1 = _cpu_pairwise_distances(b, a)
		d2 = _cpu_pairwise_distances(b, b)
	else:
		d1 = pdist_gpu(b, a, device=pdist_device)
		d2 = pdist_gpu(b, b, device=pdist_device)

	s = a.shape[0] + b.shape[0]
	apr_d = np.zeros((s, s), dtype=d1.dtype)
	apr_d[a.shape[0] :, : a.shape[0]] = d1
	apr_d[a.shape[0] :, a.shape[0] :] = d2
	return apr_d


def _lower_triangular_vector(D: np.ndarray) -> np.ndarray:
	return D[np.tril_indices(D.shape[0], k=-1)]


def barc2array(barc: dict) -> np.ndarray:
	keys = sorted(barc.keys())
	arr: List[np.ndarray] = []
	for k in keys:
		res = np.zeros((len(barc[k]), 2), dtype="<f4")
		for idx, elem in enumerate(barc[k]):
			res[idx, 0] = elem[0]
			res[idx, 1] = elem[1]
		arr.append(res)
	return np.array(arr, dtype=object)


def _compute_barcodes_from_distance(D: np.ndarray, dim: int) -> np.ndarray:
	"""
	Compute persistence barcodes from a symmetric distance matrix D up to homology 'dim'.
	Returns object array [H0, H1, ...].
	"""
	try:
		res = ripser(D, distance_matrix=True, maxdim=dim)
		dgms: Sequence[np.ndarray] = res.get("dgms", [])
		out = []
		for i in range(dim + 1):
			if i < len(dgms) and dgms[i] is not None:
				out.append(np.asarray(dgms[i], dtype="<f4"))
			else:
				out.append(np.zeros((0, 2), dtype="<f4"))
		return np.array(out, dtype=object)
	except Exception as e:
		raise RuntimeError("Need ripser installed to compute barcodes.") from e


def plot_barcodes(
	arr: np.ndarray,
	color_list: List[str] = ["deepskyblue", "limegreen", "darkkhaki"],
	dark_color_list: Optional[List[str]] = None,
	title: str = "",
	hom: Optional[Sequence[int]] = None,
) -> None:
	if dark_color_list is None:
		dark_color_list = color_list

	sh = arr.shape[0]
	step = 0
	if len(color_list) < sh:
		color_list = color_list * sh

	for i in range(sh):
		if hom is not None and i not in hom:
			continue
		barc = arr[i].copy()
		lengths = np.subtract(barc[:, 1], barc[:, 0])
		sorted_lengths = np.sort(lengths)
		nbarc = sorted_lengths.shape[0]
		topk = set(sorted_lengths[-3:].tolist()) if nbarc >= 3 else set(sorted_lengths.tolist())

		plt.plot(barc[0], np.ones(2) * step, color=color_list[i], label=f"H{i}")
		for b in barc:
			seg_color = dark_color_list[i] if (b[1] - b[0]) in topk else color_list[i]
			plt.plot(b, np.ones(2) * step, seg_color)
			step += 1

	plt.xlabel("$\\epsilon$ (time)")
	plt.ylabel("segment")
	plt.title(title)
	plt.legend(loc="lower right")
	plt.rcParams["figure.figsize"] = [6, 4]


def count_cross_barcodes(
	cloud_1: np.ndarray,
	cloud_2: np.ndarray,
	dim: int,
	title: str = "",
	is_plot: bool = False,
	pdist_device: str = "cpu",
) -> np.ndarray:
	D = sep_dist(cloud_1, cloud_2, pdist_device=pdist_device)
	m = D[cloud_1.shape[0] :, : cloud_1.shape[0]].mean()
	D[: cloud_1.shape[0], : cloud_1.shape[0]] = 0.0
	D[D < m * 1e-6] = 0.0

	# ripser fallback requires symmetric matrix
	n_a = cloud_1.shape[0]
	if D.shape[0] > n_a:
		D[:n_a, n_a:] = D[n_a:, :n_a].T

	barcodes = _compute_barcodes_from_distance(D, dim=dim)
	if is_plot:
		plot_barcodes(barcodes, title=title)
		plt.show()
	return barcodes


def calc_cross_barcodes(
	cloud_1: np.ndarray,
	cloud_2: np.ndarray,
	batch_size1: int = 4000,
	batch_size2: int = 200,
	pdist_device: str = "cpu",
	dim: int = 1,
	is_plot: bool = False,
	random_state: Optional[int] = None,
) -> np.ndarray:
	rng = np.random.default_rng(random_state)

	# Swap for consistency with the original implementation
	cloud_1, cloud_2 = cloud_2, cloud_1
	batch_size1, batch_size2 = batch_size2, batch_size1

	batch_size1 = min(batch_size1, cloud_1.shape[0])
	batch_size2 = min(batch_size2, cloud_2.shape[0])

	idx1 = rng.choice(cloud_1.shape[0], batch_size1, replace=False)
	idx2 = rng.choice(cloud_2.shape[0], batch_size2, replace=False)
	cl_1 = cloud_1[idx1]
	cl_2 = cloud_2[idx2]

	return count_cross_barcodes(cl_1, cl_2, dim=dim, is_plot=is_plot, title="", pdist_device=pdist_device)


def get_score(elem: np.ndarray, h_idx: int, kind: str = "") -> float:
	if elem.shape[0] < h_idx + 1:
		return 0.0

	barc = elem[h_idx]
	lengths = np.subtract(barc[:, 1], barc[:, 0])
	bsorted = np.sort(lengths)

	if kind == "nbarc":
		return float(bsorted.shape[0])
	if kind == "largest":
		return float(bsorted[-1]) if bsorted.size > 0 else 0.0
	if kind == "quantile":
		if bsorted.size == 0:
			return 0.0
		idx = int(0.976 * len(bsorted))
		idx = min(max(idx, 0), len(bsorted) - 1)
		return float(bsorted[idx])
	if kind == "sum_length":
		return float(np.sum(bsorted))
	if kind == "sum_sq_length":
		return float(np.sum(bsorted ** 2))

	raise ValueError("Unknown kind of score")


def mtopdiv(
	P: np.ndarray,
	Q: np.ndarray,
	batch_size1: int = 1000,
	batch_size2: int = 10000,
	n: int = 20,
	pdist_device: str = "cpu",
	is_plot: bool = False,
	random_state: Optional[int] = None,
) -> float:
	rng = np.random.default_rng(random_state)
	seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32).tolist()
	barcs = [
		calc_cross_barcodes(
			P,
			Q,
			batch_size1=batch_size1,
			batch_size2=batch_size2,
			pdist_device=pdist_device,
			dim=1,
			is_plot=is_plot,
			random_state=int(seeds[i]),
		)
		for i in range(n)
	]
	scores = [get_score(x, 1, "sum_length") for x in barcs]
	return float(np.mean(scores)) if len(scores) > 0 else 0.0
