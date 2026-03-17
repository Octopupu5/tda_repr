import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class SimplicialComplex:
	"""
	Minimal simplicial complex representation with oriented boundary matrices.
	Vertices are assumed to be hashable and orderable; simplices are stored as sorted tuples.
	"""

	def __init__(self, simplices: Iterable[Sequence[int]], closure: bool = True) -> None:
		self._simplices_by_dim: Dict[int, List[Tuple[int, ...]]] = {}
		self._index_by_simplex: Dict[int, Dict[Tuple[int, ...], int]] = {}

		all_simplices: Dict[Tuple[int, ...], None] = {}
		for s in simplices:
			t = tuple(sorted(s))
			all_simplices[t] = None
			if closure:
				for k in range(1, len(t)):
					for face in _faces_of_simplex(t, k=len(t) - k):
						all_simplices[face] = None

		# organize by dimension
		by_dim: Dict[int, List[Tuple[int, ...]]] = {}
		for s in all_simplices.keys():
			d = len(s) - 1
			by_dim.setdefault(d, []).append(s)
		for d, lst in by_dim.items():
			lst.sort()
			self._simplices_by_dim[d] = lst
			self._index_by_simplex[d] = {s: i for i, s in enumerate(lst)}

	def max_dim(self) -> int:
		return max(self._simplices_by_dim.keys(), default=-1)

	def num_simplices(self, dim: int) -> int:
		return len(self._simplices_by_dim.get(dim, []))

	def simplices(self, dim: int) -> List[Tuple[int, ...]]:
		return self._simplices_by_dim.get(dim, [])

	def index_of(self, dim: int, simplex: Sequence[int]) -> int:
		s = tuple(sorted(simplex))
		return self._index_by_simplex[dim][s]

	def has_simplex(self, dim: int, simplex: Sequence[int]) -> bool:
		s = tuple(sorted(simplex))
		return s in self._index_by_simplex.get(dim, {})


def _faces_of_simplex(simplex: Tuple[int, ...], k: int) -> Iterable[Tuple[int, ...]]:
	"""
	Yield all faces of codimension k (i.e., dimension len(simplex)-1-k).
	For closure we need k >= 1 faces; here we use it to add all lower faces if needed.
	"""
	n = len(simplex)
	if k <= 0 or k >= n:
		return
	# generate by removing any k vertices
	for idxs in combinations(range(n), n - k):
		yield tuple(simplex[i] for i in idxs)


def boundary_matrix(sc: SimplicialComplex, q: int):
	"""
	Return the oriented boundary matrix for dimension q.
	It maps q-simplices to (q-1)-simplices.
	CSR sparse matrix is returned; empty shapes are supported.
	"""
	nq = sc.num_simplices(q)
	nqm1 = sc.num_simplices(q - 1) if q - 1 >= 0 else 0
	if q <= 0 or nq == 0 or nqm1 == 0:
		return sp.csr_matrix((max(nqm1, 0), max(nq, 0)), dtype=np.float64)

	rows: List[int] = []
	cols: List[int] = []
	data: List[float] = []

	q_simplices = sc.simplices(q)
	for col, sigma in enumerate(q_simplices):
		# Oriented boundary with alternating signs by removed vertex position.
		for j in range(len(sigma)):
			face = tuple(sigma[:j] + sigma[j + 1 :])
			row = sc.index_of(q - 1, face)
			sign = -1.0 if (j % 2 == 1) else 1.0
			rows.append(row)
			cols.append(col)
			data.append(sign)

	B = sp.csr_matrix((data, (rows, cols)), shape=(nqm1, nq), dtype=np.float64)
	return B


def up_laplacian(sc: SimplicialComplex, q: int):
	"""
	Up Laplacian on q-simplices.
	"""
	Bqp1 = boundary_matrix(sc, q + 1)
	if Bqp1.shape == (0, 0) or Bqp1.shape[1] == 0:
		return sp.csr_matrix((sc.num_simplices(q), sc.num_simplices(q)), dtype=np.float64)
	return (Bqp1 @ Bqp1.T).tocsr()


def down_laplacian(sc: SimplicialComplex, q: int):
	"""
	Down Laplacian on q-simplices.
	It is zero for q equal to 0.
	"""
	if q <= 0:
		return sp.csr_matrix((sc.num_simplices(q), sc.num_simplices(q)), dtype=np.float64)
	Bq = boundary_matrix(sc, q)
	if Bq.shape == (0, 0) or Bq.shape[1] == 0:
		return sp.csr_matrix((sc.num_simplices(q), sc.num_simplices(q)), dtype=np.float64)
	return (Bq.T @ Bq).tocsr()


def hodge_laplacian(sc: SimplicialComplex, q: int):
	"""
	Full Hodge Laplacian on q-simplices: down + up.
	"""
	return down_laplacian(sc, q) + up_laplacian(sc, q)


def _partition_indices_for_q(L: SimplicialComplex, K: SimplicialComplex, q: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Return index arrays (idx_K, idx_C) that partition the q-simplices of L into those
	present in K and those present only in L but not in K.
	Assumes K is a subcomplex of L at dimension q.
	"""
	L_q = L.simplices(q)
	is_in_K: List[bool] = []
	for s in L_q:
		is_in_K.append(K.has_simplex(q, s))
	idx_K_list = [i for i, flag in enumerate(is_in_K) if flag]
	idx_C_list = [i for i, flag in enumerate(is_in_K) if not flag]
	return np.array(idx_K_list, dtype=np.int64), np.array(idx_C_list, dtype=np.int64)


def persistent_up_laplacian_operator(
	L: SimplicialComplex,
	K: SimplicialComplex,
	q: int,
	regularization: Optional[float] = None,
	use_ridge: bool = True,
):
	"""
	Build a LinearOperator for the up-persistent Laplacian via Schur complement:
	S equals A minus B times D pseudoinverse times C.
	The matrix up_laplacian(L, q) is block-partitioned
	by q-simplices into K block (A) and the block for simplices in L but not in K (D).
	We approximate D_pinv by either:
	- ridge regularization using inverse of D plus lambda times identity,
	- or min-norm least-squares via lsmr if use_ridge=False.

	Uses Schur-complement formulation.
	"""
	# Validate subcomplex condition at dimension q
	for s in K.simplices(q):
		if not L.has_simplex(q, s):
			raise ValueError("K is not a subcomplex of L at dimension q.")

	Delta_up_L = up_laplacian(L, q).tocsr()
	idx_K, idx_C = _partition_indices_for_q(L, K, q)
	nK = idx_K.shape[0]
	nC = idx_C.shape[0]
	if nK == 0:
		raise ValueError("No q-simplices from K found in L.")
	if nC == 0:
		# Nothing to eliminate; persistent up-Laplacian equals A
		A = Delta_up_L[idx_K[:, None], idx_K]
		return spla.aslinearoperator(A)

	A = Delta_up_L[idx_K[:, None], idx_K].tocsr()
	B = Delta_up_L[idx_K[:, None], idx_C].tocsr()
	C = Delta_up_L[idx_C[:, None], idx_K].tocsr()
	D = Delta_up_L[idx_C[:, None], idx_C].tocsr()

	# Set regularization if needed
	if use_ridge:
		if regularization is None:
			diag = D.diagonal() if D.shape[0] > 0 else np.array([1.0])
			scale = float(np.mean(diag)) if diag.size > 0 else 1.0
			regularization = max(1e-8 * scale, 1e-12)
		Ic = sp.identity(nC, format="csr", dtype=np.float64)
		D_reg = (D + regularization * Ic).tocsr()
		solver = spla.factorized(D_reg.tocsc()) if nC > 0 else (lambda y: y)

	def _matvec(x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float64).reshape(-1)
		t1 = A @ x
		if nC == 0:
			return t1
		y = C @ x
		if use_ridge:
			z = solver(y)
		else:
			z, *_ = spla.lsmr(D, y, atol=1e-7, btol=1e-7, maxiter=None)
		return t1 - B @ z

	def _rmatvec(x: np.ndarray) -> np.ndarray:
		# Operator is symmetric
		return _matvec(x)

	return spla.LinearOperator(shape=(nK, nK), matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64)


def eigs_persistent_up(
	L: SimplicialComplex,
	K: SimplicialComplex,
	q: int,
	k: int = 6,
	which: str = "LM",
	return_eigenvectors: bool = False,
	regularization: Optional[float] = None,
	use_ridge: bool = True,
	tol: float = 1e-6,
	maxiter: Optional[int] = None,
):
	"""
	Compute k eigenvalues (and optionally eigenvectors) of up-persistent Laplacian.
	Default is largest magnitude ('LM') for robustness; pass which='SM' or 'SA' for smallest if supported.

	Uses Schur-complement formulation for up-persistent Laplacian.
	"""
	op = persistent_up_laplacian_operator(L, K, q, regularization=regularization, use_ridge=use_ridge)
	k_eff = min(k, max(op.shape[0] - 1, 1))
	try:
		w, v = spla.eigsh(op, k=k_eff, which=which, return_eigenvectors=True, tol=tol, maxiter=maxiter)
	except spla.ArpackNoConvergence as e:  # pragma: no cover - numeric
		# Return the eigenpairs that did converge
		w = getattr(e, "eigenvalues", None)
		v = getattr(e, "eigenvectors", None)
		if w is None:
			raise
		if v is None:
			v = None
	if return_eigenvectors:
		return w, v
	return w


def persistent_down_laplacian_operator(
	L: SimplicialComplex,
	K: SimplicialComplex,
	q: int,
	regularization: Optional[float] = None,
	use_ridge: bool = True,
):
	"""
	Build a LinearOperator for the down part of the persistent Laplacian on C_q(K).

	In this implementation of persistent Laplacian for a pair K subset L,
	the "down" term is the ordinary down Laplacian of K:

	  down part is computed from the boundary matrix of K at dimension q,

	while the pair K subset L affects only the "up" term via a Schur complement.

	We keep the (L, K) signature for backward compatibility; L is not used.
	"""
	_ = (L, regularization, use_ridge)  # L is intentionally unused
	Delta_down_K = down_laplacian(K, q).tocsr()
	return spla.aslinearoperator(Delta_down_K)


def persistent_laplacian_operator(
	L: SimplicialComplex,
	K: SimplicialComplex,
	q: int,
	regularization: Optional[float] = None,
	use_ridge: bool = True,
):
	"""
	Full persistent Laplacian as a LinearOperator: down part on K plus up-persistent part.
	Its nullity equals the persistent Betti number for degree q.
	"""
	op_up = persistent_up_laplacian_operator(L, K, q, regularization=regularization, use_ridge=use_ridge)
	op_down = persistent_down_laplacian_operator(L, K, q, regularization=regularization, use_ridge=use_ridge)

	def _matvec(x: np.ndarray) -> np.ndarray:
		return op_up.matvec(x) + op_down.matvec(x)

	def _rmatvec(x: np.ndarray) -> np.ndarray:
		return op_up.rmatvec(x) + op_down.rmatvec(x)

	return spla.LinearOperator(shape=op_up.shape, matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64)


def eigs_persistent(
	L: SimplicialComplex,
	K: SimplicialComplex,
	q: int,
	k: int = 6,
	which: str = "LM",
	return_eigenvectors: bool = False,
	regularization: Optional[float] = None,
	use_ridge: bool = True,
	tol: float = 1e-6,
	maxiter: Optional[int] = None,
):
	"""
	Compute k eigenvalues (and optionally vectors) of the full persistent Laplacian.
	"""
	op = persistent_laplacian_operator(L, K, q, regularization=regularization, use_ridge=use_ridge)
	k_eff = min(k, max(op.shape[0] - 1, 1))
	try:
		w, v = spla.eigsh(op, k=k_eff, which=which, return_eigenvectors=True, tol=tol, maxiter=maxiter)
	except spla.ArpackNoConvergence as e:  # pragma: no cover - numeric
		w = getattr(e, "eigenvalues", None)
		v = getattr(e, "eigenvectors", None)
		if w is None:
			raise
		if v is None:
			v = None
	if return_eigenvectors:
		return w, v
	return w

def eigs_hodge(
	sc: SimplicialComplex,
	q: int,
	k: int = 6,
	which: str = "LM",
	return_eigenvectors: bool = False,
	tol: float = 1e-6,
	maxiter: Optional[int] = None,
):
	"""
	Compute k eigenvalues of the full Hodge Laplacian at degree q.
	"""
	Delta_q = hodge_laplacian(sc, q).tocsr()
	k_eff = min(k, max(Delta_q.shape[0] - 1, 1))
	try:
		w, v = spla.eigsh(Delta_q, k=k_eff, which=which, return_eigenvectors=True, tol=tol, maxiter=maxiter)
	except spla.ArpackNoConvergence as e:  # pragma: no cover - numeric
		w = getattr(e, "eigenvalues", None)
		v = getattr(e, "eigenvectors", None)
		if w is None:
			raise
		if v is None:
			v = None
	if return_eigenvectors:
		return w, v
	return w
