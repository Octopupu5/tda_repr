import numpy as np

from tda_repr.spectral import (
	SimplicialComplex,
	down_laplacian,
	persistent_down_laplacian_operator,
	persistent_laplacian_operator,
	persistent_up_laplacian_operator,
	up_laplacian,
)


def op_to_dense(op) -> np.ndarray:
	n = op.shape[0]
	M = np.zeros((n, n), dtype=np.float64)
	for i in range(n):
		e = np.zeros(n, dtype=np.float64)
		e[i] = 1.0
		M[:, i] = op.matvec(e)
	return M


def schur_complement_from_dense(Delta: np.ndarray, idx_K: np.ndarray, idx_C: np.ndarray) -> np.ndarray:
	if idx_C.size == 0:
		return Delta[np.ix_(idx_K, idx_K)]
	A = Delta[np.ix_(idx_K, idx_K)]
	B = Delta[np.ix_(idx_K, idx_C)]
	C = Delta[np.ix_(idx_C, idx_K)]
	D = Delta[np.ix_(idx_C, idx_C)]
	D_dag = np.linalg.pinv(D, rcond=1e-12)
	return A - (B @ D_dag @ C)


def partition_q_simplices(L: SimplicialComplex, K: SimplicialComplex, q: int) -> tuple[np.ndarray, np.ndarray]:
	L_q = L.simplices(q)
	is_in_K = np.array([K.has_simplex(q, s) for s in L_q], dtype=bool)
	idx_K = np.nonzero(is_in_K)[0].astype(np.int64)
	idx_C = np.nonzero(~is_in_K)[0].astype(np.int64)
	return idx_K, idx_C


def test_graph_kron_like_schur_q0() -> None:
	L = SimplicialComplex([(0, 1), (1, 2)], closure=True)
	K = SimplicialComplex([(0,), (2,)], closure=True)
	op = persistent_up_laplacian_operator(L, K, q=0, use_ridge=False)
	S = op_to_dense(op)
	S_expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float64)
	assert np.allclose(S, S_expected, atol=1e-10, rtol=1e-10), (S, S_expected)


def test_up_schur_matches_pinv_q1_two_triangles() -> None:
	L = SimplicialComplex(
		[
			(0, 1),
			(1, 2),
			(2, 3),
			(0, 3),
			(0, 2),
			(0, 1, 2),
			(0, 2, 3),
		],
		closure=True,
	)
	K = SimplicialComplex([(0, 1), (1, 2), (2, 3), (0, 3)], closure=True)
	op = persistent_up_laplacian_operator(L, K, q=1, use_ridge=False)
	S_op = op_to_dense(op)
	Delta_up_L = up_laplacian(L, q=1).toarray()
	idx_K, idx_C = partition_q_simplices(L, K, q=1)
	S_pinv = schur_complement_from_dense(Delta_up_L, idx_K=idx_K, idx_C=idx_C)
	assert np.allclose(S_op, S_pinv, atol=1e-8, rtol=1e-8), (S_op, S_pinv)


def test_down_part_is_plain_down_of_K_q1() -> None:
	L = SimplicialComplex([(0, 1), (0, 2), (1, 2)], closure=True)
	K = SimplicialComplex([(0, 1), (1, 2)], closure=True)
	op_down = persistent_down_laplacian_operator(L, K, q=1)
	Dp = op_to_dense(op_down)
	Dk = down_laplacian(K, q=1).toarray()
	assert np.allclose(Dp, Dk, atol=1e-12, rtol=1e-12), (Dp, Dk)
	op_full = persistent_laplacian_operator(L, K, q=1, use_ridge=False)
	F = op_to_dense(op_full)
	assert np.allclose(F, Dk, atol=1e-12, rtol=1e-12), (F, Dk)


def main() -> None:
	test_graph_kron_like_schur_q0()
	test_up_schur_matches_pinv_q1_two_triangles()
	test_down_part_is_plain_down_of_K_q1()
	print("OK: persistent Laplacian Schur-complement checks passed.")


if __name__ == "__main__":
	main()
