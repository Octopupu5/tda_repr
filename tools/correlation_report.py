import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Allow running as a script: ensure project root (parent of /tools) is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from tda_repr.viz.runlog import _epoch_records_to_scalar_maps, load_epoch_end_records


def _series_map(epoch_maps: List[Tuple[int, Dict[str, float]]], key: str) -> Dict[int, float]:
	out: Dict[int, float] = {}
	for e, m in epoch_maps:
		if key in m:
			out[int(e)] = float(m[key])
	return out


def _is_near_constant(values: List[float], atol: float = 1e-8, rtol: float = 1e-4) -> bool:
	if len(values) < 2:
		return True
	a = np.asarray(values, dtype=np.float64)
	vmin = float(np.min(a))
	vmax = float(np.max(a))
	scale = max(1.0, float(np.max(np.abs(a))))
	return (vmax - vmin) <= (float(atol) + float(rtol) * scale)


def _inject_spectral_lambdas(
	epoch_maps: List[Tuple[int, Dict[str, float]]],
	records: List[Dict[str, object]],
	max_lambda_k: int = 3,
) -> None:
	"""
	Augment scalar maps with lambda_k derived from spectral lists stored in repr logs.
	Adds keys:
	  repr.layers.<layer>.hodge_L_q0_lambda<k>
	  repr.layers.<layer>.hodge_L_q1_lambda<k>
	  repr.layers.<layer>.persistent_q0_lambda<k>
	  repr.layers.<layer>.persistent_q1_lambda<k>
	"""
	list_keys = {
		"hodge_L_q0_smallest": "hodge_L_q0",
		"hodge_L_q1_smallest": "hodge_L_q1",
		"persistent_q0_smallest": "persistent_q0",
		"persistent_q1_smallest": "persistent_q1",
	}
	for i, (_epoch, m) in enumerate(epoch_maps):
		if i >= len(records):
			break
		r = records[i]
		layers = ((r.get("repr", {}) or {}).get("layers", {}) or {})
		if not isinstance(layers, dict):
			continue
		for layer, info in layers.items():
			if not isinstance(info, dict):
				continue
			for src_key, family in list_keys.items():
				vals = info.get(src_key, None)
				if not isinstance(vals, list) or len(vals) == 0:
					continue
				arr = np.asarray(vals, dtype=np.float64)
				arr = arr[np.isfinite(arr)]
				if arr.size == 0:
					continue
				arr = np.sort(arr)
				for k in range(1, int(max_lambda_k) + 1):
					if arr.size < k:
						break
					key = f"repr.layers.{layer}.{family}_lambda{k}"
					m[key] = float(arr[k - 1])


def generate_correlation_report(
	run_dir: str,
	out_dir: str = "",
	min_common_epochs: int = 3,
	top_k: int = 100,
	bench_contains: str = "",
	repr_contains: str = "",
	spectral_max_lambda_k: int = 3,
	negate_bench_loss: bool = False,
) -> Dict[str, object]:
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	recs = load_epoch_end_records(metrics_path)
	if not recs:
		raise ValueError(f"No epoch_end records in: {metrics_path}")
	epoch_maps = _epoch_records_to_scalar_maps(recs)
	_inject_spectral_lambdas(epoch_maps, recs, max_lambda_k=int(spectral_max_lambda_k))

	all_keys = set()
	for _, m in epoch_maps:
		all_keys |= set(m.keys())
	bench_keys = sorted([k for k in all_keys if k.startswith("bench.")])
	repr_keys = sorted([k for k in all_keys if k.startswith("repr.layers.")])

	if bench_contains.strip():
		bench_keys = [k for k in bench_keys if bench_contains in k]
	if repr_contains.strip():
		repr_keys = [k for k in repr_keys if repr_contains in k]

	pairs = []
	for bk in bench_keys:
		sb = _series_map(epoch_maps, bk)
		# For objectives where "lower is better" (e.g., validation loss), it is often
		# more intuitive to correlate against -loss so that "better" aligns with larger values.
		if bool(negate_bench_loss) and (bk.endswith(".loss") or bk.endswith(".loss_assistant_only")):
			sb = {e: -float(v) for e, v in sb.items()}
		for rk in repr_keys:
			sr = _series_map(epoch_maps, rk)
			common = sorted(set(sb.keys()) & set(sr.keys()))
			if len(common) < int(min_common_epochs):
				continue
			x = [sb[e] for e in common]
			y = [sr[e] for e in common]
			if _is_near_constant(x) or _is_near_constant(y):
				continue
			rho, p = spearmanr(x, y)
			if not np.isfinite(rho):
				continue
			pairs.append(
				{
					"bench_key": bk,
					"repr_key": rk,
					"rho": float(rho),
					"abs_rho": float(abs(rho)),
					"p": float(p),
					"n": int(len(common)),
				}
			)

	pairs.sort(key=lambda d: d["abs_rho"], reverse=True)
	top = pairs[: int(top_k)]

	out_dir = out_dir.strip() or os.path.join(run_dir, "correlations_report")
	os.makedirs(out_dir, exist_ok=True)

	with open(os.path.join(out_dir, "all_pairs.csv"), "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=["bench_key", "repr_key", "rho", "abs_rho", "p", "n"])
		w.writeheader()
		for row in pairs:
			w.writerow(row)

	with open(os.path.join(out_dir, "top_pairs.csv"), "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=["bench_key", "repr_key", "rho", "abs_rho", "p", "n"])
		w.writeheader()
		for row in top:
			w.writerow(row)

	if top:
		labels = [f"{i+1}. {d['bench_key']} vs {d['repr_key']}" for i, d in enumerate(top[:30])]
		vals = [d["rho"] for d in top[:30]]
		fig_h = max(4.0, min(16.0, 0.35 * len(labels)))
		fig, ax = plt.subplots(1, 1, figsize=(12.0, fig_h))
		y = np.arange(len(labels))
		ax.barh(y, vals)
		ax.set_yticks(y)
		ax.set_yticklabels(labels, fontsize=8)
		ax.axvline(0.0, color="black", linewidth=1.0)
		ax.set_xlabel("Spearman rho")
		ax.set_title("Top correlation pairs")
		fig.tight_layout()
		fig.savefig(os.path.join(out_dir, "top_pairs.png"))
		plt.close(fig)

	return {"out_dir": out_dir, "pairs": int(len(pairs)), "top": int(len(top))}


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--run_dir", type=str, required=True)
	ap.add_argument("--out_dir", type=str, default="")
	ap.add_argument("--min_common_epochs", type=int, default=3)
	ap.add_argument("--top_k", type=int, default=100)
	ap.add_argument("--bench_contains", type=str, default="")
	ap.add_argument("--repr_contains", type=str, default="")
	ap.add_argument("--spectral_max_lambda_k", type=int, default=3)
	ap.add_argument("--negate_bench_loss", action="store_true", help="Correlate against -loss for bench.*.loss keys.")
	args = ap.parse_args()

	out = generate_correlation_report(
		run_dir=str(args.run_dir),
		out_dir=str(args.out_dir),
		min_common_epochs=int(args.min_common_epochs),
		top_k=int(args.top_k),
		bench_contains=str(args.bench_contains),
		repr_contains=str(args.repr_contains),
		spectral_max_lambda_k=int(args.spectral_max_lambda_k),
		negate_bench_loss=bool(args.negate_bench_loss),
	)
	print(f"[CorrelationReport] saved to: {out['out_dir']}")
	print(f"[CorrelationReport] pairs={out['pairs']} top={out['top']}")


if __name__ == "__main__":
	main()
