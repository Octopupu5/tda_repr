import argparse
import os

import matplotlib.pyplot as plt

from tda_repr.viz.runlog import find_metrics_file, get_series, list_scalar_series_keys, load_epoch_end_records


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--run_dir", type=str, required=True, help="Run directory containing metrics.jsonl")
	ap.add_argument("--key", type=str, default="", help="Flattened scalar key to plot (if empty, list keys)")
	args = ap.parse_args()

	metrics_path = find_metrics_file(args.run_dir)
	if metrics_path is None:
		raise SystemExit(f"No metrics.jsonl found in {args.run_dir}")

	recs = load_epoch_end_records(metrics_path)
	if not recs:
		raise SystemExit("No epoch_end records.")

	if not args.key:
		print("Available scalar keys:")
		for k in list_scalar_series_keys(recs):
			print(" ", k)
		print("\nExample keys:")
		print("  bench.sst2-val.accuracy")
		print("  repr.layers.distilbert.transformer.layer.0.mtopdiv_train_val")
		return

	series = get_series(recs, args.key)
	if not series:
		raise SystemExit(f"No values for key='{args.key}'")

	epochs = [e for e, _ in series]
	vals = [v for _, v in series]
	plt.figure(figsize=(7, 4))
	plt.plot(epochs, vals, marker="o")
	plt.title(f"{os.path.basename(args.run_dir)}\n{args.key}")
	plt.xlabel("epoch")
	plt.ylabel(args.key)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
