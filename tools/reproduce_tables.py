from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Sequence

from tools.aggregate.layer_tables_from_embeddings import write_layer_detailed_table_tex
from tools.aggregate.reproduction_paths import TABLES_DIR
from tools.aggregate.reproduction_tables import build_analysis_tex_tables
from tools.aggregate.aggregate_layer_selection_summary import write_layer_selection_summary_tex
from tools.aggregate.write_early_stopping_tables import write_both_early_stopping_tables


def _strip_table_prefix(filename: str) -> str:
	fn = str(filename)
	return fn[len("table_") :] if fn.startswith("table_") else fn


def _ensure_empty_dir(path: str) -> None:
	p = os.path.abspath(path)
	os.makedirs(p, exist_ok=True)
	for name in os.listdir(p):
		fp = os.path.join(p, name)
		if os.path.isfile(fp) or os.path.islink(fp):
			os.remove(fp)
		elif os.path.isdir(fp):
			shutil.rmtree(fp)


def main(argv: Sequence[str] | None = None) -> None:
	if argv is None:
		argv = sys.argv[1:]
	ap = argparse.ArgumentParser(description="Rebuild reproduction/tables from run corpora.")
	ap.add_argument("--runs-root", default="runs", help="Root directory with experiment runs for aggregate tables.")
	ap.add_argument(
		"--early-stop-root",
		default="runs",
		help="Root directory containing analysis/repr_early_stop_sweep.json used for early-stopping tables.",
	)
	ap.add_argument("--out-dir", default=TABLES_DIR, help="Destination directory (default: reproduction/tables).")
	ap.add_argument("--layer-detailed-seed", type=int, default=17)
	ns = ap.parse_args(list(argv))

	runs_root = os.path.abspath(str(ns.runs_root))
	early_root = os.path.abspath(str(ns.early_stop_root))
	out_dir = os.path.abspath(str(ns.out_dir))
	if not os.path.isdir(runs_root):
		raise FileNotFoundError(runs_root)
	if not os.path.isdir(early_root):
		raise FileNotFoundError(early_root)

	tmp = os.path.join(out_dir, "_tmp")
	_ensure_empty_dir(tmp)

	build_analysis_tex_tables(tables_dir=tmp, runs_root=runs_root)

	write_layer_detailed_table_tex(tmp, runs_root=runs_root, seed=int(ns.layer_detailed_seed), relpath_pins={})
	write_layer_selection_summary_tex(roots=(runs_root,), update_tex=os.path.join(tmp, "table_layer_selection_summary.tex"))

	write_both_early_stopping_tables(tmp, (early_root,))

	case_table_prefixes: Sequence[str] = (
		"table_layer_candidates_distilbert_trec6.tex",
		"table_layer_candidates_efficientnet_bloodmnist.tex",
		"table_layer_candidates_efficientnet_imagenette.tex",
	)
	for fn in case_table_prefixes:
		p = os.path.join(tmp, fn)
		if os.path.isfile(p):
			os.remove(p)

	os.makedirs(out_dir, exist_ok=True)
	for fn in sorted(os.listdir(tmp)):
		if not fn.endswith(".tex"):
			continue
		shutil.copy2(os.path.join(tmp, fn), os.path.join(out_dir, _strip_table_prefix(fn)))
	shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
	main()

