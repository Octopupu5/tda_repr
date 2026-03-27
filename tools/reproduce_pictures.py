from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence

from tools.figures.reproduction_figures import render_all_figure_artifacts, resolve_figure_bundle_paths, run_command, render_embedding_comparison
from tools.aggregate.reproduction_paths import (
	DEFAULT_FIGURES_RUNS_ROOT,
	DIPLOMA_PICTURES_DIR,
	LATEX_PICTURES_DIR,
	TABLES_DIR,
)
from tools.aggregate.layer_tables_from_embeddings import write_layer_candidate_table_tex


def _strip_table_prefix(filename: str) -> str:
	fn = str(filename)
	return fn[len("table_") :] if fn.startswith("table_") else fn


def _load_meta_model_lc(run_dir: str) -> str:
	mp = os.path.join(os.path.abspath(run_dir), "meta.json")
	if not os.path.isfile(mp):
		raise FileNotFoundError(mp)
	with open(mp, "r", encoding="utf-8") as f:
		meta = json.load(f)
	model = str(meta.get("model", "") or "").strip().lower()
	if not model:
		model = str((meta.get("args") or {}).get("model", "") or "").strip().lower()
	if not model:
		raise ValueError(f"Could not infer model from meta.json: {mp}")
	return model


def _write_case_candidate_tables(*, figures_root: str) -> None:
	root = os.path.abspath(str(figures_root))
	os.makedirs(os.path.abspath(TABLES_DIR), exist_ok=True)
	tmp = os.path.join(os.path.abspath(TABLES_DIR), "_tmp_case_candidates")
	os.makedirs(tmp, exist_ok=True)
	for fn in os.listdir(tmp):
		fp = os.path.join(tmp, fn)
		if os.path.isfile(fp):
			os.remove(fp)

	cases = {
		"efficientnet_b0_bloodmnist": "exp_20260505_033618_cv_bloodmnist_efficientnet_b0_ft-full",
		"efficientnet_b0_imagenette": "exp_20260404_182533_cv_imagenette_efficientnet_b0_ft-full",
		"distilbert_trec6": "exp_20260408_114511_nlp_trec6_distilbert_ft-full",
	}
	for spec_key, rel_run in cases.items():
		run_dir = os.path.join(root, rel_run)
		bundle_path = os.path.join(run_dir, "analysis", "embedding_retrieval_model_best_main.json")
		if not os.path.isfile(bundle_path):
			raise FileNotFoundError(bundle_path)
		with open(bundle_path, "r", encoding="utf-8") as f:
			bundle = json.load(f)
		model_lc = _load_meta_model_lc(run_dir)
		write_layer_candidate_table_tex(tmp, spec_key, bundle, model_lc=model_lc)

	for fn in os.listdir(tmp):
		if not fn.endswith(".tex"):
			continue
		src = os.path.join(tmp, fn)
		dst = os.path.join(os.path.abspath(TABLES_DIR), _strip_table_prefix(fn))
		shutil.copy2(src, dst)
	shutil.rmtree(tmp, ignore_errors=True)

def _ensure_empty_dir(path: str) -> None:
	p = os.path.abspath(path)
	os.makedirs(p, exist_ok=True)
	for name in os.listdir(p):
		fp = os.path.join(p, name)
		if os.path.isfile(fp) or os.path.islink(fp):
			os.remove(fp)
		elif os.path.isdir(fp):
			shutil.rmtree(fp)


def _copy_selected(src_dir: str, dst_dir: str, names: Sequence[str]) -> None:
	sd = os.path.abspath(src_dir)
	dd = os.path.abspath(dst_dir)
	os.makedirs(dd, exist_ok=True)
	for fn in names:
		sp = os.path.join(sd, str(fn))
		if not os.path.isfile(sp):
			raise FileNotFoundError(sp)
		shutil.copy2(sp, os.path.join(dd, str(fn)))


def _copy_neighbor_figures_from_saved_runs(*, figures_root: str, out_dir: str) -> None:
	root = os.path.abspath(str(figures_root))
	out = os.path.abspath(str(out_dir))
	os.makedirs(out, exist_ok=True)

	specs = {
		"fig_efficientnet_b0_bloodmnist_neighbors.png": (
			"exp_20260505_033618_cv_bloodmnist_efficientnet_b0_ft-full",
			"embedding_retrieval_model_best_main__layer_features.8.1.png",
		),
		"fig_efficientnet_b0_imagenette_neighbors.png": (
			"exp_20260404_182533_cv_imagenette_efficientnet_b0_ft-full",
			"embedding_retrieval_model_best_main__layer_features.7.0.png",
		),
	}
	for out_name, (run_rel, src_name) in specs.items():
		src = os.path.join(root, run_rel, "analysis", src_name)
		if not os.path.isfile(src):
			raise FileNotFoundError(src)
		shutil.copy2(src, os.path.join(out, out_name))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--figures-root",
		default=None,
		help="Directory containing exp_* run folders (defaults to saved_runs/figures_runs).",
	)
	ap.add_argument("--device", default="cpu")
	ap.add_argument("--download", action="store_true")
	args = ap.parse_args()

	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

	runs_root = args.figures_root
	if runs_root is None:
		runs_root = DEFAULT_FIGURES_RUNS_ROOT
	else:
		runs_root = runs_root if os.path.isabs(runs_root) else os.path.abspath(os.path.join(repo_root, runs_root))

	diploma_keep_from_render = (
		"fig_early_stopping_trec6.png",
		"fig_early_stopping_triplet.png",
		"fig_layer_dynamics.png",
		"fig_layerwise_dynamics_conv_blood.png",
		"fig_layerwise_dynamics_distilbert_trec6.png",
		"fig_trec6_distilbert_neighbors.png",
	)

	_ensure_empty_dir(DIPLOMA_PICTURES_DIR)
	render_all_figure_artifacts(
		out_dir=DIPLOMA_PICTURES_DIR,
		lang="ru",
		figures_run_root=runs_root,
		device=str(args.device),
		download_neighbor=bool(args.download),
		download_embedding=bool(args.download),
	)
	keep = set(diploma_keep_from_render)
	for fn in os.listdir(os.path.abspath(DIPLOMA_PICTURES_DIR)):
		fp = os.path.join(os.path.abspath(DIPLOMA_PICTURES_DIR), fn)
		if os.path.isfile(fp) and fn not in keep:
			os.remove(fp)

	_copy_neighbor_figures_from_saved_runs(figures_root=runs_root, out_dir=DIPLOMA_PICTURES_DIR)

	_ensure_empty_dir(LATEX_PICTURES_DIR)
	resolved = resolve_figure_bundle_paths(runs_root)
	py = sys.executable

	run_command(
		[
			py,
			"-m",
			"tools.figures.fig_mtopdiv_best_layer_dynamics",
			"--run_a",
			resolved.run_resnet_imagenette,
			"--run_b",
			resolved.run_smollm,
			"--out_png",
			os.path.join(os.path.abspath(LATEX_PICTURES_DIR), "Figure_1_dynamics.png"),
			"--style",
			"paper",
			"--lang",
			"en",
		]
	)

	render_embedding_comparison(
		py_exe=py,
		out_png=os.path.join(os.path.abspath(LATEX_PICTURES_DIR), "Figure_2_layer_example.png"),
		run_efficientnet_imagenette=resolved.run_eff_imagenette,
		layers_csv=resolved.eff_embedding_layers,
		device=str(args.device),
		download=bool(args.download),
	)

	run_command(
		[
			py,
			"-m",
			"tools.figures.fig_early_stopping_case",
			"--run_dir",
			resolved.run_distilbert_trec6_early_stopping,
			"--layer",
			"distilbert.transformer.layer.3.ffn",
			"--metric",
			"beta1_L_est",
			"--mode",
			"min",
			"--patience",
			"7",
			"--title",
			"__none__",
			"--out_png",
			os.path.join(os.path.abspath(LATEX_PICTURES_DIR), "Figure_3_early_stopping.png"),
			"--lang",
			"en",
		]
	)

	_write_case_candidate_tables(figures_root=runs_root)


if __name__ == "__main__":
	main()
