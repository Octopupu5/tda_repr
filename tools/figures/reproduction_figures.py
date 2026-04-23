from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.aggregate.reproduction_paths import FigureRunPaths, resolve_run_dir, effective_figure_paths


FIGURE_EMBEDDINGS_OUT_NAME = "fig_efficientnet_imagenette_embedding_compare.png"


def _safe_name(s: str) -> str:
	out = []
	for ch in str(s):
		if ch.isalnum() or ch in ("_", ".", "-"):
			out.append(ch)
		else:
			out.append("_")
	return "".join(out)


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
	proc = subprocess.run(list(argv), cwd=REPO_ROOT, capture_output=True, text=True)
	if proc.returncode != 0:
		msg = (proc.stdout or "").strip() + "\n" + (proc.stderr or "").strip()
		sys.stderr.write(f"[reproduction_figures] Command failed ({proc.returncode}): {' '.join(argv)}\n")
		if msg.strip():
			sys.stderr.write(msg + "\n")
		raise RuntimeError(f"Figure subprocess failed ({proc.returncode}): {' '.join(argv)}")
	return proc


@dataclass(frozen=True)
class ResolvedFigureRuns:
	run_resnet_imagenette: str
	run_smollm: str
	run_blood_cv: str
	run_distilbert_trec6: str
	run_distilbert_trec6_early_stopping: str
	run_eff_imagenette: str
	layers_blood_cv: str
	layers_distilbert_trec6: str
	neighbor_layer: str
	trec6_neighbor_anchor_idx: int
	eff_embedding_layers: str


def resolve_figure_bundle_paths(figures_run_root: str, paths: FigureRunPaths | None = None) -> ResolvedFigureRuns:
	rp = paths if paths is not None else effective_figure_paths()
	root = os.path.abspath(figures_run_root)
	return ResolvedFigureRuns(
		run_resnet_imagenette=resolve_run_dir(root, rp.resnet_imagenette),
		run_smollm=resolve_run_dir(root, rp.smollm_summarize),
		run_blood_cv=resolve_run_dir(root, rp.blood_cv_layerwise_run),
		run_distilbert_trec6=resolve_run_dir(root, rp.distilbert_trec6),
		run_distilbert_trec6_early_stopping=resolve_run_dir(root, rp.distilbert_trec6_early_stopping),
		run_eff_imagenette=resolve_run_dir(root, rp.efficientnet_imagenette),
		layers_blood_cv=str(rp.blood_cv_layerwise_layers),
		layers_distilbert_trec6=str(rp.trec6_distilbert_layerwise_layers),
		neighbor_layer=str(rp.distilbert_trec6_neighbor_layer),
		trec6_neighbor_anchor_idx=int(rp.trec6_neighbor_anchor_idx),
		eff_embedding_layers=str(rp.efficientnet_imagenette_embedding_layers),
	)


def validate_run_dir(path: str, label: str) -> None:
	if not os.path.isdir(path):
		raise FileNotFoundError(f"Figure run missing ({label}): {path}")
	mi = os.path.join(path, "meta.json")
	if not os.path.isfile(mi):
		raise FileNotFoundError(f"{label} meta.json missing: {mi}")


def iterate_figure_commands(
	*,
	py_exe: str,
	out_dir: str,
	lang: str,
	run: ResolvedFigureRuns,
	device: str,
	download_neighbor: bool,
	name_out: Callable[[str], str] | None = None,
) -> Iterator[Tuple[str, List[str]]]:
	def out(path_basename: str) -> str:
		if name_out:
			path_basename = name_out(path_basename)
		os.makedirs(os.path.abspath(out_dir), exist_ok=True)
		return os.path.join(os.path.abspath(out_dir), path_basename)

	yield (
		"fig_layer_dynamics",
		[
			py_exe,
			"-m",
			"tools.figures.fig_mtopdiv_best_layer_dynamics",
			"--run_a",
			run.run_resnet_imagenette,
			"--run_b",
			run.run_smollm,
			"--out_png",
			out("fig_layer_dynamics.png"),
			"--style",
			"paper",
			"--lang",
			lang,
		],
	)
	yield (
		"fig_layerwise_conv_blood",
		[
			py_exe,
			"-m",
			"tools.figures.fig_layerwise_descriptor_dynamics",
			"--run_dir",
			run.run_blood_cv,
			"--layers",
			run.layers_blood_cv,
			"--out_png",
			out("fig_layerwise_dynamics_conv_blood.png"),
			"--lang",
			lang,
		],
	)
	yield (
		"fig_layerwise_distilbert",
		[
			py_exe,
			"-m",
			"tools.figures.fig_layerwise_descriptor_dynamics",
			"--run_dir",
			run.run_distilbert_trec6,
			"--layers",
			run.layers_distilbert_trec6,
			"--out_png",
			out("fig_layerwise_dynamics_distilbert_trec6.png"),
			"--lang",
			lang,
		],
	)
	yield (
		"fig_early_stopping_trec6",
		[
			py_exe,
			"-m",
			"tools.figures.fig_early_stopping_case",
			"--run_dir",
			run.run_distilbert_trec6_early_stopping,
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
			out("fig_early_stopping_trec6.png"),
			"--lang",
			lang,
		],
	)
	yield (
		"fig_early_stopping_triplet",
		[
			py_exe,
			"-m",
			"tools.figures.fig_early_stopping_triplet_case",
			"--run_dir",
			run.run_smollm,
			"--metrics",
			"beta1_L_est,beta1_persistent_est,hodge_L_q0_lambda2",
			"--modes",
			"max,min,max",
			"--patience",
			"6",
			"--aggregate",
			"all",
			"--bench_metric",
			"ppl",
			"--out_png",
			out("fig_early_stopping_triplet.png"),
			"--lang",
			lang,
		],
	)

	_neighbor = [
		py_exe,
		"-m",
		"tools.figures.fig_trec6_distilbert_neighbors",
		"--run_dir",
		run.run_distilbert_trec6,
		"--split",
		"test",
		"--layer",
		run.neighbor_layer,
		"--anchor_idx",
		str(run.trec6_neighbor_anchor_idx),
		"--top_k",
		"20",
		"--device",
		device,
		"--out_png",
		out("fig_trec6_distilbert_neighbors.png"),
		"--lang",
		lang,
	]
	if download_neighbor:
		_neighbor.append("--download")
	yield ("fig_trec6_distilbert_neighbors", _neighbor)


def render_embedding_comparison(
	*,
	py_exe: str,
	out_png: str,
	run_efficientnet_imagenette: str,
	layers_csv: str,
	device: str,
	download: bool,
) -> None:
	argv = [
		py_exe,
		"-m",
		"tools.evaluate_embeddings",
		"--run_dir",
		run_efficientnet_imagenette,
		"--checkpoint",
		"best_main",
		"--split",
		"val",
		"--layers",
		layers_csv,
		"--compare_two_layers_top_k",
		"12",
		"--compare_out_png",
		out_png,
		"--search_best_illustration_anchor",
		"--search_min_delta_neighbors",
		"2",
		"--anchors_per_class",
		"100",
		"--seed",
		"0",
		"--device",
		device,
		"--no-skip_existing",
		"--no-write_bundle",
	]
	if download:
		argv.append("--download")
	run_command(argv)
	if not os.path.isfile(os.path.abspath(out_png)):
		raise RuntimeError(f"Embedding compare PNG was not produced: {out_png}")


def render_all_figure_artifacts(
	*,
	out_dir: str,
	lang: str,
	figures_run_root: str,
	device: str = "cpu",
	download_neighbor: bool = False,
	download_embedding: bool = False,
	path_template: FigureRunPaths | None = None,
	on_step: Callable[[str], None] | None = None,
) -> ResolvedFigureRuns:
	run = resolve_figure_bundle_paths(figures_run_root, path_template or effective_figure_paths())
	for lbl in (
		("resnet_imagenette", run.run_resnet_imagenette),
		("smollm", run.run_smollm),
		("blood_cv", run.run_blood_cv),
		("distilbert_trec6", run.run_distilbert_trec6),
		("distilbert_trec6_early_stopping", run.run_distilbert_trec6_early_stopping),
		("eff_imagenette", run.run_eff_imagenette),
	):
		validate_run_dir(lbl[1], lbl[0])

	py = sys.executable
	for name, argv in iterate_figure_commands(
		py_exe=py,
		out_dir=out_dir,
		lang=str(lang),
		run=run,
		device=device,
		download_neighbor=download_neighbor,
	):
		if on_step:
			on_step(name)
		run_command(argv)

	out_emb = os.path.join(os.path.abspath(out_dir), FIGURE_EMBEDDINGS_OUT_NAME)
	if on_step:
		on_step("embedding_compare")
	render_embedding_comparison(
		py_exe=py,
		out_png=out_emb,
		run_efficientnet_imagenette=run.run_eff_imagenette,
		layers_csv=run.eff_embedding_layers,
		device=device,
		download=download_embedding,
	)
	return run
