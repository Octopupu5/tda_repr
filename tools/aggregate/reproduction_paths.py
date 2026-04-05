from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPRODUCTION_ROOT = os.path.join(REPO_ROOT, "reproduction")
TABLES_DIR = os.path.join(REPRODUCTION_ROOT, "tables")
ANALYSIS_TABLES_DIR = TABLES_DIR
DIPLOMA_PICTURES_DIR = os.path.join(REPRODUCTION_ROOT, "diploma_pictures")
LATEX_PICTURES_DIR = os.path.join(REPRODUCTION_ROOT, "latex_pictures")
PAPER_STATIC_ANALYSIS_TABLES_DIR = os.path.join(REPO_ROOT, "paper", "analysis_tables")

DEFAULT_FIGURES_RUNS_ROOT = os.path.join(REPO_ROOT, "saved_runs", "figures_runs")
ZENODO_CASE_STUDY_DOI = "10.5281/zenodo.20114914"


@dataclass(frozen=True)
class FigureRunPaths:
	resnet_imagenette: str = "exp_20260419_160324_cv_imagenette_resnet18_ft-full"
	smollm_summarize: str = "exp_20260418_211916_nlp_smol-summarize_smollm2-135m_ft-full"
	blood_cv_layerwise_run: str = "exp_20260406_013322_cv_bloodmnist_convnext_tiny_ft-full"
	blood_cv_layerwise_layers: str = "features.1.0,features.3.1,features.5.4"
	distilbert_trec6: str = "exp_20260408_114511_nlp_trec6_distilbert_ft-full"
	distilbert_trec6_early_stopping: str = "exp_20260403_042415_nlp_trec6_distilbert_ft-full"
	trec6_distilbert_layerwise_layers: str = "distilbert.transformer.layer.0.ffn,distilbert.transformer.layer.2.ffn,distilbert.transformer.layer.5.ffn"
	distilbert_trec6_neighbor_layer: str = "distilbert.transformer.layer.5.ffn"
	trec6_neighbor_anchor_idx: int = 364
	efficientnet_imagenette: str = "exp_20260404_182533_cv_imagenette_efficientnet_b0_ft-full"
	efficientnet_imagenette_embedding_layers: str = "classifier,features.7.0"


TABLE_RUN_DIRS_BY_SPEC_KEY_OVERRIDE: Mapping[str, list[str]] | None = None
FIGURE_RUN_PATHS_OVERRIDE: FigureRunPaths | None = None


def project_root() -> str:
	return REPO_ROOT


def resolve_run_dir(figures_run_root: str, entry: str) -> str:
	if os.path.isabs(entry):
		return entry
	return os.path.join(os.path.abspath(figures_run_root), entry)


def effective_figure_paths() -> FigureRunPaths:
	return FIGURE_RUN_PATHS_OVERRIDE if FIGURE_RUN_PATHS_OVERRIDE is not None else FigureRunPaths()


LAYER_EMBEDDING_CASE_PATHS_REL_OVERRIDE: Mapping[str, str] | None = None


def default_layer_embedding_case_path_by_spec_key() -> Dict[str, str]:
	return {
		"efficientnet_b0_bloodmnist": "cv/bloodmnist/exp_20260505_033618_cv_bloodmnist_efficientnet_b0_ft-full",
		"efficientnet_b0_imagenette": "cv/imagenette/exp_20260504_182533_cv_imagenette_efficientnet_b0_ft-full",
		"distilbert_trec6": "nlp/trec6/exp_20260503_042415_nlp_trec6_distilbert_ft-full",
	}


def layer_embedding_cases_relpath_for_spec_key(
	spec_key: str,
	*,
	overrides: Mapping[str, str] | None = None,
) -> str | None:
	if overrides is not None and spec_key in overrides:
		return str(overrides[spec_key]).strip()
	if LAYER_EMBEDDING_CASE_PATHS_REL_OVERRIDE is not None and spec_key in LAYER_EMBEDDING_CASE_PATHS_REL_OVERRIDE:
		return str(LAYER_EMBEDDING_CASE_PATHS_REL_OVERRIDE[spec_key]).strip()
	return default_layer_embedding_case_path_by_spec_key().get(str(spec_key))
