from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.aggregate.reproduction_paths import (
	PAPER_STATIC_ANALYSIS_TABLES_DIR,
	TABLE_RUN_DIRS_BY_SPEC_KEY_OVERRIDE,
	ZENODO_CASE_STUDY_DOI,
	resolve_run_dir,
)
from tools.aggregate.run_meta import RunMeta, list_run_dirs, load_run_meta

COMPUTED_ANALYSIS_TABLE_TEX = (
	"table_arch_comparison.tex",
	"table_correlation_summary_cv.tex",
	"table_correlation_summary_nlp.tex",
	"table_depth_dynamics_cv.tex",
	"table_depth_dynamics_nlp.tex",
)
STATIC_ANALYSIS_TABLE_TEX = (
	"table_early_stopping.tex",
	"table_early_stopping_best.tex",
	"table_layer_candidates_distilbert_trec6.tex",
	"table_layer_candidates_efficientnet_bloodmnist.tex",
	"table_layer_candidates_efficientnet_imagenette.tex",
	"table_layer_selection_detailed.tex",
	"table_layer_selection_summary.tex",
)
ALL_ANALYSIS_TABLE_TEX = COMPUTED_ANALYSIS_TABLE_TEX + STATIC_ANALYSIS_TABLE_TEX
EXPECTED_REPRODUCTION_TEX = ALL_ANALYSIS_TABLE_TEX


def read_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def should_exclude_run_from_paper_tables(run_dir: str) -> bool:
	try:
		meta = read_json(os.path.join(str(run_dir), "meta.json"))
	except OSError:
		return True
	m = str(meta.get("model", (meta.get("args") or {}).get("model", "")) or "").strip().lower()
	return m == "mlp"


@dataclass(frozen=True)
class PaperSpec:
	key: str
	arch_tex: str
	dataset_tex: str
	cite_arch: str
	cite_ds: str
	task: str
	dataset: str
	model: str
	finetune: str
	nlp_objective: str
	bench_metric: str
	minimize_bench: bool
	task_label_tex: str
	hook_layer: str

	def matches(self, meta: RunMeta) -> bool:
		if (
			meta.task.lower(),
			meta.dataset.lower(),
			meta.model.lower(),
			meta.finetune.lower(),
		) != (
			self.task.lower(),
			self.dataset.lower(),
			self.model.lower(),
			self.finetune.lower(),
		):
			return False
		if self.task == "nlp":
			return str(meta.args.get("nlp_objective", "classification")).lower() == self.nlp_objective.lower()
		return True


PAPER_SPECS: Tuple[PaperSpec, ...] = (
	PaperSpec("mlp_mnist", "MLP", "MNIST", r"\cite{b13}", r"\cite{b6}", "cv", "mnist", "mlp", "full", "", "f1_macro", False, "(F1)", "5"),
	PaperSpec("resnet18_cifar10", "ResNet18", "CIFAR-10", r"\cite{b14}", r"\cite{b7}", "cv", "cifar10", "resnet18", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("resnet18_bloodmnist", "ResNet18", "BloodMNIST", r"\cite{b14}", r"\cite{b8}", "cv", "bloodmnist", "resnet18", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("resnet18_imagenette", "ResNet18", "ImageNette", r"\cite{b14}", r"\cite{b33}", "cv", "imagenette", "resnet18", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("efficientnet_b0_cifar10", "EfficientNet-B0", "CIFAR-10", r"\cite{b15}", r"\cite{b7}", "cv", "cifar10", "efficientnet_b0", "full", "", "f1_macro", False, "(F1)", "classifier.0"),
	PaperSpec("efficientnet_b0_bloodmnist", "EfficientNet-B0", "BloodMNIST", r"\cite{b15}", r"\cite{b8}", "cv", "bloodmnist", "efficientnet_b0", "full", "", "f1_macro", False, "(F1)", "classifier.0"),
	PaperSpec("efficientnet_b0_imagenette", "EfficientNet-B0", "ImageNette", r"\cite{b15}", r"\cite{b33}", "cv", "imagenette", "efficientnet_b0", "full", "", "f1_macro", False, "(F1)", "features.7.0"),
	PaperSpec("convnext_tiny_cifar10", "ConvNeXt-Tiny", "CIFAR-10", r"\cite{b16}", r"\cite{b7}", "cv", "cifar10", "convnext_tiny", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("convnext_tiny_bloodmnist", "ConvNeXt-Tiny", "BloodMNIST", r"\cite{b16}", r"\cite{b8}", "cv", "bloodmnist", "convnext_tiny", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("convnext_tiny_imagenette", "ConvNeXt-Tiny", "ImageNette", r"\cite{b16}", r"\cite{b33}", "cv", "imagenette", "convnext_tiny", "full", "", "f1_macro", False, "(F1)", "avgpool"),
	PaperSpec("distilbert_sst2", "DistilBERT", "SST-2", r"\cite{b17}", r"\cite{b10}", "nlp", "sst2", "distilbert", "full", "classification", "f1_macro", False, "(F1)", "distilbert.transformer.layer.5.ffn"),
	PaperSpec("distilbert_trec6", "DistilBERT", "Trec-6", r"\cite{b17}", r"\cite{b11}", "nlp", "trec6", "distilbert", "full", "classification", "f1_macro", False, "(F1)", "distilbert.transformer.layer.5.ffn"),
	PaperSpec("smollm_smol_summarize", "SmolLM", "smol-summarize", r"\cite{b18}", r"\cite{b12}", "nlp", "smol-summarize", "smollm2-135m", "full", "generation", "ppl", True, "(PPL)", "model.layers.29"),
)


def spec_by_key(key: str) -> PaperSpec:
	return next(s for s in PAPER_SPECS if s.key == key)


def find_matching_run_dirs(runs_root: str, spec: PaperSpec) -> List[str]:
	candidates: List[Tuple[bool, float, int, str]] = []
	root_abs = os.path.abspath(runs_root)
	for rd in _experiment_run_dirs_flat_and_nested(root_abs):
		m = load_run_meta(rd)
		if not spec.matches(m):
			continue
		meta_path = os.path.join(rd, "meta.json")
		try:
			meta = read_json(meta_path)
		except OSError:
			sys.stderr.write(f"[reproduction_tables] Skipping unreadable meta: {meta_path}\n")
			continue
		args, mon = meta.get("args") or {}, meta.get("monitor") or {}
		q1 = bool(args.get("compute_q1_spectra") or mon.get("compute_q1_spectra"))
		mt = os.path.getmtime(meta_path)
		try:
			sd = int(args.get("seed", 0))
		except Exception:
			sd = 0
		candidates.append((q1, mt, sd, rd))
	if not candidates:
		return []
	candidates.sort(key=lambda t: (-int(t[0]), -float(t[1])))
	best_by_seed: Dict[int, str] = {}
	for _q1, _mt, sd, rd in candidates:
		if sd not in best_by_seed:
			best_by_seed[sd] = rd
	return [best_by_seed[s] for s in sorted(best_by_seed.keys())]


def copy_static_analysis_tables(dest_dir: str) -> None:
	src = PAPER_STATIC_ANALYSIS_TABLES_DIR
	if not os.path.isdir(src):
		sys.stderr.write(
			"reproduction_tables: paper/analysis_tables missing; add .tex or fetch https://doi.org/"
			+ ZENODO_CASE_STUDY_DOI
			+ "\n"
		)
		return
	dd = os.path.abspath(dest_dir)
	os.makedirs(dd, exist_ok=True)
	for name in STATIC_ANALYSIS_TABLE_TEX:
		sp = os.path.join(src, name)
		dp = os.path.join(dd, name)
		if not os.path.isfile(sp):
			sys.stderr.write("reproduction_tables: missing source " + sp + "\n")
			continue
		try:
			if os.path.exists(dp) and os.path.samefile(sp, dp):
				continue
		except OSError:
			pass
		shutil.copy2(sp, dp)


def strip_cites_in_tex_dir(tables_dir: str) -> None:
	cite_re = re.compile(r"(?:\s|~)*\\cite[a-zA-Z]*\{[^}]*\}")
	td = os.path.abspath(str(tables_dir))
	if not os.path.isdir(td):
		raise FileNotFoundError(td)
	for fn in sorted(os.listdir(td)):
		if not str(fn).endswith(".tex"):
			continue
		fp = os.path.join(td, str(fn))
		if not os.path.isfile(fp):
			continue
		with open(fp, "r", encoding="utf-8") as rf:
			txt = rf.read()
		new_txt = cite_re.sub("", txt)
		if new_txt != txt:
			with open(fp, "w", encoding="utf-8") as wf:
				wf.write(new_txt)


def resolved_run_lists(
	*,
	runs_root: str,
	overrides: Mapping[str, List[str]] | None,
	keys: Sequence[str] | None,
) -> Dict[str, List[str]]:
	root_abs = os.path.abspath(runs_root)
	oridef = overrides if overrides is not None else TABLE_RUN_DIRS_BY_SPEC_KEY_OVERRIDE
	key_list = list(keys) if keys is not None else [s.key for s in PAPER_SPECS]
	out: Dict[str, List[str]] = {}
	skipped = []
	for key in key_list:
		spec = spec_by_key(key)
		if isinstance(oridef, Mapping) and key in oridef:
			out[key] = [resolve_run_dir(root_abs, p) for p in oridef[key]]
			continue
		found = find_matching_run_dirs(root_abs, spec)
		if found:
			out[key] = [os.path.abspath(p) for p in found]
		else:
			skipped.append(key)
	if skipped:
		sys.stderr.write(
			f"[reproduction_tables] No matching run dirs for keys (rows omitted): {', '.join(skipped)}\n"
			f"  -> set reproduction_paths.TABLE_RUN_DIRS_BY_SPEC_KEY_OVERRIDE or expand {root_abs}\n"
		)
	return out


def format_pm_tex(m: Any, s: Any, decimals: int, *, signed: bool) -> str:
	try:
		mv = float(m) if m is not None else float("nan")
	except (TypeError, ValueError):
		return r"\mathrm{n/a}"
	if isinstance(mv, float) and (math.isnan(mv) or not math.isfinite(mv)):
		return r"\mathrm{n/a}"
	try:
		sv = float(s) if s is not None else 0.0
	except (TypeError, ValueError):
		sv = 0.0
	if isinstance(sv, float) and (math.isnan(sv) or not math.isfinite(sv)):
		sv = 0.0
	body = f"{mv:.{decimals}f} \\pm {sv:.{decimals}f}"
	if signed and mv > 0:
		return f"+{body}"
	if signed and mv == 0:
		return f"+{body}"
	return body


def write_arch_tex(
	path: str,
	stats_by_key: Dict[str, Dict[str, Any]],
	specs: Sequence[PaperSpec],
	*,
	caption: Optional[str] = None,
) -> None:
	_ = caption
	body: List[str] = []
	for spec in specs:
		st = stats_by_key[spec.key]
		arch = f"{spec.arch_tex}"
		ds = f"{spec.dataset_tex}"
		dt_core = format_pm_tex(st["mean_task"], st["std_task"], 2, signed=True)
		lbl = str(spec.task_label_tex).strip()
		task_cell = f"${dt_core}$ {lbl}" if lbl else f"${dt_core}$"
		db = format_pm_tex(st["mean_b1kl"], st["std_b1kl"], 1, signed=True)
		dm = format_pm_tex(st["mean_mtop"], st["std_mtop"], 1, signed=True)
		d0 = format_pm_tex(st["mean_l0"], st["std_l0"], 2, signed=True)
		d1 = format_pm_tex(st["mean_l1"], st["std_l1"], 2, signed=True)
		body.append(f"{arch:<25} & {ds:<23} & {task_cell} & ${db}$  & ${dm}$  & ${d0}$ & ${d1}$ \\\\")
	arch_comparison_header = (
		r"\textbf{Architecture} & \textbf{Dataset} & $\mathbf{\Delta \text{Task Metric}}$ & "
		r"$\mathbf{\Delta \beta_1^{K,L}}$ & \textbf{$\Delta \text{MTopDiv}$} & "
		r"$\mathbf{\Delta \lambda_2(\Delta_0(L))}$ & $\mathbf{\Delta \lambda_1(\Delta_1^{K,L})}$ \\"
	)
	lines = [r"\toprule", arch_comparison_header, r"\midrule"] + body + [r"\bottomrule"]
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))
		f.write("\n")


def _experiment_run_dirs_flat_and_nested(runs_root: str) -> List[str]:
	root_abs = os.path.abspath(runs_root)
	paths_set: set[str] = set()
	try:
		for rd in list_run_dirs(root_abs):
			paths_set.add(os.path.abspath(rd))
	except FileNotFoundError:
		pass
	for tail in ("cv", "nlp"):
		sub_root = os.path.join(root_abs, tail)
		if not os.path.isdir(sub_root):
			continue
		for ds_name in sorted(os.listdir(sub_root)):
			ds_path = os.path.join(sub_root, ds_name)
			if not os.path.isdir(ds_path):
				continue
			for name in sorted(os.listdir(ds_path)):
				p = os.path.join(ds_path, name)
				if os.path.isdir(p) and os.path.isfile(os.path.join(p, "meta.json")):
					paths_set.add(os.path.abspath(p))
	return sorted(paths_set)


def all_experiment_run_dirs_with_metrics(runs_root: str) -> List[str]:
	root_abs = os.path.abspath(runs_root)
	out: List[str] = []
	for rd in _experiment_run_dirs_flat_and_nested(root_abs):
		if os.path.isfile(os.path.join(rd, "metrics.jsonl")):
			out.append(rd)
	return out


def split_run_dirs_by_task(run_dirs: Sequence[str]) -> Tuple[List[str], List[str]]:
	cv_dirs: List[str] = []
	nlp_dirs: List[str] = []
	for rd in run_dirs:
		mp = os.path.join(str(rd), "meta.json")
		try:
			meta = read_json(mp)
		except OSError:
			continue
		args = meta.get("args") or {}
		task = str(meta.get("task", args.get("task", "cv"))).lower().strip()
		ap = os.path.abspath(str(rd))
		if task == "nlp":
			nlp_dirs.append(ap)
		else:
			cv_dirs.append(ap)
	return cv_dirs, nlp_dirs


def build_analysis_tex_tables(
	*,
	tables_dir: str,
	runs_root: str,
	run_overrides: Mapping[str, List[str]] | None = None,
	spec_keys_or_none_for_all: Sequence[str] | None = None,
) -> Tuple[Dict[str, List[str]], List[PaperSpec]]:
	from tools.aggregate.paper_aggregate_tables import (
		aggregate_arch_table,
		arch_rows_to_stats_by_key,
		load_run_summaries_for_paper_tables,
		write_correlation_summary_tex,
		write_depth_dynamics_table_tex,
	)

	os.makedirs(os.path.abspath(tables_dir), exist_ok=True)
	copy_static_analysis_tables(tables_dir)

	run_lists = resolved_run_lists(
		runs_root=runs_root,
		overrides=run_overrides,
		keys=spec_keys_or_none_for_all,
	)
	if not run_lists:
		raise RuntimeError(
			f"No experiment runs matched under {runs_root}. "
			"Set overrides in reproduction_paths.TABLE_RUN_DIRS_BY_SPEC_KEY_OVERRIDE "
			"or add runs so at least one paper spec resolves."
		)

	td = os.path.abspath(tables_dir)
	for fn in os.listdir(td):
		if fn.endswith(".tex"):
			continue
		fp = os.path.join(td, fn)
		if os.path.isfile(fp):
			os.remove(fp)

	specs_in_order = list(PAPER_SPECS)
	summaries = load_run_summaries_for_paper_tables(runs_root)
	sys.stderr.write(f"[reproduction_tables] arch/corr pool: {len(summaries)} run summaries.\n")
	arch_rows = aggregate_arch_table(summaries)
	stats_by_key, specs_ordered = arch_rows_to_stats_by_key(arch_rows, specs_in_order)
	if not specs_ordered:
		raise RuntimeError(
			"No arch comparison rows matched bucketing under runs_root. "
			"Ensure meta matches paper specs and topo metrics are logged."
		)
	try:
		arch_table_caption = (
			r"Deep-layer representation dynamics: task column is $\\Delta$ for the whole run; "
			r"topo columns use per-run medians over monitored layers in the \emph{deep} backbone tertile only "
			r"(last third of backbone depth ranks; classifier/head layers excluded), "
			r"then mean $\\pm$ std across runs---the same layer subset as row ``Deep (Final Backbone/Pool)'' in depth-dynamics, "
			r"not the pooled early/mid/deep/head stack. "
			r"The \\texttt{mlp} row is retained for modality/architecture contrast; pooled correlation summaries omit \\texttt{mlp}. "
			r"$\\Delta$ from first logged epoch to best validation; $q=0$ spectral column uses the smallest \\emph{positive} Hodge Laplacian eigenvalue in the logged list."
		)
		write_arch_tex(os.path.join(td, "table_arch_comparison.tex"), stats_by_key, specs_ordered, caption=arch_table_caption)
		flat_all = _experiment_run_dirs_flat_and_nested(runs_root)
		corr_dirs = [d for d in flat_all if not should_exclude_run_from_paper_tables(d)]
		n_ex = len(flat_all) - len(corr_dirs)
		if n_ex:
			sys.stderr.write(f"[reproduction_tables] excluded {n_ex} MLP run dir(s) from correlation/depth pools.\n")
		sys.stderr.write(f"[reproduction_tables] correlation dirs: {len(corr_dirs)}.\n")
		corr_cv, corr_nlp = split_run_dirs_by_task(corr_dirs)
		sys.stderr.write(f"[reproduction_tables] correlation split: cv={len(corr_cv)}, nlp={len(corr_nlp)}.\n")
		write_correlation_summary_tex(os.path.join(td, "table_correlation_summary_cv.tex"), corr_cv, modality="cv")
		write_correlation_summary_tex(os.path.join(td, "table_correlation_summary_nlp.tex"), corr_nlp, modality="nlp")

		depth_all = all_experiment_run_dirs_with_metrics(runs_root)
		depth_dirs = [d for d in depth_all if not should_exclude_run_from_paper_tables(d)]
		sys.stderr.write(f"[reproduction_tables] depth pool: runs with metrics, no MLP ({len(depth_dirs)} dirs).\n")
		depth_cv, depth_nlp = split_run_dirs_by_task(depth_dirs)
		sys.stderr.write(f"[reproduction_tables] depth split: cv={len(depth_cv)}, nlp={len(depth_nlp)}.\n")
		write_depth_dynamics_table_tex(os.path.join(td, "table_depth_dynamics_cv.tex"), depth_cv, modality="cv")
		write_depth_dynamics_table_tex(os.path.join(td, "table_depth_dynamics_nlp.tex"), depth_nlp, modality="nlp")
	except OSError as exc:
		sys.stderr.write(f"[reproduction_tables] write failed: {exc}\n")
		raise

	strip_cites_in_tex_dir(td)
	return run_lists, specs_ordered


EXPECTED_REPRODUCTION_IMAGES_EN = (
	"fig_early_stopping_trec6.png",
	"fig_early_stopping_triplet.png",
	"fig_efficientnet_b0_bloodmnist_neighbors.png",
	"fig_efficientnet_b0_imagenette_neighbors.png",
	"fig_layer_dynamics.png",
	"fig_layerwise_dynamics_conv_blood.png",
	"fig_layerwise_dynamics_distilbert_trec6.png",
	"fig_trec6_distilbert_neighbors.png",
)
