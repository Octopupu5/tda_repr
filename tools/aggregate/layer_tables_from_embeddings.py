from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.aggregate.aggregate_layer_selection_summary import _proposed_candidate_layers, _r_star, _ratio_by_layer
from tools.aggregate.embedding_selection import (
	_argmax_abs_rho,
	_best_strict_edge_for_layer,
	descriptor_tex_from_repr_key,
	load_corr_edges,
)
from tools.aggregate.reproduction_paths import layer_embedding_cases_relpath_for_spec_key
from tools.aggregate.reproduction_tables import PAPER_SPECS, PaperSpec, _experiment_run_dirs_flat_and_nested
from tools.aggregate.run_meta import load_run_meta


DEFAULT_EMBEDDING_BUNDLE_REL = os.path.join("analysis", "embedding_retrieval_model_best_main.json")


def _bundle_path(run_dir: str) -> str:
	return os.path.join(str(run_dir), DEFAULT_EMBEDDING_BUNDLE_REL)


def _read_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as rf:
		return json.load(rf)


def _tex_tt(layer: Optional[str]) -> str:
	if not layer:
		return ""
	return str(layer).replace("_", r"\_")


def _escape_tex_cell(s: str) -> str:
	t = str(s).replace("&", r"\&")
	return t.replace("%", r"\%")


def _paper_layer_tt(model_lc: str, layer: Optional[str]) -> str:
	if not layer:
		return "---"
	ml = str(model_lc).strip().lower()
	ls = str(layer)
	if ml == "distilbert" and ls.startswith("distilbert.transformer.layer."):
		rest = ls[len("distilbert.transformer.layer.") :]
		idx = rest.split(".")[0]
		if idx.isdigit():
			return rf"\texttt{{layer.{idx}}}"
	return rf"\texttt{{{_tex_tt(ls)}}}"


def _runs_with_bundle(spec: PaperSpec, runs_root: str) -> List[str]:
	root_abs = os.path.abspath(str(runs_root))
	out: List[str] = []
	for rd in sorted(_experiment_run_dirs_flat_and_nested(root_abs)):
		mp = os.path.join(rd, "meta.json")
		if not os.path.isfile(mp):
			continue
		try:
			meta_rm = load_run_meta(rd)
		except Exception:
			continue
		if not spec.matches(meta_rm):
			continue
		if os.path.isfile(_bundle_path(rd)):
			out.append(os.path.abspath(rd))
	return out


def exhaustive_best_tt(spec: PaperSpec, r_by_layer: Mapping[str, float], r_star_raw: Optional[float]) -> str:
	rb = dict(r_by_layer)
	if not rb or r_star_raw is None:
		return "---"
	rs = round(float(r_star_raw), 3)
	keys = sorted(k for k, v in rb.items() if round(float(v), 3) == rs)
	if not keys:
		return "---"
	ml = spec.model.strip().lower()
	return " / ".join(_paper_layer_tt(ml, k) if ml == "distilbert" else rf"\texttt{{{_tex_tt(k)}}}" for k in keys)


def proposed_best_tt(spec: PaperSpec, run_dir: str, r_by_layer: Mapping[str, float]) -> Tuple[str, Optional[float]]:
	"""
	Proposed layer: pick the best-R layer among correlation-ranked candidates across
	(topological + spectral + MTopDiv) families.
	"""
	cands = _proposed_candidate_layers(
		str(run_dir),
		top_n_layers=8,
		min_abs_rho=0.0,
		max_p=1.0,
	)
	rbest = _r_star(r_by_layer)
	if rbest is None:
		return "---", None
	rb = dict(r_by_layer)
	hits = [float(rb[str(l)]) for l in cands if str(l) in rb]
	if not hits:
		dl = getattr(spec, "hook_layer", None)
		v = rb.get(str(dl)) if dl and str(dl) in rb else None
		label = dl if dl and v is not None else None
		return (_paper_layer_tt(spec.model, str(label)) if label else "---"), (float(v) if v is not None else None)
	r_prop = max(hits)
	pick_layers = [str(l) for l in cands if str(l) in rb and round(float(rb[str(l)]), 3) == round(float(r_prop), 3)]
	if not pick_layers:
		return "---", None
	ml = spec.model.strip().lower()
	fst = (
		_paper_layer_tt(ml, pick_layers[0])
		if ml == "distilbert"
		else rf"\texttt{{{_tex_tt(pick_layers[0])}}}"
	)
	return fst, float(r_prop)


def detailed_line(spec: PaperSpec, run_dir: str, bundle: Mapping[str, Any]) -> Optional[str]:
	rb = _ratio_by_layer(dict(bundle), _bundle_path(run_dir))
	if not rb:
		return None
	r_star_raw = _r_star(rb)
	if r_star_raw is None:
		return None
	exh = exhaustive_best_tt(spec, rb, r_star_raw)
	prop_tt, r_prop_v = proposed_best_tt(spec, run_dir, rb)
	gap_v = (
		None
		if r_prop_v is None
		else round(float(max(0.0, float(r_star_raw) - float(r_prop_v))), 3)
	)
	arch_cell = rf"{spec.arch_tex}"
	ds_cell = rf"{spec.dataset_tex}"
	gap_tex = "---" if gap_v is None else f"{gap_v:.3f}"
	r_star_tex = f"{round(float(r_star_raw), 3):.3f}"
	r_prop_tex = "---" if r_prop_v is None else f"{round(float(r_prop_v), 3):.3f}"
	return (
		f"{arch_cell} & {ds_cell} & {exh} & {r_star_tex} "
		f"& {prop_tt} & {r_prop_tex} & {gap_tex} \\\\"
	)


def _layer_tables_config() -> Tuple[Tuple[str, ...], Sequence[Sequence[str]], Mapping[str, str]]:
	selection_method_keys: Tuple[str, ...] = (
		"best_r",
		"default_layer",
		"topo_strict_best_r",
		"topo_max_abs_rho",
		"spectral_strict_best_r",
		"spectral_max_abs_rho",
		"mtopdiv_strict_best_r",
		"mtopdiv_max_abs_rho",
	)
	spec_groups_detail: Sequence[Sequence[str]] = (
		("mlp_mnist",),
		("resnet18_cifar10", "resnet18_bloodmnist", "resnet18_imagenette"),
		("efficientnet_b0_cifar10", "efficientnet_b0_bloodmnist", "efficientnet_b0_imagenette"),
		("convnext_tiny_cifar10", "convnext_tiny_bloodmnist", "convnext_tiny_imagenette"),
		("distilbert_sst2", "distilbert_trec6"),
	)
	case_file_by_spec_key: Mapping[str, str] = {
		"efficientnet_b0_bloodmnist": "table_layer_candidates_efficientnet_bloodmnist.tex",
		"efficientnet_b0_imagenette": "table_layer_candidates_efficientnet_imagenette.tex",
		"distilbert_trec6": "table_layer_candidates_distilbert_trec6.tex",
	}
	return selection_method_keys, spec_groups_detail, case_file_by_spec_key


def _descriptor_cell(tex: Optional[str]) -> str:
	if not tex:
		return "---"
	return _escape_tex_cell(str(tex).strip())


def _rho_cell(v: Optional[float]) -> str:
	if v is None:
		return "---"
	try:
		x = float(v)
	except (TypeError, ValueError):
		return "---"
	if not math.isfinite(x):
		return "---"
	return f"{float(x):.3f}"


def _parse_corr_bench_row(corr_bench_row: str) -> Tuple[Optional[str], Optional[str]]:
	s = str(corr_bench_row or "")
	pref = "bench."
	if not s.startswith(pref) or "-val." not in s:
		return None, None
	ds = s[len(pref) :].split("-val.", 1)[0]
	bench = s.split("-val.", 1)[1]
	return (ds.strip().lower() or None), (bench.strip().lower() or None)


def _maybe_backfill_candidate_row(
	row: Dict[str, Any],
	*,
	method_key: str,
	meta_sel: Mapping[str, Any],
	r_by_layer: Mapping[str, float],
) -> None:
	corr_csv = str(meta_sel.get("corr_csv", "") or "").strip()
	if not corr_csv or not os.path.isfile(corr_csv):
		return
	ds_slug, bench_metric = _parse_corr_bench_row(str(meta_sel.get("corr_bench_row", "") or ""))
	if not ds_slug or not bench_metric:
		return
	edges_all = load_corr_edges(corr_csv, dataset_slug=str(ds_slug), bench_metric=str(bench_metric))
	sr = float(meta_sel.get("strict_min_abs_rho", 0.6))
	sp = float(meta_sel.get("strict_max_p", 0.05))
	r_star = None
	try:
		r_star = float(meta_sel.get("r_star")) if meta_sel.get("r_star") is not None else None
	except (TypeError, ValueError):
		r_star = None

	def set_r_fields(layer_name: str) -> None:
		if r_star is None:
			return
		if row.get("R") is None and str(layer_name) in r_by_layer:
			rv = float(r_by_layer[str(layer_name)])
			row["R"] = float(rv)
			row["gap_vs_best"] = float(max(0.0, float(r_star) - float(rv)))

	if method_key == "default_layer" and not row.get("layer"):
		dl = meta_sel.get("default_layer")
		if dl:
			row["layer"] = str(dl)
			set_r_fields(str(dl))

	if method_key in {"topo_strict_best_r", "spectral_strict_best_r"}:
		if row.get("descriptor_tex") is not None and row.get("abs_rho_s") is not None:
			return
		layer = row.get("layer")
		if not layer:
			return
		group = "topo" if method_key.startswith("topo_") else "spectral"
		edges = [e for e in edges_all if str(e.group) == group]
		edge = _best_strict_edge_for_layer(edges, str(layer), min_abs_rho=sr, max_p=sp)
		if not edge:
			return
		row["descriptor_tex"] = descriptor_tex_from_repr_key(str(edge.repr_key))
		row["abs_rho_s"] = float(edge.abs_rho)
		set_r_fields(str(layer))
		return

	if method_key in {"topo_max_abs_rho", "spectral_max_abs_rho", "mtopdiv_max_abs_rho"}:
		if row.get("layer") and row.get("descriptor_tex") is not None and row.get("abs_rho_s") is not None:
			return
		group = "topo" if method_key.startswith("topo_") else ("spectral" if method_key.startswith("spectral_") else "mtopdiv")
		edges = [e for e in edges_all if str(e.group) == group]
		edge = _argmax_abs_rho(edges)
		if not edge:
			return
		row["layer"] = str(edge.layer)
		row["descriptor_tex"] = descriptor_tex_from_repr_key(str(edge.repr_key))
		row["abs_rho_s"] = float(edge.abs_rho)
		set_r_fields(str(edge.layer))
		return


def _scalar_cell(opt: Optional[Any]) -> str:
	if opt is None:
		return "---"
	if isinstance(opt, float) and not math.isfinite(opt):
		return "---"
	try:
		v = float(opt)
		if abs(v) < 5e-4:
			v = 0.0
		return f"{v:.3f}"
	except (TypeError, ValueError):
		return "---"


def _method_display_en(key: str, sr: float, sp: float) -> str:
	if key == "best_r":
		return r"Best by $R$"
	if key == "default_layer":
		return r"Default layer"
	if key == "topo_strict_best_r":
		return rf"Best among topological candidates ($|\rho| \ge {sr:.1f}$, $p \le {sp:.2f}$)"
	if key == "topo_max_abs_rho":
		return r"Max $|\rho_S|$ (topological)"
	if key == "spectral_strict_best_r":
		return rf"Best among spectral candidates ($|\rho| \ge {sr:.1f}$, $p \le {sp:.2f}$)"
	if key == "spectral_max_abs_rho":
		return r"Max $|\rho_S|$ (spectral)"
	if key == "mtopdiv_strict_best_r":
		return rf"Best among MTopDiv candidates ($|\rho| \ge {sr:.1f}$, $p \le {sp:.2f}$)"
	if key == "mtopdiv_max_abs_rho":
		return r"Max $|\rho_S|$ (MTopDiv)"
	return str(key)


def _candidate_body_lines(selection: Mapping[str, Any], *, model_lc: str, r_by_layer: Mapping[str, float]) -> List[str]:
	rows_raw = selection.get("rows") if isinstance(selection.get("rows"), list) else []
	meta_sel = selection.get("meta") if isinstance(selection.get("meta"), dict) else {}
	sr = float(meta_sel.get("strict_min_abs_rho", 0.6))
	sp = float(meta_sel.get("strict_max_p", 0.05))
	by_k: Dict[str, Dict[str, Any]] = {}
	for r in rows_raw:
		if isinstance(r, dict) and r.get("method_key") is not None:
			by_k[str(r["method_key"])] = r
	lines: List[str] = []
	selection_method_keys, _spec_groups_detail, _case_file_by_spec_key = _layer_tables_config()
	for mk in selection_method_keys:
		r = by_k.get(mk)
		if not r:
			continue
		if isinstance(r, dict):
			_maybe_backfill_candidate_row(r, method_key=str(mk), meta_sel=meta_sel, r_by_layer=r_by_layer)
		method_txt = _method_display_en(str(mk), sr, sp)
		lay = r.get("layer")
		l_cell = "---" if not lay else _paper_layer_tt(model_lc, str(lay))
		r_txt = _scalar_cell(r.get("R"))
		g_txt = _scalar_cell(r.get("gap_vs_best"))
		desc = _descriptor_cell(r.get("descriptor_tex"))
		ar = _rho_cell(r.get("abs_rho_s"))
		lines.append(f"{method_txt} & {l_cell} & {r_txt} & {g_txt} & {desc} & {ar} \\\\")
	return lines


def _write_tex_fragment(tex_path: str, chunks: Sequence[str]) -> None:
	os.makedirs(os.path.dirname(os.path.abspath(tex_path)), exist_ok=True)
	with open(tex_path, "w", encoding="utf-8") as wf:
		for ch in chunks:
			wf.write(ch if str(ch).endswith("\n") else str(ch) + "\n")


def try_pick_spec_bundle(
	spec_key: str,
	*,
	runs_root: str,
	rng: random.Random,
	relpath_pins: Mapping[str, str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], PaperSpec]:
	sp = next(s for s in PAPER_SPECS if s.key == spec_key)
	pin_rel = layer_embedding_cases_relpath_for_spec_key(spec_key, overrides=relpath_pins)
	root_abs = os.path.abspath(str(runs_root))

	if pin_rel:
		rd_abs = pin_rel if os.path.isabs(pin_rel) else os.path.abspath(os.path.join(root_abs, pin_rel))
		bp = _bundle_path(rd_abs)
		meta_ok = os.path.isfile(os.path.join(rd_abs, "meta.json"))
		try:
			meta_match = meta_ok and sp.matches(load_run_meta(rd_abs))
		except Exception:
			meta_match = False
		bundle_ok = os.path.isdir(rd_abs) and os.path.isfile(bp)
		if bundle_ok and meta_match:
			return rd_abs, _read_json(bp), sp
		sys.stderr.write(f"[layer_tex] pin unusable or meta mismatch ({spec_key!r}); trying random bundle.\n")

	pool = _runs_with_bundle(sp, runs_root)
	if not pool:
		sys.stderr.write(f"[layer_tex] no embedding bundle matched spec {spec_key!r} under {runs_root}.\n")
		return None, None, sp
	rd_abs = rng.choice(pool)
	bp = _bundle_path(rd_abs)
	try:
		return rd_abs, _read_json(bp), sp
	except OSError as exc:
		sys.stderr.write(f"[layer_tex] unreadable bundle ({spec_key}): {exc}\n")
		return None, None, sp


def write_layer_candidate_table_tex(
	dest_dir: str,
	spec_key: str,
	bundle: Mapping[str, Any],
	model_lc: str,
) -> None:
	_selection_method_keys, _spec_groups_detail, case_file_by_spec_key = _layer_tables_config()
	fname = case_file_by_spec_key[str(spec_key)]
	path_tex = os.path.join(os.path.abspath(dest_dir), fname)
	sel = bundle.get("selection")
	if not isinstance(sel, dict):
		raise RuntimeError(f"{path_tex}: missing selection in bundle.")
	layers_blk = bundle.get("layers") if isinstance(bundle.get("layers"), dict) else {}
	r_by_layer: Dict[str, float] = {}
	for k, v in layers_blk.items():
		if not isinstance(v, dict):
			continue
		if "macro_same_class_ratio" in v:
			try:
				r_by_layer[str(k)] = float(v["macro_same_class_ratio"])
			except (TypeError, ValueError):
				continue
	body_lines = _candidate_body_lines(sel, model_lc=model_lc, r_by_layer=r_by_layer)
	embedding_candidates_header = (
		r"\textbf{Selection method} & \textbf{Layer} & \textbf{$R$} & \textbf{Gap ($\Delta R$)} & "
		r"\textbf{Descriptor} & \textbf{$|\rho_S|$}\\"
	)
	_write_tex_fragment(
		path_tex,
		[r"\toprule", embedding_candidates_header, r"\midrule"] + body_lines + [r"\bottomrule"],
	)


def write_layer_detailed_table_tex(
	dest_dir: str,
	*,
	runs_root: str,
	seed: int,
	relpath_pins: Mapping[str, str],
) -> None:
	rng_det = random.Random(int(seed))
	tex_out = os.path.join(os.path.abspath(dest_dir), "table_layer_selection_detailed.tex")
	blocks_out: List[str] = []
	_selection_method_keys, spec_groups_detail, _case_file_by_spec_key = _layer_tables_config()
	for gi, grp in enumerate(spec_groups_detail):
		chunk: List[str] = []
		for sk in grp:
			rd_abs, blob, spec = try_pick_spec_bundle(sk, runs_root=runs_root, rng=rng_det, relpath_pins=relpath_pins)
			if blob is None or rd_abs is None:
				sys.stderr.write(f"[layer_tex] detailed row omitted for {sk} (missing embedding bundle).\n")
				spc = next(s for s in PAPER_SPECS if s.key == sk)
				arch_cell = rf"{spc.arch_tex} {spc.cite_arch}"
				ds_cell = rf"{spc.dataset_tex} {spc.cite_ds}"
				chunk.append(f"{arch_cell} & {ds_cell} & --- & --- & --- & --- & --- \\\\")
				continue
			ln = detailed_line(spec, rd_abs, blob)
			if ln is None:
				sys.stderr.write(f"[layer_tex] detailed row missing for {sk} (no metrics); placeholder.\n")
				spc = next(s for s in PAPER_SPECS if s.key == sk)
				arch_cell = rf"{spc.arch_tex} {spc.cite_arch}"
				ds_cell = rf"{spc.dataset_tex} {spc.cite_ds}"
				ln = f"{arch_cell} & {ds_cell} & --- & --- & --- & --- & --- \\\\"
			chunk.append(ln)
		if gi > 0:
			blocks_out.append("\\midrule\n")
		for c in chunk:
			blocks_out.append(c if c.endswith("\n") else c + "\n")
	layer_selection_detailed_header = (
		r"\textbf{Architecture} & \textbf{Dataset} & \textbf{Exhaustive Best Layer} & \textbf{Max $R$} & "
		r"\textbf{Proposed Best Layer} & \textbf{Yielded $R$} & \textbf{Gap ($\Delta R$)} \\"
	)
	all_chunks: List[str] = (
		[r"\toprule", layer_selection_detailed_header, r"\midrule"]
		+ blocks_out
		+ [r"\bottomrule"]
	)
	_write_tex_fragment(tex_out, all_chunks)


def build_layer_embedding_tex_tables(
	*,
	runs_root: str,
	dest_tables_dir: str,
	detailed_seed: int,
	case_study_dirs: Mapping[str, str],
) -> None:
	root = os.path.abspath(str(runs_root))
	dd = os.path.abspath(dest_tables_dir)
	_selection_method_keys, _spec_groups_detail, case_file_by_spec_key = _layer_tables_config()
	for sk, rel in case_study_dirs.items():
		if sk not in case_file_by_spec_key:
			continue
		rd = os.path.abspath(os.path.join(root, rel)) if not os.path.isabs(rel) else rel
		bp = _bundle_path(rd)
		if not os.path.isfile(bp):
			sys.stderr.write(f"[layer_tex] skip case study {sk}: bundle missing ({bp})\n")
			continue
		bundle = _read_json(bp)
		meta_mp = os.path.join(rd, "meta.json")
		model_lc = ""
		if os.path.isfile(meta_mp):
			try:
				mj = _read_json(meta_mp)
				model_lc = str(mj.get("model", "") or "").strip().lower()
				if not model_lc:
					model_lc = str((mj.get("args") or {}).get("model", "") or "").strip().lower()
			except Exception as exc:
				sys.stderr.write(f"[layer_tex] unreadable meta for {sk}: {exc}\n")
		write_layer_candidate_table_tex(dd, sk, bundle, model_lc=model_lc or "distilbert")

	write_layer_detailed_table_tex(
		dd,
		runs_root=root,
		seed=int(detailed_seed),
		relpath_pins=dict(case_study_dirs),
	)
