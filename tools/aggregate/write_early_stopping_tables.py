from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as rf:
		obj = json.load(rf)
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
	xs = [float(x) for x in vals if x == x and math.isfinite(float(x))]
	if not xs:
		return float("nan"), float("nan")
	mu = sum(xs) / len(xs)
	if len(xs) < 2:
		return mu, 0.0
	var = sum((float(x) - mu) ** 2 for x in xs) / float(len(xs) - 1)
	return mu, math.sqrt(max(0.0, var))


def _discover_sweep_files(roots: Sequence[str]) -> List[str]:
	out: List[str] = []
	for rt in roots:
		root_abs = os.path.abspath(rt)
		if not os.path.isdir(root_abs):
			sys.stderr.write(f"[warn] missing root: {root_abs}\n")
			continue
		for dp, _, fnames in os.walk(root_abs):
			if "repr_early_stop_sweep.json" in fnames:
				out.append(os.path.join(dp, "repr_early_stop_sweep.json"))
	return sorted(set(out))


def _pretty_metric_tex(metric: str) -> str:
	m = str(metric)
	if m == "beta1_L_est":
		return r"$\beta_1(L)$"
	if m == "beta1_persistent_est":
		return r"$\beta_1^{K,L}$"
	if m == "hodge_L_q0_lambda2":
		return r"$\lambda_2(\Delta_0(L))$"
	if m == "persistent_q1_lambda1":
		return r"$\lambda_1(\Delta_1^{K,L})$"
	if m == "mtopdiv_train_val":
		return r"$\mathrm{MTopDiv}$"
	return m


def _pretty_plateau(mode: str) -> str:
	return "min" if str(mode).lower() == "min" else "max"


def _short_layer_name(layer: str) -> str:
	s = str(layer or "")
	i = s.rfind("layers.")
	if i >= 0:
		return s[i:]
	j = s.rfind("layer.")
	if j >= 0:
		return s[j:]
	return s


def _trigger_cell_single(layer: str, metric: str, mode: str) -> str:
	lay = _short_layer_name(str(layer)).replace("_", r"\_")
	return rf"\texttt{{{lay}}}: {_pretty_metric_tex(metric)} {_pretty_plateau(mode)}"


def _trigger_cell_ensemble(row: Mapping[str, Any]) -> str:
	ms = row.get("metrics") or []
	mds = row.get("modes") or []
	parts: List[str] = []
	for m, mo in zip(ms, mds):
		parts.append(f"{_pretty_metric_tex(str(m))} ({_pretty_plateau(str(mo))})")
	return " + ".join(parts) + rf", {str(row.get('aggregate', '')).upper()}, $p={int(row.get('patience', 0))}$"


def _training_last_epoch(sweep_obj: Mapping[str, Any], meta_obj: Mapping[str, Any]) -> int:
	if "last_training_epoch" in sweep_obj:
		return int(sweep_obj["last_training_epoch"])
	args_o = meta_obj.get("args", {}) or {}
	e = int(args_o.get("epochs", meta_obj.get("epochs", 0)))
	if e > 0:
		return int(e) - 1
	raise RuntimeError("Cannot infer final epoch index (add last_training_epoch to sweep JSON or epochs in meta).")


def _row_from_json_best(sweep_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
	obj = _load_json(sweep_path)
	run_dir = str(obj.get("run_dir") or "").strip()
	if not run_dir or not os.path.isdir(run_dir):
		run_dir = os.path.abspath(os.path.join(os.path.dirname(sweep_path), ".."))
	meta_path = os.path.join(run_dir, "meta.json")
	if not os.path.isfile(meta_path):
		raise FileNotFoundError(meta_path)
	meta = _load_json(meta_path)
	row = obj.get("best")
	if not isinstance(row, dict):
		raise RuntimeError(f"no best in {sweep_path}")
	last_ep = _training_last_epoch(obj, meta)
	triggered = row.get("triggered")
	if triggered is None:
		triggered = row.get("stop_epoch") is not None
	row = dict(row)
	if "epochs_saved" not in row:
		es = int(row.get("effective_stop_epoch", last_ep))
		row["epochs_saved"] = max(0, int(last_ep) - es)
	row["triggered"] = bool(triggered)
	return meta, row


def _group_key(model: str, dataset: str, group_by: str) -> str:
	if group_by == "model":
		return str(model)
	if group_by == "model_dataset":
		return f"{model}::{dataset}"
	raise ValueError(group_by)


def _normalize_model_key(raw: str) -> str:
	m = str(raw or "").strip().lower()
	if m.startswith("distilbert"):
		return "distilbert"
	if m in ("smollm", "smollm2", "smollm2-135m") or "smollm" in m:
		return "smollm2-135m"
	return m


def _parse_exclude(exclude_csv: str) -> set[str]:
	return {x.strip().lower() for x in str(exclude_csv).split(",") if x.strip()}


def _ensemble_signals_only(row: Mapping[str, Any]) -> str:
	ms = row.get("metrics") or []
	mds = row.get("modes") or []
	parts = [f"{_pretty_metric_tex(str(m))} ({_pretty_plateau(str(mo))})" for m, mo in zip(ms, mds)]
	return " + ".join(parts)


def _ensemble_rule_column(row: Mapping[str, Any]) -> str:
	return rf"{str(row.get('aggregate', '')).upper()}, $p={int(row.get('patience', 0))}$"


def _primary_trigger_tex(row: Mapping[str, Any]) -> str:
	k = str(row.get("kind", "single")).lower()
	if k == "ensemble3":
		return f"{_ensemble_signals_only(row)}, {_ensemble_rule_column(row)}"
	return _trigger_cell_single(str(row.get("layer", "")), str(row.get("metric", "")), str(row.get("mode", "")))


def _gap_primary_cell(m_gr: float, s_gr: float, m_bd: float, s_bd: float) -> str:
	if not (math.isfinite(m_gr) and math.isfinite(s_gr)):
		return r"---"
	gp = f"{m_gr:.1f}\\% \\pm {s_gr:.1f}\\%"
	bd = f"{m_bd:.3f} \\pm {s_bd:.3f}" if math.isfinite(m_bd) and math.isfinite(s_bd) else "---"
	return rf"${gp} \bigm\left({bd}\right)$"


def write_table_early_stopping_tex(
	path: str,
	roots: Sequence[str],
	*,
	group_by: str = "model",
	exclude_models: str = "",
) -> bool:
	root_list = [os.path.abspath(str(r)) for r in roots]
	excl = _parse_exclude(exclude_models)
	by_group: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
	n_ok = 0
	for sp in _discover_sweep_files(root_list):
		try:
			meta, row = _row_from_json_best(sp)
			model = str(meta.get("model", "")).strip()
			ds = str(meta.get("dataset", "")).strip()
			mk = _normalize_model_key(model)
			if mk in excl:
				continue
			gk = _group_key(mk, ds, group_by)
			by_group[gk].append({"meta": meta, "row": row, "sweep_path": sp})
			n_ok += 1
		except Exception as exc:
			sys.stderr.write(f"[early_stop_tex] skip {sp}: {exc}\n")

	if n_ok == 0:
		sys.stderr.write("[early_stop_tex] no valid repr_early_stop_sweep.json files; skipping primary table.\n")
		return False

	header = (
		r"\textbf{Architecture} & \textbf{Primary Trigger Signal} & \textbf{Successful Triggers (\%)} & "
		r"\textbf{Epochs Saved} & \textbf{Gap to Empirical Optimum} \\"
	)
	out_lines = [r"\toprule", header, r"\midrule"]

	used_keys: set[str] = set()
	paper_primary_order: Tuple[Tuple[str, str], ...] = (
		("mlp", r"MLP"),
		("resnet18", r"ResNet18"),
		("efficientnet_b0", r"EfficientNet-B0"),
		("convnext_tiny", r"ConvNeXt-Tiny"),
		("distilbert", r"DistilBERT"),
		("smollm2-135m", r"SmolLM"),
	)
	for mk_paper, arch_tex in paper_primary_order:
		items = by_group.get(mk_paper, [])
		if not items:
			continue
		used_keys.add(mk_paper)
		trig = [1.0 if bool(it["row"].get("triggered", it["row"].get("stop_epoch") is not None)) else 0.0 for it in items]
		succ_pct = 100.0 * sum(trig) / max(len(trig), 1)
		ep_sav = [float(it["row"].get("epochs_saved", 0)) for it in items]
		gap_rel = [float(it["row"].get("gap_rel_pct", 0.0)) for it in items]
		bd = [float(it["row"].get("bench_delta_at_stop", 0.0)) for it in items]
		m_ep, s_ep = _mean_std(ep_sav)
		m_gr, s_gr = _mean_std(gap_rel)
		m_bd, s_bd = _mean_std(bd)
		first = items[0]["row"]
		trig_tex = _primary_trigger_tex(first)
		gap_cell = _gap_primary_cell(m_gr, s_gr, m_bd, s_bd)
		out_lines.append(
			f"{arch_tex} & {trig_tex} & {succ_pct:.1f}\\% & {m_ep:.1f} $\\pm$ {s_ep:.1f} & {gap_cell} \\\\"
		)

	remaining_keys = sorted(k for k in by_group.keys() if k not in used_keys)
	for mk in remaining_keys:
		items = by_group.get(mk, [])
		if not items:
			continue
		trig = [1.0 if bool(it["row"].get("triggered", it["row"].get("stop_epoch") is not None)) else 0.0 for it in items]
		succ_pct = 100.0 * sum(trig) / max(len(trig), 1)
		ep_sav = [float(it["row"].get("epochs_saved", 0)) for it in items]
		gap_rel = [float(it["row"].get("gap_rel_pct", 0.0)) for it in items]
		bd = [float(it["row"].get("bench_delta_at_stop", 0.0)) for it in items]
		m_ep, s_ep = _mean_std(ep_sav)
		m_gr, s_gr = _mean_std(gap_rel)
		m_bd, s_bd = _mean_std(bd)
		first = items[0]["row"]
		trig_tex = _primary_trigger_tex(first)
		gap_cell = _gap_primary_cell(m_gr, s_gr, m_bd, s_bd)
		mk_tex = str(mk).replace("_", "\\_")
		arch_tex = rf"\texttt{{{mk_tex}}}"
		out_lines.append(
			f"{arch_tex} & {trig_tex} & {succ_pct:.1f}\\% & {m_ep:.1f} $\\pm$ {s_ep:.1f} & {gap_cell} \\\\"
		)

	out_lines.append(r"\bottomrule")
	os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as wf:
		wf.write("\n".join(out_lines) + "\n")
	return True


def _ensemble_key(row: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
	if str(row.get("kind", "")).lower() != "ensemble3":
		return None
	ms = row.get("metrics") or []
	md = row.get("modes") or []
	if len(ms) != 3 or len(md) != 3:
		return None
	return (
		tuple(str(x) for x in ms),
		tuple(str(x) for x in md),
		str(row.get("aggregate", "")).lower(),
		int(row.get("patience", 0)),
	)


def write_table_early_stopping_best_tex(
	path: str,
	roots: Sequence[str],
	*,
	top_k: int = 6,
	exclude_models: str = "",
) -> bool:
	root_list = [os.path.abspath(str(r)) for r in roots]
	excl = _parse_exclude(exclude_models)

	sig_to_runs: MutableMapping[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

	for sp in _discover_sweep_files(root_list):
		try:
			run_dir = os.path.abspath(os.path.join(os.path.dirname(sp), ".."))
			meta = _load_json(os.path.join(run_dir, "meta.json"))
			mk = _normalize_model_key(str(meta.get("model", "")))
			if mk in excl:
				continue
			obj = _load_json(sp)
			ranked = obj.get("ensemble3_oracle_rows") if isinstance(obj.get("ensemble3_oracle_rows"), list) else []
			if not ranked:
				ranked = obj.get("ranked_by_gap_rel_pct") if isinstance(obj.get("ranked_by_gap_rel_pct"), list) else []
			best_by_sig: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
			for r in ranked:
				if not isinstance(r, dict):
					continue
				key = _ensemble_key(r)
				if key is None:
					continue
				gap = float(r.get("gap_rel_pct", 1e9))
				if key not in best_by_sig or gap < float(best_by_sig[key].get("gap_rel_pct", 1e9)):
					best_by_sig[key] = r
			for _k, r in best_by_sig.items():
				sig_to_runs[_k].append(dict(r))
		except Exception as exc:
			sys.stderr.write(f"[early_stop_tex] skip ensemble scan {sp}: {exc}\n")

	if not sig_to_runs:
		sys.stderr.write("[early_stop_tex] no ensemble3 rows for best table; skipping.\n")
		return False

	agg: List[Tuple[float, Tuple[Any, ...], Dict[str, Any], float, float, float, float, float]] = []
	for key, rows in sig_to_runs.items():
		if len(rows) < 1:
			continue
		trig = [1.0 if bool(r.get("triggered", r.get("stop_epoch") is not None)) else 0.0 for r in rows]
		succ_pct = 100.0 * sum(trig) / max(len(trig), 1)
		ep_sav = [float(r.get("epochs_saved", 0)) for r in rows]
		gap_rel = [float(r.get("gap_rel_pct", 0)) for r in rows]
		m_ep, s_ep = _mean_std(ep_sav)
		m_gr, s_gr = _mean_std(gap_rel)
		rep = min(rows, key=lambda r: float(r.get("gap_rel_pct", 1e9)))
		mean_gap = sum(gap_rel) / max(len(gap_rel), 1)
		agg.append((mean_gap, key, rep, succ_pct, m_ep, s_ep, m_gr, s_gr))

	agg.sort(key=lambda t: (t[0], str(t[1])))
	top = agg[: int(top_k)]

	header = (
		r"\textbf{Signals} & \textbf{Rule} & \textbf{Successful Triggers (\%)} & "
		r"\textbf{Epochs Saved} & \textbf{Quality Drop}\\"
	)
	out_lines = [r"\toprule", header, r"\midrule"]
	for _mg, _key, rep, succ_pct, m_ep, s_ep, m_gr, s_gr in top:
		sig_part = _ensemble_signals_only(rep)
		rule_part = _ensemble_rule_column(rep)
		out_lines.append(
			f"{sig_part} & {rule_part} & {succ_pct:.1f}\\% & {m_ep:.1f} $\\pm$ {s_ep:.1f} & "
			f"{m_gr:.2f}\\% $\\pm$ {s_gr:.2f}\\% \\\\"
		)
	out_lines.append(r"\bottomrule")
	os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as wf:
		wf.write("\n".join(out_lines) + "\n")
	return True


def write_both_early_stopping_tables(
	tables_dir: str,
	runs_roots: Sequence[str],
	*,
	exclude_models: str = "",
	top_k_best: int = 6,
) -> None:
	td = os.path.abspath(tables_dir)
	p1 = os.path.join(td, "table_early_stopping.tex")
	p2 = os.path.join(td, "table_early_stopping_best.tex")
	a = write_table_early_stopping_tex(p1, runs_roots, exclude_models=exclude_models)
	b = write_table_early_stopping_best_tex(p2, runs_roots, top_k=int(top_k_best), exclude_models=exclude_models)
	if a or b:
		sys.stderr.write(f"[early_stop_tex] wrote primary={a} best={b} -> {td}\n")
