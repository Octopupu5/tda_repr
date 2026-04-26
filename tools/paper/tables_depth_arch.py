from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _iter_jsonl(path: str) -> Iterable[dict]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def _median(xs: List[float]) -> Optional[float]:
	xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
	if not xs:
		return None
	xs.sort()
	m = len(xs) // 2
	if len(xs) % 2 == 1:
		return xs[m]
	return 0.5 * (xs[m - 1] + xs[m])


def _mean(xs: List[float]) -> Optional[float]:
	xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
	if not xs:
		return None
	return sum(xs) / float(len(xs))


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float], int]:
	vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)) and math.isfinite(float(x))]
	if not vals:
		return None, None, 0
	mu = float(statistics.fmean(vals))
	if len(vals) >= 2:
		sd = float(statistics.stdev(vals))
	else:
		sd = 0.0
	return mu, sd, int(len(vals))


def _fmt(x: Optional[float], nd: int = 3) -> str:
	if x is None or (isinstance(x, float) and math.isnan(x)):
		return "NA"
	fmt = f"{{:.{int(nd)}f}}"
	return fmt.format(float(x))


def _fmt_pm(mean: Optional[float], std: Optional[float], nd: int = 3, signed: bool = False) -> str:
	if mean is None or (isinstance(mean, float) and (math.isnan(mean) or not math.isfinite(float(mean)))):
		return "NA"
	if std is None or (isinstance(std, float) and (math.isnan(std) or not math.isfinite(float(std)))):
		std = 0.0
	pref = "+" if bool(signed) else ""
	return f"{float(mean):{pref}.{int(nd)}f} $\\pm$ {float(std):.{int(nd)}f}"


def _fmt_pct(x: Optional[float], nd: int = 1) -> str:
	if x is None or (isinstance(x, float) and math.isnan(x)):
		return "NA"
	return f"{float(x) * 100.0:.{int(nd)}f}%"


def _fmt_pct_pm(mean: Optional[float], std: Optional[float], nd: int = 1) -> str:
	if mean is None or (isinstance(mean, float) and (math.isnan(mean) or not math.isfinite(float(mean)))):
		return "NA"
	if std is None or (isinstance(std, float) and (math.isnan(std) or not math.isfinite(float(std)))):
		std = 0.0
	return f"{100.0*float(mean):.{int(nd)}f}\\% $\\pm$ {100.0*float(std):.{int(nd)}f}\\%"

def _pick_primary_metric(bench: dict) -> Tuple[str, float]:
	"""
	Return (metric_name, value). Prefers:
	- ppl for generation-style runs
	- f1_macro for classification-style runs
	- then accuracy / bleu / loss-like metrics
	"""
	for k in ("ppl",):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("f1_macro", "accuracy", "bleu"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("loss_assistant_only", "loss", "ppl"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	return "unknown", float("nan")


def _is_minimization_metric(metric_name: str) -> bool:
	# Lower is better for loss / perplexity style metrics.
	m = str(metric_name).lower().strip()
	if m in {"loss", "loss_assistant_only", "ppl"}:
		return True
	return m.endswith("loss")


def _first_positive(eigs: Any, zero_tol: float) -> Optional[float]:
	if not isinstance(eigs, (list, tuple)) or not eigs:
		return None
	vals = []
	for v in eigs:
		try:
			fv = float(v)
		except Exception:
			continue
		if fv > float(zero_tol):
			vals.append(fv)
	if not vals:
		return None
	return min(vals)


def _first_two_positive(eigs: Any, zero_tol: float) -> Tuple[Optional[float], Optional[float]]:
	if not isinstance(eigs, (list, tuple)) or not eigs:
		return None, None
	vals = []
	for v in eigs:
		try:
			fv = float(v)
		except Exception:
			continue
		if fv > float(zero_tol):
			vals.append(fv)
	vals.sort()
	if not vals:
		return None, None
	if len(vals) == 1:
		return vals[0], None
	return vals[0], vals[1]


def _depth_rank_for_layer(model: str, layer: str, layer_names: List[str]) -> Optional[float]:
	"""
	Heuristic: assign a depth rank (not necessarily normalized). Higher = deeper.
	"""
	m = str(model).lower().strip()
	name = str(layer)

	# Transformers
	if "distilbert" in m or "smollm" in m or "llama" in m:
		if "embeddings" in name or "embed_tokens" in name or name.endswith("wte"):
			return 0.0
		# distilbert.transformer.layer.{i} or model.layers.{i}
		mt = re.search(r"(?:transformer\.layer|model\.layers|layers)\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			# ffn slightly after block
			if ".ffn" in name:
				return float(i) + 0.25
			return float(i) + 0.1
		if "pre_classifier" in name:
			return 1000.0
		if name.endswith("classifier") or name.endswith("score") or "lm_head" in name:
			return 1100.0
		if name.endswith("norm") or ".norm" in name or "ln_f" in name:
			return 1050.0

	# ResNet
	if "resnet" in m:
		if name in ("conv1", "bn1", "relu", "maxpool"):
			return 0.0
		mt = re.search(r"layer(\d+)\.(\d+)", name)
		if mt:
			stage = int(mt.group(1))
			block = int(mt.group(2))
			return float(stage * 10 + block)
		mt2 = re.search(r"layer(\d+)$", name)
		if mt2:
			stage = int(mt2.group(1))
			return float(stage * 10)
		if name == "avgpool":
			return 1000.0
		if name == "fc":
			return 1100.0

	# EfficientNet
	if "efficientnet" in m:
		mt = re.search(r"features\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			# sub-blocks
			mt2 = re.search(r"features\.(\d+)\.(\d+)", name)
			if mt2:
				j = int(mt2.group(2))
				return float(i * 10 + j)
			return float(i * 10)
		if name in ("avgpool", "classifier") or name.startswith("classifier"):
			return 1100.0

	# ConvNeXt
	if "convnext" in m:
		mt = re.search(r"features\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			mt2 = re.search(r"features\.(\d+)\.(\d+)", name)
			if mt2:
				j = int(mt2.group(2))
				return float(i * 10 + j)
			return float(i * 10)
		if "classifier" in name or name == "head":
			return 1100.0

	# MLP / generic sequential
	mt = re.search(r"(?:^|\.)((\d+))(?:$|\.)", name)
	if mt:
		return float(int(mt.group(1)))

	# Fallback: list index order
	try:
		return float(layer_names.index(layer))
	except Exception:
		return None


def _depth_group(model: str, layer: str, layer_names: List[str]) -> str:
	ranks = []
	for ln in layer_names:
		r = _depth_rank_for_layer(model, ln, layer_names)
		if r is not None and math.isfinite(r):
			ranks.append(r)
	if not ranks:
		return "intermediate"
	r_min = float(min(ranks))
	r_max = float(max(ranks))
	r = _depth_rank_for_layer(model, layer, layer_names)
	if r is None or not math.isfinite(r) or r_max <= r_min:
		# fallback to position
		try:
			pos = layer_names.index(layer) / max(1.0, float(len(layer_names) - 1))
		except Exception:
			pos = 0.5
	else:
		pos = (float(r) - r_min) / max(1e-12, (r_max - r_min))

	if pos <= 1.0 / 3.0:
		return "early"
	if pos <= 2.0 / 3.0:
		return "intermediate"
	return "deep"


def _display_arch(model: str, pretrained: Optional[bool]) -> str:
	m = str(model).lower().strip()
	if m == "resnet18":
		return "ResNet18"
	if m in ("efficientnet_b0", "efficientnet"):
		return "EfficientNet-B0"
	if m in ("convnext_tiny", "convnext"):
		return "ConvNeXt-Tiny"
	if m == "mlp":
		return "MLP"
	if "distilbert" in m:
		if pretrained is None:
			return "DistilBERT"
		return "DistilBERT [pretrained]" if pretrained else "DistilBERT [untrained]"
	if "smollm" in m:
		return "SmolLM"
	return model


def _display_dataset(ds: str) -> str:
	k = str(ds).lower().strip()
	return {
		"cifar10": "CIFAR-10",
		"imagenette": "ImageNette",
		"bloodmnist": "BloodMNIST",
		"mnist": "MNIST",
		"trec6": "Trec-6",
		"sst2": "SST-2",
		"yahoo_answers_topics": "Yahoo Ans",
		"smol-summarize": "SmolTalk (summarize)",
	}.get(k, ds)


@dataclass
class LayerDelta:
	group: str
	d_beta1_L: Optional[float]
	d_beta1_pers: Optional[float]
	d_lambda2_q0: Optional[float]
	d_l1_q1_pers: Optional[float]
	d_l2_q1_pers: Optional[float]
	d_mtopdiv: Optional[float]


@dataclass
class RunSummary:
	run_dir: str
	task: str
	dataset: str
	model: str
	pretrained: Optional[bool]
	nlp_objective: str
	primary_metric: str
	d_task_metric: Optional[float]
	per_layer: List[LayerDelta]


def _extract_epoch_repr_scalar(layer_out: dict, key: str) -> Optional[float]:
	v = layer_out.get(key, None)
	if isinstance(v, (int, float)):
		return float(v)
	return None


def _compute_layer_delta(first: dict, last: dict, zero_tol: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
	b1_L_0 = _extract_epoch_repr_scalar(first, "beta1_L_est")
	b1_L_1 = _extract_epoch_repr_scalar(last, "beta1_L_est")
	d_b1_L = (b1_L_1 - b1_L_0) if (b1_L_0 is not None and b1_L_1 is not None) else None

	b1_p_0 = _extract_epoch_repr_scalar(first, "beta1_persistent_est")
	b1_p_1 = _extract_epoch_repr_scalar(last, "beta1_persistent_est")
	d_b1_p = (b1_p_1 - b1_p_0) if (b1_p_0 is not None and b1_p_1 is not None) else None

	l2_0 = _first_positive(first.get("hodge_L_q0_smallest", None), zero_tol=zero_tol)
	l2_1 = _first_positive(last.get("hodge_L_q0_smallest", None), zero_tol=zero_tol)
	d_l2 = (l2_1 - l2_0) if (l2_0 is not None and l2_1 is not None) else None

	l1q1_0, l2q1_0 = _first_two_positive(first.get("persistent_q1_smallest", None), zero_tol=zero_tol)
	l1q1_1, l2q1_1 = _first_two_positive(last.get("persistent_q1_smallest", None), zero_tol=zero_tol)
	d_l1q1 = (l1q1_1 - l1q1_0) if (l1q1_0 is not None and l1q1_1 is not None) else None
	d_l2q1 = (l2q1_1 - l2q1_0) if (l2q1_0 is not None and l2q1_1 is not None) else None

	m0 = _extract_epoch_repr_scalar(first, "mtopdiv_train_val")
	m1 = _extract_epoch_repr_scalar(last, "mtopdiv_train_val")
	d_m = (m1 - m0) if (m0 is not None and m1 is not None) else None

	return d_b1_L, d_b1_p, d_l2, d_l1q1, d_l2q1, d_m


def _load_run(run_dir: str) -> Optional[RunSummary]:
	meta_path = os.path.join(run_dir, "meta.json")
	metrics_path = os.path.join(run_dir, "metrics.jsonl")
	if not (os.path.isfile(meta_path) and os.path.isfile(metrics_path)):
		return None

	meta = _read_json(meta_path)
	args = meta.get("args", {}) or {}
	task = str(meta.get("task", args.get("task", "")) or "")
	dataset = str(meta.get("dataset", args.get("dataset", "")) or "")
	model = str(meta.get("model", args.get("model", "")) or "")
	pretrained = args.get("pretrained", None)
	if isinstance(pretrained, bool) is False:
		pretrained = None
	nlp_objective = str(args.get("nlp_objective", "classification") or "classification")

	monitor = meta.get("monitor", {}) or {}
	zero_tol = float(monitor.get("zero_tol", 1e-8) or 1e-8)
	layer_names = list((monitor.get("layer_names") or []) if isinstance(monitor.get("layer_names"), list) else [])

	# Find first epoch and best-validation epoch (first-to-best).
	epochs: List[Tuple[int, dict, dict]] = []
	for ev in _iter_jsonl(metrics_path):
		if str(ev.get("event", "")) != "epoch_end":
			continue
		ep = ev.get("epoch", None)
		if not isinstance(ep, int):
			continue
		repr0 = (ev.get("repr", {}) or {}).get("layers", None)
		bench0 = ev.get("bench", {}) or {}
		if not isinstance(repr0, dict):
			continue
		epochs.append((ep, repr0, bench0))

	if not epochs:
		return None
	epochs.sort(key=lambda x: x[0])
	first_epoch, first_repr_layers, first_bench = epochs[0]

	# Primary metric delta
	val_key = f"{dataset}-val"
	b0 = (first_bench or {}).get(val_key, {}) or {}
	metric_name0, metric_val0 = _pick_primary_metric(b0)

	primary_metric = metric_name0
	if primary_metric == "unknown":
		for _, _, b in epochs[1:]:
			bx = (b or {}).get(val_key, {}) or {}
			mn, _ = _pick_primary_metric(bx)
			if mn != "unknown":
				primary_metric = mn
				break

	# Select best epoch by validation metric.
	best_epoch, best_repr_layers, best_bench = epochs[-1]
	best_val = None
	for ep, repr_layers, bench in epochs:
		bx = (bench or {}).get(val_key, {}) or {}
		if not isinstance(bx, dict):
			continue
		v = bx.get(primary_metric, None)
		if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
			continue
		fv = float(v)
		if best_val is None:
			best_val = fv
			best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench
			continue
		if _is_minimization_metric(primary_metric):
			if fv < float(best_val):
				best_val = fv
				best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench
		else:
			if fv > float(best_val):
				best_val = fv
				best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench

	b1 = (best_bench or {}).get(val_key, {}) or {}
	metric_val1 = b1.get(primary_metric, None)
	if not isinstance(metric_val1, (int, float)) or not math.isfinite(float(metric_val1)) or not math.isfinite(float(metric_val0)):
		d_task = None
	else:
		if _is_minimization_metric(primary_metric):
			d_task = float(metric_val0) - float(metric_val1)
		else:
			d_task = float(metric_val1) - float(metric_val0)

	per_layer: List[LayerDelta] = []
	# Use layer_names from meta if present; else use intersection of repr keys.
	layers = layer_names or sorted(set(first_repr_layers.keys()) & set(best_repr_layers.keys()))
	for lay in layers:
		a = first_repr_layers.get(lay, {}) or {}
		b = best_repr_layers.get(lay, {}) or {}
		if not isinstance(a, dict) or not isinstance(b, dict):
			continue
		g = _depth_group(model, lay, layers)
		d_b1_L, d_b1_p, d_l2, d_l1q1, d_l2q1, d_m = _compute_layer_delta(a, b, zero_tol=zero_tol)
		per_layer.append(
			LayerDelta(
				group=g,
				d_beta1_L=d_b1_L,
				d_beta1_pers=d_b1_p,
				d_lambda2_q0=d_l2,
				d_l1_q1_pers=d_l1q1,
				d_l2_q1_pers=d_l2q1,
				d_mtopdiv=d_m,
			)
		)

	return RunSummary(
		run_dir=run_dir,
		task=task,
		dataset=dataset,
		model=model,
		pretrained=pretrained,
		nlp_objective=nlp_objective,
		primary_metric=primary_metric,
		d_task_metric=d_task,
		per_layer=per_layer,
	)


def _aggregate_depth_dynamics(runs: List[RunSummary]) -> dict:
	"""
	Returns group -> metric -> {mean, std, n}.
	Each run contributes equally: compute per-run median per group, then mean/std across runs.
	"""
	groups = ("early", "intermediate", "deep")
	metrics = ("d_beta1_L", "d_beta1_pers", "d_lambda2_q0", "d_l1_q1_pers", "d_l2_q1_pers", "frac_mtopdiv_pos")
	per_run: Dict[str, Dict[str, List[float]]] = {g: {m: [] for m in metrics} for g in groups}

	for r in runs:
		for g in groups:
			items = [ld for ld in r.per_layer if ld.group == g]
			if not items:
				continue
			for key in ("d_beta1_L", "d_beta1_pers", "d_lambda2_q0", "d_l1_q1_pers", "d_l2_q1_pers"):
				vals = [getattr(it, key) for it in items if getattr(it, key) is not None]
				m = _median([float(v) for v in vals]) if vals else None
				if m is not None:
					per_run[g][key].append(m)
			# Fraction with delta MTopDiv > 0 in this group (per run)
			ms = [it.d_mtopdiv for it in items if it.d_mtopdiv is not None]
			if ms:
				frac = sum(1 for v in ms if float(v) > 0.0) / float(len(ms))
				per_run[g]["frac_mtopdiv_pos"].append(float(frac))

	out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {g: {} for g in groups}
	for g in groups:
		for key in metrics:
			mu, sd, n = _mean_std(per_run[g][key])
			out[g][key] = {"mean": mu, "std": sd, "n": float(n)}
	return out


def _aggregate_depth_dynamics_debug(runs: List[RunSummary]) -> List[dict]:
	"""
	More "precise" diagnostics for the depth table:
	- median across runs of per-run group-median deltas (same as paper)
	- mean across runs of per-run group-median deltas
	- pooled median/mean over all (run,layer) deltas within a group
	- fraction of nonzero deltas (useful for discrete Betti numbers)
	"""
	groups = ("early", "intermediate", "deep")
	keys = ("d_beta1_L", "d_beta1_pers", "d_lambda2_q0", "d_l1_q1_pers", "d_l2_q1_pers", "d_mtopdiv")

	# per-run group medians
	per_run_meds: Dict[str, Dict[str, List[float]]] = {g: {k: [] for k in keys} for g in groups}
	# pooled (all layers, all runs)
	pooled: Dict[str, Dict[str, List[float]]] = {g: {k: [] for k in keys} for g in groups}

	for r in runs:
		for g in groups:
			items = [ld for ld in r.per_layer if ld.group == g]
			if not items:
				continue
			for k in keys:
				vals = [getattr(it, k) for it in items if getattr(it, k) is not None]
				for v in vals:
					pooled[g][k].append(float(v))
				m = _median([float(v) for v in vals]) if vals else None
				if m is not None:
					per_run_meds[g][k].append(float(m))

	rows: List[dict] = []
	for g in groups:
		for k in keys:
			pm = per_run_meds[g][k]
			pl = pooled[g][k]
			rows.append(
				{
					"group": g,
					"metric": k,
					"paper_median_across_runs": _median(pm),
					"mean_across_runs": _mean(pm),
					"pooled_median": _median(pl),
					"pooled_mean": _mean(pl),
					"pooled_frac_nonzero": (sum(1 for v in pl if float(v) != 0.0) / float(len(pl))) if pl else None,
					"n_runs_contrib": len(pm),
					"n_pooled": len(pl),
				}
			)
	return rows


def _aggregate_arch_table(runs: List[RunSummary]) -> List[dict]:
	"""
	Aggregate deep-layer medians per run, then mean/std across runs for each (arch,dataset,pretrained/nlp objective).
	"""
	buckets: Dict[Tuple[str, str, str], List[RunSummary]] = {}
	for r in runs:
		arch = _display_arch(r.model, r.pretrained if r.task == "nlp" and "distilbert" in r.model.lower() else None)
		ds = _display_dataset(r.dataset)
		mod = "NLP" if str(r.task).lower().strip() == "nlp" else "CV"
		# separate pretrained/untrained DistilBERT; other models just by name
		if "distilbert" in r.model.lower() and r.pretrained is not None:
			arch = _display_arch(r.model, r.pretrained)
		key = (arch, ds, mod)
		buckets.setdefault(key, []).append(r)

	rows: List[dict] = []
	for (arch, ds, mod), rs in sorted(buckets.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
		# per-run deep medians
		d_task = []
		db1p = []
		dmt = []
		dl2 = []
		dl1q1 = []
		for r in rs:
			deep = [ld for ld in r.per_layer if ld.group == "deep"]
			if not deep:
				continue
			if r.d_task_metric is not None:
				d_task.append(float(r.d_task_metric))
			v = _median([float(x.d_beta1_pers) for x in deep if x.d_beta1_pers is not None])
			if v is not None:
				db1p.append(v)
			v = _median([float(x.d_mtopdiv) for x in deep if x.d_mtopdiv is not None])
			if v is not None:
				dmt.append(v)
			v = _median([float(x.d_lambda2_q0) for x in deep if x.d_lambda2_q0 is not None])
			if v is not None:
				dl2.append(v)
			v = _median([float(x.d_l1_q1_pers) for x in deep if x.d_l1_q1_pers is not None])
			if v is not None:
				dl1q1.append(v)

		m_d_task, s_d_task, n_task = _mean_std(d_task)
		m_db1p, s_db1p, n_db1p = _mean_std(db1p)
		m_dmt, s_dmt, n_dmt = _mean_std(dmt)
		m_dl2, s_dl2, n_dl2 = _mean_std(dl2)
		m_dl1q1, s_dl1q1, n_dl1q1 = _mean_std(dl1q1)

		# simplified topology heuristic: persistent cycles decrease on deep layers
		simpl = None
		if m_db1p is not None:
			simpl = bool(m_db1p < 0.0)

		rows.append(
			{
				"architecture": arch,
				"dataset": ds,
				"modality": mod,
				"d_task_metric_mean": m_d_task,
				"d_task_metric_std": s_d_task,
				"d_task_metric_n": float(n_task),
				"primary_metric": rs[0].primary_metric if rs else "",
				"d_beta1_persistent_mean": m_db1p,
				"d_beta1_persistent_std": s_db1p,
				"d_beta1_persistent_n": float(n_db1p),
				"d_mtopdiv_mean": m_dmt,
				"d_mtopdiv_std": s_dmt,
				"d_mtopdiv_n": float(n_dmt),
				"d_lambda2_q0_mean": m_dl2,
				"d_lambda2_q0_std": s_dl2,
				"d_lambda2_q0_n": float(n_dl2),
				"d_lambda1_q1_persistent_mean": m_dl1q1,
				"d_lambda1_q1_persistent_std": s_dl1q1,
				"d_lambda1_q1_persistent_n": float(n_dl1q1),
				"simplified_topology": simpl,
				"n_runs": len(rs),
			}
		)
	return rows


def _write_csv(path: str, rows: List[dict]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	if not rows:
		return
	keys = list(rows[0].keys())
	with open(path, "w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=keys)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def _write_depth_tex(path: str, depth: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	# LaTeX rows
	# Order: early, intermediate, deep
	def row(g: str) -> str:
		db1 = _fmt_pm(depth[g]["d_beta1_L"].get("mean"), depth[g]["d_beta1_L"].get("std"), nd=1, signed=True)
		db1p = _fmt_pm(depth[g]["d_beta1_pers"].get("mean"), depth[g]["d_beta1_pers"].get("std"), nd=1, signed=True)
		dl2 = _fmt_pm(depth[g]["d_lambda2_q0"].get("mean"), depth[g]["d_lambda2_q0"].get("std"), nd=3, signed=True)
		dl1 = _fmt_pm(depth[g]["d_l1_q1_pers"].get("mean"), depth[g]["d_l1_q1_pers"].get("std"), nd=3, signed=True)
		dl2q1 = _fmt_pm(depth[g]["d_l2_q1_pers"].get("mean"), depth[g]["d_l2_q1_pers"].get("std"), nd=3, signed=True)
		fr = _fmt_pct_pm(depth[g]["frac_mtopdiv_pos"].get("mean"), depth[g]["frac_mtopdiv_pos"].get("std"), nd=1)
		label = {"early": "Early (Stem/Embeddings)", "intermediate": "Intermediate (Mid Blocks)", "deep": "Deep (Final Blocks/Heads)"}[g]
		return f"{label}  & {db1} & {db1p} & {dl2} & {dl1} & {dl2q1} & {fr} \\\\"

	tex = "\n".join([row("early"), row("intermediate"), row("deep")]) + "\n"
	with open(path, "w", encoding="utf-8") as f:
		f.write(tex)


def _write_arch_tex(path: str, rows: List[dict]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	lines = []
	for r in rows:
		arch = r["architecture"]
		ds = r["dataset"]
		mod = r["modality"]
		pm = str(r.get("primary_metric", "") or "")
		pm_disp = {"accuracy": "Acc", "f1_macro": "F1", "bleu": "BLEU", "loss": "Loss", "loss_assistant_only": "Loss", "ppl": "PPL"}.get(pm, pm)
		dtm_m = r.get("d_task_metric_mean", None)
		dtm_s = r.get("d_task_metric_std", None)
		if dtm_m is None or (isinstance(dtm_m, float) and math.isnan(dtm_m)):
			task = "NA"
		else:
			nd = 3 if pm == "bleu" else 2
			task = f"{_fmt_pm(dtm_m, dtm_s, nd=nd, signed=True)} ({pm_disp})"
		db1p = _fmt_pm(r.get("d_beta1_persistent_mean", None), r.get("d_beta1_persistent_std", None), nd=1, signed=True)
		dmt = _fmt_pm(r.get("d_mtopdiv_mean", None), r.get("d_mtopdiv_std", None), nd=1, signed=True)
		dl2 = _fmt_pm(r.get("d_lambda2_q0_mean", None), r.get("d_lambda2_q0_std", None), nd=2, signed=True)
		dl1 = _fmt_pm(
			r.get("d_lambda1_q1_persistent_mean", None),
			r.get("d_lambda1_q1_persistent_std", None),
			nd=2,
			signed=True,
		)
		s = r.get("simplified_topology", None)
		s_txt = "Yes" if s is True else ("No" if s is False else "NA")
		lines.append(f"{arch} & {ds} ({mod}) & {task} & {db1p} & {dmt} & {dl2} & {dl1} & {s_txt} \\\\")
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")


def _write_arch_paper_subset_tex(path: str, rows: List[dict]) -> None:
	"""
	Write rows in the exact order used in the paper table, filling missing combos with NA.
	"""
	want = [
		("MLP", "MNIST", "CV"),
		("ResNet18", "CIFAR-10", "CV"),
		("ResNet18", "BloodMNIST", "CV"),
		("ResNet18", "ImageNette", "CV"),
		("EfficientNet-B0", "CIFAR-10", "CV"),
		("EfficientNet-B0", "BloodMNIST", "CV"),
		("EfficientNet-B0", "ImageNette", "CV"),
		("ConvNeXt-Tiny", "CIFAR-10", "CV"),
		("ConvNeXt-Tiny", "BloodMNIST", "CV"),
		("ConvNeXt-Tiny", "ImageNette", "CV"),
		("DistilBERT [pretrained]", "SST-2", "NLP"),
		("DistilBERT [pretrained]", "Trec-6", "NLP"),
		("DistilBERT [untrained]", "SST-2", "NLP"),
		("DistilBERT [untrained]", "Trec-6", "NLP"),
		("SmolLM", "SmolTalk (summarize)", "NLP"),
	]
	by_key = {(r["architecture"], r["dataset"], r["modality"]): r for r in rows}
	out_rows = []
	for key in want:
		if key in by_key:
			out_rows.append(by_key[key])
		else:
			out_rows.append(
				{
					"architecture": key[0],
					"dataset": key[1],
					"modality": key[2],
					"d_task_metric": None,
					"primary_metric": "accuracy" if key[2] != "NLP" else "bleu",
					"d_beta1_persistent": None,
					"d_mtopdiv": None,
					"d_lambda2_q0": None,
					"d_lambda1_q1_persistent": None,
					"simplified_topology": None,
				}
			)
	_write_arch_tex(path, out_rows)


def main() -> None:
	ap = argparse.ArgumentParser(description="Aggregate topo/spectral dynamics from runs/* into tables.")
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--out_dir", type=str, default="paper/analysis_tables")
	ap.add_argument("--include", type=str, default="", help="Regex to include run dir names")
	ap.add_argument("--exclude", type=str, default="", help="Regex to exclude run dir names")
	args = ap.parse_args()

	inc_re = re.compile(args.include) if args.include.strip() else None
	exc_re = re.compile(args.exclude) if args.exclude.strip() else None

	runs: List[RunSummary] = []
	for name in sorted(os.listdir(args.runs_dir)):
		if not name.startswith("exp_"):
			continue
		if inc_re and not inc_re.search(name):
			continue
		if exc_re and exc_re.search(name):
			continue
		rd = os.path.join(args.runs_dir, name)
		if not os.path.isdir(rd):
			continue
		rs = _load_run(rd)
		if rs is None:
			continue
		# keep only completed runs with at least some per-layer entries
		if rs.per_layer:
			runs.append(rs)

	if not runs:
		raise SystemExit("No runs found with meta.json + metrics.jsonl + epoch_end repr data.")

	out_dir = str(args.out_dir)
	os.makedirs(out_dir, exist_ok=True)

	# Per-run CSV (debuggable)
	per_run_rows = []
	for r in runs:
		per_run_rows.append(
			{
				"run_dir": r.run_dir,
				"task": r.task,
				"dataset": r.dataset,
				"model": r.model,
				"pretrained": r.pretrained,
				"nlp_objective": r.nlp_objective,
				"primary_metric": r.primary_metric,
				"d_task_metric": r.d_task_metric,
				"n_layers": len(r.per_layer),
			}
		)
	_write_csv(os.path.join(out_dir, "runs_summary.csv"), per_run_rows)

	# Depth dynamics across all runs
	depth = _aggregate_depth_dynamics(runs)
	depth_rows = []
	for g in ("early", "intermediate", "deep"):
		depth_rows.append({"group": g, **depth[g]})
	_write_csv(os.path.join(out_dir, "depth_dynamics.csv"), depth_rows)
	_write_depth_tex(os.path.join(out_dir, "table_depth_dynamics_rows.tex"), depth)
	_write_csv(os.path.join(out_dir, "depth_dynamics_debug.csv"), _aggregate_depth_dynamics_debug(runs))

	# Architecture comparison (deep layers)
	arch_rows = _aggregate_arch_table(runs)
	_write_csv(os.path.join(out_dir, "arch_comparison.csv"), arch_rows)
	_write_arch_tex(os.path.join(out_dir, "table_arch_comparison_rows.tex"), arch_rows)
	_write_arch_paper_subset_tex(os.path.join(out_dir, "table_arch_comparison_rows_paper.tex"), arch_rows)

	print(f"[OK] runs={len(runs)} out_dir={out_dir}")
	print("[OK] wrote:", os.path.join(out_dir, "depth_dynamics.csv"))
	print("[OK] wrote:", os.path.join(out_dir, "arch_comparison.csv"))
	print("[OK] LaTeX row snippets:")
	print(" -", os.path.join(out_dir, "table_depth_dynamics_rows.tex"))
	print(" -", os.path.join(out_dir, "table_arch_comparison_rows.tex"))
	print(" -", os.path.join(out_dir, "table_arch_comparison_rows_paper.tex"))


if __name__ == "__main__":
	main()

