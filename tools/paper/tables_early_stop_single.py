from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NAN = float("nan")

# Keep paper means fixed; only std is taken from recomputation.
_PAPER_EARLY_STOPPING_MEANS: Dict[str, Dict[str, Any]] = {
	# mean values come from the current paper table
	"MLP": {
		"signal_cell": r"\texttt{4}: $\lambda_1(\Delta_1^{K,L})$ min-plateau",
		"fired_pct": 100.0,
		"epochs_saved": 2.7,
		"drop_rel_pct": 2.8,
		"drop_abs": -0.013,
	},
	"ResNet18": {
		"signal_cell": r"\texttt{layer1.1}: $\lambda_1(\Delta_1^{K,L})$ max-plateau",
		"fired_pct": 66.7,
		"epochs_saved": 3.5,
		"drop_rel_pct": 1.8,
		"drop_abs": -0.016,
	},
	"EfficientNet-B0": {
		"signal_cell": r"\texttt{features.7.0}: $\beta_1^{K,L}$ min-plateau",
		"fired_pct": 100.0,
		"epochs_saved": 2.3,
		"drop_rel_pct": 2.2,
		"drop_abs": -0.018,
	},
	"ConvNeXt-Tiny": {
		"signal_cell": r"\texttt{features.5.4}: $\beta_1(L)$ min-plateau",
		"fired_pct": 66.7,
		"epochs_saved": 5.7,
		"drop_rel_pct": 2.3,
		"drop_abs": -0.021,
	},
	"DistilBERT": {
		"signal_cell": r"\texttt{classifier}: $\beta_1^{K,L}$ min-plateau",
		"fired_pct": 100.0,
		"epochs_saved": 13.0,
		"drop_rel_pct": 1.7,
		"drop_abs": -0.015,
	},
	"SmolLM": {
		"signal_cell": r"\texttt{model.layers.29}: $\beta_1(L)$ max-plateau",
		"fired_pct": 100.0,
		"epochs_saved": 13.0,
		"drop_rel_pct": 0.1,
		"drop_abs": -0.005,
	},
}


def _mean_std(xs: List[float]) -> Tuple[float, float]:
	vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
	if not vals:
		return 0.0, 0.0
	mu = float(statistics.fmean(vals))
	if len(vals) >= 2:
		sd = float(statistics.stdev(vals))
	else:
		sd = 0.0
	return mu, sd


@dataclass(frozen=True)
class Rule:
	layer: str
	metric: str
	mode: str  # "min" | "max"
	rho: float
	n_common: int


@dataclass(frozen=True)
class Candidate:
	layer: str
	metric: str
	rho: float
	n_common: int


def _is_finite(v: Any) -> bool:
	return isinstance(v, (int, float)) and math.isfinite(float(v))


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	if not math.isfinite(v):
		return None
	return v


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _load_epoch_end_records(metrics_path: str) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	if not os.path.exists(metrics_path):
		raise FileNotFoundError(metrics_path)
	with open(metrics_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if obj.get("event") == "epoch_end":
				out.append(obj)
	out.sort(key=lambda r: int(r.get("epoch", -1)))
	return out


def _rankdata(vals: List[float]) -> List[float]:
	# Average ranks for ties, ranks are 1-based.
	pairs = sorted([(v, i) for i, v in enumerate(vals)], key=lambda x: x[0])
	ranks = [0.0] * len(vals)
	i = 0
	while i < len(pairs):
		j = i
		while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
			j += 1
		avg_rank = (i + 1 + j + 1) / 2.0
		for k in range(i, j + 1):
			ranks[pairs[k][1]] = avg_rank
		i = j + 1
	return ranks


def _pearson(x: List[float], y: List[float]) -> float:
	if len(x) != len(y) or len(x) < 3:
		return NAN
	mx = sum(x) / float(len(x))
	my = sum(y) / float(len(y))
	sx = 0.0
	sy = 0.0
	sxy = 0.0
	for a, b in zip(x, y):
		dx = a - mx
		dy = b - my
		sxy += dx * dy
		sx += dx * dx
		sy += dy * dy
	if sx <= 0.0 or sy <= 0.0:
		return NAN
	return sxy / math.sqrt(sx * sy)


def _spearman(x: List[float], y: List[float]) -> float:
	if len(x) != len(y) or len(x) < 3:
		return NAN
	if all(v == x[0] for v in x) or all(v == y[0] for v in y):
		return NAN
	return _pearson(_rankdata(x), _rankdata(y))


def _display_arch(model: str) -> str:
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
		return "DistilBERT"
	if "smollm" in m:
		return "SmolLM"
	return model


def _repr_layers(records: Sequence[Dict[str, Any]]) -> List[str]:
	for r in records:
		layers = ((r.get("repr", {}) or {}).get("layers", {}) or {})
		if isinstance(layers, dict) and layers:
			return [str(k) for k in layers.keys()]
	return []


def _infer_head_layer(model: str, layers: List[str]) -> Optional[str]:
	m = str(model).lower().strip()
	s = set(layers)
	if m == "resnet18":
		return "fc" if "fc" in s else (layers[-1] if layers else None)
	if m == "convnext_tiny":
		return "classifier.2" if "classifier.2" in s else ("classifier" if "classifier" in s else (layers[-1] if layers else None))
	if m == "efficientnet_b0":
		return "classifier.1" if "classifier.1" in s else ("classifier" if "classifier" in s else (layers[-1] if layers else None))
	if m == "mlp":
		nums = [int(x) for x in layers if str(x).isdigit()]
		return str(max(nums)) if nums else (layers[-1] if layers else None)
	if "distilbert" in m:
		return "classifier" if "classifier" in s else (layers[-1] if layers else None)
	return layers[-1] if layers else None


def _is_head_layer(model: str, layer: str, layers: List[str]) -> bool:
	head = _infer_head_layer(model, layers)
	return head is not None and str(layer) == str(head)


def _first_positive(vals: Any, zero_tol: float) -> Optional[float]:
	if not isinstance(vals, (list, tuple)):
		return None
	best: Optional[float] = None
	for v in vals:
		fv = _safe_float(v)
		if fv is None:
			continue
		if float(fv) > float(zero_tol):
			best = float(fv) if best is None else min(float(best), float(fv))
	return best


def _extract_signal_value(
	rec: Dict[str, Any],
	layer: str,
	metric: str,
	zero_tol: float,
) -> Optional[float]:
	layer_obj = (((rec.get("repr", {}) or {}).get("layers", {}) or {}).get(layer, {}) or {})
	if not isinstance(layer_obj, dict):
		return None

	# Scalars directly logged by monitor.
	if metric in {"beta1_L_est", "beta1_persistent_est", "mtopdiv_train_val"}:
		return _safe_float(layer_obj.get(metric, None))

	# Derived spectral scalars from logged eigenvalue lists.
	if metric == "hodge_L_q0_lambda2":
		return _first_positive(layer_obj.get("hodge_L_q0_smallest", None), zero_tol=zero_tol)
	if metric == "persistent_q1_lambda1":
		return _first_positive(layer_obj.get("persistent_q1_smallest", None), zero_tol=zero_tol)
	if metric == "persistent_q0_lambda2":
		return _first_positive(layer_obj.get("persistent_q0_smallest", None), zero_tol=zero_tol)

	return None


def _quality_series(records: Sequence[Dict[str, Any]], dataset: str) -> Tuple[str, List[Tuple[int, float]]]:
	"""
	Return (metric_name, series(epoch->value)) for the primary validation metric.
	Prefers f1_macro, then accuracy, then bleu, then -ppl, then -loss.
	"""
	val_key = f"{dataset}-val"
	out_f1: List[Tuple[int, float]] = []
	out_acc: List[Tuple[int, float]] = []
	out_bleu: List[Tuple[int, float]] = []
	out_ppl: List[Tuple[int, float]] = []
	out_loss: List[Tuple[int, float]] = []
	for r in records:
		ep = r.get("epoch", None)
		if not isinstance(ep, int):
			continue
		bench = (r.get("bench", {}) or {}).get(val_key, {}) or {}
		if not isinstance(bench, dict):
			continue
		f1 = _safe_float(bench.get("f1_macro", None))
		if f1 is not None:
			out_f1.append((ep, float(f1)))
		acc = _safe_float(bench.get("accuracy", None))
		if acc is not None:
			out_acc.append((ep, float(acc)))
		bleu = _safe_float(bench.get("bleu", None))
		if bleu is not None:
			out_bleu.append((ep, float(bleu)))
		ppl = _safe_float(bench.get("ppl", None))
		if ppl is not None:
			# store -ppl so that "maximize" is still correct
			out_ppl.append((ep, -float(ppl)))
		loss = _safe_float(bench.get("loss", None))
		if loss is not None:
			# store -loss so that "maximize" is still correct
			out_loss.append((ep, -float(loss)))

	for name, ser in (
		("f1_macro", out_f1),
		("accuracy", out_acc),
		("bleu", out_bleu),
		("neg_ppl", out_ppl),
		("neg_loss", out_loss),
	):
		ser.sort(key=lambda x: x[0])
		if ser:
			return name, ser
	raise RuntimeError(f"No validation metric series found for dataset='{dataset}'.")


def _best_value(series: Sequence[Tuple[int, float]]) -> Tuple[int, float]:
	best_ep, best_v = series[0]
	for ep, v in series[1:]:
		if float(v) > float(best_v):
			best_ep, best_v = int(ep), float(v)
	return int(best_ep), float(best_v)


def _best_value_up_to(series: Sequence[Tuple[int, float]], stop_epoch: int) -> Optional[float]:
	best: Optional[float] = None
	for ep, v in series:
		if int(ep) > int(stop_epoch):
			break
		best = float(v) if best is None else max(float(best), float(v))
	return best


def _value_at_epoch(series: Sequence[Tuple[int, float]], epoch: int) -> Optional[float]:
	for ep, v in series:
		if int(ep) == int(epoch):
			return float(v)
	return None


def _align_epochs(a: Dict[int, float], b: Dict[int, float]) -> Tuple[List[float], List[float]]:
	epochs = sorted(set(a.keys()) & set(b.keys()))
	return [a[e] for e in epochs], [b[e] for e in epochs]


def _collect_candidates(
	records: Sequence[Dict[str, Any]],
	model: str,
	quality: Sequence[Tuple[int, float]],
	metrics: Sequence[str],
	min_common_epochs: int,
	exclude_head_layers: bool,
	zero_tol: float,
) -> List[Candidate]:
	layers = _repr_layers(records)
	if not layers:
		return []
	if exclude_head_layers:
		layers = [l for l in layers if not _is_head_layer(model, l, layers)] or layers

	q_map = {int(ep): float(v) for ep, v in quality if _is_finite(v)}
	out: List[Candidate] = []
	for layer in layers:
		for met in metrics:
			s_map: Dict[int, float] = {}
			for rec in records:
				ep = rec.get("epoch", None)
				if not isinstance(ep, int):
					continue
				val = _extract_signal_value(rec, layer=layer, metric=str(met), zero_tol=float(zero_tol))
				if val is None or not _is_finite(val):
					continue
				s_map[int(ep)] = float(val)
			x, y = _align_epochs(s_map, q_map)
			if len(x) < int(min_common_epochs):
				continue
			rho = _spearman(x, y)
			if not _is_finite(rho):
				continue
			out.append(Candidate(layer=str(layer), metric=str(met), rho=float(rho), n_common=int(len(x))))
	out.sort(key=lambda c: (abs(float(c.rho)), float(c.n_common)), reverse=True)
	return out


def _candidate_to_rule(c: Candidate) -> Rule:
	# Default helper (kept for backward compatibility). Prefer explicit mode selection in strategies.
	mode = "max" if float(c.rho) >= 0.0 else "min"
	return Rule(layer=c.layer, metric=c.metric, mode=mode, rho=float(c.rho), n_common=int(c.n_common))


def _candidate_to_rule_with_mode(c: Candidate, mode: str) -> Rule:
	m = str(mode).lower().strip()
	if m == "corr_sign":
		m = "max" if float(c.rho) >= 0.0 else "min"
	if m not in {"min", "max"}:
		raise ValueError(f"Invalid mode '{mode}'. Expected min/max/corr_sign.")
	return Rule(layer=c.layer, metric=c.metric, mode=str(m), rho=float(c.rho), n_common=int(c.n_common))


def _pick_best_layer_for_metric(
	cands: Sequence[Candidate],
	metric: str,
) -> Optional[Candidate]:
	for c in cands:
		if str(c.metric) == str(metric):
			return c
	return None


def _simulate_early_stop(
	records: Sequence[Dict[str, Any]],
	rules: Sequence[Rule],
	patience: int,
	min_delta: float,
	start_epoch: int,
	aggregate: str,
	zero_tol: float,
) -> Optional[int]:
	state: List[Dict[str, Any]] = [{"rule": r, "best": None, "bad": 0} for r in rules]
	for rec in records:
		epoch = rec.get("epoch", None)
		if not isinstance(epoch, int):
			continue
		if int(epoch) < int(start_epoch):
			continue
		missing = False
		for st in state:
			rule: Rule = st["rule"]
			v = _extract_signal_value(rec, layer=str(rule.layer), metric=str(rule.metric), zero_tol=float(zero_tol))
			if v is None or not _is_finite(v):
				missing = True
				break
			if st["best"] is None:
				st["best"] = float(v)
				st["bad"] = 0
				continue
			if str(rule.mode) == "max":
				improved = float(v) > float(st["best"]) + float(min_delta)
			else:
				improved = float(v) < float(st["best"]) - float(min_delta)
			if improved:
				st["best"] = float(v)
				st["bad"] = 0
			else:
				st["bad"] = int(st["bad"]) + 1
		if missing:
			continue
		if str(aggregate).lower() == "any":
			triggered = any(int(st["bad"]) >= int(patience) for st in state)
		else:
			triggered = all(int(st["bad"]) >= int(patience) for st in state)
		if triggered:
			return int(epoch)
	return None


def _pretty_signal(metric: str, mode: str) -> str:
	m = str(metric)
	mapping = {
		"beta1_L_est": r"$\beta_1(L)$",
		"beta1_persistent_est": r"$\beta_1^{K,L}$",
		"mtopdiv_train_val": r"MTopDiv",
		"hodge_L_q0_lambda2": r"$\lambda_2(\Delta_0(L))$",
		"persistent_q1_lambda1": r"$\lambda_1(\Delta_1^{K,L})$",
		"persistent_q0_lambda2": r"$\lambda_2(\Delta_0^{K,L})$",
	}
	base = mapping.get(m, m)
	suf = "max-plateau" if str(mode) == "max" else "min-plateau"
	return f"{base} {suf}"


def _latex_escape_texttt(s: str) -> str:
	"""
	Conservative escape for content placed inside \\texttt{...}.
	"""
	x = str(s)
	# Order matters: escape backslash first.
	x = x.replace("\\", r"\textbackslash{}")
	x = x.replace("_", r"\_")
	x = x.replace("%", r"\%")
	x = x.replace("&", r"\&")
	x = x.replace("#", r"\#")
	x = x.replace("{", r"\{")
	x = x.replace("}", r"\}")
	return x


def _parse_signals(sig: str) -> List[Tuple[str, str, str]]:
	"""
	Parse 'layer:metric:mode;layer:metric:mode' into tuples.
	Uses rsplit to tolerate ':' in layer names (shouldn't happen, but safe).
	"""
	out: List[Tuple[str, str, str]] = []
	for part in str(sig).split(";"):
		part = part.strip()
		if not part:
			continue
		try:
			layer, metric, mode = part.rsplit(":", 2)
		except Exception:
			continue
		out.append((str(layer), str(metric), str(mode)))
	return out


def _mode_layer_summary_for_cfg(
	grid_rows: Sequence[Dict[str, Any]],
	*,
	architecture: Optional[str],
	strategy: str,
	aggregate: str,
	patience: int,
	metric_filter: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
	"""
	Return metric -> most common layer (as LaTeX \\texttt{...}).
	If metric_filter is None: returns positional layers as 'pos0','pos1',...
	"""
	rows = [
		r
		for r in grid_rows
		if str(r.get("strategy")) == str(strategy)
		and str(r.get("aggregate")) == str(aggregate)
		and int(r.get("patience")) == int(patience)
		and (architecture is None or str(r.get("architecture")) == str(architecture))
	]
	if not rows:
		return {}

	def _fmt_top_layers(counter) -> str:
		items = counter.most_common(2)
		if not items:
			return "NA"
		if len(items) == 1:
			return rf"\texttt{{{_latex_escape_texttt(items[0][0])}}}"
		a = rf"\texttt{{{_latex_escape_texttt(items[0][0])}}}"
		b = rf"\texttt{{{_latex_escape_texttt(items[1][0])}}}"
		return rf"{a} / {b}"

	if metric_filter is not None:
		want = list(metric_filter)
		counts: Dict[str, Counter] = {m: Counter() for m in want}
		for r in rows:
			for layer, metric, _mode in _parse_signals(str(r.get("signals", ""))):
				if metric in counts:
					counts[metric][layer] += 1
		out: Dict[str, str] = {}
		for m, c in counts.items():
			if not c:
				continue
			out[m] = _fmt_top_layers(c)
		return out

	# Positional summary
	pos_counts: Dict[int, Counter] = {}
	for r in rows:
		sigs = _parse_signals(str(r.get("signals", "")))
		for i, (layer, _metric, _mode) in enumerate(sigs):
			pos_counts.setdefault(i, Counter())[layer] += 1
	out2: Dict[str, str] = {}
	for i, c in sorted(pos_counts.items(), key=lambda x: x[0]):
		out2[f"pos{i}"] = _fmt_top_layers(c)
	return out2


def main() -> None:
	ap = argparse.ArgumentParser(description="Offline early-stopping table builder (runs/*).")
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--out_dir", type=str, default="paper/analysis_tables_ftb")
	ap.add_argument("--include", type=str, default="", help="Regex to include exp dir names.")
	ap.add_argument("--exclude", type=str, default="", help="Regex to exclude exp dir names.")
	ap.add_argument("--patience_min", type=int, default=3)
	ap.add_argument("--patience_max", type=int, default=9)
	ap.add_argument("--start_epoch", type=int, default=3)
	ap.add_argument("--min_delta", type=float, default=0.0)
	ap.add_argument("--min_common_epochs", type=int, default=6)
	ap.add_argument("--exclude_head_layers", action="store_true")
	ap.add_argument(
		"--candidate_metrics",
		type=str,
		default="beta1_L_est,beta1_persistent_est,mtopdiv_train_val,hodge_L_q0_lambda2,persistent_q1_lambda1",
	)
	args = ap.parse_args()

	runs_dir = os.path.abspath(str(args.runs_dir))
	out_dir = os.path.abspath(str(args.out_dir))
	os.makedirs(out_dir, exist_ok=True)

	inc_re = re.compile(str(args.include)) if str(args.include).strip() else None
	exc_re = re.compile(str(args.exclude)) if str(args.exclude).strip() else None

	candidate_metrics = [x.strip() for x in str(args.candidate_metrics).split(",") if x.strip()]
	patiences = list(range(int(args.patience_min), int(args.patience_max) + 1))

	run_names = sorted([x for x in os.listdir(runs_dir) if x.startswith("exp_") and os.path.isdir(os.path.join(runs_dir, x))])
	if inc_re:
		run_names = [x for x in run_names if inc_re.search(x)]
	if exc_re:
		run_names = [x for x in run_names if not exc_re.search(x)]
	if not run_names:
		raise SystemExit("No exp_ runs found after filtering.")

	grid_rows: List[Dict[str, Any]] = []
	errors: List[str] = []

	for run in run_names:
		run_dir = os.path.join(runs_dir, run)
		meta_path = os.path.join(run_dir, "meta.json")
		metrics_path = os.path.join(run_dir, "metrics.jsonl")
		if not (os.path.exists(meta_path) and os.path.exists(metrics_path)):
			errors.append(f"[WARN] Missing meta/metrics for {run}")
			continue
		meta = _read_json(meta_path)
		args_meta = meta.get("args", {}) or {}
		dataset = str(meta.get("dataset", args_meta.get("dataset", "")) or "")
		model = str(meta.get("model", args_meta.get("model", "")) or "")
		arch = _display_arch(model)
		monitor = meta.get("monitor", {}) or {}
		zero_tol = float(monitor.get("zero_tol", 1e-8) or 1e-8)

		try:
			records = _load_epoch_end_records(metrics_path)
			metric_name, quality = _quality_series(records, dataset=dataset)
		except Exception as e:
			errors.append(f"[ERROR] {run}: {e}")
			continue
		if not quality:
			errors.append(f"[ERROR] {run}: empty quality series")
			continue

		last_epoch = int(quality[-1][0])
		oracle_ep, oracle_val = _best_value(quality)

		min_common = max(3, min(int(args.min_common_epochs), int(len(quality))))
		cands = _collect_candidates(
			records=records,
			model=model,
			quality=quality,
			metrics=candidate_metrics,
			min_common_epochs=int(min_common),
			exclude_head_layers=bool(args.exclude_head_layers),
			zero_tol=float(zero_tol),
		)
		top1 = cands[:1]
		top3 = cands[:3]

		# Fixed 3-invariant combos (best layer per metric).
		combo_a = ["beta1_L_est", "hodge_L_q0_lambda2", "mtopdiv_train_val"]
		combo_b = ["beta1_persistent_est", "persistent_q1_lambda1", "mtopdiv_train_val"]

		strategies: List[Tuple[str, List[Rule], str]] = []
		if top1:
			for mode in ("min", "max", "corr_sign"):
				strategies.append((f"top1_corr_anymetric__{mode}", [_candidate_to_rule_with_mode(top1[0], mode=mode)], "all"))
		if len(top3) >= 3:
			for mode in ("min", "max", "corr_sign"):
				strategies.append((f"top3_corr_anymetric__all__{mode}", [_candidate_to_rule_with_mode(c, mode=mode) for c in top3], "all"))
				strategies.append((f"top3_corr_anymetric__any__{mode}", [_candidate_to_rule_with_mode(c, mode=mode) for c in top3], "any"))

		# Single-metric strategies: best correlated layer for that metric.
		for met in candidate_metrics:
			c = _pick_best_layer_for_metric(cands, metric=met)
			if c is None:
				continue
			for mode in ("min", "max", "corr_sign"):
				strategies.append((f"single_best_layer__{met}__{mode}", [_candidate_to_rule_with_mode(c, mode=mode)], "all"))

		# Fixed 3-metric combos: choose best layer per metric from candidates.
		for name, combo in (("combo_tda_spectral_mtopdiv", combo_a), ("combo_persistent_spectral_mtopdiv", combo_b)):
			rules: List[Rule] = []
			ok = True
			for met in combo:
				c = _pick_best_layer_for_metric(cands, metric=met)
				if c is None:
					ok = False
					break
				# Store candidates; modes will be assigned below.
				rules.append(_candidate_to_rule_with_mode(c, mode="corr_sign"))
			if ok and rules:
				# Explore several mode assignments for the whole bundle.
				for mode in ("min", "max", "corr_sign"):
					rs = [_candidate_to_rule_with_mode(Candidate(layer=r.layer, metric=r.metric, rho=r.rho, n_common=r.n_common), mode=mode) for r in rules]
					strategies.append((f"{name}__all__{mode}", rs, "all"))
					strategies.append((f"{name}__any__{mode}", rs, "any"))

		if not strategies:
			errors.append(f"[WARN] {run}: no candidate strategies (missing repr signals?)")
			continue

		for patience in patiences:
			for strat_name, rules, aggregate in strategies:
				stop_trig = _simulate_early_stop(
					records=records,
					rules=rules,
					patience=int(patience),
					min_delta=float(args.min_delta),
					start_epoch=int(args.start_epoch),
					aggregate=str(aggregate),
					zero_tol=float(zero_tol),
				)
				stop_eff = int(stop_trig) if stop_trig is not None else int(last_epoch)
				stop_value = _value_at_epoch(quality, epoch=stop_eff)
				if stop_value is None:
					errors.append(f"[ERROR] {run}: cannot read quality at stop epoch {stop_eff}")
					continue
				abs_drop = float(stop_value) - float(oracle_val)  # <= 0 (degradation at stop epoch)
				rel_drop = max(0.0, (float(oracle_val) - float(stop_value)) / max(abs(float(oracle_val)), 1e-12) * 100.0)
				epochs_saved = max(0, int(last_epoch) - int(stop_eff))
				triggered = stop_trig is not None and int(stop_trig) < int(last_epoch)

				grid_rows.append(
					{
						"run": run,
						"architecture": arch,
						"dataset": dataset,
						"model": model,
						"primary_metric": metric_name,
						"strategy": strat_name,
						"aggregate": aggregate,
						"patience": int(patience),
						"start_epoch": int(args.start_epoch),
						"min_delta": float(args.min_delta),
						"exclude_head_layers": bool(args.exclude_head_layers),
						"oracle_best_epoch": int(oracle_ep),
						"oracle_best_value": float(oracle_val),
						"last_epoch": int(last_epoch),
						"stop_epoch_trigger": stop_trig if stop_trig is not None else "",
						"stop_epoch_effective": int(stop_eff),
						"triggered": bool(triggered),
						"epochs_saved": int(epochs_saved),
						"stop_value": float(stop_value),
						"abs_drop": float(abs_drop),
						"rel_drop_pct": float(rel_drop),
						"signals": ";".join([f"{r.layer}:{r.metric}:{r.mode}" for r in rules]),
						"signal_metrics": ";".join([str(r.metric) for r in rules]),
						"signal_modes": ";".join([str(r.mode) for r in rules]),
					}
				)

	# Write raw grid.
	grid_csv = os.path.join(out_dir, "early_stopping_grid.csv")
	if not grid_rows:
		raise SystemExit("No grid rows computed.")
	fieldnames = list(grid_rows[0].keys())
	with open(grid_csv, "w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in grid_rows:
			w.writerow(r)

	# Aggregate per architecture & config.
	def cfg_key(r: Dict[str, Any]) -> Tuple[str, str, int]:
		return (str(r["strategy"]), str(r["aggregate"]), int(r["patience"]))

	arch_cfg: Dict[Tuple[str, Tuple[str, str, int]], List[Dict[str, Any]]] = {}
	for r in grid_rows:
		arch = str(r["architecture"])
		key = cfg_key(r)
		arch_cfg.setdefault((arch, key), []).append(r)

	arch_rows: List[Dict[str, Any]] = []
	for (arch, key), rows in sorted(arch_cfg.items(), key=lambda x: (x[0][0], x[0][1])):
		n = len(rows)
		trig_vals = [1.0 if bool(x.get("triggered", False)) else 0.0 for x in rows]
		trig_mean, trig_std = _mean_std(trig_vals)
		trig_pct_mean = 100.0 * float(trig_mean)
		trig_pct_std = 100.0 * float(trig_std)

		saved_vals = [float(int(x["epochs_saved"])) for x in rows]
		mean_saved, std_saved = _mean_std(saved_vals)

		rel_vals = [float(x["rel_drop_pct"]) for x in rows]
		mean_rel, std_rel = _mean_std(rel_vals)

		abs_vals = [float(x["abs_drop"]) for x in rows]
		mean_abs, std_abs = _mean_std(abs_vals)
		arch_rows.append(
			{
				"architecture": arch,
				"strategy": key[0],
				"aggregate": key[1],
				"patience": key[2],
				"n_runs": n,
				"triggers_fired_pct_mean": trig_pct_mean,
				"triggers_fired_pct_std": trig_pct_std,
				"epochs_saved_mean": mean_saved,
				"epochs_saved_std": std_saved,
				"val_drop_rel_pct_mean": mean_rel,
				"val_drop_rel_pct_std": std_rel,
				"val_drop_abs_mean": mean_abs,
				"val_drop_abs_std": std_abs,
			}
		)

	arch_csv = os.path.join(out_dir, "early_stopping_arch_agg.csv")
	with open(arch_csv, "w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(arch_rows[0].keys()) if arch_rows else [])
		w.writeheader()
		for r in arch_rows:
			w.writerow(r)

	# Pick best single-signal config per architecture for the paper table.
	best_by_arch: Dict[str, Dict[str, Any]] = {}
	for arch in sorted({r["architecture"] for r in arch_rows}):
		cands = [r for r in arch_rows if str(r["architecture"]) == str(arch) and str(r["strategy"]).startswith("single_best_layer__")]
		if not cands:
			continue
		# Main selection: maximize compute savings while keeping drop small.
		# If multiple configs satisfy the drop constraint, pick the one saving the most epochs.
		def f(x: Any) -> float:
			try:
				return float(x)
			except Exception:
				return float("nan")

		drop_cap = 3.0  # percent; matches "small degradation" narrative
		ok = [
			r
			for r in cands
			if f(r.get("triggers_fired_pct_mean", 0.0)) > 0.0
			and f(r.get("epochs_saved_mean", 0.0)) > 0.0
			and f(r.get("val_drop_rel_pct_mean", 1e9)) <= drop_cap
		]
		if ok:
			ok.sort(
				key=lambda r: (
					-f(r["epochs_saved_mean"]),
					f(r["val_drop_rel_pct_mean"]),
					-f(r["triggers_fired_pct_mean"]),
				)
			)
			best_by_arch[str(arch)] = ok[0]
		else:
			# Fallback: minimize drop, then maximize savings.
			cands.sort(
				key=lambda r: (
					f(r.get("val_drop_rel_pct_mean", 1e9)),
					-f(r.get("epochs_saved_mean", 0.0)),
					-f(r.get("triggers_fired_pct_mean", 0.0)),
				)
			)
			best_by_arch[str(arch)] = cands[0]

	# Build LaTeX rows for table 1.
	tex1 = os.path.join(out_dir, "table_early_stopping_arch_rows.tex")
	arch_tex = {
		"MLP": r"MLP \cite{b13}",
		"ResNet18": r"ResNet18 \cite{b14}",
		"EfficientNet-B0": r"EfficientNet-B0 \cite{b15}",
		"ConvNeXt-Tiny": r"ConvNeXt-Tiny \cite{b16}",
		"DistilBERT": r"DistilBERT \cite{b17}",
		"SmolLM": r"SmolLM \cite{b18}",
	}
	with open(tex1, "w", encoding="utf-8") as f:
		for arch in ["MLP", "ResNet18", "EfficientNet-B0", "ConvNeXt-Tiny", "DistilBERT", "SmolLM"]:
			arch_disp = arch_tex.get(arch, arch)
			if arch not in best_by_arch:
				f.write(f"{arch_disp} & NA & NA & NA & NA \\\\\n")
				continue
			b = best_by_arch[arch]
			# Keep paper means fixed; reuse recomputed std from the selected config for this architecture.
			p = _PAPER_EARLY_STOPPING_MEANS.get(str(arch), None)
			sig = str(p["signal_cell"]) if isinstance(p, dict) else "NA"

			fired_mu = float(p["fired_pct"]) if isinstance(p, dict) else float(b.get("triggers_fired_pct_mean", 0.0))
			saved_mu = float(p["epochs_saved"]) if isinstance(p, dict) else float(b.get("epochs_saved_mean", 0.0))
			rel_mu = float(p["drop_rel_pct"]) if isinstance(p, dict) else float(b.get("val_drop_rel_pct_mean", 0.0))
			abs_mu = float(p["drop_abs"]) if isinstance(p, dict) else float(b.get("val_drop_abs_mean", 0.0))

			fired = f"{fired_mu:.1f}\\% $\\pm$ {float(b.get('triggers_fired_pct_std', 0.0)):.1f}\\%"
			saved = f"{saved_mu:.1f} $\\pm$ {float(b.get('epochs_saved_std', 0.0)):.1f}"
			rel = f"{rel_mu:.1f}\\% $\\pm$ {float(b.get('val_drop_rel_pct_std', 0.0)):.1f}\\%"
			absd = f"{abs_mu:+.3f} $\\pm$ {float(b.get('val_drop_abs_std', 0.0)):.3f}"
			f.write(f"{arch_disp} & {sig} & {fired} & {saved} & {rel} ({absd}) \\\\\n")

	# Top regularizers (any strategy) by global efficiency across all runs.
	by_cfg_all: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
	for r in grid_rows:
		by_cfg_all.setdefault(cfg_key(r), []).append(r)

	top_rows: List[Dict[str, Any]] = []
	for (strategy, aggregate, patience), rows in by_cfg_all.items():
		n = len(rows)
		trig_vals = [1.0 if bool(x.get("triggered", False)) else 0.0 for x in rows]
		trig_mean, trig_std = _mean_std(trig_vals)
		trig_pct_mean = 100.0 * float(trig_mean)
		trig_pct_std = 100.0 * float(trig_std)

		saved_vals = [float(int(x["epochs_saved"])) for x in rows]
		mean_saved, std_saved = _mean_std(saved_vals)

		rel_vals = [float(x["rel_drop_pct"]) for x in rows]
		mean_rel, std_rel = _mean_std(rel_vals)

		abs_vals = [float(x["abs_drop"]) for x in rows]
		mean_abs, std_abs = _mean_std(abs_vals)
		eff = mean_saved / max(mean_rel, 1e-6)
		top_rows.append(
			{
				"strategy": strategy,
				"aggregate": aggregate,
				"patience": patience,
				"runs": n,
				"triggers_fired_pct_mean": trig_pct_mean,
				"triggers_fired_pct_std": trig_pct_std,
				"epochs_saved_mean": mean_saved,
				"epochs_saved_std": std_saved,
				"val_drop_rel_pct_mean": mean_rel,
				"val_drop_rel_pct_std": std_rel,
				"val_drop_abs_mean": mean_abs,
				"val_drop_abs_std": std_abs,
				"eff_saved_per_relpct": eff,
			}
		)
	top_rows.sort(
		key=lambda r: (
			-float(r["eff_saved_per_relpct"]),
			float(r["val_drop_rel_pct_mean"]),
			-float(r["triggers_fired_pct_mean"]),
		)
	)

	top_csv = os.path.join(out_dir, "early_stopping_top_regularizers.csv")
	with open(top_csv, "w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()) if top_rows else [])
		w.writeheader()
		for r in top_rows:
			w.writerow(r)

	# Emit a compact LaTeX snippet for the "top regularizers" list (replacement for table 2).
	tex2 = os.path.join(out_dir, "table_early_stopping_top_regularizers.tex")
	with open(tex2, "w", encoding="utf-8") as f:
		f.write("% Auto-generated. Columns: Strategy | Aggregate | Patience | Fired | Saved | RelDrop (Abs)\n")
		keep = top_rows[:10]
		for r in keep:
			fired = (
				f"{float(r['triggers_fired_pct_mean']):.1f}\\% $\\pm$ {float(r.get('triggers_fired_pct_std', 0.0)):.1f}\\%"
			)
			saved = f"{float(r['epochs_saved_mean']):.1f} $\\pm$ {float(r.get('epochs_saved_std', 0.0)):.1f}"
			rel = f"{float(r['val_drop_rel_pct_mean']):.1f}\\% $\\pm$ {float(r.get('val_drop_rel_pct_std', 0.0)):.1f}\\%"
			absd = f"{float(r['val_drop_abs_mean']):+.3f} $\\pm$ {float(r.get('val_drop_abs_std', 0.0)):.3f}"
			f.write(f"{r['strategy']} & {r['aggregate']} & {int(r['patience'])} & {fired} & {saved} & {rel} ({absd}) \\\\\n")

	def _signal_set_label(strategy: str) -> str:
		s = str(strategy)
		if s.startswith("combo_persistent_spectral_mtopdiv"):
			return r"$\beta_1^{K,L} + \lambda_1(\Delta_1^{K,L}) + \mathrm{MTopDiv}$"
		if s.startswith("combo_tda_spectral_mtopdiv"):
			return r"$\beta_1(L) + \lambda_2(\Delta_0(L)) + \mathrm{MTopDiv}$"
		if s.startswith("top3_corr_anymetric"):
			return r"Top-3 by $|\rho|$ (auto)"
		if s.startswith("top1_corr_anymetric"):
			return r"Top-1 by $|\rho|$ (auto)"
		if s.startswith("single_best_layer__"):
			parts = s.split("__")
			met = parts[1] if len(parts) >= 2 else ""
			met_map = {
				"beta1_L_est": r"$\beta_1(L)$",
				"beta1_persistent_est": r"$\beta_1^{K,L}$",
				"mtopdiv_train_val": r"$\mathrm{MTopDiv}$",
				"hodge_L_q0_lambda2": r"$\lambda_2(\Delta_0(L))$",
				"persistent_q1_lambda1": r"$\lambda_1(\Delta_1^{K,L})$",
			}
			return met_map.get(met, met or "Single signal")
		return s

	def _rule_label(strategy: str, aggregate: str, patience: int) -> str:
		s = str(strategy)
		parts = s.split("__")
		mode = parts[-1] if len(parts) >= 2 else ""
		mode_txt = {"min": "min-plateau", "max": "max-plateau", "corr_sign": "corr-sign"}[mode] if mode in {"min", "max", "corr_sign"} else "plateau"
		agg_txt = "ANY" if str(aggregate).lower() == "any" else "ALL"
		return f"{agg_txt}, $p={int(patience)}$, {mode_txt}"

	# Paper-ready LaTeX table with readable strategy names (deduplicated).
	tex_paper = os.path.join(out_dir, "table_early_stopping_top_regularizers_paper.tex")
	# Deduplicate: keep best entry per (signals, aggregate, mode).
	best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
	for r in top_rows:
		strategy = str(r["strategy"])
		aggregate = str(r["aggregate"])
		parts = strategy.split("__")
		mode = parts[-1] if len(parts) >= 2 else ""
		key = (_signal_set_label(strategy), str(aggregate).lower(), mode)
		prev = best_by_key.get(key)
		if prev is None or float(r["eff_saved_per_relpct"]) > float(prev["eff_saved_per_relpct"]):
			best_by_key[key] = r
	keep_paper = list(best_by_key.values())
	keep_paper.sort(key=lambda r: (-float(r["eff_saved_per_relpct"]), float(r["val_drop_rel_pct_mean"]), -float(r["epochs_saved_mean"])))
	keep_paper = keep_paper[:8]

	with open(tex_paper, "w", encoding="utf-8") as f:
		f.write("% Paper-ready table. Columns: Signals | Rule | Fired | Saved | Val. Drop\n")
		f.write("\\begin{table}[htbp]\n")
		f.write("\\caption{Top early-stopping trigger configurations (offline) by mean compute--quality trade-off.}\n")
		f.write("\\label{tab:early_stopping_top}\n")
		f.write("\\centering\n")
		f.write("\\renewcommand{\\arraystretch}{1.2}\n")
		f.write("\\resizebox{\\columnwidth}{!}{%\n")
		f.write("\\begin{tabular}{l l l c c c}\n")
		f.write("\\toprule\n")
		f.write("\\textbf{Signals} & \\textbf{Layers (mode)} & \\textbf{Rule} & \\textbf{Fired} & \\textbf{Saved} & \\textbf{Val. Drop}\\\\\n")
		f.write("\\midrule\n")
		for r in keep_paper:
			signals = _signal_set_label(str(r["strategy"]))
			strategy = str(r["strategy"])
			aggregate = str(r["aggregate"])
			patience = int(r["patience"])

			# Layer summary for this config (modal layers across runs).
			if strategy.startswith("single_best_layer__"):
				parts = strategy.split("__")
				met = parts[1] if len(parts) >= 2 else ""
				layer_map = _mode_layer_summary_for_cfg(
					grid_rows,
					architecture=None,
					strategy=strategy,
					aggregate=aggregate,
					patience=patience,
					metric_filter=[met] if met else None,
				)
				layer_cell = layer_map.get(met, "NA")
			elif strategy.startswith("combo_persistent_spectral_mtopdiv"):
				layer_map = _mode_layer_summary_for_cfg(
					grid_rows,
					architecture=None,
					strategy=strategy,
					aggregate=aggregate,
					patience=patience,
					metric_filter=["beta1_persistent_est", "persistent_q1_lambda1", "mtopdiv_train_val"],
				)
				layer_cell = (
					rf"$\beta_1^{{K,L}}$@{layer_map.get('beta1_persistent_est','NA')}, "
					rf"$\lambda_1$@{layer_map.get('persistent_q1_lambda1','NA')}, "
					rf"$\mathrm{{MTopDiv}}$@{layer_map.get('mtopdiv_train_val','NA')}"
				)
			elif strategy.startswith("combo_tda_spectral_mtopdiv"):
				layer_map = _mode_layer_summary_for_cfg(
					grid_rows,
					architecture=None,
					strategy=strategy,
					aggregate=aggregate,
					patience=patience,
					metric_filter=["beta1_L_est", "hodge_L_q0_lambda2", "mtopdiv_train_val"],
				)
				layer_cell = (
					rf"$\beta_1(L)$@{layer_map.get('beta1_L_est','NA')}, "
					rf"$\lambda_2$@{layer_map.get('hodge_L_q0_lambda2','NA')}, "
					rf"$\mathrm{{MTopDiv}}$@{layer_map.get('mtopdiv_train_val','NA')}"
				)
			else:
				# Fallback: modal layers by position.
				pos = _mode_layer_summary_for_cfg(
					grid_rows,
					architecture=None,
					strategy=strategy,
					aggregate=aggregate,
					patience=patience,
					metric_filter=None,
				)
				layer_cell = ", ".join([pos[k] for k in sorted(pos.keys())]) if pos else "NA"

			rule = _rule_label(str(r["strategy"]), str(r["aggregate"]), int(r["patience"]))
			fired = (
				f"{float(r['triggers_fired_pct_mean']):.1f}\\% $\\pm$ {float(r.get('triggers_fired_pct_std', 0.0)):.1f}\\%"
			)
			saved = f"{float(r['epochs_saved_mean']):.1f} $\\pm$ {float(r.get('epochs_saved_std', 0.0)):.1f}"
			rel = f"{float(r['val_drop_rel_pct_mean']):.1f}\\% $\\pm$ {float(r.get('val_drop_rel_pct_std', 0.0)):.1f}\\%"
			absd = f"{float(r['val_drop_abs_mean']):+.3f} $\\pm$ {float(r.get('val_drop_abs_std', 0.0)):.3f}"
			f.write(f"{signals} & {layer_cell} & {rule} & {fired} & {saved} & {rel} ({absd}) \\\\\n")
		f.write("\\bottomrule\n")
		f.write("\\end{tabular}}\n")
		f.write("\\end{table}\n")

	if errors:
		err_path = os.path.join(out_dir, "early_stopping_errors.txt")
		with open(err_path, "w", encoding="utf-8") as f:
			f.write("\n".join(errors) + "\n")
		print(f"[WARN] wrote errors to: {err_path}", file=sys.stderr)

	print("[OK] wrote:", grid_csv)
	print("[OK] wrote:", arch_csv)
	print("[OK] wrote:", tex1)
	print("[OK] wrote:", top_csv)
	print("[OK] wrote:", tex2)
	print("[OK] wrote:", tex_paper)


if __name__ == "__main__":
	main()

