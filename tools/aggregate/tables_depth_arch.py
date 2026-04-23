from __future__ import annotations

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
	for k in ("ppl",):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("f1_macro", "accuracy"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	for k in ("loss_assistant_only", "loss", "ppl"):
		if k in bench and isinstance(bench.get(k), (int, float)):
			return k, float(bench[k])
	return "unknown", float("nan")


def _is_minimization_metric(metric_name: str) -> bool:
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


def _is_head_layer(model: str, layer: str) -> bool:
	"""
	Classification / projection head (not backbone depth). Used to split pooled layers from
	backbone early/intermediate/deep tertiles in depth-dynamics aggregation.
	"""
	m = str(model).lower().strip()
	n = str(layer)
	if "efficientnet" in m:
		if n == "classifier" or n.startswith("classifier."):
			return True
	if "convnext" in m:
		if n == "head" or n == "classifier" or n.startswith("classifier.") or "classifier" in n:
			return True
	if "resnet" in m:
		if n == "fc":
			return True
	if any(x in m for x in ("distilbert", "smollm", "llama")):
		if "pre_classifier" in n:
			return True
		if n in ("classifier", "score") or (
			n.endswith("classifier") and "transformer.layer" not in n
		):
			return True
		if "lm_head" in n:
			return True
	return False


def _backbone_depth_rank(model: str, layer: str, layer_names: List[str]) -> Optional[float]:
	m = str(model).lower().strip()
	name = str(layer)

	if _is_head_layer(model, layer):
		return None

	if "resnet" in m:
		if name in ("conv1", "bn1", "relu", "maxpool"):
			return 0.0
		mt = re.search(r"layer(\d+)\.(\d+)", name)
		if mt:
			return float(int(mt.group(1)) * 10 + int(mt.group(2)))
		mt2 = re.search(r"layer(\d+)$", name)
		if mt2:
			return float(int(mt2.group(1)) * 10)
		if name == "avgpool":
			mx_blk = 0.0
			for ln in layer_names:
				if _is_head_layer(model, ln):
					continue
				ls = str(ln)
				mtb = re.search(r"layer(\d+)\.(\d+)", ls)
				if mtb:
					mx_blk = max(mx_blk, float(int(mtb.group(1)) * 10 + int(mtb.group(2))))
				mtw = re.search(r"layer(\d+)$", ls)
				if mtw:
					mx_blk = max(mx_blk, float(int(mtw.group(1)) * 10))
			return mx_blk + 0.5 if mx_blk > 0.0 else 0.5
		return None

	if "efficientnet" in m or "convnext" in m:
		max_stage = 0
		for ln in layer_names:
			if _is_head_layer(model, ln):
				continue
			mt0 = re.search(r"features\.(\d+)", str(ln))
			if mt0:
				max_stage = max(max_stage, int(mt0.group(1)))
		mt = re.search(r"features\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			mt2 = re.search(r"features\.(\d+)\.(\d+)", name)
			if mt2:
				return float(int(mt2.group(1)) * 100 + int(mt2.group(2)))
			return float(i * 100)
		if name == "avgpool":
			return float(max_stage * 100 + 50)
		return None

	if any(x in m for x in ("distilbert", "smollm", "llama")):
		if "embeddings" in name or "embed_tokens" in name or name.endswith("wte"):
			return 0.0
		mt = re.search(r"(?:transformer\.layer|model\.layers|layers)\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			if ".ffn" in name:
				return float(i) + 0.25
			return float(i) + 0.1
		if name.endswith("norm") or ".norm" in name or "ln_f" in name:
			return 1000.0
		return None

	mt = re.search(r"(?:^|\.)((\d+))(?:$|\.)", name)
	if mt:
		return float(int(mt.group(1)))
	try:
		back = [ln for ln in layer_names if not _is_head_layer(model, ln)]
		return float(back.index(layer))
	except Exception:
		return None


def _depth_rank_for_layer(model: str, layer: str, layer_names: List[str]) -> Optional[float]:
	m = str(model).lower().strip()
	name = str(layer)

	if "distilbert" in m or "smollm" in m or "llama" in m:
		if "embeddings" in name or "embed_tokens" in name or name.endswith("wte"):
			return 0.0
		mt = re.search(r"(?:transformer\.layer|model\.layers|layers)\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			if ".ffn" in name:
				return float(i) + 0.25
			return float(i) + 0.1
		if "pre_classifier" in name:
			return 1000.0
		if name.endswith("classifier") or name.endswith("score") or "lm_head" in name:
			return 1100.0
		if name.endswith("norm") or ".norm" in name or "ln_f" in name:
			return 1050.0

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

	if "efficientnet" in m:
		mt = re.search(r"features\.(\d+)", name)
		if mt:
			i = int(mt.group(1))
			mt2 = re.search(r"features\.(\d+)\.(\d+)", name)
			if mt2:
				j = int(mt2.group(2))
				return float(i * 10 + j)
			return float(i * 10)
		if name in ("avgpool", "classifier") or name.startswith("classifier"):
			return 1100.0

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

	mt = re.search(r"(?:^|\.)((\d+))(?:$|\.)", name)
	if mt:
		return float(int(mt.group(1)))

	try:
		return float(layer_names.index(layer))
	except Exception:
		return None


def _depth_group(model: str, layer: str, layer_names: List[str]) -> str:
	if _is_head_layer(model, layer):
		return "head"

	r = _backbone_depth_rank(model, layer, layer_names)
	back_ranks: List[float] = []
	for ln in layer_names:
		if _is_head_layer(model, ln):
			continue
		br = _backbone_depth_rank(model, ln, layer_names)
		if br is not None and math.isfinite(br):
			back_ranks.append(float(br))
	if not back_ranks:
		return "intermediate"
	r_min = float(min(back_ranks))
	r_max = float(max(back_ranks))
	if r is None or not math.isfinite(r):
		try:
			back_names = [ln for ln in layer_names if not _is_head_layer(model, ln)]
			pos = back_names.index(layer) / max(1.0, float(len(back_names) - 1)) if layer in back_names else 0.5
		except Exception:
			pos = 0.5
	elif r_max <= r_min:
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

	_best_epoch, best_repr_layers, best_bench = epochs[-1]
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
			_best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench
			continue
		if _is_minimization_metric(primary_metric):
			if fv < float(best_val):
				best_val = fv
				_best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench
		else:
			if fv > float(best_val):
				best_val = fv
				_best_epoch, best_repr_layers, best_bench = ep, repr_layers, bench

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
	groups = ("early", "intermediate", "deep", "head")
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


def _aggregate_arch_table(runs: List[RunSummary]) -> List[dict]:
	buckets: Dict[Tuple[str, str, str], List[RunSummary]] = {}
	for r in runs:
		arch = _display_arch(r.model, r.pretrained if r.task == "nlp" and "distilbert" in r.model.lower() else None)
		ds = _display_dataset(r.dataset)
		mod = "NLP" if str(r.task).lower().strip() == "nlp" else "CV"
		if "distilbert" in r.model.lower() and r.pretrained is not None:
			arch = _display_arch(r.model, r.pretrained)
		key = (arch, ds, mod)
		buckets.setdefault(key, []).append(r)

	rows: List[dict] = []
	for (arch, ds, mod), rs in sorted(buckets.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
		d_task = []
		db1p = []
		dmt = []
		dl2 = []
		dl1q1 = []
		for r in rs:
			if r.d_task_metric is not None:
				d_task.append(float(r.d_task_metric))
			deep_layers = [ld for ld in r.per_layer if ld.group == "deep"]
			if not deep_layers:
				continue
			v = _median([float(x.d_beta1_pers) for x in deep_layers if x.d_beta1_pers is not None])
			if v is not None:
				db1p.append(v)
			v = _median([float(x.d_mtopdiv) for x in deep_layers if x.d_mtopdiv is not None])
			if v is not None:
				dmt.append(v)
			v = _median([float(x.d_lambda2_q0) for x in deep_layers if x.d_lambda2_q0 is not None])
			if v is not None:
				dl2.append(v)
			v = _median([float(x.d_l1_q1_pers) for x in deep_layers if x.d_l1_q1_pers is not None])
			if v is not None:
				dl1q1.append(v)

		m_d_task, s_d_task, n_task = _mean_std(d_task)
		m_db1p, s_db1p, n_db1p = _mean_std(db1p)
		m_dmt, s_dmt, n_dmt = _mean_std(dmt)
		m_dl2, s_dl2, n_dl2 = _mean_std(dl2)
		m_dl1q1, s_dl1q1, n_dl1q1 = _mean_std(dl1q1)

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
