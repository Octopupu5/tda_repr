from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypedDict


class SelectionRowDict(TypedDict, total=False):
	method_key: str
	method_label: str
	layer: Optional[str]
	R: Optional[float]
	gap_vs_best: Optional[float]
	descriptor_repr_key: Optional[str]
	descriptor_tex: Optional[str]
	abs_rho_s: Optional[float]

def infer_default_layer(model_name: str) -> Optional[str]:
	default_layer_by_model: Mapping[str, str] = {
		"mlp": "1",
		"resnet18": "avgpool",
		"efficientnet_b0": "classifier.0",
		"convnext_tiny": "avgpool",
		"convnext": "avgpool",
		"distilbert": "pre_classifier",
		"smollm2-135m": "model.layers.29",
	}
	m = str(model_name or "").strip().lower()
	for k, v in default_layer_by_model.items():
		if k in m:
			return str(v)
	return None


def _metric_group_from_repr_key(repr_key: str) -> str:
	k = str(repr_key)
	if ".mtopdiv" in k:
		return "mtopdiv"
	if any(x in k for x in (".hodge_L_q", ".persistent_q")):
		return "spectral"
	if any(x in k for x in (".beta", ".gudhi_", ".graph_")):
		return "topo"
	return "other"


def _layer_metric_from_repr_key(repr_key: str) -> Tuple[Optional[str], Optional[str]]:
	p = str(repr_key)
	if not p.startswith("repr.layers."):
		return None, None
	rest = p[len("repr.layers.") :]
	if "." not in rest:
		return None, None
	layer, metric = rest.rsplit(".", 1)
	return (layer or None), (metric or None)


@dataclass(frozen=True)
class CorrEdge:
	bench_key: str
	repr_key: str
	layer: str
	metric: str
	group: str
	rho: float
	abs_rho: float
	p: float


def load_corr_edges(corr_csv_path: str, *, dataset_slug: str, bench_metric: str) -> List[CorrEdge]:
	path = os.path.abspath(str(corr_csv_path))
	if not os.path.isfile(path):
		return []
	want_bench = f"bench.{str(dataset_slug).lower()}-val.{str(bench_metric).lower()}"
	out: List[CorrEdge] = []
	with open(path, "r", encoding="utf-8") as f:
		rd = csv.DictReader(f)
		for r in rd:
			bk = str(r.get("bench_key", "") or "")
			if bk != want_bench:
				continue
			rk = str(r.get("repr_key", "") or "")
			layer, metric = _layer_metric_from_repr_key(rk)
			if not layer or not metric:
				continue
			try:
				rho = float(r.get("rho", "nan"))
				abs_rho = float(r.get("abs_rho", "nan"))
				pv = float(r.get("p", "nan"))
			except Exception:
				continue
			if not (math.isfinite(rho) and math.isfinite(abs_rho) and math.isfinite(pv)):
				continue
			out.append(
				CorrEdge(
					bench_key=bk,
					repr_key=rk,
					layer=str(layer),
					metric=str(metric),
					group=_metric_group_from_repr_key(rk),
					rho=float(rho),
					abs_rho=float(abs_rho),
					p=float(pv),
				)
			)
	return out


def _argmax_abs_rho(edges: Sequence[CorrEdge]) -> Optional[CorrEdge]:
	best: Optional[CorrEdge] = None
	for e in edges:
		if best is None or float(e.abs_rho) > float(best.abs_rho):
			best = e
	return best


def _best_strict_edge_for_layer(
	edges: Sequence[CorrEdge], layer: str, *, min_abs_rho: float, max_p: float
) -> Optional[CorrEdge]:
	layer_s = str(layer)
	best: Optional[CorrEdge] = None
	for e in edges:
		if str(e.layer) != layer_s:
			continue
		if float(e.abs_rho) < float(min_abs_rho) or float(e.p) > float(max_p):
			continue
		if best is None or float(e.abs_rho) > float(best.abs_rho):
			best = e
	return best


def descriptor_tex_from_repr_key(repr_key: str) -> str:
	_layer, metric = _layer_metric_from_repr_key(str(repr_key))
	m = str(metric or "")
	if m == "beta1_L_est":
		return r"$\beta_1\left(L\right)$"
	if m == "beta1_persistent_est":
		return r"$\beta_1^{K,L}$"
	if m == "mtopdiv_train_val":
		return r"$\mathrm{MTopDiv}$"
	mt = re.match(r"^(hodge_L_q(\d+)|persistent_q(\d+))_lambda(\d+)$", m)
	if mt:
		q = mt.group(2) or mt.group(3) or "?"
		k = mt.group(4) or "?"
		if mt.group(1).startswith("hodge_L"):
			return rf"$\lambda_{k}\left(\Delta_{q}\left(L\right)\right)$"
		return rf"$\lambda_{k}\left(\Delta_{q}^{{K,L}}\right)$"
	return str(repr_key)


def _round3(x: Optional[float]) -> Optional[float]:
	if x is None:
		return None
	try:
		v = float(x)
	except Exception:
		return None
	return None if not math.isfinite(v) else round(v, 3)


def _best_r_among_layers(r_by_layer: Mapping[str, float], layers: Sequence[str]) -> Optional[Tuple[str, float]]:
	best_layer: Optional[str] = None
	best_val: Optional[float] = None
	for lay in layers:
		if str(lay) not in r_by_layer:
			continue
		v = float(r_by_layer[str(lay)])
		if best_val is None or v > float(best_val):
			best_val = float(v)
			best_layer = str(lay)
	if best_layer is None or best_val is None:
		return None
	return best_layer, float(best_val)


def _strict_layer_set(edges: Sequence[CorrEdge], *, min_abs_rho: float, max_p: float) -> List[str]:
	seen: set[str] = set()
	out: List[str] = []
	for e in edges:
		if float(e.abs_rho) < float(min_abs_rho) or float(e.p) > float(max_p):
			continue
		if e.layer in seen:
			continue
		seen.add(e.layer)
		out.append(e.layer)
	return out


def _blank_row(method_key: str, method_label: str) -> SelectionRowDict:
	return SelectionRowDict(
		method_key=method_key,
		method_label=method_label,
		layer=None,
		R=None,
		gap_vs_best=None,
		descriptor_repr_key=None,
		descriptor_tex=None,
		abs_rho_s=None,
	)


def _filled_row(
	*,
	method_key: str,
	method_label: str,
	layer: Optional[str],
	r_val: Optional[float],
	r_star: float,
	descriptor_repr_key: Optional[str],
	abs_rho_s: Optional[float],
) -> SelectionRowDict:
	return SelectionRowDict(
		method_key=method_key,
		method_label=method_label,
		layer=(str(layer) if layer is not None else None),
		R=_round3(r_val),
		gap_vs_best=None if r_val is None else _round3(max(0.0, float(r_star) - float(r_val))),
		descriptor_repr_key=str(descriptor_repr_key) if descriptor_repr_key else None,
		descriptor_tex=descriptor_tex_from_repr_key(descriptor_repr_key) if descriptor_repr_key else None,
		abs_rho_s=_round3(abs_rho_s),
	)


def build_selection_rows(
	r_by_layer: Mapping[str, float],
	*,
	model_name: str,
	dataset_slug: str,
	corr_csv_path: str,
	strict_min_abs_rho: float,
	strict_max_p: float,
	bench_metric: str = "f1_macro",
) -> Tuple[List[SelectionRowDict], Dict[str, Any]]:
	r_star = max((float(v) for v in r_by_layer.values()), default=0.0)
	default_layer = infer_default_layer(model_name)
	edges = load_corr_edges(corr_csv_path, dataset_slug=dataset_slug, bench_metric=bench_metric)

	by_g: MutableMapping[str, List[CorrEdge]] = {"topo": [], "spectral": [], "mtopdiv": []}
	for e in edges:
		if e.group in by_g:
			by_g[str(e.group)].append(e)

	rows: List[SelectionRowDict] = []

	br = _best_r_among_layers(r_by_layer, list(r_by_layer.keys()))
	rows.append(
		_blank_row("best_r", "Наилучший слой по метрике R")
		if not br
		else _filled_row(
			method_key="best_r",
			method_label="Наилучший слой по метрике R",
			layer=br[0],
			r_val=br[1],
			r_star=r_star,
			descriptor_repr_key=None,
			abs_rho_s=None,
		)
	)

	r_def = float(r_by_layer[str(default_layer)]) if default_layer and str(default_layer) in r_by_layer else None
	rows.append(
		_filled_row(
			method_key="default_layer",
			method_label="Слой по умолчанию",
			layer=default_layer,
			r_val=r_def,
			r_star=r_star,
			descriptor_repr_key=None,
			abs_rho_s=None,
		)
	)

	def strict_and_argmax(group: str, strict_key: str, strict_label: str, max_key: str, max_label: str) -> None:
		g_edges = by_g[group]
		strict_layers = _strict_layer_set(g_edges, min_abs_rho=float(strict_min_abs_rho), max_p=float(strict_max_p))
		best = _best_r_among_layers(r_by_layer, strict_layers)
		if best:
			e = _best_strict_edge_for_layer(
				g_edges,
				str(best[0]),
				min_abs_rho=float(strict_min_abs_rho),
				max_p=float(strict_max_p),
			)
			rows.append(
				_filled_row(
					method_key=strict_key,
					method_label=strict_label,
					layer=best[0],
					r_val=best[1],
					r_star=r_star,
					descriptor_repr_key=None if not e else e.repr_key,
					abs_rho_s=None if not e else e.abs_rho,
				)
			)
		else:
			rows.append(_blank_row(strict_key, strict_label))

		e2 = _argmax_abs_rho(g_edges)
		rows.append(
			_blank_row(max_key, max_label)
			if not e2
			else _filled_row(
				method_key=max_key,
				method_label=max_label,
				layer=e2.layer,
				r_val=float(r_by_layer[e2.layer]) if e2.layer in r_by_layer else None,
				r_star=r_star,
				descriptor_repr_key=e2.repr_key,
				abs_rho_s=e2.abs_rho,
			)
		)

	strict_and_argmax(
		"topo",
		"topo_strict_best_r",
		"Наилучший слой среди топологически отобранных при строгом пороге",
		"topo_max_abs_rho",
		"Слой с наибольшей по модулю топологической корреляцией",
	)
	strict_and_argmax(
		"spectral",
		"spectral_strict_best_r",
		"Наилучший слой среди спектрально отобранных при строгом пороге",
		"spectral_max_abs_rho",
		"Слой с наибольшей по модулю спектральной корреляцией",
	)
	strict_and_argmax(
		"mtopdiv",
		"mtopdiv_strict_best_r",
		"Наилучший слой среди слоёв, отобранных по MTopDiv",
		"mtopdiv_max_abs_rho",
		"Слой с наибольшей по модулю корреляцией по MTopDiv",
	)

	meta = {
		"r_star": _round3(r_star),
		"default_layer": default_layer or None,
		"corr_csv": os.path.abspath(str(corr_csv_path)),
		"corr_bench_row": f"bench.{str(dataset_slug).lower()}-val.{str(bench_metric).lower()}",
		"strict_min_abs_rho": float(strict_min_abs_rho),
		"strict_max_p": float(strict_max_p),
		"n_corr_edges_used": len(edges),
	}
	return rows, meta

