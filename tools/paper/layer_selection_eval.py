from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
	try:
		v = float(x)
	except Exception:
		return None
	if v != v:  # nan
		return None
	return v


def _find_corr_csv(run_dir: str) -> Optional[str]:
	cands = [
		os.path.join(run_dir, "correlations_report", "all_pairs.csv"),
		os.path.join(run_dir, "all_pairs.csv"),
	]
	for p in cands:
		if os.path.exists(p):
			return p
	return None


def _infer_task_from_meta(run_dir: str) -> str:
	meta_path = os.path.join(run_dir, "meta.json")
	if not os.path.exists(meta_path):
		return "cv"
	meta = _read_json(meta_path)
	task = str(meta.get("task", (meta.get("args", {}) or {}).get("task", "cv"))).lower().strip()
	return task or "cv"


def _default_device_from_meta(run_dir: str) -> str:
	meta_path = os.path.join(run_dir, "meta.json")
	if not os.path.exists(meta_path):
		return ""
	meta = _read_json(meta_path)
	return str((meta.get("args", {}) or {}).get("device", "") or "")


def _run_eval(cmd: List[str]) -> None:
	subprocess.check_call(cmd)


def _summary_path(run_dir: str, checkpoint: str) -> str:
	ck = str(checkpoint).strip()
	tag = {
		"best_main": "model_best_main",
		"best": "model_best_main",
		"early_signal": "model_early_signal",
		"early": "model_early_signal",
	}.get(ck, os.path.splitext(os.path.basename(ck))[0])
	return os.path.join(run_dir, "analysis", f"embedding_retrieval_{tag}__summary.json")


def _pick_best_layer(summary: dict) -> Tuple[Optional[str], Optional[float]]:
	rows = [r for r in (summary.get("rows", []) or []) if isinstance(r, dict) and "macro_avg_same_class_ratio" in r]
	if not rows:
		return None, None
	rows.sort(key=lambda r: float(r.get("macro_avg_same_class_ratio", -1.0)), reverse=True)
	best = rows[0]
	return str(best.get("layer", "")) or None, _safe_float(best.get("macro_avg_same_class_ratio", None))


def _pick_top_layers(summary: dict, eps: float = 1e-3, max_layers: int = 2) -> Tuple[str, float]:
	rows = [r for r in (summary.get("rows", []) or []) if isinstance(r, dict) and "macro_avg_same_class_ratio" in r]
	rows.sort(key=lambda r: float(r.get("macro_avg_same_class_ratio", -1.0)), reverse=True)
	best_layer = str(rows[0].get("layer", ""))
	best_r = float(rows[0].get("macro_avg_same_class_ratio", 0.0))
	alts = [best_layer]
	for rr in rows[1:]:
		if len(alts) >= int(max_layers):
			break
		v = float(rr.get("macro_avg_same_class_ratio", -1.0))
		if abs(v - best_r) <= float(eps):
			alts.append(str(rr.get("layer", "")))
	if len(alts) == 1:
		return best_layer, best_r
	return " / ".join(alts), best_r


@dataclass
class Row:
	architecture: str
	run_dir: str
	oracle_layer: Optional[str]
	oracle_r: Optional[float]
	proposed_layer: Optional[str]
	proposed_r: Optional[float]
	gap: Optional[float]


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


def main() -> None:
	ap = argparse.ArgumentParser(description="Run embedding retrieval eval and build layer-selection table.")
	ap.add_argument("--runs_dir", type=str, default="runs")
	ap.add_argument("--out_dir", type=str, default="analysis_tables")
	ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
	ap.add_argument("--checkpoint", type=str, default="best_main")
	ap.add_argument("--device", type=str, default="", help="Override device (mps/cuda:0/cpu). Default: inferred.")
	ap.add_argument("--top_k", type=int, default=1)
	ap.add_argument("--anchors_per_class", type=int, default=1)
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--max_batches", type=int, default=0)
	ap.add_argument("--max_samples", type=int, default=0)
	ap.add_argument("--corr_group", type=str, default="topo_mtopdiv", choices=["topo_mtopdiv", "spectral"])
	ap.add_argument("--top_corr_layers", type=int, default=1)
	ap.add_argument("--corr_min_abs_rho", type=float, default=0.6)
	ap.add_argument("--corr_max_p", type=float, default=0.05)
	ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
	ap.add_argument("--include", type=str, default="", help="Substring filter for exp_ dir name.")
	args = ap.parse_args()

	runs_dir = os.path.abspath(str(args.runs_dir))
	out_dir = os.path.abspath(str(args.out_dir))
	os.makedirs(out_dir, exist_ok=True)

	rows: List[Row] = []
	for name in sorted(os.listdir(runs_dir)):
		if not name.startswith("exp_"):
			continue
		if args.include and str(args.include) not in name:
			continue
		run_dir = os.path.join(runs_dir, name)
		meta_path = os.path.join(run_dir, "meta.json")
		if not os.path.exists(meta_path):
			continue
		meta = _read_json(meta_path)
		model = str(meta.get("model", (meta.get("args", {}) or {}).get("model", "")) or "")
		arch = _display_arch(model)

		task = _infer_task_from_meta(run_dir)
		# Table in paper wants CV+NLP; current evaluate_embeddings supports CV and NLP classification.
		if task not in {"cv", "nlp"}:
			continue

		device = str(args.device).strip() or _default_device_from_meta(run_dir)
		cmd_base = [
			sys.executable,
			os.path.join("tools", "evaluate_embeddings.py"),
			"--run_dir",
			run_dir,
			"--checkpoint",
			str(args.checkpoint),
			"--split",
			str(args.split),
			"--top_k",
			str(int(args.top_k)),
			"--anchors_per_class",
			str(int(args.anchors_per_class)),
			"--seed",
			str(int(args.seed)),
			"--max_batches",
			str(int(args.max_batches)),
			"--max_samples",
			str(int(args.max_samples)),
		]
		if device:
			cmd_base += ["--device", device]
		if bool(args.skip_existing):
			cmd_base += ["--skip_existing"]
		else:
			cmd_base += ["--no-skip_existing"]

		# 1) Oracle (all main layers)
		_run_eval(cmd_base + ["--all_main_layers"])
		oracle_summary_path = _summary_path(run_dir, str(args.checkpoint))
		oracle = _read_json(oracle_summary_path) if os.path.exists(oracle_summary_path) else {}
		oracle_layer, oracle_r = _pick_best_layer(oracle)

		# 2) Proposed (from correlations)
		corr_csv = _find_corr_csv(run_dir)
		proposed_layer = None
		proposed_r = None
		if corr_csv:
			_run_eval(
				cmd_base
				+ [
					"--layers_from_correlations",
					str(args.corr_group),
					"--correlation_csv",
					corr_csv,
					"--top_corr_layers",
					str(int(args.top_corr_layers)),
					"--corr_min_abs_rho",
					str(float(args.corr_min_abs_rho)),
					"--corr_max_p",
					str(float(args.corr_max_p)),
				]
			)
			proposed_summary_path = _summary_path(run_dir, str(args.checkpoint))
			proposed = _read_json(proposed_summary_path) if os.path.exists(proposed_summary_path) else {}
			if proposed:
				proposed_layer, proposed_r = _pick_top_layers(proposed, eps=1e-3, max_layers=2)

		gap = None
		if oracle_r is not None and proposed_r is not None:
			gap = float(oracle_r) - float(proposed_r)

		rows.append(
			Row(
				architecture=arch,
				run_dir=run_dir,
				oracle_layer=oracle_layer,
				oracle_r=oracle_r,
				proposed_layer=proposed_layer,
				proposed_r=proposed_r,
				gap=gap,
			)
		)

	# write CSV and a simple LaTeX snippet (one row per run)
	csv_path = os.path.join(out_dir, "layer_selection_by_run.csv")
	with open(csv_path, "w", encoding="utf-8") as f:
		f.write("architecture,run_dir,oracle_layer,oracle_r,proposed_layer,proposed_r,gap\n")
		for r in rows:
			f.write(
				f"{r.architecture},{r.run_dir},{r.oracle_layer or ''},{r.oracle_r if r.oracle_r is not None else ''},"
				f"{r.proposed_layer or ''},{r.proposed_r if r.proposed_r is not None else ''},{r.gap if r.gap is not None else ''}\n"
			)

	tex_path = os.path.join(out_dir, "table_layer_selection_rows_by_run.tex")
	with open(tex_path, "w", encoding="utf-8") as f:
		for r in rows:
			ol = f"\\texttt{{{r.oracle_layer}}}" if r.oracle_layer else "NA"
			orv = f"{r.oracle_r:.3f}" if r.oracle_r is not None else "NA"
			pl = f"\\texttt{{{r.proposed_layer}}}" if r.proposed_layer else "NA"
			prv = f"{r.proposed_r:.3f}" if r.proposed_r is not None else "NA"
			g = f"{r.gap:.3f}" if r.gap is not None else "NA"
			f.write(f"{r.architecture} & {ol} & {orv} & {pl} & {prv} & {g} \\\\\n")

	print("[OK] wrote:", csv_path)
	print("[OK] wrote:", tex_path)


if __name__ == "__main__":
	main()

