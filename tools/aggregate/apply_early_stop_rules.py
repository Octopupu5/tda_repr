from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tools.repr_early_stop_sweep import evaluate_ensemble3_rule_on_run, evaluate_single_signal_rule_on_run


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as rf:
		return json.load(rf)


def _discover_run_dirs(roots: Sequence[str]) -> List[str]:
	out: List[str] = []
	for rt in roots:
		root_abs = os.path.abspath(str(rt))
		if not os.path.isdir(root_abs):
			raise FileNotFoundError(root_abs)
		for dp, _dirs, fnames in os.walk(root_abs):
			if os.path.basename(dp).startswith("exp_") and "meta.json" in fnames and "metrics.jsonl" in fnames:
				out.append(os.path.abspath(dp))
	return sorted(set(out))


def _group_key(meta: Mapping[str, Any]) -> str:
	model = str(meta.get("model", "") or "").strip()
	if not model:
		model = str((meta.get("args") or {}).get("model", "") or "").strip()
	ds = str(meta.get("dataset", "") or "").strip()
	if not model or not ds:
		raise ValueError("meta.json missing model/dataset.")
	return f"{model}::{ds}"


def _rule_rows_for_run(run_dir: str, rules_obj: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	meta = _load_json(os.path.join(run_dir, "meta.json"))
	gk = _group_key(meta)

	primary = (rules_obj.get("primary") or {}).get(gk)
	ens_top = (rules_obj.get("ensembles_top_k") or {}).get(gk) or []
	if not primary:
		raise KeyError(f"Missing primary rule for group {gk!r} in rules JSON.")

	specs: List[Dict[str, Any]] = []
	specs.append(dict(primary))
	for it in ens_top:
		if isinstance(it, dict):
			specs.append(dict(it))

	seen = set()
	specs_u: List[Dict[str, Any]] = []
	for sp in specs:
		key = json.dumps(sp, sort_keys=True, ensure_ascii=False)
		if key not in seen:
			seen.add(key)
			specs_u.append(sp)

	rows: List[Dict[str, Any]] = []
	ens_rows: List[Dict[str, Any]] = []
	for sp in specs_u:
		t = str(sp.get("type", "single")).lower().strip()
		start_epoch = int(sp.get("start_epoch", 3))
		min_delta = float(sp.get("min_delta", 0.0))
		patience = int(sp.get("patience", 3))
		bench_metric_override = str(sp.get("bench_metric", "") or "").strip()
		if t == "single":
			row = evaluate_single_signal_rule_on_run(
				run_dir,
				metric=str(sp["metric"]),
				mode=str(sp["mode"]),
				layer=str(sp["layer"]),
				patience=patience,
				start_epoch=start_epoch,
				min_delta=min_delta,
				bench_metric_override=bench_metric_override,
			)
		elif t == "ensemble3":
			row = evaluate_ensemble3_rule_on_run(
				run_dir,
				metrics=list(sp["metrics"]),
				modes=list(sp["modes"]),
				layers=list(sp["layers"]),
				aggregate=str(sp.get("aggregate", "all")),
				patience=patience,
				start_epoch=start_epoch,
				min_delta=min_delta,
				bench_metric_override=bench_metric_override,
			)
			ens_rows.append(dict(row))
		else:
			raise ValueError(f"Unknown rule type: {t}")
		rows.append(dict(row))

	rows.sort(key=lambda r: (float(r.get("gap_rel_pct", 1e9)), float(r.get("effective_stop_epoch", 0))))
	return rows, ens_rows


def main(argv: Sequence[str] | None = None) -> None:
	if argv is None:
		argv = sys.argv[1:]
	ap = argparse.ArgumentParser(description="Apply fixed early-stop rules to evaluation runs.")
	ap.add_argument("--roots", type=str, nargs="+", required=True)
	ap.add_argument("--rules_json", type=str, required=True)
	ap.add_argument("--out_suffix", type=str, default="repr_early_stop_sweep.json")
	ap.add_argument("--skip_existing", action="store_true")
	args = ap.parse_args(list(argv))

	rules_obj = _load_json(str(args.rules_json))
	if not isinstance(rules_obj, dict):
		raise SystemExit("--rules_json must be a JSON object.")

	run_dirs = _discover_run_dirs(args.roots)
	if not run_dirs:
		raise SystemExit("No run dirs found under roots.")

	for rd in run_dirs:
		out_json = os.path.join(rd, "analysis", str(args.out_suffix))
		if args.skip_existing and os.path.isfile(out_json) and os.path.getsize(out_json) > 10:
			print("[skip existing]", out_json, file=sys.stderr)
			continue
		rows, ens_rows = _rule_rows_for_run(rd, rules_obj)
		best = rows[0] if rows else None
		obj = {
			"run_dir": os.path.abspath(rd),
			"eval_note": "fixed_rules: evaluated only selected primary rule and top-K ensemble rules from selection corpus.",
			"ranked_by_gap_rel_pct": rows,
			"ensemble3_oracle_rows": ens_rows,
			"best": best,
		}
		os.makedirs(os.path.join(rd, "analysis"), exist_ok=True)
		with open(out_json, "w", encoding="utf-8") as wf:
			json.dump(obj, wf, indent=2, ensure_ascii=False)
		print("[ok] wrote", out_json, file=sys.stderr)


if __name__ == "__main__":
	main()
