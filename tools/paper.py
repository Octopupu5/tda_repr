from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class _Cmd:
	name: str
	argv: List[str]


def _run(cmd: _Cmd) -> None:
	proc = subprocess.run(cmd.argv)
	if proc.returncode != 0:
		raise SystemExit(f"[FAIL] {cmd.name} (exit={proc.returncode})")


def main() -> None:
	ap = argparse.ArgumentParser(description="Paper helpers (tables + figures) with a compact interface.")
	sub = ap.add_subparsers(dest="cmd", required=True)

	p_tables = sub.add_parser("tables", help="Build all paper tables from existing runs/* artifacts.")
	p_tables.add_argument("--runs_dir", type=str, default="runs")
	p_tables.add_argument("--analysis_out_dir", type=str, default="paper/analysis_tables")
	p_tables.add_argument("--paper_out_dir", type=str, default="paper/analysis_tables_ftb")
	p_tables.add_argument("--runs_summary_csv", type=str, default="")
	p_tables.add_argument("--abs_rho_threshold", type=float, default=0.6)
	p_tables.add_argument("--corr_min_abs_rho", type=float, default=0.6)
	p_tables.add_argument("--corr_max_p", type=float, default=0.05)
	p_tables.add_argument("--skip_depth_arch", action="store_true")
	p_tables.add_argument("--skip_corr_summary", action="store_true")
	p_tables.add_argument("--skip_layer_selection", action="store_true")
	p_tables.add_argument("--skip_layer_rel_dev", action="store_true")
	p_tables.add_argument("--skip_early_stop_single", action="store_true")
	p_tables.add_argument("--skip_early_stop_ensemble", action="store_true")

	p_fig = sub.add_parser("fig", help="Render a specific paper figure.")
	p_fig.add_argument(
		"kind",
		type=str,
		choices=["mtopdiv_best_layer_dynamics", "layerwise_descriptor_dynamics", "early_stopping_case", "early_stopping_triplet_case"],
	)
	p_fig.add_argument("--help_args", action="store_true", help="Print the underlying script help and exit.")
	p_fig.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the underlying script after '--'.")

	args = ap.parse_args()

	root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	py = sys.executable

	if args.cmd == "tables":
		scripts_dir = os.path.join(root, "tools", "paper")
		runs_dir = os.path.abspath(str(args.runs_dir))
		analysis_out_dir = os.path.abspath(str(args.analysis_out_dir))
		paper_out_dir = os.path.abspath(str(args.paper_out_dir))
		runs_summary_csv = str(args.runs_summary_csv).strip()
		if not runs_summary_csv:
			runs_summary_csv = os.path.join(analysis_out_dir, "runs_summary.csv")
		runs_summary_csv = os.path.abspath(str(runs_summary_csv))

		cmds: List[_Cmd] = []
		if not bool(args.skip_depth_arch):
			cmds.append(
				_Cmd(
					"tables_depth_arch",
					[
						py,
						os.path.join(scripts_dir, "tables_depth_arch.py"),
						"--runs_dir",
						runs_dir,
						"--out_dir",
						analysis_out_dir,
					],
				)
			)
		if not bool(args.skip_corr_summary):
			cmds.append(
				_Cmd(
					"tables_corr_summary",
					[
						py,
						os.path.join(scripts_dir, "tables_corr_summary.py"),
						"--runs_dir",
						runs_dir,
						"--out_tex",
						os.path.join(paper_out_dir, "table_correlation_summary.tex"),
						"--out_rows_tex",
						os.path.join(paper_out_dir, "table_correlation_summary_rows.tex"),
						"--abs_rho_threshold",
						str(float(args.abs_rho_threshold)),
					],
				)
			)
		if (not bool(args.skip_layer_selection)) or (not bool(args.skip_layer_rel_dev)):
			if bool(args.skip_depth_arch) and not os.path.exists(runs_summary_csv):
				raise SystemExit(
					"Missing runs_summary.csv for layer-selection tables.\n"
					f"Looked for: {runs_summary_csv}\n"
					"Either run without --skip_depth_arch, or pass --runs_summary_csv explicitly."
				)
		if not bool(args.skip_layer_selection):
			cmds.append(
				_Cmd(
					"tables_layer_selection",
					[
						py,
						os.path.join(scripts_dir, "tables_layer_selection.py"),
						"--runs_dir",
						runs_dir,
						"--runs_summary_csv",
						runs_summary_csv,
						"--out_dir",
						paper_out_dir,
						"--corr_min_abs_rho",
						str(float(args.corr_min_abs_rho)),
						"--corr_max_p",
						str(float(args.corr_max_p)),
					],
				)
			)
		if not bool(args.skip_layer_rel_dev):
			cmds.append(
				_Cmd(
					"tables_layer_rel_dev",
					[
						py,
						os.path.join(scripts_dir, "tables_layer_rel_dev.py"),
						"--runs_dir",
						runs_dir,
						"--runs_summary_csv",
						runs_summary_csv,
						"--out_rows_tex",
						os.path.join(paper_out_dir, "table_layer_selection_relative_deviation_rows.tex"),
						"--corr_min_abs_rho",
						str(float(args.corr_min_abs_rho)),
						"--corr_max_p",
						str(float(args.corr_max_p)),
					],
				)
			)
		if not bool(args.skip_early_stop_single):
			cmds.append(
				_Cmd(
					"tables_early_stop_single",
					[
						py,
						os.path.join(scripts_dir, "tables_early_stop_single.py"),
						"--runs_dir",
						runs_dir,
						"--out_dir",
						paper_out_dir,
					],
				)
			)
		if not bool(args.skip_early_stop_ensemble):
			cmds.append(
				_Cmd(
					"tables_early_stop_ensemble",
					[
						py,
						os.path.join(scripts_dir, "tables_early_stop_ensemble.py"),
						"--runs_dir",
						runs_dir,
						"--out_tex",
						os.path.join(paper_out_dir, "table_early_stopping_best_online_paper.tex"),
						"--out_tex_heldout",
						os.path.join(paper_out_dir, "table_early_stopping_best_heldout_paper.tex"),
					],
				)
			)

		for c in cmds:
			_run(c)
		return

	if args.cmd == "fig":
		scripts_dir = os.path.join(root, "tools", "paper")
		entry = {
			"mtopdiv_best_layer_dynamics": "fig_mtopdiv_best_layer_dynamics.py",
			"layerwise_descriptor_dynamics": "fig_layerwise_descriptor_dynamics.py",
			"early_stopping_case": "fig_early_stopping_case.py",
			"early_stopping_triplet_case": "fig_early_stopping_triplet_case.py",
		}[str(args.kind)]
		argv = [py, os.path.join(scripts_dir, entry)]
		if bool(args.help_args):
			argv += ["--help"]
		else:
			rest = list(args.args or [])
			if rest and rest[0] == "--":
				rest = rest[1:]
			argv += rest
		_run(_Cmd(f"fig:{args.kind}", argv))
		return


if __name__ == "__main__":
	main()

