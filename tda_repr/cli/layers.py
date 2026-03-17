from __future__ import annotations

import argparse

import torch

from tda_repr.models import csv_to_list, get_model_info, list_module_names, select_names


def main() -> None:
	ap = argparse.ArgumentParser(description="Inspect model modules and default monitor layers.")
	ap.add_argument("--model", type=str, required=True, help="Registry key, e.g. resnet18")
	ap.add_argument("--device", type=str, default="cpu")
	ap.add_argument("--pretrained", action="store_true")
	ap.add_argument("--leaf_only", action="store_true")
	ap.add_argument("--include", type=str, default="", help="CSV include patterns (glob by default)")
	ap.add_argument("--exclude", type=str, default="", help="CSV exclude patterns")
	ap.add_argument("--regex", action="store_true")
	ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False, help="Fail if any include patterns are unmatched.")
	args = ap.parse_args()

	mi = get_model_info(args.model, device=torch.device(args.device), pretrained=bool(args.pretrained))
	all_modules = list_module_names(mi.model, leaf_only=bool(args.leaf_only))
	rep = select_names(
		all_modules,
		include=csv_to_list(args.include),
		exclude=csv_to_list(args.exclude),
		use_regex=bool(args.regex),
	)
	if args.strict and rep.unmatched:
		raise SystemExit(f"Unmatched patterns: {rep.unmatched}")

	print(f"[model] {mi.name} ({mi.family})")
	print(f"[default_layer_names] {len(mi.layer_names)}")
	for n in mi.layer_names:
		print("  ", n)

	print(f"[modules] total={len(all_modules)} selected={len(rep.selected)}")
	for n in rep.selected:
		print("  ", n)
	if rep.unmatched:
		print("[unmatched_patterns]", ", ".join(rep.unmatched))


if __name__ == "__main__":
	main()

