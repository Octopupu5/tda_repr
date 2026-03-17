import argparse
import os
import sys

import torch

# Allow running as a script: ensure project root (parent of /tools) is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from tda_repr.models import csv_to_list, get_model_info, list_module_names, select_names


def main() -> None:
	ap = argparse.ArgumentParser(description="Inspect model modules/parameters and pattern selections.")
	ap.add_argument("--model", type=str, required=True, help="Model key from registry, e.g. resnet18.")
	ap.add_argument("--device", type=str, default="cpu", help="Use cpu by default.")
	ap.add_argument("--pretrained", action="store_true")
	ap.add_argument("--leaf_only", action="store_true", help="List only leaf modules.")
	ap.add_argument("--include", type=str, default="", help="CSV include patterns (glob by default).")
	ap.add_argument("--exclude", type=str, default="", help="CSV exclude patterns (glob by default).")
	ap.add_argument("--regex", action="store_true", help="Treat include/exclude patterns as regex.")
	ap.add_argument("--show_params", action="store_true", help="Also print parameter names and shapes.")
	args = ap.parse_args()

	mi = get_model_info(args.model, device=torch.device(args.device), pretrained=bool(args.pretrained))
	model = mi.model
	modules = list_module_names(model, leaf_only=bool(args.leaf_only))
	include = csv_to_list(args.include)
	exclude = csv_to_list(args.exclude)
	rep = select_names(modules, include=include, exclude=exclude, use_regex=bool(args.regex))

	print(f"[Model] {mi.name} ({mi.family})")
	print(f"[Default layer_names] {len(mi.layer_names)}")
	for x in mi.layer_names:
		print("  ", x)

	print(f"\n[Modules] total={len(modules)} selected={len(rep.selected)}")
	for x in rep.selected:
		print("  ", x)
	if rep.unmatched:
		print("[Unmatched include patterns]")
		for p in rep.unmatched:
			print("  ", p)

	if args.show_params:
		print("\n[Parameters]")
		for n, p in model.named_parameters():
			print(f"  {n}\tshape={tuple(p.shape)}\trequires_grad={bool(p.requires_grad)}")


if __name__ == "__main__":
	main()

