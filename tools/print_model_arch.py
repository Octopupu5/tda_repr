import argparse
import importlib.util
import os
from typing import List

import torch


def _load_registry_module():
	# Load registry.py directly to avoid importing the full `tda_repr` package.
	# This is useful when optional deps are not installed yet.
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	reg_path = os.path.join(root, "tda_repr", "models", "registry.py")
	spec = importlib.util.spec_from_file_location("tda_repr_models_registry", reg_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Failed to create module spec for: {reg_path}")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _split_csv(s: str) -> List[str]:
	return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--kinds",
		type=str,
		default="mlp,resnet18,convnext_tiny,efficientnet_b0,distilbert,smollm",
		help="Comma-separated model keys from tda_repr/models/registry.py",
	)
	ap.add_argument("--pretrained", action="store_true", help="If set, request pretrained configs/weights where supported.")
	ap.add_argument("--max_names", type=int, default=120, help="How many module paths to print from named_modules().")
	args = ap.parse_args()

	mod = _load_registry_module()
	get_model_info = mod.get_model_info

	for kind in _split_csv(args.kinds):
		print("\n" + "=" * 80)
		print("KIND:", kind)
		mi = get_model_info(kind, device=torch.device("cpu"), pretrained=bool(args.pretrained))
		model = mi.model
		print(model)
		print("layer_names:", mi.layer_names)
		all_names = list(dict(model.named_modules()).keys())
		print("named_modules:", len(all_names))
		print("first:", all_names[: max(0, min(int(args.max_names), len(all_names)))])


if __name__ == "__main__":
	main()
