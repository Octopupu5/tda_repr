import torch

from tda_repr.models.registry import get_model_info


def test_registry_default_layers_exist_for_cv_models() -> None:
	kinds = ["mlp", "resnet18", "convnext_tiny", "efficientnet_b0"]
	for kind in kinds:
		mi = get_model_info(kind, device=torch.device("cpu"), pretrained=False)
		all_mods = set(dict(mi.model.named_modules()).keys())
		missing = [n for n in mi.layer_names if n not in all_mods]
		assert not missing, f"{kind}: missing layers {missing}"
		assert len(mi.layer_names) > 0

