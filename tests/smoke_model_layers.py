import torch
from tda_repr import LayerTaps, get_model_info


def run(kind: str) -> None:
	info = get_model_info(kind)
	model = info.model.eval()

	if info.family == "cnn":
		x = torch.rand(2, 3, 224, 224)
	elif info.family == "mlp":
		x = torch.rand(2, 1, 28, 28)
	else:
		# transformer: random token ids
		input_ids = torch.randint(0, 30522, (2, 16))
		attn = torch.ones_like(input_ids)
		with LayerTaps(model, info.layer_names) as taps:
			_ = model(input_ids=input_ids, attention_mask=attn)
			print(kind, {k: (v[0] if isinstance(v, tuple) else v).shape if hasattr(v, "shape") else type(v) for k, v in taps.outputs.items()})
		return

	if info.preprocess:
		x = info.preprocess(x)

	with LayerTaps(model, info.layer_names) as taps:
		_ = model(x)
		print(kind, {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in taps.outputs.items()})


if __name__ == "__main__":
	for key in ["mlp", "resnet18", "convnext_tiny", "efficientnet_b0", "distilbert"]:
		try:
			run(key)
		except Exception as e:
			print(f"skip {key}: {e}")
