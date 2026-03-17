import torch.nn as nn

from tda_repr.models.layers import (
	SelectionValidationError,
	freeze_all,
	list_module_names,
	list_parameter_names,
	select_names,
	set_trainable_by_name_selection,
)


def _toy_model() -> nn.Module:
	return nn.Sequential(
		nn.Linear(8, 16),
		nn.ReLU(),
		nn.Linear(16, 4),
	)


def test_list_module_names_non_empty() -> None:
	model = _toy_model()
	names = list_module_names(model)
	assert "0" in names
	assert "1" in names
	assert "2" in names


def test_select_names_glob_include_exclude() -> None:
	names = ["layer1.0", "layer1.1", "layer2.0", "fc"]
	rep = select_names(names, include=["layer*"], exclude=["*.1"], use_regex=False)
	assert rep.selected == ["layer1.0", "layer2.0"]
	assert rep.unmatched == []


def test_set_trainable_by_name_selection_strict_unmatched_raises() -> None:
	model = _toy_model()
	try:
		set_trainable_by_name_selection(model, include=["does_not_exist*"], strict=True)
		assert False, "Expected SelectionValidationError"
	except SelectionValidationError:
		pass


def test_set_trainable_by_name_selection_applies_mask() -> None:
	model = _toy_model()
	freeze_all(model)
	rep = set_trainable_by_name_selection(model, include=["2.*"], strict=True)
	assert rep.selected
	trainable = set(list_parameter_names(model, trainable_only=True))
	assert trainable == {"2.weight", "2.bias"}

