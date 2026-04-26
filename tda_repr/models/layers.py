from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import re
from typing import Iterable, List, Sequence

import torch.nn as nn


@dataclass
class SelectionReport:
	selected: List[str]
	unmatched: List[str]


class SelectionValidationError(ValueError):
	pass


def list_module_names(model: nn.Module, leaf_only: bool = False) -> List[str]:
	"""
	Return named_modules() keys excluding root "".
	If leaf_only=True, only return modules without children.
	"""
	out: List[str] = []
	for name, module in model.named_modules():
		if not name:
			continue
		if leaf_only and any(True for _ in module.children()):
			continue
		out.append(name)
	return out


def list_parameter_names(model: nn.Module, trainable_only: bool = False) -> List[str]:
	out = []
	for name, p in model.named_parameters():
		if trainable_only and not bool(p.requires_grad):
			continue
		out.append(name)
	return out


def _match_any(name: str, patterns: Sequence[str], use_regex: bool) -> bool:
	if not patterns:
		return True
	if use_regex:
		for pat in patterns:
			if re.search(pat, name):
				return True
		return False
	for pat in patterns:
		if fnmatch(name, pat):
			return True
	return False


def select_names(
	names: Iterable[str],
	include: Sequence[str] | None = None,
	exclude: Sequence[str] | None = None,
	use_regex: bool = False,
) -> SelectionReport:
	"""
	Select names by include/exclude patterns.
	- include/exclude support glob by default and regex when use_regex=True.
	- returns both selected names and unmatched include patterns for UX.
	"""
	include = [x for x in (include or []) if str(x).strip()]
	exclude = [x for x in (exclude or []) if str(x).strip()]
	names_list = list(names)

	selected: List[str] = []
	for n in names_list:
		if include and not _match_any(n, include, use_regex=use_regex):
			continue
		if exclude and _match_any(n, exclude, use_regex=use_regex):
			continue
		selected.append(n)

	unmatched: List[str] = []
	if include:
		for pat in include:
			if not any(_match_any(n, [pat], use_regex=use_regex) for n in names_list):
				unmatched.append(pat)

	return SelectionReport(selected=selected, unmatched=unmatched)


def freeze_all(model: nn.Module) -> None:
	for p in model.parameters():
		p.requires_grad = False


def set_trainable_by_name_selection(
	model: nn.Module,
	include: Sequence[str] | None = None,
	exclude: Sequence[str] | None = None,
	use_regex: bool = False,
	strict: bool = False,
) -> SelectionReport:
	"""
	Freeze all params, then unfreeze parameters selected by name patterns.
	Patterns are matched against parameter names from model.named_parameters().
	"""
	all_names = list_parameter_names(model, trainable_only=False)
	rep = select_names(all_names, include=include, exclude=exclude, use_regex=use_regex)
	if strict and rep.unmatched:
		raise SelectionValidationError(f"Unmatched parameter selection patterns: {rep.unmatched}")
	if strict and not rep.selected:
		raise SelectionValidationError("Parameter selection produced an empty set.")
	freeze_all(model)
	selected_set = set(rep.selected)
	for n, p in model.named_parameters():
		p.requires_grad = n in selected_set
	return rep


def csv_to_list(s: str) -> List[str]:
	return [x.strip() for x in str(s).split(",") if x.strip()]

