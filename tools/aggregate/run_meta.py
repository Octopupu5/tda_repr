from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RunMeta:
	run_dir: str
	task: str
	dataset: str
	model: str
	finetune: str
	args: Dict[str, Any]


def _read_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def list_run_dirs(runs_root: str) -> List[str]:
	root = os.path.abspath(str(runs_root))
	if not os.path.isdir(root):
		raise FileNotFoundError(f"runs_root does not exist or is not a directory: {root!r}")
	out: List[str] = []
	for name in sorted(os.listdir(root)):
		p = os.path.join(root, name)
		if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
			out.append(p)
	return out


def load_run_meta(run_dir: str) -> RunMeta:
	rd = os.path.abspath(str(run_dir))
	meta_path = os.path.join(rd, "meta.json")
	meta = _read_json(meta_path)
	args = meta.get("args", None)
	if not isinstance(args, dict):
		raise ValueError(f"Run meta is missing a dict field 'args': run_dir={rd!r}")
	task = str(meta.get("task", args.get("task", ""))).strip().lower()
	dataset = str(meta.get("dataset", args.get("dataset", ""))).strip().lower()
	model = str(meta.get("model", args.get("model", ""))).strip().lower()
	finetune = str(meta.get("finetune", args.get("finetune", "full"))).strip().lower()
	if not task or not dataset or not model:
		raise ValueError(
			"Run meta is missing required fields (task/dataset/model). "
			f"run_dir={rd!r} task={task!r} dataset={dataset!r} model={model!r}"
		)
	return RunMeta(run_dir=rd, task=task, dataset=dataset, model=model, finetune=finetune, args=dict(args))

