import argparse
import gc
import json
import os
import platform
import sys
import time
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _project_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_matplotlib_cache(root: str) -> None:
	"""
	Keep matplotlib from writing into unwritable locations (common on shared systems / WSL).
	"""
	mplcfg = os.path.join(root, ".mplconfig")
	os.makedirs(mplcfg, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", mplcfg)


def _load_run_experiment_module(root: str) -> ModuleType:
	tools_dir = os.path.join(root, "tools")
	if not os.path.isdir(tools_dir):
		raise RuntimeError(f"Expected tools/ directory at {tools_dir!r}")
	sys.path.insert(0, tools_dir)
	try:
		import run_experiment as rexp  # type: ignore
	except Exception as e:
		raise RuntimeError("Failed to import tools/run_experiment.py as a module.") from e
	return rexp


def _cleanup_memory(device: str) -> None:
	gc.collect()
	if torch.cuda.is_available() and str(device).startswith("cuda"):
		torch.cuda.empty_cache()
	if platform.system() == "Linux":
		# Best-effort: WSL/glibc may keep arenas; trimming helps return memory to OS.
		try:
			import ctypes

			libc = ctypes.CDLL("libc.so.6")
			trim = getattr(libc, "malloc_trim", None)
			if trim is not None:
				trim(0)
		except Exception:
			pass


def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
	"""
	Move optimizer state tensors to device after load_state_dict.
	"""
	for state in opt.state.values():
		for k, v in list(state.items()):
			if torch.is_tensor(v):
				state[k] = v.to(device)


def _score_for_best(metrics: Dict[str, Any]) -> Tuple[float, str]:
	"""
	Return (score, name) where higher is better.
	For generation we minimize ppl/loss via negative score.
	"""
	if metrics.get("ppl", None) is not None:
		return -float(metrics["ppl"]), "ppl_min"
	if metrics.get("loss_assistant_only", None) is not None:
		return -float(metrics["loss_assistant_only"]), "loss_assistant_only_min"
	if metrics.get("loss", None) is not None:
		return -float(metrics["loss"]), "loss_min"
	if metrics.get("bleu", None) is not None:
		return float(metrics["bleu"]), "bleu"
	return -float("inf"), "none"


def _schedule(epoch: int, base: bool, every: int) -> bool:
	if int(every) and int(every) > 0:
		return (int(epoch) % int(every)) == 0
	return bool(base)


def _build_generation_tokenizer(model_name_or_path: str) -> Any:
	from transformers import AutoTokenizer

	tok = AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)
	if getattr(tok, "pad_token", None) is None:
		eos = getattr(tok, "eos_token", None)
		if eos is not None:
			tok.pad_token = eos
		else:
			tok.add_special_tokens({"pad_token": "[PAD]"})
	if getattr(tok, "pad_token_id", None) is None:
		raise ValueError("Tokenizer is missing pad_token_id after pad token setup.")
	try:
		tok.padding_side = "left"
	except Exception as e:
		raise RuntimeError("Failed to set tokenizer.padding_side='left'.") from e
	return tok


def _ensure_model_generation_padding(model: Any, tokenizer: Any, device: torch.device) -> None:
	pad_id = getattr(tokenizer, "pad_token_id", None)
	if pad_id is None:
		eos_id = getattr(tokenizer, "eos_token_id", None)
		if isinstance(eos_id, (tuple, list)):
			eos_id = eos_id[0] if eos_id else None
		if eos_id is None:
			raise ValueError("Tokenizer is missing eos_token_id; required for generation.")
		pad_id = int(eos_id)
	try:
		model.config.pad_token_id = int(pad_id)
	except Exception as e:
		raise RuntimeError("Failed to set model.config.pad_token_id.") from e
	if hasattr(model, "generation_config") and getattr(model, "generation_config", None) is not None:
		try:
			model.generation_config.pad_token_id = int(pad_id)
		except Exception as e:
			raise RuntimeError("Failed to set model.generation_config.pad_token_id.") from e

	# If tokenizer grew (PAD added), resize embeddings and keep model on the intended device.
	tok_len = int(len(tokenizer))
	emb = model.get_input_embeddings()
	if emb is not None and hasattr(emb, "weight") and hasattr(emb.weight, "shape"):
		emb_n = int(emb.weight.shape[0])
		if tok_len != emb_n:
			model.resize_token_embeddings(tok_len)
			model.to(device)


def _build_smol_summarize_loaders(
	*,
	args: Dict[str, Any],
	tokenizer: Any,
	model_eos_token: str,
	smol_kept_pos: Optional[List[int]],
	tokenizer_name: str,
) -> Tuple[Any, Any, Any]:
	"""
	Rebuild the same smol-summarize train/val split and collate_fn used by tools/run_experiment.py.
	"""
	from collections.abc import Mapping

	from tda_repr.data import get_dataset, make_dataloaders

	bundle = get_dataset(
		str(args["dataset"]),
		root=str(args.get("data_root", "data")),
		download=bool(args.get("download", True)),
		tokenizer_name=str(tokenizer_name),
	)
	if bundle.train is None:
		raise RuntimeError("Dataset 'smol-summarize' failed to load (train split is None).")

	base = bundle.train.shuffle(seed=int(args.get("nlp_subset_seed", 0)))
	if smol_kept_pos is not None:
		base = base.select([int(x) for x in smol_kept_pos])

	split = base.train_test_split(test_size=0.1, seed=int(args.get("nlp_subset_seed", 0)))
	bundle.train = split["train"]
	bundle.val = split["test"]
	bundle.test = None

	eos = model_eos_token

	def _qa_gen_collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
		if not batch:
			raise ValueError("Empty batch for generation collate.")
		if not isinstance(batch[0], dict):
			raise ValueError("Expected HF dataset items to be dicts.")

		def _prompt_target_from_messages(ex: dict) -> Optional[Tuple[str, str]]:
			for k in ("messages", "conversation", "conversations"):
				msgs = ex.get(k, None)
				if not isinstance(msgs, list) or not msgs:
					continue
				if not isinstance(msgs[0], dict):
					continue
				if not ("role" in msgs[0] and "content" in msgs[0]):
					continue
				last_assistant = None
				for i in range(len(msgs) - 1, -1, -1):
					role = str((msgs[i] or {}).get("role", "")).lower().strip()
					if role in ("assistant", "bot", "gpt"):
						last_assistant = i
						break
				if last_assistant is None:
					return None
				target = str((msgs[last_assistant] or {}).get("content", "")).strip()
				ctx = msgs[:last_assistant]
				parts: List[str] = []
				for m in ctx:
					if not isinstance(m, dict):
						continue
					role = str(m.get("role", "")).strip().lower()
					content = str(m.get("content", "")).strip()
					if not content:
						continue
					if role == "system":
						parts.append(f"System:\n{content}")
					elif role == "user":
						parts.append(f"User:\n{content}")
					elif role == "assistant":
						parts.append(f"Assistant:\n{content}")
					else:
						parts.append(f"{role.capitalize()}:\n{content}")
				prompt = "\n\n".join(parts).strip()
				prompt = (prompt + "\n\nAssistant:\n") if prompt else "Assistant:\n"
				return prompt, target
			return None

		prompts: List[str] = []
		targets: List[str] = []
		for ex in batch:
			mt = _prompt_target_from_messages(ex)
			if mt is None:
				raise ValueError(f"Could not infer prompt/target fields from keys={list(ex.keys())}")
			p, t = mt
			if not str(t).strip():
				continue
			prompts.append(str(p))
			targets.append(str(t))

		if not prompts:
			raise ValueError("All samples in batch had empty targets; cannot build generation batch.")

		prompt_texts = prompts
		target_texts = [t + eos for t in targets]
		full_texts = [p + t + eos for p, t in zip(prompts, targets)]

		prompt_enc = tokenizer(
			prompt_texts,
			add_special_tokens=False,
			padding=True,
			truncation=False,
			return_tensors="pt",
		)
		full_enc = tokenizer(
			full_texts,
			add_special_tokens=False,
			padding=True,
			truncation=False,
			return_tensors="pt",
		)
		target_enc = tokenizer(
			target_texts,
			add_special_tokens=False,
			padding=True,
			truncation=False,
			return_tensors="pt",
		)

		labels = full_enc["input_ids"].clone()
		labels[full_enc["attention_mask"] == 0] = -100
		prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()
		for i, pl in enumerate(prompt_lens):
			pl = int(pl)
			if pl > 0:
				labels[i, :pl] = -100

		out = dict(full_enc)
		out["labels"] = labels
		out["gen_input_ids"] = prompt_enc["input_ids"]
		out["gen_attention_mask"] = prompt_enc["attention_mask"]
		ref_labels = target_enc["input_ids"].clone()
		ref_labels[target_enc["attention_mask"] == 0] = -100
		out["gen_ref_labels"] = ref_labels
		return out

	bundle.collate_fn = _qa_gen_collate
	loaders = make_dataloaders(bundle, batch_size=int(args["batch_size"]), num_workers=0)
	train_loader = loaders["train"]
	val_loader = loaders["val"] or loaders["test"]
	if train_loader is None or val_loader is None:
		raise RuntimeError("Dataset did not provide required splits.")
	return bundle, train_loader, val_loader


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--run_dir", type=str, required=True, help="Existing run directory under runs/..., containing checkpoints/ and metrics.jsonl")
	ap.add_argument("--device", type=str, default="", help="Override device (e.g. cuda:0). If empty, uses device from checkpoint.")
	args_cli = ap.parse_args()

	root = _project_root()
	_ensure_matplotlib_cache(root)
	rexp = _load_run_experiment_module(root)

	run_dir = os.path.abspath(str(args_cli.run_dir))
	ckpt_dir = os.path.join(run_dir, "checkpoints")
	ckpt_path = os.path.join(ckpt_dir, "model_last.pt")
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path!r}. Run at least 1 epoch with save_models enabled.")

	obj = torch.load(ckpt_path, map_location="cpu")
	if not isinstance(obj, dict):
		raise TypeError("Checkpoint must be a dict.")
	payload = obj.get("payload", {}) or {}
	if not isinstance(payload, dict):
		raise TypeError("Checkpoint payload must be a dict.")

	ckpt_args = payload.get("args", None)
	if not isinstance(ckpt_args, dict):
		raise KeyError("Checkpoint payload is missing 'args' dict.")
	nlp_objective = str(payload.get("nlp_objective", ckpt_args.get("nlp_objective", ""))).lower().strip()
	if nlp_objective != "generation":
		raise ValueError(f"This resume script supports only nlp_objective='generation' (got {nlp_objective!r}).")

	model_name_or_path = str(payload.get("model_name_or_path", "")).strip()
	if not model_name_or_path:
		raise KeyError("Checkpoint payload is missing 'model_name_or_path'.")

	device_str = str(args_cli.device).strip() or str(payload.get("device", ckpt_args.get("device", "cpu"))).strip()
	device = torch.device(device_str)

	# Rebuild tokenizer + model.
	tokenizer = _build_generation_tokenizer(model_name_or_path)
	from transformers import AutoConfig, AutoModelForCausalLM

	if bool(ckpt_args.get("pretrained", True)):
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
	else:
		cfg = AutoConfig.from_pretrained(model_name_or_path)
		model = AutoModelForCausalLM.from_config(cfg)

	model.to(device)
	try:
		model.config.use_cache = False
	except Exception as e:
		raise RuntimeError("Failed to set model.config.use_cache=False.") from e

	_ensure_model_generation_padding(model, tokenizer, device=device)

	# Load model weights.
	state = obj.get("state_dict", None)
	if not isinstance(state, dict):
		raise KeyError("Checkpoint is missing 'state_dict'.")
	model.load_state_dict(state, strict=True)
	model.to(device)

	# Data / loaders.
	if str(ckpt_args.get("dataset", "")) != "smol-summarize":
		raise ValueError("This resume script currently supports only dataset='smol-summarize'.")
	smol_kept_pos = payload.get("smol_summarize_kept_pos", None)
	if smol_kept_pos is not None and not isinstance(smol_kept_pos, list):
		raise TypeError("payload['smol_summarize_kept_pos'] must be a list of ints or None.")
	eos_tok = str(getattr(tokenizer, "eos_token", "") or "")
	_bundle, train_loader, val_loader = _build_smol_summarize_loaders(
		args=ckpt_args,
		tokenizer=tokenizer,
		model_eos_token=eos_tok,
		smol_kept_pos=smol_kept_pos,
		tokenizer_name=model_name_or_path,
	)

	# Monitor / tracker.
	from tda_repr.training import BenchmarkSpec, ExperimentTracker, RepresentationMonitor, RepresentationMonitorConfig, RunStore, TrackerConfig

	monitor_cfg_d = payload.get("monitor", None)
	if not isinstance(monitor_cfg_d, dict):
		raise KeyError("Checkpoint payload is missing 'monitor' dict.")
	monitor_cfg = RepresentationMonitorConfig(**monitor_cfg_d)
	monitor = RepresentationMonitor(monitor_cfg)

	bench_metrics = payload.get("bench_metrics", None)
	if not isinstance(bench_metrics, list) or not bench_metrics:
		bench_metrics = ["loss_assistant_only", "ppl", "loss"]
	bench_specs = [
		BenchmarkSpec(
			name=f"{ckpt_args['dataset']}-val",
			dataloader_key="val",
			kind="generation",
			metrics=tuple(str(x) for x in bench_metrics),
			tokenizer=tokenizer,
		)
	]
	store = RunStore(run_dir, unique=False)
	tracker = ExperimentTracker(
		monitor=monitor,
		benchmarks=bench_specs,
		store=store,
		cfg=TrackerConfig(run_dir=store.run_dir, eval_every=1, max_eval_batches=(int(ckpt_args.get("max_val_batches", 0)) or None)),
	)

	# Optimizer / scheduler.
	params = [p for p in model.parameters() if p.requires_grad]
	if not params:
		raise RuntimeError("No trainable parameters found (requires_grad=False on all params).")
	opt = torch.optim.AdamW(params, lr=float(ckpt_args["lr"]), weight_decay=float(ckpt_args["weight_decay"]))
	opt_state = obj.get("optimizer", None)
	if not isinstance(opt_state, dict):
		raise KeyError("Checkpoint is missing 'optimizer' state.")
	opt.load_state_dict(opt_state)
	_optimizer_to_device(opt, device=device)

	lr_sched = None
	sched_d = payload.get("scheduler", {}) or {}
	if not isinstance(sched_d, dict):
		raise TypeError("payload['scheduler'] must be a dict.")
	if str(sched_d.get("kind", "none")) == "linear":
		warmup_steps = int(sched_d.get("warmup_steps", 0) or 0)
		total_steps = int(sched_d.get("total_steps", 1) or 1)

		def _lr_lambda(step: int) -> float:
			if warmup_steps > 0 and step < warmup_steps:
				return float(step) / float(max(1, warmup_steps))
			den = float(max(1, total_steps - warmup_steps))
			return max(0.0, float(total_steps - step) / den)

		lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
		sd = obj.get("lr_scheduler", None)
		if not isinstance(sd, dict):
			raise KeyError("Checkpoint payload expects 'lr_scheduler' state dict for linear scheduler.")
		lr_sched.load_state_dict(sd)

	last_epoch = int(obj.get("epoch", -1))
	global_step = int(obj.get("global_step", 0))
	next_epoch = last_epoch + 1
	try:
		cur_lr = float(opt.param_groups[0].get("lr", float("nan")))
	except Exception:
		cur_lr = float("nan")
	print(f"[ResumeOneEpoch] start epoch={next_epoch} global_step={global_step} lr={cur_lr}")

	# Match tools/run_experiment.py: optionally schedule heavy computations by epoch.
	dim2_on = _schedule(next_epoch, base=bool(ckpt_args.get("build_triangles", False)), every=int(ckpt_args.get("dim2_every", 0) or 0))
	q1_on = _schedule(next_epoch, base=bool(ckpt_args.get("compute_q1_spectra", False)), every=int(ckpt_args.get("q1_every", 0) or 0))
	monitor.cfg.build_triangles = bool(dim2_on)
	monitor.cfg.compute_q1_spectra = bool(q1_on)

	# Train exactly one epoch.
	t0_epoch = time.perf_counter()
	monitor.reset_epoch()
	model.train()
	t0_train = time.perf_counter()
	with monitor.attach(model):
		max_train_batches = int(ckpt_args.get("max_train_batches", 0) or 0)
		try:
			train_total = int(len(train_loader))
		except Exception:
			train_total = None
		if max_train_batches:
			train_total = int(min(int(max_train_batches), train_total)) if train_total is not None else int(max_train_batches)
		train_it = tqdm(
			enumerate(train_loader),
			total=train_total,
			desc=f"train e{next_epoch}",
			leave=False,
			dynamic_ncols=True,
			mininterval=1.0,
			miniters=10,
			smoothing=0.05,
		)
		for bi, batch in train_it:
			if max_train_batches and bi >= max_train_batches:
				break
			opt.zero_grad()
			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in (batch or {}).items()}
			fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
			out = model(**fwd)
			if not (hasattr(out, "loss") and out.loss is not None):
				raise RuntimeError("Model forward did not return .loss for generation batch.")
			loss = out.loss
			if not torch.isfinite(loss).all():
				raise RuntimeError("Non-finite loss (NaN/Inf) encountered.")
			loss.backward()
			torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
			opt.step()
			global_step += 1
			if lr_sched is not None:
				lr_sched.step()
			try:
				lr_now = float(opt.param_groups[0].get("lr", float("nan")))
			except Exception:
				lr_now = float("nan")
			train_it.set_postfix(loss=float(loss.detach().item()), lr=lr_now, step=int(global_step))
			attn = batch.get("attention_mask", None)
			monitor.collect("train", attention_mask=attn)
	t_train = float(time.perf_counter() - t0_train)

	# Val pass for repr collection (no grad).
	model.eval()
	t0_val = time.perf_counter()
	with torch.no_grad(), monitor.attach(model):
		max_val_batches = int(ckpt_args.get("max_val_batches", 0) or 0)
		try:
			val_total = int(len(val_loader))
		except Exception:
			val_total = None
		if max_val_batches:
			val_total = int(min(int(max_val_batches), val_total)) if val_total is not None else int(max_val_batches)
		val_it = tqdm(
			enumerate(val_loader),
			total=val_total,
			desc=f"val e{next_epoch}",
			leave=False,
			dynamic_ncols=True,
			mininterval=1.0,
			miniters=10,
			smoothing=0.05,
		)
		for bi, batch in val_it:
			if max_val_batches and bi >= max_val_batches:
				break
			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in (batch or {}).items()}
			fwd = {k: v for k, v in batch.items() if k not in ("gen_input_ids", "gen_attention_mask", "gen_ref_labels")}
			out = model(**fwd)
			try:
				vloss = out.loss if hasattr(out, "loss") else None
				if isinstance(vloss, torch.Tensor):
					val_it.set_postfix(loss=float(vloss.detach().item()))
			except Exception:
				pass
			attn = batch.get("attention_mask", None)
			monitor.collect("val", attention_mask=attn)
	t_val = float(time.perf_counter() - t0_val)

	out_rec = tracker.on_epoch_end(
		next_epoch,
		model=model,
		dataloaders={"train": train_loader, "val": val_loader, "test": None},
		loss_fn=None,
		extra={"train_s": t_train, "val_s": t_val, "dim2_on": bool(dim2_on), "q1_on": bool(q1_on)},
	)

	bench = out_rec.get("bench", {}) or {}
	main_key = f"{ckpt_args['dataset']}-val"
	mm = (bench.get(main_key, {}) or {})
	score, score_name = _score_for_best(mm)

	# Save updated last checkpoint.
	os.makedirs(ckpt_dir, exist_ok=True)
	last_obj: Dict[str, Any] = {
		"epoch": int(next_epoch),
		"global_step": int(global_step),
		"state_dict": model.state_dict(),
		"optimizer": opt.state_dict(),
		"payload": dict(payload),
	}
	if lr_sched is not None:
		last_obj["lr_scheduler"] = lr_sched.state_dict()
	torch.save(last_obj, ckpt_path)

	# Update best checkpoint if improved.
	best_path = os.path.join(ckpt_dir, "model_best_main.pt")
	best_score = -float("inf")
	if os.path.exists(best_path):
		try:
			best_obj = torch.load(best_path, map_location="cpu")
			best_pl = (best_obj or {}).get("payload", {}) if isinstance(best_obj, dict) else {}
			best_score = float((best_pl or {}).get("main_metric", -float("inf")))
		except Exception:
			best_score = -float("inf")
	if score > best_score:
		best_payload = {
			"kind": "best_main",
			"main_metric": float(score),
			"main_metric_name": str(score_name),
			"task": str(ckpt_args.get("task", "nlp")),
			"dataset": str(ckpt_args.get("dataset", "")),
			"model": str(ckpt_args.get("model", "")),
			"finetune": str(payload.get("finetune", "")),
			"epoch": int(next_epoch),
		}
		torch.save({"epoch": int(next_epoch), "state_dict": model.state_dict(), "payload": best_payload}, best_path)

	# Rewrite progress figures to include the new epoch.
	if bool(ckpt_args.get("live_plots", True)):
		rexp._rewrite_progress_figures(run_dir, dataset=str(ckpt_args.get("dataset", "")))

	_cleanup_memory(device=device_str)
	print(
		f"[ResumeOneEpoch] run_dir={run_dir} epoch={next_epoch} "
		f"val_loss_assistant_only={mm.get('loss_assistant_only', None)} val_ppl={mm.get('ppl', None)} "
		f"score={score}({score_name}) total_s={time.perf_counter() - t0_epoch:.2f}"
	)


if __name__ == "__main__":
	main()

