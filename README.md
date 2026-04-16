## tda-repr

Topological and spectral analysis toolkit for neural network representations.

`tda-repr` helps you monitor hidden-layer geometry during training and compare it to benchmark quality metrics (loss/accuracy/F1), with reproducible logs and plots.

It supports iterative research workflows with explicit run artifacts: configuration metadata, per-epoch structured logs, progress figures, checkpoint snapshots, and correlation reports. This makes comparisons between datasets, architectures, and fine-tune regimes reproducible and auditable.

`tda_repr/` is the library. `tools/` contains scripts used in the thesis/repro.

Run commands from the repo root. The `tools/` scripts are for the repo checkout (not the PyPI package).

### Install (from source)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Run an experiment

Interactive:

```bash
python -m tools.run_experiment --interactive --interactive_ui tui
```

Non-interactive:

```bash
python -m tools.run_experiment \
  --no-interactive \
  --task cv \
  --dataset cifar10 \
  --model resnet18 \
  --device cpu \
  --finetune full \
  --epochs 20 \
  --batch_size 128 \
  --download
```

Outputs go to `runs/exp_*/` (`meta.json`, `metrics.jsonl`, `figures/`, `checkpoints/`, `correlations_report/`, `analysis/`).

### Analysis after successful run

Correlation report:

```bash
python -m tools.correlation_report --run_dir runs/<run_dir>
```

Embedding quality / layer selection:

```bash
python -m tools.evaluate_embeddings --run_dir runs/<run_dir> --checkpoint best_main --split val --device cpu --download --skip_existing
```

Early-stop sweep (offline):

```bash
python -m tools.repr_early_stop_sweep --roots runs --skip_existing
```

### Reproducibility

Zenodo [ https://doi.org/10.5281/zenodo.20114914 ] archive (`saved_runs/`) -> figures + 3 case tables:

```bash
./reproduction_cases.sh
```

Full regeneration (`runs/`) -> all remaining tables:

```bash
./reproduction_runs.sh
```
