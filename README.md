# tda-repr

Topological and spectral analysis toolkit for neural network representations.

`tda-repr` helps you monitor hidden-layer geometry during training and compare it
to benchmark quality metrics (loss/accuracy/F1), with reproducible logs and plots.

The package supports iterative research workflows with explicit run artifacts:
configuration metadata, per-epoch structured logs, progress figures, checkpoint
snapshots, and correlation reports. This makes comparisons between datasets,
architectures, and fine-tune regimes reproducible and auditable.

## Methodological Positioning

Traditional monitoring (loss/accuracy/F1) is often enough for pure engineering
optimization, but it can hide critical behavior changes inside hidden layers.
`tda-repr` complements benchmark metrics with geometric and topological layer
signals to make internal dynamics observable across epochs.

This is especially useful when:

- validation quality plateaus but training still changes;
- different fine-tune modes produce similar scores but different internal behavior;
- you need evidence-based layer selection for monitoring or unfreezing;
- you want a representation-based early-stop signal for post-hoc analysis.

## Key Features

- Track layer representations on train/val/test splits.
- Compute graph/Hodge/persistent characteristics per layer and epoch.
- Compute MTopDiv distance between stages (for example, train vs val).
- Run interactive experiments from terminal (TUI) or non-interactive CLI.
- Export run artifacts: JSONL logs, progress figures, correlation report.

## Practical Workflow

Typical usage is iterative and looks like this:

1. Start from a short baseline run with a small number of layers and epochs to verify
  that data, model, and logging are configured correctly.
2. Expand monitored layers only after baseline metrics are stable, because excessive
  layer coverage increases runtime and can make analysis noisy.
3. Compare fine-tune modes (`full`, `linear_probe`, selective unfreezing) on the same
  dataset/model pair to separate optimization effects from representation effects.
4. Treat early-stop as an analysis signal, not an automatic controller: it indicates a
  structural change point, but should be interpreted together with validation curves.
5. Use correlation reports for ranking relationships, then validate top candidates by
  re-running with the same seed/config to check reproducibility.

## Installation

### Editable install (development)

```bash
pip install -e .
```

### Install with all optional extras

```bash
pip install -e .[all]
```

### Main extras

- `.[nlp]` for text datasets/models (`transformers`, `datasets`)
- `.[topology]` for advanced topology backends (`gudhi`, `ripser`)
- `.[medical]` for MedMNIST support
- `.[dev]` for tests/lint/release tooling

## Environment Sanity Check

Before long runs, verify that CLI entrypoints and PyTorch device detection work:

```bash
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
tda-repr-layers --model resnet18 --leaf_only
```

If CUDA is unavailable, run with `--device cpu` explicitly.

## Quick Start

### 1) Inspect available layers

```bash
tda-repr-layers --model efficientnet_b0 --leaf_only
```

Strict include validation:

```bash
tda-repr-layers --model efficientnet_b0 --include "features.0.0,features.7,avgpool,classifier.1" --strict
```

### 2) Run an interactive experiment (TUI)

```bash
python tools/run_experiment.py --interactive --interactive_ui tui
```

In TUI you can:

- select task/dataset/model/device;
- choose monitor layers with readable names;
- choose benchmark metrics;
- configure fine-tune mode and early-stop signals;
- go back to previous steps (`b` where supported).

### 3) Run non-interactive experiment (CLI)

```bash
python tools/run_experiment.py \
  --no-interactive \
  --task cv \
  --dataset cifar10 \
  --model resnet18 \
  --device cuda:0 \
  --finetune full \
  --epochs 20 \
  --batch_size 256 \
  --download \
  --layer_include "conv1,layer1.*,layer2.*,layer3.*,layer4.*,avgpool,fc"
```

## Common Recipes

### Recipe A: Fine-tune only selected modules

Useful when full fine-tune is too expensive or noisy.

```bash
python tools/run_experiment.py \
  --no-interactive \
  --task cv \
  --dataset imagenette \
  --model efficientnet_b0 \
  --device cuda:0 \
  --finetune selected_layers \
  --train_layers "features.0.0,features.7,features.8.0,avgpool,classifier.1" \
  --layer_include "features.0.0,features.7,features.8.0,avgpool,classifier.1" \
  --epochs 20 \
  --batch_size 256 \
  --download
```

### Recipe B: Parameter-pattern based fine-tune

```bash
python tools/run_experiment.py \
  --no-interactive \
  --task cv \
  --dataset cifar10 \
  --model resnet18 \
  --device cuda:0 \
  --finetune named_patterns \
  --train_patterns "layer4.*,fc.*" \
  --train_exclude_patterns "*.bn*" \
  --epochs 20 \
  --batch_size 256 \
  --download
```

### Recipe C: Multi-signal early-stop monitor (signal only)

`--early_stop` now emits an early-stop signal event and figure, but does **not**
terminate training automatically. This is intended for post-hoc analysis and fair
comparison against fixed-epoch training.

```bash
python tools/run_experiment.py \
  --no-interactive \
  --task cv \
  --dataset cifar10 \
  --model efficientnet_b0 \
  --device cuda:0 \
  --finetune full \
  --layer_include "features.0.0,features.7,features.8.0,avgpool" \
  --early_stop \
  --early_stop_signals "features.0.0:mtopdiv_train_val:max;features.7:mtopdiv_train_val:max;features.8.0:beta1_persistent_est:max;avgpool:beta1_persistent_est:max" \
  --early_stop_aggregate any \
  --early_stop_patience 4 \
  --early_stop_start_epoch 3 \
  --epochs 20 \
  --batch_size 256 \
  --download
```

## Fine-Tune Modes

`--finetune` supports:

- `full` - all parameters trainable.
- `linear_probe` - classifier head only.
- `last_n_params` - unfreeze last N parameters (`--last_n_params`).
- `named_prefixes` - unfreeze parameters by name prefixes (`--train_prefixes`).
- `named_patterns` - unfreeze by glob patterns (`--train_patterns` / `--train_exclude_patterns`).
- `selected_layers` - unfreeze explicit module list (`--train_layers`).

If `--finetune_list` is provided (CSV), modes run sequentially in one command.

## Output Artifacts

Each run writes into `runs/<experiment_name>/`:

- `meta.json` - run setup and metadata
- `metrics.jsonl` - per-epoch structured logs
- `figures/fig_quality_progress.png` - benchmark metrics over epochs
- `figures/fig_repr_progress.png` - representation metrics over epochs
- `figures/fig_early_stop_metric.png` - early-stop signal curve (if enabled)

### Correlation report

```bash
python tools/correlation_report.py --run_dir runs/<your_run_dir>
```

Outputs:

- `all_pairs.csv`
- `top_pairs.csv`
- `top_pairs.png`

### Paper/report tables (one-shot)

Build all paper-style tables from existing `runs/*` artifacts (no training rerun):

```bash
python tools/paper.py tables
```

This writes LaTeX row snippets and paper tables into:

- `paper/analysis_tables/` (depth dynamics, arch comparison, runs index)
- `paper/analysis_tables_ftb/` (paper tables: correlation summary, layer selection, early stopping)

### Checkpoints and best epoch

When `--save_models` is enabled (default), each run also stores:

- `checkpoints/model_best_main.pt` - best validation model by main metric (`f1_macro` or `accuracy`)
- `checkpoints/model_early_signal.pt` - model snapshot at representation-based early-stop signal (if triggered)

## Troubleshooting

- `SelectionValidationError` on `--layer_include` or `--train_patterns`: run `tda-repr-layers --model <model> --leaf_only` and adjust names/patterns.
- Empty/invalid early-stop signal: make sure metric keys are topo/spectral (`beta*`, `hodge*`, `persistent*`, `mtopdiv*`, `gudhi*`, `graph_*`) and selected layer is monitored.
- Dataset errors: pass `--download` for first run and confirm dataset key (`cifar10`, `imagenette`, `pathmnist`, etc.).
- No trainable parameters: check `--finetune` mode arguments (`--train_layers`, `--train_prefixes`, `--train_patterns`, `--last_n_params`).
- Slow experiments: start with lower `--epochs`, smaller `--max_train_batches`, and fewer monitored layers.

## Typical Error Messages


| Error message pattern                              | Meaning                                            | Action                                                                                |
| -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `SelectionValidationError` with unmatched patterns | Selection rules did not match model names          | Print available layers/params and refine include/exclude patterns                     |
| `Early stopping requires at least one signal`      | Early-stop enabled without valid signal config     | Set `--early_stop_signals` or pair of `--early_stop_layer` and `--early_stop_metric`  |
| `Early-stop metric ... must be topology/spectral`  | Selected metric key is not from supported family   | Use keys starting with `beta`, `hodge`, `persistent`, `mtopdiv`, `gudhi`, or `graph_` |
| `No trainable parameters for finetune mode`        | Current fine-tune config freezes everything        | Recheck mode-specific arguments and selected layers/patterns                          |
| `Dataset did not provide required splits`          | Loaded dataset does not expose train/val structure | Use a supported dataset key and ensure expected splits are available                  |


## Resource and Reproducibility Guidance

Compute and memory costs depend on model size, monitored layer count, and graph/
topology settings. For stable and efficient iteration:

- start from short baseline runs, then increase scope gradually;
- keep seed, dataset, and monitor config fixed during comparative studies;
- interpret single-run correlations as hypotheses and confirm by reruns;
- prioritize trend consistency over isolated epoch peaks.

## Python API Example

Minimal layer-hook example:

```python
import torch
from tda_repr import LayerTaps, get_model_info

info = get_model_info("resnet18")
model = info.model.eval()
x = torch.rand(4, 3, 224, 224)

layer_names = ["layer1.0", "layer4.1", "fc"]
with LayerTaps(model, layer_names) as taps:
    _ = model(x)

for name, tensor in taps.outputs.items():
    print(name, tuple(tensor.shape))
```

## Release Checklist

```bash
pip install -e .[dev]
pytest
python -m build
python -m twine check dist/*
```

Optional smoke runs:

```bash
python -m tests.smoke_model_layers
python -m tests.smoke_train_monitor
```

## Notes

- Default package on PyPI includes only `tda_repr*` code (not `runs/`, `data/`, `tests/`, or dot-directories).
