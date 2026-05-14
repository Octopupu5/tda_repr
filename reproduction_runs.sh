ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate

rm -rf "runs/reproduction_select" "runs/reproduction_eval" "reproduction/tables" "reproduction/early_stop_rules.json"
mkdir -p "runs/reproduction_select" "runs/reproduction_eval" "reproduction"

run_one () {
  root="$1"
  task="$2"
  dataset="$3"
  model="$4"
  seed="$5"
  epochs="$6"
  shift 6
  extra_args=("$@")

  runs_base="${root}/exp"
  python -m tools.run_experiment \
    --no-interactive \
    --runs_base "$runs_base" \
    --task "$task" \
    --dataset "$dataset" \
    --model "$model" \
    --device "${DEVICE:-cpu}" \
    --pretrained \
    --finetune full \
    --epochs "$epochs" \
    --batch_size 16 \
    --download \
    --compute_mtopdiv \
    --compute_q1_spectra \
    --seed "$seed" \
    "${extra_args[@]}" \
    1>&2

  ls -1dt "${runs_base}"_* | head -n 1
}

run_selection_seed () {
  seed="$1"

  rd="$(run_one "runs/reproduction_select" cv mnist mlp "$seed" 20)"
  python -m tools.evaluate_embeddings --run_dir "$rd" --checkpoint best_main --split val --device "${DEVICE:-cpu}" --download --skip_existing

  for model in resnet18 efficientnet_b0 convnext_tiny; do
    for ds in cifar10 bloodmnist imagenette; do
      rd="$(run_one "runs/reproduction_select" cv "$ds" "$model" "$seed" 20)"
      python -m tools.evaluate_embeddings --run_dir "$rd" --checkpoint best_main --split val --device "${DEVICE:-cpu}" --download --skip_existing
    done
  done

  for ds in sst2 trec6; do
    rd="$(run_one "runs/reproduction_select" nlp "$ds" distilbert "$seed" 20)"
    python -m tools.evaluate_embeddings --run_dir "$rd" --checkpoint best_main --split test --device "${DEVICE:-cpu}" --download --skip_existing
  done

  run_one "runs/reproduction_select" nlp smol-summarize smollm2-135m "$seed" 20 \
    --nlp_max_train_examples 50000 \
    --nlp_max_val_examples 10000 \
    --nlp_max_test_examples 10000 \
    >/dev/null 2>&1
}

run_eval_seed () {
  seed="$1"

  run_one "runs/reproduction_eval" cv mnist mlp "$seed" 20 >/dev/null 2>&1

  for model in resnet18 efficientnet_b0 convnext_tiny; do
    for ds in cifar10 bloodmnist imagenette; do
      run_one "runs/reproduction_eval" cv "$ds" "$model" "$seed" 20 >/dev/null 2>&1
    done
  done

  for ds in sst2 trec6; do
    run_one "runs/reproduction_eval" nlp "$ds" distilbert "$seed" 20 >/dev/null 2>&1
  done

  run_one "runs/reproduction_eval" nlp smol-summarize smollm2-135m "$seed" 20 \
    --nlp_max_train_examples 50000 \
    --nlp_max_val_examples 10000 \
    --nlp_max_test_examples 10000 \
    >/dev/null 2>&1
}

for seed in 0 1 2; do
  run_selection_seed "$seed"
done

python -m tools.repr_early_stop_sweep --roots "runs/reproduction_select" --skip_existing
python -m tools.aggregate.select_early_stop_rules --roots "runs/reproduction_select" --out_json "reproduction/early_stop_rules.json"

for seed in 3 4 5; do
  run_eval_seed "$seed"
done

python -m tools.aggregate.apply_early_stop_rules --roots "runs/reproduction_eval" --rules_json "reproduction/early_stop_rules.json" --skip_existing
python -m tools.reproduce_tables --runs-root "runs/reproduction_select" --early-stop-root "runs/reproduction_eval" --out-dir "reproduction/tables"
