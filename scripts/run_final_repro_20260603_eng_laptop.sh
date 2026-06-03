#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/g24tka12/ABSA-QUAD}"
cd "$ROOT"

if [[ -z "${PY:-}" ]]; then
  if [[ -x /home/g24tka12/miniforge3/envs/absa/bin/python ]]; then
    PY="/home/g24tka12/miniforge3/envs/absa/bin/python"
  elif [[ -x /tmp/g24tka12_absa/miniforge3/envs/absa/bin/python ]]; then
    PY="/tmp/g24tka12_absa/miniforge3/envs/absa/bin/python"
  else
    PY="python3"
  fi
fi

CONDA_PREFIX_DIR="$(dirname "$(dirname "$PY")")"
export CONDA_PREFIX="$CONDA_PREFIX_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"

TAG=final_repro_20260603
mkdir -p "logs/${TAG}" "predictions/${TAG}" "submission/${TAG}"

SEED_DIRS=(
  checkpoints/final_eng_laptop_v5_bt_seed42
  checkpoints/final_eng_laptop_v5_bt_seed123
  checkpoints/final_eng_laptop_v5_bt_seed456
  checkpoints/final_eng_laptop_v5_bt_seed789
  checkpoints/final_eng_laptop_v5_bt_seed1024
)

for d in "${SEED_DIRS[@]}"; do
  test -f "$d/checkpoint_best.pt"
done

CFG=final_configs/eng_laptop_v5_bt.yaml
DEV=data/subtask_3/eng/eng_laptop_dev_task3.jsonl
TEST=data/subtask_3/eng/eng_laptop_test_task3.jsonl

"$PY" -u scripts/inference.py \
  --multi_seed_ensemble --seed_dirs "${SEED_DIRS[@]}" \
  --config "$CFG" --test_file "$DEV" \
  --output "predictions/${TAG}/eng_laptop_dev.jsonl" \
  --vote_threshold 3 --span_topk_threshold 0.0 --dual_layer --batch_size 16 \
  2>&1 | tee "logs/${TAG}/eng_laptop_dev_predict.log"

"$PY" scripts/official_eval.py \
  -p "predictions/${TAG}/eng_laptop_dev.jsonl" \
  -g "$DEV" -t 3 \
  2>&1 | tee "logs/${TAG}/eng_laptop_dev_eval.log"

"$PY" -u scripts/inference.py \
  --multi_seed_ensemble --seed_dirs "${SEED_DIRS[@]}" \
  --config "$CFG" --test_file "$TEST" \
  --output "submission/${TAG}/eng_laptop_test.jsonl" \
  --vote_threshold 3 --span_topk_threshold 0.0 --dual_layer --batch_size 16 \
  2>&1 | tee "logs/${TAG}/eng_laptop_test_predict.log"

wc -l "predictions/${TAG}/eng_laptop_dev.jsonl" "submission/${TAG}/eng_laptop_test.jsonl" | tee "logs/${TAG}/eng_laptop_counts.log"
