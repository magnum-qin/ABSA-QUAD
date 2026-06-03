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

run_task() {
  local task="$1"
  local cfg="$2"
  local dev="$3"
  local test="$4"
  local prefix="$5"
  local vote="$6"
  local topk="$7"

  local seed_dirs=(
    "checkpoints/${prefix}42"
    "checkpoints/${prefix}123"
    "checkpoints/${prefix}456"
    "checkpoints/${prefix}789"
    "checkpoints/${prefix}1024"
  )

  for d in "${seed_dirs[@]}"; do
    test -f "$d/checkpoint_best.pt"
  done

  "$PY" -u scripts/inference.py \
    --multi_seed_ensemble --seed_dirs "${seed_dirs[@]}" \
    --config "$cfg" --test_file "$dev" \
    --output "predictions/${TAG}/${task}_dev.jsonl" \
    --vote_threshold "$vote" --span_topk_threshold "$topk" --batch_size 16 \
    2>&1 | tee "logs/${TAG}/${task}_dev_predict.log"

  "$PY" scripts/official_eval.py \
    -p "predictions/${TAG}/${task}_dev.jsonl" \
    -g "$dev" -t 3 \
    2>&1 | tee "logs/${TAG}/${task}_dev_eval.log"

  "$PY" -u scripts/inference.py \
    --multi_seed_ensemble --seed_dirs "${seed_dirs[@]}" \
    --config "$cfg" --test_file "$test" \
    --output "submission/${TAG}/${task}_test.jsonl" \
    --vote_threshold "$vote" --span_topk_threshold "$topk" --batch_size 16 \
    2>&1 | tee "logs/${TAG}/${task}_test_predict.log"
}

run_task jpn_hotel \
  final_configs/jpn_hotel_v16_no_aug.yaml \
  data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
  data/subtask_3/jpn/jpn_hotel_test_task3.jsonl \
  final_jpn_hotel_v16_no_aug_seed \
  4 0.0

run_task rus_restaurant \
  final_configs/rus_restaurant_v15.yaml \
  data/subtask_3/rus/rus_restaurant_dev_task3.jsonl \
  data/subtask_3/rus/rus_restaurant_test_task3.jsonl \
  final_rus_restaurant_v15_seed \
  4 0.10

run_task ukr_restaurant \
  final_configs/ukr_restaurant_v13b.yaml \
  data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl \
  data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl \
  final_ukr_restaurant_v13b_seed \
  3 0.0

wc -l "predictions/${TAG}/"*"_dev.jsonl" "submission/${TAG}/"*"_test.jsonl" | tee "logs/${TAG}/jpn_rus_ukr_counts.log"
