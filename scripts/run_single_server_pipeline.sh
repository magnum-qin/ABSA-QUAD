#!/usr/bin/env bash
# Single-server runner for the post-V20 experiment scripts.
#
# Usage:
#   scripts/run_single_server_pipeline.sh <task> [--augment] [--train] [--infer] [--eval]
#
# Tasks:
#   jpn_v21
#   rus_deepseek_aug
#   ukr_deepseek_aug
#   eng_laptop_deepseek_aug
#   eng_laptop_no_aug_large
#
# Environment:
#   PYTHON_BIN=/path/to/python   Optional. Defaults to python.
#   SEEDS="42 123 456 789 1024" Optional.
#   WORKERS=8                   Optional for --augment.
#   DEEPSEEK_API_KEY=...        Required only for --augment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 123 456 789 1024}"
WORKERS="${WORKERS:-8}"

usage() {
  sed -n '1,28p' "$0" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TASK="$1"
shift

DO_AUGMENT=0
DO_TRAIN=0
DO_INFER=0
DO_EVAL=0

if [[ $# -eq 0 ]]; then
  DO_TRAIN=1
  DO_INFER=1
  DO_EVAL=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --augment) DO_AUGMENT=1 ;;
    --train) DO_TRAIN=1 ;;
    --infer) DO_INFER=1 ;;
    --eval) DO_EVAL=1 ;;
    --all) DO_AUGMENT=1; DO_TRAIN=1; DO_INFER=1; DO_EVAL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

CONFIG=""
TRAIN_FILE=""
DEV_FILE=""
AUG_INPUT=""
AUG_OUTPUT=""
SOURCE_LANG=""
PIVOTS=()
SAVE_PREFIX=""
OUTPUT=""
VOTE_THRESHOLD=""
SPAN_TOPK="0.0"

case "$TASK" in
  jpn_v21)
    CONFIG="configs/config_jpn_v16_aug.yaml"
    TRAIN_FILE="data/subtask_3/jpn/jpn_hotel_train_augmented.jsonl"
    DEV_FILE="data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl"
    AUG_INPUT="data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl"
    AUG_OUTPUT="$TRAIN_FILE"
    SOURCE_LANG="jpn"
    PIVOTS=("Korean")
    SAVE_PREFIX="jpn_v21_deepseek_aug_seed"
    OUTPUT="predictions/ensemble_jpn_v21_deepseek_aug_5seed.jsonl"
    VOTE_THRESHOLD="4"
    ;;
  rus_deepseek_aug)
    CONFIG="configs/config_rus_rest_deepseek_aug.yaml"
    TRAIN_FILE="data/subtask_3/rus/rus_restaurant_train_task3_augmented.jsonl"
    DEV_FILE="data/subtask_3/rus/rus_restaurant_dev_task3.jsonl"
    AUG_INPUT="data/subtask_3/rus/rus_restaurant_train_alltasks.jsonl"
    AUG_OUTPUT="$TRAIN_FILE"
    SOURCE_LANG="rus"
    PIVOTS=("English" "Polish")
    SAVE_PREFIX="rus_rest_deepseek_aug_seed"
    OUTPUT="predictions/ensemble_rus_rest_deepseek_aug_5seed.jsonl"
    VOTE_THRESHOLD="4"
    SPAN_TOPK="0.10"
    ;;
  ukr_deepseek_aug)
    CONFIG="configs/config_ukr_rest_deepseek_aug.yaml"
    TRAIN_FILE="data/subtask_3/ukr/ukr_restaurant_train_task3_augmented.jsonl"
    DEV_FILE="data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl"
    AUG_INPUT="data/subtask_3/ukr/ukr_restaurant_train_alltasks.jsonl"
    AUG_OUTPUT="$TRAIN_FILE"
    SOURCE_LANG="ukr"
    PIVOTS=("English" "Polish")
    SAVE_PREFIX="ukr_rest_deepseek_aug_seed"
    OUTPUT="predictions/ensemble_ukr_rest_deepseek_aug_5seed.jsonl"
    VOTE_THRESHOLD="3"
    ;;
  eng_laptop_deepseek_aug)
    CONFIG="configs/config_eng_laptop_deepseek_aug.yaml"
    TRAIN_FILE="data/subtask_3/eng/eng_laptop_train_task3_augmented.jsonl"
    DEV_FILE="data/subtask_3/eng/eng_laptop_dev_task3.jsonl"
    AUG_INPUT="data/subtask_3/eng/eng_laptop_train_alltasks.jsonl"
    AUG_OUTPUT="$TRAIN_FILE"
    SOURCE_LANG="eng"
    PIVOTS=("French" "Spanish")
    SAVE_PREFIX="eng_laptop_deepseek_aug_seed"
    OUTPUT="predictions/ensemble_eng_laptop_deepseek_aug_5seed.jsonl"
    VOTE_THRESHOLD="4"
    ;;
  eng_laptop_no_aug_large)
    CONFIG="configs/config_eng_laptop_no_aug_large.yaml"
    TRAIN_FILE="data/subtask_3/eng/eng_laptop_train_alltasks.jsonl"
    DEV_FILE="data/subtask_3/eng/eng_laptop_dev_task3.jsonl"
    SAVE_PREFIX="eng_laptop_no_aug_large_seed"
    OUTPUT="predictions/ensemble_eng_laptop_no_aug_large_5seed.jsonl"
    VOTE_THRESHOLD="4"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    usage
    exit 2
    ;;
esac

read -r -a SEED_ARRAY <<< "$SEEDS"
SEED_DIRS=()
for seed in "${SEED_ARRAY[@]}"; do
  SEED_DIRS+=("checkpoints/${SAVE_PREFIX}${seed}")
done

mkdir -p logs predictions

if [[ "$DO_AUGMENT" -eq 1 ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is required for --augment" >&2
    exit 1
  fi
  if [[ -z "$AUG_INPUT" || -z "$AUG_OUTPUT" ]]; then
    echo "Task $TASK does not define augmentation inputs" >&2
    exit 1
  fi
  "$PYTHON_BIN" scripts/augment_data_deepseek.py \
    --input "$AUG_INPUT" \
    --output "$AUG_OUTPUT" \
    --source_lang "$SOURCE_LANG" \
    --pivot_langs "${PIVOTS[@]}" \
    --workers "$WORKERS" \
    2>&1 | tee "logs/${TASK}_augment.log"
fi

if [[ "$DO_TRAIN" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/train_multi_seed.py \
    --config "$CONFIG" \
    --seeds "${SEED_ARRAY[@]}" \
    --save_prefix "$SAVE_PREFIX" \
    --train_file "$TRAIN_FILE" \
    2>&1 | tee "logs/${TASK}_train.log"
fi

if [[ "$DO_INFER" -eq 1 ]]; then
  infer_args=(
    scripts/inference.py
    --config "$CONFIG"
    --multi_seed_ensemble
    --seed_dirs "${SEED_DIRS[@]}"
    --test_file "$DEV_FILE"
    --output "$OUTPUT"
    --span_topk_threshold "$SPAN_TOPK"
  )
  if [[ -n "$VOTE_THRESHOLD" ]]; then
    infer_args+=(--vote_threshold "$VOTE_THRESHOLD")
  fi
  "$PYTHON_BIN" "${infer_args[@]}" 2>&1 | tee "logs/${TASK}_infer.log"
fi

if [[ "$DO_EVAL" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/official_eval.py \
    -p "$OUTPUT" \
    -g "$DEV_FILE" \
    -t 3 \
    2>&1 | tee "logs/${TASK}_eval.log"
fi
