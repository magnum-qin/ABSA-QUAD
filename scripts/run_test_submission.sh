#!/usr/bin/env bash
# Frozen-submission entry point. Default is read-only audit; reproduction writes elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-audit}"
SEEDS=(42 123 456 789 1024)
OUT_DIR="${OUT_DIR:-submission/reproduced}"
TASKS="${TASKS:-eng_laptop rus_restaurant tat_restaurant ukr_restaurant zho_laptop zho_restaurant}"

if [[ "$MODE" == "audit" ]]; then
  python scripts/audit_final_submission.py
  exit 0
fi
if [[ "$MODE" != "reproduce" ]]; then
  echo "Usage: $0 [audit|reproduce]" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

predict() {
  local task="$1" prefix="$2" config="$3" test_file="$4" vote="$5" topk="$6"
  local seed_dirs=()
  local seed checkpoint
  for seed in "${SEEDS[@]}"; do
    checkpoint="checkpoints/${prefix}${seed}/checkpoint_best.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      return 1
    fi
    seed_dirs+=("checkpoints/${prefix}${seed}")
  done
  conda run -n absa --no-capture-output python -u scripts/inference.py \
    --multi_seed_ensemble --seed_dirs "${seed_dirs[@]}" \
    --config "$config" --test_file "$test_file" \
    --output "$OUT_DIR/${task}_test.jsonl" \
    --vote_threshold "$vote" --span_topk_threshold "$topk" --batch_size 16
}

for task in $TASKS; do
  case "$task" in
    eng_laptop)
      predict "$task" v23_eng_laptop_deepseek_seed configs/pipeline_eng_laptop_v23_deepseek.yaml data/subtask_3/eng/eng_laptop_test_task3.jsonl 4 0.0 ;;
    rus_restaurant)
      predict "$task" v23_rus_restaurant_deepseek_gen4k_seed configs/pipeline_rus_restaurant_v23_deepseek.yaml data/subtask_3/rus/rus_restaurant_test_task3.jsonl 3 0.0 ;;
    tat_restaurant)
      predict "$task" v23_tat_restaurant_qwen_seed configs/pipeline_tat_restaurant_v23_deepseek.yaml data/subtask_3/tat/tat_restaurant_test_task3.jsonl 3 0.0 ;;
    ukr_restaurant)
      predict "$task" v23_ukr_restaurant_deepseek_seed configs/pipeline_ukr_restaurant_v23_deepseek.yaml data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl 3 0.0 ;;
    zho_laptop)
      predict "$task" final_zho_laptop_v16_seed configs/pipeline_zho_laptop_v16.yaml data/subtask_3/zho/zho_laptop_test_task3.jsonl 4 0.1 ;;
    zho_restaurant)
      predict "$task" v23_zho_restaurant_deepseek_real_seed configs/pipeline_zho_restaurant_v23_qwen.yaml data/subtask_3/zho/zho_restaurant_test_task3.jsonl 3 0.0 ;;
    eng_restaurant|jpn_hotel)
      echo "$task is frozen by prediction hash only; its historical five checkpoints are not local." >&2
      exit 1 ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
done

echo "Reproduced available frozen tasks under $OUT_DIR (submission/final was not modified)."
