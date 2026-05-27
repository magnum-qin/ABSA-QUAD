#!/bin/bash
# V16 ensemble sweep for the 3 missing datasets just trained.

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"
PRED_DIR=predictions/v16_missing_sweep
mkdir -p $PRED_DIR
RESULT=/tmp/v16_missing_sweep_results.txt
: > $RESULT

run_eval () {
  local lang=$1
  local dev=$2
  local cfg=/tmp/pipeline_${lang}_v16.yaml
  local seed_dirs=""
  for s in $SEEDS; do seed_dirs="$seed_dirs checkpoints/v16_native_${lang}_seed${s}"; done

  for vote in 2 3 4; do
    for topk in 0.0 0.10 0.15 0.20; do
      local tag="v16_${lang}_vote${vote}_topk${topk}"
      local out=$PRED_DIR/${tag}.jsonl
      echo "[$(date +%H:%M:%S)] === $tag ===" | tee -a $RESULT
      conda run -n absa --no-capture-output python -u scripts/inference.py \
        --multi_seed_ensemble --seed_dirs $seed_dirs \
        --config $cfg --test_file $dev --output $out \
        --vote_threshold $vote --span_topk_threshold $topk \
        --batch_size 16 > /tmp/v16m_eval_${tag}.log 2>&1
      local cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py \
        -p $out -g $dev -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
      echo "  → vote=$vote topk=$topk  $cf1" | tee -a $RESULT
    done
  done
}

run_eval eng_restaurant data/subtask_3/eng/eng_restaurant_dev_task3.jsonl
run_eval eng_laptop     data/subtask_3/eng/eng_laptop_dev_task3.jsonl
run_eval zho_restaurant data/subtask_3/zho/zho_restaurant_dev_task3.jsonl

echo "================ MISSING SWEEP DONE ================" | tee -a $RESULT
