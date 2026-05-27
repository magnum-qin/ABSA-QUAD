#!/bin/bash
# V15 5-seed ensemble + vote threshold + top-k span sweep per language.
# Reports best (vote, topk) config per lang vs prior V13/V13b/V13c ensemble baseline.

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"
PRED_DIR=predictions/v15_sweep
mkdir -p $PRED_DIR
RESULT=/tmp/v15_ensemble_sweep_results.txt
: > $RESULT

run_eval () {
  local lang=$1
  local dev=$2
  local cfg=/tmp/pipeline_${lang}_v15.yaml
  local seed_dirs=""
  for s in $SEEDS; do seed_dirs="$seed_dirs checkpoints/v15_xlmr_${lang}_seed${s}"; done

  for vote in 2 3 4; do
    for topk in 0.0 0.10 0.15 0.20; do
      local tag="v15_${lang}_vote${vote}_topk${topk}"
      local out=$PRED_DIR/${tag}.jsonl

      echo "[$(date +%H:%M:%S)] === $tag ===" | tee -a $RESULT
      conda run -n absa --no-capture-output python -u scripts/inference.py \
        --multi_seed_ensemble \
        --seed_dirs $seed_dirs \
        --config $cfg \
        --test_file $dev \
        --output $out \
        --vote_threshold $vote \
        --span_topk_threshold $topk \
        --batch_size 16 \
        > /tmp/v15_eval_${tag}.log 2>&1

      local cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py \
        -p $out -g $dev -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
      echo "  → vote=$vote topk=$topk  $cf1" | tee -a $RESULT
    done
  done
}

# Per-lang priority: rus (likely win), tat (high var, ensemble decides), zho_laptop/ukr (confirm)
run_eval rus        data/subtask_3/rus/rus_restaurant_dev_task3.jsonl
run_eval tat        data/subtask_3/tat/tat_restaurant_dev_task3.jsonl
run_eval zho_laptop data/subtask_3/zho/zho_laptop_dev_task3.jsonl
run_eval ukr        data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl

echo "================ SWEEP COMPLETE ================" | tee -a $RESULT
echo "Results: $RESULT"
