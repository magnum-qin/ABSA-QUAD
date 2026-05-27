#!/bin/bash
# Finer top-k sweep for V15-tat. Current best: vote=3 topk=0.20 = 0.4821 (boundary).
# Extend topk grid + try vote=3.5 (not possible — integer only) so re-confirm vote=2,3,4 at extreme topk.

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"
SD=""; for s in $SEEDS; do SD="$SD checkpoints/v15_xlmr_tat_seed${s}"; done

DEV=data/subtask_3/tat/tat_restaurant_dev_task3.jsonl
CFG=/tmp/pipeline_tat_v15.yaml
PRED_DIR=predictions/v15_sweep_tat
mkdir -p $PRED_DIR
RESULT=/tmp/v15_tat_finer_results.txt
: > $RESULT

for vote in 3; do
  for topk in 0.05 0.25 0.30 0.35 0.40 0.50; do
    tag="v15_tat_vote${vote}_topk${topk}"
    out=$PRED_DIR/${tag}.jsonl
    echo "[$(date +%H:%M:%S)] === $tag ===" | tee -a $RESULT

    conda run -n absa --no-capture-output python -u scripts/inference.py \
      --multi_seed_ensemble --seed_dirs $SD \
      --config $CFG --test_file $DEV --output $out \
      --vote_threshold $vote --span_topk_threshold $topk \
      --batch_size 16 > /tmp/v15_tat_finer_${tag}.log 2>&1

    cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py \
      -p $out -g $DEV -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
    echo "  → vote=$vote topk=$topk  $cf1" | tee -a $RESULT
  done
done

echo "================ tat FINER SWEEP DONE ================" | tee -a $RESULT
