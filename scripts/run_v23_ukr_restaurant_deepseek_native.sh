#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== V23 native-ukr remaining 4 seeds at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config configs/pipeline_ukr_restaurant_v23_deepseek_native.yaml --seeds 123 456 789 1024 \
  --save_prefix v23_ukr_restaurant_deepseek_native_seed \
  --train_file data/subtask_3/ukr/ukr_restaurant_train_v23_deepseek_native.jsonl \
  >> /tmp/ukr_native_train.log 2>&1
echo "==== native-ukr 5 seeds DONE; running ensemble ===="
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_ukr_restaurant_deepseek_native_seed${s}"; done
DEV=data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl
RES=/tmp/ukr_native_results.txt; : > $RES
echo "=== native-ukr KD ensemble (vs translated 0.5560 / PAI SOTA 0.5437) ===" | tee -a $RES
for vote in 2 3 4; do
  out=/tmp/ukr_native_v${vote}.jsonl
  conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
    --seed_dirs $SD --config configs/pipeline_ukr_restaurant_v23_deepseek_native.yaml --test_file $DEV --output $out \
    --vote_threshold $vote --span_topk_threshold 0.0 --batch_size 16 > /tmp/ukr_native_eval_${vote}.log 2>&1
  cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $DEV -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
  echo "  native vote=$vote  →  $cf1" | tee -a $RES
done
echo "===== native-ukr DONE =====" | tee -a $RES
