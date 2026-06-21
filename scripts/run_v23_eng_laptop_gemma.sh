#!/bin/bash
# Route D (local-teacher): eng_laptop KD distilled from LOCAL gemma-4-31B-it-qat
# instead of cloud DeepSeek. Identical config / gold+BT base / 5-seed dual-layer
# ensemble — only the silver teacher differs. Compares to DeepSeek arm = 0.4141.
set -e
cd /home/g24tka12/ABSA-QUAD

echo "==== [1/3] build gemma distill data at $(date) ===="
python3 scripts/build_v23_eng_laptop_gemma.py

echo "==== [2/3] train 5 seeds (gemma silver) at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config configs/pipeline_eng_laptop_v23_deepseek.yaml \
  --seeds 42 123 456 789 1024 \
  --save_prefix v23_eng_laptop_gemma_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_v23_gemma.jsonl \
  > /tmp/v23_eng_laptop_gemma_train.log 2>&1

echo "==== [3/3] ensemble eval sweep at $(date) ===="
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_eng_laptop_gemma_seed${s}"; done
DEV=data/subtask_3/eng/eng_laptop_dev_task3.jsonl
RES=/tmp/eng_gemma_kd_results.txt; : > $RES
echo "=== eng_laptop gemma-KD ensemble (vs DeepSeek-KD 0.4141 / V5-repro 0.3254) ===" | tee -a $RES
for vote in 2 3 4; do
  for topk in 0.0 0.20; do
    out=/tmp/eng_gemma_kd_v${vote}_t${topk}.jsonl
    conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
      --seed_dirs $SD --config configs/pipeline_eng_laptop_v23_deepseek.yaml --test_file $DEV --output $out \
      --vote_threshold $vote --span_topk_threshold $topk --batch_size 16 > /tmp/eng_gemma_eval_${vote}_${topk}.log 2>&1
    cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $DEV -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
    echo "  eng_laptop gemma vote=$vote topk=$topk  ->  $cf1" | tee -a $RES
  done
done
echo "===== eng_laptop GEMMA-KD PIPELINE DONE at $(date) =====" | tee -a $RES
