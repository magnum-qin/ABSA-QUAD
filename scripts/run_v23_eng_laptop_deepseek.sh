#!/bin/bash
# Route D: eng_laptop knowledge distillation
#   deberta-v3-base + raw + 4-pivot BT + 8541 DeepSeek-v4-pro silver labels (23590 rows)
#   5-seed dual-layer ensemble. LLM priors distilled into weights; no LLM at inference.
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== Route D V23 silver-distill eng_laptop (5 seeds) at $(date) ===="
echo "Train rows: $(wc -l < data/subtask_3/eng/eng_laptop_train_v23_deepseek.jsonl)"
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config configs/pipeline_eng_laptop_v23_deepseek.yaml \
  --seeds 42 123 456 789 1024 \
  --save_prefix v23_eng_laptop_deepseek_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_v23_deepseek.jsonl \
  > /tmp/v23_eng_laptop_deepseek_train.log 2>&1
echo "==== Route D V23 DONE at $(date) ===="
