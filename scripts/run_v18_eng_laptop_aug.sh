#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== Starting V18 eng_laptop (V13 aug: 4076+3960 paraphrases) at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_eng_laptop_v18_v13aug.yaml \
  --seeds 42 123 456 789 1024 \
  --save_prefix v18_v13aug_eng_laptop_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_v13.jsonl \
  > /tmp/v18_eng_laptop_train.log 2>&1
echo "==== V18 eng_laptop COMPLETE at $(date) ===="
