#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== Starting V20 eng_laptop (roberta-large + BT + paraphrase, V13b) at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_eng_laptop_v20_v13b.yaml \
  --seeds 42 123 456 789 1024 \
  --save_prefix v20_rob_v13b_eng_laptop_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_v13b.jsonl \
  > /tmp/v20_eng_laptop_train.log 2>&1
echo "==== V20 DONE at $(date) ===="
