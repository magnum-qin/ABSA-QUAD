#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== V23 KD zho_laptop (chinese-roberta + silver) at $(date) ===="
echo "rows: $(wc -l < data/subtask_3/zho/zho_laptop_train_v23.jsonl)"
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_zho_laptop_v23.yaml --seeds 42 123 456 789 1024 \
  --save_prefix v23_silver_zho_laptop_seed \
  --train_file data/subtask_3/zho/zho_laptop_train_v23.jsonl \
  > /tmp/v23_silver_zho_laptop_train.log 2>&1
echo "==== V23 KD zho_laptop DONE at $(date) ===="
