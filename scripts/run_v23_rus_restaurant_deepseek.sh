#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== V23 KD rus_restaurant (xlm-roberta-large + silver) at $(date) ===="
echo "rows: $(wc -l < data/subtask_3/rus/rus_restaurant_train_v23.jsonl)"
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_rus_v23.yaml --seeds 42 123 456 789 1024 \
  --save_prefix v23_silver_rus_seed \
  --train_file data/subtask_3/rus/rus_restaurant_train_v23.jsonl \
  > /tmp/v23_silver_rus_train.log 2>&1
echo "==== V23 KD rus DONE at $(date) ===="
