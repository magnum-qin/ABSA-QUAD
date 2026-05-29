#!/bin/bash
# V19 roberta-large 4 remaining seeds (42 already done in smoke test = 0.3024).

set -e
cd /home/g24tka12/ABSA-QUAD

echo "==== Starting V19 roberta-large 4 more seeds at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_eng_laptop_v19_roberta_large.yaml \
  --seeds 123 456 789 1024 \
  --save_prefix v19_roberta_large_eng_laptop_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_alltasks.jsonl \
  > /tmp/v19_roberta_large_more_train.log 2>&1
echo "==== V19 roberta-large DONE at $(date) ===="
