#!/bin/bash
# V19: smoke-test 3 alternative encoders for eng_laptop with seed 42.
# Order: roberta-large (fast) → ernie2-large (fast) → deberta-v2-xxlarge (slow, 3×).

set -e
cd /home/g24tka12/ABSA-QUAD

train_one () {
  local tag=$1
  local cfg=$2
  local prefix=v19_${tag}_eng_laptop_seed
  local logfile=/tmp/v19_${tag}_train.log

  echo "==== Starting V19-$tag (seed 42) at $(date) ===="
  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg --seeds 42 \
    --save_prefix $prefix \
    --train_file data/subtask_3/eng/eng_laptop_train_alltasks.jsonl \
    > $logfile 2>&1
  echo "[$(date)] V19-$tag done. Log: $logfile"
}

train_one roberta_large       /tmp/pipeline_eng_laptop_v19_roberta_large.yaml
train_one ernie2_large        /tmp/pipeline_eng_laptop_v19_ernie2_large.yaml
train_one deberta_v2_xxlarge  /tmp/pipeline_eng_laptop_v19_deberta_v2_xxlarge.yaml

echo "================ V19 SMOKE TEST DONE at $(date) ================"
