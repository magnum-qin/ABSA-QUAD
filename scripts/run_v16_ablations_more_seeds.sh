#!/bin/bash
# Confirm ablation findings with 4 more seeds each.
# Order: jpn (faster, 1h/seed × 4 = 4h) → eng_restaurant (3h/seed × 4 = 12h)
# Total ~16h sequential.

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="123 456 789 1024"

train_one () {
  local name=$1
  local cfg=$2
  local data=$3
  local prefix=$4
  local logfile=/tmp/ablation_${name}_more_train.log

  echo "========================================"
  echo " Starting ablation $name (seeds $SEEDS) at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg --seeds $SEEDS \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] ablation $name done. Log: $logfile"
}

# Run faster one first
train_one jpn_no_aug \
  /tmp/pipeline_jpn_v16_no_aug.yaml \
  data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
  v16_ablation_jpn_no_aug_seed

train_one eng_restaurant_no_rethink \
  /tmp/pipeline_eng_restaurant_v16_no_rethink.yaml \
  data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  v16_ablation_eng_restaurant_no_rethink_seed

echo "================ MORE-SEED ABLATIONS COMPLETE at $(date) ================"
