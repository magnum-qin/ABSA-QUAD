#!/bin/bash
# Train V16 for the 3 datasets not yet covered: eng_restaurant, eng_laptop, zho_restaurant.
# Order: eng_restaurant (smallest 2284) → eng_laptop (4076) → zho_restaurant (6050)

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"

train_one () {
  local name=$1
  local data=$2
  local cfg=/tmp/pipeline_${name}_v16.yaml
  local prefix=v16_native_${name}_seed
  local logfile=/tmp/v16_${name}_train.log

  echo "========================================"
  echo " Starting V16-${name} at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg \
    --seeds $SEEDS \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] V16-${name} done. Log: $logfile"
}

train_one eng_restaurant data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl
train_one eng_laptop     data/subtask_3/eng/eng_laptop_train_alltasks.jsonl
train_one zho_restaurant data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl

echo "========================================"
echo " V16 missing datasets COMPLETE at $(date)"
echo "========================================"
