#!/bin/bash
# V16: language-native encoders (vs V15 universal XLM-R-large)
# Order: zho_laptop (biggest gap) → jpn → rus
# Skip ukr (V13b already best) and tat (no Tatar-specific encoder exists).

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"

train_one () {
  local lang=$1
  local data=$2
  local cfg=/tmp/pipeline_${lang}_v16.yaml
  local prefix=v16_native_${lang}_seed
  local logfile=/tmp/v16_${lang}_train.log

  echo "========================================"
  echo " Starting V16-${lang} at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg \
    --seeds $SEEDS \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] V16-${lang} done. Log: $logfile"
}

train_one zho_laptop data/subtask_3/zho/zho_laptop_train_v13c.jsonl
train_one jpn        data/subtask_3/jpn/jpn_hotel_train_v13.jsonl
train_one rus        data/subtask_3/rus/rus_restaurant_train_v13.jsonl

echo "========================================"
echo " V16 native-encoder training COMPLETE at $(date)"
echo "========================================"
