#!/bin/bash
# 2 ablations on V16, single seed (42) each:
#   #2: eng_restaurant WITHOUT Rethink module
#   #3: jpn_hotel WITHOUT data augmentation (raw alltasks instead of v13)

set -e
cd /home/g24tka12/ABSA-QUAD

train_one () {
  local name=$1
  local cfg=$2
  local data=$3
  local prefix=$4
  local logfile=/tmp/ablation_${name}_train.log

  echo "========================================"
  echo " Starting ablation $name at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg --seeds 42 \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] ablation $name done. Log: $logfile"
}

# #2: V16-eng_restaurant WITHOUT rethink module
train_one eng_restaurant_no_rethink \
  /tmp/pipeline_eng_restaurant_v16_no_rethink.yaml \
  data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  v16_ablation_eng_restaurant_no_rethink_seed

# #3: V16-jpn WITHOUT data augmentation
train_one jpn_no_aug \
  /tmp/pipeline_jpn_v16_no_aug.yaml \
  data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
  v16_ablation_jpn_no_aug_seed

echo "================ V16 ABLATIONS COMPLETE at $(date) ================"
