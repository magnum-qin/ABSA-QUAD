#!/bin/bash
# V17: Uncertainty-Weighted Multi-task Loss (UML) — adopted from LogSigma's Subtask 1 method.
# First: train seed 42 only as sanity check. If it converges normally, expand to 5 seeds.
#
# UML learns per-task log(sigma^2) and weights each loss by 0.5 * exp(-log_sigma).
# Replaces fixed loss_weights from V16 config.

set -e
cd /home/g24tka12/ABSA-QUAD

train_one () {
  local name=$1
  local cfg=$2
  local data=$3
  local prefix=$4
  local seeds=$5
  local logfile=/tmp/v17_uml_${name}_train.log

  echo "========================================"
  echo " Starting V17-UML $name (seeds $seeds) at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg --seeds $seeds \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] V17-UML $name done. Log: $logfile"
}

# Sanity-check seed 42 first (faster: jpn = ~1h, eng_rest = ~2.5h)
# Use no_aug for jpn (since §7.6.3 showed no_aug ensemble slightly better)
train_one jpn_no_aug \
  /tmp/pipeline_jpn_v17_uml_no_aug.yaml \
  data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
  v17_uml_jpn_no_aug_seed "42"

train_one eng_restaurant \
  /tmp/pipeline_eng_restaurant_v17_uml.yaml \
  data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  v17_uml_eng_restaurant_seed "42"

echo "================ V17-UML seed 42 sanity check DONE at $(date) ================"
