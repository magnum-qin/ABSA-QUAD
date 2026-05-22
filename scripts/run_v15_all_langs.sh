#!/bin/bash
# Sequentially train V15 (XLM-R-large) on 4 non-jpn languages.
# Order: rus → zho_laptop → ukr → tat (rough size order)

set -e
cd /home/g24tka12/ABSA-QUAD

PRETRAIN=checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt
SEEDS="42 123 456 789 1024"

train_one () {
  local lang=$1
  local data=$2
  local cfg=/tmp/pipeline_${lang}_v15.yaml
  local prefix=v15_xlmr_${lang}_seed
  local logfile=/tmp/v15_${lang}_train.log

  echo "========================================"
  echo " Starting V15-${lang} at $(date)"
  echo " data=$data, prefix=$prefix"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg \
    --seeds $SEEDS \
    --save_prefix $prefix \
    --train_file $data \
    --pretrained_checkpoint $PRETRAIN \
    > $logfile 2>&1

  echo "[$(date)] V15-${lang} done. Log: $logfile"
}

train_one rus        data/subtask_3/rus/rus_restaurant_train_v13.jsonl
train_one zho_laptop data/subtask_3/zho/zho_laptop_train_v13c.jsonl
train_one ukr        data/subtask_3/ukr/ukr_restaurant_train_v13b.jsonl
train_one tat        data/subtask_3/tat/tat_restaurant_train_v13.jsonl

echo "========================================"
echo " ALL V15 trainings complete at $(date)"
echo "========================================"
