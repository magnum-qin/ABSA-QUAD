#!/bin/bash
# Recovery run for V15 trainings killed by ENOSPC on 2026-05-20:
#   - zho_laptop seeds 789, 1024 (42/123/456 already complete)
#   - ukr all 5 seeds
#   - tat all 5 seeds

set -e
cd /home/g24tka12/ABSA-QUAD

# Note: prior V15 trainings used --pretrained_checkpoint v5_zho_restaurant
# (mDeBERTa-base, 768d), but the loader showed "Loaded & averaged 0 encoder
# parameters" every time — shape mismatch with XLM-R-large (1024d) → no-op.
# Apples-to-apples with prior V15 results means dropping the flag entirely.
FULL_SEEDS="42 123 456 789 1024"

train_one () {
  local lang=$1
  local data=$2
  local seeds=$3
  local cfg=/tmp/pipeline_${lang}_v15.yaml
  local prefix=v15_xlmr_${lang}_seed
  local logfile=/tmp/v15_${lang}_train.log

  echo "========================================"
  echo " Starting V15-${lang} (seeds: $seeds) at $(date)"
  echo "========================================"

  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $cfg \
    --seeds $seeds \
    --save_prefix $prefix \
    --train_file $data \
    > $logfile 2>&1

  echo "[$(date)] V15-${lang} done. Log: $logfile"
}

train_one zho_laptop data/subtask_3/zho/zho_laptop_train_v13c.jsonl "789 1024"
train_one ukr        data/subtask_3/ukr/ukr_restaurant_train_v13b.jsonl "$FULL_SEEDS"
train_one tat        data/subtask_3/tat/tat_restaurant_train_v13.jsonl "$FULL_SEEDS"

echo "========================================"
echo " V15 RECOVERY complete at $(date)"
echo "========================================"
