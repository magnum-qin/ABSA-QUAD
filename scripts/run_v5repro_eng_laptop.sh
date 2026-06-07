#!/bin/bash
# Reproduce historical eng_laptop best (V5, cF1=0.3304):
#   deberta-v3-base + 4-pivot BT (en→de/fr/zh/ru→en) + dual-layer 5-seed ensemble.
# Waits for the 4-pivot BT data file to exist before training.

set -e
cd /home/g24tka12/ABSA-QUAD

DATA=data/subtask_3/eng/eng_laptop_train_v5repro.jsonl
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: $DATA not found. Run augment_data.py --langs de fr zh ru first." >&2
  exit 1
fi
echo "Train rows: $(wc -l < $DATA)"

echo "==== Starting V5-repro eng_laptop (deberta-v3-base, 5 seeds, dual-layer) at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_eng_laptop_v5repro.yaml \
  --seeds 42 123 456 789 1024 \
  --save_prefix v5repro_eng_laptop_seed \
  --train_file "$DATA" \
  > /tmp/v5repro_eng_laptop_train.log 2>&1
echo "==== V5-repro eng_laptop DONE at $(date) ===="
