#!/bin/bash
# Route A: eng_laptop = deberta-v3-base + 6-pivot BT (de/fr/zh/ru/es/it, 21603 rows)
#          + 10-seed dual-layer ensemble. Builds on verified V5-repro (0.3254).
set -e
cd /home/g24tka12/ABSA-QUAD
echo "==== Route A V5-6pivot eng_laptop (10 seeds) at $(date) ===="
echo "Train rows: $(wc -l < data/subtask_3/eng/eng_laptop_train_v5_6pivot.jsonl)"
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_eng_laptop_v5_6pivot.yaml \
  --seeds 42 123 456 789 1024 2025 31337 7 64 256 \
  --save_prefix v5_6pivot_eng_laptop_seed \
  --train_file data/subtask_3/eng/eng_laptop_train_v5_6pivot.jsonl \
  > /tmp/v5_6pivot_eng_laptop_train.log 2>&1
echo "==== Route A DONE at $(date) ===="
