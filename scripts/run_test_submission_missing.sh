#!/bin/bash
set -e
cd /home/g24tka12/ABSA-QUAD
SEEDS="42 123 456 789 1024"
OUT_DIR=submission/final
mkdir -p $OUT_DIR
RESULT=/tmp/test_submission_missing.txt
: > $RESULT

predict () {
  local lang=$1
  local cfg=$2
  local test_file=$3
  local vote=$4
  local topk=$5
  local seed_dirs=""
  for s in $SEEDS; do seed_dirs="$seed_dirs checkpoints/v16_native_${lang}_seed${s}"; done
  local out=$OUT_DIR/${lang}_test.jsonl
  echo "[$(date +%H:%M:%S)] === $lang vote=$vote topk=$topk ===" | tee -a $RESULT
  conda run -n absa --no-capture-output python -u scripts/inference.py \
    --multi_seed_ensemble --seed_dirs $seed_dirs \
    --config $cfg --test_file $test_file --output $out \
    --vote_threshold $vote --span_topk_threshold $topk \
    --batch_size 16 > /tmp/test_pred_${lang}.log 2>&1
  local n=$(wc -l < $out); local q=$(python -c "import json; print(sum(len(json.loads(l).get('Quadruplet',[])) for l in open('$out')))")
  echo "  → $n samples, $q quads → $out" | tee -a $RESULT
}

predict eng_restaurant /tmp/pipeline_eng_restaurant_v16.yaml data/subtask_3/eng/eng_restaurant_test_task3.jsonl 4 0.0
predict eng_laptop     /tmp/pipeline_eng_laptop_v16.yaml     data/subtask_3/eng/eng_laptop_test_task3.jsonl     3 0.0
predict zho_restaurant /tmp/pipeline_zho_restaurant_v16.yaml data/subtask_3/zho/zho_restaurant_test_task3.jsonl 4 0.0

echo "===== MISSING TEST DONE =====" | tee -a $RESULT
ls -la $OUT_DIR/
