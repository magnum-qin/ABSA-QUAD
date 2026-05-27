#!/bin/bash
# Generate test-set predictions for final submission using per-language best configs.
# Per §5.23 doc:
#   jpn:        V16 (ku-nlp deberta-v2-large) vote=4 topk=0.0
#   rus:        V15 (xlm-roberta-large)       vote=4 topk=0.10
#   zho_laptop: V16 (chinese-roberta-wwm-ext-large) vote=4 topk=0.10
#   ukr:        V13b (mdeberta-v3-base)       vote=3 topk=0.0
#   tat:        V15 (xlm-roberta-large)       vote=3 topk=0.20

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"
OUT_DIR=submission/final
mkdir -p $OUT_DIR
RESULT=/tmp/test_submission_results.txt
: > $RESULT

predict () {
  local lang=$1
  local prefix=$2
  local cfg=$3
  local test_file=$4
  local vote=$5
  local topk=$6

  local seed_dirs=""
  for s in $SEEDS; do seed_dirs="$seed_dirs checkpoints/${prefix}${s}"; done
  local out=$OUT_DIR/${lang}_test.jsonl

  echo "[$(date +%H:%M:%S)] === $lang ($prefix vote=$vote topk=$topk) ===" | tee -a $RESULT
  conda run -n absa --no-capture-output python -u scripts/inference.py \
    --multi_seed_ensemble --seed_dirs $seed_dirs \
    --config $cfg --test_file $test_file --output $out \
    --vote_threshold $vote --span_topk_threshold $topk \
    --batch_size 16 > /tmp/test_pred_${lang}.log 2>&1

  local n_samples=$(wc -l < $out)
  local n_quads=$(python -c "import json; print(sum(len(json.loads(l).get('Quadruplet',[])) for l in open('$out')))")
  echo "  → $n_samples samples, $n_quads total quads → $out" | tee -a $RESULT
}

# jpn: V16 deberta-v2
predict jpn_hotel       v16_native_jpn_seed        /tmp/pipeline_jpn_v16.yaml        data/subtask_3/jpn/jpn_hotel_test_task3.jsonl       4 0.0
# rus: V15 XLM-R
predict rus_restaurant  v15_xlmr_rus_seed          /tmp/pipeline_rus_v15.yaml        data/subtask_3/rus/rus_restaurant_test_task3.jsonl 4 0.10
# zho_laptop: V16 chinese-roberta
predict zho_laptop      v16_native_zho_laptop_seed /tmp/pipeline_zho_laptop_v16.yaml data/subtask_3/zho/zho_laptop_test_task3.jsonl     4 0.10
# ukr: V13b mDeBERTa
predict ukr_restaurant  v13b_ukr_seed              /tmp/pipeline_ukr_v13b.yaml       data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl 3 0.0
# tat: V15 XLM-R
predict tat_restaurant  v15_xlmr_tat_seed          /tmp/pipeline_tat_v15.yaml        data/subtask_3/tat/tat_restaurant_test_task3.jsonl 3 0.20

echo "================ TEST SUBMISSION DONE ================" | tee -a $RESULT
ls -la $OUT_DIR/

# ===== APPENDED: missing 3 datasets =====
