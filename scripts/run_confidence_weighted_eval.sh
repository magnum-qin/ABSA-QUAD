#!/bin/bash
# Compare confidence-weighted ensemble vs uniform ensemble on dev set for all 8 datasets.

set -e
cd /home/g24tka12/ABSA-QUAD

SEEDS="42 123 456 789 1024"
PRED_DIR=predictions/cw_eval
mkdir -p $PRED_DIR
RESULT=/tmp/cw_ensemble_results.txt
: > $RESULT

run_cw () {
  local lang=$1
  local prefix=$2          # e.g. v16_native_jpn_seed
  local cfg=$3             # /tmp/pipeline_jpn_v16.yaml
  local dev=$4
  local vote=$5
  local topk=$6
  shift 6
  local weights="$@"       # 5 dev cF1 floats

  local seed_dirs=""
  for s in $SEEDS; do seed_dirs="$seed_dirs checkpoints/${prefix}${s}"; done
  local out=$PRED_DIR/cw_${lang}.jsonl

  echo "[$(date +%H:%M:%S)] === $lang vote=$vote topk=$topk weights=$weights ===" | tee -a $RESULT
  conda run -n absa --no-capture-output python -u scripts/confidence_weighted_ensemble.py \
    --seed_dirs $seed_dirs --seed_weights $weights \
    --config $cfg --test_file $dev --output $out \
    --vote_threshold $vote --span_topk_threshold $topk \
    --batch_size 16 > /tmp/cw_${lang}.log 2>&1

  local cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py \
    -p $out -g $dev -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
  echo "  → $cf1" | tee -a $RESULT
}

# Per-dataset: prefix, config, dev, vote, topk, seed_weights (42 123 456 789 1024)
# Weights are per-seed best dev cF1 (from training logs)
run_cw jpn_hotel       v16_ablation_jpn_no_aug_seed  /tmp/pipeline_jpn_v16_no_aug.yaml \
    data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl    3 0.0   0.4109 0.3967 0.3940 0.4055 0.3978
run_cw rus_restaurant  v15_xlmr_rus_seed             /tmp/pipeline_rus_v15.yaml \
    data/subtask_3/rus/rus_restaurant_dev_task3.jsonl 4 0.10 0.4817 0.4469 0.4785 0.4646 0.4780
run_cw zho_laptop      v16_native_zho_laptop_seed    /tmp/pipeline_zho_laptop_v16.yaml \
    data/subtask_3/zho/zho_laptop_dev_task3.jsonl   4 0.10  0.3970 0.3861 0.3941 0.4010 0.3917
run_cw ukr_restaurant  v13b_ukr_seed                 /tmp/pipeline_ukr_v13b.yaml \
    data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl 3 0.0 0.4410 0.4472 0.4674 0.4736 0.4415
run_cw tat_restaurant  v15_xlmr_tat_seed             /tmp/pipeline_tat_v15.yaml \
    data/subtask_3/tat/tat_restaurant_dev_task3.jsonl 3 0.20 0.3984 0.4282 0.4047 0.3942 0.3979
run_cw eng_restaurant  v16_native_eng_restaurant_seed /tmp/pipeline_eng_restaurant_v16.yaml \
    data/subtask_3/eng/eng_restaurant_dev_task3.jsonl 4 0.0 0.6759 0.6697 0.6740 0.6715 0.6731
run_cw eng_laptop      v16_native_eng_laptop_seed    /tmp/pipeline_eng_laptop_v16.yaml \
    data/subtask_3/eng/eng_laptop_dev_task3.jsonl   3 0.0  0.2943 0.2929 0.2887 0.2874 0.2829
run_cw zho_restaurant  v16_native_zho_restaurant_seed /tmp/pipeline_zho_restaurant_v16.yaml \
    data/subtask_3/zho/zho_restaurant_dev_task3.jsonl 4 0.0 0.4952 0.5162 0.4956 0.5234 0.5113

echo "================ CW ENSEMBLE EVAL DONE ================" | tee -a $RESULT
