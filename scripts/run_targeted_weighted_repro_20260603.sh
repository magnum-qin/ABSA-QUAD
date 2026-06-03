#!/usr/bin/env bash
set -euo pipefail

cd /home/g24tka12/ABSA-QUAD

if [[ -x /tmp/g24tka12_absa/miniforge3/envs/absa/bin/python ]]; then
  PY=/tmp/g24tka12_absa/miniforge3/envs/absa/bin/python
elif [[ -x /home/g24tka12/miniforge3/envs/absa/bin/python ]]; then
  PY=/home/g24tka12/miniforge3/envs/absa/bin/python
else
  PY=python3
fi

CONDA_PREFIX_DIR="$(dirname "$(dirname "$PY")")"
export CONDA_PREFIX="$CONDA_PREFIX_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"

TAG=targeted_weighted_repro_20260603
mkdir -p "predictions/${TAG}" "logs/${TAG}"
RESULT="logs/${TAG}/results.tsv"
: > "$RESULT"
printf "task\tmethod\tvote\ttopk\tcf1\tpred\n" >> "$RESULT"

run_uniform() {
  local task="$1" cfg="$2" dev="$3" prefix="$4" vote="$5" topk="$6"
  local out="predictions/${TAG}/${task}_uniform_vote${vote}_topk${topk}.jsonl"
  local log="logs/${TAG}/${task}_uniform_vote${vote}_topk${topk}.log"
  "$PY" -u scripts/inference.py \
    --multi_seed_ensemble \
    --seed_dirs checkpoints/${prefix}42 checkpoints/${prefix}123 checkpoints/${prefix}456 checkpoints/${prefix}789 checkpoints/${prefix}1024 \
    --config "$cfg" --test_file "$dev" --output "$out" \
    --vote_threshold "$vote" --span_topk_threshold "$topk" --batch_size 16 \
    > "$log" 2>&1
  local cf1
  cf1="$("$PY" scripts/official_eval.py -p "$out" -g "$dev" -t 3 2>/dev/null | grep -oE 'cF1: [0-9.]+' | tail -1 | awk '{print $2}')"
  printf "%s\tuniform\t%s\t%s\t%s\t%s\n" "$task" "$vote" "$topk" "$cf1" "$out" | tee -a "$RESULT"
}

run_weighted() {
  local task="$1" cfg="$2" dev="$3" prefix="$4" vote="$5" topk="$6"
  shift 6
  local weights=("$@")
  local out="predictions/${TAG}/${task}_weighted_vote${vote}_topk${topk}.jsonl"
  local log="logs/${TAG}/${task}_weighted_vote${vote}_topk${topk}.log"
  "$PY" -u scripts/confidence_weighted_ensemble.py \
    --seed_dirs checkpoints/${prefix}42 checkpoints/${prefix}123 checkpoints/${prefix}456 checkpoints/${prefix}789 checkpoints/${prefix}1024 \
    --seed_weights "${weights[@]}" \
    --config "$cfg" --test_file "$dev" --output "$out" \
    --vote_threshold "$vote" --span_topk_threshold "$topk" --batch_size 16 \
    > "$log" 2>&1
  local cf1
  cf1="$("$PY" scripts/official_eval.py -p "$out" -g "$dev" -t 3 2>/dev/null | grep -oE 'cF1: [0-9.]+' | tail -1 | awk '{print $2}')"
  printf "%s\tconfidence_weighted\t%s\t%s\t%s\t%s\n" "$task" "$vote" "$topk" "$cf1" "$out" | tee -a "$RESULT"
}

run_uniform rus_restaurant final_configs/rus_restaurant_v15.yaml data/subtask_3/rus/rus_restaurant_dev_task3.jsonl final_rus_restaurant_v15_seed 4 0.10
run_weighted rus_restaurant final_configs/rus_restaurant_v15.yaml data/subtask_3/rus/rus_restaurant_dev_task3.jsonl final_rus_restaurant_v15_seed 4 0.10 0.4817 0.4469 0.4785 0.4646 0.4780

run_uniform ukr_restaurant final_configs/ukr_restaurant_v13b.yaml data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl final_ukr_restaurant_v13b_seed 3 0.0
run_weighted ukr_restaurant final_configs/ukr_restaurant_v13b.yaml data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl final_ukr_restaurant_v13b_seed 3 0.0 0.4410 0.4472 0.4674 0.4736 0.4415

sort -t $'\t' -k5,5nr "$RESULT" | tee "logs/${TAG}/results_sorted.tsv"
