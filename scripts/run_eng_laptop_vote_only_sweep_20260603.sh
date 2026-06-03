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

TAG=eng_laptop_vote_only_20260603
mkdir -p "predictions/${TAG}" "logs/${TAG}"
RESULT="logs/${TAG}/results.tsv"
: > "$RESULT"
printf "vote\ttopk\tdual\tcf1\tpred\n" >> "$RESULT"

SEED_DIRS=(
  checkpoints/final_eng_laptop_v5_bt_seed42
  checkpoints/final_eng_laptop_v5_bt_seed123
  checkpoints/final_eng_laptop_v5_bt_seed456
  checkpoints/final_eng_laptop_v5_bt_seed789
  checkpoints/final_eng_laptop_v5_bt_seed1024
)
CFG=final_configs/eng_laptop_v5_bt.yaml
DEV=data/subtask_3/eng/eng_laptop_dev_task3.jsonl

for dual in 1 0; do
  for vote in 2 3 4 5; do
    out="predictions/${TAG}/vote${vote}_topk0_dual${dual}.jsonl"
    log="logs/${TAG}/vote${vote}_topk0_dual${dual}.log"
    args=(
      -u scripts/inference.py
      --multi_seed_ensemble
      --seed_dirs "${SEED_DIRS[@]}"
      --config "$CFG"
      --test_file "$DEV"
      --output "$out"
      --vote_threshold "$vote"
      --span_topk_threshold 0.0
      --batch_size 16
    )
    if [[ "$dual" == "1" ]]; then
      args+=(--dual_layer)
    fi
    "$PY" "${args[@]}" > "$log" 2>&1
    cf1="$("$PY" scripts/official_eval.py -p "$out" -g "$DEV" -t 3 2>/dev/null | grep -oE 'cF1: [0-9.]+' | tail -1 | awk '{print $2}')"
    printf "%s\t0.0\t%s\t%s\t%s\n" "$vote" "$dual" "$cf1" "$out" | tee -a "$RESULT"
  done
done

sort -t $'\t' -k4,4nr "$RESULT" | tee "logs/${TAG}/results_sorted.tsv"
