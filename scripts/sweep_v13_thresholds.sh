#!/bin/bash
# Vote × pair_threshold sweep on V13 checkpoints (no retraining).
# Outputs cF1 for each combination to help find the optimal post-hoc thresholds.

set -e
cd /home/g24tka12/ABSA-QUAD

SEED_DIRS=(
  "checkpoints/v13_comet096_jpn_hotel_seed42"
  "checkpoints/v13_comet096_jpn_hotel_seed123"
  "checkpoints/v13_comet096_jpn_hotel_seed456"
  "checkpoints/v13_comet096_jpn_hotel_seed789"
  "checkpoints/v13_comet096_jpn_hotel_seed1024"
)

CONFIG=/tmp/pipeline_jpn_v13.yaml
GOLD=data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl
OUT_DIR=predictions/sweep_v13
mkdir -p "$OUT_DIR"

RESULTS_FILE=/tmp/v13_sweep_results.tsv
echo -e "vote\tpair\tcP\tcR\tcF1" > "$RESULTS_FILE"

run_combo () {
  local vote=$1
  local pair=$2
  local pred="$OUT_DIR/v13_vote${vote}_pair${pair}.jsonl"
  local pred_mecab="${pred%.jsonl}_mecab.jsonl"

  echo ">>> vote=$vote pair=$pair"
  conda run -n absa python3 scripts/inference.py \
    --multi_seed_ensemble --dual_layer \
    --seed_dirs "${SEED_DIRS[@]}" \
    --vote_threshold "$vote" \
    --config "$CONFIG" \
    --test_file "$GOLD" \
    --output "$pred" \
    --batch_size 32 \
    --pair_threshold "$pair" 2>&1 | tail -2

  conda run -n absa python3 scripts/postprocess_jpn.py \
    --input "$pred" --gold "$GOLD" --output "$pred_mecab" 2>&1 | tail -2

  # Parse cP/cR/cF1
  local eval_out
  eval_out=$(conda run -n absa python3 scripts/official_eval.py \
    -p "$pred_mecab" -g "$GOLD" -t 3 2>&1)
  local cp cr cf1
  cp=$(echo "$eval_out"  | awk '/cPrecision:/ {print $2}')
  cr=$(echo "$eval_out"  | awk '/cRecall:/    {print $2}')
  cf1=$(echo "$eval_out" | awk '/cF1:/        {print $2}' | head -1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$vote" "$pair" "$cp" "$cr" "$cf1" \
    | tee -a "$RESULTS_FILE"
}

# Phase 1: vote sweep at pair=0.3
echo "=== PHASE 1: vote sweep (pair=0.3) ==="
for vote in 1 2 3 4 5; do
  run_combo "$vote" "0.3"
done

# Phase 2: pair sweep at best vote
BEST_VOTE=$(awk -F'\t' 'NR>1 {print $5"\t"$1}' "$RESULTS_FILE" | sort -rn | head -1 | cut -f2)
echo ""
echo "=== PHASE 2: pair sweep at best vote=$BEST_VOTE ==="
for pair in 0.2 0.4 0.5; do
  run_combo "$BEST_VOTE" "$pair"
done

echo ""
echo "=== FINAL RESULTS (sorted by cF1) ==="
(head -1 "$RESULTS_FILE"; tail -n +2 "$RESULTS_FILE" | sort -t$'\t' -k5 -rn) \
  | column -t -s $'\t'
