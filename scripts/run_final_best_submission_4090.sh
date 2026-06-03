#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/g24tka12/ABSA-QUAD}"
REPRO_MODE="${REPRO_MODE:-historical}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x /home/g24tka12/miniforge3/envs/absa/bin/python ]]; then
    PY="/home/g24tka12/miniforge3/envs/absa/bin/python"
  elif [[ -x /tmp/g24tka12_absa/miniforge3/envs/absa/bin/python ]]; then
    PY="/tmp/g24tka12_absa/miniforge3/envs/absa/bin/python"
  else
    PY="python"
  fi
fi
CONDA_PREFIX_DIR="$(dirname "$(dirname "$PY")")"

export CONDA_PREFIX="$CONDA_PREFIX_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"
mkdir -p final_configs logs submission/final predictions

SEEDS=(42 123 456 789 1024)
TASKS="${TASKS:-eng_restaurant eng_laptop jpn_hotel rus_restaurant tat_restaurant ukr_restaurant zho_restaurant zho_laptop}"
TASKS="${TASKS//,/ }"

has_task() {
  local wanted="$1"
  [[ " $TASKS " == *" $wanted "* ]]
}

copy_cfg() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "Missing config: $src" >&2
    exit 1
  fi
  cp "$src" "$dst"
}

copy_cfg /tmp/pipeline_eng_restaurant_v16.yaml final_configs/eng_restaurant_v16.yaml
copy_cfg /tmp/pipeline_jpn_v16_no_aug.yaml final_configs/jpn_hotel_v16_no_aug.yaml
copy_cfg /tmp/pipeline_rus_v15.yaml final_configs/rus_restaurant_v15.yaml
copy_cfg /tmp/pipeline_tat_v15.yaml final_configs/tat_restaurant_v15.yaml
copy_cfg /tmp/pipeline_ukr_v13b.yaml final_configs/ukr_restaurant_v13b.yaml
copy_cfg /tmp/pipeline_zho_laptop_v16.yaml final_configs/zho_laptop_v16.yaml
copy_cfg /tmp/pipeline_zho_restaurant_v16.yaml final_configs/zho_restaurant_v16.yaml

cat > final_configs/eng_laptop_v5_bt.yaml <<'YAML'
model:
  encoder_name: "microsoft/deberta-v3-base"
  hidden_size: 768
  num_entity_labels: 24
  num_attribute_labels: 10
  dropout: 0.1
  use_crf: false
  va_hidden_size: 256
  va_min: 1.0
  va_max: 9.0
  rethink:
    enabled: true
    num_iterations: 2
    label_embedding_dim: 64
data:
  train_file: "data/subtask_3/eng/eng_laptop_train_alltasks_augmented.jsonl"
  dev_file: "data/subtask_3/eng/eng_laptop_dev_task3.jsonl"
  test_file: "data/subtask_3/eng/eng_laptop_test_task3.jsonl"
  max_seq_length: 256
  entity_labels: ["BATTERY", "COMPANY", "CPU", "DISPLAY", "FANS_COOLING", "GRAPHICS", "HARDWARE", "HARD_DISK", "KEYBOARD", "LAPTOP", "MEMORY", "MOTHERBOARD", "MOUSE", "MULTIMEDIA_DEVICES", "NULL", "OPTICAL_DRIVES", "OS", "OUT_OF_SCOPE", "PORTS", "POWER_SUPPLY", "SHIPPING", "SOFTWARE", "SUPPORT", "WARRANTY"]
  attribute_labels: ["CONNECTIVITY", "DESIGN_FEATURES", "GENERAL", "MISCELLANEOUS", "NULL", "OPERATION_PERFORMANCE", "PORTABILITY", "PRICE", "QUALITY", "USABILITY"]
training:
  batch_size: 8
  learning_rate: 1.5e-5
  encoder_lr: 1.5e-5
  head_lr: 7.5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_epochs: 30
  max_grad_norm: 1.0
  fgm: {enabled: true, epsilon: 0.1}
  ema: {enabled: true, decay: 0.999}
  loss_weights:
    span: 2.0
    pair: 1.0
    entity: 1.0
    attribute: 1.0
    va: 1.0
    va_valence: 1.0
    va_arousal: 2.0
    consistency: 0.5
  patience: 8
  save_dir: "checkpoints/final_eng_laptop_v5_bt"
  save_top_k: 3
evaluation:
  va_tolerance: 0.5
  metrics: ["continuous_f1", "exact_f1", "va_rmse", "va_mae"]
device: "cuda"
seed: 42
YAML

count_lines() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "MISSING"
    return
  fi
  wc -l < "$file" | tr -d ' '
}

require_line_count() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(count_lines "$file")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Preflight failed: $file has $actual lines, expected $expected." >&2
    return 1
  fi
}

require_seed_dirs() {
  local name="$1"
  local prefix="$2"
  local missing=0
  for seed in "${SEEDS[@]}"; do
    local ckpt="checkpoints/${prefix}${seed}/checkpoint_best.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "Preflight failed: missing $name checkpoint $ckpt" >&2
      missing=1
    fi
  done
  return "$missing"
}

preflight_historical() {
  local failed=0

  if has_task eng_laptop; then
    require_line_count data/subtask_3/eng/eng_laptop_train_alltasks_augmented.jsonl 15049 || failed=1
    require_seed_dirs eng_laptop v5_eng_laptop_seed || failed=1
  fi
  if has_task eng_restaurant; then
    require_line_count data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl 2284 || failed=1
    require_seed_dirs eng_restaurant v16_native_eng_restaurant_seed || failed=1
  fi
  if has_task jpn_hotel; then
    require_line_count data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl 1600 || failed=1
    require_seed_dirs jpn_hotel v16_ablation_jpn_no_aug_seed || failed=1
  fi
  if has_task rus_restaurant; then
    require_line_count data/subtask_3/rus/rus_restaurant_train_v13.jsonl 1766 || failed=1
    require_seed_dirs rus_restaurant v15_xlmr_rus_seed || failed=1
  fi
  if has_task tat_restaurant; then
    require_line_count data/subtask_3/tat/tat_restaurant_train_v13.jsonl 1599 || failed=1
    require_seed_dirs tat_restaurant v15_xlmr_tat_seed || failed=1
  fi
  if has_task ukr_restaurant; then
    require_line_count data/subtask_3/ukr/ukr_restaurant_train_v13b.jsonl 2967 || failed=1
    require_seed_dirs ukr_restaurant v13b_ukr_seed || failed=1
  fi
  if has_task zho_restaurant; then
    require_line_count data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl 6050 || failed=1
    require_seed_dirs zho_restaurant v16_native_zho_restaurant_seed || failed=1
  fi
  if has_task zho_laptop; then
    require_line_count data/subtask_3/zho/zho_laptop_train_v13c.jsonl 5534 || failed=1
    require_seed_dirs zho_laptop v16_native_zho_laptop_seed || failed=1
  fi

  if [[ "$failed" != "0" ]]; then
    echo
    echo "Historical reproduction is not ready." >&2
    echo "Restore the missing historical checkpoints/data, or set REPRO_MODE=retrain to run a fresh retrain that is not proof of the recorded best dev cF1." >&2
    exit 2
  fi
}

make_seed_config() {
  local base_cfg="$1"
  local seed="$2"
  local save_dir="$3"
  local tmp_cfg="/tmp/final_${seed}_$(basename "$base_cfg")"
  "$PY" - "$base_cfg" "$seed" "$save_dir" "$tmp_cfg" <<'PY'
import sys, yaml
base_cfg, seed, save_dir, tmp_cfg = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["seed"] = seed
cfg["training"]["save_dir"] = save_dir
with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(tmp_cfg)
PY
}

train_task() {
  local name="$1"
  local cfg="$2"
  local prefix="$3"
  echo "[$(date '+%F %T')] TRAIN $name prefix=$prefix" | tee -a logs/final_submission_4090.log
  for seed in "${SEEDS[@]}"; do
    local save_dir="checkpoints/${prefix}${seed}"
    if [[ -f "$save_dir/checkpoint_best.pt" ]]; then
      echo "[$(date '+%F %T')] SKIP $name seed=$seed existing $save_dir/checkpoint_best.pt" | tee -a logs/final_submission_4090.log
      continue
    fi
    local tmp_cfg
    tmp_cfg="$(make_seed_config "$cfg" "$seed" "$save_dir" | tail -n 1)"
    echo "[$(date '+%F %T')] RUN $name seed=$seed save_dir=$save_dir" | tee -a logs/final_submission_4090.log
    "$PY" -u scripts/train.py --config "$tmp_cfg" 2>&1 | tee "logs/final_train_${name}_seed${seed}.log"
    rm -f "$tmp_cfg"
  done
}

predict_task() {
  local name="$1"
  local cfg="$2"
  local test_file="$3"
  local prefix="$4"
  local vote="$5"
  local topk="$6"
  local dual_layer="$7"
  local out="submission/final/${name}_test.jsonl"
  local seed_dirs=()
  for seed in "${SEEDS[@]}"; do
    local d="checkpoints/${prefix}${seed}"
    if [[ ! -f "$d/checkpoint_best.pt" ]]; then
      echo "Missing checkpoint for $name seed=$seed: $d/checkpoint_best.pt" >&2
      exit 1
    fi
    seed_dirs+=("$d")
  done
  local args=(
    -u scripts/inference.py
    --multi_seed_ensemble
    --seed_dirs "${seed_dirs[@]}"
    --config "$cfg"
    --test_file "$test_file"
    --output "$out"
    --vote_threshold "$vote"
    --span_topk_threshold "$topk"
    --batch_size 16
  )
  if [[ "$dual_layer" == "1" ]]; then
    args+=(--dual_layer)
  fi
  echo "[$(date '+%F %T')] PREDICT $name -> $out" | tee -a logs/final_submission_4090.log
  "$PY" "${args[@]}" 2>&1 | tee "logs/final_predict_${name}.log"
  "$PY" - "$out" <<'PY' | tee -a logs/final_submission_4090.log
import json, sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
print(f"{path}: samples={len(rows)} quads={sum(len(r.get('Quadruplet', [])) for r in rows)}")
PY
}

if [[ "$REPRO_MODE" == "historical" ]]; then
  echo "[$(date '+%F %T')] HISTORICAL REPRODUCTION PREFLIGHT" | tee -a logs/final_submission_4090.log
  preflight_historical
  rm -f submission/final/*_backup.jsonl

  has_task eng_restaurant && predict_task eng_restaurant final_configs/eng_restaurant_v16.yaml data/subtask_3/eng/eng_restaurant_test_task3.jsonl v16_native_eng_restaurant_seed 4 0.0 0
  has_task eng_laptop && predict_task eng_laptop final_configs/eng_laptop_v5_bt.yaml data/subtask_3/eng/eng_laptop_test_task3.jsonl v5_eng_laptop_seed 3 0.0 1
  has_task jpn_hotel && predict_task jpn_hotel final_configs/jpn_hotel_v16_no_aug.yaml data/subtask_3/jpn/jpn_hotel_test_task3.jsonl v16_ablation_jpn_no_aug_seed 4 0.0 0
  has_task rus_restaurant && predict_task rus_restaurant final_configs/rus_restaurant_v15.yaml data/subtask_3/rus/rus_restaurant_test_task3.jsonl v15_xlmr_rus_seed 4 0.10 0
  has_task tat_restaurant && predict_task tat_restaurant final_configs/tat_restaurant_v15.yaml data/subtask_3/tat/tat_restaurant_test_task3.jsonl v15_xlmr_tat_seed 3 0.20 0
  has_task ukr_restaurant && predict_task ukr_restaurant final_configs/ukr_restaurant_v13b.yaml data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl v13b_ukr_seed 3 0.0 0
  has_task zho_restaurant && predict_task zho_restaurant final_configs/zho_restaurant_v16.yaml data/subtask_3/zho/zho_restaurant_test_task3.jsonl v16_native_zho_restaurant_seed 4 0.0 0
  has_task zho_laptop && predict_task zho_laptop final_configs/zho_laptop_v16.yaml data/subtask_3/zho/zho_laptop_test_task3.jsonl v16_native_zho_laptop_seed 4 0.10 0

  echo "[$(date '+%F %T')] HISTORICAL SUBMISSION DONE" | tee -a logs/final_submission_4090.log
  ls -lh submission/final | tee -a logs/final_submission_4090.log
  exit 0
elif [[ "$REPRO_MODE" != "retrain" ]]; then
  echo "Unknown REPRO_MODE=$REPRO_MODE. Use historical or retrain." >&2
  exit 2
fi

echo "[$(date '+%F %T')] RETRAIN MODE: this produces a fresh run, not proof of historical best reproduction." | tee -a logs/final_submission_4090.log

has_task eng_restaurant && train_task eng_restaurant final_configs/eng_restaurant_v16.yaml final_eng_restaurant_v16_seed
has_task eng_laptop && train_task eng_laptop final_configs/eng_laptop_v5_bt.yaml final_eng_laptop_v5_bt_seed
has_task jpn_hotel && train_task jpn_hotel final_configs/jpn_hotel_v16_no_aug.yaml final_jpn_hotel_v16_no_aug_seed
has_task rus_restaurant && train_task rus_restaurant final_configs/rus_restaurant_v15.yaml final_rus_restaurant_v15_seed
has_task tat_restaurant && train_task tat_restaurant final_configs/tat_restaurant_v15.yaml final_tat_restaurant_v15_seed
has_task ukr_restaurant && train_task ukr_restaurant final_configs/ukr_restaurant_v13b.yaml final_ukr_restaurant_v13b_seed
has_task zho_restaurant && train_task zho_restaurant final_configs/zho_restaurant_v16.yaml final_zho_restaurant_v16_seed
has_task zho_laptop && train_task zho_laptop final_configs/zho_laptop_v16.yaml final_zho_laptop_v16_seed

rm -f submission/final/*_backup.jsonl

has_task eng_restaurant && predict_task eng_restaurant final_configs/eng_restaurant_v16.yaml data/subtask_3/eng/eng_restaurant_test_task3.jsonl final_eng_restaurant_v16_seed 4 0.0 0
has_task eng_laptop && predict_task eng_laptop final_configs/eng_laptop_v5_bt.yaml data/subtask_3/eng/eng_laptop_test_task3.jsonl final_eng_laptop_v5_bt_seed 3 0.0 1
has_task jpn_hotel && predict_task jpn_hotel final_configs/jpn_hotel_v16_no_aug.yaml data/subtask_3/jpn/jpn_hotel_test_task3.jsonl final_jpn_hotel_v16_no_aug_seed 4 0.0 0
has_task rus_restaurant && predict_task rus_restaurant final_configs/rus_restaurant_v15.yaml data/subtask_3/rus/rus_restaurant_test_task3.jsonl final_rus_restaurant_v15_seed 4 0.10 0
has_task tat_restaurant && predict_task tat_restaurant final_configs/tat_restaurant_v15.yaml data/subtask_3/tat/tat_restaurant_test_task3.jsonl final_tat_restaurant_v15_seed 3 0.20 0
has_task ukr_restaurant && predict_task ukr_restaurant final_configs/ukr_restaurant_v13b.yaml data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl final_ukr_restaurant_v13b_seed 3 0.0 0
has_task zho_restaurant && predict_task zho_restaurant final_configs/zho_restaurant_v16.yaml data/subtask_3/zho/zho_restaurant_test_task3.jsonl final_zho_restaurant_v16_seed 4 0.0 0
has_task zho_laptop && predict_task zho_laptop final_configs/zho_laptop_v16.yaml data/subtask_3/zho/zho_laptop_test_task3.jsonl final_zho_laptop_v16_seed 4 0.10 0

echo "[$(date '+%F %T')] FINAL SUBMISSION DONE" | tee -a logs/final_submission_4090.log
ls -lh submission/final | tee -a logs/final_submission_4090.log
