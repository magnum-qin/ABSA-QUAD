#!/bin/bash
# V23 KD for the 3 remaining datasets, teacher = local qwen3.6:latest (ollama @11433).
# Per dataset, sequentially (single 24GB GPU, qwen labeling and training cannot overlap):
#   1) qwen3.6 generates ~4000 native in-domain reviews + add test text -> unlabeled pool
#   2) qwen3.6 silver-labels the pool
#   3) unload qwen -> free GPU
#   4) build distill train = best-config base train + qwen silver
#   5) train 5-seed dual-layer ensemble with the dataset's V16 encoder
#   6) ensemble sweep eval on dev, compare to current best
set -e
cd /home/g24tka12/ABSA-QUAD
OLU="http://localhost:11433"
QM="qwen3.6:latest"

# dataset | dir | base_train | alltasks(categories) | domain | v16_config | dev | test | curr_best
DATASETS=(
"jpn_hotel|jpn|data/subtask_3/jpn/jpn_hotel_train_v13.jsonl|data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl|hotel|/tmp/pipeline_jpn_v16.yaml|data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl|data/subtask_3/jpn/jpn_hotel_test_task3.jsonl|0.4466"
"eng_restaurant|eng|data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl|data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl|restaurant|/tmp/pipeline_eng_restaurant_v16.yaml|data/subtask_3/eng/eng_restaurant_dev_task3.jsonl|data/subtask_3/eng/eng_restaurant_test_task3.jsonl|0.6554"
"zho_restaurant|zho|data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl|data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl|restaurant|/tmp/pipeline_zho_restaurant_v16.yaml|data/subtask_3/zho/zho_restaurant_dev_task3.jsonl|data/subtask_3/zho/zho_restaurant_test_task3.jsonl|0.4745"
)

RES=/tmp/qwen_kd_multi_results.txt; : > $RES

for row in "${DATASETS[@]}"; do
  IFS='|' read -r name dir base alltasks domain cfg dev test curr <<< "$row"
  echo "##################### $name  ($(date)) #####################" | tee -a $RES
  POOL=data/subtask_3/$dir/${name}_qwen_pool.jsonl
  NATIVE=data/subtask_3/$dir/${name}_qwen_native.jsonl
  SILVER=data/subtask_3/$dir/${name}_silver_qwen.jsonl
  DISTILL=data/subtask_3/$dir/${name}_train_v23_qwen.jsonl
  V23CFG=/tmp/pipeline_${name}_v23_qwen.yaml

  # 1) native generation + test text -> pool
  echo "[$name 1/6] gen native pool ($(date))"
  if [ ! -s "$POOL" ]; then
    OLLAMA_URL=$OLU conda run -n absa --no-capture-output python scripts/gen_native_ollama.py \
      --dataset $name --n 4000 --output $NATIVE --model "$QM" --workers 1
    python3 - "$NATIVE" "$test" "$POOL" <<'PY'
import json, sys
native, test, out = sys.argv[1:4]
pool = {}
for l in open(native):
    d = json.loads(l); pool[d['Text'][:60]] = d['Text']
nt = 0
for l in open(test):
    t = json.loads(l).get('Text', '').strip()
    if t and t[:60] not in pool:
        pool[t[:60]] = t; nt += 1
with open(out, 'w') as f:
    for i, t in enumerate(pool.values()):
        f.write(json.dumps({"ID": f"unlab_{i}", "Text": t}, ensure_ascii=False) + "\n")
print(f"  pool={len(pool)} (native+{nt} test)")
PY
  fi
  echo "[$name] pool rows: $(wc -l < $POOL)"

  # 2) qwen silver-labeling
  echo "[$name 2/6] qwen silver-labeling ($(date))"
  OLLAMA_URL=$OLU conda run -n absa --no-capture-output python scripts/llm_silver_label_ollama.py \
    --input $POOL --train_file $alltasks --output $SILVER \
    --model "$QM" --domain $domain --workers 2 --n_shots 6

  # 3) unload qwen -> free GPU
  echo "[$name 3/6] unload qwen ($(date))"
  curl -s $OLU/api/generate -d "{\"model\":\"$QM\",\"keep_alive\":0}" >/dev/null 2>&1; sleep 6
  nvidia-smi --query-gpu=memory.used --format=csv,noheader

  # 4) build distill train = base + non-empty silver
  echo "[$name 4/6] build distill train ($(date))"
  python3 - "$base" "$SILVER" "$DISTILL" <<'PY'
import json, sys
base, silver, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(base)]
n0 = len(rows)
ns = 0
for l in open(silver):
    d = json.loads(l)
    if d.get('_err') or not d.get('Quadruplet'):
        continue
    rows.append({"ID": d['ID'], "Text": d['Text'], "Quadruplet": d['Quadruplet']}); ns += 1
with open(out, 'w') as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"  base={n0} + silver={ns} = {len(rows)}")
PY

  # 5) V23 config (copy V16, swap train_file) + train 5 seeds
  echo "[$name 5/6] train 5 seeds ($(date))"
  python3 - "$cfg" "$DISTILL" "$V23CFG" <<'PY'
import sys, re
src, distill, dst = sys.argv[1:4]
txt = open(src).read()
txt = re.sub(r'(train_file:\s*)"[^"]*"', r'\1"%s"' % distill, txt, count=1)
open(dst, 'w').write(txt)
print(f"  wrote {dst}")
PY
  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $V23CFG --seeds 42 123 456 789 1024 \
    --save_prefix v23_qwen_${name}_seed --train_file $DISTILL \
    > /tmp/v23_qwen_${name}_train.log 2>&1

  # 6) ensemble eval sweep
  echo "[$name 6/6] ensemble eval ($(date))" | tee -a $RES
  SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_qwen_${name}_seed${s}"; done
  echo "=== $name qwen-KD ensemble (vs current best $curr) ===" | tee -a $RES
  for vote in 2 3 4; do
    for topk in 0.0 0.10; do
      out=/tmp/${name}_qwen_kd_v${vote}_t${topk}.jsonl
      conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
        --seed_dirs $SD --config $V23CFG --test_file $dev --output $out \
        --vote_threshold $vote --span_topk_threshold $topk --batch_size 16 \
        > /tmp/${name}_qwen_eval_${vote}_${topk}.log 2>&1
      cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $dev -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
      echo "  $name qwen vote=$vote topk=$topk  ->  $cf1" | tee -a $RES
    done
  done
  echo "" | tee -a $RES
done
echo "##################### ALL QWEN-KD MULTI DONE ($(date)) #####################" | tee -a $RES
