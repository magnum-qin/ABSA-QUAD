#!/bin/bash
# Pool-source ablation: re-run the FAILED datasets with REAL public-corpus pools
# (eng_restaurant=Yelp, zho_laptop=amazon_zh+opencc) + minority silver, to test
# whether their KD failure was due to LLM-generated pools vs the real mechanism.
set -e
cd /home/g24tka12/ABSA-QUAD
OLU="http://localhost:11433"; QM="qwen3.6:latest"
RES=/tmp/real_pool_kd_results.txt; : > $RES

# name|dir|base|alltasks|domain|cfg|dev|pool|cap(0=all)|baseline
ROWS=(
"eng_restaurant|eng|data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl|data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl|restaurant|/tmp/pipeline_eng_restaurant_v16.yaml|data/subtask_3/eng/eng_restaurant_dev_task3.jsonl|data/subtask_3/eng/eng_restaurant_real_pool.jsonl|1800|0.6554"
"zho_laptop|zho|data/subtask_3/zho/zho_laptop_train_v13c.jsonl|data/subtask_3/zho/zho_laptop_train_alltasks.jsonl|laptop|/tmp/pipeline_zho_laptop_v16.yaml|data/subtask_3/zho/zho_laptop_dev_task3.jsonl|data/subtask_3/zho/zho_laptop_real_pool.jsonl|0|0.4487"
)

for row in "${ROWS[@]}"; do
  IFS='|' read -r name dir base alltasks domain cfg dev pool cap baseline <<< "$row"
  echo "##### $name REAL-pool KD ($(date)) #####" | tee -a $RES
  SILVER=data/subtask_3/$dir/${name}_silver_qwen_real.jsonl
  DISTILL=data/subtask_3/$dir/${name}_train_v23_qwen_real.jsonl
  V23CFG=/tmp/pipeline_${name}_v23_qwen_real.yaml

  echo "[$name 1/4] silver-label real pool ($(date))"
  OLLAMA_URL=$OLU conda run -n absa --no-capture-output python scripts/llm_silver_label_ollama.py \
    --input $pool --train_file $alltasks --output $SILVER \
    --model "$QM" --domain $domain --workers 2 --n_shots 6

  echo "[$name 2/4] unload qwen ($(date))"
  curl -s $OLU/api/generate -d "{\"model\":\"$QM\",\"keep_alive\":0}" >/dev/null 2>&1; sleep 6
  nvidia-smi --query-gpu=memory.used --format=csv,noheader

  echo "[$name 3/4] build distill (cap=$cap) + train ($(date))"
  python3 - "$base" "$SILVER" "$DISTILL" "$cap" <<'PY'
import json, sys, random
base, silver, out, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows=[json.loads(l) for l in open(base)]; n0=len(rows)
sv=[]
for l in open(silver):
    d=json.loads(l)
    if d.get('_err') or not d.get('Quadruplet'): continue
    sv.append({"ID":d['ID'],"Text":d['Text'],"Quadruplet":d['Quadruplet']})
if cap>0:
    random.seed(42); random.shuffle(sv); sv=sv[:cap]
rows+=sv
with open(out,'w') as f:
    for d in rows: f.write(json.dumps(d,ensure_ascii=False)+"\n")
print(f"  base={n0} + silver={len(sv)} = {len(rows)}  ratio={len(sv)/n0:.2f}")
PY
  python3 - "$cfg" "$DISTILL" "$V23CFG" <<'PY'
import sys, re
src, distill, dst = sys.argv[1:4]
t=open(src).read(); t=re.sub(r'(train_file:\s*)"[^"]*"', r'\1"%s"'%distill, t, count=1); open(dst,'w').write(t)
PY
  conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
    --config $V23CFG --seeds 42 123 456 789 1024 \
    --save_prefix v23_qwen_real_${name}_seed --train_file $DISTILL \
    > /tmp/v23_qwen_real_${name}_train.log 2>&1

  echo "[$name 4/4] ensemble eval ($(date))" | tee -a $RES
  SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_qwen_real_${name}_seed${s}"; done
  echo "=== $name REAL-pool KD (vs baseline $baseline) ===" | tee -a $RES
  for vote in 2 3 4; do for topk in 0.0 0.10; do
    out=/tmp/${name}_real_v${vote}_t${topk}.jsonl
    conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
      --seed_dirs $SD --config $V23CFG --test_file $dev --output $out \
      --vote_threshold $vote --span_topk_threshold $topk --batch_size 16 \
      > /tmp/${name}_real_eval_${vote}_${topk}.log 2>&1
    cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $dev -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
    echo "  $name real vote=$vote topk=$topk  ->  $cf1" | tee -a $RES
  done; done
  # re-warm qwen for next dataset
  curl -s $OLU/api/generate -d "{\"model\":\"$QM\",\"options\":{\"num_gpu\":99},\"keep_alive\":\"2h\"}" >/dev/null 2>&1
  echo "" | tee -a $RES
done
echo "##### REAL-pool KD ALL DONE ($(date)) #####" | tee -a $RES
