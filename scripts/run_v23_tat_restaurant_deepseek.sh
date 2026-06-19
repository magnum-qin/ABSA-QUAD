#!/bin/bash
# tat KD full pipeline: silver-label native pool -> distill data -> 5-seed -> ensemble eval
set -e
cd /home/g24tka12/ABSA-QUAD
KEY="REDACTED_DEEPSEEK_KEY"

# 1) append test transductive text to pool
conda run -n absa --no-capture-output python -c "
import json
pool={}
for l in open('data/subtask_3/tat/tat_native_pool.jsonl'):
    d=json.loads(l); pool[d['Text'][:60]]=d['Text']
tn=0
for l in open('data/subtask_3/tat/tat_restaurant_test_task3.jsonl'):
    t=json.loads(l).get('Text','').strip()
    if t and t[:60] not in pool: pool[t[:60]]=t; tn+=1
with open('data/subtask_3/tat/tat_unlabeled_pool.jsonl','w') as f:
    for i,t in enumerate(pool.values()): f.write(json.dumps({'ID':f'unlab_{i}','Text':t},ensure_ascii=False)+'\n')
print(f'pool={len(pool)} (test added {tn})')
"

# 2) silver label
conda run -n absa --no-capture-output env DEEPSEEK_API_KEY="$KEY" python -u scripts/llm_silver_label.py \
  --input data/subtask_3/tat/tat_unlabeled_pool.jsonl \
  --train_file data/subtask_3/tat/tat_restaurant_train_alltasks.jsonl \
  --output data/subtask_3/tat/tat_silver.jsonl \
  --model deepseek-v4-pro --domain "Tatar restaurant" \
  --workers 14 --n_shots 4

# 3) build distill data (V15 base data = v13 + silver)
conda run -n absa --no-capture-output python -c "
import json, random
random.seed(42)
best={}
for l in open('data/subtask_3/tat/tat_silver.jsonl'):
    d=json.loads(l)
    if '_err' in d or not d.get('Quadruplet'): continue
    best[d['ID']]={'ID':d['ID'],'Text':d['Text'],'Quadruplet':d['Quadruplet']}
silver=list(best.values())
base=[json.loads(l) for l in open('data/subtask_3/tat/tat_restaurant_train_v13.jsonl')]
comb=base+silver; random.shuffle(comb)
with open('data/subtask_3/tat/tat_restaurant_train_v23.jsonl','w') as f:
    for s in comb: f.write(json.dumps(s,ensure_ascii=False)+'\n')
print(f'v13={len(base)} silver={len(silver)} ({sum(len(s[\"Quadruplet\"]) for s in silver)} quads) total={len(comb)}')
"

# 4) config + 5-seed train (xlm-roberta-large, same as V15 tat)
sed -e 's|tat_restaurant_train_v13.jsonl|tat_restaurant_train_v23.jsonl|' \
    -e 's|v15_xlmr_tat|v23_silver_tat|' \
    /tmp/pipeline_tat_v15.yaml > /tmp/pipeline_tat_v23.yaml
echo "==== tat KD 5-seed training at $(date) ===="
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config /tmp/pipeline_tat_v23.yaml --seeds 42 123 456 789 1024 \
  --save_prefix v23_silver_tat_seed \
  --train_file data/subtask_3/tat/tat_restaurant_train_v23.jsonl \
  > /tmp/v23_silver_tat_train.log 2>&1

# 5) ensemble eval
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_silver_tat_seed${s}"; done
DEV=data/subtask_3/tat/tat_restaurant_dev_task3.jsonl
RES=/tmp/tat_kd_results.txt; : > $RES
echo "=== tat KD ensemble (vs V15 0.4821 / Takoyaki SOTA 0.4736) ===" | tee -a $RES
for vote in 2 3 4; do
  for topk in 0.0 0.20; do
    out=/tmp/tat_kd_v${vote}_t${topk}.jsonl
    conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
      --seed_dirs $SD --config /tmp/pipeline_tat_v23.yaml --test_file $DEV --output $out \
      --vote_threshold $vote --span_topk_threshold $topk --batch_size 16 > /tmp/tat_kd_eval_${vote}_${topk}.log 2>&1
    cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $DEV -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
    echo "  tat vote=$vote topk=$topk  →  $cf1" | tee -a $RES
  done
done
echo "===== tat KD PIPELINE DONE =====" | tee -a $RES
