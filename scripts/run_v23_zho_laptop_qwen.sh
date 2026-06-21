#!/bin/bash
# V23 KD zho_laptop, teacher = local qwen3.6. Silver CAPPED to keep silver:gold < 1
# (replicate the zho_restaurant winning condition: silver stays a minority so it adds
# diversity to the sparse 113-category space without drowning the gold).
#   base = zho_laptop_train_v13c (5534, V16 best recipe) + capped qwen silver (2700)
set -e
cd /home/g24tka12/ABSA-QUAD
OLU="http://localhost:11433"; QM="qwen3.6:latest"
CAP=2700
RES=/tmp/zho_laptop_qwen_kd_results.txt; : > $RES

POOL=data/subtask_3/zho/zho_laptop_pool_v23_gen.jsonl
NATIVE=data/subtask_3/zho/zho_laptop_pool_v23_gen_raw.jsonl
SILVER=data/subtask_3/zho/zho_laptop_silver_v23_qwen.jsonl
DISTILL=data/subtask_3/zho/zho_laptop_train_v23_qwen.jsonl
BASE=data/subtask_3/zho/zho_laptop_train_v13c.jsonl
ALLTASKS=data/subtask_3/zho/zho_laptop_train_alltasks.jsonl
CFG=/tmp/pipeline_zho_laptop_v16.yaml
V23CFG=configs/pipeline_zho_laptop_v23_qwen.yaml
DEV=data/subtask_3/zho/zho_laptop_dev_task3.jsonl
TEST=data/subtask_3/zho/zho_laptop_test_task3.jsonl

echo "##### zho_laptop qwen-KD (capped silver) $(date) #####" | tee -a $RES

# 1) native pool + test text
echo "[1/6] gen native pool ($(date))"
if [ ! -s "$POOL" ]; then
  OLLAMA_URL=$OLU conda run -n absa --no-capture-output python scripts/gen_native_ollama.py \
    --dataset zho_laptop --n 4000 --output $NATIVE --model "$QM" --workers 1
  python3 - "$NATIVE" "$TEST" "$POOL" <<'PY'
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
echo "[zho_laptop] pool rows: $(wc -l < $POOL)"

# 2) qwen silver-labeling (domain=laptop)
echo "[2/6] qwen silver-labeling ($(date))"
OLLAMA_URL=$OLU conda run -n absa --no-capture-output python scripts/llm_silver_label_ollama.py \
  --input $POOL --train_file $ALLTASKS --output $SILVER \
  --model "$QM" --domain laptop --workers 2 --n_shots 6

# 3) unload qwen
echo "[3/6] unload qwen ($(date))"
curl -s $OLU/api/generate -d "{\"model\":\"$QM\",\"keep_alive\":0}" >/dev/null 2>&1; sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 4) build distill = base + CAPPED non-empty silver (deterministic shuffle)
echo "[4/6] build distill (cap silver=$CAP) ($(date))"
python3 - "$BASE" "$SILVER" "$DISTILL" "$CAP" <<'PY'
import json, sys, random
base, silver, out, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(base)]; n0 = len(rows)
sv = []
for l in open(silver):
    d = json.loads(l)
    if d.get('_err') or not d.get('Quadruplet'): continue
    sv.append({"ID": d['ID'], "Text": d['Text'], "Quadruplet": d['Quadruplet']})
random.seed(42); random.shuffle(sv)
sv = sv[:cap]
rows += sv
with open(out, 'w') as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"  base={n0} + silver(capped)={len(sv)} = {len(rows)}  ratio={len(sv)/n0:.2f}")
PY

# 5) V23 config + train 5 seeds
echo "[5/6] train 5 seeds ($(date))"
python3 - "$CFG" "$DISTILL" "$V23CFG" <<'PY'
import sys, re
src, distill, dst = sys.argv[1:4]
txt = open(src).read()
txt = re.sub(r'(train_file:\s*)"[^"]*"', r'\1"%s"' % distill, txt, count=1)
open(dst, 'w').write(txt)
print(f"  wrote {dst}")
PY
conda run -n absa --no-capture-output python -u scripts/train_multi_seed.py \
  --config $V23CFG --seeds 42 123 456 789 1024 \
  --save_prefix v23_zho_laptop_qwen_seed --train_file $DISTILL \
  > /tmp/v23_zho_laptop_qwen_train.log 2>&1

# 6) ensemble eval sweep
echo "[6/6] ensemble eval ($(date))" | tee -a $RES
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/v23_zho_laptop_qwen_seed${s}"; done
echo "=== zho_laptop qwen-KD ensemble (vs V16 0.4487) ===" | tee -a $RES
for vote in 2 3 4; do for topk in 0.0 0.10; do
  out=/tmp/zho_laptop_qwen_kd_v${vote}_t${topk}.jsonl
  conda run -n absa --no-capture-output python -u scripts/inference.py --multi_seed_ensemble \
    --seed_dirs $SD --config $V23CFG --test_file $DEV --output $out \
    --vote_threshold $vote --span_topk_threshold $topk --batch_size 16 \
    > /tmp/zho_laptop_qwen_eval_${vote}_${topk}.log 2>&1
  cf1=$(conda run -n absa --no-capture-output python scripts/official_eval.py -p $out -g $DEV -t 3 2>/dev/null | grep -oE "cF1: [0-9.]+" | tail -1)
  echo "  zho_laptop qwen vote=$vote topk=$topk  ->  $cf1" | tee -a $RES
done; done
echo "##### zho_laptop qwen-KD DONE $(date) #####" | tee -a $RES
