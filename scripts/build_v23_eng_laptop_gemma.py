"""
Build eng_laptop_train_v23_gemma.jsonl — identical recipe to train_v23.jsonl but
the silver portion comes from the LOCAL gemma-4-31B teacher instead of DeepSeek.

train_v23 = [gold + 4-pivot BT  (non-`unlab_` rows)]  +  [DeepSeek silver (`unlab_`)]
train_v23_gemma = [SAME gold + BT base, verbatim]      +  [gemma silver (`unlab_`)]

Only the teacher differs; the gold/BT base is byte-identical, so a downstream cF1
delta isolates teacher quality (cloud DeepSeek vs local gemma).
"""
import json

V23 = "data/subtask_3/eng/eng_laptop_train_v23_deepseek.jsonl"
GEMMA_SILVER = "data/subtask_3/eng/eng_laptop_silver_v23_gemma.jsonl"
OUT = "data/subtask_3/eng/eng_laptop_train_v23_gemma.jsonl"

# 1. fixed base = everything in train_v23 that is NOT silver (no `unlab_` prefix)
base = []
for l in open(V23):
    d = json.loads(l)
    if not str(d["ID"]).startswith("unlab_"):
        base.append(d)

# 2. gemma silver = non-empty, non-error rows
gem = []
for l in open(GEMMA_SILVER):
    d = json.loads(l)
    if d.get("_err"):
        continue
    if d.get("Quadruplet"):
        gem.append({"ID": d["ID"], "Text": d["Text"], "Quadruplet": d["Quadruplet"]})

with open(OUT, "w", encoding="utf-8") as f:
    for d in base + gem:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"base(gold+BT)={len(base)}  gemma_silver={len(gem)}  total={len(base)+len(gem)}")
print(f"(DeepSeek arm was base=15049 + silver=8541 = 23590)")
print(f"wrote {OUT}")
