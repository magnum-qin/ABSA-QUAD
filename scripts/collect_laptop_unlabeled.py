"""
Collect unlabeled English laptop-domain review text for LLM knowledge distillation (route D).

Source: naga-jay/amazon-laptop-reviews-enriched (Amazon laptop reviews, parquet-native)
        + eng_laptop test set (in-domain transductive).

Output: data/subtask_3/eng/eng_laptop_pool_v23.jsonl
  Each line: {"ID": "...", "Text": "..."}  (no labels; ready for LLM silver-labeling)
"""
import json, re
from pathlib import Path

LAPTOP_KW = re.compile(r'\b(laptop|notebook|keyboard|touchpad|trackpad|battery|screen|display|'
                       r'charger|ssd|ram|processor|cpu|gpu|fan|hinge|webcam|ultrabook|chromebook|'
                       r'macbook|thinkpad|bezel|backlit|trackpoint|resolution|boot|port|usb|wifi)\b', re.I)

TARGET = 8000
out_path = Path("data/subtask_3/eng/eng_laptop_pool_v23.jsonl")
pool = {}

from datasets import load_dataset
print("Streaming naga-jay/amazon-laptop-reviews-enriched ...", flush=True)
ds = load_dataset("naga-jay/amazon-laptop-reviews-enriched", split="train", streaming=True)
n = 0
for ex in ds:
    txt = (ex.get("text") or "").strip()
    txt = re.sub(r'\s+', ' ', txt)
    # keep concise review-sentence-like text in DimASQP length range
    if 25 < len(txt) < 260 and LAPTOP_KW.search(txt):
        key = txt.lower()[:80]
        if key not in pool:
            pool[key] = txt
            n += 1
            if n % 1000 == 0:
                print(f"  amazon: {n} collected", flush=True)
    if n >= TARGET:
        break
print(f"Amazon laptop collected: {n}", flush=True)

# In-domain transductive: eng_laptop test set
test_n = 0
for l in open("data/subtask_3/eng/eng_laptop_test_task3.jsonl", encoding="utf-8"):
    t = json.loads(l).get("Text", "").strip()
    if t:
        key = t.lower()[:80]
        if key not in pool:
            pool[key] = t
            test_n += 1
print(f"Test-set in-domain added: {test_n}", flush=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for i, txt in enumerate(pool.values()):
        f.write(json.dumps({"ID": f"unlab_{i}", "Text": txt}, ensure_ascii=False) + "\n")
print(f"\nWrote {len(pool)} unlabeled samples to {out_path}", flush=True)
