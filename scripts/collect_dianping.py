import json, re
from datasets import load_dataset
from opencc import OpenCC
cc = OpenCC('s2twp')  # simplified -> traditional (matches zho_restaurant test text)
FOOD = re.compile(r'(味道|菜|吃|餐廳|餐厅|口味|服務|服务|环境|環境|點|点菜|好吃|難吃|难吃|份量|分量|价格|價格|店|喝|飲|饮|甜|辣|鮮|鲜|香)')
pool = {}
ds = load_dataset("twn39/dianping", split="train", streaming=True)
scan = 0
for r in ds:
    scan += 1
    t = r.get("sentence", "")
    if not isinstance(t, str): continue
    t = " ".join(t.split())
    if 15 < len(t) < 200 and FOOD.search(t):
        pool[t[:50]] = cc.convert(t)
    if len(pool) >= 4000 or scan >= 200000: break
print(f"collected {len(pool)} (scanned {scan})", flush=True)
# add zho_restaurant test text (already traditional)
out = {f"unlab_{i}": t for i, t in enumerate(pool.values())}
nt = 0
seen = set(out.values())
for l in open("data/subtask_3/zho/zho_restaurant_test_task3.jsonl", encoding="utf-8"):
    t = json.loads(l).get("Text", "").strip()
    if t and t not in seen:
        out[f"unlab_t{nt}"] = t; nt += 1
with open("data/subtask_3/zho/zho_restaurant_pool_v23_real.jsonl", "w", encoding="utf-8") as f:
    for k, t in out.items():
        f.write(json.dumps({"ID": k, "Text": t}, ensure_ascii=False) + "\n")
print(f"+{nt} test -> wrote {len(out)} to zho_restaurant_pool_v23_real.jsonl", flush=True)
