"""
Collect unlabeled in-domain review text for KD on non-English datasets (route D extension).

Targets:
  zho_laptop     : SetFit/amazon_reviews_multi_zh, keyword-filtered to laptop/computer
  rus_restaurant : d0rj/geo-reviews-dataset-2023, rubric-filtered to restaurant/cafe
  (ukr handled separately: test-set transductive only)

Output: data/subtask_3/<lang>/<dataset>_unlabeled_pool.jsonl  ({ID, Text})
Always appends the dataset's own test-set text (in-domain transductive).
"""
import json, re, argparse
from pathlib import Path
from datasets import load_dataset

ZH_KW = re.compile(r'(笔记本|电脑|笔电|本子|游戏本|超极本|键盘|触控板|屏幕|显卡|内存|硬盘|固态|处理器|续航|电池|散热|开机|系统|轻薄)')
RU_REST_RUBRIC = re.compile(r'(ресторан|кафе|кофейн|бар|пиццери|столов|食|еда|кухн|суши|пельмен)', re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["zho_laptop", "rus_restaurant"])
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_n", type=int, default=8000)
    args = ap.parse_args()

    pool = {}
    if args.target == "zho_laptop":
        print("Streaming SetFit/amazon_reviews_multi_zh ...", flush=True)
        for split in ("train", "test"):
            try:
                ds = load_dataset("SetFit/amazon_reviews_multi_zh", split=split, streaming=True)
            except Exception as e:
                print(f"  {split} fail: {e}"); continue
            for ex in ds:
                t = re.sub(r'\s+', ' ', (ex.get("text") or "")).strip()
                if 15 < len(t) < 200 and ZH_KW.search(t):
                    pool[t[:60]] = t
                if len(pool) >= args.target_n:
                    break
            if len(pool) >= args.target_n:
                break
        print(f"  zh laptop collected: {len(pool)}", flush=True)

    elif args.target == "rus_restaurant":
        print("Streaming d0rj/geo-reviews-dataset-2023 ...", flush=True)
        ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train", streaming=True)
        for ex in ds:
            rub = (ex.get("rubrics") or "")
            t = re.sub(r'\s+', ' ', (ex.get("text") or "")).strip()
            if 20 < len(t) < 260 and RU_REST_RUBRIC.search(rub):
                pool[t[:60]] = t
            if len(pool) >= args.target_n:
                break
        print(f"  ru restaurant collected: {len(pool)}", flush=True)

    # Append in-domain test text (transductive)
    tn = 0
    for l in open(args.test_file, encoding="utf-8"):
        t = json.loads(l).get("Text", "").strip()
        if t and t[:60] not in pool:
            pool[t[:60]] = t
            tn += 1
    print(f"  test-set transductive added: {tn}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, t in enumerate(pool.values()):
            f.write(json.dumps({"ID": f"unlab_{i}", "Text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pool)} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
