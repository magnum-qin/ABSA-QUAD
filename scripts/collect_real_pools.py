"""
Collect REAL public-corpus unlabeled pools for the KD pool-source ablation
(eng_restaurant via Yelp, zho_laptop via amazon_reviews_multi_zh + opencc s2twp).

Tests whether the 3 failed datasets failed because of LLM-generated pools vs a
real corpus. Pool stays a minority vs gold (capped downstream) to also control the
silver:gold drowning confound.
"""
import re, json, argparse
from datasets import load_dataset

RESTAURANT = re.compile(r'\b(restaurant|food|menu|waiter|waitress|dish|meal|brunch|dinner|lunch|appetizer|entree|server|tasty|delicious|ambiance|reservation)\b', re.I)
ZH_LAPTOP = re.compile(r'(笔记本|笔电|超极本|游戏本|轻薄本|散热|续航|开机速度|触控板|独显|固态|内存条|外接显卡)')


def collect_yelp(n, max_scan):
    # stream segfaults on this repo -> download parquet directly (robust)
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("Yelp/yelp_review_full",
                           "yelp_review_full/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["text"])
    pool = {}
    scan = 0
    for t in df["text"]:
        scan += 1
        if not isinstance(t, str):
            continue
        t = " ".join(t.split())
        if 20 < len(t) < 300 and RESTAURANT.search(t):
            pool[t[:60]] = t
        if len(pool) >= n or scan >= max_scan:
            break
    return list(pool.values()), scan


def collect_zho_laptop(n, max_scan):
    from opencc import OpenCC
    cc = OpenCC('s2twp')  # simplified -> traditional (Taiwan, with phrases)
    pool = {}
    ds = load_dataset("SetFit/amazon_reviews_multi_zh", split="train", streaming=True)
    scan = 0
    for r in ds:
        scan += 1
        t = " ".join(r.get("text", "").split())
        if 15 < len(t) < 200 and ZH_LAPTOP.search(t):
            trad = cc.convert(t)
            pool[trad[:50]] = trad
        if len(pool) >= n or scan >= max_scan:
            break
    return list(pool.values()), scan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["eng_restaurant", "zho_laptop"])
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--max_scan", type=int, default=400000)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.dataset == "eng_restaurant":
        texts, scan = collect_yelp(args.n, args.max_scan)
    else:
        texts, scan = collect_zho_laptop(args.n, args.max_scan)
    print(f"collected {len(texts)} real reviews (scanned {scan})", flush=True)

    pool = {t[:60]: t for t in texts}
    nt = 0
    for l in open(args.test_file, encoding="utf-8"):
        t = json.loads(l).get("Text", "").strip()
        if t and t[:60] not in pool:
            pool[t[:60]] = t; nt += 1
    with open(args.output, "w", encoding="utf-8") as f:
        for i, t in enumerate(pool.values()):
            f.write(json.dumps({"ID": f"unlab_{i}", "Text": t}, ensure_ascii=False) + "\n")
    print(f"+{nt} test texts -> wrote {len(pool)} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
