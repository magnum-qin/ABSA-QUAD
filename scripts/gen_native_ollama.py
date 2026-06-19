"""
Generate native in-domain reviews with a LOCAL ollama model (qwen3.6) for KD pools.

Same idea as gen_native_{ukr,tat}_reviews.py (DeepSeek) but the generator is the
local qwen3.6 teacher via ollama's /api/chat with think:false. Used to build the
unlabeled pool for datasets that have no in-domain corpus (eng_restaurant,
jpn_hotel, zho_restaurant). Each call asks for N short diverse reviews.

Usage:
    OLLAMA_URL=http://localhost:11433 python scripts/gen_native_ollama.py \
        --dataset jpn_hotel --n 4000 --output <pool.jsonl> --workers 1
"""
import os, sys, json, argparse, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11433")

# Per-dataset generation spec: (system prompt, user template, aspects, tones)
SPECS = {
    "eng_restaurant": {
        "sys": ("You generate realistic, natural English restaurant/cafe reviews, like real "
                "Yelp/Google reviews. Write in lively colloquial English. Each review 1-3 "
                'sentences. Return ONLY JSON: {"reviews": ["...", "..."]}'),
        "user": ('Generate {k} different short English restaurant reviews. Focus on: {focus}. '
                 'Tone: {tone}. Avoid identical phrasing. Return JSON {{"reviews":[...]}}.'),
        "aspects": ["food", "service", "ambiance", "price", "drinks", "dessert", "portion size",
                    "menu", "location", "staff", "wait time", "cleanliness", "wine list", "decor"],
        "tones": ["enthusiastic positive", "mildly positive", "mixed", "disappointed negative", "neutral"],
    },
    "jpn_hotel": {
        "sys": ("あなたはホテルのリアルで自然な日本語の口コミを生成するアシスタントです。"
                "楽天トラベルやじゃらんの実際の口コミのように、生き生きとした話し言葉で書いてください。"
                '各口コミは1〜3文。JSONのみ返す: {"reviews": ["...", "..."]}'),
        "user": ('ホテルについて{k}件の異なる短い日本語の口コミを生成してください。'
                 '焦点: {focus}。トーン: {tone}。同じ表現を繰り返さないこと。JSON {{"reviews":[...]}} を返す。'),
        "aspects": ["部屋", "朝食", "接客・サービス", "立地・アクセス", "大浴場・風呂", "清潔さ",
                    "コストパフォーマンス", "設備・アメニティ", "周辺環境", "ベッド・寝具",
                    "フロント対応", "価格", "眺望", "駐車場"],
        "tones": ["とても満足", "やや満足", "賛否両論", "がっかり・不満", "普通・中立"],
    },
    "zho_laptop": {
        "sys": ("你是生成真實、自然的繁體中文筆記型電腦（筆電）評價的助手。像 PChome、Mobile01 上真實用戶寫的那樣，"
                '用生動口語化的繁體中文。每條評價1-3句。只返回JSON: {"reviews": ["...", "..."]}'),
        "user": ('生成{k}條不同的繁體中文筆電短評。側重: {focus}。語氣: {tone}。'
                 '避免重複的表達。返回JSON {{"reviews":[...]}}。'),
        "aspects": ["螢幕", "電池續航", "鍵盤", "效能", "散熱", "做工質感", "價格", "作業系統",
                    "觸控板", "外觀", "重量", "喇叭音質", "連接埠", "開機速度", "風扇噪音", "顯卡"],
        "tones": ["非常滿意", "比較滿意", "褒貶不一", "失望不滿", "一般中立"],
    },
    "zho_restaurant": {
        "sys": ("你是生成真实、自然的中文餐厅/咖啡馆评价的助手。像大众点评、美团上真实用户写的那样，"
                '用生动的口语化中文。每条评价1-3句。只返回JSON: {"reviews": ["...", "..."]}'),
        "user": ('生成{k}条不同的中文餐厅短评。侧重: {focus}。语气: {tone}。'
                 '避免重复的表达。返回JSON {{"reviews":[...]}}。'),
        "aspects": ["味道", "服务", "环境", "价格", "分量", "菜品", "卫生", "性价比", "位置",
                    "排队等位", "招牌菜", "饮料", "装修", "上菜速度"],
        "tones": ["非常满意", "比较满意", "褒贬不一", "失望不满", "一般中立"],
    },
}


def call(model, sys_p, user_p, timeout=180):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model, "think": False,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "stream": False, "options": {"temperature": 1.0, "num_predict": 1024, "num_gpu": 99},
    }, timeout=timeout)
    r.raise_for_status()
    c = (r.json().get("message") or {}).get("content") or ""
    if "```" in c:
        import re
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", c, re.S)
        if m:
            c = m.group(1)
    try:
        return json.loads(c).get("reviews", [])
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(SPECS))
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="qwen3.6:latest")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--per_call", type=int, default=8)
    args = p.parse_args()

    spec = SPECS[args.dataset]
    n_calls = args.n // args.per_call + 1
    jobs = [(random.choice(spec["aspects"]), random.choice(spec["tones"])) for _ in range(n_calls)]

    pool = {}
    done = 0
    lock = threading.Lock()

    def work(job):
        focus, tone = job
        user_p = spec["user"].format(k=args.per_call, focus=focus, tone=tone)
        return call(args.model, spec["sys"], user_p)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            for rev in fut.result():
                rev = " ".join(str(rev).split())
                if 12 < len(rev) < 300:
                    pool[rev[:60]] = rev
            done += 1
            if done % 20 == 0:
                with lock:
                    print(f"  [{done}/{n_calls} calls] unique reviews={len(pool)}", flush=True)
            if len(pool) >= args.n:
                break

    with open(args.output, "w", encoding="utf-8") as f:
        for i, t in enumerate(pool.values()):
            f.write(json.dumps({"ID": f"gen_{i}", "Text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pool)} native {args.dataset} reviews to {args.output}", flush=True)


if __name__ == "__main__":
    main()
