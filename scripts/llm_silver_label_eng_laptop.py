"""
LLM silver-labeling for eng_laptop knowledge distillation (route D).

Ask DeepSeek to produce DimASQP quads (Aspect, Category, Opinion, VA) for each
unlabeled laptop review. Validated silver labels become extra training data so a
small DeBERTa-v3-base model can absorb the LLM's priors for the sparse 24x10
category space — without any LLM at inference time (unlike the failed §5.20.4
LLM-fusion and §7.7.1 post-correction experiments).

Validation (drop quad unless ALL hold):
  - Category appears in the valid category set built from eng_laptop train
  - Aspect and Opinion are NULL or verbatim substrings of the text
  - VA parses as V#A with both in [1, 9]

Resume-safe: re-running skips IDs already written. Failures recorded with _err.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/llm_silver_label_eng_laptop.py \
        --input data/subtask_3/eng/eng_laptop_pool_v23.jsonl \
        --train_file data/subtask_3/eng/eng_laptop_train_alltasks.jsonl \
        --output data/subtask_3/eng/eng_laptop_silver_v23_deepseek.jsonl \
        --model deepseek-chat --workers 12 --n_shots 6
"""
import os
import sys
import json
import re
import time
import argparse
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

_print_lock = threading.Lock()


def build_category_list(train_file):
    cats = Counter()
    for l in open(train_file, encoding="utf-8"):
        for q in json.loads(l).get("Quadruplet", []):
            cats[q["Category"]] += 1
    return [c for c, _ in cats.most_common()]


def build_system_prompt(valid_categories):
    cat_str = ", ".join(valid_categories)
    return f"""You are an expert at Aspect-Based Sentiment Analysis for laptop product reviews.
Given a review, extract every opinion as a quadruplet (Aspect, Category, Opinion, VA).

Rules:
1. Aspect = the target term in the text (e.g. "keyboard", "battery"). Use "NULL" if the opinion targets the laptop implicitly with no explicit aspect word.
2. Opinion = the evaluative expression in the text (e.g. "fast", "too hot"). Use "NULL" if absent.
3. Aspect and Opinion (when not NULL) MUST be verbatim substrings of the review text — do not paraphrase.
4. Category MUST be exactly one of these {len(valid_categories)} valid categories:
{cat_str}
5. VA = "Valence#Arousal", both floats in 1.0-9.0. Valence: 1=very negative, 5=neutral, 9=very positive. Arousal: 1=calm, 9=excited/intense.
6. Output JSON ONLY, no markdown: {{"Quadruplet": [{{"Aspect":"...","Category":"...","Opinion":"...","VA":"7.0#6.0"}}, ...]}}
7. If the review has no laptop-related opinion, output {{"Quadruplet": []}}."""


def build_user_prompt(text, examples=None):
    shots = ""
    if examples:
        for ex in examples:
            shots += f"Review: {ex['Text']}\nOutput: {json.dumps({'Quadruplet': ex['Quadruplet']}, ensure_ascii=False)}\n\n"
    return f"{shots}Review: {text}\nOutput:"


def call_deepseek(prompt, sys_prompt, api_key, model="deepseek-chat"):
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=180)
    # Disable thinking/CoT: v4-pro defaults to reasoning mode, which emits ~2200
    # output tokens/call (mostly hidden chain-of-thought). For structured quad
    # extraction we only need the JSON (~200 tokens), so we turn thinking off
    # → ~10x cheaper output, faster, no truncation risk.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message.content


def validate_quad(q, text, valid_cats):
    if not isinstance(q, dict):
        return False
    cat = q.get("Category", "")
    if cat not in valid_cats:
        return False
    va = q.get("VA", "")
    try:
        v, a = va.split("#")
        v, a = float(v), float(a)
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return False
    except Exception:
        return False
    tl = text.lower()
    for f in ("Aspect", "Opinion"):
        s = q.get(f, "")
        if s != "NULL" and s.lower() not in tl:
            return False
    return True


def load_resume(path):
    done = {}
    if Path(path).exists():
        for l in open(path, encoding="utf-8"):
            try:
                d = json.loads(l)
                done[d["ID"]] = d
            except Exception:
                pass
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Unlabeled pool jsonl ({ID, Text})")
    p.add_argument("--train_file", required=True, help="For building valid category list + few-shot examples")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--api_key", default=None)
    args = p.parse_args()

    if OpenAI is None:
        print("ERROR: pip install openai", file=sys.stderr); sys.exit(1)
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY", file=sys.stderr); sys.exit(1)

    valid_cats = build_category_list(args.train_file)
    sys_prompt = build_system_prompt(valid_cats)

    # Few-shot examples: short, non-empty quad samples from train
    train = [json.loads(l) for l in open(args.train_file, encoding="utf-8")]
    examples = [t for t in train if t.get("Quadruplet") and len(t["Text"]) < 160][: args.n_shots]

    pool = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        pool = pool[: args.limit]

    done = load_resume(args.output)
    pending = [s for s in pool if s["ID"] not in done or "_err" in done.get(s["ID"], {})]
    print(f"Pool: {len(pool)}, already done: {len(done)}, pending: {len(pending)}, "
          f"valid_cats: {len(valid_cats)}, workers: {args.workers}", flush=True)

    out_f = open(args.output, "a", encoding="utf-8")
    write_lock = threading.Lock()
    stats = Counter()

    def work(sample):
        text = sample["Text"]
        prompt = build_user_prompt(text, examples)
        try:
            raw = call_deepseek(prompt, sys_prompt, api_key, args.model)
            data = json.loads(raw)
            quads = data.get("Quadruplet", [])
            valid = [q for q in quads if validate_quad(q, text, valid_cats)]
            # normalize fields
            clean = [{"Aspect": q["Aspect"], "Category": q["Category"],
                      "Opinion": q["Opinion"], "VA": q["VA"]} for q in valid]
            return {"ID": sample["ID"], "Text": text, "Quadruplet": clean}, len(quads), len(clean)
        except Exception as e:
            return {"ID": sample["ID"], "Text": text, "Quadruplet": [], "_err": str(e)[:80]}, 0, 0

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in pending}
        for fut in as_completed(futs):
            rec, n_raw, n_kept = fut.result()
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            stats["raw"] += n_raw
            stats["kept"] += n_kept
            if "_err" in rec:
                stats["err"] += 1
            n_done += 1
            if n_done % 200 == 0:
                with _print_lock:
                    print(f"  [{n_done}/{len(pending)}] raw={stats['raw']} kept={stats['kept']} "
                          f"err={stats['err']} keep_rate={stats['kept']/max(1,stats['raw']):.2f}", flush=True)

    out_f.close()
    print(f"\nDone. {n_done} processed. raw_quads={stats['raw']} kept_quads={stats['kept']} "
          f"errors={stats['err']}", flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
