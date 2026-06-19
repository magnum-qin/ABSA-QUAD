"""
LLM silver-labeling via a LOCAL ollama model (route D, local-teacher variant).

Mirror of scripts/llm_silver_label.py but the teacher is a locally-served model
through ollama's native /api/chat endpoint instead of the DeepSeek cloud API.
Built to answer: "does distilling from a local 31B (gemma-4-31B-it-qat) match
distilling from cloud DeepSeek?" — same pool, same validation, same downstream
KD config, only the teacher differs.

Key ollama specifics (vs DeepSeek):
  - gemma-4-qat is a *reasoning* model: it emits a thinking channel that ollama
    routes to message.thinking, leaving message.content EMPTY unless given a huge
    token budget. We pass "think": false to suppress CoT and get clean JSON.
  - num_gpu:99 forces all layers onto the GPU (ollama otherwise CPU-offloads if
    the GPU looked busy at first load → ~10x slower).
  - The model wraps JSON in ```json fences; we strip them before parsing.

Validation, resume-safety, few-shot, and output format are identical to the
DeepSeek script so the silver files are drop-in interchangeable.

Usage:
    python scripts/llm_silver_label_ollama.py \
        --input data/subtask_3/eng/eng_laptop_unlabeled_pool.jsonl \
        --train_file data/subtask_3/eng/eng_laptop_train_alltasks.jsonl \
        --output data/subtask_3/eng/eng_laptop_silver_gemma.jsonl \
        --model "hf.co/unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL" \
        --domain laptop --workers 4 --n_shots 6
"""
import os
import sys
import json
import re
import argparse
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

_print_lock = threading.Lock()
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def build_category_list(train_file):
    cats = Counter()
    for l in open(train_file, encoding="utf-8"):
        for q in json.loads(l).get("Quadruplet", []):
            cats[q["Category"]] += 1
    return [c for c, _ in cats.most_common()]


def build_system_prompt(valid_categories, domain="product"):
    cat_str = ", ".join(valid_categories)
    return f"""You are an expert at Aspect-Based Sentiment Analysis for {domain} reviews.
Given a review, extract every opinion as a quadruplet (Aspect, Category, Opinion, VA).

Rules:
1. Aspect = the target term in the text (e.g. "keyboard", "battery"). Use "NULL" if the opinion targets the {domain} implicitly with no explicit aspect word.
2. Opinion = the evaluative expression in the text (e.g. "fast", "too hot"). Use "NULL" if absent.
3. Aspect and Opinion (when not NULL) MUST be verbatim substrings of the review text — do not paraphrase.
4. Category MUST be exactly one of these {len(valid_categories)} valid categories:
{cat_str}
5. VA = "Valence#Arousal", both floats in 1.0-9.0. Valence: 1=very negative, 5=neutral, 9=very positive. Arousal: 1=calm, 9=excited/intense.
6. Output JSON ONLY, no markdown: {{"Quadruplet": [{{"Aspect":"...","Category":"...","Opinion":"...","VA":"7.0#6.0"}}, ...]}}
7. If the review has no {domain}-related opinion, output {{"Quadruplet": []}}."""


def build_user_prompt(text, examples=None):
    shots = ""
    if examples:
        for ex in examples:
            shots += f"Review: {ex['Text']}\nOutput: {json.dumps({'Quadruplet': ex['Quadruplet']}, ensure_ascii=False)}\n\n"
    return f"{shots}Review: {text}\nOutput:"


def call_ollama(prompt, sys_prompt, model, timeout=180):
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "think": False,  # suppress gemma-qat reasoning channel -> non-empty content
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_gpu": 99, "temperature": 0.3, "num_predict": 1024},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    content = (r.json().get("message") or {}).get("content") or ""
    m = _FENCE.search(content)
    if m:
        content = m.group(1)
    return content


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
        s = q.get(f)
        if s is None:  # some teachers (qwen3.6) emit JSON null instead of "NULL"
            s = "NULL"
        if s != "NULL" and str(s).lower() not in tl:
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
    p.add_argument("--input", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="hf.co/unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL")
    p.add_argument("--domain", default="product")
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    valid_cats = build_category_list(args.train_file)
    sys_prompt = build_system_prompt(valid_cats, args.domain)

    train = [json.loads(l) for l in open(args.train_file, encoding="utf-8")]
    examples = [t for t in train if t.get("Quadruplet") and len(t["Text"]) < 160][: args.n_shots]

    pool = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        pool = pool[: args.limit]

    done = load_resume(args.output)
    pending = [s for s in pool if s["ID"] not in done or "_err" in done.get(s["ID"], {})]
    print(f"Pool: {len(pool)}, already done: {len(done)}, pending: {len(pending)}, "
          f"valid_cats: {len(valid_cats)}, workers: {args.workers}, model: {args.model}", flush=True)

    out_f = open(args.output, "a", encoding="utf-8")
    write_lock = threading.Lock()
    stats = Counter()

    def work(sample):
        text = sample["Text"]
        prompt = build_user_prompt(text, examples)
        try:
            raw = call_ollama(prompt, sys_prompt, args.model)
            data = json.loads(raw)
            quads = data.get("Quadruplet", [])
            valid = [q for q in quads if validate_quad(q, text, valid_cats)]
            clean = [{"Aspect": q["Aspect"] if q.get("Aspect") is not None else "NULL",
                      "Category": q["Category"],
                      "Opinion": q["Opinion"] if q.get("Opinion") is not None else "NULL",
                      "VA": q["VA"]} for q in valid]
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
            if n_done % 100 == 0:
                with _print_lock:
                    print(f"  [{n_done}/{len(pending)}] raw={stats['raw']} kept={stats['kept']} "
                          f"err={stats['err']} keep_rate={stats['kept']/max(1,stats['raw']):.2f}", flush=True)

    out_f.close()
    print(f"\nDone. {n_done} processed. raw_quads={stats['raw']} kept_quads={stats['kept']} "
          f"errors={stats['err']}", flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
