"""
LLM silver-labeling via any OpenAI-COMPATIBLE endpoint (Unsloth Studio, vLLM, etc.).

Same validation/output as llm_silver_label{,_ollama}.py, but talks to a /v1/chat/completions
endpoint with a Bearer API key. Built for Unsloth Studio (http://localhost:8888/v1,
key sk-unsloth-..., model loaded in the Studio UI).

Reasoning models (Qwen3): we disable the thinking channel via
  extra_body={"chat_template_kwargs": {"enable_thinking": false}}
and also strip any leftover <think>...</think> from the content before JSON parse.

Usage:
  python scripts/llm_silver_label_openai.py \
    --input <pool>.jsonl --train_file <ds>_train_alltasks.jsonl --output <ds>_silver_v23_qwen.jsonl \
    --base_url http://localhost:8888/v1 --api_key sk-unsloth-xxx --model "<model-id>" \
    --domain restaurant --workers 4 --n_shots 6
"""
import os, sys, json, re, argparse, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

_print_lock = threading.Lock()
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


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
1. Aspect = the target term in the text. Use "NULL" if the opinion targets the {domain} implicitly.
2. Opinion = the evaluative expression in the text. Use "NULL" if absent.
3. Aspect and Opinion (when not NULL) MUST be verbatim substrings of the review text.
4. Category MUST be exactly one of these {len(valid_categories)} valid categories:
{cat_str}
5. VA = "Valence#Arousal", both floats 1.0-9.0. Valence: 1=very negative, 9=very positive. Arousal: 1=calm, 9=intense.
6. Output JSON ONLY: {{"Quadruplet": [{{"Aspect":"...","Category":"...","Opinion":"...","VA":"7.0#6.0"}}, ...]}}
7. If no {domain}-related opinion, output {{"Quadruplet": []}}."""


def build_user_prompt(text, examples=None):
    shots = ""
    if examples:
        for ex in examples:
            shots += f"Review: {ex['Text']}\nOutput: {json.dumps({'Quadruplet': ex['Quadruplet']}, ensure_ascii=False)}\n\n"
    return f"{shots}Review: {text}\nOutput:"


def call_llm(client, model, prompt, sys_prompt):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:  # endpoint may reject the extra_body kwarg -> retry plain
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=2048,
        )
    content = resp.choices[0].message.content or ""
    content = _THINK.sub("", content)
    m = _FENCE.search(content)
    if m:
        content = m.group(1)
    return content.strip()


def validate_quad(q, text, valid_cats):
    if not isinstance(q, dict):
        return False
    if q.get("Category", "") not in valid_cats:
        return False
    try:
        v, a = str(q.get("VA", "")).split("#")
        v, a = float(v), float(a)
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return False
    except Exception:
        return False
    tl = text.lower()
    for f in ("Aspect", "Opinion"):
        s = q.get(f)
        if s is None:
            s = "NULL"
        if s != "NULL" and str(s).lower() not in tl:
            return False
    return True


def load_resume(path):
    done = {}
    if Path(path).exists():
        for l in open(path, encoding="utf-8"):
            try:
                d = json.loads(l); done[d["ID"]] = d
            except Exception:
                pass
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base_url", default="http://localhost:8888/v1")
    p.add_argument("--api_key", default=os.environ.get("UNSLOTH_API_KEY", ""))
    p.add_argument("--model", required=True)
    p.add_argument("--domain", default="product")
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: set --api_key or UNSLOTH_API_KEY", file=sys.stderr); sys.exit(1)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180)

    valid_cats = build_category_list(args.train_file)
    sys_prompt = build_system_prompt(valid_cats, args.domain)
    train = [json.loads(l) for l in open(args.train_file, encoding="utf-8")]
    examples = [t for t in train if t.get("Quadruplet") and len(t["Text"]) < 160][: args.n_shots]

    pool = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        pool = pool[: args.limit]
    done = load_resume(args.output)
    pending = [s for s in pool if s["ID"] not in done or "_err" in done.get(s["ID"], {})]
    print(f"Pool: {len(pool)}, done: {len(done)}, pending: {len(pending)}, "
          f"valid_cats: {len(valid_cats)}, model: {args.model}", flush=True)

    out_f = open(args.output, "a", encoding="utf-8")
    write_lock = threading.Lock()
    stats = Counter()

    def work(sample):
        text = sample["Text"]
        try:
            raw = call_llm(client, args.model, build_user_prompt(text, examples), sys_prompt)
            quads = json.loads(raw).get("Quadruplet", [])
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
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
            stats["raw"] += n_raw; stats["kept"] += n_kept
            if "_err" in rec: stats["err"] += 1
            n_done += 1
            if n_done % 100 == 0:
                with _print_lock:
                    print(f"  [{n_done}/{len(pending)}] raw={stats['raw']} kept={stats['kept']} "
                          f"err={stats['err']} keep_rate={stats['kept']/max(1,stats['raw']):.2f}", flush=True)
    out_f.close()
    print(f"\nDone. {n_done} processed. raw={stats['raw']} kept={stats['kept']} err={stats['err']}", flush=True)


if __name__ == "__main__":
    main()
