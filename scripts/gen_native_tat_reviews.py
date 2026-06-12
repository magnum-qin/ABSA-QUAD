"""
Generate native Tatar restaurant reviews with DeepSeek (route D, tat variant).

Same recipe as gen_native_ukr_reviews.py but with Tatar prompts. Probe confirmed
DeepSeek generates fluent Tatar (Tatar-specific Cyrillic letters present) and
labels it correctly.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/gen_native_tat_reviews.py --n 4000 --output <pool.jsonl> --workers 12
"""
import os, sys, json, argparse, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

ASPECTS = ["ашлар", "хезмәт күрсәтү", "атмосфера", "бәяләр", "интерьер", "эчемлекләр",
           "десертлар", "официантлар", "музыка", "порцияләр", "меню", "урнашу урыны",
           "чисталык", "тизлек", "милли ашлар", "чәй"]
TONES = ["бик уңай", "уртача уңай", "катнаш", "тискәре", "нейтраль"]

SYS = ("Син татар телендә ресторан/кафе турында табигый, реалистик отзывлар языштыручы ярдәмче. "
       "Чын кешеләр Google Maps'та язган кебек ит, җанлы сөйләм телендә яз. "
       "Һәр отзыв 1-3 җөмлә. ТИК JSON кайтар: {\"reviews\": [\"...\", \"...\"]}")


def call(client, model, focus, tone, k):
    user = (f"Татар телендә ресторан турында {k} төрле кыска отзыв яз. "
            f"Игътибар үзәге: {focus}. Тон: {tone}. "
            "Бер үк сүзләрне кабатлама. JSON кайтар: {\"reviews\":[...]}")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        temperature=1.0, max_tokens=2048,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    try:
        return json.loads(r.choices[0].message.content).get("reviews", [])
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--per_call", type=int, default=8)
    args = p.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY", file=sys.stderr); sys.exit(1)
    client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, timeout=180)

    n_calls = args.n // args.per_call + 1
    jobs = [(random.choice(ASPECTS), random.choice(TONES)) for _ in range(n_calls)]

    pool = {}
    done = 0
    lock = threading.Lock()

    def work(job):
        focus, tone = job
        return call(client, args.model, focus, tone, args.per_call)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            for rev in fut.result():
                rev = " ".join(str(rev).split())
                if 20 < len(rev) < 240:
                    pool[rev[:60]] = rev
            done += 1
            if done % 50 == 0:
                with lock:
                    print(f"  [{done}/{n_calls} calls] unique reviews={len(pool)}", flush=True)
            if len(pool) >= args.n:
                break

    with open(args.output, "w", encoding="utf-8") as f:
        for i, t in enumerate(pool.values()):
            f.write(json.dumps({"ID": f"gen_{i}", "Text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pool)} native Tatar reviews to {args.output}", flush=True)


if __name__ == "__main__":
    main()