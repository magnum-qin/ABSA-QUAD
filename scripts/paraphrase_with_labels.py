"""
Paraphrase existing Japanese hotel reviews while preserving aspect/opinion spans.

Unlike generate_pseudo_labels.py (which synthesises new text AND new labels),
this script REUSES the original gold labels and only rewrites the text.
The paraphrased text is only accepted if every non-NULL aspect/opinion phrase
from the gold annotation appears verbatim as a substring.

Usage:
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/paraphrase_with_labels.py \
        --train_file data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
        --output data/subtask_3/jpn/jpn_hotel_paraphrase.jsonl \
        --model qwen3.6-flash-2026-04-16 \
        --n_variants 2
"""

import json
import argparse
import os
import re
import sys
import time
from pathlib import Path
from difflib import SequenceMatcher


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


SYSTEM_PROMPT = """あなたは日本語のホテルレビューを言い換える専門アシスタントです。元のレビューの意味と感情を保ちつつ、自然な日本語で書き直してください。

厳守すべきルール:
1. 入力の PRESERVE_PHRASES にあるすべての文字列を、出力テキストに必ず**そのまま**（部分文字列として）含めること。省略・変形・スペース挿入・漢字変換の変更は禁止。
2. それ以外の部分は、語順を変えたり、同義語に置き換えたり、敬体⇔常体を切り替えたりして言い換えてよい。
3. 元の文と意味が同じで、感情の極性と強さも保つこと。
4. 自然で流暢な日本語を書くこと。
5. 出力は JSON のみ。フォーマット: {"paraphrases": ["...", "..."]}
6. 元のテキストと完全に同一の出力は避けること。
"""


def build_prompt(text: str, preserve_phrases: list, n_variants: int) -> str:
    phrases_json = json.dumps(preserve_phrases, ensure_ascii=False)
    return (
        f"原文:\n{text}\n\n"
        f"PRESERVE_PHRASES: {phrases_json}\n\n"
        f"上記の原文を言い換えた新しい日本語テキストを {n_variants} 件、"
        f"PRESERVE_PHRASES のすべての文字列を部分文字列として必ず含めたうえで生成してください。"
        f"JSON のみを返してください: {{\"paraphrases\": [...]}}"
    )


def collect_preserve_phrases(sample: dict) -> list:
    """Return unique non-NULL aspect/opinion spans."""
    phrases = set()
    for q in sample.get("Quadruplet", []):
        for field in ("Aspect", "Opinion"):
            v = q.get(field, "")
            if v and v != "NULL":
                phrases.add(v)
    # sort longest first so that substring checks catch the most constraining spans
    return sorted(phrases, key=len, reverse=True)


def validate_paraphrase(new_text: str, preserve_phrases: list,
                         original_text: str,
                         min_diff_ratio: float = 0.15) -> tuple:
    """
    Returns (ok: bool, reason: str).
    - new_text must contain every preserve_phrase verbatim
    - new_text must differ from original by at least min_diff_ratio
    """
    if not new_text or not isinstance(new_text, str):
        return False, "empty text"
    for p in preserve_phrases:
        if p not in new_text:
            return False, f"missing phrase: {p!r}"
    # reject near-identical outputs
    ratio = SequenceMatcher(None, original_text, new_text).ratio()
    if ratio > (1.0 - min_diff_ratio):
        return False, f"too similar (ratio={ratio:.3f})"
    return True, "ok"


def call_qwen(prompt: str, api_key: str, model: str) -> list:
    """Call Qwen via DashScope OpenAI-compatible endpoint. Returns list[str]."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL, timeout=120)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON (robust to extra text around it)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\{[\s\S]*\}', raw)
                if not m:
                    print(f"    WARNING: no JSON object in response: {raw[:200]!r}")
                    return []
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError as e:
                    print(f"    WARNING: JSON parse failed: {e}")
                    return []

            items = data.get("paraphrases", [])
            if not isinstance(items, list):
                return []
            return [s for s in items if isinstance(s, str)]

        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"    API error (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(wait)
    return []


def load_resume(output_path: Path) -> set:
    """Return set of original IDs already processed (based on paraphrase output)."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                src = obj.get("SourceID")
                if src:
                    done.add(src)
            except json.JSONDecodeError:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="qwen3.6-flash-2026-04-16")
    parser.add_argument("--n_variants", type=int, default=2,
                        help="Paraphrases requested per sample")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of source samples (for testing)")
    parser.add_argument("--min_diff_ratio", type=float, default=0.15,
                        help="Minimum textual difference from original (0-1)")
    parser.add_argument("--skip_null_only", action="store_true",
                        help="Skip samples whose only quads are NULL/NULL")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set. Use --api_key or export DASHSCOPE_API_KEY=...")
        sys.exit(1)

    # Load source training data
    with open(args.train_file, encoding="utf-8") as f:
        train_data = [json.loads(l) for l in f]
    if args.limit:
        train_data = train_data[:args.limit]
    print(f"Loaded {len(train_data)} source samples from {args.train_file}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_resume(output_path)
    if done_ids:
        print(f"Resume: {len(done_ids)} samples already done, skipping them")

    n_written = 0
    n_kept = 0
    n_rejected = 0
    start = time.time()

    with open(output_path, "a", encoding="utf-8") as fout:
        for i, sample in enumerate(train_data):
            src_id = sample.get("ID") or f"src_{i}"
            if src_id in done_ids:
                continue

            text = sample.get("Text", "").strip()
            if not text:
                continue

            preserve = collect_preserve_phrases(sample)
            if args.skip_null_only and not preserve:
                continue  # nothing to preserve → skip

            prompt = build_prompt(text, preserve, args.n_variants)
            paraphrases = call_qwen(prompt, api_key, args.model)

            accepted = []
            for p in paraphrases:
                ok, reason = validate_paraphrase(p, preserve, text,
                                                 min_diff_ratio=args.min_diff_ratio)
                if ok:
                    accepted.append(p)
                else:
                    n_rejected += 1

            for j, new_text in enumerate(accepted):
                out = {
                    "ID": f"para_{src_id}_{j+1}",
                    "SourceID": src_id,
                    "Text": new_text,
                    "Quadruplet": sample["Quadruplet"],  # reuse gold labels
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_written += 1
                n_kept += 1
            fout.flush()

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(train_data) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(train_data)}] kept={n_kept} rejected={n_rejected} "
                      f"rate={rate:.2f}/s ETA={eta/60:.1f}min")

    print(f"\nDone. Wrote {n_written} paraphrases to {output_path}")
    print(f"Accepted: {n_kept}  Rejected: {n_rejected}")
    if n_kept + n_rejected > 0:
        print(f"Acceptance rate: {n_kept/(n_kept+n_rejected)*100:.1f}%")


if __name__ == "__main__":
    main()
