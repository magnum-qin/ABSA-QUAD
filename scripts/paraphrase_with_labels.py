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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from difflib import SequenceMatcher


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Per-provider API key env var names
PROVIDER_ENV_KEY = {
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
}


def model_provider(model: str) -> str:
    if model.startswith(("deepseek",)):
        return "deepseek"
    return "dashscope"


def model_base_url(model: str) -> str:
    return DEEPSEEK_BASE_URL if model_provider(model) == "deepseek" else QWEN_BASE_URL


# Fallback model chain — supports DeepSeek (deepseek.com) and DashScope (阿里云百炼).
# When the primary model's quota is exhausted, the script automatically rotates
# to the next entry. Order = preference (primary first).
FALLBACK_MODELS = [
    "deepseek-v4-pro",                # primary (highest semantic fidelity)
    "deepseek-chat",                  # DeepSeek fallback
    "qwen3.6-flash-2026-04-16",
    "tongyi-xiaomi-analysis-pro",
    "qwen3.5-122b-a10b",
    "qwen3.5-flash",
    "qwen3.5-35b-a3b",
    "qwen3.5-plus",
    "qwen3.5-plus-2026-02-15",
    "glm-5",
]

# Quota-related error patterns that should trigger model rotation
QUOTA_ERROR_PATTERNS = [
    "insufficient_quota", "quota", "balance", "billing",
    "exhausted", "InsufficientBalance", "AccountFlowControl",
    "rate_limit", "Throttling", "429",
    "FreeAllocationExceeded", "Arrearage",
]


def is_quota_error(err_msg: str) -> bool:
    msg = str(err_msg).lower()
    return any(p.lower() in msg for p in QUOTA_ERROR_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Language-aware system prompts. Each entry: (language_name, domain, prompt).
# All prompts share the same JSON output contract: {"paraphrases": ["...", ...]}
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_PROMPTS = {
    "jpn": """あなたは日本語のホテルレビューを言い換える専門アシスタントです。元のレビューの意味と感情を保ちつつ、自然な日本語で書き直してください。

厳守すべきルール:
1. 入力の PRESERVE_PHRASES にあるすべての文字列を、出力テキストに必ず**そのまま**（部分文字列として）含めること。省略・変形・スペース挿入・漢字変換の変更は禁止。
2. それ以外の部分は、語順を変えたり、同義語に置き換えたり、敬体⇔常体を切り替えたりして言い換えてよい。
3. 元の文と意味が同じで、感情の極性と強さも保つこと。
4. 自然で流暢な日本語を書くこと。
5. 出力は JSON のみ。フォーマット: {"paraphrases": ["...", "..."]}
6. 元のテキストと完全に同一の出力は避けること。
""",
    "zho": """你是一个中文评论改写专家。请保持原评论的含义和情感不变,用自然流畅的中文重新表达。

必须遵守的规则:
1. PRESERVE_PHRASES 中的所有字符串必须**逐字**(作为子字符串)出现在输出中,禁止省略、变形、增删空格或更换汉字。
2. 其他部分可以调整语序、替换同义词、改变句式等。
3. 必须保持原文的意思一致,情感极性和强度也要保留。
4. 用自然流畅的中文表达。
5. 仅输出 JSON, 格式: {"paraphrases": ["...", "..."]}
6. 避免输出与原文完全相同的句子。
""",
    "rus": """Вы — эксперт по перефразированию русских отзывов о ресторанах. Перефразируйте отзыв, сохраняя смысл и эмоциональную окраску.

Строгие правила:
1. Все строки из PRESERVE_PHRASES должны **дословно** (как подстроки) присутствовать в выводе. Запрещены пропуски, изменения, добавление пробелов или замена символов.
2. Остальные части можно перестраивать: менять порядок слов, заменять синонимами, менять структуру предложения.
3. Сохраняйте исходный смысл, полярность и интенсивность эмоций.
4. Пишите естественным, беглым русским языком.
5. Возвращайте ТОЛЬКО JSON в формате: {"paraphrases": ["...", "..."]}
6. Избегайте вывода, идентичного исходному тексту.
""",
    "ukr": """Ви — експерт із перефразування українських відгуків про ресторани. Перефразуйте відгук, зберігаючи зміст і емоційне забарвлення.

Суворі правила:
1. Усі рядки з PRESERVE_PHRASES мають **дослівно** (як підрядки) бути присутні у виводі. Заборонено пропуски, зміни, додавання пробілів або заміну символів.
2. Інші частини можна перебудовувати: міняти порядок слів, заміняти синонімами, змінювати структуру речення.
3. Зберігайте вихідний зміст, полярність та інтенсивність емоцій.
4. Пишіть природною, плавною українською мовою.
5. Повертайте ТІЛЬКИ JSON у форматі: {"paraphrases": ["...", "..."]}
6. Уникайте виводу, ідентичного оригінальному тексту.
""",
    "tat": """You are a Tatar restaurant-review paraphraser. Rewrite the review in natural, fluent Tatar while preserving meaning and emotional polarity.

Strict rules:
1. Every string in PRESERVE_PHRASES MUST appear verbatim (as a substring) in the output. No omissions, modifications, added spaces, or character substitutions.
2. Other parts may be reordered, paraphrased with synonyms, or restructured.
3. Preserve the original meaning, sentiment polarity and intensity.
4. Write natural, fluent Tatar (NOT Russian).
5. Return JSON ONLY in format: {"paraphrases": ["...", "..."]}
6. Avoid outputs identical to the original text.
""",
    "eng": """You are an expert paraphraser of English reviews. Rewrite the review while preserving its meaning and emotional polarity.

Strict rules:
1. Every string in PRESERVE_PHRASES MUST appear verbatim (as a substring) in the output. No omissions, modifications, added spaces, or character substitutions.
2. Other parts may be reordered, paraphrased with synonyms, or restructured.
3. Preserve the original meaning, sentiment polarity and intensity.
4. Write natural, fluent English.
5. Return JSON ONLY in format: {"paraphrases": ["...", "..."]}
6. Avoid outputs identical to the original text.
""",
}

LANGUAGE_USER_TEMPLATE = {
    "jpn": ("原文:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "上記の原文を言い換えた新しい日本語テキストを {n} 件、"
            "PRESERVE_PHRASES のすべての文字列を部分文字列として必ず含めたうえで生成してください。"
            "JSON のみを返してください: {{\"paraphrases\": [...]}}"),
    "zho": ("原文:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "请将上述原文改写为 {n} 个新的中文句子,"
            "PRESERVE_PHRASES 中的所有字符串必须作为子字符串包含在内。"
            "仅返回 JSON: {{\"paraphrases\": [...]}}"),
    "rus": ("Оригинал:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "Перефразируйте оригинал в {n} новых русских вариантах, "
            "обязательно включая все строки из PRESERVE_PHRASES как подстроки. "
            "Возвращайте ТОЛЬКО JSON: {{\"paraphrases\": [...]}}"),
    "ukr": ("Оригінал:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "Перефразуйте оригінал у {n} нових українських варіантах, "
            "обов'язково включаючи всі рядки з PRESERVE_PHRASES як підрядки. "
            "Повертайте ТІЛЬКИ JSON: {{\"paraphrases\": [...]}}"),
    "tat": ("Original:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "Paraphrase the original into {n} new Tatar sentences, "
            "MUST include every string from PRESERVE_PHRASES as a substring. "
            "Return JSON ONLY: {{\"paraphrases\": [...]}}"),
    "eng": ("Original:\n{text}\n\nPRESERVE_PHRASES: {phrases}\n\n"
            "Paraphrase the original into {n} new English sentences, "
            "MUST include every string from PRESERVE_PHRASES as a substring. "
            "Return JSON ONLY: {{\"paraphrases\": [...]}}"),
}


def get_system_prompt(language: str) -> str:
    if language not in LANGUAGE_PROMPTS:
        raise ValueError(f"Unsupported language: {language}. "
                         f"Available: {sorted(LANGUAGE_PROMPTS.keys())}")
    return LANGUAGE_PROMPTS[language]


def build_prompt(text: str, preserve_phrases: list, n_variants: int,
                 language: str = "jpn") -> str:
    phrases_json = json.dumps(preserve_phrases, ensure_ascii=False)
    template = LANGUAGE_USER_TEMPLATE[language]
    return template.format(text=text, phrases=phrases_json, n=n_variants)


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


# Module-level rotation state: index into FALLBACK_MODELS.
# Once a model is permanently exhausted, _rotation_index is bumped past it
# so subsequent calls skip it directly.
_rotation_index = 0


def _call_one_model(prompt: str, api_keys: dict, model: str,
                    system_prompt: str) -> tuple:
    """Single attempt at one model. api_keys is a {provider: key} dict.
    Returns (paraphrases, err_msg or None)."""
    from openai import OpenAI
    provider = model_provider(model)
    key = api_keys.get(provider)
    if not key:
        return [], f"NO_KEY_FOR_PROVIDER:{provider}"
    client = OpenAI(api_key=key, base_url=model_base_url(model), timeout=120)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1900,  # tongyi-xiaomi-analysis-pro caps at 2000
            # NOTE: Cyrillic/CJK uses ~2-3x more tokens than Latin per char.
            # 1900 is a tight ceiling for languages with multi-byte scripts.
        )
        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', raw)
            if not m:
                print(f"    WARNING: no JSON object in response: {raw[:200]!r}")
                return [], None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError as e:
                print(f"    WARNING: JSON parse failed: {e}")
                return [], None

        items = data.get("paraphrases", [])
        if not isinstance(items, list):
            return [], None
        return [s for s in items if isinstance(s, str)], None

    except Exception as e:
        return [], str(e)


def call_qwen(prompt: str, api_keys: dict, system_prompt: str,
              _ignored: str = None) -> list:
    """
    Call LLM with automatic model rotation on quota exhaustion.
    Supports cross-provider fallback (DeepSeek → DashScope → ...).

    api_keys: {"deepseek": "sk-...", "dashscope": "sk-..."}

    Strategy:
      1. Try the current rotation model with up to 3 retries (transient errors).
      2. If error matches a quota/billing pattern OR provider key missing,
         mark this model exhausted and advance _rotation_index permanently.
      3. Try the next model in FALLBACK_MODELS.
      4. Once all models are exhausted, return [] and the caller skips the sample.
    """
    global _rotation_index

    while _rotation_index < len(FALLBACK_MODELS):
        model = FALLBACK_MODELS[_rotation_index]

        for attempt in range(3):
            paraphrases, err = _call_one_model(prompt, api_keys, model, system_prompt)
            if err is None:
                return paraphrases  # success (possibly empty)

            if err.startswith("NO_KEY_FOR_PROVIDER:") or is_quota_error(err):
                print(f"    ✗ Model '{model}' unavailable ({err[:120]}). Rotating to next.")
                _rotation_index += 1
                break  # exit retry loop, jump to next model

            wait = 30 * (attempt + 1)
            print(f"    API error on '{model}' (attempt {attempt+1}/3): {err[:120]}")
            if attempt < 2:
                time.sleep(wait)
        else:
            # All 3 retries failed without quota signal — give up on this sample
            return []

    print("    ✗ ALL fallback models exhausted. No more API calls possible.")
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
    parser.add_argument("--api_key", default=None,
                        help="DashScope API key (alternatively set DASHSCOPE_API_KEY env)")
    parser.add_argument("--deepseek_api_key", default=None,
                        help="DeepSeek API key (alternatively set DEEPSEEK_API_KEY env)")
    parser.add_argument("--model", default="qwen3.6-flash-2026-04-16",
                        help="Primary model name (e.g. deepseek-v4-pro, qwen3.6-flash...). "
                             "On quota exhaustion the script rotates through FALLBACK_MODELS.")
    parser.add_argument("--n_variants", type=int, default=2,
                        help="Paraphrases requested per sample")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of source samples (for testing)")
    parser.add_argument("--min_diff_ratio", type=float, default=0.15,
                        help="Minimum textual difference from original (0-1)")
    parser.add_argument("--skip_null_only", action="store_true",
                        help="Skip samples whose only quads are NULL/NULL")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API workers (1 = sequential)")
    parser.add_argument("--language", default="jpn",
                        choices=sorted(LANGUAGE_PROMPTS.keys()),
                        help="Source language code for choosing the system prompt")
    args = parser.parse_args()

    # If user-specified model isn't already in the chain, prepend it as primary
    global _rotation_index
    if args.model and args.model not in FALLBACK_MODELS:
        FALLBACK_MODELS.insert(0, args.model)
    elif args.model in FALLBACK_MODELS:
        # Start rotation at the user-specified model (skip earlier ones)
        _rotation_index = FALLBACK_MODELS.index(args.model)
    print(f"Model rotation chain (starting at index {_rotation_index}):")
    for i, m in enumerate(FALLBACK_MODELS):
        marker = "→" if i == _rotation_index else " "
        print(f"  {marker} [{i}] {m}")

    # Build per-provider key map; missing keys cause those providers' models
    # to be skipped in rotation (not a hard error).
    api_keys = {
        "dashscope": args.api_key or os.environ.get("DASHSCOPE_API_KEY"),
        "deepseek":  args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY"),
    }
    if not any(api_keys.values()):
        print("ERROR: No API key found. Set DASHSCOPE_API_KEY and/or DEEPSEEK_API_KEY.")
        sys.exit(1)
    print("API keys available for: " + ", ".join(p for p, k in api_keys.items() if k))

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

    # Filter to pending samples
    pending = []
    for i, sample in enumerate(train_data):
        src_id = sample.get("ID") or f"src_{i}"
        if src_id in done_ids:
            continue
        if not sample.get("Text", "").strip():
            continue
        preserve = collect_preserve_phrases(sample)
        if args.skip_null_only and not preserve:
            continue
        pending.append((src_id, sample, preserve))
    print(f"Pending: {len(pending)} samples (workers={args.workers})")

    system_prompt = get_system_prompt(args.language)
    print(f"Language: {args.language} (system prompt loaded, {len(system_prompt)} chars)")

    # Per-sample worker function
    def process_sample(item):
        src_id, sample, preserve = item
        text = sample["Text"].strip()
        prompt = build_prompt(text, preserve, args.n_variants, language=args.language)
        paraphrases = call_qwen(prompt, api_keys, system_prompt, args.model)
        accepted = []
        n_rej = 0
        for p in paraphrases:
            ok, _ = validate_paraphrase(p, preserve, text,
                                        min_diff_ratio=args.min_diff_ratio)
            if ok:
                accepted.append(p)
            else:
                n_rej += 1
        return src_id, sample, accepted, n_rej

    n_written = 0
    n_kept = 0
    n_rejected = 0
    start = time.time()
    write_lock = threading.Lock()
    completed = 0
    total = len(pending)

    with open(output_path, "a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_sample, item) for item in pending]
        for fut in as_completed(futures):
            try:
                src_id, sample, accepted, n_rej = fut.result()
            except Exception as e:
                print(f"    worker error: {e}")
                completed += 1
                continue

            with write_lock:
                for j, new_text in enumerate(accepted):
                    out = {
                        "ID": f"para_{src_id}_{j+1}",
                        "SourceID": src_id,
                        "Text": new_text,
                        "Quadruplet": sample["Quadruplet"],
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_written += 1
                    n_kept += 1
                n_rejected += n_rej
                fout.flush()

            completed += 1
            if completed % 20 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] kept={n_kept} rejected={n_rejected} "
                      f"rate={rate:.2f}/s ETA={eta/60:.1f}min")

    print(f"\nDone. Wrote {n_written} paraphrases to {output_path}")
    print(f"Accepted: {n_kept}  Rejected: {n_rejected}")
    if n_kept + n_rejected > 0:
        print(f"Acceptance rate: {n_kept/(n_kept+n_rejected)*100:.1f}%")


if __name__ == "__main__":
    main()
