"""
Extract ABSA quadruplets from jpn_hotel test text using DeepSeek v4-pro.

For each test sample, ask LLM to produce (Aspect, Category, Opinion, VA) quads.
Used for LLM fusion: merge LLM quads with our V15 model predictions.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/llm_extract_quads_jpn.py \
        --test_file data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
        --train_file data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
        --output predictions/llm_quads_jpn_dev.jsonl \
        --workers 8 \
        --n_shots 5
"""
import os
import sys
import json
import re
import time
import random
import argparse
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


# Built up dynamically from train data
VALID_CATEGORIES_JPN = []


def build_category_list(train_file):
    cats = Counter()
    with open(train_file, encoding='utf-8') as f:
        for line in f:
            for q in json.loads(line).get('Quadruplet', []):
                cats[q['Category']] += 1
    # Only keep categories that appear >= 3 times in train
    valid = [c for c, n in cats.most_common() if n >= 3]
    return valid


def build_system_prompt(valid_categories):
    cat_str = ', '.join(valid_categories)
    return f"""あなたは日本語ホテルレビューの ABSA(Aspect-Based Sentiment Analysis) 専門家です。
レビュー文から sentiment quadruplets を抽出してください。

各 quadruplet は次の 4 要素:
- Aspect: 評価対象の本文中の文字列 (例: "部屋", "サービス"), または "NULL" (暗黙的な場合)
- Category: Entity#Attribute 形式. 以下のリストから厳密に選択
- Opinion: 評価/感情を表す本文中の文字列 (例: "親切", "とても良かった")
- VA: "Valence#Arousal" の形 (両方とも 1.0 〜 9.0 の小数)
  - Valence: 1=非常にネガティブ, 5=中立, 9=非常にポジティブ
  - Arousal: 1=穏やか/落ち着いた, 5=普通, 9=興奮/強烈

許可された Category 一覧 (これ以外を使用してはいけない):
{cat_str}

厳守ルール:
1. Aspect と Opinion は元の本文に**そのまま**(部分文字列として)現れる文字列か "NULL"。 言い換え・要約禁止
2. Category は上のリストから完全一致で選択
3. VA は 1.0〜9.0 の範囲
4. 1 つの文に複数の quadruplets が含まれる場合は全て列挙
5. 評価対象/感情が無い文では空リスト
6. 出力は JSON のみ: {{"Quadruplet": [{{"Aspect":"...","Category":"...","Opinion":"...","VA":"7.0#6.0"}}, ...]}}
"""


def build_user_prompt(text, examples=None):
    if examples:
        shots = []
        for e in examples:
            quads = e.get('Quadruplet', [])
            shots.append(f"Text: {e['Text']}\n出力: {json.dumps({'Quadruplet': quads}, ensure_ascii=False)}")
        shots_str = "\n\n".join(shots)
        return f"以下は例:\n\n{shots_str}\n\n以下の新しいレビューを分析してください。\n\nText: {text}\n出力:"
    return f"Text: {text}\n出力:"


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def call_deepseek(prompt, sys_prompt, api_key, model='deepseek-v4-pro'):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=120)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.2,  # low temp for extraction stability
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\{[\s\S]*\}', raw)
                if not m:
                    return [], f"no_json: {raw[:120]}"
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError as e:
                    return [], f"json_err: {e}"
            quads = data.get('Quadruplet') or data.get('quadruplets') or []
            return quads, None
        except Exception as e:
            err = str(e)
            if 'Insufficient Balance' in err or '402' in err:
                return [], f"BALANCE: {err[:100]}"
            wait = 30 * (attempt + 1)
            if attempt < 2:
                time.sleep(wait)
            else:
                return [], f"api_err: {err[:120]}"
    return [], "all_retry_failed"


def validate_quad(q, text, valid_cats):
    if not isinstance(q, dict):
        return False
    asp = q.get('Aspect', '')
    cat = q.get('Category', '')
    opn = q.get('Opinion', '')
    va = q.get('VA', '')
    if cat not in valid_cats:
        return False
    if asp != 'NULL' and asp not in text:
        return False
    if opn != 'NULL' and opn not in text:
        return False
    parts = str(va).split('#')
    if len(parts) != 2:
        return False
    try:
        v, a = float(parts[0]), float(parts[1])
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return False
    except ValueError:
        return False
    return True


def load_resume(path):
    done = set()
    if not Path(path).exists():
        return done
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                done.add(json.loads(line)['ID'])
            except Exception:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--train_file', required=True,
                        help='Used to build valid category list + few-shot examples')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print('ERROR: DEEPSEEK_API_KEY not set')
        sys.exit(1)

    valid_cats = build_category_list(args.train_file)
    print(f'Valid categories: {len(valid_cats)} ({valid_cats[:5]}...)')

    sys_prompt = build_system_prompt(valid_cats)

    # Few-shot pool from train
    with open(args.train_file, encoding='utf-8') as f:
        train_pool = [json.loads(l) for l in f if json.loads(l).get('Quadruplet')]
    print(f'Train pool: {len(train_pool)}')

    # Test samples
    with open(args.test_file, encoding='utf-8') as f:
        test_samples = [json.loads(l) for l in f]
    print(f'Test samples: {len(test_samples)}')

    # Resume
    done_ids = load_resume(args.output)
    pending = [s for s in test_samples if s['ID'] not in done_ids]
    print(f'Pending: {len(pending)} (resumed {len(done_ids)})')

    def process_one(sample):
        examples = random.sample(train_pool, min(args.n_shots, len(train_pool)))
        prompt = build_user_prompt(sample['Text'], examples=examples)
        quads_raw, err = call_deepseek(prompt, sys_prompt, api_key, args.model)
        if err:
            return sample['ID'], [], err
        # Validate
        valid_quads = [q for q in quads_raw if validate_quad(q, sample['Text'], valid_cats)]
        return sample['ID'], valid_quads, None

    n_done = 0
    n_quads_total = 0
    n_invalid = 0
    n_balance_err = 0
    start = time.time()
    write_lock = threading.Lock()

    with open(args.output, 'a', encoding='utf-8') as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, s) for s in pending]
        for fut in as_completed(futs):
            sid, quads, err = fut.result()
            with write_lock:
                if err and 'BALANCE' in err:
                    n_balance_err += 1
                    if n_balance_err > 5:
                        print('FATAL: DeepSeek balance exhausted, stopping.')
                        break
                # Even on error, write empty (to avoid re-trying)
                out = {'ID': sid, 'Quadruplet': quads}
                if err:
                    out['_err'] = err[:100]
                fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                fout.flush()
                n_done += 1
                n_quads_total += len(quads)
                if (n_done) % 20 == 0:
                    elapsed = time.time() - start
                    rate = n_done / elapsed
                    eta = (len(pending) - n_done) / rate if rate > 0 else 0
                    print(f'  [{n_done}/{len(pending)}] quads_avg={n_quads_total/max(1,n_done):.2f} '
                          f'rate={rate:.2f}/s ETA={eta/60:.1f}min')

    print(f'\nDone. Wrote {n_done} samples, total {n_quads_total} quads')
    print(f'Avg quads/sample: {n_quads_total/max(1,n_done):.2f}')


if __name__ == '__main__':
    main()
