import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

JPN_CATEGORIES = [
    'FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT', 'FACILITIES#DESIGN_FEATURES', 'FACILITIES#GENERAL', 'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICE', 'FACILITIES#QUALITY', 'FACILITIES#STYLE_OPTIONS', 
    'FOOD_DRINKS#COMFORT', 'FOOD_DRINKS#DESIGN_FEATURES', 'FOOD_DRINKS#GENERAL', 'FOOD_DRINKS#PRICE', 'FOOD_DRINKS#QUALITY', 'FOOD_DRINKS#STYLE_OPTIONS', 
    'HOTEL#CLEANLINESS', 'HOTEL#COMFORT', 'HOTEL#DESIGN_FEATURES', 'HOTEL#GENERAL', 'HOTEL#PRICE', 'HOTEL#QUALITY', 'HOTEL#STYLE_OPTIONS', 
    'LOCATION#COMFORT', 'LOCATION#GENERAL', 'LOCATION#QUALITY', 'LOCATION#STYLE_OPTIONS', 
    'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN_FEATURES', 'ROOMS#GENERAL', 'ROOMS#PRICE', 'ROOMS#QUALITY', 'ROOMS#STYLE_OPTIONS', 
    'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN_FEATURES', 'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#QUALITY', 'ROOM_AMENITIES#STYLE_OPTIONS', 
    'SERVICE#COMFORT', 'SERVICE#DESIGN_FEATURES', 'SERVICE#GENERAL', 'SERVICE#PRICE', 'SERVICE#QUALITY', 'SERVICE#STYLE_OPTIONS'
]

SYSTEM_PROMPT = """あなたは日本語のホテルレビューからの情報抽出エキスパートです。
ユーザーは、レビューの「原文」、および予測モデルが抽出した「評価対象（Aspect）」、「評価表現（Opinion）」、「カテゴリ（Category）」の候補を提供します。

あなたの任務は 2 つです：
1. 【Category（カテゴリ）の確認・修正】
モデルが予測した Category を確認してください。もしそれが文脈から明らかに間違っている場合のみ、修正してください。
もし妥当であれば、元の Category をそのまま出力してください。
有効なカテゴリは以下の 44 個のみです：
{categories}

2. 【Aspect（評価対象）と Opinion（評価表現）の境界修正】
モデルが抽出した文字列に、余分な助詞（例：「は」「が」「も」）、句読点（例：「。」「、」「・」）、または文脈上不要な接尾辞が含まれている場合、それらを削除して最もコアなフレーズのみにしてください。
例1: Aspect "部屋は" -> "部屋"
例2: Opinion "とても良い。" -> "とても良い"
例3: Aspect "温泉、" -> "温泉"
もし修正の必要がなければ、そのまま出力してください。

【出力形式（厳守）】
以下の JSON 形式で出力してください。Markdown のコードブロックや JSON 以外の余分なテキストは一切含めないでください。
{{
    "Aspect": "修正後の評価対象",
    "Opinion": "修正後の評価表現",
    "Category": "修正後のカテゴリ"
}}
"""

USER_PROMPT = """原文: {text}
モデル予測 Aspect: {aspect}
モデル予測 Opinion: {opinion}
モデル予測 Category: {category}

出力 (JSON):"""

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None

def call_llm(client, text, aspect, opinion, category, model="deepseek-chat"):
    sys_prompt = SYSTEM_PROMPT.format(categories=", ".join(JPN_CATEGORIES))
    usr_prompt = USER_PROMPT.format(text=text, aspect=aspect, opinion=opinion, category=category)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ],
                temperature=0.0,
                max_tokens=64
            )
            raw = response.choices[0].message.content.strip()
            
            parsed = extract_json(raw)
            if parsed and "Category" in parsed and "Aspect" in parsed and "Opinion" in parsed:
                return parsed
            
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return None

def process_sample(sample, client, model):
    text = sample.get("Text", "")
    quads = sample.get("Quadruplet", [])
    
    new_quads = []
    for quad in quads:
        aspect = quad.get("Aspect", "")
        opinion = quad.get("Opinion", "")
        category = quad.get("Category", "")
        
        parsed = call_llm(client, text, aspect, opinion, category, model)
        
        new_quad = dict(quad)
        if parsed:
            if parsed.get("Category") in JPN_CATEGORIES:
                new_quad["Category"] = parsed["Category"]
            new_quad["Aspect"] = parsed.get("Aspect", aspect)
            new_quad["Opinion"] = parsed.get("Opinion", opinion)
            
        new_quads.append(new_quad)
        
    new_sample = dict(sample)
    new_sample["Quadruplet"] = new_quads
    return new_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input predictions JSONL")
    parser.add_argument("--output", required=True, help="Output corrected JSONL")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        return
        
    if OpenAI is None:
        print("Error: 'openai' package not installed. run 'pip install openai'")
        return
        
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    print(f"Loading {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Loaded {len(samples)} samples. Starting V2 correction with {args.workers} workers...")
    
    results = [None] * len(samples)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, sample, client, args.model): i for i, sample in enumerate(samples)}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results[i] = samples[i]
                
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
                
    print(f"Saved V2 corrected predictions to {args.output}")

if __name__ == "__main__":
    main()
