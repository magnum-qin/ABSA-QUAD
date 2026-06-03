import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

PROMPT_TEMPLATES = {
    "jpn": {
        "desc": "あなたは翻訳アシスタントです。以下の日本語のレビューを、一度{pivot_lang}に翻訳し、その後再び自然な日本語に翻訳し直してください（バックトランスレーション）。",
        "constraint": "ただし、以下の PRESERVE_PHRASES に含まれる文字列は、最終的なテキストの中に**必ずそのまま（部分文字列として）**含めてください。",
        "output": "出力は JSON 形式のみとしてください。\nフォーマット: {{\"paraphrase\": \"バックトランスレーション後の日本語テキスト\"}}"
    },
    "eng": {
        "desc": "You are a translation assistant. Please translate the following English review into {pivot_lang}, and then translate it back into natural English (back-translation).",
        "constraint": "However, the exact strings listed in PRESERVE_PHRASES MUST be preserved exactly as substrings in your final back-translated output.",
        "output": "Output ONLY in JSON format.\nFormat: {{\"paraphrase\": \"The back-translated English text\"}}"
    },
    "rus": {
        "desc": "You are a translation assistant. Please translate the following Russian review into {pivot_lang}, and then translate it back into natural Russian (back-translation).",
        "constraint": "However, the exact strings listed in PRESERVE_PHRASES MUST be preserved exactly as substrings in your final back-translated Russian output.",
        "output": "Output ONLY in JSON format.\nFormat: {{\"paraphrase\": \"The back-translated Russian text\"}}"
    },
    "ukr": {
        "desc": "You are a translation assistant. Please translate the following Ukrainian review into {pivot_lang}, and then translate it back into natural Ukrainian (back-translation).",
        "constraint": "However, the exact strings listed in PRESERVE_PHRASES MUST be preserved exactly as substrings in your final back-translated Ukrainian output.",
        "output": "Output ONLY in JSON format.\nFormat: {{\"paraphrase\": \"The back-translated Ukrainian text\"}}"
    }
}

def get_prompt(source_lang, pivot_lang, text, phrases):
    tpl = PROMPT_TEMPLATES.get(source_lang, PROMPT_TEMPLATES["eng"])
    sys_msg = f"{tpl['desc'].format(pivot_lang=pivot_lang)}\n{tpl['constraint']}\n\nOriginal Text:\n{text}\n\nPRESERVE_PHRASES:\n{json.dumps(phrases, ensure_ascii=False)}\n\n{tpl['output']}"
    return sys_msg

def call_llm(client, text, phrases, pivot_lang, source_lang, model="deepseek-chat"):
    prompt = get_prompt(source_lang, pivot_lang, text, phrases)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
            
            try:
                if "{" in raw:
                    raw = raw[raw.find("{"):raw.rfind("}")+1]
                data = json.loads(raw)
                return data.get("paraphrase")
            except:
                pass
                
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return None

def process_sample(sample, client, model, pivot_lang, source_lang):
    text = sample.get("Text", "")
    quads = sample.get("Quadruplet", [])
    
    phrases = []
    for q in quads:
        for f in ["Aspect", "Opinion"]:
            if q.get(f) and q.get(f) != "NULL":
                phrases.append(q.get(f))
                
    new_text = call_llm(client, text, phrases, pivot_lang, source_lang, model)
    
    if new_text and all(p in new_text for p in phrases):
        new_sample = dict(sample)
        new_sample["Text"] = new_text
        new_sample["ID"] = f"{sample.get('ID', '')}_bt_{pivot_lang}"
        return new_sample
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_lang", required=True, choices=["jpn", "eng", "rus", "ukr"])
    parser.add_argument("--pivot_langs", nargs="+", default=["Japanese", "Chinese"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set.")
        return
        
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    with open(args.input, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
        
    all_augmented = []
    
    for pivot_lang in args.pivot_langs:
        print(f"Generating paraphrases for {args.source_lang} using {pivot_lang} as pivot...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_sample, sample, client, args.model, pivot_lang, args.source_lang) for sample in samples]
            for future in tqdm(as_completed(futures), total=len(futures)):
                res = future.result()
                if res:
                    all_augmented.append(res)
                    
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in samples + all_augmented:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(samples + all_augmented)} samples (Original {len(samples)}, Augmented {len(all_augmented)}) to {args.output}")

if __name__ == "__main__":
    main()
