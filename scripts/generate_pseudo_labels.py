"""
Generate pseudo-labelled Japanese hotel reviews using an LLM API.

Usage:
    python scripts/generate_pseudo_labels.py \
        --train_file data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
        --output data/subtask_3/jpn/jpn_hotel_pseudo.jsonl \
        --api_key YOUR_KEY \
        --model gpt-4o \
        --n_samples 200

Output format matches gold JSONL:
    {"ID": "pseudo_1", "Text": "...", "Quadruplet": [...]}

The script:
1. Builds a few-shot prompt from real training examples
2. Asks the LLM to write N new hotel reviews and label them
3. Parses and validates the JSON output
4. Filters low-quality samples (no quads, invalid VA, unknown categories)
"""

import json
import argparse
import os
import random
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Prompt construction
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a Japanese hotel review annotator for sentiment analysis research.
Your task is to generate realistic Japanese hotel reviews and annotate them with quadruplets.

Each quadruplet has:
- Aspect: a text span from the review (the thing being discussed), or "NULL"
- Category: Entity#Attribute format (see valid categories below)
- Opinion: a text span expressing sentiment, or "NULL"
- VA: Valence#Arousal (both 1.0-9.0, where 1=very negative/calm, 9=very positive/excited)

Valid categories:
SERVICE#QUALITY, SERVICE#GENERAL, SERVICE#STYLE_OPTIONS,
FOOD_DRINKS#QUALITY, FOOD_DRINKS#STYLE_OPTIONS, FOOD_DRINKS#PRICES,
FOOD_DRINKS#COMFORT, FOOD_DRINKS#GENERAL,
ROOMS#QUALITY, ROOMS#COMFORT, ROOMS#CLEANLINESS, ROOMS#DESIGN_FEATURES,
ROOMS#GENERAL, ROOMS#PRICES,
HOTEL#GENERAL, HOTEL#DESIGN_FEATURES, HOTEL#CLEANLINESS,
HOTEL#COMFORT, HOTEL#QUALITY, HOTEL#PRICES,
FACILITIES#GENERAL, FACILITIES#QUALITY, FACILITIES#PRICES,
FACILITIES#COMFORT, FACILITIES#DESIGN_FEATURES, FACILITIES#CLEANLINESS,
LOCATION#GENERAL, LOCATION#COMFORT,
EXPERIENCES#GENERAL, EXPERIENCES#QUALITY,
EXPERIENCE_HOTELS#GENERAL,
NULL#GENERAL

Rules:
- Aspect and Opinion must be exact substrings of Text, or "NULL"
- VA values must be floats between 1.0 and 9.0
- Generate diverse reviews covering different hotel aspects
- Use natural, colloquial Japanese
"""

FEW_SHOT_N = 5  # examples to include in prompt


def build_prompt(examples: list, n_generate: int) -> str:
    shots = "\n\n".join([
        f"Text: {e['Text']}\n"
        f"Quadruplets: {json.dumps(e['Quadruplet'], ensure_ascii=False)}"
        for e in examples
    ])

    return f"""Here are {len(examples)} example annotated reviews:

{shots}

Now generate {n_generate} NEW Japanese hotel reviews with their quadruplet annotations.
Return ONLY a JSON array, no other text. Each element must have keys: "Text" and "Quadruplet".
Example format:
[
  {{
    "Text": "...",
    "Quadruplet": [
      {{"Aspect": "...", "Category": "...", "Opinion": "...", "VA": "7.0#6.5"}}
    ]
  }}
]"""


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #

VALID_CATEGORIES = {
    "SERVICE#QUALITY", "SERVICE#GENERAL", "SERVICE#STYLE_OPTIONS",
    "FOOD_DRINKS#QUALITY", "FOOD_DRINKS#STYLE_OPTIONS", "FOOD_DRINKS#PRICES",
    "FOOD_DRINKS#COMFORT", "FOOD_DRINKS#GENERAL",
    "ROOMS#QUALITY", "ROOMS#COMFORT", "ROOMS#CLEANLINESS", "ROOMS#DESIGN_FEATURES",
    "ROOMS#GENERAL", "ROOMS#PRICES",
    "HOTEL#GENERAL", "HOTEL#DESIGN_FEATURES", "HOTEL#CLEANLINESS",
    "HOTEL#COMFORT", "HOTEL#QUALITY", "HOTEL#PRICES",
    "FACILITIES#GENERAL", "FACILITIES#QUALITY", "FACILITIES#PRICES",
    "FACILITIES#COMFORT", "FACILITIES#DESIGN_FEATURES", "FACILITIES#CLEANLINESS",
    "LOCATION#GENERAL", "LOCATION#COMFORT",
    "EXPERIENCES#GENERAL", "EXPERIENCES#QUALITY",
    "EXPERIENCE_HOTELS#GENERAL",
    "NULL#GENERAL",
}


def validate_sample(sample: dict) -> bool:
    text = sample.get("Text", "")
    quads = sample.get("Quadruplet", [])
    if not text or not quads:
        return False

    for q in quads:
        # Category
        if q.get("Category") not in VALID_CATEGORIES:
            return False
        # VA format
        va = q.get("VA", "")
        parts = va.split("#")
        if len(parts) != 2:
            return False
        try:
            v, a = float(parts[0]), float(parts[1])
            if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
                return False
        except ValueError:
            return False
        # Aspect/Opinion must be substring or NULL
        for field in ["Aspect", "Opinion"]:
            val = q.get(field, "")
            if val != "NULL" and val not in text:
                return False

    return True


# --------------------------------------------------------------------------- #
#  Provider detection
# --------------------------------------------------------------------------- #

def detect_provider(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("glm"):
        return "zhipu"
    if model.startswith(("qwen", "deepseek")):
        return "qwen"
    return "openai"


# --------------------------------------------------------------------------- #
#  LLM call — Gemini (google-genai SDK)
# --------------------------------------------------------------------------- #

def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.9,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    return response.text.strip()


# --------------------------------------------------------------------------- #
#  LLM call — Qwen / GLM via Alibaba Bailian (OpenAI-compatible)
# --------------------------------------------------------------------------- #

QWEN_BASE_URL  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

def _call_openai_compat(prompt: str, api_key: str, model: str, base_url: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()

def _call_qwen(prompt: str, api_key: str, model: str) -> str:
    return _call_openai_compat(prompt, api_key, model, QWEN_BASE_URL)


# --------------------------------------------------------------------------- #
#  Unified call with retry
# --------------------------------------------------------------------------- #

def call_llm(prompt: str, api_key: str, model: str) -> list:
    import time
    provider = detect_provider(model)

    for attempt in range(3):
        try:
            if provider == "gemini":
                text = _call_gemini(prompt, api_key, model)
            elif provider == "qwen":
                text = _call_qwen(prompt, api_key, model)
            elif provider == "zhipu":
                text = _call_openai_compat(prompt, api_key, model, ZHIPU_BASE_URL)
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Parse JSON
            try:
                data = json.loads(text)
                # json_object mode may return {"samples": [...]} or directly [...]
                if isinstance(data, list):
                    return data
                for v in data.values():
                    if isinstance(v, list):
                        return v
                return []
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if not match:
                    print("WARNING: No JSON array found in response")
                    return []
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError as e:
                    print(f"WARNING: JSON parse error: {e}")
                    return []

        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"  API error (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return []


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Gold training JSONL for few-shot examples")
    parser.add_argument("--output", required=True, help="Output pseudo-labelled JSONL")
    parser.add_argument("--api_key", default=None,
                        help="API key (or set GEMINI_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Samples per API call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve API key from args or environment
    provider = detect_provider(args.model)
    if args.api_key:
        api_key = args.api_key
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
    elif provider == "zhipu":
        api_key = os.environ.get("ZHIPU_API_KEY")
    elif provider == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        env_map = {"gemini": "GEMINI_API_KEY", "zhipu": "ZHIPU_API_KEY",
                   "qwen": "DASHSCOPE_API_KEY", "openai": "OPENAI_API_KEY"}
        print(f"ERROR: No API key found. Set {env_map.get(provider, 'OPENAI_API_KEY')} or pass --api_key")
        sys.exit(1)
    print(f"Provider: {provider} | Model: {args.model}")

    # Load training examples
    with open(args.train_file) as f:
        train_data = [json.loads(l) for l in f]
    print(f"Loaded {len(train_data)} training examples")

    # Generate in batches
    all_generated = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

    for i in range(n_batches):
        batch_target = min(args.batch_size, args.n_samples - len(all_generated))
        if batch_target <= 0:
            break

        # Sample few-shot examples
        examples = random.sample(train_data, min(FEW_SHOT_N, len(train_data)))
        prompt = build_prompt(examples, batch_target)

        print(f"  Batch {i+1}/{n_batches}: requesting {batch_target} samples...")
        raw = call_llm(prompt, api_key, args.model)

        valid = [s for s in raw if validate_sample(s)]
        print(f"    Got {len(raw)} raw, {len(valid)} valid")
        all_generated.extend(valid)

        if len(all_generated) >= args.n_samples:
            break

    all_generated = all_generated[:args.n_samples]
    print(f"\nTotal valid samples: {len(all_generated)}")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(all_generated):
            out = {
                "ID": f"pseudo_{idx+1}",
                "Text": sample["Text"],
                "Quadruplet": sample["Quadruplet"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
