"""
Back-Translation Data Augmentation for DimASQP

Translates training sentences through pivot languages (en→pivot→en)
to generate paraphrased training samples while preserving quad annotations.

Usage:
    python scripts/augment_data.py \
        --input data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
        --output data/subtask_3/eng/eng_restaurant_train_augmented.jsonl \
        --langs de fr zh
"""

import json
import random
import argparse
import copy
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


# --- Constants ---
VA_MIN, VA_MAX = 1.0, 9.0

# MarianMT model names: source_lang -> {pivot_lang: (fwd_model, bwd_model)}
# fwd = source->pivot, bwd = pivot->source
MARIAN_MODELS_BY_SOURCE = {
    "en": {
        "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
        "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
        "zh": ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"),
        "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
    },
    "ja": {
        "en": ("Helsinki-NLP/opus-mt-jap-en", "Helsinki-NLP/opus-mt-en-jap"),
    },
    "zh": {
        "en": ("Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-en-zh"),
    },
    "ru": {
        "en": ("Helsinki-NLP/opus-mt-ru-en", "Helsinki-NLP/opus-mt-en-ru"),
        "uk": ("Helsinki-NLP/opus-mt-ru-uk", "Helsinki-NLP/opus-mt-uk-ru"),
    },
    "uk": {
        "en": ("Helsinki-NLP/opus-mt-uk-en", "Helsinki-NLP/opus-mt-en-uk"),
        "ru": ("Helsinki-NLP/opus-mt-uk-ru", "Helsinki-NLP/opus-mt-ru-uk"),
    },
    "tat": {
        # Turkic language group model with >>tat<< tag
        "en": ("Helsinki-NLP/opus-mt-trk-en", "Helsinki-NLP/opus-mt-en-trk"),
    },
}

# Legacy alias for backward compatibility
MARIAN_MODELS = MARIAN_MODELS_BY_SOURCE["en"]


# --- Validation ---

def build_valid_categories(samples: List[dict]) -> set:
    """Build valid category set from training data."""
    cats = set()
    for s in samples:
        for q in s.get("Quadruplet", []):
            cats.add(q["Category"])
    return cats


def validate_quad(text: str, quad: dict, valid_categories: set = None) -> bool:
    """Validate a single quadruplet against the text."""
    # Check category
    if valid_categories:
        cat = quad.get("Category", "")
        if cat not in valid_categories:
            return False

    # Check VA
    try:
        va_parts = quad["VA"].split("#")
        v, a = float(va_parts[0]), float(va_parts[1])
        if not (VA_MIN <= v <= VA_MAX and VA_MIN <= a <= VA_MAX):
            return False
    except (ValueError, IndexError, KeyError):
        return False

    # Check spans exist in text
    for field in ("Aspect", "Opinion"):
        span = quad.get(field, "")
        if span != "NULL" and span.lower() not in text.lower():
            return False

    return True


def validate_sample(sample: dict, valid_categories: set = None) -> Tuple[bool, dict]:
    """Validate sample, remove invalid quads. Returns (is_valid, cleaned)."""
    text = sample.get("Text", "")
    quads = sample.get("Quadruplet", [])
    valid_quads = [q for q in quads if validate_quad(text, q, valid_categories)]
    if not valid_quads:
        return False, sample
    cleaned = copy.deepcopy(sample)
    cleaned["Quadruplet"] = valid_quads
    return True, cleaned


def word_jaccard(text1: str, text2: str) -> float:
    """Word-level Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


# --- Back-Translation ---

class BackTranslationAugmenter:
    """Back-translate via pivot languages to generate paraphrases."""

    def __init__(self, pivot_langs: List[str], device: str = "cuda",
                 batch_size: int = 32, fuzzy_threshold: float = 0.8,
                 source_lang: str = "en", lang_tags: dict = None):
        self.pivot_langs = pivot_langs
        self.device = device
        self.batch_size = batch_size
        self.fuzzy_threshold = fuzzy_threshold
        self.source_lang = source_lang
        self.lang_tags = lang_tags or {}  # e.g. {"fwd": ">>tat<<", "bwd": None}
        self.translators = {}

        models = MARIAN_MODELS_BY_SOURCE.get(source_lang, {})
        for lang in pivot_langs:
            if lang not in models:
                print(f"WARNING: No MarianMT model for '{source_lang}'->{lang}, skipping")
                continue
            fwd_name, bwd_name = models[lang]
            print(f"Loading MarianMT for {lang}...")
            print(f"  Forward: {fwd_name}")
            print(f"  Backward: {bwd_name}")
            fwd_tok = MarianTokenizer.from_pretrained(fwd_name)
            fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device).eval()
            bwd_tok = MarianTokenizer.from_pretrained(bwd_name)
            bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device).eval()
            self.translators[lang] = (fwd_model, fwd_tok, bwd_model, bwd_tok)
            print(f"  Loaded {lang} models")

    def _translate_batch(self, texts: List[str], model, tokenizer,
                         lang_tag: str = None) -> List[str]:
        """Translate a batch of texts. Optionally prepend a language tag."""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if lang_tag:
                batch = [f"{lang_tag} {t}" for t in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=512)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            results.extend(decoded)
        return results

    def back_translate_batch(self, texts: List[str], pivot_lang: str) -> List[str]:
        """source -> pivot -> source for a batch of texts."""
        fwd_model, fwd_tok, bwd_model, bwd_tok = self.translators[pivot_lang]
        fwd_tag = self.lang_tags.get("fwd")
        bwd_tag = self.lang_tags.get("bwd")
        # Forward: source -> pivot
        pivot_texts = self._translate_batch(texts, fwd_model, fwd_tok, lang_tag=fwd_tag)
        # Backward: pivot -> source
        back_texts = self._translate_batch(pivot_texts, bwd_model, bwd_tok, lang_tag=bwd_tag)
        return back_texts

    def _find_span_in_text(self, span: str, text: str) -> Optional[str]:
        """Find span in text, return actual-case version or None."""
        if span == "NULL":
            return "NULL"

        # 1. Exact case-insensitive match
        text_lower = text.lower()
        span_lower = span.lower()
        idx = text_lower.find(span_lower)
        if idx >= 0:
            return text[idx:idx + len(span)]

        # 2. Fuzzy match via n-gram comparison
        words = text.split()
        span_words = span.split()
        span_len = len(span_words)
        best_ratio = 0.0
        best_match = None

        for n in range(max(1, span_len - 1), span_len + 2):
            for start in range(len(words) - n + 1):
                candidate = " ".join(words[start:start + n])
                ratio = SequenceMatcher(None, span_lower, candidate.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate

        if best_ratio >= self.fuzzy_threshold:
            return best_match

        return None

    def relocate_spans(self, translated_text: str,
                       quads: List[dict]) -> List[dict]:
        """Relocate aspect/opinion spans in translated text."""
        relocated = []
        for quad in quads:
            new_quad = copy.deepcopy(quad)
            # Relocate aspect
            new_aspect = self._find_span_in_text(quad["Aspect"], translated_text)
            if new_aspect is None:
                continue
            new_quad["Aspect"] = new_aspect

            # Relocate opinion
            new_opinion = self._find_span_in_text(quad["Opinion"], translated_text)
            if new_opinion is None:
                continue
            new_quad["Opinion"] = new_opinion

            relocated.append(new_quad)
        return relocated

    def augment_dataset(self, samples: List[dict],
                        original_texts: set,
                        valid_categories: set = None) -> List[dict]:
        """Generate augmented samples for all pivot languages."""
        augmented = []

        for lang in self.pivot_langs:
            if lang not in self.translators:
                continue

            print(f"\n--- Back-translating via {lang.upper()} ---")
            texts = [s["Text"] for s in samples]

            # Batch back-translate
            bt_texts = self.back_translate_batch(texts, lang)

            n_kept = 0
            n_dropped = 0
            n_identical = 0

            for sample, bt_text in zip(samples, bt_texts):
                orig_text = sample["Text"]

                # Skip identical translations
                if bt_text.strip().lower() == orig_text.strip().lower():
                    n_identical += 1
                    continue

                # Skip extreme length changes
                ratio = len(bt_text) / max(len(orig_text), 1)
                if ratio < 0.5 or ratio > 2.0:
                    n_dropped += 1
                    continue

                # Skip near-duplicates with original data
                if any(word_jaccard(bt_text, ot) > 0.9 for ot in
                       random.sample(list(original_texts),
                                     min(50, len(original_texts)))):
                    n_dropped += 1
                    continue

                # Relocate spans
                relocated_quads = self.relocate_spans(bt_text, sample["Quadruplet"])
                if not relocated_quads:
                    n_dropped += 1
                    continue

                aug_sample = {
                    "ID": f"{sample['ID']}_bt_{lang}",
                    "Text": bt_text,
                    "Quadruplet": relocated_quads
                }

                # Final validation
                is_valid, cleaned = validate_sample(aug_sample, valid_categories)
                if is_valid:
                    augmented.append(cleaned)
                    n_kept += 1
                else:
                    n_dropped += 1

            total = len(samples)
            print(f"  Results: {n_kept}/{total} kept "
                  f"({n_kept/total*100:.1f}%), "
                  f"{n_dropped} dropped, {n_identical} identical")

            # Free GPU memory between languages
            torch.cuda.empty_cache()

        return augmented


# --- I/O ---

def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[dict], path: str):
    """Save as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


def print_category_distribution(samples: List[dict], label: str = ""):
    """Print category distribution."""
    from collections import Counter
    cats = Counter()
    for s in samples:
        for q in s.get("Quadruplet", []):
            cats[q["Category"]] += 1
    print(f"\n{label} Category distribution ({sum(cats.values())} total quads):")
    for cat, count in cats.most_common():
        print(f"  {cat:<30} {count:>5}")


def main():
    parser = argparse.ArgumentParser(description="Back-Translation Data Augmentation")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to original training JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: input with _augmented suffix)")
    parser.add_argument("--langs", type=str, nargs="+", default=["de", "fr", "zh"],
                        help="Pivot languages for back-translation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fuzzy_threshold", type=float, default=0.8,
                        help="Fuzzy match threshold for span relocation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_merge", action="store_true",
                        help="Output augmented only, don't merge with original")
    parser.add_argument("--source_lang", type=str, default="en",
                        help="Source language code (default: en)")
    parser.add_argument("--bwd_lang_tag", type=str, default=None,
                        help="Language tag for backward translation (e.g. >>tat<< for Tatar)")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output path
    if args.output is None:
        inp = Path(args.input)
        args.output = str(inp.parent / f"{inp.stem}_augmented{inp.suffix}")

    # Load data
    print(f"Loading data from: {args.input}")
    samples = load_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")
    original_texts = {s["Text"] for s in samples}
    valid_categories = build_valid_categories(samples)
    print(f"Valid categories: {len(valid_categories)}")
    print_category_distribution(samples, "Original")

    # Build language tags
    lang_tags = {}
    if args.bwd_lang_tag:
        lang_tags["bwd"] = args.bwd_lang_tag

    # Run back-translation
    augmenter = BackTranslationAugmenter(
        pivot_langs=args.langs,
        device=device,
        batch_size=args.batch_size,
        fuzzy_threshold=args.fuzzy_threshold,
        source_lang=args.source_lang,
        lang_tags=lang_tags,
    )
    augmented = augmenter.augment_dataset(samples, original_texts,
                                          valid_categories=valid_categories)
    print(f"\nTotal augmented samples: {len(augmented)}")
    print_category_distribution(augmented, "Augmented")

    # Save augmented-only file
    aug_only_path = str(Path(args.output).parent /
                        f"{Path(args.output).stem}_bt_only.jsonl")
    save_jsonl(augmented, aug_only_path)

    # Merge
    if args.no_merge:
        save_jsonl(augmented, args.output)
    else:
        merged = samples + augmented
        random.shuffle(merged)
        save_jsonl(merged, args.output)
        print_category_distribution(merged, "Merged")

    # Summary
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Original samples:  {len(samples)}")
    print(f"Augmented samples: {len(augmented)}")
    if not args.no_merge:
        print(f"Merged total:      {len(samples) + len(augmented)}")
    print(f"Expansion ratio:   {(len(samples) + len(augmented)) / len(samples):.2f}x")
    print(f"\nOutput: {args.output}")
    print(f"\nNext: train with augmented data:")
    print(f"  python scripts/train.py \\")
    print(f"    --config configs/config_rethink_v3.yaml \\")
    print(f"    --train_file {args.output}")


if __name__ == "__main__":
    main()
