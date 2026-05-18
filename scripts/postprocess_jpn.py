"""
Post-process Japanese predictions using MeCab morphological analysis.

Fixes span boundary errors by aligning to morpheme boundaries:
- Trailing particles/copula: 広々と → 広々, 良かったです → 良かった
- Trailing particles: も多くて → 多く, 快適に → 快適
- Non-content spans filtered out

Usage:
    python scripts/postprocess_jpn.py \
        --input predictions/jpn_hotel.jsonl \
        --gold data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
        --output predictions/jpn_hotel_mecab.jsonl
"""

import json
import argparse
from pathlib import Path

import MeCab


# Content POS tags (not particles/auxiliary)
CONTENT_POS = {'名詞', '動詞', '形容詞', '副詞', '連体詞', '形状詞'}
FILLER_POS = {'助詞', '助動詞', '補助記号', '空白', '記号'}

# v2: STRICT whitelist of true intensifiers (not all adverbs).
# These are degree/manner modifiers that typically belong inside opinion span.
INTENSIFIER_WHITELIST = {
    'とても', '非常に', '大変', 'かなり', '本当に', 'すごく', 'すこく',
    'めっちゃ', '結構', 'すごい', '物凄く', 'ものすごく',
    'ちょっと', '少し', 'やや', 'まあまあ', 'まあ',
    '全然', '全く', 'まったく', 'まじで',
    '最高に', '最も',
}


def get_morphemes(tagger, text):
    """Get list of morpheme dicts with surface, pos, start, end."""
    morphemes = []
    node = tagger.parseToNode(text)
    pos = 0
    while node:
        if node.surface:
            start = text.find(node.surface, pos)
            if start == -1:
                start = pos
            end = start + len(node.surface)
            feature = node.feature.split(',')
            morphemes.append({
                'surface': node.surface,
                'pos': feature[0],
                'start': start,
                'end': end,
            })
            pos = end
        node = node.next
    return morphemes


def align_span(text, span, morphemes):
    """Align a span to morpheme boundaries.

    1. Find span position in text
    2. Find all morphemes overlapping with the span
    3. Trim leading/trailing filler morphemes (particles, aux verbs)
    4. Return the aligned span
    """
    if span in ('NULL', '', None):
        return span

    span_start = text.find(span)
    if span_start == -1:
        return span
    span_end = span_start + len(span)

    # Find overlapping morphemes
    overlapping = []
    for m in morphemes:
        if m['end'] > span_start and m['start'] < span_end:
            overlapping.append(m)

    if not overlapping:
        return span

    # Trim leading filler morphemes
    while len(overlapping) > 1 and overlapping[0]['pos'] in FILLER_POS:
        overlapping = overlapping[1:]

    # Trim trailing filler morphemes
    while len(overlapping) > 1 and overlapping[-1]['pos'] in FILLER_POS:
        overlapping = overlapping[:-1]

    if not overlapping:
        return span

    # Build aligned span from remaining morphemes
    aligned_start = overlapping[0]['start']
    aligned_end = overlapping[-1]['end']
    aligned = text[aligned_start:aligned_end]

    return aligned


def expand_opinion_with_adverb(opinion, text, morphemes):
    """v2: extend opinion span leftward to include a preceding adverb/intensifier.

    Pattern: 副詞 (adverb) directly preceding the opinion → merge.
    E.g. "親切" → "とても親切" when text contains "とても親切".

    Only fires when the adverb is IMMEDIATELY before the opinion (or separated
    only by particles which themselves precede the opinion).
    """
    if opinion in ('NULL', '', None):
        return opinion
    pos = text.find(opinion)
    if pos == -1:
        return opinion

    # Find the morpheme starting exactly at `pos`
    start_morph_idx = None
    for i, m in enumerate(morphemes):
        if m['start'] == pos:
            start_morph_idx = i
            break
    if start_morph_idx is None or start_morph_idx == 0:
        return opinion

    # Walk backwards: skip 助詞 (particles like の/が) until we hit something
    look_back = start_morph_idx - 1
    while look_back >= 0 and morphemes[look_back]['pos'] == '助詞':
        look_back -= 1
    if look_back < 0:
        return opinion

    # Only expand if the previous morpheme is in the strict intensifier whitelist
    prev = morphemes[look_back]
    if prev['surface'] not in INTENSIFIER_WHITELIST:
        return opinion

    # Build extended span from adverb to end of opinion
    new_start = prev['start']
    new_end = pos + len(opinion)
    return text[new_start:new_end]


def is_content_span(span, text, morphemes):
    """Check if a span contains at least one content morpheme."""
    if span in ('NULL', '', None):
        return True

    span_start = text.find(span)
    if span_start == -1:
        return True
    span_end = span_start + len(span)

    for m in morphemes:
        if m['end'] > span_start and m['start'] < span_end:
            if m['pos'] in CONTENT_POS:
                return True

    return False


def postprocess_predictions(predictions, gold_data, tagger,
                             expand_opinion_adverb: bool = False):
    """Apply MeCab-based post-processing to predictions."""
    gold_map = {g['ID']: g['Text'] for g in gold_data}

    processed = []
    stats = {'total': 0, 'aligned': 0, 'filtered': 0}

    for pred in predictions:
        text = gold_map.get(pred['ID'], '')
        if not text:
            processed.append(pred)
            continue

        morphemes = get_morphemes(tagger, text)

        new_quads = []
        for q in pred.get('Quadruplet', []):
            stats['total'] += 1
            aspect = q['Aspect']
            opinion = q['Opinion']

            # Align spans to morpheme boundaries
            new_aspect = align_span(text, aspect, morphemes)
            new_opinion = align_span(text, opinion, morphemes)

            # v2: extend opinion to include preceding adverb (とても/かなり/...)
            if expand_opinion_adverb:
                new_opinion = expand_opinion_with_adverb(new_opinion, text, morphemes)

            if new_aspect != aspect or new_opinion != opinion:
                stats['aligned'] += 1

            # Filter out non-content spans
            if not is_content_span(new_aspect, text, morphemes) and new_aspect != 'NULL':
                stats['filtered'] += 1
                continue
            if not is_content_span(new_opinion, text, morphemes):
                stats['filtered'] += 1
                continue

            new_quads.append({
                'Aspect': new_aspect,
                'Category': q['Category'],
                'Opinion': new_opinion,
                'VA': q['VA'],
            })

        processed.append({'ID': pred['ID'], 'Quadruplet': new_quads})

    return processed, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input predictions JSONL')
    parser.add_argument('--gold', required=True, help='Gold data JSONL (for text)')
    parser.add_argument('--output', required=True, help='Output processed JSONL')
    parser.add_argument('--expand_opinion_adverb', action='store_true',
                        help='v2: extend opinion to include preceding adverb '
                             '(e.g. "親切" → "とても親切")')
    args = parser.parse_args()

    tagger = MeCab.Tagger()

    with open(args.input) as f:
        predictions = [json.loads(l) for l in f]
    with open(args.gold) as f:
        gold_data = [json.loads(l) for l in f]

    processed, stats = postprocess_predictions(predictions, gold_data, tagger,
                                                expand_opinion_adverb=args.expand_opinion_adverb)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for p in processed:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Processed {len(processed)} samples")
    print(f"  Total quads: {stats['total']}")
    print(f"  Aligned: {stats['aligned']}")
    print(f"  Filtered: {stats['filtered']}")


if __name__ == '__main__':
    main()
