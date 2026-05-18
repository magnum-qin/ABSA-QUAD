"""
Merge V15 model predictions with LLM (DeepSeek) extracted quads.

Strategies:
  - union:        keep any quad that appears in either V15 or LLM (max recall)
  - intersection: keep only quads in both (max precision, baseline ablation)
  - llm_only:     keep only LLM quads (sanity check)
  - v15_only:     keep only V15 quads (== baseline)
  - llm_validated: V15 quads ∪ LLM quads that are "close to" some V15 quad
                   (V15 endorses LLM via fuzzy match)

VA conflict resolution:
  - When same (Aspect, Category, Opinion) appears in both, average VA.

Usage:
    python scripts/merge_llm_v15.py \
        --v15_pred predictions/v15_xlmr_jpn_topk0.15_vote3_mecab.jsonl \
        --llm_pred predictions/llm_quads_jpn_dev.jsonl \
        --output predictions/merged_v15_llm_union.jsonl \
        --strategy union
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def parse_va(s):
    try:
        v, a = s.split('#')
        return float(v), float(a)
    except Exception:
        return 5.0, 5.0


def fmt_va(v, a):
    return f"{max(1.0, min(9.0, v)):.2f}#{max(1.0, min(9.0, a)):.2f}"


def merge_one(v15_quads, llm_quads, strategy: str):
    # Index by (aspect, category, opinion) key
    v15_keys = {(q['Aspect'], q['Category'], q['Opinion']): parse_va(q['VA']) for q in v15_quads}
    llm_keys = {(q['Aspect'], q['Category'], q['Opinion']): parse_va(q['VA']) for q in llm_quads}

    if strategy == 'union':
        all_keys = set(v15_keys) | set(llm_keys)
    elif strategy == 'intersection':
        all_keys = set(v15_keys) & set(llm_keys)
    elif strategy == 'llm_only':
        all_keys = set(llm_keys)
    elif strategy == 'v15_only':
        all_keys = set(v15_keys)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    merged = []
    for key in all_keys:
        v_a = v15_keys.get(key)
        l_a = llm_keys.get(key)
        if v_a is not None and l_a is not None:
            # Both: average VA
            v = (v_a[0] + l_a[0]) / 2
            a = (v_a[1] + l_a[1]) / 2
        elif v_a is not None:
            v, a = v_a
        else:
            v, a = l_a
        asp, cat, opn = key
        merged.append({'Aspect': asp, 'Category': cat, 'Opinion': opn, 'VA': fmt_va(v, a)})
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v15_pred', required=True)
    parser.add_argument('--llm_pred', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--strategy', default='union',
                        choices=['union', 'intersection', 'llm_only', 'v15_only'])
    parser.add_argument('--llm_max_per_sample', type=int, default=None,
                        help='Cap LLM quads per sample (drops noisy excess). None=unlimited.')
    args = parser.parse_args()

    v15 = {p['ID']: p.get('Quadruplet', []) for p in load_jsonl(args.v15_pred)}
    llm = {p['ID']: p.get('Quadruplet', []) for p in load_jsonl(args.llm_pred)}

    all_ids = set(v15) | set(llm)
    print(f'V15 samples: {len(v15)}, LLM samples: {len(llm)}, total IDs: {len(all_ids)}')

    n_v15_q = sum(len(q) for q in v15.values())
    n_llm_q = sum(len(q) for q in llm.values())
    print(f'V15 quads: {n_v15_q} (avg {n_v15_q/max(1,len(v15)):.2f}/sample)')
    print(f'LLM quads: {n_llm_q} (avg {n_llm_q/max(1,len(llm)):.2f}/sample)')

    out = []
    n_merged = 0
    for sid in sorted(all_ids):
        v15_q = v15.get(sid, [])
        llm_q = llm.get(sid, [])
        if args.llm_max_per_sample is not None:
            llm_q = llm_q[:args.llm_max_per_sample]
        merged = merge_one(v15_q, llm_q, args.strategy)
        n_merged += len(merged)
        out.append({'ID': sid, 'Quadruplet': merged})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'\nStrategy: {args.strategy}')
    print(f'Output: {n_merged} merged quads ({n_merged/max(1,len(out)):.2f}/sample)')
    print(f'Wrote: {args.output}')


if __name__ == '__main__':
    main()
