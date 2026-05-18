"""
Per-category vote threshold calibration.

Runs each seed of an ensemble individually, then applies vote thresholds that
differ by category. Over-predicted categories (compared to gold) get a higher
threshold; under-predicted get a lower threshold.

Usage:
    python scripts/per_category_vote.py \
        --seed_dirs checkpoints/v15_xlmr_jpn_seed{42,123,456,789,1024} \
        --config /tmp/pipeline_jpn_v15.yaml \
        --test_file data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
        --output predictions/v15_per_cat_vote.jsonl \
        --span_topk_threshold 0.15 \
        --pair_threshold 0.3 \
        --base_vote 3 \
        --over_categories HOTEL#COMFORT ROOMS#QUALITY HOTEL#PRICE \
        --over_vote 4
"""
import os, sys, json, copy, argparse
from pathlib import Path
from collections import defaultdict, Counter

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import DimASQPDataset, collate_fn
from scripts.inference import (
    _merge_config_from_checkpoint,
    load_model,
    load_ensemble_model,
    generate_predictions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_dirs', nargs='+', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base_vote', type=int, default=3,
                        help="Default vote threshold for categories not in --over_categories")
    parser.add_argument('--over_categories', nargs='*', default=[],
                        help="Categories to apply --over_vote threshold")
    parser.add_argument('--over_vote', type=int, default=4,
                        help="Vote threshold for over-predicted categories")
    parser.add_argument('--under_categories', nargs='*', default=[],
                        help="Categories to apply --under_vote threshold")
    parser.add_argument('--under_vote', type=int, default=2,
                        help="Vote threshold for under-predicted categories")
    parser.add_argument('--pair_threshold', type=float, default=0.3)
    parser.add_argument('--span_topk_threshold', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dual_layer', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config + tokenizer
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    first_ckpt = torch.load(Path(args.seed_dirs[0]) / "checkpoint_best.pt",
                            map_location='cpu', weights_only=False)
    config = _merge_config_from_checkpoint(copy.deepcopy(base_config), first_ckpt)
    del first_ckpt
    tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])

    dataset = DimASQPDataset(
        data_path=args.test_file, tokenizer=tokenizer,
        max_length=config['data']['max_seq_length'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    # Run each seed
    all_preds = []
    for i, sd in enumerate(args.seed_dirs):
        print(f"[{i+1}/{len(args.seed_dirs)}] {sd}")
        if args.dual_layer:
            epoch_ckpts = sorted(Path(sd).glob('checkpoint_epoch*.pt'))
            if epoch_ckpts:
                peek = torch.load(epoch_ckpts[0], map_location='cpu', weights_only=False)
                seed_cfg = _merge_config_from_checkpoint(copy.deepcopy(config), peek)
                model = load_ensemble_model(sd, seed_cfg, device)
            else:
                model = load_model(str(Path(sd) / "checkpoint_best.pt"),
                                   copy.deepcopy(config), device)
        else:
            model = load_model(str(Path(sd) / "checkpoint_best.pt"),
                               copy.deepcopy(config), device)
        if args.span_topk_threshold > 0:
            model.span_extractor.topk_threshold = args.span_topk_threshold
            if hasattr(model, 'span_extractor_stage2'):
                model.span_extractor_stage2.topk_threshold = args.span_topk_threshold
        preds = generate_predictions(model, loader, device, args.pair_threshold)
        all_preds.append(preds)
        del model; torch.cuda.empty_cache()

    # Per-category vote merge
    over_set = set(args.over_categories)
    under_set = set(args.under_categories)
    n_samples = len(all_preds[0])
    merged = []
    stats = Counter()
    for s in range(n_samples):
        sid = all_preds[0][s]['ID']
        votes = Counter()
        va_sum = defaultdict(lambda: [0.0, 0.0, 0])
        for seed_preds in all_preds:
            seen = set()
            for q in seed_preds[s].get('Quadruplet', []):
                key = (q['Aspect'], q['Category'], q['Opinion'])
                if key in seen: continue
                seen.add(key)
                votes[key] += 1
                try:
                    v, a = q['VA'].split('#')
                    v, a = float(v), float(a)
                    va_sum[key][0] += v
                    va_sum[key][1] += a
                    va_sum[key][2] += 1
                except Exception:
                    pass
        kept = []
        for key, vote in votes.items():
            cat = key[1]
            if cat in over_set:
                thr = args.over_vote
            elif cat in under_set:
                thr = args.under_vote
            else:
                thr = args.base_vote
            if vote < thr:
                stats[f"drop_{cat}"] += 1
                continue
            stats[f"keep_{cat}"] += 1
            vs, as_, n = va_sum[key]
            if n > 0:
                v = max(1.0, min(9.0, vs / n))
                a = max(1.0, min(9.0, as_ / n))
            else:
                v, a = 5.0, 5.0
            asp, _, opn = key
            kept.append({"Aspect": asp, "Category": cat, "Opinion": opn,
                         "VA": f"{v:.2f}#{a:.2f}"})
        merged.append({"ID": sid, "Quadruplet": kept})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"\nWrote {len(merged)} predictions to {args.output}")
    print(f"Over-categories (vote>={args.over_vote}): {sorted(over_set)}")
    print(f"Under-categories (vote>={args.under_vote}): {sorted(under_set)}")
    print(f"Default (vote>={args.base_vote})")


if __name__ == '__main__':
    main()
