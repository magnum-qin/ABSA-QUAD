"""
Cross-encoder ensemble for DimASQP.

Runs inference with checkpoints that use DIFFERENT encoders (e.g. V13 mDeBERTa
+ V15 XLM-R-large) on the SAME test set. Each seed is loaded with its own
tokenizer/encoder. Predictions are merged by majority voting across all seeds.

Usage:
    python scripts/cross_encoder_ensemble.py \
        --group v13:checkpoints/v13_comet096_jpn_hotel_seed{42,123,456,789,1024} \
        --group v15:checkpoints/v15_xlmr_jpn_seed{42,123,456,789,1024} \
        --test_file data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
        --output predictions/cross_v13_v15_vote5.jsonl \
        --vote_threshold 5 \
        --span_topk_threshold 0.15 \
        --pair_threshold 0.3
"""
import os
import sys
import json
import argparse
import copy
from pathlib import Path
from collections import defaultdict, Counter

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dimabsa_model import DimASQPModel
from src.data.dataset import DimASQPDataset, collate_fn

# Reuse logic from inference.py
from scripts.inference import (
    _merge_config_from_checkpoint,
    load_model,
    load_ensemble_model,
    generate_predictions,
)


def parse_group_arg(arg: str):
    """Parse 'name:path1,path2,...' or 'name:path1 name:path2 ...' format."""
    if ':' not in arg:
        raise ValueError(f"--group must be 'name:paths' format, got {arg!r}")
    name, paths = arg.split(':', 1)
    paths = [p.strip() for p in paths.split(',') if p.strip()]
    return name, paths


def run_group(group_name: str, seed_dirs: list, test_file: str,
              base_config_path: str, batch_size: int, device: str,
              pair_threshold: float, span_topk_threshold: float,
              dual_layer: bool) -> list:
    """Run inference for one encoder group. Returns per-sample list of quads
    aggregated over all seeds in this group (each seed votes once)."""
    print(f"\n{'='*60}\nGroup '{group_name}' ({len(seed_dirs)} seeds)\n{'='*60}")

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    # Peek first checkpoint to get correct encoder_name + tokenizer
    first_ckpt_path = Path(seed_dirs[0]) / "checkpoint_best.pt"
    first_ckpt = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
    group_config = _merge_config_from_checkpoint(copy.deepcopy(base_config), first_ckpt)
    del first_ckpt

    encoder_name = group_config['model']['encoder_name']
    print(f"  Encoder: {encoder_name}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    test_dataset = DimASQPDataset(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=group_config['data']['max_seq_length']
    )
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    all_preds = []  # list of (per-sample) list of quads, one per seed
    for i, seed_dir in enumerate(seed_dirs):
        print(f"  [{i+1}/{len(seed_dirs)}] {seed_dir}")
        if dual_layer:
            epoch_ckpts = sorted(Path(seed_dir).glob('checkpoint_epoch*.pt'))
            if epoch_ckpts:
                peek_ckpt = torch.load(epoch_ckpts[0], map_location='cpu', weights_only=False)
                seed_config = _merge_config_from_checkpoint(copy.deepcopy(group_config), peek_ckpt)
                model = load_ensemble_model(seed_dir, seed_config, device)
            else:
                ckpt_path = Path(seed_dir) / "checkpoint_best.pt"
                model = load_model(str(ckpt_path), copy.deepcopy(group_config), device)
        else:
            ckpt_path = Path(seed_dir) / "checkpoint_best.pt"
            model = load_model(str(ckpt_path), copy.deepcopy(group_config), device)

        if span_topk_threshold > 0:
            model.span_extractor.topk_threshold = span_topk_threshold
            if hasattr(model, 'span_extractor_stage2'):
                model.span_extractor_stage2.topk_threshold = span_topk_threshold

        preds = generate_predictions(model, dataloader, device, pair_threshold)
        all_preds.append(preds)
        del model
        torch.cuda.empty_cache()

    return all_preds


def vote_merge(all_seed_preds: list, vote_threshold: int) -> list:
    """Merge per-seed predictions by voting on (Aspect, Category, Opinion) keys.

    all_seed_preds: list[seed][sample] -> dict with 'ID' and 'Quadruplet'.
    Returns: list[sample] -> dict ready to write.
    """
    n_samples = len(all_seed_preds[0])
    n_seeds = len(all_seed_preds)
    merged = []
    for s_idx in range(n_samples):
        sample_id = all_seed_preds[0][s_idx]['ID']

        # Count quad votes; track ONE va per key (mean across voting seeds)
        quad_votes = Counter()
        quad_va_sum = defaultdict(lambda: [0.0, 0.0, 0])  # [v_sum, a_sum, n]

        for seed_preds in all_seed_preds:
            seen_keys = set()
            for q in seed_preds[s_idx].get('Quadruplet', []):
                key = (q['Aspect'], q['Category'], q['Opinion'])
                if key in seen_keys:
                    continue  # avoid double-counting within a seed
                seen_keys.add(key)
                quad_votes[key] += 1
                try:
                    v, a = q['VA'].split('#')
                    v, a = float(v), float(a)
                except (ValueError, KeyError, AttributeError):
                    continue
                quad_va_sum[key][0] += v
                quad_va_sum[key][1] += a
                quad_va_sum[key][2] += 1

        kept = []
        for key, count in quad_votes.items():
            if count < vote_threshold:
                continue
            v_sum, a_sum, n = quad_va_sum[key]
            if n > 0:
                v = max(1.0, min(9.0, v_sum / n))
                a = max(1.0, min(9.0, a_sum / n))
            else:
                v, a = 5.0, 5.0
            asp, cat, opn = key
            kept.append({"Aspect": asp, "Category": cat, "Opinion": opn,
                         "VA": f"{v:.2f}#{a:.2f}"})

        merged.append({"ID": sample_id, "Quadruplet": kept})
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', action='append', required=True,
                        help="Encoder group as 'name:dir1,dir2,...'. Repeat for multiple groups.")
    parser.add_argument('--config', required=True,
                        help="Base config (only used for non-encoder fields like data paths)")
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vote_threshold', type=int, required=True,
                        help="Min seeds that must predict a quad to keep it")
    parser.add_argument('--pair_threshold', type=float, default=0.3)
    parser.add_argument('--span_topk_threshold', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dual_layer', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Vote threshold: {args.vote_threshold}")
    print(f"Span top-k threshold: {args.span_topk_threshold}")

    all_seed_preds = []
    for group_spec in args.group:
        name, dirs = parse_group_arg(group_spec)
        group_preds = run_group(
            group_name=name, seed_dirs=dirs, test_file=args.test_file,
            base_config_path=args.config, batch_size=args.batch_size,
            device=device, pair_threshold=args.pair_threshold,
            span_topk_threshold=args.span_topk_threshold,
            dual_layer=args.dual_layer,
        )
        all_seed_preds.extend(group_preds)
    print(f"\nTotal seeds across all groups: {len(all_seed_preds)}")

    merged = vote_merge(all_seed_preds, args.vote_threshold)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Wrote {len(merged)} predictions to {args.output}")


if __name__ == '__main__':
    main()
