"""
Confidence-Weighted Multi-Seed Ensemble (ablation #5).

Standard ensemble: each seed contributes one equal-weight vote.
Confidence-weighted: weight each seed by its dev cF1 (better seeds count more).

Threshold semantic preserved: vote_threshold=3 still means "3 effective votes".
Implemented as: keep quad if (sum of weights for seeds that voted) >= vote_threshold * (mean weight).
With uniform weights = 1.0 this is identical to the original ensemble.

Usage:
    python scripts/confidence_weighted_ensemble.py \
        --seed_dirs checkpoints/v16_native_jpn_seed{42,123,456,789,1024} \
        --seed_weights 0.4019 0.4035 0.4128 0.3926 0.4023 \
        --config /tmp/pipeline_jpn_v16.yaml \
        --test_file data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl \
        --output predictions/v16_jpn_weighted_vote.jsonl \
        --vote_threshold 4 --span_topk_threshold 0.0
"""
import os, sys, json, copy, argparse
from pathlib import Path
from collections import defaultdict

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import DimASQPDataset, collate_fn
from scripts.inference import (
    _merge_config_from_checkpoint,
    load_model,
    generate_predictions,
)


def weighted_merge(all_model_preds, seed_weights, vote_threshold):
    """Merge per-seed predictions with confidence-weighted vote.

    Quad kept if (sum of weights for seeds that voted for it) >= vote_threshold * mean_weight.
    VA averaged with same weights (weighted mean).
    """
    n_models = len(all_model_preds)
    mean_w = sum(seed_weights) / n_models
    threshold_w = vote_threshold * mean_w
    n_samples = len(all_model_preds[0])

    merged = []
    for idx in range(n_samples):
        sample_id = all_model_preds[0][idx]["ID"]
        quad_weights = defaultdict(list)  # key -> list of weights
        quad_va = defaultdict(list)       # key -> list of (v, a)
        for seed_idx, model_preds in enumerate(all_model_preds):
            w = seed_weights[seed_idx]
            for quad in model_preds[idx]["Quadruplet"]:
                key = (quad["Aspect"], quad["Category"], quad["Opinion"])
                v, a = quad["VA"].split("#")
                quad_weights[key].append(w)
                quad_va[key].append((float(v), float(a)))

        final_quads = []
        for key, ws in quad_weights.items():
            sum_w = sum(ws)
            if sum_w < threshold_w:
                continue
            vas = quad_va[key]
            avg_v = sum(w * v for w, (v, _) in zip(ws, vas)) / sum_w
            avg_a = sum(w * a for w, (_, a) in zip(ws, vas)) / sum_w
            avg_v = max(1.0, min(9.0, avg_v))
            avg_a = max(1.0, min(9.0, avg_a))
            asp, cat, opn = key
            final_quads.append({
                "Aspect": asp, "Category": cat, "Opinion": opn,
                "VA": f"{avg_v:.2f}#{avg_a:.2f}",
            })
        merged.append({"ID": sample_id, "Quadruplet": final_quads})
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", required=True)
    p.add_argument("--seed_weights", nargs="+", type=float, required=True,
                   help="Per-seed weights (typically dev cF1)")
    p.add_argument("--config", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vote_threshold", type=int, default=3)
    p.add_argument("--pair_threshold", type=float, default=0.3)
    p.add_argument("--span_topk_threshold", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    assert len(args.seed_dirs) == len(args.seed_weights), \
        "Number of seed_dirs must equal number of seed_weights"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load config & tokenizer
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    peek_ckpt = torch.load(Path(args.seed_dirs[0]) / "checkpoint_best.pt",
                           map_location="cpu", weights_only=False)
    config = _merge_config_from_checkpoint(copy.deepcopy(base_config), peek_ckpt)
    del peek_ckpt
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])

    dataset = DimASQPDataset(
        data_path=args.test_file, tokenizer=tokenizer,
        max_length=config["data"]["max_seq_length"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    # Per-seed inference
    all_preds = []
    for i, sd in enumerate(args.seed_dirs):
        ckpt = str(Path(sd) / "checkpoint_best.pt")
        print(f"[{i+1}/{len(args.seed_dirs)}] {ckpt} (weight={args.seed_weights[i]:.4f})")
        model = load_model(ckpt, copy.deepcopy(config), device)
        if args.span_topk_threshold > 0:
            model.span_extractor.topk_threshold = args.span_topk_threshold
            if hasattr(model, "span_extractor_stage2"):
                model.span_extractor_stage2.topk_threshold = args.span_topk_threshold
        preds = generate_predictions(model, loader, device, args.pair_threshold)
        all_preds.append(preds)
        del model
        torch.cuda.empty_cache()

    # Weighted merge
    n = len(args.seed_dirs)
    mean_w = sum(args.seed_weights) / n
    print(f"\nWeighted vote: n_models={n}, mean_weight={mean_w:.4f}, "
          f"vote_threshold={args.vote_threshold}, "
          f"effective threshold_weight={args.vote_threshold * mean_w:.4f}")
    merged = weighted_merge(all_preds, args.seed_weights, args.vote_threshold)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(merged)} predictions to {args.output}")


if __name__ == "__main__":
    main()
