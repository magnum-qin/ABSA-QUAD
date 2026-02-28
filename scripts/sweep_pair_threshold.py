"""
Sweep pair_threshold on dev set to find optimal value.
Loads model once, then iterates over thresholds.

Usage:
    python scripts/sweep_pair_threshold.py \
        --checkpoint checkpoints/rethink_v4_crf/checkpoint_best.pt \
        --config configs/config_rethink_v4_ablation.yaml
"""

import sys
import json
import yaml
import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dimabsa_model import DimASQPModel
from src.data.dataset import DimASQPDataset, collate_fn
from transformers import AutoTokenizer
from scripts.inference import load_model


def run_eval(pred_path: str, gold_path: str) -> dict:
    """Run official eval via subprocess, return metrics dict."""
    python = sys.executable
    cmd = [python, 'scripts/official_eval.py', '-g', gold_path, '-p', pred_path, '-t', '3']
    result = subprocess.run(cmd, capture_output=True, text=True)
    metrics = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith('cF1:'):
            metrics['cF1'] = float(line.split(':')[1].strip())
        elif line.startswith('cPrecision:'):
            metrics['cPrecision'] = float(line.split(':')[1].strip())
        elif line.startswith('cRecall:'):
            metrics['cRecall'] = float(line.split(':')[1].strip())
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Sweep pair_threshold on dev set')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config_rethink_v4_ablation.yaml')
    parser.add_argument('--dev_file', type=str, default=None)
    parser.add_argument('--start', type=float, default=0.05)
    parser.add_argument('--end', type=float, default=0.70)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dev_file = args.dev_file or config['data']['dev_file']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Dev file: {dev_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # Load model once
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])

    # Create dev dataset and dataloader
    print("Loading dev data...")
    dev_dataset = DimASQPDataset(
        data_path=dev_file,
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    print(f"Dev samples: {len(dev_dataset)}")

    # Generate thresholds
    thresholds = []
    t = args.start
    while t <= args.end + 1e-9:
        thresholds.append(round(t, 3))
        t += args.step

    print(f"\nSweeping pair_threshold: {thresholds}\n")

    results = []
    for threshold in thresholds:
        # Generate predictions with this threshold
        all_predictions = []
        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                texts = batch['texts']
                sample_ids = batch['sample_ids']

                predictions = model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    texts=texts,
                    offset_mapping=batch['offset_mapping'],
                    pair_threshold=threshold
                )

                for sample_id, quads in zip(sample_ids, predictions):
                    all_predictions.append({
                        "ID": sample_id,
                        "Quadruplet": quads
                    })

        # Write predictions to temp file
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            for pred in all_predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
            pred_path = f.name

        # Evaluate
        metrics = run_eval(pred_path, dev_file)
        Path(pred_path).unlink(missing_ok=True)

        cf1 = metrics.get('cF1', 0)
        cp = metrics.get('cPrecision', 0)
        cr = metrics.get('cRecall', 0)
        n_quads = sum(len(p['Quadruplet']) for p in all_predictions)
        results.append((threshold, cf1, cp, cr, n_quads))
        print(f"  threshold={threshold:.3f}  cF1={cf1:.4f}  cP={cp:.4f}  cR={cr:.4f}  quads={n_quads}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Threshold':>10} {'cF1':>8} {'cP':>8} {'cR':>8} {'#Quads':>8}")
    print("-" * 70)
    best_t, best_f1 = 0, 0
    for t, cf1, cp, cr, nq in results:
        marker = " <-- best" if cf1 > best_f1 else ""
        if cf1 > best_f1:
            best_t, best_f1 = t, cf1
        print(f"{t:>10.3f} {cf1:>8.4f} {cp:>8.4f} {cr:>8.4f} {nq:>8}{marker}")

    print("=" * 70)
    print(f"\nBest threshold: {best_t:.3f} (cF1={best_f1:.4f})")


if __name__ == '__main__':
    main()
