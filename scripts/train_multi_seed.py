"""
Multi-Seed Training Script for Ensemble

Trains the same V3 config with multiple random seeds.
Each seed produces an independent checkpoint for later ensemble inference.

Usage:
    python scripts/train_multi_seed.py --config configs/config_rethink_v3.yaml
    python scripts/train_multi_seed.py --config configs/config_rethink_v3.yaml --seeds 42 123 456
"""

import os
import sys
import yaml
import copy
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def run_training(config_path: str, seed: int, base_config: dict,
                 save_prefix: str = "ensemble_seed", train_file: str = None,
                 pretrained_checkpoint: str = None,
                 pretrained_checkpoints: list = None,
                 resume_existing: bool = False):
    """Run a single training with the given seed."""
    config = copy.deepcopy(base_config)
    config['seed'] = seed
    config['training']['save_dir'] = f"checkpoints/{save_prefix}{seed}"
    if train_file:
        config['data']['train_file'] = train_file

    # Write temporary config
    tmp_config_path = f"/tmp/ensemble_seed{seed}.yaml"
    with open(tmp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Training with seed={seed}")
    print(f"  Save dir: {config['training']['save_dir']}")
    print(f"{'='*60}\n")

    save_dir = Path(config['training']['save_dir'])
    complete_marker = save_dir / 'training_complete.json'
    if resume_existing and complete_marker.exists():
        print(f"  Already complete: {complete_marker} (skipping seed={seed})")
        return 0

    cmd = [sys.executable, "scripts/train.py", "--config", tmp_config_path]
    if pretrained_checkpoints:
        cmd.extend(["--pretrained_checkpoints"] + pretrained_checkpoints)
    elif pretrained_checkpoint:
        cmd.extend(["--pretrained_checkpoint", pretrained_checkpoint])
    if resume_existing and (save_dir / 'checkpoint_latest.pt').exists():
        cmd.extend(["--resume_checkpoint", str(save_dir / 'checkpoint_latest.pt')])
    # Ensure LD_LIBRARY_PATH includes conda lib (for libstdc++ compatibility)
    env = os.environ.copy()
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode == 0:
        complete_marker.write_text(
            f'{{"seed": {seed}, "completed_at": "{datetime.now().isoformat()}"}}\n'
        )

    Path(tmp_config_path).unlink(missing_ok=True)
    return result.returncode


def collect_results(seed: int, save_prefix: str = "ensemble_seed") -> dict:
    """Collect best metrics from a seed's checkpoint."""
    ckpt_path = Path(f"checkpoints/{save_prefix}{seed}/checkpoint_best.pt")
    if not ckpt_path.exists():
        return {"seed": seed, "cF1": 0.0, "error": True}

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return {
        "seed": seed,
        "cF1": metrics.get("continuous_f1", 0.0),
        "exact_f1": metrics.get("exact_f1", 0.0),
        "va_rmse": metrics.get("va_rmse", 0.0),
        "epoch": ckpt.get("epoch", -1),
        "error": False
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Training for Ensemble")
    parser.add_argument("--config", type=str, default="configs/config_rethink_v3.yaml",
                        help="Base config file")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to train (default: {DEFAULT_SEEDS})")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Override training data file in config")
    parser.add_argument("--save_prefix", type=str, default="ensemble_seed",
                        help="Checkpoint directory prefix (default: ensemble_seed)")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Pretrained checkpoint for cross-lingual transfer")
    parser.add_argument("--pretrained_checkpoints", type=str, nargs='+', default=None,
                        help="Multiple pretrained checkpoints for multi-source transfer (weights averaged)")
    parser.add_argument("--resume_existing", action="store_true",
                        help="Resume interrupted seeds and skip seeds with a completion marker")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    seeds = args.seeds or DEFAULT_SEEDS

    print(f"Multi-seed training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds}")
    print(f"Base config: {args.config}")

    if args.train_file:
        print(f"Train file override: {args.train_file}")
    if args.pretrained_checkpoints:
        print(f"Pretrained checkpoints (multi-source): {args.pretrained_checkpoints}")
    elif args.pretrained_checkpoint:
        print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Save prefix: {args.save_prefix}")

    # Run each seed
    for seed in seeds:
        returncode = run_training(args.config, seed, base_config,
                                  save_prefix=args.save_prefix,
                                  train_file=args.train_file,
                                  pretrained_checkpoint=args.pretrained_checkpoint,
                                  pretrained_checkpoints=args.pretrained_checkpoints,
                                  resume_existing=args.resume_existing)
        if returncode != 0:
            print(f"WARNING: seed={seed} exited with code {returncode}")

    # Summary
    print(f"\n{'='*70}")
    print("MULTI-SEED TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"{'Seed':<8} {'cF1':>8} {'Exact F1':>10} {'VA RMSE':>9} {'Epoch':>7}")
    print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")

    results = []
    for seed in seeds:
        r = collect_results(seed, save_prefix=args.save_prefix)
        results.append(r)
        if r["error"]:
            print(f"{r['seed']:<8} {'ERROR':>8}")
        else:
            print(f"{r['seed']:<8} {r['cF1']:>8.4f} {r['exact_f1']:>10.4f} {r['va_rmse']:>9.4f} {r['epoch']:>7d}")

    valid = [r for r in results if not r["error"]]
    if valid:
        avg_cf1 = sum(r['cF1'] for r in valid) / len(valid)
        best = max(valid, key=lambda x: x['cF1'])
        print(f"\nAverage cF1: {avg_cf1:.4f}")
        print(f"Best single: seed={best['seed']} (cF1={best['cF1']:.4f})")
        print(f"\nNext: run ensemble inference with:")
        print(f"  python scripts/inference.py --multi_seed_ensemble \\")
        print(f"    --seed_dirs " + " ".join(f"checkpoints/ensemble_seed{s}" for s in seeds) + " \\")
        print(f"    --config {args.config} \\")
        print(f"    --test_file {base_config['data']['dev_file']} \\")
        print(f"    --output predictions/ensemble_pred.jsonl")


if __name__ == "__main__":
    main()
