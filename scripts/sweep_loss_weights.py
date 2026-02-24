"""
Loss Weights Sweep Script for V3 Optimization

Runs multiple training runs with different loss_weights configurations,
using the same random seed for fair comparison. Results are printed as
a summary table at the end.

Usage:
    python scripts/sweep_loss_weights.py --config configs/config_rethink_v3.yaml
    python scripts/sweep_loss_weights.py --config configs/config_rethink_v3.yaml --groups baseline span-focus
"""

import os
import sys
import yaml
import copy
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Predefined weight groups
WEIGHT_GROUPS = {
    "baseline": {
        "span": 1.0, "pair": 1.0, "entity": 1.0,
        "attribute": 1.0, "va": 1.0, "consistency": 0.5
    },
    "span-focus": {
        "span": 2.0, "pair": 1.0, "entity": 1.0,
        "attribute": 1.0, "va": 1.0, "consistency": 0.5
    },
    "va-boost": {
        "span": 1.0, "pair": 1.0, "entity": 1.0,
        "attribute": 1.0, "va": 2.0, "consistency": 0.5
    },
    "span+va": {
        "span": 1.5, "pair": 1.0, "entity": 1.0,
        "attribute": 1.0, "va": 1.5, "consistency": 0.3
    },
    "pair-down": {
        "span": 1.5, "pair": 0.5, "entity": 1.0,
        "attribute": 1.0, "va": 1.5, "consistency": 0.3
    },
}


def run_training(config_path: str, group_name: str, weights: dict, base_config: dict):
    """Run a single training with the given loss weights."""
    config = copy.deepcopy(base_config)
    config['training']['loss_weights'] = weights
    config['training']['save_dir'] = f"checkpoints/sweep_{group_name}"

    # Write temporary config
    tmp_config_path = f"/tmp/sweep_{group_name}.yaml"
    with open(tmp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Running group: {group_name}")
    print(f"  Weights: {weights}")
    print(f"  Save dir: {config['training']['save_dir']}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "scripts/train.py",
        "--config", tmp_config_path
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Clean up
    Path(tmp_config_path).unlink(missing_ok=True)

    return result.returncode


def collect_results(group_name: str) -> dict:
    """Collect best metrics from a training run's checkpoint."""
    ckpt_path = Path(f"checkpoints/sweep_{group_name}/checkpoint_best.pt")
    if not ckpt_path.exists():
        return {"group": group_name, "cF1": 0.0, "exact_f1": 0.0, "va_rmse": 0.0, "error": True}

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return {
        "group": group_name,
        "cF1": metrics.get("continuous_f1", 0.0),
        "exact_f1": metrics.get("exact_f1", 0.0),
        "va_rmse": metrics.get("va_rmse", 0.0),
        "epoch": ckpt.get("epoch", -1),
        "error": False
    }


def main():
    parser = argparse.ArgumentParser(description="Loss Weights Sweep")
    parser.add_argument("--config", type=str, default="configs/config_rethink_v3.yaml",
                        help="Base config file")
    parser.add_argument("--groups", nargs="+", default=None,
                        help=f"Which groups to run (default: all). Choices: {list(WEIGHT_GROUPS.keys())}")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    groups = args.groups or list(WEIGHT_GROUPS.keys())
    invalid = [g for g in groups if g not in WEIGHT_GROUPS]
    if invalid:
        print(f"Unknown groups: {invalid}. Available: {list(WEIGHT_GROUPS.keys())}")
        sys.exit(1)

    print(f"Sweep started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Groups to run: {groups}")
    print(f"Base config: {args.config}")
    print(f"Seed: {base_config.get('seed', 42)}")

    # Run each group
    for group_name in groups:
        weights = WEIGHT_GROUPS[group_name]
        returncode = run_training(args.config, group_name, weights, base_config)
        if returncode != 0:
            print(f"WARNING: Group '{group_name}' exited with code {returncode}")

    # Collect and display results
    print(f"\n{'='*70}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Group':<15} {'cF1':>8} {'Exact F1':>10} {'VA RMSE':>9} {'Epoch':>7}")
    print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")

    results = []
    for group_name in groups:
        r = collect_results(group_name)
        results.append(r)
        if r["error"]:
            print(f"{r['group']:<15} {'ERROR':>8}")
        else:
            print(f"{r['group']:<15} {r['cF1']:>8.4f} {r['exact_f1']:>10.4f} {r['va_rmse']:>9.4f} {r['epoch']:>7d}")

    # Find best
    valid = [r for r in results if not r["error"]]
    if valid:
        best = max(valid, key=lambda x: x["cF1"])
        print(f"\nBest group: {best['group']} (cF1={best['cF1']:.4f})")
        print(f"Weights: {WEIGHT_GROUPS[best['group']]}")

    # Save results to file
    results_path = "checkpoints/sweep_results.json"
    Path("checkpoints").mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
