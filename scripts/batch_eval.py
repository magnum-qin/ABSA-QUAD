"""
Batch inference and evaluation for all language/domain datasets.

Generates predictions for each dataset using trained checkpoints,
then evaluates using the official DimABSA2026 evaluation script.

Usage:
    # Dev set: inference + evaluation
    python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

    # Test set: inference only (no gold labels)
    python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --test

    # Specific datasets
    python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --datasets eng_restaurant jpn_hotel

    # Ensemble
    python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --ensemble
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Import dataset definitions from train script
from train_all_languages import DATASETS


def run_inference(config_path: str, dataset_name: str, ds_info: dict,
                  checkpoint_dir: str, output_dir: str,
                  ensemble: bool = False, use_test: bool = False) -> bool:
    """Run inference for a single dataset. Returns True if successful."""
    ckpt_dir = Path(checkpoint_dir) / dataset_name
    suffix = "test_pred" if use_test else "pred"
    output_path = Path(output_dir) / f"{dataset_name}_{suffix}.jsonl"
    input_file = ds_info['test'] if use_test else ds_info['dev']

    if not Path(input_file).exists():
        print(f"  SKIP: file not found: {input_file}")
        return False

    cmd = [sys.executable, "scripts/inference.py",
           "--config", config_path,
           "--test_file", input_file,
           "--output", str(output_path)]

    if ensemble:
        cmd += ["--ensemble", "--checkpoint_dir", str(ckpt_dir)]
    else:
        ckpt_path = ckpt_dir / "checkpoint_best.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found: {ckpt_path}")
            return False
        cmd += ["--checkpoint", str(ckpt_path)]

    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: inference failed")
        print(result.stderr[-500:] if result.stderr else "")
        return False
    return True


def run_evaluation(dataset_name: str, ds_info: dict, output_dir: str) -> dict:
    """Run official evaluation for a single dataset. Returns metrics dict."""
    pred_path = Path(output_dir) / f"{dataset_name}_pred.jsonl"
    gold_path = ds_info['dev']

    cmd = [sys.executable, "scripts/official_eval.py",
           "-p", str(pred_path),
           "-g", gold_path,
           "-t", "3"]

    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: evaluation failed")
        print(result.stderr[-500:] if result.stderr else "")
        return {"error": True}

    # Parse evaluation output
    metrics = {}
    for line in result.stdout.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                pass

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference and evaluation")
    parser.add_argument("--base_config", type=str, default="configs/config_rethink_v3.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/all_lang")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to evaluate (default: all). Choices: {list(DATASETS.keys())}")
    parser.add_argument("--ensemble", action="store_true", help="Use weight-averaged ensemble")
    parser.add_argument("--test", action="store_true",
                        help="Use test set (inference only, no evaluation)")
    args = parser.parse_args()

    dataset_names = args.datasets or list(DATASETS.keys())
    invalid = [d for d in dataset_names if d not in DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = "TEST (inference only)" if args.test else "DEV (inference + evaluation)"
    print(f"Batch Evaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {mode}")
    print(f"Datasets: {dataset_names}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print()

    # Run inference (+ evaluation for dev mode)
    results = []
    successful = []
    for name in dataset_names:
        ds_info = DATASETS[name]
        print(f"{'='*60}")
        print(f"Dataset: {name}")

        # Inference
        print("  Running inference...")
        if not run_inference(args.base_config, name, ds_info,
                             args.checkpoint_dir, args.output_dir,
                             args.ensemble, args.test):
            results.append({"dataset": name, "error": True})
            continue

        if args.test:
            # Test mode: no gold labels, just record success
            suffix = "test_pred"
            output_path = Path(args.output_dir) / f"{name}_{suffix}.jsonl"
            print(f"  Saved: {output_path}")
            results.append({"dataset": name, "output": str(output_path)})
            successful.append(name)
        else:
            # Dev mode: run evaluation
            print("  Running evaluation...")
            metrics = run_evaluation(name, ds_info, args.output_dir)
            metrics["dataset"] = name
            results.append(metrics)

            if not metrics.get("error"):
                cf1 = metrics.get("cF1", 0)
                print(f"  cF1: {cf1:.4f}")
                successful.append(name)

    # Summary
    print(f"\n{'='*75}")
    if args.test:
        print("TEST PREDICTIONS SUMMARY")
        print(f"{'='*75}")
        print(f"{'Dataset':<20} {'Status':>10} {'Output File'}")
        print(f"{'-'*20} {'-'*10} {'-'*40}")
        for r in results:
            if r.get("error"):
                print(f"{r['dataset']:<20} {'ERROR':>10}")
            else:
                print(f"{r['dataset']:<20} {'OK':>10} {r.get('output', '')}")
        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        print(f"Output dir: {args.output_dir}/")
    else:
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*75}")
        print(f"{'Dataset':<20} {'cF1':>10} {'cPrecision':>12} {'cRecall':>10}")
        print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*10}")

        valid = []
        for r in results:
            if r.get("error"):
                print(f"{r['dataset']:<20} {'ERROR':>10}")
            else:
                cf1 = r.get("cF1", 0)
                cp = r.get("cPrecision", 0)
                cr = r.get("cRecall", 0)
                print(f"{r['dataset']:<20} {cf1:>10.4f} {cp:>12.4f} {cr:>10.4f}")
                valid.append(r)

        if valid:
            avg_cf1 = sum(r.get("cF1", 0) for r in valid) / len(valid)
            print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*10}")
            print(f"{'AVERAGE':<20} {avg_cf1:>10.4f}")
            print(f"\nSuccessful: {len(valid)}/{len(results)}")

            best = max(valid, key=lambda x: x.get("cF1", 0))
            print(f"Best: {best['dataset']} (cF1={best.get('cF1', 0):.4f})")

    # Save results
    results_path = Path(args.output_dir) / ("test_results.json" if args.test else "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": args.base_config,
            "mode": "test" if args.test else "dev",
            "ensemble": args.ensemble,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
