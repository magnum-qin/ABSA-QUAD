"""
Train V3 model across all language/domain datasets.

Loops through all 8 datasets, generates per-dataset configs with correct
label sets and encoder, trains each, and prints a summary table.

Usage:
    python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml
    python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml --datasets eng_restaurant jpn_hotel
    python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml --num_epochs 5  # quick test
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

# All 8 dataset configurations
DATASETS = {
    "eng_restaurant": {
        "train": "data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl",
        "dev": "data/subtask_3/eng/eng_restaurant_dev_task3.jsonl",
        "entity_labels": [
            "AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "GENERAL", "MISCELLANEOUS", "PRICES", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/deberta-v3-base"
    },
    "eng_laptop": {
        "train": "data/subtask_3/eng/eng_laptop_train_alltasks.jsonl",
        "dev": "data/subtask_3/eng/eng_laptop_dev_task3.jsonl",
        "entity_labels": [
            "BATTERY", "COMPANY", "CPU", "DISPLAY", "FANS_COOLING", "GRAPHICS",
            "HARD_DISK", "HARDWARE", "KEYBOARD", "LAPTOP", "MEMORY", "MOTHERBOARD",
            "MOUSE", "MULTIMEDIA_DEVICES", "OPTICAL_DRIVES", "OS", "OUT_OF_SCOPE",
            "PORTS", "POWER_SUPPLY", "SHIPPING", "SOFTWARE", "SUPPORT", "WARRANTY", "NULL"
        ],
        "attribute_labels": [
            "CONNECTIVITY", "DESIGN_FEATURES", "GENERAL", "MISCELLANEOUS",
            "OPERATION_PERFORMANCE", "PORTABILITY", "PRICE", "QUALITY", "USABILITY", "NULL"
        ],
        "encoder": "microsoft/deberta-v3-base"
    },
    "jpn_hotel": {
        "train": "data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl",
        "dev": "data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl",
        "entity_labels": [
            "FACILITIES", "FOOD_DRINKS", "HOTEL", "LOCATION", "ROOMS",
            "ROOM_AMENITIES", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "CLEANLINESS", "COMFORT", "DESIGN_FEATURES", "GENERAL",
            "MISCELLANEOUS", "PRICE", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
    "rus_restaurant": {
        "train": "data/subtask_3/rus/rus_restaurant_train_alltasks.jsonl",
        "dev": "data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
        "entity_labels": [
            "AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "GENERAL", "MISCELLANEOUS", "PRICES", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
    "tat_restaurant": {
        "train": "data/subtask_3/tat/tat_restaurant_train_alltasks.jsonl",
        "dev": "data/subtask_3/tat/tat_restaurant_dev_task3.jsonl",
        "entity_labels": [
            "AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "GENERAL", "MISCELLANEOUS", "PRICES", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
    "ukr_restaurant": {
        "train": "data/subtask_3/ukr/ukr_restaurant_train_alltasks.jsonl",
        "dev": "data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl",
        "entity_labels": [
            "AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "GENERAL", "MISCELLANEOUS", "PRICES", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
    "zho_restaurant": {
        "train": "data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl",
        "dev": "data/subtask_3/zho/zho_restaurant_dev_task3.jsonl",
        "entity_labels": [
            "AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE", "NULL"
        ],
        "attribute_labels": [
            "GENERAL", "MISCELLANEOUS", "PRICES", "QUALITY", "STYLE_OPTIONS", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
    "zho_laptop": {
        "train": "data/subtask_3/zho/zho_laptop_train_alltasks.jsonl",
        "dev": "data/subtask_3/zho/zho_laptop_dev_task3.jsonl",
        "entity_labels": [
            "BATTERY", "COMPANY", "CPU", "DISPLAY", "FANS_COOLING", "GRAPHICS",
            "HARD_DISK", "HARDWARE", "KEYBOARD", "LAPTOP", "MEMORY", "MOTHERBOARD",
            "MOUSE", "MULTIMEDIA_DEVICES", "OPTICAL_DRIVES", "OS",
            "PORTS", "POWER_SUPPLY", "SHIPPING", "SOFTWARE", "SUPPORT", "WARRANTY", "NULL"
        ],
        "attribute_labels": [
            "CONNECTIVITY", "DESIGN_FEATURES", "GENERAL",
            "OPERATION_PERFORMANCE", "PORTABILITY", "PRICE", "QUALITY", "USABILITY", "NULL"
        ],
        "encoder": "microsoft/mdeberta-v3-base"
    },
}


def build_config(base_config: dict, dataset_name: str, ds_info: dict,
                 num_epochs: int = None) -> dict:
    """Build a training config for a specific dataset."""
    config = copy.deepcopy(base_config)

    # Data paths
    config['data']['train_file'] = ds_info['train']
    config['data']['dev_file'] = ds_info['dev']

    # Label lists
    config['data']['entity_labels'] = ds_info['entity_labels']
    config['data']['attribute_labels'] = ds_info['attribute_labels']

    # Model labels must match
    config['model']['num_entity_labels'] = len(ds_info['entity_labels'])
    config['model']['num_attribute_labels'] = len(ds_info['attribute_labels'])

    # Encoder
    config['model']['encoder_name'] = ds_info['encoder']

    # Save directory
    config['training']['save_dir'] = f"checkpoints/all_lang/{dataset_name}"

    # Override epochs if specified
    if num_epochs is not None:
        config['training']['num_epochs'] = num_epochs

    return config


def run_training(config: dict, dataset_name: str) -> int:
    """Run training for a single dataset. Returns exit code."""
    tmp_config_path = f"/tmp/train_{dataset_name}.yaml"
    with open(tmp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Training: {dataset_name}")
    print(f"  Encoder: {config['model']['encoder_name']}")
    print(f"  Entity labels ({config['model']['num_entity_labels']}): "
          f"{config['data']['entity_labels']}")
    print(f"  Attribute labels ({config['model']['num_attribute_labels']}): "
          f"{config['data']['attribute_labels']}")
    print(f"  Train: {config['data']['train_file']}")
    print(f"  Dev:   {config['data']['dev_file']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Save:  {config['training']['save_dir']}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, "scripts/train.py", "--config", tmp_config_path]
    result = subprocess.run(cmd, text=True)

    Path(tmp_config_path).unlink(missing_ok=True)
    return result.returncode


def collect_results(dataset_name: str) -> dict:
    """Collect best metrics from a training run."""
    ckpt_path = Path(f"checkpoints/all_lang/{dataset_name}/checkpoint_best.pt")
    if not ckpt_path.exists():
        return {"dataset": dataset_name, "error": True}

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    return {
        "dataset": dataset_name,
        "cF1": metrics.get("continuous_f1", 0.0),
        "exact_f1": metrics.get("exact_f1", 0.0),
        "va_rmse": metrics.get("va_rmse", 0.0),
        "epoch": ckpt.get("epoch", -1),
        "error": False
    }


def main():
    parser = argparse.ArgumentParser(description="Train V3 on all languages/domains")
    parser.add_argument("--base_config", type=str, default="configs/config_rethink_v3.yaml",
                        help="Base V3 config file")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Which datasets to run (default: all). "
                             f"Choices: {list(DATASETS.keys())}")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of epochs (useful for quick testing)")
    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    dataset_names = args.datasets or list(DATASETS.keys())
    invalid = [d for d in dataset_names if d not in DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    print(f"All-Language Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {dataset_names}")
    print(f"Base config: {args.base_config}")
    print(f"Seed: {base_config.get('seed', 42)}")
    if args.num_epochs:
        print(f"Epochs override: {args.num_epochs}")

    # Train each dataset
    for name in dataset_names:
        ds_info = DATASETS[name]
        config = build_config(base_config, name, ds_info, args.num_epochs)
        returncode = run_training(config, name)
        if returncode != 0:
            print(f"WARNING: {name} exited with code {returncode}")

    # Collect and display results
    print(f"\n{'='*75}")
    print("ALL-LANGUAGE RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f"{'Dataset':<20} {'cF1':>8} {'Exact F1':>10} {'VA RMSE':>9} {'Epoch':>7}")
    print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")

    results = []
    for name in dataset_names:
        r = collect_results(name)
        results.append(r)
        if r.get("error"):
            print(f"{r['dataset']:<20} {'ERROR':>8}")
        else:
            print(f"{r['dataset']:<20} {r['cF1']:>8.4f} {r['exact_f1']:>10.4f} "
                  f"{r['va_rmse']:>9.4f} {r['epoch']:>7d}")

    # Compute averages
    valid = [r for r in results if not r.get("error")]
    if valid:
        avg_cf1 = sum(r["cF1"] for r in valid) / len(valid)
        avg_ef1 = sum(r["exact_f1"] for r in valid) / len(valid)
        avg_rmse = sum(r["va_rmse"] for r in valid) / len(valid)
        print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")
        print(f"{'AVERAGE':<20} {avg_cf1:>8.4f} {avg_ef1:>10.4f} {avg_rmse:>9.4f}")
        print(f"\nSuccessful: {len(valid)}/{len(results)}")

        # Find best
        best = max(valid, key=lambda x: x["cF1"])
        print(f"Best: {best['dataset']} (cF1={best['cF1']:.4f})")

    # Save results
    results_path = "checkpoints/all_lang/results.json"
    Path("checkpoints/all_lang").mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_config": args.base_config,
            "results": results,
            "average": {
                "cF1": avg_cf1 if valid else 0,
                "exact_f1": avg_ef1 if valid else 0,
                "va_rmse": avg_rmse if valid else 0
            } if valid else {}
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
