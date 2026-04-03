"""
Full Pipeline: Augmentation + Multi-Seed Training + Ensemble Inference for ALL languages.

Runs sequentially on a single GPU:
  1. Back-translation augmentation per language
  2. 5-seed training per language
  3. Dual-layer ensemble inference + official eval

Usage:
    python scripts/pipeline_all_langs.py --config configs/config_rethink_v3.yaml
    python scripts/pipeline_all_langs.py --config configs/config_rethink_v3.yaml --skip_augmented eng_restaurant
"""

import os
import sys
import json
import yaml
import copy
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SEEDS = [42, 123, 456, 789, 1024]

# Language/domain configurations
LANG_CONFIGS = [
    {
        "name": "eng_restaurant",
        "train_file": "data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/eng/eng_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/deberta-v3-base",
        "source_lang": "en",
        "pivot_langs": ["de", "fr", "zh", "ru"],
        "lang_tags": {},
    },
    {
        "name": "eng_laptop",
        "train_file": "data/subtask_3/eng/eng_laptop_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/eng/eng_laptop_dev_task3.jsonl",
        "encoder": "microsoft/deberta-v3-base",
        "source_lang": "en",
        "pivot_langs": ["de", "fr", "zh", "ru"],
        "lang_tags": {},
    },
    {
        "name": "jpn_hotel",
        "train_file": "data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "ja",
        "pivot_langs": ["en"],
        "lang_tags": {},
    },
    {
        "name": "rus_restaurant",
        "train_file": "data/subtask_3/rus/rus_restaurant_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "ru",
        "pivot_langs": ["en", "uk"],
        "lang_tags": {},
    },
    {
        "name": "tat_restaurant",
        "train_file": "data/subtask_3/tat/tat_restaurant_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/tat/tat_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "tat",
        "pivot_langs": ["en"],
        "lang_tags": {"bwd": ">>tat<<"},
    },
    {
        "name": "ukr_restaurant",
        "train_file": "data/subtask_3/ukr/ukr_restaurant_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "uk",
        "pivot_langs": ["en", "ru"],
        "lang_tags": {},
    },
    {
        "name": "zho_laptop",
        "train_file": "data/subtask_3/zho/zho_laptop_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/zho/zho_laptop_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "zh",
        "pivot_langs": ["en"],
        "lang_tags": {},
    },
    {
        "name": "zho_restaurant",
        "train_file": "data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl",
        "dev_file": "data/subtask_3/zho/zho_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "source_lang": "zh",
        "pivot_langs": ["en"],
        "lang_tags": {},
    },
]


def run_cmd(cmd: list, desc: str = "", timeout: int = None) -> int:
    """Run a command with conda env, return exit code."""
    env = os.environ.copy()
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"\n>>> {desc}")
    print(f"    CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, timeout=timeout)
    return result.returncode


def step1_augment(lang_cfg: dict, force: bool = False) -> str:
    """Run back-translation augmentation. Returns augmented file path."""
    name = lang_cfg["name"]
    train_file = lang_cfg["train_file"]
    inp = Path(train_file)
    output = str(inp.parent / f"{inp.stem}_augmented{inp.suffix}")

    if Path(output).exists() and not force:
        with open(output) as f:
            n = sum(1 for _ in f)
        print(f"  [{name}] Augmented data already exists: {output} ({n} samples), skipping")
        return output

    cmd = [
        sys.executable, "scripts/augment_data.py",
        "--input", train_file,
        "--output", output,
        "--langs", *lang_cfg["pivot_langs"],
        "--source_lang", lang_cfg["source_lang"],
        "--batch_size", "32",
        "--fuzzy_threshold", "0.8",
    ]
    if lang_cfg["lang_tags"].get("bwd"):
        cmd.extend(["--bwd_lang_tag", lang_cfg["lang_tags"]["bwd"]])

    rc = run_cmd(cmd, f"Augmenting {name}")
    if rc != 0:
        print(f"  WARNING: Augmentation failed for {name}, using original data")
        return train_file

    return output


def step2_train_multi_seed(lang_cfg: dict, base_config: dict,
                            augmented_file: str,
                            pretrained_checkpoint: str = None,
                            save_prefix: str = None) -> str:
    """Train 5 seeds. Returns save_prefix."""
    name = lang_cfg["name"]
    if save_prefix is None:
        save_prefix = f"v5_{name}_seed"

    # Check if already trained
    all_done = True
    for seed in SEEDS:
        ckpt = Path(f"checkpoints/{save_prefix}{seed}/checkpoint_best.pt")
        if not ckpt.exists():
            all_done = False
            break
    if all_done:
        print(f"  [{name}] All 5 seeds already trained, skipping")
        return save_prefix

    # Build config for this language
    config = copy.deepcopy(base_config)
    config["model"]["encoder_name"] = lang_cfg["encoder"]
    config["data"]["train_file"] = augmented_file
    config["data"]["dev_file"] = lang_cfg["dev_file"]

    # Extract entity/attribute labels from training data
    entities = set()
    attributes = set()
    with open(lang_cfg["train_file"]) as f:
        for line in f:
            s = json.loads(line)
            for q in s.get("Quadruplet", []):
                cat = q["Category"]
                parts = cat.split("#")
                entities.add(parts[0])
                attributes.add(parts[1])
    entities.add("NULL")
    attributes.add("NULL")
    config["data"]["entity_labels"] = sorted(entities)
    config["data"]["attribute_labels"] = sorted(attributes)
    config["model"]["num_entity_labels"] = len(config["data"]["entity_labels"])
    config["model"]["num_attribute_labels"] = len(config["data"]["attribute_labels"])

    cmd = [
        sys.executable, "scripts/train_multi_seed.py",
        "--config", "configs/config_rethink_v3.yaml",
        "--seeds", *[str(s) for s in SEEDS],
        "--save_prefix", save_prefix,
        "--train_file", augmented_file,
    ]

    # Write a temp config with correct encoder and labels
    tmp_config = f"/tmp/pipeline_{name}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        sys.executable, "scripts/train_multi_seed.py",
        "--config", tmp_config,
        "--seeds", *[str(s) for s in SEEDS],
        "--save_prefix", save_prefix,
        "--train_file", augmented_file,
    ]
    if pretrained_checkpoint:
        cmd.extend(["--pretrained_checkpoint", pretrained_checkpoint])
        print(f"  [{name}] Transfer learning from: {pretrained_checkpoint}")

    rc = run_cmd(cmd, f"Training 5 seeds for {name}")
    if rc != 0:
        print(f"  WARNING: Training failed for {name}")

    Path(tmp_config).unlink(missing_ok=True)
    return save_prefix


def step3_ensemble_inference(lang_cfg: dict, save_prefix: str,
                              base_config: dict) -> str:
    """Run dual-layer ensemble inference. Returns prediction file path."""
    name = lang_cfg["name"]
    pred_file = f"predictions/v5_{name}.jsonl"

    seed_dirs = [f"checkpoints/{save_prefix}{s}" for s in SEEDS]

    # Check all seed dirs exist
    existing = [d for d in seed_dirs if Path(d).exists()]
    if len(existing) < 3:
        print(f"  WARNING: Only {len(existing)} seed dirs found for {name}, skipping ensemble")
        return ""

    # Build config
    config = copy.deepcopy(base_config)
    config["model"]["encoder_name"] = lang_cfg["encoder"]

    tmp_config = f"/tmp/pipeline_inf_{name}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Per-language vote threshold (tuned from ablation on dev set)
    VOTE_THRESHOLDS = {
        "jpn_hotel":      2,  # vote≥2 + MeCab=0.3323 > vote≥4+MeCab=0.3312
        "rus_restaurant": 5,  # vote=5: 0.4826 vs vote=3: 0.4542 (+2.84%)
        "tat_restaurant": 4,  # vote=4: 0.4630 vs vote=3: 0.4471 (+1.59%)
        "ukr_restaurant": 5,  # vote=5: 0.4706 vs vote=3: 0.4575 (+1.31%)
    }
    vote_threshold = VOTE_THRESHOLDS.get(name, (len(existing) // 2) + 1)

    cmd = [
        sys.executable, "scripts/inference.py",
        "--multi_seed_ensemble",
        "--dual_layer",
        "--seed_dirs", *existing,
        "--vote_threshold", str(vote_threshold),
        "--config", tmp_config,
        "--test_file", lang_cfg["dev_file"],
        "--output", pred_file,
        "--batch_size", "32",
        "--pair_threshold", "0.3",
    ]

    rc = run_cmd(cmd, f"Ensemble inference for {name}")
    Path(tmp_config).unlink(missing_ok=True)

    if rc != 0:
        print(f"  WARNING: Inference failed for {name}")
        return ""

    return pred_file


def step4_eval(lang_cfg: dict, pred_file: str) -> dict:
    """Run official evaluation. Returns metrics dict."""
    name = lang_cfg["name"]
    if not pred_file or not Path(pred_file).exists():
        return {"name": name, "cF1": 0, "error": True}

    cmd = [
        sys.executable, "scripts/official_eval.py",
        "-p", pred_file,
        "-g", lang_cfg["dev_file"],
        "-t", "3",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # Parse cF1 from output
    metrics = {"name": name, "error": False}
    for line in output.split("\n"):
        if "cF1:" in line:
            try:
                metrics["cF1"] = float(line.split("cF1:")[1].strip())
            except (ValueError, IndexError):
                pass
        if "cPrecision:" in line:
            try:
                metrics["cP"] = float(line.split("cPrecision:")[1].strip())
            except (ValueError, IndexError):
                pass
        if "cRecall:" in line:
            try:
                metrics["cR"] = float(line.split("cRecall:")[1].strip())
            except (ValueError, IndexError):
                pass

    if "cF1" not in metrics:
        metrics["cF1"] = 0
        metrics["error"] = True

    print(f"  [{name}] cF1={metrics.get('cF1', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline for All Languages")
    parser.add_argument("--config", type=str, default="configs/config_rethink_v3.yaml")
    parser.add_argument("--skip_augmented", nargs="*", default=[],
                        help="Languages to skip augmentation (already done)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these languages")
    parser.add_argument("--skip_augment", action="store_true",
                        help="Skip augmentation step entirely")
    parser.add_argument("--force_augment", action="store_true",
                        help="Force re-run augmentation even if output exists")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training step entirely")
    parser.add_argument("--transfer", action="store_true",
                        help="Use cross-lingual transfer from English checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # Cross-lingual transfer: map target language -> source checkpoint
    # All non-English use mdeberta-v3-base, so can transfer between them.
    # Use zho_restaurant (largest data, 6050 samples) as source for restaurant domain,
    # zho_laptop as source for laptop domain.
    TRANSFER_SOURCE = {
        "rus_restaurant": "checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt",
        "tat_restaurant": "checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt",
        "ukr_restaurant": "checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt",
        "jpn_hotel":      "checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt",
        "zho_laptop":     "checkpoints/v5_zho_restaurant_seed1024/checkpoint_best.pt",
    }

    langs = LANG_CONFIGS
    if args.only:
        langs = [l for l in langs if l["name"] in args.only]

    print(f"{'='*70}")
    print(f"FULL PIPELINE: {len(langs)} languages")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    all_results = []

    for i, lang_cfg in enumerate(langs):
        name = lang_cfg["name"]
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(langs)}] {name}")
        print(f"{'='*70}")

        # Step 1: Augmentation
        if args.skip_augment:
            aug_file = lang_cfg["train_file"]
            print(f"  Skipping augmentation, using original: {aug_file}")
        else:
            aug_file = step1_augment(lang_cfg, force=args.force_augment)
        print(f"  Training file: {aug_file}")

        # Step 2: Multi-seed training
        save_prefix = f"v5_{name}_seed"
        if args.transfer:
            save_prefix = f"v6_transfer_{name}_seed"

        if not args.skip_train:
            # Resolve pretrained checkpoint for transfer learning
            pretrained_ckpt = None
            if args.transfer and name in TRANSFER_SOURCE:
                ckpt_path = TRANSFER_SOURCE[name]
                if Path(ckpt_path).exists():
                    pretrained_ckpt = ckpt_path
                else:
                    print(f"  WARNING: Transfer source not found: {ckpt_path}, training from scratch")
            save_prefix = step2_train_multi_seed(lang_cfg, base_config, aug_file,
                                                  pretrained_checkpoint=pretrained_ckpt,
                                                  save_prefix=save_prefix)

        # Step 3: Ensemble inference
        pred_file = step3_ensemble_inference(lang_cfg, save_prefix, base_config)

        # Step 4: Evaluation
        metrics = step4_eval(lang_cfg, pred_file)
        all_results.append(metrics)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Language':<20} {'cPrecision':>11} {'cRecall':>9} {'cF1':>8}")
    print(f"{'-'*20} {'-'*11} {'-'*9} {'-'*8}")
    for r in all_results:
        if r.get("error"):
            print(f"{r['name']:<20} {'ERROR':>30}")
        else:
            print(f"{r['name']:<20} {r.get('cP', 0):>11.4f} {r.get('cR', 0):>9.4f} {r['cF1']:>8.4f}")

    valid = [r for r in all_results if not r.get("error")]
    if valid:
        avg = sum(r["cF1"] for r in valid) / len(valid)
        print(f"\nAverage cF1: {avg:.4f}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    results_path = "predictions/v5_all_lang_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
