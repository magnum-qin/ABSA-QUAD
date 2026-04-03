"""
Sweep vote thresholds (1-5) for each non-English language and report best.
Usage:
    python scripts/tune_vote_threshold.py
"""
import subprocess
import sys
import yaml
import copy
import json
from pathlib import Path

SEEDS = [42, 123, 456, 789, 1024]

LANGS = [
    {
        "name": "jpn_hotel",
        "dev_file": "data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "save_prefix": "v6_transfer_jpn_hotel_seed",
    },
    {
        "name": "rus_restaurant",
        "dev_file": "data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "save_prefix": "v6_transfer_rus_restaurant_seed",
    },
    {
        "name": "tat_restaurant",
        "dev_file": "data/subtask_3/tat/tat_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "save_prefix": "v6_transfer_tat_restaurant_seed",
    },
    {
        "name": "ukr_restaurant",
        "dev_file": "data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "save_prefix": "v6_transfer_ukr_restaurant_seed",
    },
    {
        "name": "zho_laptop",
        "dev_file": "data/subtask_3/zho/zho_laptop_dev_task3.jsonl",
        "encoder": "microsoft/mdeberta-v3-base",
        "save_prefix": "v6_transfer_zho_laptop_seed",
    },
]

BASE_CONFIG = "configs/config_rethink_v3.yaml"


def run_inference(lang, vote, config_path, pred_file):
    seed_dirs = [f"checkpoints/{lang['save_prefix']}{s}" for s in SEEDS]
    existing = [d for d in seed_dirs if Path(d).exists()]
    if len(existing) < 3:
        return None

    cmd = [
        sys.executable, "scripts/inference.py",
        "--multi_seed_ensemble",
        "--dual_layer",
        "--seed_dirs", *existing,
        "--vote_threshold", str(vote),
        "--config", config_path,
        "--test_file", lang["dev_file"],
        "--output", pred_file,
        "--batch_size", "32",
        "--pair_threshold", "0.3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_eval(pred_file, gold_file):
    cmd = [
        sys.executable, "scripts/official_eval.py",
        "-p", pred_file,
        "-g", gold_file,
        "-t", "3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    cf1 = 0.0
    for line in result.stdout.split("\n"):
        if "cF1:" in line:
            try:
                cf1 = float(line.split("cF1:")[1].strip())
            except (ValueError, IndexError):
                pass
    return cf1


def main():
    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    best_votes = {}
    all_results = {}

    for lang in LANGS:
        name = lang["name"]
        print(f"\n{'='*50}")
        print(f"Tuning vote threshold for: {name}")

        config = copy.deepcopy(base_config)
        config["model"]["encoder_name"] = lang["encoder"]
        tmp_config = f"/tmp/tune_vote_{name}.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        results = {}
        for vote in range(1, 6):
            pred_file = f"predictions/tune_{name}_vote{vote}.jsonl"
            ok = run_inference(lang, vote, tmp_config, pred_file)
            if not ok:
                print(f"  vote={vote}: inference failed")
                continue
            cf1 = run_eval(pred_file, lang["dev_file"])
            results[vote] = cf1
            print(f"  vote={vote}: cF1={cf1:.4f}")

        Path(tmp_config).unlink(missing_ok=True)

        if results:
            best_vote = max(results, key=results.get)
            best_votes[name] = best_vote
            all_results[name] = results
            print(f"  >>> Best: vote={best_vote} cF1={results[best_vote]:.4f}")

    print(f"\n{'='*50}")
    print("BEST VOTE THRESHOLDS:")
    for name, vote in best_votes.items():
        cf1 = all_results[name][vote]
        print(f"  {name}: vote={vote}  cF1={cf1:.4f}")

    out = {"best_votes": best_votes, "all_results": {k: {str(vv): vf for vv, vf in v.items()} for k, v in all_results.items()}}
    with open("predictions/vote_threshold_tuning.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to predictions/vote_threshold_tuning.json")


if __name__ == "__main__":
    main()
