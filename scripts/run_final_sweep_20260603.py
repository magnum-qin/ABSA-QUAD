#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/g24tka12/ABSA-QUAD")
SEEDS = [42, 123, 456, 789, 1024]
VOTES = [2, 3, 4, 5]
TOPKS = [0.0, 0.10, 0.15, 0.20]


SPECS = {
    "eng_laptop": {
        "cfg": "final_configs/eng_laptop_v5_bt.yaml",
        "dev": "data/subtask_3/eng/eng_laptop_dev_task3.jsonl",
        "prefix": "final_eng_laptop_v5_bt_seed",
        "dual": [True, False],
        "postprocess": [],
    },
    "jpn_hotel": {
        "cfg": "final_configs/jpn_hotel_v16_no_aug.yaml",
        "dev": "data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl",
        "prefix": "final_jpn_hotel_v16_no_aug_seed",
        "dual": [False],
        "postprocess": ["none", "mecab", "mecab_adverb"],
    },
    "rus_restaurant": {
        "cfg": "final_configs/rus_restaurant_v15.yaml",
        "dev": "data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
        "prefix": "final_rus_restaurant_v15_seed",
        "dual": [False],
        "postprocess": [],
        "cw_weights": [0.4817, 0.4469, 0.4785, 0.4646, 0.4780],
    },
    "ukr_restaurant": {
        "cfg": "final_configs/ukr_restaurant_v13b.yaml",
        "dev": "data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl",
        "prefix": "final_ukr_restaurant_v13b_seed",
        "dual": [False],
        "postprocess": [],
        "cw_weights": [0.4410, 0.4472, 0.4674, 0.4736, 0.4415],
    },
}


def py_path() -> str:
    for path in (
        "/home/g24tka12/miniforge3/envs/absa/bin/python",
        "/tmp/g24tka12_absa/miniforge3/envs/absa/bin/python",
    ):
        if Path(path).exists():
            return path
    return sys.executable


def run(cmd, log_path=None):
    env = os.environ.copy()
    py = py_path()
    conda_prefix = str(Path(py).parents[1])
    env["CONDA_PREFIX"] = conda_prefix
    env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
    if log_path:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True).stdout


def eval_cf1(pred, gold) -> float:
    out = run([py_path(), "scripts/official_eval.py", "-p", pred, "-g", gold, "-t", "3"])
    matches = re.findall(r"cF1:\s*([0-9.]+)", out)
    return float(matches[-1]) if matches else -1.0


def seed_dirs(prefix):
    dirs = [f"checkpoints/{prefix}{seed}" for seed in SEEDS]
    for d in dirs:
        if not (ROOT / d / "checkpoint_best.pt").exists():
            raise FileNotFoundError(f"missing checkpoint_best.pt in {d}")
    return dirs


def run_inference(task, cfg, dev, prefix, vote, topk, dual, pred_dir, log_dir):
    tag = f"{task}_vote{vote}_topk{topk:.2f}_dual{int(dual)}"
    out = pred_dir / f"{tag}.jsonl"
    log = log_dir / f"{tag}.log"
    cmd = [
        py_path(), "-u", "scripts/inference.py",
        "--multi_seed_ensemble",
        "--seed_dirs", *seed_dirs(prefix),
        "--config", cfg,
        "--test_file", dev,
        "--output", str(out),
        "--vote_threshold", str(vote),
        "--span_topk_threshold", str(topk),
        "--batch_size", "16",
    ]
    if dual:
        cmd.append("--dual_layer")
    rc = run(cmd, log)
    if rc != 0:
        return []
    rows = [{"task": task, "method": "uniform", "vote": vote, "topk": topk,
             "dual": dual, "postprocess": "none", "cf1": eval_cf1(str(out), dev),
             "pred": str(out)}]
    return rows


def run_jpn_postprocess(base_row, dev, pred_dir, log_dir):
    rows = []
    base = Path(base_row["pred"])
    for mode, extra in (("mecab", []), ("mecab_adverb", ["--expand_opinion_adverb"])):
        out = pred_dir / f"{base.stem}_{mode}.jsonl"
        log = log_dir / f"{base.stem}_{mode}.log"
        cmd = [py_path(), "scripts/postprocess_jpn.py", "--input", str(base),
               "--gold", dev, "--output", str(out), *extra]
        rc = run(cmd, log)
        if rc == 0:
            row = dict(base_row)
            row["postprocess"] = mode
            row["cf1"] = eval_cf1(str(out), dev)
            row["pred"] = str(out)
            rows.append(row)
    return rows


def run_confidence_weighted(task, cfg, dev, prefix, weights, vote, topk, pred_dir, log_dir):
    tag = f"{task}_cw_vote{vote}_topk{topk:.2f}"
    out = pred_dir / f"{tag}.jsonl"
    log = log_dir / f"{tag}.log"
    cmd = [
        py_path(), "-u", "scripts/confidence_weighted_ensemble.py",
        "--seed_dirs", *seed_dirs(prefix),
        "--seed_weights", *[str(w) for w in weights],
        "--config", cfg,
        "--test_file", dev,
        "--output", str(out),
        "--vote_threshold", str(vote),
        "--span_topk_threshold", str(topk),
        "--batch_size", "16",
    ]
    rc = run(cmd, log)
    if rc != 0:
        return None
    return {"task": task, "method": "confidence_weighted", "vote": vote,
            "topk": topk, "dual": False, "postprocess": "none",
            "cf1": eval_cf1(str(out), dev), "pred": str(out)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(SPECS))
    args = parser.parse_args()

    pred_dir = ROOT / "predictions" / "final_sweep_20260603"
    log_dir = ROOT / "logs" / "final_sweep_20260603"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = log_dir / ("results_" + "_".join(args.tasks) + ".jsonl")

    all_rows = []
    with open(result_path, "w", encoding="utf-8") as rf:
        for task in args.tasks:
            spec = SPECS[task]
            for dual, vote, topk in itertools.product(spec["dual"], VOTES, TOPKS):
                rows = run_inference(task, spec["cfg"], spec["dev"], spec["prefix"],
                                     vote, topk, dual, pred_dir, log_dir)
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rf.flush()
                    all_rows.append(row)
                    if task == "jpn_hotel" and row["postprocess"] == "none":
                        for pp_row in run_jpn_postprocess(row, spec["dev"], pred_dir, log_dir):
                            print(json.dumps(pp_row, ensure_ascii=False), flush=True)
                            rf.write(json.dumps(pp_row, ensure_ascii=False) + "\n")
                            rf.flush()
                            all_rows.append(pp_row)

            if "cw_weights" in spec:
                for vote, topk in itertools.product(VOTES, TOPKS):
                    row = run_confidence_weighted(task, spec["cfg"], spec["dev"], spec["prefix"],
                                                  spec["cw_weights"], vote, topk, pred_dir, log_dir)
                    if row:
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rf.flush()
                        all_rows.append(row)

    for task in args.tasks:
        rows = [r for r in all_rows if r["task"] == task]
        if rows:
            best = max(rows, key=lambda r: r["cf1"])
            print("BEST", task, json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
