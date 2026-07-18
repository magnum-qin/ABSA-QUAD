#!/usr/bin/env python
"""Strict Tatar DeepSeek-vs-Qwen matched-text teacher control.

The existing Qwen arm is frozen.  This workflow re-labels its exact 3,915
silver texts with DeepSeek using the same prompt recipe, trains only the five
missing DeepSeek seeds, and compares frozen vote=3 ensembles.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tallip_common import (
    ROOT, evaluate_records, normalize_text, paired_bootstrap, read_jsonl,
    root_path, sha256_file, write_jsonl,
)


SEEDS = [42, 123, 456, 789, 1024]
GOLD = root_path("data/subtask_3/tat/tat_restaurant_train_alltasks.jsonl")
QWEN_TRAIN = root_path("data/subtask_3/tat/tat_restaurant_train_v23_qwen.jsonl")
DEV = root_path("data/subtask_3/tat/tat_restaurant_dev_task3.jsonl")
TEST = root_path("data/subtask_3/tat/tat_restaurant_test_task3.jsonl")
CONFIG = root_path("configs/pipeline_tat_restaurant_v23_deepseek.yaml")
AUDIT_ROOT = root_path("data/audits/tallip_kd/tat_teacher_matched")
DOC_ROOT = root_path("docs/audits/tallip_kd")
INPUT = AUDIT_ROOT / "input_qwen_silver_texts.jsonl"
SILVER = AUDIT_ROOT / "deepseek_silver.jsonl"
RAW = AUDIT_ROOT / "deepseek_raw.jsonl"
METADATA = AUDIT_ROOT / "deepseek_metadata.json"
TRAIN = root_path("data/subtask_3/tat/tat_restaurant_train_v23_deepseek_teacher_matched.jsonl")
PREFIX = "v23_tat_restaurant_deepseek_teacher_matched_seed"
PRED_ROOT = AUDIT_ROOT / "predictions"
QWEN_PRED_ROOT = root_path("data/audits/tallip_kd/inductive_predictions")


def load_env(path: str = ".env.local") -> None:
    env = root_path(path)
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def prepare(_: argparse.Namespace) -> int:
    gold = read_jsonl(GOLD)
    qwen = read_jsonl(QWEN_TRAIN)
    if qwen[:len(gold)] != gold:
        raise SystemExit("Qwen train prefix is not the frozen 1,240-row gold base")
    silver = qwen[len(gold):]
    if len(silver) != 3915 or any(not row.get("Quadruplet") for row in silver):
        raise SystemExit(f"Unexpected frozen Qwen silver tail: {len(silver)} rows")
    probes = [{"ID": row["ID"], "Text": row["Text"]} for row in silver]
    if len({str(row["ID"]) for row in probes}) != len(probes):
        raise SystemExit("Duplicate IDs in frozen Qwen silver tail")
    write_jsonl(INPUT, probes)
    print(f"prepared={len(probes)} input={INPUT.relative_to(ROOT)} sha256={sha256_file(INPUT)}")
    return 0


def label(_: argparse.Namespace) -> int:
    load_env()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    command = [
        sys.executable, str(ROOT / "scripts/llm_silver_label.py"),
        "--input", str(INPUT), "--train_file", str(GOLD), "--output", str(SILVER),
        "--model", "deepseek-v4-pro", "--domain", "restaurant", "--workers", "12",
        "--n_shots", "6", "--raw_output", str(RAW), "--metadata_output", str(METADATA),
        "--prompt_version", "tat-teacher-matched-v1",
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


def build(_: argparse.Namespace) -> int:
    gold, probes, outputs = read_jsonl(GOLD), read_jsonl(INPUT), read_jsonl(SILVER)
    by_id = {str(row.get("ID")): row for row in outputs}
    if len(by_id) != len(outputs):
        raise SystemExit("Duplicate DeepSeek output IDs")
    missing = [str(row["ID"]) for row in probes if str(row["ID"]) not in by_id]
    errors = [str(row.get("ID")) for row in outputs if row.get("_err")]
    if missing or errors or len(outputs) != len(probes):
        raise SystemExit(f"Invalid DeepSeek run: missing={missing[:5]} errors={errors[:5]} rows={len(outputs)}")
    matched = []
    for probe in probes:
        output = by_id[str(probe["ID"])]
        if output.get("Text") != probe["Text"]:
            raise SystemExit(f"Text mismatch for {probe['ID']}")
        matched.append({"ID": probe["ID"], "Text": probe["Text"],
                        "Quadruplet": output.get("Quadruplet", [])})
    train = gold + matched
    write_jsonl(TRAIN, train)
    references = {normalize_text(row["Text"]) for path in (DEV, TEST) for row in read_jsonl(path)}
    overlap = sorted({normalize_text(row["Text"]) for row in train} & references)
    if overlap:
        raise SystemExit(f"Matched train leaks dev/test text: {len(overlap)} rows")
    qwen = read_jsonl(QWEN_TRAIN)
    if [row["Text"] for row in train] != [row["Text"] for row in qwen]:
        raise SystemExit("Matched train text/order differs from frozen Qwen train")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "same 1,240 gold + same 3,915 silver texts/order; teacher labels only",
        "rows": len(train), "gold_rows": len(gold), "silver_rows": len(matched),
        "empty_deepseek_silver_rows": sum(not row["Quadruplet"] for row in matched),
        "normalized_dev_test_overlap": len(overlap),
        "input_sha256": sha256_file(INPUT), "silver_sha256": sha256_file(SILVER),
        "train_sha256": sha256_file(TRAIN), "qwen_train_sha256": sha256_file(QWEN_TRAIN),
        "prompt_metadata": str(METADATA.relative_to(ROOT)),
    }
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    path = DOC_ROOT / "tat_teacher_matched_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def train(_: argparse.Namespace) -> int:
    command = [
        sys.executable, str(ROOT / "scripts/train_multi_seed.py"), "--config", str(CONFIG),
        "--seeds", *map(str, SEEDS), "--save_prefix", PREFIX, "--train_file", str(TRAIN),
        "--resume_existing",
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


def inference_command(seed_dirs: list[str], output: Path, vote: int) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts/inference.py"), "--multi_seed_ensemble",
        "--seed_dirs", *seed_dirs, "--config", str(CONFIG), "--test_file", str(DEV),
        "--output", str(output), "--vote_threshold", str(vote),
        "--span_topk_threshold", "0.0", "--batch_size", "16",
    ]


def infer(args: argparse.Namespace) -> int:
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    seed_dirs = [f"checkpoints/{PREFIX}{seed}" for seed in SEEDS]
    missing = [path for path in seed_dirs if not root_path(path).joinpath("checkpoint_best.pt").exists()]
    if missing:
        raise SystemExit(f"Missing best checkpoints: {missing}")
    for seed, seed_dir in zip(SEEDS, seed_dirs):
        output = PRED_ROOT / f"deepseek_seed{seed}.jsonl"
        if output.exists() and not args.force:
            continue
        command = inference_command([seed_dir], output, 1)
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    output = PRED_ROOT / "deepseek_ensemble.jsonl"
    if not output.exists() or args.force:
        command = inference_command(seed_dirs, output, 3)
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


def collect(_: argparse.Namespace) -> int:
    gold = read_jsonl(DEV)
    deepseek_scores, qwen_scores = [], []
    for seed in SEEDS:
        deepseek_scores.append(evaluate_records(gold, read_jsonl(PRED_ROOT / f"deepseek_seed{seed}.jsonl"))["cF1"])
        qwen_scores.append(evaluate_records(
            gold, read_jsonl(QWEN_PRED_ROOT / f"tat_restaurant_original_seed{seed}.jsonl")
        )["cF1"])
    deepseek_ensemble = read_jsonl(PRED_ROOT / "deepseek_ensemble.jsonl")
    qwen_ensemble = read_jsonl(QWEN_PRED_ROOT / "tat_restaurant_original_ensemble.jsonl")
    deepseek_metrics = evaluate_records(gold, deepseek_ensemble)
    qwen_metrics = evaluate_records(gold, qwen_ensemble)
    comparison = paired_bootstrap(gold, deepseek_ensemble, qwen_ensemble, 10000, 20260713)
    report = {
        "seeds": SEEDS, "vote": 3, "topk": 0.0,
        "deepseek": {"single_scores": deepseek_scores,
                     "single_mean": statistics.mean(deepseek_scores),
                     "single_sd": statistics.stdev(deepseek_scores),
                     "ensemble_cF1": deepseek_metrics["cF1"]},
        "qwen": {"single_scores": qwen_scores,
                 "single_mean": statistics.mean(qwen_scores),
                 "single_sd": statistics.stdev(qwen_scores),
                 "ensemble_cF1": qwen_metrics["cF1"]},
        "paired_seed_deltas_deepseek_minus_qwen": [a - b for a, b in zip(deepseek_scores, qwen_scores)],
        "ensemble_deepseek_minus_qwen": comparison,
    }
    output = DOC_ROOT / "tat_teacher_matched_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    low, high = comparison["ci95"]
    markdown = [
        "# Tatar matched-text teacher control", "",
        "Same 1,240 gold rows and same 3,915 silver texts/order; only teacher labels differ.", "",
        "| Teacher | Five-seed mean±SD | vote=3 ensemble cF1 |", "|---|---:|---:|",
        f"| DeepSeek | {report['deepseek']['single_mean']:.4f}±{report['deepseek']['single_sd']:.4f} | {deepseek_metrics['cF1']:.4f} |",
        f"| Qwen | {report['qwen']['single_mean']:.4f}±{report['qwen']['single_sd']:.4f} | {qwen_metrics['cF1']:.4f} |",
        "", f"DeepSeek−Qwen ensemble Δ={comparison['delta_a_minus_b']:+.4f}, "
        f"95% CI [{low:+.4f}, {high:+.4f}], {comparison['classification']}.", "",
        "Paired bootstrap: 10,000 resamples, seed=20260713.", "",
    ]
    table = DOC_ROOT / "tat_teacher_matched_table.md"
    table.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("prepare", prepare), ("label", label), ("build", build),
                       ("train", train), ("collect", collect)):
        command = sub.add_parser(name)
        command.set_defaults(func=func)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--force", action="store_true")
    infer_parser.set_defaults(func=infer)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
