#!/usr/bin/env python3
"""Preflight audit for reproducing recorded best ABSA-QUAD results.

This script is intentionally read-only. It checks whether the current project
has the historical data/checkpoint assets required to reproduce the best dev
cF1 values recorded in docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SEEDS = (42, 123, 456, 789, 1024)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    version: str
    checkpoint_prefix: str
    config: str
    train_file: str | None
    expected_train_lines: int | None
    dev_file: str
    test_file: str
    vote: int
    topk: float
    dual_layer: bool
    expected_dev_cf1: float


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "eng_restaurant",
        "V16",
        "v16_native_eng_restaurant_seed",
        "/tmp/pipeline_eng_restaurant_v16.yaml",
        "data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl",
        2284,
        "data/subtask_3/eng/eng_restaurant_dev_task3.jsonl",
        "data/subtask_3/eng/eng_restaurant_test_task3.jsonl",
        4,
        0.0,
        False,
        0.6949,
    ),
    TaskSpec(
        "eng_laptop",
        "V5/V6",
        "v5_eng_laptop_seed",
        "/tmp/pipeline_eng_laptop_v5.yaml",
        "data/subtask_3/eng/eng_laptop_train_alltasks_augmented.jsonl",
        15049,
        "data/subtask_3/eng/eng_laptop_dev_task3.jsonl",
        "data/subtask_3/eng/eng_laptop_test_task3.jsonl",
        3,
        0.0,
        True,
        0.3304,
    ),
    TaskSpec(
        "jpn_hotel",
        "V16 no_aug",
        "v16_ablation_jpn_no_aug_seed",
        "/tmp/pipeline_jpn_v16_no_aug.yaml",
        "data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl",
        1600,
        "data/subtask_3/jpn/jpn_hotel_dev_task3.jsonl",
        "data/subtask_3/jpn/jpn_hotel_test_task3.jsonl",
        4,
        0.0,
        False,
        0.4500,
    ),
    TaskSpec(
        "rus_restaurant",
        "V15",
        "v15_xlmr_rus_seed",
        "/tmp/pipeline_rus_v15.yaml",
        "data/subtask_3/rus/rus_restaurant_train_v13.jsonl",
        1766,
        "data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
        "data/subtask_3/rus/rus_restaurant_test_task3.jsonl",
        4,
        0.10,
        False,
        0.5267,
    ),
    TaskSpec(
        "tat_restaurant",
        "V15",
        "v15_xlmr_tat_seed",
        "/tmp/pipeline_tat_v15.yaml",
        "data/subtask_3/tat/tat_restaurant_train_v13.jsonl",
        1599,
        "data/subtask_3/tat/tat_restaurant_dev_task3.jsonl",
        "data/subtask_3/tat/tat_restaurant_test_task3.jsonl",
        3,
        0.20,
        False,
        0.4821,
    ),
    TaskSpec(
        "ukr_restaurant",
        "V13b",
        "v13b_ukr_seed",
        "/tmp/pipeline_ukr_v13b.yaml",
        "data/subtask_3/ukr/ukr_restaurant_train_v13b.jsonl",
        2967,
        "data/subtask_3/ukr/ukr_restaurant_dev_task3.jsonl",
        "data/subtask_3/ukr/ukr_restaurant_test_task3.jsonl",
        3,
        0.0,
        False,
        0.5068,
    ),
    TaskSpec(
        "zho_restaurant",
        "V16",
        "v16_native_zho_restaurant_seed",
        "/tmp/pipeline_zho_restaurant_v16.yaml",
        "data/subtask_3/zho/zho_restaurant_train_alltasks.jsonl",
        6050,
        "data/subtask_3/zho/zho_restaurant_dev_task3.jsonl",
        "data/subtask_3/zho/zho_restaurant_test_task3.jsonl",
        4,
        0.0,
        False,
        0.5669,
    ),
    TaskSpec(
        "zho_laptop",
        "V16",
        "v16_native_zho_laptop_seed",
        "/tmp/pipeline_zho_laptop_v16.yaml",
        "data/subtask_3/zho/zho_laptop_train_v13c.jsonl",
        5534,
        "data/subtask_3/zho/zho_laptop_dev_task3.jsonl",
        "data/subtask_3/zho/zho_laptop_test_task3.jsonl",
        4,
        0.10,
        False,
        0.4487,
    ),
)


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def audit_task(root: Path, spec: TaskSpec) -> dict:
    train_lines = count_lines(root / spec.train_file) if spec.train_file else None
    missing_ckpts = [
        seed
        for seed in SEEDS
        if not (root / "checkpoints" / f"{spec.checkpoint_prefix}{seed}" / "checkpoint_best.pt").exists()
    ]
    data_ok = spec.expected_train_lines is None or train_lines == spec.expected_train_lines
    ckpt_ok = not missing_ckpts
    return {
        "name": spec.name,
        "version": spec.version,
        "expected_dev_cf1": spec.expected_dev_cf1,
        "train_file": spec.train_file,
        "train_lines": train_lines,
        "expected_train_lines": spec.expected_train_lines,
        "data_ok": data_ok,
        "checkpoint_prefix": spec.checkpoint_prefix,
        "missing_checkpoint_seeds": missing_ckpts,
        "checkpoints_ok": ckpt_ok,
        "inference": {
            "vote_threshold": spec.vote,
            "span_topk_threshold": spec.topk,
            "dual_layer": spec.dual_layer,
            "config": spec.config,
        },
        "ready": data_ok and ckpt_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit historical best reproduction assets.")
    parser.add_argument("--root", default=".", help="Project root to audit")
    parser.add_argument("--tasks", nargs="*", help="Optional task names to audit")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    wanted = set(args.tasks or [])
    specs = [spec for spec in TASKS if not wanted or spec.name in wanted]
    rows = [audit_task(root, spec) for spec in specs]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        header = (
            f"{'task':<16} {'ver':<10} {'data':<14} {'ckpt':<16} "
            f"{'vote':<4} {'topk':<5} {'dev':<6} status"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            data = f"{row['train_lines']}/{row['expected_train_lines']}"
            ckpt = "ok" if row["checkpoints_ok"] else "missing:" + ",".join(map(str, row["missing_checkpoint_seeds"]))
            status = "READY" if row["ready"] else "NOT_READY"
            print(
                f"{row['name']:<16} {row['version']:<10} {data:<14} {ckpt:<16} "
                f"{row['inference']['vote_threshold']:<4} {row['inference']['span_topk_threshold']:<5} "
                f"{row['expected_dev_cf1']:<6.4f} {status}"
            )

    return 0 if all(row["ready"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
