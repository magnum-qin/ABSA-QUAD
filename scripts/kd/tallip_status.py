#!/usr/bin/env python
"""Compact status for long-running TALLIP KD training and labeling."""

from __future__ import annotations

import re
from pathlib import Path

from tallip_common import ROOT, load_manifest, read_jsonl, root_path
from teacher_fidelity import paths_for


def latest_epoch(log_path: Path) -> str:
    if not log_path.exists():
        return "not-started"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    seed_matches = re.findall(r"(?:Training with seed=|_seed)(\d+)", text)
    epoch_matches = re.findall(r"Epoch (\d+)/(\d+)", text)
    seed = seed_matches[-1] if seed_matches else "?"
    epoch = f"{epoch_matches[-1][0]}/{epoch_matches[-1][1]}" if epoch_matches else "loading"
    return f"seed={seed}, epoch={epoch}"


def main() -> int:
    manifest = load_manifest()
    seeds = manifest["defaults"]["seeds"]
    expected_jobs = complete_jobs = best_files = 0
    print("Training")
    for dataset, spec in manifest["datasets"].items():
        jobs = []
        if spec.get("clean_rerun"):
            jobs.append(("kd", spec["clean_prefix"]))
        if spec.get("clean_baseline_rerun"):
            jobs.append(("baseline", spec["baseline_prefix"]))
        for system, prefix in jobs:
            expected_jobs += len(seeds)
            complete = sum(root_path(f"checkpoints/{prefix}{seed}/training_complete.json").exists() for seed in seeds)
            best = sum(root_path(f"checkpoints/{prefix}{seed}/checkpoint_best.pt").exists() for seed in seeds)
            complete_jobs += complete
            best_files += best
            log = Path(f"/tmp/tallip_{dataset}_{system}_train.log")
            progress = "done" if complete == len(seeds) else latest_epoch(log)
            print(f"  {dataset:<20} {system:<8} complete={complete}/5 best={best}/5 {progress}")
    print(f"  total complete={complete_jobs}/{expected_jobs}, best={best_files}/{expected_jobs}")

    print("Teacher fidelity")
    for teacher in ("deepseek", "qwen"):
        complete_rows = expected_rows = errors = 0
        for dataset, spec in manifest["datasets"].items():
            paths = paths_for(manifest, dataset, teacher)
            expected = len(read_jsonl(paths["input"])) if paths["input"].exists() else 0
            rows = read_jsonl(paths["output"]) if paths["output"].exists() else []
            expected_rows += expected
            complete_rows += len(rows)
            errors += sum(bool(row.get("_err")) for row in rows)
        print(f"  {teacher:<8} rows={complete_rows}/{expected_rows} errors={errors}")

    prediction_root = root_path(manifest["defaults"]["output_root"]) / "inductive_predictions"
    prediction_files = list(prediction_root.glob("*.jsonl")) if prediction_root.exists() else []
    print(f"Frozen inference\n  prediction files={len(prediction_files)}/60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
