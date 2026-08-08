#!/usr/bin/env python
"""Build, train, infer, audit, and report post-lock dev-only KD controls.

The post-lock protocol is deliberately isolated from frozen official-test
evidence. It covers exposure-matched repeated-gold controls and same-text
student self-training controls under fixed pre-test thresholds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tallip_common import ROOT, evaluate_records, normalize_text, paired_bootstrap, read_jsonl


DEFAULT_MANIFEST = ROOT / "scripts/kd/postlock_v1.yaml"


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with root_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evict_file_cache(path: str | Path) -> None:
    """Best-effort eviction of a large, clean checkpoint from the page cache."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    descriptor = os.open(root_path(path), os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    manifest = yaml.safe_load(root_path(path).read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    if protocol["official_test_evaluation"] != "forbidden":
        raise SystemExit("post-lock protocol must forbid official-test evaluation")
    if protocol["model_selection_split"] != "dev_only":
        raise SystemExit("post-lock protocol must be dev-only")
    return manifest


def select(specs: dict, names: list[str]) -> list[tuple[str, dict]]:
    if names == ["all"]:
        return list(specs.items())
    unknown = set(names) - set(specs)
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    return [(name, specs[name]) for name in names]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repeat_gold(rows: list[dict], target_rows: int, seed: int) -> list[dict]:
    """Repeat all rows as evenly as possible to exactly match target exposure."""
    if not rows:
        raise ValueError("cannot repeat an empty gold dataset")
    if target_rows < len(rows):
        raise ValueError(f"target {target_rows} is smaller than source {len(rows)}")

    full_copies, remainder = divmod(target_rows, len(rows))
    output: list[dict] = []
    copy_number = 0
    for _ in range(full_copies):
        for row in rows:
            item = copy.deepcopy(row)
            if copy_number:
                item["ID"] = f"{row['ID']}__postlock_repeat{copy_number}"
            output.append(item)
        copy_number += 1

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    for index in indices[:remainder]:
        row = rows[index]
        item = copy.deepcopy(row)
        item["ID"] = f"{row['ID']}__postlock_repeat{copy_number}"
        output.append(item)
    return output


def overlap_count(train: list[dict], references: list[dict]) -> int:
    reference_texts = {normalize_text(row["Text"]) for row in references}
    return sum(normalize_text(row["Text"]) in reference_texts for row in train)


def build_gold_repeat(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seed = manifest["protocol"]["build_seed"]
    output_root = root_path(manifest["protocol"]["output_root"])
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest["protocol"]["name"],
        "build_seed": seed,
        "datasets": {},
    }
    for dataset, spec in select(manifest["gold_repeat"], args.datasets):
        source = read_jsonl(spec["source_train"])
        target_rows = sum(1 for _ in root_path(spec["target_train"]).open(encoding="utf-8"))
        repeated = repeat_gold(source, target_rows, seed)
        dev = read_jsonl(spec["dev"])
        test = read_jsonl(spec["test_text_audit"])
        dev_overlap = overlap_count(repeated, dev)
        test_overlap = overlap_count(repeated, test)
        if dev_overlap or test_overlap:
            raise SystemExit(
                f"{dataset}: repeated gold overlaps references: dev={dev_overlap}, test={test_overlap}"
            )
        ids = [str(row["ID"]) for row in repeated]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{dataset}: repeated-gold IDs are not unique")
        output = root_path(spec["output_train"])
        write_jsonl(output, repeated)
        record = {
            "source_train": spec["source_train"],
            "source_rows": len(source),
            "source_sha256": sha256_file(spec["source_train"]),
            "target_reference": spec["target_train"],
            "target_rows": target_rows,
            "output_train": spec["output_train"],
            "output_rows": len(repeated),
            "output_sha256": sha256_file(output),
            "exposure_ratio": target_rows / len(source),
            "dev_normalized_overlap_rows": dev_overlap,
            "test_normalized_overlap_rows": test_overlap,
        }
        audit["datasets"][dataset] = record
        print(
            f"{dataset:<20} {len(source):>6} -> {len(repeated):>6} "
            f"({record['exposure_ratio']:.2f}x), overlap=0, sha256={record['output_sha256'][:12]}"
        )
    audit_path = output_root / "gold_repeat_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"audit: {audit_path.relative_to(ROOT)}")
    return 0


def build_self_inputs(args: argparse.Namespace) -> int:
    """Extract exactly the silver-text portion used by each adopted LLM arm."""
    manifest = load_manifest(args.manifest)
    output_root = root_path(manifest["protocol"]["output_root"])
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest["protocol"]["name"],
        "policy": "same texts/order as the LLM silver tail; labels removed",
        "datasets": {},
    }
    for dataset, spec in select(manifest["self_training"], args.datasets):
        base = read_jsonl(spec["base_train"])
        llm_train = read_jsonl(spec["llm_reference_train"])
        if len(llm_train) <= len(base):
            raise SystemExit(f"{dataset}: LLM reference has no silver tail")
        base_by_id = {str(row["ID"]): row for row in base}
        if len(base_by_id) != len(base):
            raise SystemExit(f"{dataset}: base train IDs are not unique")
        seen_base = set()
        silver = []
        for row in llm_train:
            row_id = str(row["ID"])
            if row_id in base_by_id:
                if row["Text"] != base_by_id[row_id]["Text"]:
                    raise SystemExit(f"{dataset}: base ID={row_id} has mismatched text in LLM reference")
                seen_base.add(row_id)
            else:
                silver.append(row)
        missing_base = set(base_by_id) - seen_base
        if missing_base:
            raise SystemExit(f"{dataset}: LLM reference is missing {len(missing_base)} base IDs")
        if len(silver) != len(llm_train) - len(base):
            raise SystemExit(f"{dataset}: could not separate base and silver rows exactly")
        if any(not row.get("Quadruplet") for row in silver):
            raise SystemExit(f"{dataset}: LLM silver tail contains empty labels")
        inputs = [{"ID": row["ID"], "Text": row["Text"]} for row in silver]
        if len({str(row["ID"]) for row in inputs}) != len(inputs):
            raise SystemExit(f"{dataset}: self-training input IDs are not unique")
        output = root_path(spec["input"])
        write_jsonl(output, inputs)
        record = {
            "base_train": spec["base_train"],
            "base_rows": len(base),
            "llm_reference_train": spec["llm_reference_train"],
            "llm_reference_rows": len(llm_train),
            "self_training_input": spec["input"],
            "input_rows": len(inputs),
            "input_sha256": sha256_file(output),
        }
        audit["datasets"][dataset] = record
        print(
            f"{dataset:<20} base={len(base):>6}, same-text pseudo inputs={len(inputs):>6}, "
            f"target={len(llm_train):>6}, sha256={record['input_sha256'][:12]}"
        )
    audit_path = output_root / "self_training_input_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"audit: {audit_path.relative_to(ROOT)}")
    return 0


def cuda_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
        cwd=ROOT,
        check=False,
    )
    return probe.returncode == 0


def train_gold_repeat(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if not args.dry_run and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to start training")
    output_root = root_path(manifest["protocol"]["output_root"])
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    seeds = manifest["protocol"]["seeds"]
    for dataset, spec in select(manifest["gold_repeat"], args.datasets):
        train_file = root_path(spec["output_train"])
        if not train_file.exists():
            raise SystemExit(f"missing repeated-gold data for {dataset}: run build-gold-repeat first")
        command = [
            sys.executable,
            "scripts/train_multi_seed.py",
            "--config",
            spec["config"],
            "--seeds",
            *map(str, seeds),
            "--save_prefix",
            spec["checkpoint_prefix"],
            "--train_file",
            spec["output_train"],
            "--resume_existing",
        ]
        print("$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        log_path = log_root / f"{dataset}_gold_repeat_train.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# started {datetime.now(timezone.utc).isoformat()}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise SystemExit(f"{dataset}: training failed with code {result.returncode}; see {log_path}")
        print(f"completed {dataset}; log={log_path.relative_to(ROOT)}", flush=True)
    return 0


def pseudo_label(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["protocol"]["seeds"]
    output_root = root_path(manifest["protocol"]["output_root"])
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to start pseudo-label inference")
    for dataset, spec in select(manifest["self_training"], args.datasets):
        input_path = root_path(spec["input"])
        if not input_path.exists():
            raise SystemExit(f"{dataset}: missing self-training input; run build-self-inputs")
        seed_dirs = [f"checkpoints/{spec['gold_teacher_prefix']}{seed}" for seed in seeds]
        missing = [
            directory
            for directory in seed_dirs
            if not (ROOT / directory / "checkpoint_best.pt").exists()
        ]
        if missing:
            raise SystemExit(f"{dataset}: missing gold-teacher best checkpoints: {missing}")
        output = root_path(spec["pseudo"])
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/inference.py",
            "--multi_seed_ensemble",
            "--seed_dirs",
            *seed_dirs,
            "--config",
            spec["config"],
            "--test_file",
            spec["input"],
            "--output",
            spec["pseudo"],
            "--vote_threshold",
            str(spec["pseudo_vote"]),
            "--span_topk_threshold",
            str(spec["pseudo_topk"]),
            "--batch_size",
            str(args.batch_size),
        ]
        print("$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        log_path = log_root / f"{dataset}_self_training_pseudo.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise SystemExit(f"{dataset}: pseudo labeling failed; see {log_path}")
        print(f"completed pseudo labeling for {dataset}: {output.relative_to(ROOT)}", flush=True)
    return 0


def build_self_train(args: argparse.Namespace) -> int:
    """Combine gold with same-text student pseudo labels, retaining empty predictions."""
    manifest = load_manifest(args.manifest)
    output_root = root_path(manifest["protocol"]["output_root"])
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest["protocol"]["name"],
        "empty_policy": "retain empty student predictions as valid no-quad pseudo labels",
        "datasets": {},
    }
    for dataset, spec in select(manifest["self_training"], args.datasets):
        base = read_jsonl(spec["base_train"])
        inputs = read_jsonl(spec["input"])
        pseudo = read_jsonl(spec["pseudo"])
        if len(inputs) != len(pseudo):
            raise SystemExit(f"{dataset}: pseudo coverage mismatch {len(pseudo)} != {len(inputs)}")
        by_id = {str(row["ID"]): row for row in pseudo}
        if len(by_id) != len(pseudo):
            raise SystemExit(f"{dataset}: duplicate pseudo IDs")
        pseudo_rows = []
        for row in inputs:
            predicted = by_id.get(str(row["ID"]))
            if predicted is None:
                raise SystemExit(f"{dataset}: missing pseudo output for ID={row['ID']}")
            pseudo_rows.append(
                {
                    "ID": row["ID"],
                    "Text": row["Text"],
                    "Quadruplet": predicted.get("Quadruplet", []),
                }
            )
        train = base + pseudo_rows
        expected_rows = sum(
            1 for _ in root_path(spec["llm_reference_train"]).open(encoding="utf-8")
        )
        if len(train) != expected_rows:
            raise SystemExit(f"{dataset}: self-training rows {len(train)} != target {expected_rows}")
        output = root_path(spec["output_train"])
        write_jsonl(output, train)
        nonempty = sum(bool(row["Quadruplet"]) for row in pseudo_rows)
        record = {
            "base_rows": len(base),
            "pseudo_rows": len(pseudo_rows),
            "pseudo_nonempty_rows": nonempty,
            "pseudo_empty_rows": len(pseudo_rows) - nonempty,
            "pseudo_nonempty_rate": nonempty / len(pseudo_rows) if pseudo_rows else 0.0,
            "output_train": spec["output_train"],
            "output_rows": len(train),
            "output_sha256": sha256_file(output),
        }
        audit["datasets"][dataset] = record
        print(
            f"{dataset:<20} base={len(base):>6} + pseudo={len(pseudo_rows):>6} "
            f"(nonempty={record['pseudo_nonempty_rate']:.1%}) -> {len(train):>6}"
        )
    audit_path = output_root / "self_training_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"audit: {audit_path.relative_to(ROOT)}")
    return 0


def train_self(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if not args.dry_run and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to start training")
    output_root = root_path(manifest["protocol"]["output_root"])
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    seeds = manifest["protocol"]["seeds"]
    for dataset, spec in select(manifest["self_training"], args.datasets):
        train_file = root_path(spec["output_train"])
        if not train_file.exists():
            raise SystemExit(f"missing self-training data for {dataset}: run build-self-train first")
        command = [
            sys.executable,
            "scripts/train_multi_seed.py",
            "--config",
            spec["config"],
            "--seeds",
            *map(str, seeds),
            "--save_prefix",
            spec["checkpoint_prefix"],
            "--train_file",
            spec["output_train"],
            "--resume_existing",
        ]
        print("$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        log_path = log_root / f"{dataset}_self_training_train.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# started {datetime.now(timezone.utc).isoformat()}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise SystemExit(f"{dataset}: training failed with code {result.returncode}; see {log_path}")
        print(f"completed self-training {dataset}; log={log_path.relative_to(ROOT)}", flush=True)
    return 0


def prediction_path(output_root: Path, dataset: str, system: str, seed: int | str) -> Path:
    suffix = "ensemble" if seed == "ensemble" else f"seed{seed}"
    return output_root / "predictions" / f"{dataset}_{system}_{suffix}.jsonl"


def merge_seed_predictions(prediction_sets: list[list[dict]], vote_threshold: int) -> list[dict]:
    """Merge already-materialized seed predictions using inference.py voting semantics.

    Loading each multi-GB checkpoint in one long-lived Python process can retain
    enough allocator/page-cache memory for systemd-oomd to terminate inference.
    The per-seed predictions are already required by this protocol, so merging
    those JSONL files is equivalent and keeps the ensemble step memory-bounded.
    """
    if not prediction_sets:
        raise ValueError("cannot merge an empty prediction set")
    row_count = len(prediction_sets[0])
    if any(len(rows) != row_count for rows in prediction_sets):
        raise ValueError("seed prediction row counts differ")

    merged = []
    for index in range(row_count):
        sample_id = prediction_sets[0][index]["ID"]
        if any(rows[index]["ID"] != sample_id for rows in prediction_sets[1:]):
            raise ValueError(f"seed prediction ID/order mismatch at row {index}")
        quad_votes: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
        for rows in prediction_sets:
            for quad in rows[index].get("Quadruplet", []):
                key = (quad["Aspect"], quad["Category"], quad["Opinion"])
                valence, arousal = quad["VA"].split("#")
                quad_votes[key].append((float(valence), float(arousal)))

        final_quads = []
        for (aspect, category, opinion), values in quad_votes.items():
            if len(values) < vote_threshold:
                continue
            avg_v = max(1.0, min(9.0, sum(v for v, _ in values) / len(values)))
            avg_a = max(1.0, min(9.0, sum(a for _, a in values) / len(values)))
            final_quads.append(
                {
                    "Aspect": aspect,
                    "Category": category,
                    "Opinion": opinion,
                    "VA": f"{avg_v:.2f}#{avg_a:.2f}",
                }
            )
        merged.append({"ID": sample_id, "Quadruplet": final_quads})
    return merged


def inference_specs(manifest: dict, dataset: str, systems: list[str]) -> list[tuple[str, dict]]:
    resolved = []
    if "gold_repeat" in systems and dataset in manifest["gold_repeat"]:
        spec = manifest["gold_repeat"][dataset]
        resolved.append(
            (
                "gold_repeat",
                {
                    "checkpoint_prefix": spec["checkpoint_prefix"],
                    "config": spec["config"],
                    "dev": spec["dev"],
                    "vote": spec["vote"],
                    "topk": spec["topk"],
                },
            )
        )
    if "self_training" in systems and dataset in manifest["self_training"]:
        spec = manifest["self_training"][dataset]
        resolved.append(
            (
                "self_training",
                {
                    "checkpoint_prefix": spec["checkpoint_prefix"],
                    "config": spec["config"],
                    "dev": spec["dev"],
                    "vote": spec["eval_vote"],
                    "topk": spec["eval_topk"],
                },
            )
        )
    return resolved


def infer(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["protocol"]["seeds"]
    output_root = root_path(manifest["protocol"]["output_root"])
    log_root = output_root / "logs"
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    dataset_names = (
        list(manifest["gold_repeat"])
        if args.datasets == ["all"]
        else args.datasets
    )
    known = set(manifest["gold_repeat"]) | set(manifest["self_training"])
    unknown = set(dataset_names) - known
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    if not args.dry_run and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to start inference")
    for dataset in dataset_names:
        for system, spec in inference_specs(manifest, dataset, args.systems):
            seed_dirs = [f"checkpoints/{spec['checkpoint_prefix']}{seed}" for seed in seeds]
            missing = [
                directory
                for directory in seed_dirs
                if not (ROOT / directory / "checkpoint_best.pt").exists()
            ]
            if missing:
                raise SystemExit(f"{dataset}/{system}: missing best checkpoints: {missing}")
            for seed, directory in zip(seeds, seed_dirs):
                output = prediction_path(output_root, dataset, system, seed)
                command = [
                    sys.executable,
                    "scripts/inference.py",
                    "--multi_seed_ensemble",
                    "--seed_dirs",
                    directory,
                    "--config",
                    spec["config"],
                    "--test_file",
                    spec["dev"],
                    "--output",
                    str(output.relative_to(ROOT)),
                    "--vote_threshold",
                    "1",
                    "--span_topk_threshold",
                    str(spec["topk"]),
                    "--batch_size",
                    str(args.batch_size),
                ]
                print("$ " + " ".join(command), flush=True)
                if not args.dry_run and (args.force or not output.exists()):
                    log_path = log_root / f"{dataset}_{system}_seed{seed}_infer.log"
                    try:
                        with log_path.open("a", encoding="utf-8") as log:
                            subprocess.run(
                                command,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    finally:
                        evict_file_cache(ROOT / directory / "checkpoint_best.pt")
            output = prediction_path(output_root, dataset, system, "ensemble")
            seed_outputs = [prediction_path(output_root, dataset, system, seed) for seed in seeds]
            print(
                f"merge {len(seed_outputs)} seed predictions -> {output.relative_to(ROOT)} "
                f"(vote_threshold={spec['vote']})",
                flush=True,
            )
            if not args.dry_run and (args.force or not output.exists()):
                missing_outputs = [str(path.relative_to(ROOT)) for path in seed_outputs if not path.exists()]
                if missing_outputs:
                    raise SystemExit(
                        f"{dataset}/{system}: missing per-seed predictions before merge: {missing_outputs}"
                    )
                merged = merge_seed_predictions(
                    [read_jsonl(path) for path in seed_outputs],
                    vote_threshold=spec["vote"],
                )
                write_jsonl(output, merged)
    return 0


def load_prediction(path: Path) -> list[dict] | None:
    return read_jsonl(path) if path.exists() else None


def compact_metrics(metrics: dict) -> dict:
    """Drop per-example bootstrap internals from the durable public report."""
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checked_bootstrap(
    check_id: str,
    gold: list[dict],
    pred_a: list[dict],
    pred_b: list[dict],
    protocol: dict,
    checks: list[dict],
) -> dict:
    """Run a paired bootstrap twice and require byte-equivalent statistics."""
    kwargs = {
        "iterations": protocol["bootstrap_iterations"],
        "seed": protocol["bootstrap_seed"],
    }
    first = paired_bootstrap(gold, pred_a, pred_b, **kwargs)
    second = paired_bootstrap(gold, pred_a, pred_b, **kwargs)
    if first != second:
        raise RuntimeError(f"non-reproducible paired bootstrap: {check_id}")
    checks.append(
        {
            "id": check_id,
            "exact_repeat": True,
            "result_sha256": sha256_json(first),
        }
    )
    return first


def validate_prediction_rows(path: Path, gold: list[dict]) -> dict:
    """Validate one prediction file at the dev-example grain."""
    rows = read_jsonl(path)
    gold_ids = [str(row["ID"]) for row in gold]
    prediction_ids = [str(row.get("ID")) for row in rows]
    if prediction_ids != gold_ids:
        raise ValueError(f"{path}: prediction IDs/order do not match dev")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(f"{path}: duplicate prediction IDs")
    invalid_rows = 0
    invalid_quads = 0
    for row in rows:
        if set(row) - {"ID", "Quadruplet"} or not isinstance(row.get("Quadruplet"), list):
            invalid_rows += 1
            continue
        for quad in row["Quadruplet"]:
            if set(quad) != {"Aspect", "Category", "Opinion", "VA"}:
                invalid_quads += 1
                continue
            try:
                valence, arousal = (float(value) for value in str(quad["VA"]).split("#"))
            except (TypeError, ValueError):
                invalid_quads += 1
                continue
            if not (1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0):
                invalid_quads += 1
    if invalid_rows or invalid_quads:
        raise ValueError(
            f"{path}: invalid rows={invalid_rows}, invalid quadruplets={invalid_quads}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_ids": len(set(prediction_ids)),
        "id_order_matches_dev": True,
        "invalid_rows": 0,
        "invalid_quadruplets": 0,
    }


def build_result_audit(manifest: dict, report: dict, bootstrap_checks: list[dict]) -> dict:
    """Audit all post-lock predictions, thresholds, and checkpoint lineage."""
    protocol = manifest["protocol"]
    output_root = root_path(protocol["output_root"])
    seeds = protocol["seeds"]
    expected_predictions = set()
    checkpoint_records = []
    prediction_records = []
    ensemble_records = []
    threshold_records = []

    for dataset, gold_spec in manifest["gold_repeat"].items():
        gold = read_jsonl(gold_spec["dev"])
        for system, spec in inference_specs(
            manifest, dataset, ["gold_repeat", "self_training"]
        ):
            seed_rows = []
            for seed in seeds:
                directory = ROOT / "checkpoints" / f"{spec['checkpoint_prefix']}{seed}"
                best = directory / "checkpoint_best.pt"
                marker = directory / "training_complete.json"
                if not best.exists() or not marker.exists():
                    raise FileNotFoundError(f"missing completed best checkpoint: {directory}")
                completion = json.loads(marker.read_text(encoding="utf-8"))
                if int(completion.get("seed", -1)) != seed:
                    raise ValueError(f"{marker}: completion seed mismatch")
                best_stat = best.stat()
                checkpoint_records.append(
                    {
                        "dataset": dataset,
                        "system": system,
                        "seed": seed,
                        "checkpoint": str(best.relative_to(ROOT)),
                        "checkpoint_bytes": best_stat.st_size,
                        "checkpoint_mtime_ns": best_stat.st_mtime_ns,
                        "completion_marker": str(marker.relative_to(ROOT)),
                        "completion_marker_sha256": sha256_file(marker),
                    }
                )
                evict_file_cache(best)

                prediction = prediction_path(output_root, dataset, system, seed)
                expected_predictions.add(prediction.resolve())
                record = validate_prediction_rows(prediction, gold)
                record.update({"dataset": dataset, "system": system, "seed": seed})
                prediction_records.append(record)
                seed_rows.append(read_jsonl(prediction))

            ensemble = prediction_path(output_root, dataset, system, "ensemble")
            expected_predictions.add(ensemble.resolve())
            ensemble_record = validate_prediction_rows(ensemble, gold)
            actual_ensemble = read_jsonl(ensemble)
            rebuilt = merge_seed_predictions(seed_rows, vote_threshold=spec["vote"])
            if rebuilt != actual_ensemble:
                raise ValueError(f"{dataset}/{system}: ensemble does not match fixed-vote merge")
            ensemble_record.update(
                {
                    "dataset": dataset,
                    "system": system,
                    "seed": "ensemble",
                    "rebuilt_from_seed_predictions": True,
                }
            )
            prediction_records.append(ensemble_record)
            ensemble_records.append(
                {
                    "dataset": dataset,
                    "system": system,
                    "vote": spec["vote"],
                    "topk": spec["topk"],
                    "seed_prediction_count": len(seed_rows),
                    "exact_rebuild": True,
                    "output": str(ensemble.relative_to(ROOT)),
                }
            )
            threshold_records.append(
                {
                    "dataset": dataset,
                    "system": system,
                    "single_seed_vote": 1,
                    "ensemble_vote": spec["vote"],
                    "topk": spec["topk"],
                    "source": "scripts/kd/postlock_v1.yaml",
                }
            )

    actual_predictions = {
        path.resolve() for path in (output_root / "predictions").glob("*.jsonl")
    }
    if actual_predictions != expected_predictions:
        missing = sorted(str(path) for path in expected_predictions - actual_predictions)
        extra = sorted(str(path) for path in actual_predictions - expected_predictions)
        raise ValueError(f"prediction inventory mismatch: missing={missing}, extra={extra}")
    if len(checkpoint_records) != 40 or len(prediction_records) != 48:
        raise ValueError(
            f"unexpected inventory: checkpoints={len(checkpoint_records)}, "
            f"predictions={len(prediction_records)}"
        )

    checkpoint_dirs = {
        (ROOT / record["checkpoint"]).parent for record in checkpoint_records
    }
    non_best = sorted(
        str(path.relative_to(ROOT))
        for directory in checkpoint_dirs
        for path in directory.glob("checkpoint_*.pt")
        if path.name != "checkpoint_best.pt"
    )
    comparisons = sum(
        len(item["systems"]) + len(item.get("control_comparisons", {}))
        for item in report["datasets"].values()
    )
    if comparisons != len(bootstrap_checks):
        raise ValueError(
            f"bootstrap audit mismatch: report={comparisons}, checks={len(bootstrap_checks)}"
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["name"],
        "manifest": {
            "path": str(DEFAULT_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(DEFAULT_MANIFEST),
        },
        "scope": {
            "split": protocol["model_selection_split"],
            "official_test_evaluation": protocol["official_test_evaluation"],
            "checkpoint_policy": protocol["checkpoint_policy"],
            "threshold_policy": protocol["threshold_policy"],
        },
        "summary": {
            "prediction_files_expected": 48,
            "prediction_files_validated": len(prediction_records),
            "best_checkpoints_expected": 40,
            "best_checkpoints_validated": len(checkpoint_records),
            "ensemble_outputs_rebuilt": len(ensemble_records),
            "bootstrap_comparisons_repeated": len(bootstrap_checks),
            "all_prediction_ids_and_order_match_dev": True,
            "all_ensemble_rebuilds_exact": True,
            "all_bootstrap_repeats_exact": all(
                item["exact_repeat"] for item in bootstrap_checks
            ),
            "all_checkpoint_sources_are_best": all(
                record["checkpoint"].endswith("/checkpoint_best.pt")
                for record in checkpoint_records
            ),
            "retention_policy_compliant": not non_best,
        },
        "thresholds": threshold_records,
        "bootstrap": {
            "iterations": protocol["bootstrap_iterations"],
            "seed": protocol["bootstrap_seed"],
            "checks": bootstrap_checks,
        },
        "ensembles": ensemble_records,
        "predictions": prediction_records,
        "checkpoints": checkpoint_records,
        "non_best_checkpoint_files": non_best,
    }


def collect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    protocol = manifest["protocol"]
    seeds = protocol["seeds"]
    output_root = root_path(protocol["output_root"])
    publish_root = root_path(protocol["publish_root"])
    inductive_root = ROOT / "data/audits/tallip_kd/inductive_predictions"
    dataset_names = list(manifest["gold_repeat"]) if args.datasets == ["all"] else args.datasets
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["name"],
        "split": "dev_only",
        "bootstrap_seed": protocol["bootstrap_seed"],
        "bootstrap_iterations": protocol["bootstrap_iterations"],
        "comparison_basis": (
            "same dev examples; fixed pre-test thresholds; LLM-KD predictions were frozen "
            "before this post-lock protocol"
        ),
        "datasets": {},
    }
    bootstrap_checks = []
    table = [
        "# Post-lock dev-only control results",
        "",
        "All comparisons use the same gold dev examples, fixed pre-test thresholds, "
        "10,000 paired bootstrap resamples, and seed 20260713. Official-test labels are forbidden.",
        "",
        "| Dataset | Control | Five-seed mean±SD | Ensemble cF1 | LLM KD − control [95% CI] | Decision | Evidence grade / allowed claim |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    control_table = [
        "",
        "## Self-training versus exposure-matched repeated gold",
        "",
        "| Dataset | Self-training − repeated gold [95% CI] | Decision | Allowed claim |",
        "|---|---:|---|---|",
    ]
    for dataset in dataset_names:
        if dataset not in manifest["gold_repeat"]:
            raise SystemExit(f"Unknown dataset: {dataset}")
        gold_spec = manifest["gold_repeat"][dataset]
        gold = read_jsonl(gold_spec["dev"])
        llm_system = gold_spec["llm_system"]
        llm_path = inductive_root / f"{dataset}_{llm_system}_ensemble.jsonl"
        llm = load_prediction(llm_path)
        if llm is None:
            raise SystemExit(f"missing frozen LLM KD prediction: {llm_path}")
        dataset_report = {
            "dev": gold_spec["dev"],
            "llm_kd_prediction": str(llm_path.relative_to(ROOT)),
            "systems": {},
            "control_comparisons": {},
        }
        systems = ["gold_repeat"]
        if dataset in manifest["self_training"]:
            systems.append("self_training")
        for system in systems:
            ensemble_path = prediction_path(output_root, dataset, system, "ensemble")
            ensemble = load_prediction(ensemble_path)
            if ensemble is None:
                continue
            seed_scores = []
            for seed in seeds:
                rows = load_prediction(prediction_path(output_root, dataset, system, seed))
                if rows is not None:
                    seed_scores.append({"seed": seed, "cF1": evaluate_records(gold, rows)["cF1"]})
            ensemble_metrics = evaluate_records(gold, ensemble)
            comparison = checked_bootstrap(
                f"{dataset}:llm_kd_minus_{system}",
                gold,
                llm,
                ensemble,
                protocol,
                bootstrap_checks,
            )
            values = [item["cF1"] for item in seed_scores]
            summary = {
                "prediction": str(ensemble_path.relative_to(ROOT)),
                "ensemble": compact_metrics(ensemble_metrics),
                "single_seeds": seed_scores,
                "single_mean": statistics.mean(values) if values else None,
                "single_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "llm_kd_minus_control": comparison,
            }
            dataset_report["systems"][system] = summary
            low, high = comparison["ci95"]
            if comparison["classification"] == "supported":
                allowed = (
                    "High: LLM benefit exceeds exposure"
                    if system == "gold_repeat"
                    else "High: LLM teacher adds value over student pseudo-labels"
                )
            else:
                allowed = (
                    "High: no reliable LLM advantage over exposure"
                    if system == "gold_repeat"
                    else "High: no reliable LLM advantage over student pseudo-labels"
                )
            table.append(
                f"| {dataset} | {system} | {summary['single_mean']:.4f}±{summary['single_sd']:.4f} | "
                f"{ensemble_metrics['cF1']:.4f} | {comparison['delta_a_minus_b']:+.4f} "
                f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} | {allowed} |"
            )
        if {"gold_repeat", "self_training"} <= set(dataset_report["systems"]):
            repeated = load_prediction(
                prediction_path(output_root, dataset, "gold_repeat", "ensemble")
            )
            self_trained = load_prediction(
                prediction_path(output_root, dataset, "self_training", "ensemble")
            )
            comparison = checked_bootstrap(
                f"{dataset}:self_training_minus_gold_repeat",
                gold,
                self_trained,
                repeated,
                protocol,
                bootstrap_checks,
            )
            dataset_report["control_comparisons"][
                "self_training_minus_gold_repeat"
            ] = comparison
            low, high = comparison["ci95"]
            allowed = {
                "supported": "student pseudo-labels improve over repeated exposure",
                "harmful": "student pseudo-labels harm relative to repeated exposure",
                "inconclusive": "no reliable difference between the two non-LLM controls",
            }[comparison["classification"]]
            control_table.append(
                f"| {dataset} | {comparison['delta_a_minus_b']:+.4f} "
                f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} | {allowed} |"
            )
        report["datasets"][dataset] = dataset_report
    report["bootstrap_reproducibility"] = {
        "comparison_count": len(bootstrap_checks),
        "all_exact_repeats": True,
        "checks": bootstrap_checks,
    }
    table.extend(control_table)
    audit = build_result_audit(manifest, report, bootstrap_checks)
    metrics_path = output_root / "postlock_metrics.json"
    table_path = output_root / "postlock_table.md"
    audit_path = output_root / "postlock_audit.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    publish_root.mkdir(parents=True, exist_ok=True)
    (publish_root / "postlock_metrics.json").write_text(
        metrics_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (publish_root / "postlock_table.md").write_text(
        table_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (publish_root / "postlock_audit.json").write_text(
        audit_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print("\n".join(table))
    print(
        f"audit: predictions={audit['summary']['prediction_files_validated']}/48, "
        f"best={audit['summary']['best_checkpoints_validated']}/40, "
        f"bootstrap={audit['summary']['bootstrap_comparisons_repeated']} exact repeats"
    )
    return 0


def status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["protocol"]["seeds"]
    complete = 0
    total = 0
    for dataset, spec in select(manifest["gold_repeat"], args.datasets):
        states = []
        for seed in seeds:
            total += 1
            directory = ROOT / "checkpoints" / f"{spec['checkpoint_prefix']}{seed}"
            best = directory / "checkpoint_best.pt"
            marker = directory / "training_complete.json"
            if marker.exists() and best.exists():
                state = "complete"
                complete += 1
            elif best.exists():
                state = "running/partial"
            else:
                state = "pending"
            states.append(f"{seed}:{state}")
        print(f"{dataset:<20} " + ", ".join(states))
    print(f"gold-repeat progress: {complete}/{total} complete")
    self_complete = 0
    self_total = 0
    self_names = (
        ["all"]
        if args.datasets == ["all"]
        else [name for name in args.datasets if name in manifest["self_training"]]
    )
    self_selected = select(manifest["self_training"], self_names) if self_names else []
    for dataset, spec in self_selected:
        states = []
        for seed in seeds:
            self_total += 1
            directory = ROOT / "checkpoints" / f"{spec['checkpoint_prefix']}{seed}"
            best = directory / "checkpoint_best.pt"
            marker = directory / "training_complete.json"
            if marker.exists() and best.exists():
                state = "complete"
                self_complete += 1
            elif best.exists():
                state = "running/partial"
            else:
                state = "pending"
            states.append(f"{seed}:{state}")
        print(f"{dataset + ' self':<20} " + ", ".join(states))
    print(f"self-training progress: {self_complete}/{self_total} complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, function in (
        ("build-gold-repeat", build_gold_repeat),
        ("build-self-inputs", build_self_inputs),
        ("train-gold-repeat", train_gold_repeat),
        ("pseudo-label", pseudo_label),
        ("build-self-train", build_self_train),
        ("train-self", train_self),
        ("infer", infer),
        ("collect", collect),
        ("status", status),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--datasets", nargs="+", default=["all"])
        if name in {"train-gold-repeat", "pseudo-label", "train-self", "infer"}:
            command.add_argument("--dry-run", action="store_true")
        if name == "pseudo-label":
            command.add_argument("--batch-size", type=int, default=16)
        if name == "infer":
            command.add_argument(
                "--systems",
                nargs="+",
                choices=["gold_repeat", "self_training"],
                default=["gold_repeat", "self_training"],
            )
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--force", action="store_true")
        command.set_defaults(function=function)

    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
