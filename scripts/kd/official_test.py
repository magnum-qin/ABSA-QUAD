#!/usr/bin/env python
"""One-shot official-test inference, audit, scoring, and reproducibility record."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tallip_common import ROOT, evaluate_records, paired_bootstrap, read_jsonl, sha256_file

SEEDS = (42, 123, 456, 789, 1024)
MANIFEST = ROOT / "scripts/kd/official_test.yaml"


def load_protocol(path: str | Path = MANIFEST) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def path_for(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def output_path(protocol: dict, system: str) -> Path:
    return path_for(protocol["protocol"]["output_root"]) / "predictions" / f"{system}.jsonl"


def system_path(protocol: dict, system: str, source: str) -> Path:
    return output_path(protocol, system) if source == "generated" else path_for(source)


def run(command: list[str], log: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            return subprocess.run(command, cwd=ROOT, check=True, text=True,
                                  stdout=handle, stderr=subprocess.STDOUT)
    return subprocess.run(command, cwd=ROOT, check=True, text=True,
                          capture_output=True)


def validate_unlabelled_test(path: Path) -> tuple[int, list[str]]:
    rows = read_jsonl(path)
    ids = [str(row.get("ID", "")) for row in rows]
    if not rows or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"Invalid local test IDs: {path}")
    leaked = [str(row["ID"]) for row in rows if "Quadruplet" in row]
    if leaked:
        raise SystemExit(f"Local test unexpectedly contains labels: {path} ({len(leaked)} rows)")
    if any("Text" not in row for row in rows):
        raise SystemExit(f"Local test is missing Text: {path}")
    return len(rows), ids


def command_preflight(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    summary = {"datasets": {}, "inference": {}}
    for dataset, spec in protocol["datasets"].items():
        local = path_for(spec["local_test"])
        count, _ = validate_unlabelled_test(local)
        summary["datasets"][dataset] = {
            "rows": count,
            "local_test": str(local.relative_to(ROOT)),
            "local_test_sha256": sha256_file(local),
            "main_system": spec["main_system"],
        }
        for system, source in spec["systems"].items():
            if source != "generated":
                prediction = system_path(protocol, system, source)
                if not prediction.exists():
                    raise SystemExit(f"Missing frozen prediction: {prediction}")
    for system, spec in protocol["inference"].items():
        checkpoints = []
        for seed in SEEDS:
            checkpoint = ROOT / "checkpoints" / f"{spec['checkpoint_prefix']}{seed}" / "checkpoint_best.pt"
            if not checkpoint.exists():
                raise SystemExit(f"Missing best checkpoint: {checkpoint}")
            checkpoints.append({"seed": seed, "path": str(checkpoint.relative_to(ROOT)),
                                "sha256": sha256_file(checkpoint)})
        summary["inference"][system] = {
            "dataset": spec["dataset"], "config": spec["config"],
            "config_sha256": sha256_file(spec["config"]), "vote": spec["vote"],
            "topk": spec["topk"], "checkpoints": checkpoints,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_infer(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    root = path_for(protocol["protocol"]["output_root"])
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for system, spec in protocol["inference"].items():
        output = output_path(protocol, system)
        if output.exists() and not args.force:
            print(f"SKIP existing {output.relative_to(ROOT)}")
            continue
        dataset = protocol["datasets"][spec["dataset"]]
        seed_dirs = [f"checkpoints/{spec['checkpoint_prefix']}{seed}" for seed in SEEDS]
        command = [
            sys.executable, "scripts/inference.py", "--multi_seed_ensemble",
            "--seed_dirs", *seed_dirs, "--config", spec["config"],
            "--test_file", dataset["local_test"], "--output", str(output),
            "--vote_threshold", str(spec["vote"]),
            "--span_topk_threshold", str(spec["topk"]),
            "--batch_size", str(args.batch_size),
        ]
        run(command, root / "logs" / f"infer_{system}.log")
    return 0


def validate_prediction(gold_ids: list[str], path: Path) -> dict:
    rows = read_jsonl(path)
    ids = [str(row.get("ID", "")) for row in rows]
    missing = sorted(set(gold_ids) - set(ids))
    extra = sorted(set(ids) - set(gold_ids))
    duplicates = sorted(item for item in set(ids) if ids.count(item) > 1)
    malformed = [str(row.get("ID", "")) for row in rows
                 if not isinstance(row.get("Quadruplet"), list)]
    if missing or extra or duplicates or malformed or len(ids) != len(gold_ids):
        raise SystemExit(
            f"Invalid prediction {path}: missing={len(missing)} extra={len(extra)} "
            f"duplicates={len(duplicates)} malformed={len(malformed)}"
        )
    return {"rows": len(rows), "sha256": sha256_file(path)}


def command_freeze(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    audit = path_for(protocol["protocol"]["audit_root"])
    audit.mkdir(parents=True, exist_ok=True)
    git_head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    lock = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.manifest),
        "git_head": git_head,
        "official_commit": protocol["protocol"]["official_commit"],
        "selection_basis": protocol["protocol"]["selection_basis"],
        "no_test_tuning": protocol["protocol"]["no_test_tuning"],
        "datasets": {},
    }
    for dataset, spec in protocol["datasets"].items():
        local = path_for(spec["local_test"])
        count, ids = validate_unlabelled_test(local)
        entry = {
            "local_test": spec["local_test"], "local_test_sha256": sha256_file(local),
            "rows": count, "main_system": spec["main_system"], "systems": {},
        }
        for system, source in spec["systems"].items():
            prediction = system_path(protocol, system, source)
            if not prediction.exists():
                raise SystemExit(f"Cannot freeze missing prediction: {prediction}")
            validated = validate_prediction(ids, prediction)
            entry["systems"][system] = {
                "path": str(prediction.relative_to(ROOT)), **validated,
            }
        lock["datasets"][dataset] = entry
    lock_path = audit / "protocol_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen: {lock_path.relative_to(ROOT)}")
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    meta = protocol["protocol"]
    destination = path_for(meta["gold_root"])
    evaluator_destination = destination / meta["evaluator"]
    if destination.exists() and evaluator_destination.exists() and not args.force:
        print(f"SKIP existing {destination.relative_to(ROOT)}")
        return 0
    lock = path_for(meta["audit_root"]) / "protocol_lock.json"
    if not lock.exists():
        raise SystemExit("Refusing to fetch gold before protocol_lock.json exists")
    with tempfile.TemporaryDirectory(prefix="dimabsa-official-") as tmp:
        clone = Path(tmp) / "repo"
        run(["git", "clone", "--quiet", meta["official_repository"], str(clone)])
        run(["git", "-C", str(clone), "checkout", "--quiet", meta["official_commit"]])
        actual = run(["git", "-C", str(clone), "rev-parse", "HEAD"]).stdout.strip()
        if actual != meta["official_commit"]:
            raise SystemExit(f"Official commit mismatch: {actual}")
        destination.mkdir(parents=True, exist_ok=True)
        subtree = clone / meta["official_subtree"]
        for spec in protocol["datasets"].values():
            source = subtree / spec["gold"]
            target = destination / spec["gold"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        evaluator_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clone / meta["evaluator"], evaluator_destination)
        shutil.copy2(clone / "evaluation_script/requirements.txt",
                     evaluator_destination.parent / "requirements.txt")
    print(f"Fetched official gold at commit {meta['official_commit']}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    meta = protocol["protocol"]
    lock_path = path_for(meta["audit_root"]) / "protocol_lock.json"
    if not lock_path.exists():
        raise SystemExit("Missing protocol lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = {"official_commit": meta["official_commit"], "datasets": {}}
    for dataset, spec in protocol["datasets"].items():
        local_path = path_for(spec["local_test"])
        gold_path = path_for(meta["gold_root"]) / spec["gold"]
        local, gold = read_jsonl(local_path), read_jsonl(gold_path)
        local_by_id = {str(row["ID"]): row for row in local}
        gold_by_id = {str(row["ID"]): row for row in gold}
        same_order = [str(row["ID"]) for row in local] == [str(row["ID"]) for row in gold]
        same_ids = set(local_by_id) == set(gold_by_id)
        text_mismatches = [item for item in local_by_id
                           if item in gold_by_id and local_by_id[item].get("Text") != gold_by_id[item].get("Text")]
        label_rows = sum(isinstance(row.get("Quadruplet"), list) for row in gold)
        if not same_order or not same_ids or text_mismatches or label_rows != len(gold):
            raise SystemExit(
                f"Gold audit failed for {dataset}: order={same_order} ids={same_ids} "
                f"text_mismatches={len(text_mismatches)} labels={label_rows}/{len(gold)}"
            )
        locked = lock["datasets"][dataset]
        for system, item in locked["systems"].items():
            path = path_for(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise SystemExit(f"Prediction changed after freeze: {system}")
            validate_prediction(list(gold_by_id), path)
        report["datasets"][dataset] = {
            "rows": len(gold), "id_set_equal": same_ids, "id_order_equal": same_order,
            "text_exact_equal": not text_mismatches, "gold_sha256": sha256_file(gold_path),
            "local_unlabelled_sha256": sha256_file(local_path),
        }
    output = path_for(meta["audit_root"]) / "gold_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit passed: {output.relative_to(ROOT)}")
    return 0


def official_score(evaluator: Path, prediction: Path, gold: Path) -> tuple[float, str]:
    result = run([sys.executable, str(evaluator), "-t", "3", "-p", str(prediction), "-g", str(gold)])
    matches = re.findall(r"['\"]cF1['\"]:\s*([0-9.eE+-]+)", result.stdout)
    if not matches:
        matches = re.findall(r"cF1:\s*([0-9.eE+-]+)", result.stdout)
    if not matches:
        raise SystemExit(f"Could not parse official evaluator output for {prediction}:\n{result.stdout}")
    # The official script iterates over a set of IDs, so summation order can
    # vary at ~1e-16 across Python processes. Twelve decimals preserve far more
    # precision than reported while making the audit JSON byte-reproducible.
    return round(float(matches[-1]), 12), result.stdout


def command_score(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.manifest)
    meta = protocol["protocol"]
    audit_root = path_for(meta["audit_root"])
    audit_report = audit_root / "gold_audit.json"
    if not audit_report.exists():
        raise SystemExit("Refusing to score before gold_audit.json exists")
    lock = json.loads((audit_root / "protocol_lock.json").read_text(encoding="utf-8"))
    evaluator = path_for(meta["gold_root"]) / meta["evaluator"]
    raw_root = audit_root / "official_evaluator_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "post-competition evaluation on released official test labels",
        "selection_basis": "dev-only; prediction hashes frozen before gold retrieval",
        "official_commit": meta["official_commit"],
        "official_evaluator_sha256": sha256_file(evaluator),
        "bootstrap_iterations": meta["bootstrap_iterations"],
        "bootstrap_seed": meta["bootstrap_seed"],
        "datasets": {}, "comparisons": [],
    }
    loaded: dict[tuple[str, str], list[dict]] = {}
    for dataset, spec in protocol["datasets"].items():
        gold_path = path_for(meta["gold_root"]) / spec["gold"]
        gold = read_jsonl(gold_path)
        systems = {}
        for system, item in lock["datasets"][dataset]["systems"].items():
            prediction_path = path_for(item["path"])
            if sha256_file(prediction_path) != item["sha256"]:
                raise SystemExit(f"Prediction changed after lock: {system}")
            score, raw = official_score(evaluator, prediction_path, gold_path)
            (raw_root / f"{dataset}__{system}.txt").write_text(raw, encoding="utf-8")
            internal = evaluate_records(gold, read_jsonl(prediction_path))
            if abs(score - internal["cF1"]) > 5e-6:
                raise SystemExit(f"Official/internal cF1 mismatch for {system}: {score} vs {internal['cF1']}")
            systems[system] = {
                "official_cF1": score, "prediction": item["path"],
                "prediction_sha256": item["sha256"],
            }
            loaded[(dataset, system)] = read_jsonl(prediction_path)
        result["datasets"][dataset] = {
            "gold_sha256": sha256_file(gold_path), "main_system": spec["main_system"],
            "main_cF1": systems[spec["main_system"]]["official_cF1"], "systems": systems,
        }
    for comparison in protocol["comparisons"]:
        dataset = comparison["dataset"]
        gold_path = path_for(meta["gold_root"]) / protocol["datasets"][dataset]["gold"]
        bootstrap = paired_bootstrap(
            read_jsonl(gold_path), loaded[(dataset, comparison["a"])],
            loaded[(dataset, comparison["b"])], meta["bootstrap_iterations"],
            meta["bootstrap_seed"],
        )
        result["comparisons"].append({**comparison, **bootstrap})
    metrics_path = audit_root / "official_test_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Official Task-3 test results", "",
        "Post-competition evaluation on the released official test labels; all system choices and prediction hashes were frozen before gold retrieval.", "",
        "| Dataset | Frozen main system | Official test cF1 |", "|---|---|---:|",
    ]
    for dataset, entry in result["datasets"].items():
        lines.append(f"| {dataset} | {entry['main_system']} | {entry['main_cF1']:.4f} |")
    lines.extend(["", "| Dataset | Comparison | ΔcF1 [95% CI] | Decision |", "|---|---|---:|---|"])
    for item in result["comparisons"]:
        low, high = item["ci95"]
        lines.append(
            f"| {item['dataset']} | {item['a']} − {item['b']} | "
            f"{item['delta_a_minus_b']:+.4f} [{low:+.4f}, {high:+.4f}] | {item['classification']} |"
        )
    lines.extend(["", f"Paired bootstrap: {meta['bootstrap_iterations']:,} resamples, seed={meta['bootstrap_seed']}.", ""])
    table_path = audit_root / "official_test_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=command_preflight)
    infer = sub.add_parser("infer")
    infer.add_argument("--batch-size", type=int, default=16)
    infer.add_argument("--force", action="store_true")
    infer.set_defaults(func=command_infer)
    freeze = sub.add_parser("freeze")
    freeze.set_defaults(func=command_freeze)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=command_fetch)
    audit = sub.add_parser("audit")
    audit.set_defaults(func=command_audit)
    score = sub.add_parser("score")
    score.set_defaults(func=command_score)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
