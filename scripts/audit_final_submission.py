#!/usr/bin/env python3
"""Validate the frozen Task-3 submission against its manifest and test inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 1024)
QUAD_KEYS = {"Aspect", "Category", "Opinion", "VA"}
VA_RE = re.compile(r"^-?\d+(?:\.\d+)?#-?\d+(?:\.\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def allowed_categories(config_path: Path) -> set[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    entities = data.get("entity_labels", [])
    attributes = data.get("attribute_labels", [])
    return {f"{entity}#{attribute}" for entity in entities for attribute in attributes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="submission/final_manifest.json")
    parser.add_argument("--output", default="docs/audits/final_submission_audit.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = {Path(item["prediction"]).name for item in manifest["datasets"]}
    actual_names = {
        path.name for path in (ROOT / "submission/final").glob("*_test.jsonl")
        if "backup" not in path.name
    }
    issues: list[dict] = []
    results: list[dict] = []

    if actual_names != expected_names:
        issues.append({"severity": "error", "check": "final_file_set", "detail": {
            "missing": sorted(expected_names - actual_names),
            "extra": sorted(actual_names - expected_names),
        }})

    archive_path = ROOT / manifest["archive"]["path"]
    archive_hash = sha256(archive_path)
    if archive_hash != manifest["archive"]["sha256"]:
        issues.append({"severity": "error", "check": "archive_hash", "detail": archive_hash})

    with zipfile.ZipFile(archive_path) as archive:
        zip_names = set(archive.namelist())
        if zip_names != expected_names:
            issues.append({"severity": "error", "check": "archive_members", "detail": {
                "missing": sorted(expected_names - zip_names), "extra": sorted(zip_names - expected_names)
            }})
        zip_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in zip_names}

    for item in manifest["datasets"]:
        pred_path = ROOT / item["prediction"]
        test_path = ROOT / item["test_file"]
        config_path = ROOT / item["config"]
        local_issues = []
        pred_hash = sha256(pred_path)
        config_hash = sha256(config_path)
        if pred_hash != item["prediction_sha256"]:
            local_issues.append("prediction_hash_mismatch")
        if config_hash != item["config_sha256"]:
            local_issues.append("config_hash_mismatch")
        if zip_hashes.get(pred_path.name) != pred_hash:
            local_issues.append("archive_bytes_differ")

        predictions = read_jsonl(pred_path)
        tests = read_jsonl(test_path)
        if len(predictions) != len(tests):
            local_issues.append("row_count_mismatch")
        ids = [row.get("ID") for row in predictions]
        if len(ids) != len(set(ids)):
            local_issues.append("duplicate_prediction_ids")
        categories = allowed_categories(config_path)
        invalid_quads = 0
        invalid_spans = 0
        invalid_categories = 0
        invalid_va = 0
        total_quads = 0
        for index, (pred, test) in enumerate(zip(predictions, tests)):
            if pred.get("ID") != test.get("ID"):
                local_issues.append(f"id_or_order_mismatch_at_{index}")
                break
            if "Text" in pred and pred["Text"] != test.get("Text"):
                local_issues.append(f"text_mismatch_at_{index}")
                break
            if set(pred) - {"ID", "Text", "Quadruplet"} or not isinstance(pred.get("Quadruplet"), list):
                local_issues.append(f"invalid_row_schema_at_{index}")
                break
            text = test.get("Text", "")
            for quad in pred["Quadruplet"]:
                total_quads += 1
                if not isinstance(quad, dict) or set(quad) != QUAD_KEYS:
                    invalid_quads += 1
                    continue
                if quad["Category"] not in categories:
                    invalid_categories += 1
                if not isinstance(quad["VA"], str) or not VA_RE.match(quad["VA"]):
                    invalid_va += 1
                else:
                    values = [float(value) for value in quad["VA"].split("#")]
                    invalid_va += int(any(value < 1 or value > 9 for value in values))
                for field in ("Aspect", "Opinion"):
                    value = quad[field]
                    if value != "NULL" and value not in text:
                        invalid_spans += 1

        if invalid_quads:
            local_issues.append(f"invalid_quad_schema:{invalid_quads}")
        if invalid_categories:
            local_issues.append(f"invalid_categories:{invalid_categories}")
        if invalid_va:
            local_issues.append(f"invalid_va:{invalid_va}")
        # Span extraction may retain normalized punctuation/casing; keep it visible but non-fatal.
        if invalid_spans:
            issues.append({"severity": "warning", "check": "non_exact_spans", "dataset": item["dataset"], "detail": invalid_spans})

        checkpoint_paths = [ROOT / f"{item['checkpoint_prefix']}{seed}" / "checkpoint_best.pt" for seed in SEEDS]
        checkpoint_count = sum(path.exists() for path in checkpoint_paths)
        expected_checkpoint_count = 5 if item["checkpoint_state"] == "complete" else 0
        if checkpoint_count != expected_checkpoint_count:
            local_issues.append(f"checkpoint_count:{checkpoint_count}_expected:{expected_checkpoint_count}")
        if item["checkpoint_state"] == "historical_missing":
            issues.append({"severity": "warning", "check": "historical_checkpoints_missing", "dataset": item["dataset"], "detail": item["provenance_note"]})

        results.append({
            "dataset": item["dataset"], "rows": len(predictions), "quadruplets": total_quads,
            "prediction_sha256": pred_hash, "zip_bytes_match": zip_hashes.get(pred_path.name) == pred_hash,
            "checkpoint_best_files": checkpoint_count, "status": "pass" if not local_issues else "fail",
            "issues": local_issues,
        })
        for detail in local_issues:
            issues.append({"severity": "error", "check": "dataset_validation", "dataset": item["dataset"], "detail": detail})

    errors = sum(issue["severity"] == "error" for issue in issues)
    report = {
        "schema_version": 1,
        "manifest": args.manifest,
        "status": "pass_with_warnings" if not errors and issues else ("pass" if not errors else "fail"),
        "summary": {"datasets": len(results), "rows": sum(row["rows"] for row in results),
                    "errors": errors, "warnings": sum(issue["severity"] == "warning" for issue in issues)},
        "archive": {"path": manifest["archive"]["path"], "sha256": archive_hash,
                    "member_count": len(zip_hashes)},
        "datasets": results,
        "issues": issues,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"status={report['status']} output={output_path.relative_to(ROOT)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
