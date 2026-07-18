#!/usr/bin/env python
"""Audit KD data leakage and build deterministic clean-inductive files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tallip_common import (
    ROOT, load_manifest, normalize_text, read_jsonl, root_path, sha256_file, write_jsonl,
)


def index_reference(rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(row.get("Text", "")) for row in rows},
        {normalize_text(row.get("Text", "")) for row in rows},
        {str(row.get("ID", "")) for row in rows},
    )


def inspect_file(path: str, dev_rows: list[dict], test_rows: list[dict]) -> dict:
    rows = read_jsonl(path)
    dev_exact, dev_norm, dev_ids = index_reference(dev_rows)
    test_exact, test_norm, test_ids = index_reference(test_rows)
    matches = []
    reason_counts = Counter()
    for line_no, row in enumerate(rows, 1):
        text = str(row.get("Text", ""))
        normalized = normalize_text(text)
        row_id = str(row.get("ID", ""))
        reasons = []
        for split, exact, norm, ids in (
            ("dev", dev_exact, dev_norm, dev_ids), ("test", test_exact, test_norm, test_ids)
        ):
            if text in exact:
                reasons.append(f"{split}:exact_text")
            if normalized in norm:
                reasons.append(f"{split}:normalized_text")
            if row_id and row_id in ids:
                reasons.append(f"{split}:id")
        if reasons:
            reason_counts.update(reasons)
            matches.append({
                "line": line_no, "ID": row_id, "Text": text,
                "normalized_text": normalized, "reasons": reasons,
            })
    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "matched_rows": len(matches),
        "reason_counts": dict(reason_counts),
        "matches": matches,
    }


def discover_dataset_files(dataset: str, spec: dict, registry: dict) -> dict[str, str]:
    result = {}
    for key in ("gold_train", "adopted_train", "baseline_train", "clean_train", "clean_baseline_train"):
        path = spec.get(key)
        if path and root_path(path).exists():
            result[key] = path
    for exp_name, exp in registry.get("experiments", {}).items():
        if exp.get("dataset") != dataset:
            continue
        for role, key in (("gold", "gold_base"), ("pool", "pool"), ("silver", "silver_file"), ("distill", "distill_file")):
            if key == "pool":
                path = exp.get("pool", {}).get("file")
            else:
                path = exp.get(key)
            if not path and key == "silver_file":
                suffix = f"_{exp['variant']}" if exp.get("variant") else ""
                path = f"data/subtask_3/{exp['lang']}/{dataset}_silver_v23_{exp['teacher']}{suffix}.jsonl"
            if not path and key == "distill_file":
                suffix = f"_{exp['variant']}" if exp.get("variant") else ""
                path = f"data/subtask_3/{exp['lang']}/{dataset}_train_v23_{exp['teacher']}{suffix}.jsonl"
            if path and root_path(path).exists():
                result[f"registry:{exp_name}:{role}"] = path
    return dict(sorted(result.items()))


def run_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    with root_path(args.registry).open(encoding="utf-8") as handle:
        import yaml
        registry = yaml.safe_load(handle)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalization": "Unicode NFKC + casefold + Unicode whitespace collapse",
        "manifest_sha256": sha256_file(args.manifest),
        "datasets": {},
    }
    detail_dir = root_path(manifest["defaults"]["output_root"]) / "contamination"
    for dataset, spec in manifest["datasets"].items():
        dev_rows, test_rows = read_jsonl(spec["dev"]), read_jsonl(spec["test"])
        files = discover_dataset_files(dataset, spec, registry)
        file_reports = {}
        for role, path in files.items():
            file_report = inspect_file(path, dev_rows, test_rows)
            matches = file_report.pop("matches")
            if matches:
                safe_role = role.replace(":", "__")
                detail_path = detail_dir / f"{dataset}__{safe_role}.jsonl"
                write_jsonl(detail_path, matches)
                file_report["match_detail"] = str(detail_path.relative_to(ROOT))
                file_report["match_detail_sha256"] = sha256_file(detail_path)
            file_reports[role] = file_report
        report["datasets"][dataset] = {
            "dev": {"path": spec["dev"], "rows": len(dev_rows), "sha256": sha256_file(spec["dev"])},
            "test": {"path": spec["test"], "rows": len(test_rows), "sha256": sha256_file(spec["test"])},
            "files": file_reports,
        }
    out = root_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    for dataset, dataset_report in report["datasets"].items():
        adopted = dataset_report["files"].get("adopted_train")
        if adopted:
            print(f"{dataset:<20} adopted overlap={adopted['matched_rows']}/{adopted['rows']}")
    return 0


def filter_one(dataset: str, label: str, source: str, output: str, expected: int,
               dev: str, test: str, removed_dir: Path) -> dict:
    rows, dev_rows, test_rows = read_jsonl(source), read_jsonl(dev), read_jsonl(test)
    dev_norm = {normalize_text(row.get("Text", "")) for row in dev_rows}
    test_norm = {normalize_text(row.get("Text", "")) for row in test_rows}
    kept, removed = [], []
    for line_no, row in enumerate(rows, 1):
        normalized = normalize_text(row.get("Text", ""))
        splits = [split for split, values in (("dev", dev_norm), ("test", test_norm)) if normalized in values]
        if splits:
            removed.append({
                "source_line": line_no, "source": source, "target_splits": splits,
                "normalized_text": normalized, "row": row,
            })
        else:
            kept.append(row)
    if len(kept) != expected:
        raise SystemExit(f"{dataset}/{label}: expected {expected} clean rows, got {len(kept)}; refusing to write")
    write_jsonl(output, kept)
    removed_path = removed_dir / f"{dataset}_{label}_removed.jsonl"
    write_jsonl(removed_path, removed)
    return {
        "dataset": dataset, "label": label, "source": source, "source_sha256": sha256_file(source),
        "output": output, "output_sha256": sha256_file(output), "source_rows": len(rows),
        "kept_rows": len(kept), "removed_rows": len(removed), "removed_file": str(removed_path.relative_to(ROOT)),
        "removed_sha256": sha256_file(removed_path),
        "dev_removed": sum("dev" in row["target_splits"] for row in removed),
        "test_removed": sum("test" in row["target_splits"] for row in removed),
    }


def run_filter(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    output_root = root_path(manifest["defaults"]["output_root"])
    removed_dir = output_root / "removed"
    summaries = []
    for dataset, spec in manifest["datasets"].items():
        if spec.get("clean_rerun"):
            summaries.append(filter_one(
                dataset, "kd", spec["adopted_train"], spec["clean_train"], spec["expected_clean_rows"],
                spec["dev"], spec["test"], removed_dir,
            ))
        if spec.get("clean_baseline_rerun"):
            summaries.append(filter_one(
                dataset, "baseline", spec["baseline_train"], spec["clean_baseline_train"],
                spec["expected_clean_baseline_rows"], spec["dev"], spec["test"], removed_dir,
            ))
    summary_path = root_path(args.output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"normalization": "NFKC+casefold+whitespace", "files": summaries},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    for summary in summaries:
        print(f"{summary['dataset']:<20} {summary['label']:<8} {summary['source_rows']} -> {summary['kept_rows']} "
              f"(removed {summary['removed_rows']})")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    failed = False
    for dataset, spec in manifest["datasets"].items():
        candidates = []
        if spec.get("adopted"):
            candidates.append(("adopted", spec.get("clean_train", spec.get("adopted_train"))))
        if spec.get("clean_baseline_train"):
            candidates.append(("baseline", spec["clean_baseline_train"]))
        for label, path in candidates:
            result = inspect_file(path, read_jsonl(spec["dev"]), read_jsonl(spec["test"]))
            normalized = sum(value for key, value in result["reason_counts"].items() if key.endswith(":normalized_text"))
            print(f"{dataset:<20} {label:<8} normalized overlap={normalized}")
            failed |= normalized != 0
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="scripts/kd/tallip.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--registry", default="scripts/kd/registry.yaml")
    audit.add_argument("--output", default="docs/audits/tallip_kd/contamination_audit.json")
    audit.set_defaults(func=run_audit)
    clean = sub.add_parser("filter")
    clean.add_argument("--output", default="docs/audits/tallip_kd/cleaning_manifest.json")
    clean.set_defaults(func=run_filter)
    verify = sub.add_parser("verify")
    verify.set_defaults(func=run_verify)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
