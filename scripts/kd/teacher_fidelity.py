#!/usr/bin/env python
"""Prepare, run, validate, and evaluate the 8x2 TALLIP teacher-fidelity probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tallip_common import (
    ROOT, evaluate_records, load_manifest, normalize_text, paired_bootstrap, read_jsonl, root_path,
    sha256_file, write_jsonl,
)


def load_env_file(path: str | None) -> None:
    """Load simple KEY=VALUE entries without logging secret values."""
    if not path:
        return
    env_path = root_path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def selected_datasets(manifest: dict, names: list[str]) -> list[tuple[str, dict]]:
    if names == ["all"]:
        return list(manifest["datasets"].items())
    unknown = sorted(set(names) - set(manifest["datasets"]))
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")
    return [(name, manifest["datasets"][name]) for name in names]


def paths_for(manifest: dict, dataset: str, teacher: str) -> dict[str, Path]:
    root = root_path(manifest["defaults"]["output_root"]) / "teacher_fidelity"
    return {
        "input": root / "inputs" / f"{dataset}.jsonl",
        "output": root / "validated" / f"{dataset}_{teacher}.jsonl",
        "raw": root / "raw" / f"{dataset}_{teacher}.jsonl",
        "metadata": root / "metadata" / f"{dataset}_{teacher}.json",
    }


def prepare(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    rows_total = 0
    for dataset, spec in selected_datasets(manifest, args.datasets):
        gold = read_jsonl(spec["dev"])
        probes = [{key: value for key, value in row.items() if key != "Quadruplet"} for row in gold]
        if any("Quadruplet" in row for row in probes):
            raise AssertionError("Gold labels leaked into teacher input")
        paths = paths_for(manifest, dataset, "deepseek")
        write_jsonl(paths["input"], probes)
        rows_total += len(probes)
        print(f"{dataset:<20} {len(probes):>4} -> {paths['input'].relative_to(ROOT)}")
    print(f"Prepared {rows_total} label-free dev probes")
    return 0


def build_command(args: argparse.Namespace, manifest: dict, dataset: str, spec: dict,
                  teacher: str, limit: int | None) -> list[str]:
    paths = paths_for(manifest, dataset, teacher)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    common = [
        "--input", str(paths["input"]), "--train_file", str(root_path(spec["gold_train"])),
        "--output", str(paths["output"]), "--domain", spec["domain"], "--n_shots", "6",
        "--raw_output", str(paths["raw"]), "--metadata_output", str(paths["metadata"]),
        "--prompt_version", "dim-asqp-fidelity-v1",
    ]
    if limit is not None:
        common.extend(["--limit", str(limit)])
    if teacher == "deepseek":
        return [sys.executable, str(ROOT / "scripts/llm_silver_label.py"), *common,
                "--model", args.deepseek_model, "--workers", str(args.deepseek_workers)]
    return [
        sys.executable, str(ROOT / "scripts/llm_silver_label_openai.py"), *common,
        "--base_url", args.qwen_base_url, "--model", args.qwen_model,
        "--workers", str(args.qwen_workers),
    ]


def validate_one(manifest: dict, dataset: str, teacher: str, expected: int,
                 max_error_rate: float, strict_model: bool = False) -> dict:
    paths = paths_for(manifest, dataset, teacher)
    inputs = read_jsonl(paths["input"])
    outputs = read_jsonl(paths["output"]) if paths["output"].exists() else []
    raw = read_jsonl(paths["raw"]) if paths["raw"].exists() else []
    input_ids = [str(row["ID"]) for row in inputs[:expected]]
    output_ids = [str(row.get("ID")) for row in outputs]
    raw_ids = [str(row.get("ID")) for row in raw]
    duplicate_ids = [row_id for row_id, count in Counter(output_ids).items() if count != 1]
    errors = sum(bool(row.get("_err")) for row in outputs)
    response_models = sorted({str(row.get("response_model")) for row in raw if row.get("response_model")})
    missing = sorted(set(input_ids) - set(output_ids))
    extra = sorted(set(output_ids) - set(input_ids)) if expected < len(inputs) else sorted(set(output_ids) - set(input_ids))
    model_ok = bool(response_models)
    if strict_model:
        needle = "deepseek" if teacher == "deepseek" else "qwen"
        model_ok = model_ok and all(needle in model.casefold() for model in response_models)
    result = {
        "dataset": dataset, "teacher": teacher, "expected_rows": expected,
        "output_rows": len(outputs), "raw_rows": len(raw), "errors": errors,
        "error_rate": errors / expected if expected else 0.0,
        "duplicate_ids": duplicate_ids, "missing_ids": missing, "extra_ids": extra,
        "response_models": response_models, "model_identity_ok": model_ok,
        "input_sha256": sha256_file(paths["input"]),
        "output_sha256": sha256_file(paths["output"]) if paths["output"].exists() else None,
    }
    result["ok"] = (
        len(outputs) == expected and len(raw) == expected and not duplicate_ids and not missing and not extra
        and result["error_rate"] <= max_error_rate and model_ok
        and set(output_ids) == set(raw_ids)
    )
    return result


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    limit = 10 if args.phase == "smoke" else None
    max_error_rate = 0.0 if args.phase == "smoke" else 0.02
    failed = False
    for dataset, spec in selected_datasets(manifest, args.datasets):
        paths = paths_for(manifest, dataset, args.teacher)
        if not paths["input"].exists():
            raise SystemExit(f"Missing {paths['input']}; run prepare first")
        command = build_command(args, manifest, dataset, spec, args.teacher, limit)
        print("$ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
            expected = min(10, len(read_jsonl(paths["input"]))) if limit else len(read_jsonl(paths["input"]))
            result = validate_one(manifest, dataset, args.teacher, expected, max_error_rate, args.strict_model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["ok"]:
                failed = True
                if args.phase == "smoke":
                    print(f"Smoke gate failed for {dataset}/{args.teacher}; stopping.", file=sys.stderr)
                    break
    return 1 if failed else 0


def validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    results = []
    failed = False
    for dataset, _ in selected_datasets(manifest, args.datasets):
        for teacher in args.teachers:
            expected = len(read_jsonl(paths_for(manifest, dataset, teacher)["input"]))
            result = validate_one(manifest, dataset, teacher, expected, args.max_error_rate, args.strict_model)
            results.append(result)
            failed |= not result["ok"]
            print(f"{dataset:<20} {teacher:<8} rows={result['output_rows']}/{expected} "
                  f"errors={result['error_rate']:.2%} model={','.join(result['response_models']) or 'MISSING'} "
                  f"{'OK' if result['ok'] else 'FAIL'}")
    if args.output:
        path = root_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


def evaluate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    iterations = manifest["defaults"]["bootstrap_iterations"]
    seed = manifest["defaults"]["bootstrap_seed"]
    report = {"bootstrap_iterations": iterations, "bootstrap_seed": seed, "datasets": {}}
    table_rows = []
    for dataset, spec in selected_datasets(manifest, args.datasets):
        gold, train = read_jsonl(spec["dev"]), read_jsonl(spec["gold_train"])
        for teacher in ("deepseek", "qwen"):
            validation = validate_one(manifest, dataset, teacher, len(gold), 0.02, strict_model=True)
            if not validation["ok"]:
                raise SystemExit(f"Refusing to evaluate invalid run {dataset}/{teacher}: {validation}")
        predictions = {teacher: read_jsonl(paths_for(manifest, dataset, teacher)["output"])
                       for teacher in ("deepseek", "qwen")}
        teacher_metrics = {}
        for teacher, rows in predictions.items():
            metrics = evaluate_records(gold, rows, train)
            metrics.pop("_row_counts")
            teacher_metrics[teacher] = metrics
        bootstrap = paired_bootstrap(gold, predictions["deepseek"], predictions["qwen"], iterations, seed)
        report["datasets"][dataset] = {"teachers": teacher_metrics, "deepseek_minus_qwen": bootstrap}
        ds, qw = teacher_metrics["deepseek"], teacher_metrics["qwen"]
        table_rows.append((dataset, ds["cF1"], qw["cF1"], bootstrap["delta_a_minus_b"],
                           bootstrap["ci95"][0], bootstrap["ci95"][1], bootstrap["classification"]))
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# TALLIP teacher fidelity (gold dev, labels hidden from teachers)", "",
        "| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, ds, qw, delta, low, high, decision in table_rows:
        markdown.append(f"| {dataset} | {ds:.4f} | {qw:.4f} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] | {decision} |")
    markdown.extend(["", f"Paired bootstrap: {iterations:,} resamples, seed={seed}.", ""])
    markdown_path = root_path(args.markdown)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


def evaluate_single(args: argparse.Namespace) -> int:
    """Evaluate one complete teacher run without implying a paired comparison."""
    manifest = load_manifest(args.manifest)
    report = {
        "scope": f"standalone full-dev fidelity for {args.teacher}; no cross-teacher inference",
        "teacher": args.teacher,
        "datasets": {},
    }
    markdown = [
        f"# {args.teacher} standalone full-dev teacher fidelity", "",
        "This table is unpaired evidence and must not be used as an 8x2 teacher comparison.", "",
        "| Dataset | n | cF1 | Exact F1 | Aspect F1 | Opinion F1 | Category F1 | VA MAE | VA RMSE | Empty | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, spec in selected_datasets(manifest, args.datasets):
        gold = read_jsonl(spec["dev"])
        validation = validate_one(manifest, dataset, args.teacher, len(gold), 0.02, strict_model=True)
        if not validation["ok"]:
            raise SystemExit(f"Refusing to evaluate invalid run {dataset}/{args.teacher}: {validation}")
        predictions = read_jsonl(paths_for(manifest, dataset, args.teacher)["output"])
        metrics = evaluate_records(gold, predictions, read_jsonl(spec["gold_train"]))
        metrics.pop("_row_counts")
        report["datasets"][dataset] = {"validation": validation, "metrics": metrics}
        coverage = metrics["coverage"]
        markdown.append(
            f"| {dataset} | {len(gold)} | {metrics['cF1']:.4f} | {metrics['exact_f1']:.4f} | "
            f"{metrics['aspect_span']['f1']:.4f} | {metrics['opinion_span']['f1']:.4f} | "
            f"{metrics['category_span']['f1']:.4f} | {metrics['va_mae']:.4f} | {metrics['va_rmse']:.4f} | "
            f"{coverage['empty_rate']:.2%} | {coverage['error_rate']:.2%} |"
        )
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.extend(["", "Full gold-dev labels were hidden from the teacher; few-shot examples came from train only.", ""])
    markdown_path = root_path(args.markdown)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


def evaluate_pilot(args: argparse.Namespace) -> int:
    """Paired 8x10 dev comparison used when external full-dev labeling is unavailable."""
    manifest = load_manifest(args.manifest)
    iterations = manifest["defaults"]["bootstrap_iterations"]
    seed = manifest["defaults"]["bootstrap_seed"]
    report = {
        "scope": "paired 8x10 dev pilot; secondary evidence due external-upload restriction",
        "bootstrap_iterations": iterations, "bootstrap_seed": seed, "datasets": {},
    }
    markdown = [
        "# Paired 8x10 teacher-fidelity pilot", "",
        "Secondary evidence: only the first 10 gold-dev examples per dataset are compared.", "",
        "| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, spec in selected_datasets(manifest, args.datasets):
        gold = read_jsonl(spec["dev"])[:10]
        train = read_jsonl(spec["gold_train"])
        predictions = {}
        teacher_metrics = {}
        for teacher in ("deepseek", "qwen"):
            validation = validate_one(manifest, dataset, teacher, len(gold), 0.0, strict_model=True)
            if not validation["ok"]:
                raise SystemExit(f"Invalid pilot run {dataset}/{teacher}: {validation}")
            predictions[teacher] = read_jsonl(paths_for(manifest, dataset, teacher)["output"])
            metrics = evaluate_records(gold, predictions[teacher], train)
            metrics.pop("_row_counts")
            teacher_metrics[teacher] = metrics
        comparison = paired_bootstrap(gold, predictions["deepseek"], predictions["qwen"], iterations, seed)
        report["datasets"][dataset] = {"teachers": teacher_metrics, "deepseek_minus_qwen": comparison}
        low, high = comparison["ci95"]
        markdown.append(f"| {dataset} | {teacher_metrics['deepseek']['cF1']:.4f} | "
                        f"{teacher_metrics['qwen']['cF1']:.4f} | {comparison['delta_a_minus_b']:+.4f} | "
                        f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} |")
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.extend(["", f"Paired bootstrap: {iterations:,} resamples, seed={seed}; n=10 per dataset.", ""])
    markdown_path = root_path(args.markdown)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


def align_existing(gold: list[dict], silver: list[dict]) -> tuple[list[dict], dict]:
    """Align already-created silver to gold without making any external calls."""
    by_id = {}
    by_text = {}
    for row in silver:
        if row.get("_err"):
            continue
        by_id.setdefault(str(row.get("ID")), []).append(row)
        by_text.setdefault(normalize_text(row.get("Text", "")), []).append(row)
    aligned = []
    ambiguous = []
    methods = Counter()
    for gold_row in gold:
        key = normalize_text(gold_row.get("Text", ""))
        candidates = [row for row in by_id.get(str(gold_row.get("ID")), [])
                      if normalize_text(row.get("Text", "")) == key]
        method = "id_text"
        if not candidates:
            candidates = by_text.get(key, [])
            method = "normalized_text"
        if not candidates:
            continue
        unique = {}
        for row in candidates:
            quad_key = json.dumps(row.get("Quadruplet", []), ensure_ascii=False, sort_keys=True)
            unique[quad_key] = row
        if len(unique) != 1:
            ambiguous.append(str(gold_row.get("ID")))
            continue
        selected = next(iter(unique.values()))
        aligned.append({"ID": gold_row["ID"], "Text": gold_row["Text"],
                        "Quadruplet": selected.get("Quadruplet", [])})
        methods[method] += 1
    return aligned, {"aligned": len(aligned), "ambiguous_gold_ids": ambiguous, "methods": dict(methods)}


def retrospective(args: argparse.Namespace) -> int:
    """Secondary evidence from silver responses that already exist locally."""
    manifest = load_manifest(args.manifest)
    iterations = manifest["defaults"]["bootstrap_iterations"]
    seed = manifest["defaults"]["bootstrap_seed"]
    root = root_path(manifest["defaults"]["output_root"]) / "retrospective_fidelity"
    report = {
        "scope": "secondary retrospective evidence; existing silver aligned to gold test; no new API calls",
        "bootstrap_iterations": iterations, "bootstrap_seed": seed, "datasets": {},
    }
    markdown = [
        "# Retrospective teacher fidelity from existing silver", "",
        "Secondary evidence only: these predictions were already present locally and are aligned to gold test text.", "",
        "| Dataset | Common n | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, spec in selected_datasets(manifest, args.datasets):
        sources = spec.get("retrospective_silver", {})
        if not sources:
            continue
        gold_all = read_jsonl(spec["test"])
        if not gold_all or any("Quadruplet" not in row for row in gold_all):
            report["datasets"][dataset] = {
                "status": "unavailable", "reason": "test file has no gold Quadruplet labels",
                "gold": spec["test"], "gold_sha256": sha256_file(spec["test"]),
            }
            markdown.append(f"| {dataset} | — | — | — | — | — | unavailable: unlabeled test |")
            continue
        aligned = {}
        align_reports = {}
        for teacher, source in sources.items():
            aligned[teacher], align_reports[teacher] = align_existing(gold_all, read_jsonl(source))
            write_jsonl(root / f"{dataset}_{teacher}.jsonl", aligned[teacher])
            align_reports[teacher].update({"source": source, "source_sha256": sha256_file(source)})
        dataset_report = {"gold": spec["test"], "gold_sha256": sha256_file(spec["test"]),
                          "alignment": align_reports, "teachers": {}}
        if {"deepseek", "qwen"} <= set(aligned):
            ds_ids = {str(row["ID"]) for row in aligned["deepseek"]}
            qw_ids = {str(row["ID"]) for row in aligned["qwen"]}
            common_ids = ds_ids & qw_ids
            gold = [row for row in gold_all if str(row["ID"]) in common_ids]
            predictions = {teacher: [row for row in rows if str(row["ID"]) in common_ids]
                           for teacher, rows in aligned.items()}
            for teacher in ("deepseek", "qwen"):
                metrics = evaluate_records(gold, predictions[teacher], read_jsonl(spec["gold_train"]))
                metrics.pop("_row_counts")
                dataset_report["teachers"][teacher] = metrics
            comparison = paired_bootstrap(gold, predictions["deepseek"], predictions["qwen"], iterations, seed)
            dataset_report["deepseek_minus_qwen"] = comparison
            low, high = comparison["ci95"]
            markdown.append(f"| {dataset} | {len(gold)} | {dataset_report['teachers']['deepseek']['cF1']:.4f} | "
                            f"{dataset_report['teachers']['qwen']['cF1']:.4f} | "
                            f"{comparison['delta_a_minus_b']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                            f"{comparison['classification']} |")
        else:
            teacher = next(iter(aligned))
            gold_ids = {str(row["ID"]) for row in aligned[teacher]}
            gold = [row for row in gold_all if str(row["ID"]) in gold_ids]
            metrics = evaluate_records(gold, aligned[teacher], read_jsonl(spec["gold_train"]))
            metrics.pop("_row_counts")
            dataset_report["teachers"][teacher] = metrics
            markdown.append(f"| {dataset} | {len(gold)} | {metrics['cF1']:.4f} | — | — | — | DS-only |")
        report["datasets"][dataset] = dataset_report
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.extend(["", f"Paired bootstrap: {iterations:,} resamples, seed={seed}.", ""])
    markdown_path = root_path(args.markdown)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teacher", choices=["deepseek", "qwen"], required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--deepseek-workers", type=int, default=12)
    parser.add_argument("--qwen-model", default="unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    parser.add_argument("--qwen-base-url", default="http://localhost:8888/v1")
    parser.add_argument("--qwen-workers", type=int, default=4)
    parser.add_argument("--strict-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="scripts/kd/tallip.yaml")
    parser.add_argument("--env-file", default=".env.local")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--datasets", nargs="+", default=["all"])
    prepare_parser.set_defaults(func=prepare)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--datasets", nargs="+", default=["all"])
    add_runtime_arguments(run_parser)
    run_parser.set_defaults(func=run)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--datasets", nargs="+", default=["all"])
    validate_parser.add_argument("--teachers", nargs="+", choices=["deepseek", "qwen"], default=["deepseek", "qwen"])
    validate_parser.add_argument("--max-error-rate", type=float, default=0.02)
    validate_parser.add_argument("--strict-model", action="store_true")
    validate_parser.add_argument("--output", default="docs/audits/tallip_kd/teacher_fidelity_validation.json")
    validate_parser.set_defaults(func=validate)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--datasets", nargs="+", default=["all"])
    eval_parser.add_argument("--output", default="docs/audits/tallip_kd/teacher_fidelity_metrics.json")
    eval_parser.add_argument("--markdown", default="docs/audits/tallip_kd/teacher_fidelity_table.md")
    eval_parser.set_defaults(func=evaluate)
    single_parser = sub.add_parser("evaluate-single")
    single_parser.add_argument("--teacher", choices=["deepseek", "qwen"], required=True)
    single_parser.add_argument("--datasets", nargs="+", default=["all"])
    single_parser.add_argument("--output", default="docs/audits/tallip_kd/teacher_fidelity_single_metrics.json")
    single_parser.add_argument("--markdown", default="docs/audits/tallip_kd/teacher_fidelity_single_table.md")
    single_parser.set_defaults(func=evaluate_single)
    pilot_parser = sub.add_parser("evaluate-pilot")
    pilot_parser.add_argument("--datasets", nargs="+", default=["all"])
    pilot_parser.add_argument("--output", default="docs/audits/tallip_kd/teacher_fidelity_pilot_metrics.json")
    pilot_parser.add_argument("--markdown", default="docs/audits/tallip_kd/teacher_fidelity_pilot_table.md")
    pilot_parser.set_defaults(func=evaluate_pilot)
    retrospective_parser = sub.add_parser("retrospective")
    retrospective_parser.add_argument("--datasets", nargs="+", default=["all"])
    retrospective_parser.add_argument("--output", default="docs/audits/tallip_kd/retrospective_fidelity_metrics.json")
    retrospective_parser.add_argument("--markdown", default="docs/audits/tallip_kd/retrospective_fidelity_table.md")
    retrospective_parser.set_defaults(func=retrospective)
    args = parser.parse_args()
    load_env_file(args.env_file)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
