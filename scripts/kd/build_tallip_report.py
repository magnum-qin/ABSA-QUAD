#!/usr/bin/env python
"""Build or verify the frozen TALLIP KD tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tallip_common import ROOT, load_manifest, root_path


def load_optional(path: str) -> dict | None:
    resolved = root_path(path)
    return json.loads(resolved.read_text(encoding="utf-8")) if resolved.exists() else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def fmt(value, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def teacher_table(report: dict | None, pilot: dict | None, qwen_full: dict | None) -> list[str]:
    lines = [
        "## Table 1. Direct teacher fidelity on label-hidden gold dev", "",
    ]
    if report:
        lines.extend([
            "| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for dataset, item in report["datasets"].items():
            ds = item["teachers"]["deepseek"]["cF1"]
            qw = item["teachers"]["qwen"]["cF1"]
            comparison = item["deepseek_minus_qwen"]
            low, high = comparison["ci95"]
            lines.append(f"| {dataset} | {ds:.4f} | {qw:.4f} | {comparison['delta_a_minus_b']:+.4f} | "
                         f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} |")
        return lines
    lines.extend([
        "Full 8x2 pairing is unavailable: external DeepSeek upload was not run. The two panels below have different scope.", "",
        "### Table 1a. Qwen full-dev fidelity (unpaired)", "",
        "| Dataset | n | Qwen cF1 | Exact F1 | Aspect F1 | Opinion F1 | Category F1 | VA MAE | VA RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if qwen_full:
        for dataset, item in qwen_full["datasets"].items():
            metrics = item["metrics"]
            lines.append(f"| {dataset} | {metrics['coverage']['gold_rows']} | {metrics['cF1']:.4f} | "
                         f"{metrics['exact_f1']:.4f} | {metrics['aspect_span']['f1']:.4f} | "
                         f"{metrics['opinion_span']['f1']:.4f} | {metrics['category_span']['f1']:.4f} | "
                         f"{metrics['va_mae']:.4f} | {metrics['va_rmse']:.4f} |")
    else:
        lines.append("| _Qwen full run missing_ | — | — | — | — | — | — | — | — |")
    lines.extend([
        "", "### Table 1b. DeepSeek-Qwen paired pilot (first 10 examples per dataset)", "",
        "| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ])
    if pilot:
        for dataset, item in pilot["datasets"].items():
            comparison = item["deepseek_minus_qwen"]
            low, high = comparison["ci95"]
            lines.append(f"| {dataset} | {item['teachers']['deepseek']['cF1']:.4f} | "
                         f"{item['teachers']['qwen']['cF1']:.4f} | {comparison['delta_a_minus_b']:+.4f} | "
                         f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} |")
    else:
        lines.append("| _paired pilot missing_ | — | — | — | — | pending |")
    return lines


def inductive_tables(report: dict | None) -> list[str]:
    lines = [
        "## Table 2. Inductive-safe KD five-seed and ensemble results", "",
        "| Dataset | Training input | Seed scores | Mean±SD | Ensemble cF1 |",
        "|---|---|---|---:|---:|",
    ]
    comparisons = [
        "## Table 3. Paired dev comparisons under the frozen protocol", "",
        "| Dataset | Contrast | ΔcF1 | 95% CI | Decision |",
        "|---|---|---:|---:|---|",
    ]
    if not report:
        lines.append("| _pending clean training_ | — | — | — | — |")
        comparisons.append("| _pending clean inference_ | — | — | — | pending |")
        return lines + [""] + comparisons
    for dataset, item in report["datasets"].items():
        system = "clean" if item["systems"].get("clean", {}).get("status") == "complete" else "original"
        training_label = "cleaned" if system == "clean" else "original (zero dev/test overlap)"
        result = item["systems"].get(system, {})
        scores = ", ".join(f"{row['seed']}:{row['cF1']:.4f}" for row in result.get("single_seeds", [])) or "—"
        mean_sd = (f"{result['single_mean']:.4f}±{result['single_sd']:.4f}"
                   if "single_mean" in result else "—")
        ensemble = result.get("ensemble", {}).get("cF1")
        lines.append(f"| {dataset} | {training_label} | {scores} | {mean_sd} | {fmt(ensemble)} |")
        for contrast, comparison in item.get("comparisons", {}).items():
            contrast_label = {
                "clean_minus_baseline": "clean KD − baseline",
                "clean_minus_transductive": "clean KD − transductive KD",
                "original_minus_baseline": "inductive-safe KD − baseline",
            }.get(contrast, contrast)
            low, high = comparison["ci95"]
            comparisons.append(f"| {dataset} | {contrast_label} | {comparison['delta_a_minus_b']:+.4f} | "
                               f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} |")
    return lines + [""] + comparisons


def ablation_result(kind: str, row: dict, teacher: dict, tat: dict,
                    registry: dict, freeze: dict) -> str:
    if kind == "literal":
        return row["result"]
    if kind == "teacher_direct":
        datasets = teacher["datasets"]
        higher = sum(
            item["teachers"]["deepseek"]["cF1"] > item["teachers"]["qwen"]["cF1"]
            for item in datasets.values()
        )
        supported = [
            name for name, item in datasets.items()
            if item["deepseek_minus_qwen"]["classification"] == "supported"
        ]
        return (f"DeepSeek cF1 is higher in {higher}/{len(datasets)}; supported on "
                f"{', '.join(supported)}")
    if kind == "teacher_downstream_tat":
        comparison = tat["ensemble_deepseek_minus_qwen"]
        low, high = comparison["ci95"]
        return (f"DeepSeek {tat['deepseek']['ensemble_cF1']:.4f} vs Qwen "
                f"{tat['qwen']['ensemble_cF1']:.4f}; Δ={comparison['delta_a_minus_b']:+.4f}, "
                f"CI [{low:+.4f},{high:+.4f}]")
    experiments = registry["experiments"]
    if kind == "pool_rus":
        values = {
            "generated": experiments["rus_restaurant_deepseek_gen4k"]["result"],
            "translated": experiments["rus_restaurant_deepseek_trans4k"]["result"],
            "real": experiments["rus_restaurant_deepseek_real4k"]["result"],
        }
        return (f"generated ({values['generated']:.4f}) > translated "
                f"({values['translated']:.4f}) > real ({values['real']:.4f})")
    if kind == "pool_zho":
        real = experiments["zho_restaurant_deepseek_real"]["result"]
        generated = experiments["zho_restaurant_deepseek"]["result"]
        return f"real − generated = {(real - generated) * 100:+.1f}pp"
    if kind == "quantity_rus":
        performance = {
            item["system"]: item["dev_cf1"]
            for item in freeze["snapshot"]["datasets"]["performance"]
        }
        full = performance["gen8k full paired mean"]
        capped = performance["gen8k cap3000 paired mean"]
        registered = experiments["rus_restaurant_deepseek_gen8k_cap3k"]
        if abs(capped - registered["paired_mean"]) > 1e-9:
            raise ValueError("rus cap3000 paired mean disagrees between registry and freeze artifact")
        return f"{(capped - full) * 100:+.2f}pp; gate failed"
    raise ValueError(f"Unsupported ablation result_kind: {kind}")


def ablation_table(spec: dict, teacher: dict, tat: dict,
                   registry: dict, freeze: dict) -> list[str]:
    lines = [
        "## Table 4. Ablation claims and evidence grade", "",
        "| Axis | Controlled evidence | Result | Grade / allowed claim |",
        "|---|---|---|---|",
    ]
    seen = set()
    for row in spec["rows"]:
        if row["id"] in seen:
            raise ValueError(f"Duplicate ablation evidence id: {row['id']}")
        seen.add(row["id"])
        for source in row["sources"]:
            if not root_path(source).exists():
                raise FileNotFoundError(f"Missing ablation source: {source}")
        result = ablation_result(row["result_kind"], row, teacher, tat, registry, freeze)
        lines.append(
            f"| {row['axis']} | {row['controlled_evidence']} | {result} | "
            f"{row['grade_allowed_claim']} |"
        )
    return lines


def official_test_table(report: dict | None) -> list[str]:
    lines = [
        "## Table 5. One-shot official Task-3 test evaluation", "",
        "Post-competition local evaluation. Systems and prediction hashes were frozen from dev before gold retrieval.", "",
        "| Dataset | Frozen main system | Official test cF1 |",
        "|---|---|---:|",
    ]
    if not report:
        lines.append("| _official test not run_ | — | — |")
        return lines
    for dataset, item in report["datasets"].items():
        lines.append(f"| {dataset} | {item['main_system']} | {item['main_cF1']:.4f} |")
    lines.extend([
        "", "| Dataset | Pre-registered contrast | ΔcF1 | 95% CI | Decision |",
        "|---|---|---:|---:|---|",
    ])
    for item in report["comparisons"]:
        low, high = item["ci95"]
        lines.append(
            f"| {item['dataset']} | {item['a']} − {item['b']} | "
            f"{item['delta_a_minus_b']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{item['classification']} |"
        )
    return lines


def build_report(args: argparse.Namespace) -> tuple[str, list[Path]]:
    load_manifest()  # fail early if the frozen protocol is invalid YAML
    teacher = load_optional(args.teacher)
    teacher_pilot = load_optional(args.teacher_pilot)
    qwen_full = load_optional(args.qwen_full)
    inductive = load_optional(args.inductive)
    official_test = load_optional(args.official_test)
    ablation = load_optional(args.ablation)
    tat = load_optional(args.tat_matched)
    freeze = load_optional(args.freeze_artifact)
    registry = yaml.safe_load(root_path(args.registry).read_text(encoding="utf-8"))
    if not all((teacher, inductive, official_test, ablation, tat, freeze, registry)):
        raise RuntimeError("Frozen core tables require all final evidence artifacts")
    lines = [
        "# TALLIP KD core evidence tables", "",
        "**Status: frozen on 2026-07-18.** Tables 1–4 use development-set evidence; "
        "Table 5 is a one-shot post-competition evaluation on released official test labels. "
        "Run `python scripts/kd/build_tallip_report.py --verify` before citation.", "",
        *teacher_table(teacher, teacher_pilot, qwen_full), "", *inductive_tables(inductive), "",
        *ablation_table(ablation, teacher, tat, registry, freeze), "",
        *official_test_table(official_test), "",
    ]
    input_names = [
        path for path in (
            "scripts/kd/tallip.yaml",
            "scripts/kd/official_test.yaml",
            args.teacher,
            args.inductive,
            args.official_test,
            args.ablation,
            args.tat_matched,
            args.registry,
            args.freeze_artifact,
        )
    ]
    input_names.extend(
        source for row in ablation["rows"] for source in row["sources"]
    )
    inputs = []
    seen_inputs = set()
    for name in input_names:
        path = root_path(name)
        resolved = path.resolve()
        if resolved not in seen_inputs:
            inputs.append(path)
            seen_inputs.add(resolved)
    return "\n".join(lines), inputs


def update_lock(args: argparse.Namespace, report: str, inputs: list[Path]) -> None:
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    generator = Path(__file__).resolve()
    locked_at = args.locked_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock = {
        "schema_version": 1,
        "status": "frozen",
        "locked_at": locked_at,
        "table_count": 5,
        "output": {"path": relative(output), "sha256": sha256_file(output)},
        "generator": {"path": relative(generator), "sha256": sha256_file(generator)},
        "inputs": [
            {"path": relative(path), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "verification_command": "python scripts/kd/build_tallip_report.py --verify",
        "update_policy": "Do not update after test-gold inspection without a new protocol version and explicit evidence audit."
    }
    lock_path = root_path(args.lock)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {relative(output)} and {relative(lock_path)}")


def verify_lock(args: argparse.Namespace, report: str) -> None:
    lock_path = root_path(args.lock)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing core-table lock: {relative(lock_path)}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen" or lock.get("table_count") != 5:
        raise ValueError("Core-table lock is not a five-table frozen manifest")
    checks = [lock["generator"], *lock["inputs"], lock["output"]]
    errors = []
    for item in checks:
        path = root_path(item["path"])
        actual = sha256_file(path) if path.exists() else "missing"
        if actual != item["sha256"]:
            errors.append(f"{item['path']}: expected {item['sha256']}, got {actual}")
    rendered_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
    if rendered_hash != lock["output"]["sha256"]:
        errors.append(
            f"rendered report: expected {lock['output']['sha256']}, got {rendered_hash}"
        )
    if errors:
        raise RuntimeError("Frozen TALLIP core-table verification failed:\n" + "\n".join(errors))
    print(f"verified {lock['table_count']} frozen tables ({lock['output']['sha256'][:12]})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="docs/audits/tallip_kd/teacher_fidelity_metrics.json")
    parser.add_argument("--teacher-pilot", default="docs/audits/tallip_kd/teacher_fidelity_pilot_metrics.json")
    parser.add_argument("--qwen-full", default="docs/audits/tallip_kd/qwen_full_fidelity_metrics.json")
    parser.add_argument("--inductive", default="docs/audits/tallip_kd/inductive_metrics.json")
    parser.add_argument("--official-test", default="docs/audits/tallip_kd/official_test/official_test_metrics.json")
    parser.add_argument("--ablation", default="docs/audits/tallip_kd/ablation_evidence.json")
    parser.add_argument("--tat-matched", default="docs/audits/tallip_kd/tat_teacher_matched_metrics.json")
    parser.add_argument("--registry", default="scripts/kd/registry.yaml")
    parser.add_argument("--freeze-artifact", default="docs/audits/kd_freeze_report/artifact.json")
    parser.add_argument("--output", default="docs/audits/tallip_kd/tallip_core_tables.md")
    parser.add_argument("--lock", default="docs/audits/tallip_kd/tallip_core_tables.lock.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="verify hashes and deterministic rendering (default)")
    mode.add_argument("--update-lock", action="store_true", help="explicitly rewrite the tables and lock manifest")
    parser.add_argument("--locked-at", help="timestamp recorded by --update-lock")
    args = parser.parse_args()
    report, inputs = build_report(args)
    if args.update_lock:
        update_lock(args, report, inputs)
    else:
        verify_lock(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
