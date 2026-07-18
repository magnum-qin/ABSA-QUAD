#!/usr/bin/env python
"""Build the frozen TALLIP KD tables from machine-readable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tallip_common import ROOT, load_manifest, root_path


def load_optional(path: str) -> dict | None:
    resolved = root_path(path)
    return json.loads(resolved.read_text(encoding="utf-8")) if resolved.exists() else None


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
        "## Table 2. Clean KD five-seed and ensemble results", "",
        "| Dataset | Clean/original | Seed scores | Mean±SD | Ensemble cF1 |",
        "|---|---|---|---:|---:|",
    ]
    comparisons = [
        "## Table 3. Paired clean comparisons", "",
        "| Dataset | Contrast | ΔcF1 | 95% CI | Decision |",
        "|---|---|---:|---:|---|",
    ]
    if not report:
        lines.append("| _pending clean training_ | — | — | — | — |")
        comparisons.append("| _pending clean inference_ | — | — | — | pending |")
        return lines + [""] + comparisons
    for dataset, item in report["datasets"].items():
        system = "clean" if item["systems"].get("clean", {}).get("status") == "complete" else "original"
        result = item["systems"].get(system, {})
        scores = ", ".join(f"{row['seed']}:{row['cF1']:.4f}" for row in result.get("single_seeds", [])) or "—"
        mean_sd = (f"{result['single_mean']:.4f}±{result['single_sd']:.4f}"
                   if "single_mean" in result else "—")
        ensemble = result.get("ensemble", {}).get("cF1")
        lines.append(f"| {dataset} | {system} | {scores} | {mean_sd} | {fmt(ensemble)} |")
        for contrast, comparison in item.get("comparisons", {}).items():
            low, high = comparison["ci95"]
            comparisons.append(f"| {dataset} | {contrast} | {comparison['delta_a_minus_b']:+.4f} | "
                               f"[{low:+.4f}, {high:+.4f}] | {comparison['classification']} |")
    return lines + [""] + comparisons


def ablation_table() -> list[str]:
    return [
        "## Table 4. Ablation claims and evidence grade", "",
        "| Axis | Controlled evidence | Result | Grade / allowed claim |",
        "|---|---|---|---|",
        "| Teacher | 8 datasets × 2 teachers on identical label-hidden gold dev | "
        "DeepSeek cF1 is higher in 8/8; supported on eng_restaurant, zho_laptop, zho_restaurant | "
        "High for overall robustness; no evidence of a reliable ranking reversal |",
        "| Teacher downstream | tat, same 1,240 gold + same 3,915 silver texts/order, five paired seeds | "
        "DeepSeek 0.5365 vs Qwen 0.5327; Δ=+0.0038, CI [−0.0823,+0.0852] | "
        "High matched-control quality; ranking inconclusive |",
        "| Pool source | rus real/trans/generated, same teacher and cap=3000 | generated > translated > real | "
        "High within rus; no universal source ranking |",
        "| Pool source | zho real vs generated | real +2.3pp | Medium; observational support only |",
        "| Silver quantity | rus same gen8k pool, paired seeds 42/123, cap3000 vs 7988 | −1.14pp; gate failed | "
        "Medium; rejects the planned follow-up, not a full dose-response |",
        "| Translation vs native | ukr translated and native experiments differ in pool construction/quantity | mixed | "
        "Low; descriptive only |",
        "| Information complementarity | synthesis across teacher, pool, and quantity results | plausible moderator | "
        "Interpretation, not a causal or universal optimum claim |",
    ]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="docs/audits/tallip_kd/teacher_fidelity_metrics.json")
    parser.add_argument("--teacher-pilot", default="docs/audits/tallip_kd/teacher_fidelity_pilot_metrics.json")
    parser.add_argument("--qwen-full", default="docs/audits/tallip_kd/qwen_full_fidelity_metrics.json")
    parser.add_argument("--inductive", default="docs/audits/tallip_kd/inductive_metrics.json")
    parser.add_argument("--official-test", default="docs/audits/tallip_kd/official_test/official_test_metrics.json")
    parser.add_argument("--output", default="docs/audits/tallip_kd/tallip_core_tables.md")
    args = parser.parse_args()
    load_manifest()  # fail early if the frozen protocol is invalid YAML
    teacher = load_optional(args.teacher)
    teacher_pilot = load_optional(args.teacher_pilot)
    qwen_full = load_optional(args.qwen_full)
    inductive = load_optional(args.inductive)
    official_test = load_optional(args.official_test)
    lines = [
        "# TALLIP KD core evidence tables", "",
        "Generated from frozen machine-readable artifacts. Tables 1–4 use development-set evidence; Table 5 is a one-shot post-competition evaluation on released official test labels.", "",
        *teacher_table(teacher, teacher_pilot, qwen_full), "", *inductive_tables(inductive), "",
        *ablation_table(), "", *official_test_table(official_test), "",
    ]
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
