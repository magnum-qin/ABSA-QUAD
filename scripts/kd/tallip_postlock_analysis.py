#!/usr/bin/env python
"""Build the final post-lock evidence closure without using official test for selection.

Outputs a merged five-dataset control report, direct complementarity/error
measurements, family-wise bootstrap multiplicity diagnostics, and an auditable
cost/efficiency inventory. All model-development analyses use dev only. The
released official test is read solely to add post-hoc multiplicity diagnostics
to the already frozen one-shot comparisons.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from tallip_common import (
    ROOT,
    evaluate_records,
    metrics_from_counts,
    official_row_counts,
    read_jsonl,
    sum_counts,
)


SEED = 20260713
ITERATIONS = 10000
ADOPTED = (
    "eng_laptop",
    "rus_restaurant",
    "tat_restaurant",
    "ukr_restaurant",
    "zho_restaurant",
)
TEACHERS = {
    "eng_laptop": "deepseek",
    "rus_restaurant": "deepseek",
    "tat_restaurant": "qwen",
    "ukr_restaurant": "deepseek",
    "zho_restaurant": "deepseek",
}
OUTPUT_ROOT = ROOT / "docs/audits/tallip_kd"


def load_json(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_span(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.strip().casefold().split())


def category_counts(rows: list[dict]) -> Counter:
    return Counter(
        normalize_span(quad.get("Category"))
        for row in rows
        for quad in (row.get("Quadruplet") or [])
        if normalize_span(quad.get("Category"))
    )


def span_set(rows: list[dict], field: str) -> set[str]:
    return {
        value
        for row in rows
        for quad in (row.get("Quadruplet") or [])
        if (value := normalize_span(quad.get(field))) and value != "null"
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def js_divergence(left: Counter, right: Counter) -> float | None:
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not keys or not left_total or not right_total:
        return None
    result = 0.0
    # Stable accumulation order is required for byte-identical frozen JSON
    # across Python processes with different hash randomization seeds.
    for key in sorted(keys):
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def extract_silver(base: list[dict], llm_train: list[dict]) -> list[dict]:
    base_by_id = {str(row["ID"]): row for row in base}
    if len(base_by_id) != len(base):
        raise ValueError("base IDs are not unique")
    seen = set()
    silver = []
    for row in llm_train:
        row_id = str(row["ID"])
        if row_id in base_by_id:
            if row.get("Text") != base_by_id[row_id].get("Text"):
                raise ValueError(f"base text mismatch for ID={row_id}")
            seen.add(row_id)
        else:
            silver.append(row)
    if seen != set(base_by_id):
        raise ValueError(f"LLM train misses {len(set(base_by_id) - seen)} base rows")
    if len(silver) != len(llm_train) - len(base):
        raise ValueError("base/silver separation changed the intended grain")
    return silver


def prediction_map(rows: list[dict]) -> dict[str, dict]:
    result = {str(row["ID"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("prediction IDs are not unique")
    return result


def row_cf1(gold: dict, pred: dict) -> float:
    counts = official_row_counts(
        gold.get("Quadruplet") or [], pred.get("Quadruplet") or []
    )
    return metrics_from_counts(counts)["cF1"]


def slice_metrics(
    gold: list[dict], pred_a: list[dict], pred_b: list[dict], indices: list[int]
) -> dict:
    if not indices:
        return {"rows": 0, "llm_kd_cF1": None, "baseline_cF1": None, "delta": None}
    selected_gold = [gold[index] for index in indices]
    ids = {str(row["ID"]) for row in selected_gold}
    selected_a = [row for row in pred_a if str(row["ID"]) in ids]
    selected_b = [row for row in pred_b if str(row["ID"]) in ids]
    a = evaluate_records(selected_gold, selected_a)["cF1"]
    b = evaluate_records(selected_gold, selected_b)["cF1"]
    return {"rows": len(indices), "llm_kd_cF1": a, "baseline_cF1": b, "delta": a - b}


def build_complementarity() -> dict:
    v1 = load_yaml("scripts/kd/postlock_v1.yaml")
    v2 = load_yaml("scripts/kd/postlock_v2.yaml")
    tallip = load_yaml("scripts/kd/tallip.yaml")
    inductive = load_json("docs/audits/tallip_kd/inductive_metrics.json")
    fidelity = load_json("docs/audits/tallip_kd/teacher_fidelity_metrics.json")
    self_specs = {**v1["self_training"], **v2["self_training"]}
    datasets = {}
    for dataset in ADOPTED:
        spec = self_specs[dataset]
        base = read_jsonl(spec["base_train"])
        llm_train = read_jsonl(spec["llm_reference_train"])
        silver = extract_silver(base, llm_train)
        dev_path = v1["gold_repeat"].get(dataset, v2["gold_repeat"].get(dataset))["dev"]
        dev = read_jsonl(dev_path)
        gold_train = read_jsonl(tallip["datasets"][dataset]["gold_train"])

        systems = inductive["datasets"][dataset]["systems"]
        llm_key = "clean" if systems.get("clean", {}).get("status") == "complete" else "original"
        llm_path = systems[llm_key]["source"]
        baseline_path = systems["baseline"]["source"]
        llm_pred = read_jsonl(llm_path)
        baseline_pred = read_jsonl(baseline_path)
        gold_ids = [str(row["ID"]) for row in dev]
        if [str(row["ID"]) for row in llm_pred] != gold_ids:
            raise ValueError(f"{dataset}: LLM prediction order differs from dev")
        if [str(row["ID"]) for row in baseline_pred] != gold_ids:
            raise ValueError(f"{dataset}: baseline prediction order differs from dev")

        base_categories = category_counts(base)
        silver_categories = category_counts(silver)
        dev_categories = category_counts(dev)
        rare_categories = {
            name for name in dev_categories if base_categories[name] <= 5
        }
        dev_rare_quads = sum(dev_categories[name] for name in rare_categories)
        silver_covered_rare_quads = sum(
            dev_categories[name] for name in rare_categories if silver_categories[name]
        )
        novel_silver_categories = set(silver_categories) - set(base_categories)
        novel_relevant = novel_silver_categories & set(dev_categories)

        allowed_categories = set(category_counts(gold_train))
        silver_quads = [
            (row, quad)
            for row in silver
            for quad in (row.get("Quadruplet") or [])
        ]
        invalid_aspects = invalid_opinions = invalid_va = invalid_categories = 0
        for row, quad in silver_quads:
            raw_text = str(row.get("Text", ""))
            normalized_text = normalize_span(raw_text)
            aspect = normalize_span(quad.get("Aspect"))
            opinion = normalize_span(quad.get("Opinion"))
            raw_aspect = str(quad.get("Aspect", ""))
            raw_opinion = str(quad.get("Opinion", ""))
            if (
                aspect != "null"
                and raw_aspect not in raw_text
                and aspect not in normalized_text
            ):
                invalid_aspects += 1
            if (
                opinion != "null"
                and raw_opinion not in raw_text
                and opinion not in normalized_text
            ):
                invalid_opinions += 1
            if normalize_span(quad.get("Category")) not in allowed_categories:
                invalid_categories += 1
            try:
                valence, arousal = (
                    float(value) for value in str(quad.get("VA", "")).split("#")
                )
                if not (1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0):
                    invalid_va += 1
            except (TypeError, ValueError):
                invalid_va += 1

        base_aspects, silver_aspects, dev_aspects = (
            span_set(rows, "Aspect") for rows in (base, silver, dev)
        )
        base_opinions, silver_opinions, dev_opinions = (
            span_set(rows, "Opinion") for rows in (base, silver, dev)
        )

        llm_by_id, baseline_by_id = prediction_map(llm_pred), prediction_map(baseline_pred)
        repaired, harmed, tied = [], [], []
        for row in dev:
            row_id = str(row["ID"])
            delta = row_cf1(row, llm_by_id[row_id]) - row_cf1(row, baseline_by_id[row_id])
            target = repaired if delta > 1e-12 else harmed if delta < -1e-12 else tied
            target.append(row_id)

        implicit = [
            index for index, row in enumerate(dev)
            if any(
                normalize_span(quad.get("Aspect")) == "null"
                or normalize_span(quad.get("Opinion")) == "null"
                for quad in (row.get("Quadruplet") or [])
            )
        ]
        rare = [
            index for index, row in enumerate(dev)
            if any(
                normalize_span(quad.get("Category")) in rare_categories
                for quad in (row.get("Quadruplet") or [])
            )
        ]
        multi = [
            index for index, row in enumerate(dev)
            if len(row.get("Quadruplet") or []) >= 2
        ]
        teacher = TEACHERS[dataset]
        teacher_cf1 = fidelity["datasets"][dataset]["teachers"][teacher]["cF1"]
        datasets[dataset] = {
            "sources": {
                "base_train": spec["base_train"],
                "base_sha256": sha256(spec["base_train"]),
                "llm_train": spec["llm_reference_train"],
                "llm_train_sha256": sha256(spec["llm_reference_train"]),
                "dev": dev_path,
                "dev_sha256": sha256(dev_path),
                "llm_prediction": llm_path,
                "llm_prediction_sha256": sha256(llm_path),
                "baseline_prediction": baseline_path,
                "baseline_prediction_sha256": sha256(baseline_path),
            },
            "grain": {
                "base_rows": len(base),
                "silver_rows": len(silver),
                "llm_train_rows": len(llm_train),
                "dev_rows": len(dev),
            },
            "teacher": teacher,
            "teacher_fidelity_cF1": teacher_cf1,
            "silver_label_validity": {
                "quadruplets": len(silver_quads),
                "empty_rows": sum(not (row.get("Quadruplet") or []) for row in silver),
                "invalid_explicit_aspect_substrings": invalid_aspects,
                "invalid_explicit_opinion_substrings": invalid_opinions,
                "categories_absent_from_gold_train_inventory": invalid_categories,
                "invalid_va_format_or_range": invalid_va,
            },
            "category_complementarity": {
                "base_unique_categories": len(base_categories),
                "silver_unique_categories": len(silver_categories),
                "dev_unique_categories": len(dev_categories),
                "dev_categories_covered_by_base": len(set(dev_categories) & set(base_categories)),
                "dev_categories_covered_by_silver": len(set(dev_categories) & set(silver_categories)),
                "silver_categories_absent_from_base": len(novel_silver_categories),
                "novel_silver_categories_present_in_dev": len(novel_relevant),
                "rare_category_definition": "base frequency <= 5 quadruplets",
                "dev_rare_quadruplets": dev_rare_quads,
                "dev_rare_quadruplets_with_category_in_silver": silver_covered_rare_quads,
                "rare_dev_quadruplet_category_coverage": ratio(
                    silver_covered_rare_quads, dev_rare_quads
                ),
                "base_to_dev_js_divergence": js_divergence(base_categories, dev_categories),
                "silver_to_dev_js_divergence": js_divergence(silver_categories, dev_categories),
            },
            "lexical_complementarity": {
                "silver_aspect_types": len(silver_aspects),
                "silver_aspect_novel_vs_base": len(silver_aspects - base_aspects),
                "silver_aspect_novelty_rate": ratio(
                    len(silver_aspects - base_aspects), len(silver_aspects)
                ),
                "silver_opinion_types": len(silver_opinions),
                "silver_opinion_novel_vs_base": len(silver_opinions - base_opinions),
                "silver_opinion_novelty_rate": ratio(
                    len(silver_opinions - base_opinions), len(silver_opinions)
                ),
                "dev_aspect_types_covered_by_base": len(dev_aspects & base_aspects),
                "dev_aspect_types_covered_by_silver": len(dev_aspects & silver_aspects),
                "dev_opinion_types_covered_by_base": len(dev_opinions & base_opinions),
                "dev_opinion_types_covered_by_silver": len(dev_opinions & silver_opinions),
            },
            "error_transition": {
                "definition": "row-level official cF1: LLM-KD higher/lower/equal than baseline",
                "repaired_rows": len(repaired),
                "harmed_rows": len(harmed),
                "tied_rows": len(tied),
                "net_repaired_minus_harmed": len(repaired) - len(harmed),
                "example_repaired_ids": repaired[:10],
                "example_harmed_ids": harmed[:10],
            },
            "dev_slices": {
                "all": slice_metrics(dev, llm_pred, baseline_pred, list(range(len(dev)))),
                "implicit_aspect_or_opinion": slice_metrics(dev, llm_pred, baseline_pred, implicit),
                "rare_category": slice_metrics(dev, llm_pred, baseline_pred, rare),
                "multi_quadruplet": slice_metrics(dev, llm_pred, baseline_pred, multi),
            },
        }
    span_coverage_delta = {
        dataset: (
            item["lexical_complementarity"]["dev_aspect_types_covered_by_silver"]
            - item["lexical_complementarity"]["dev_aspect_types_covered_by_base"]
            + item["lexical_complementarity"]["dev_opinion_types_covered_by_silver"]
            - item["lexical_complementarity"]["dev_opinion_types_covered_by_base"]
        )
        for dataset, item in datasets.items()
    }
    category_alignment_gain = {
        dataset: (
            item["category_complementarity"]["base_to_dev_js_divergence"]
            - item["category_complementarity"]["silver_to_dev_js_divergence"]
        )
        for dataset, item in datasets.items()
    }
    dev_gain = {
        dataset: item["dev_slices"]["all"]["delta"]
        for dataset, item in datasets.items()
    }
    fidelity = {
        dataset: item["teacher_fidelity_cF1"]
        for dataset, item in datasets.items()
    }
    cross_domain_summary = {
        "largest_kd_dev_gain": max(dev_gain, key=dev_gain.get),
        "smallest_kd_dev_gain": min(dev_gain, key=dev_gain.get),
        "largest_combined_dev_span_coverage_gain": max(
            span_coverage_delta, key=span_coverage_delta.get
        ),
        "highest_teacher_fidelity": max(fidelity, key=fidelity.get),
        "silver_category_distribution_closer_to_dev_than_base": [
            dataset for dataset, value in category_alignment_gain.items() if value > 0
        ],
        "silver_improves_both_dev_aspect_and_opinion_type_coverage": [
            dataset
            for dataset, item in datasets.items()
            if (
                item["lexical_complementarity"]["dev_aspect_types_covered_by_silver"]
                > item["lexical_complementarity"]["dev_aspect_types_covered_by_base"]
                and item["lexical_complementarity"]["dev_opinion_types_covered_by_silver"]
                > item["lexical_complementarity"]["dev_opinion_types_covered_by_base"]
            )
        ],
        "allowed_interpretation": (
            "The measured fidelity, distribution, and lexical-coverage descriptors "
            "are non-monotonic across five domains. They support a multi-factor "
            "quality/coverage/complementarity interpretation, not a single causal "
            "score or universal ordering."
        ),
    }
    return {
        "schema_version": 1,
        "as_of": "2026-08-13",
        "scope": "five adopted KD datasets; dev-only model comparisons",
        "interpretation_limit": (
            "Coverage and error transitions are direct descriptive measurements; "
            "with five domains they do not establish a universal causal moderator."
        ),
        "cross_domain_summary": cross_domain_summary,
        "datasets": datasets,
    }


def bootstrap_inference(gold: list[dict], pred_a: list[dict], pred_b: list[dict]) -> dict:
    a = evaluate_records(gold, pred_a)["_row_counts"]
    b = evaluate_records(gold, pred_b)["_row_counts"]
    observed_a = metrics_from_counts(sum_counts(a))["cF1"]
    observed_b = metrics_from_counts(sum_counts(b))["cF1"]
    observed = observed_a - observed_b
    rng = random.Random(SEED)
    n = len(gold)
    deltas = []
    for _ in range(ITERATIONS):
        indices = [rng.randrange(n) for _ in range(n)]
        score_a = metrics_from_counts(sum_counts(a[index] for index in indices))["cF1"]
        score_b = metrics_from_counts(sum_counts(b[index] for index in indices))["cF1"]
        deltas.append(score_a - score_b)
    ordered = sorted(deltas)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    null_extreme = sum(abs(delta - observed) >= abs(observed) for delta in deltas)
    p_value = (null_extreme + 1) / (ITERATIONS + 1)
    low, high = percentile(0.025), percentile(0.975)
    return {
        "score_a": observed_a,
        "score_b": observed_b,
        "delta_a_minus_b": observed,
        "ci95": [low, high],
        "bootstrap_p_two_sided": p_value,
        "classification": "supported" if low > 0 else "harmful" if high < 0 else "inconclusive",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def bh_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__, reverse=True)
    adjusted = [0.0] * len(p_values)
    running = 1.0
    m = len(p_values)
    for reverse_rank, index in enumerate(order):
        rank = m - reverse_rank
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def add_adjustments(items: list[dict]) -> None:
    p_values = [item["bootstrap_p_two_sided"] for item in items]
    for item, holm, bh in zip(items, holm_adjust(p_values), bh_adjust(p_values)):
        item["holm_adjusted_p"] = holm
        item["bh_adjusted_p"] = bh
        item["holm_reject_0.05"] = holm < 0.05
        item["bh_reject_0.05"] = bh < 0.05


def comparison_item(
    item_id: str,
    dataset: str,
    gold_path: str,
    a_path: str,
    b_path: str,
) -> dict:
    result = bootstrap_inference(
        read_jsonl(gold_path), read_jsonl(a_path), read_jsonl(b_path)
    )
    return {
        "id": item_id,
        "dataset": dataset,
        "gold": gold_path,
        "prediction_a": a_path,
        "prediction_b": b_path,
        **result,
    }


def build_statistics(complete: dict) -> dict:
    tallip = load_yaml("scripts/kd/tallip.yaml")
    inductive = load_json("docs/audits/tallip_kd/inductive_metrics.json")
    official_manifest = load_yaml("scripts/kd/official_test.yaml")
    official = load_json("docs/audits/tallip_kd/official_test/official_test_metrics.json")
    families: dict[str, list[dict]] = {}

    teacher = []
    for dataset, spec in tallip["datasets"].items():
        teacher.append(
            comparison_item(
                f"{dataset}:deepseek_minus_qwen",
                dataset,
                spec["dev"],
                f"data/audits/tallip_kd/teacher_fidelity/validated/{dataset}_deepseek.jsonl",
                f"data/audits/tallip_kd/teacher_fidelity/validated/{dataset}_qwen.jsonl",
            )
        )
    families["teacher_fidelity_8x2"] = teacher

    dev_kd = []
    for dataset in ADOPTED:
        spec = inductive["datasets"][dataset]
        systems = spec["systems"]
        llm_key = "clean" if systems.get("clean", {}).get("status") == "complete" else "original"
        dev_kd.append(
            comparison_item(
                f"{dataset}:llm_kd_minus_baseline",
                dataset,
                spec["dev"],
                systems[llm_key]["source"],
                systems["baseline"]["source"],
            )
        )
    families["adopted_kd_vs_baseline_dev"] = dev_kd

    exposure, self_training = [], []
    for dataset in ADOPTED:
        spec = complete["datasets"][dataset]
        exposure.append(
            comparison_item(
                f"{dataset}:llm_kd_minus_gold_repeat",
                dataset,
                spec["dev"],
                spec["llm_kd_prediction"],
                spec["systems"]["gold_repeat"]["prediction"],
            )
        )
        self_training.append(
            comparison_item(
                f"{dataset}:llm_kd_minus_self_training",
                dataset,
                spec["dev"],
                spec["llm_kd_prediction"],
                spec["systems"]["self_training"]["prediction"],
            )
        )
    families["exposure_control_dev"] = exposure
    families["same_text_self_training_dev"] = self_training

    frozen_test = []
    for comparison in official["comparisons"]:
        if comparison["role"] != "adopted_kd_vs_baseline":
            continue
        dataset = comparison["dataset"]
        spec = official["datasets"][dataset]
        gold_path = str(
            Path(official_manifest["protocol"]["gold_root"])
            / official_manifest["datasets"][dataset]["gold"]
        )
        frozen_test.append(
            comparison_item(
                f"{dataset}:official_frozen_{comparison['a']}_minus_{comparison['b']}",
                dataset,
                gold_path,
                spec["systems"][comparison["a"]]["prediction"],
                spec["systems"][comparison["b"]]["prediction"],
            )
        )
    families["frozen_official_test_posthoc"] = frozen_test

    for items in families.values():
        add_adjustments(items)
    return {
        "schema_version": 1,
        "as_of": "2026-08-13",
        "method": {
            "resampling": "paired example-level bootstrap",
            "iterations": ITERATIONS,
            "seed": SEED,
            "p_value": (
                "two-sided centered-bootstrap tail probability with +1 correction"
            ),
            "families": "Holm FWER and Benjamini-Hochberg FDR applied separately per named family",
        },
        "scope_caveat": (
            "Multiplicity diagnostics are post-hoc robustness analyses. They do not "
            "alter frozen systems or authorize official-test selection."
        ),
        "families": families,
    }


LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - INFO -")
PARAMETERS = re.compile(r"Total parameters: ([\d,]+)")


def observed_training_seconds(path: Path) -> tuple[float, int | None]:
    if not path.exists():
        return 0.0, None
    total = 0.0
    started = None
    latest = None
    parameters = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        timestamp = LOG_TIMESTAMP.match(line)
        if timestamp:
            latest = datetime.strptime(timestamp.group(1), "%Y-%m-%d %H:%M:%S,%f")
            if "Using device:" in line:
                if started is not None and latest is not None:
                    total += max(0.0, (latest - started).total_seconds())
                started = latest
        match = PARAMETERS.search(line)
        if match and parameters is None:
            parameters = int(match.group(1).replace(",", ""))
        if "Training completed!" in line and started is not None and latest is not None:
            total += max(0.0, (latest - started).total_seconds())
            started = None
    if started is not None and latest is not None:
        total += max(0.0, (latest - started).total_seconds())
    return total, parameters


def build_cost_efficiency(complementarity: dict) -> dict:
    v1 = load_yaml("scripts/kd/postlock_v1.yaml")
    v2 = load_yaml("scripts/kd/postlock_v2.yaml")
    controls = []
    for version, manifest in (("v1", v1), ("v2", v2)):
        root = ROOT / manifest["protocol"]["output_root"] / "logs"
        for system_key, suffix in (("gold_repeat", "gold_repeat"), ("self_training", "self_training")):
            for dataset, spec in manifest[system_key].items():
                if version == "v2" and system_key == "gold_repeat":
                    continue
                log = root / f"{dataset}_{suffix}_train.log"
                seconds, parameters = observed_training_seconds(log)
                config = load_yaml(spec["config"])
                best_paths = [
                    ROOT / "checkpoints" / f"{spec['checkpoint_prefix']}{seed}"
                    / "checkpoint_best.pt"
                    for seed in manifest["protocol"]["seeds"]
                ]
                controls.append(
                    {
                        "protocol": manifest["protocol"]["name"],
                        "dataset": dataset,
                        "system": system_key,
                        "seeds": len(manifest["protocol"]["seeds"]),
                        "encoder": config["model"]["encoder_name"],
                        "parameters": parameters,
                        "batch_size": config["training"]["batch_size"],
                        "max_epochs": config["training"]["num_epochs"],
                        "checkpoint_storage": (
                            "local NVMe staging then archive"
                            if version == "v2"
                            else "workspace filesystem"
                        ),
                        "observed_run_wall_hours": seconds / 3600,
                        "retained_best_checkpoint_bytes": sum(
                            path.stat().st_size for path in best_paths if path.exists()
                        ),
                        "best_checkpoints_found": sum(path.exists() for path in best_paths),
                        "log": str(log.relative_to(ROOT)),
                    }
                )

    provider_totals = {}
    for provider in ("deepseek", "qwen"):
        validated_rows = timed_processed_rows = errors = 0
        wall_seconds = 0.0
        requested_models = set()
        for dataset in load_yaml("scripts/kd/tallip.yaml")["datasets"]:
            path = ROOT / f"data/audits/tallip_kd/teacher_fidelity/metadata/{dataset}_{provider}.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            timed_processed_rows += metadata["processed"]
            errors += metadata["errors"]
            requested_models.add(metadata["requested_model"])
            validated = (
                ROOT
                / "data/audits/tallip_kd/teacher_fidelity/validated"
                / f"{dataset}_{provider}.jsonl"
            )
            validated_rows += sum(1 for _ in validated.open(encoding="utf-8"))
            start = datetime.fromisoformat(metadata["started_at"])
            end = datetime.fromisoformat(metadata["completed_at"])
            wall_seconds += (end - start).total_seconds()
        provider_totals[provider] = {
            "requested_models": sorted(requested_models),
            "annotated_rows": validated_rows,
            "timed_processed_rows": timed_processed_rows,
            "errors": errors,
            "summed_dataset_wall_seconds": wall_seconds,
            "effective_timed_rows_per_second": (
                timed_processed_rows / wall_seconds if wall_seconds else None
            ),
            "token_usage": None,
            "monetary_cost": None,
            "cost_caveat": (
                "historical raw responses did not retain provider token-usage fields; "
                "throughput uses only rows represented by the retained timing metadata"
            ),
        }

    adopted_silver = {
        dataset: {
            "teacher": TEACHERS[dataset],
            "silver_rows": item["grain"]["silver_rows"],
            "dev_delta_cF1": item["dev_slices"]["all"]["delta"],
            "dev_delta_cF1_per_1000_silver_rows": (
                item["dev_slices"]["all"]["delta"]
                / (item["grain"]["silver_rows"] / 1000)
            ),
        }
        for dataset, item in complementarity["datasets"].items()
    }
    return {
        "schema_version": 1,
        "as_of": "2026-08-13",
        "scope": (
            "Measured provider throughput for the 8x2 fidelity campaign, adopted "
            "silver row counts, logged post-lock control training wall time, and "
            "student deployment requirements. Historical silver-generation billing "
            "and original adopted-KD run wall time are unavailable."
        ),
        "hardware": "single NVIDIA GeForce RTX 4090 (24 GB) for recorded post-lock runs",
        "teacher_fidelity": provider_totals,
        "adopted_silver_inventory": adopted_silver,
        "postlock_training": controls,
        "deployment": {
            "teacher_calls_at_student_inference": 0,
            "student_seed_models_per_frozen_ensemble": 5,
            "claim": "LLMs are offline annotators only; deployment uses student checkpoints",
        },
        "limitations": [
            "Historical silver-generation token counts and billing records were not stored; no monetary estimate is fabricated.",
            "Observed training-run wall hours are parsed from timestamped logs, include checkpoint I/O, and are hardware/workload specific.",
            "Teacher-fidelity wall time uses concurrent dataset jobs and is reported as summed per-dataset wall time, not elapsed campaign makespan.",
            "Delta per 1,000 silver rows is descriptive only and is not comparable as a causal efficiency estimate across datasets with different baselines and domains.",
            "Post-lock v2 used local NVMe checkpoint staging before archival, so v1/v2 wall times should not be interpreted as a controlled storage-speed comparison.",
        ],
    }


def build_complete_postlock() -> tuple[dict, dict]:
    v1 = load_json("docs/audits/tallip_kd/postlock_metrics.json")
    v2 = load_json("docs/audits/tallip_kd/postlock_v2_metrics.json")
    complete = copy.deepcopy(v1)
    complete["created_at"] = max(v1["created_at"], v2["created_at"])
    complete["protocol"] = "postlock_dev_only_v1+v2"
    complete["protocols"] = [v1["protocol"], v2["protocol"]]
    for dataset in ("tat_restaurant", "ukr_restaurant"):
        complete["datasets"][dataset]["systems"]["self_training"] = copy.deepcopy(
            v2["datasets"][dataset]["systems"]["self_training"]
        )
        complete["datasets"][dataset]["control_comparisons"] = copy.deepcopy(
            v2["datasets"][dataset]["control_comparisons"]
        )
    v1_checks = v1["bootstrap_reproducibility"]["checks"]
    v2_checks = [
        item for item in v2["bootstrap_reproducibility"]["checks"]
        if "llm_kd_minus_gold_repeat" not in item["id"]
    ]
    complete["bootstrap_reproducibility"] = {
        "comparison_count": len(v1_checks) + len(v2_checks),
        "all_exact_repeats": True,
        "checks": [*v1_checks, *v2_checks],
    }

    audit1 = load_json("docs/audits/tallip_kd/postlock_audit.json")
    audit2 = load_json("docs/audits/tallip_kd/postlock_v2_audit.json")
    v2_self = lambda rows: [row for row in rows if row.get("system") == "self_training"]
    checkpoints = [*audit1["checkpoints"], *v2_self(audit2["checkpoints"])]
    predictions = [*audit1["predictions"], *v2_self(audit2["predictions"])]
    ensembles = [*audit1["ensembles"], *v2_self(audit2["ensembles"])]
    thresholds = [*audit1["thresholds"], *v2_self(audit2["thresholds"])]
    non_best = sorted(set(audit1["non_best_checkpoint_files"] + audit2["non_best_checkpoint_files"]))
    checks = complete["bootstrap_reproducibility"]["checks"]
    audit = {
        "schema_version": 1,
        "as_of": "2026-08-13",
        "protocol": complete["protocol"],
        "source_audits": [
            {"path": "docs/audits/tallip_kd/postlock_audit.json", "sha256": sha256("docs/audits/tallip_kd/postlock_audit.json")},
            {"path": "docs/audits/tallip_kd/postlock_v2_audit.json", "sha256": sha256("docs/audits/tallip_kd/postlock_v2_audit.json")},
        ],
        "scope": audit1["scope"],
        "summary": {
            "prediction_files_expected": 60,
            "prediction_files_validated": len(predictions),
            "best_checkpoints_expected": 50,
            "best_checkpoints_validated": len(checkpoints),
            "ensemble_outputs_rebuilt": len(ensembles),
            "bootstrap_comparisons_repeated": len(checks),
            "all_prediction_ids_and_order_match_dev": all(row["id_order_matches_dev"] for row in predictions),
            "all_ensemble_rebuilds_exact": all(row["exact_rebuild"] for row in ensembles),
            "all_bootstrap_repeats_exact": all(row["exact_repeat"] for row in checks),
            "all_checkpoint_sources_are_best": all(row["checkpoint"].endswith("/checkpoint_best.pt") for row in checkpoints),
            "retention_policy_compliant": not non_best,
        },
        "thresholds": thresholds,
        "bootstrap": {"iterations": ITERATIONS, "seed": SEED, "checks": checks},
        "ensembles": ensembles,
        "predictions": predictions,
        "checkpoints": checkpoints,
        "non_best_checkpoint_files": non_best,
    }
    expected = audit["summary"]
    if (
        expected["prediction_files_validated"] != 60
        or expected["best_checkpoints_validated"] != 50
        or expected["ensemble_outputs_rebuilt"] != 10
        or expected["bootstrap_comparisons_repeated"] != 15
        or not all(value for key, value in expected.items() if key.startswith("all_"))
        or not expected["retention_policy_compliant"]
    ):
        raise ValueError(f"incomplete merged post-lock audit: {expected}")
    return complete, audit


def markdown_outputs(
    complete: dict, complementarity: dict, statistics_report: dict, cost: dict
) -> dict[str, str]:
    controls = [
        "# Complete five-domain post-lock controls", "",
        "All systems are dev-only and use fixed pre-test thresholds.", "",
        "| Dataset | LLM−repeat [95% CI] | LLM−self [95% CI] | Self−repeat [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ADOPTED:
        item = complete["datasets"][dataset]
        repeat = item["systems"]["gold_repeat"]["llm_kd_minus_control"]
        self_result = item["systems"]["self_training"]["llm_kd_minus_control"]
        control = item["control_comparisons"]["self_training_minus_gold_repeat"]
        cell = lambda result: (
            f"{result['delta_a_minus_b']:+.4f} "
            f"[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] "
            f"({result['classification']})"
        )
        controls.append(f"| {dataset} | {cell(repeat)} | {cell(self_result)} | {cell(control)} |")

    comp = [
        "# Five-domain information complementarity and error analysis", "",
        "Descriptive dev-only measurements; they do not establish a universal causal moderator.", "",
        "| Dataset | Teacher fidelity | Silver rows | Dev span coverage Δ A/O | Category JSD Δ silver−base | Repaired / harmed rows | All-dev KD ΔcF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in complementarity["datasets"].items():
        category = item["category_complementarity"]
        lexical = item["lexical_complementarity"]
        transition = item["error_transition"]
        delta = item["dev_slices"]["all"]["delta"]
        aspect_delta = (
            lexical["dev_aspect_types_covered_by_silver"]
            - lexical["dev_aspect_types_covered_by_base"]
        )
        opinion_delta = (
            lexical["dev_opinion_types_covered_by_silver"]
            - lexical["dev_opinion_types_covered_by_base"]
        )
        js_delta = (
            category["silver_to_dev_js_divergence"]
            - category["base_to_dev_js_divergence"]
        )
        comp.append(
            f"| {dataset} | {item['teacher_fidelity_cF1']:.4f} | {item['grain']['silver_rows']} | "
            f"{aspect_delta:+d} / {opinion_delta:+d} | {js_delta:+.4f} | "
            f"{transition['repaired_rows']} / {transition['harmed_rows']} | {delta:+.4f} |"
        )
    summary = complementarity["cross_domain_summary"]
    comp.extend([
        "",
        f"Largest KD dev gain: {summary['largest_kd_dev_gain']}; largest combined "
        f"dev span-coverage gain: {summary['largest_combined_dev_span_coverage_gain']}; "
        f"highest direct teacher fidelity: {summary['highest_teacher_fidelity']}.",
        "",
        summary["allowed_interpretation"],
        "",
        "## Dev error slices", "",
        "| Dataset | All rows / Δ | Multi-quad rows / Δ | Rare-category rows / Δ | Implicit rows / Δ |",
        "|---|---:|---:|---:|---:|",
    ])
    def slice_cell(value: dict) -> str:
        delta = "—" if value["delta"] is None else f"{value['delta']:+.4f}"
        return f"{value['rows']} / {delta}"

    for dataset, item in complementarity["datasets"].items():
        slices = item["dev_slices"]
        comp.append(
            f"| {dataset} | {slice_cell(slices['all'])} | "
            f"{slice_cell(slices['multi_quadruplet'])} | "
            f"{slice_cell(slices['rare_category'])} | "
            f"{slice_cell(slices['implicit_aspect_or_opinion'])} |"
        )

    stats = [
        "# Bootstrap multiplicity robustness", "",
        "Post-hoc inference only; official-test choices remain frozen.", "",
    ]
    for family, items in statistics_report["families"].items():
        stats.extend([
            f"## {family}", "",
            "| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |",
            "|---|---:|---:|---:|---:|---|---|",
        ])
        for item in items:
            stats.append(
                f"| {item['dataset']} | {item['delta_a_minus_b']:+.4f} | "
                f"{item['bootstrap_p_two_sided']:.4f} | {item['holm_adjusted_p']:.4f} | "
                f"{item['bh_adjusted_p']:.4f} | {item['classification']} | "
                f"{'yes' if item['holm_reject_0.05'] else 'no'} |"
            )
        stats.append("")

    resource = [
        "# KD cost and efficiency inventory", "",
        "Token usage and monetary billing were not retained historically; no cost is fabricated.", "",
        "| Provider | Validated rows | Timed rows | Errors | Summed timed wall (s) | Timed rows/s | Token/cost status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for provider, item in cost["teacher_fidelity"].items():
        resource.append(
            f"| {provider} | {item['annotated_rows']} | {item['timed_processed_rows']} | "
            f"{item['errors']} | {item['summed_dataset_wall_seconds']:.1f} | "
            f"{item['effective_timed_rows_per_second']:.2f} | unavailable |"
        )
    resource.extend([
        "", "| Dataset | Teacher | Silver rows | Dev ΔcF1 | Descriptive Δ per 1k silver |",
        "|---|---|---:|---:|---:|",
    ])
    for dataset, item in cost["adopted_silver_inventory"].items():
        resource.append(
            f"| {dataset} | {item['teacher']} | {item['silver_rows']} | "
            f"{item['dev_delta_cF1']:+.4f} | "
            f"{item['dev_delta_cF1_per_1000_silver_rows']:+.4f} |"
        )
    resource.extend([
        "", "| Dataset | System | Encoder | Parameters | 5-seed run-wall hours | Retained best GiB |",
        "|---|---|---|---:|---:|---:|",
    ])
    for item in cost["postlock_training"]:
        resource.append(
            f"| {item['dataset']} | {item['system']} | {item['encoder']} | "
            f"{item['parameters'] or '—'} | {item['observed_run_wall_hours']:.2f} | "
            f"{item['retained_best_checkpoint_bytes'] / (1024 ** 3):.2f} |"
        )
    return {
        "postlock_complete_table.md": "\n".join(controls) + "\n",
        "complementarity_error_analysis.md": "\n".join(comp) + "\n",
        "statistical_robustness.md": "\n".join(stats).rstrip() + "\n",
        "cost_efficiency.md": "\n".join(resource) + "\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    complete, audit = build_complete_postlock()
    complementarity = build_complementarity()
    statistics_report = build_statistics(complete)
    cost = build_cost_efficiency(complementarity)
    outputs = {
        "postlock_complete_metrics.json": complete,
        "postlock_complete_audit.json": audit,
        "complementarity_error_analysis.json": complementarity,
        "statistical_robustness.json": statistics_report,
        "cost_efficiency.json": cost,
    }
    markdown = markdown_outputs(complete, complementarity, statistics_report, cost)
    if args.verify:
        for name, value in outputs.items():
            expected = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            actual = (OUTPUT_ROOT / name).read_text(encoding="utf-8")
            if actual != expected:
                raise RuntimeError(f"non-reproducible artifact: {name}")
        for name, expected in markdown.items():
            actual = (OUTPUT_ROOT / name).read_text(encoding="utf-8")
            if actual != expected:
                raise RuntimeError(f"non-reproducible artifact: {name}")
        print("verified post-lock completion, complementarity, statistics, and cost artifacts")
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        write_json(OUTPUT_ROOT / name, value)
    for name, value in markdown.items():
        (OUTPUT_ROOT / name).write_text(value, encoding="utf-8")
    print("wrote post-lock completion, complementarity, statistics, and cost artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
