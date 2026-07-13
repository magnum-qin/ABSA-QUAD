#!/usr/bin/env python3
"""Low-cost, reproducible quality audit for the two Russian silver batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ALLOWED_ENTITIES = {"AMBIENCE", "DRINKS", "FOOD", "LOCATION", "NULL", "RESTAURANT", "SERVICE"}
ALLOWED_ATTRIBUTES = {"GENERAL", "MISCELLANEOUS", "NULL", "PRICES", "QUALITY", "STYLE_OPTIONS"}
ALLOWED_CATEGORIES = {f"{e}#{a}" for e in ALLOWED_ENTITIES for a in ALLOWED_ATTRIBUTES}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            row["__line__"] = line_no
            rows.append(row)
    return rows


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def stable_sample(rows: list[dict], size: int) -> list[dict]:
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f"{row.get('ID', '')}\0{row.get('Text', '')}".encode("utf-8")
    ).digest())
    return ranked[: min(size, len(ranked))]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max", "mean", "std")}
    ordered = sorted(values)
    def q(frac: float) -> float:
        index = frac * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    return {"min": ordered[0], "p25": q(.25), "median": median(ordered), "p75": q(.75),
            "max": ordered[-1], "mean": mean(ordered), "std": pstdev(ordered)}


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def js_divergence(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return 0.0
    p = {key: left[key] / left_total for key in keys}
    q = {key: right[key] / right_total for key in keys}
    m = {key: (p[key] + q[key]) / 2 for key in keys}
    def kl(a: dict, b: dict) -> float:
        return sum(value * math.log2(value / b[key]) for key, value in a.items() if value)
    return (kl(p, m) + kl(q, m)) / 2


def distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {key: count / total for key, count in counter.most_common()}


def validate_row(row: dict) -> list[str]:
    flags = []
    text = row.get("Text")
    quads = row.get("Quadruplet")
    if not isinstance(row.get("ID"), str) or not isinstance(text, str) or not isinstance(quads, list):
        return ["invalid_row_schema"]
    if not text.strip():
        flags.append("empty_text")
    for index, quad in enumerate(quads):
        if not isinstance(quad, dict) or set(quad) != {"Aspect", "Opinion", "Category", "VA"}:
            flags.append(f"quad_{index}_schema")
            continue
        if quad["Category"] not in ALLOWED_CATEGORIES:
            flags.append(f"quad_{index}_category")
        for field in ("Aspect", "Opinion"):
            value = quad[field]
            if not isinstance(value, str) or (value != "NULL" and value not in text):
                flags.append(f"quad_{index}_{field.casefold()}_span")
        try:
            va = [float(value) for value in quad["VA"].split("#")]
            if len(va) != 2 or any(value < 1 or value > 9 for value in va):
                raise ValueError
        except (AttributeError, ValueError):
            flags.append(f"quad_{index}_va")
    return flags


def lexical_metrics(rows: list[dict], reference_rows: list[dict]) -> dict:
    stream = [token for row in rows for token in tokens(row.get("Text", ""))]
    reference_stream = [token for row in reference_rows for token in tokens(row.get("Text", ""))]
    token_set, reference_set = set(stream), set(reference_stream)
    bigrams = set(zip(stream, stream[1:]))
    reference_bigrams = set(zip(reference_stream, reference_stream[1:]))
    segments = [stream[index:index + 100] for index in range(0, len(stream), 100) if len(stream[index:index + 100]) == 100]
    return {
        "tokens": len(stream),
        "types": len(token_set),
        "type_token_ratio": len(token_set) / max(1, len(stream)),
        "msttr_100": mean(len(set(segment)) / 100 for segment in segments) if segments else None,
        "unigram_novelty_vs_reference": len(token_set - reference_set) / max(1, len(token_set)),
        "bigram_novelty_vs_reference": len(bigrams - reference_bigrams) / max(1, len(bigrams)),
    }


def batch_metrics(rows: list[dict], lexical_reference: list[dict]) -> tuple[dict, Counter, list[float], list[float]]:
    flags = [validate_row(row) for row in rows]
    texts = [row.get("Text", "") for row in rows]
    normalized = [" ".join(tokens(text)) for text in texts]
    quads = [quad for row in rows for quad in row.get("Quadruplet", []) if isinstance(quad, dict)]
    categories = Counter(quad.get("Category") for quad in quads)
    valence, arousal = [], []
    for quad in quads:
        try:
            v, a = (float(value) for value in quad["VA"].split("#"))
            valence.append(v); arousal.append(a)
        except (KeyError, AttributeError, ValueError):
            pass
    null_aspect = sum(quad.get("Aspect") == "NULL" for quad in quads)
    null_opinion = sum(quad.get("Opinion") == "NULL" for quad in quads)
    both_null = sum(quad.get("Aspect") == quad.get("Opinion") == "NULL" for quad in quads)
    invalid_counter = Counter(flag.split("_", 2)[-1] for row_flags in flags for flag in row_flags)
    category_entropy = entropy(categories)
    return ({
        "rows": len(rows),
        "unique_ids": len({row.get("ID") for row in rows}),
        "unique_texts": len(set(texts)),
        "normalized_duplicate_rows": len(rows) - len(set(normalized)),
        "invalid_rows": sum(bool(row_flags) for row_flags in flags),
        "validation_flags": dict(invalid_counter),
        "text_characters": quantiles([len(text) for text in texts]),
        "text_tokens": quantiles([len(tokens(text)) for text in texts]),
        "quadruplets": len(quads),
        "quadruplets_per_row": quantiles([len(row.get("Quadruplet", [])) for row in rows]),
        "null_aspect_rate": null_aspect / max(1, len(quads)),
        "null_opinion_rate": null_opinion / max(1, len(quads)),
        "both_null_rate": both_null / max(1, len(quads)),
        "category_count": len(categories),
        "category_entropy_bits": category_entropy,
        "category_entropy_normalized": category_entropy / math.log2(max(2, len(categories))),
        "category_top_share": max(categories.values(), default=0) / max(1, sum(categories.values())),
        "category_distribution": distribution(categories),
        "valence": quantiles(valence),
        "arousal": quantiles(arousal),
        "lexical": lexical_metrics(rows, lexical_reference),
    }, categories, valence, arousal)


def write_manual_sample(path: Path, batches: dict[str, list[dict]], per_batch: int = 50) -> None:
    rng = random.Random(20260713)
    records = []
    for batch, rows in batches.items():
        selected = rng.sample(rows, min(per_batch, len(rows)))
        for row in selected:
            records.append({
                "batch": batch, "source_line": row["__line__"], "ID": row.get("ID"),
                "Text": row.get("Text"), "Quadruplet": row.get("Quadruplet"),
                "automated_flags": validate_row(row),
                "manual_label_correct": None, "manual_notes": ""
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_analysis(sample_size: int = 3000) -> dict:
    paths = {
        "gen4k": ROOT / "data/subtask_3/rus/rus_restaurant_silver_v23_deepseek_gen4k.jsonl",
        "gen8k": ROOT / "data/subtask_3/rus/rus_restaurant_silver_v23_deepseek_gen.jsonl",
        "gold": ROOT / "data/subtask_3/rus/rus_restaurant_train_alltasks.jsonl",
        "test": ROOT / "data/subtask_3/rus/rus_restaurant_test_task3.jsonl",
    }
    data = {name: load_jsonl(path) for name, path in paths.items()}
    reference = data["gold"] + data["test"]
    sampled = {name: stable_sample(data[name], sample_size) for name in ("gen4k", "gen8k")}
    gold_metrics, gold_categories, _, _ = batch_metrics(data["gold"], reference)
    batch_results, category_counters = {}, {}
    for name in ("gen4k", "gen8k"):
        full_metrics, categories, _, _ = batch_metrics(data[name], reference)
        sample_metrics, _, _, _ = batch_metrics(sampled[name], reference)
        full_metrics["equal_size_lexical_sample"] = sample_metrics["lexical"]
        full_metrics["category_jsd_vs_gold"] = js_divergence(categories, gold_categories)
        batch_results[name] = full_metrics
        category_counters[name] = categories
    gen4k_text = {" ".join(tokens(row.get("Text", ""))) for row in data["gen4k"]}
    gen8k_text = {" ".join(tokens(row.get("Text", ""))) for row in data["gen8k"]}
    return {
        "schema_version": 1,
        "generated_at": "2026-07-13T00:00:00+09:00",
        "scope": "Automated low-cost silver quality audit; semantic label correctness is not automatically established.",
        "source_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "equal_size_sample_rows_per_batch": min(sample_size, len(sampled["gen4k"]), len(sampled["gen8k"])),
        "batches": batch_results,
        "gold_reference": gold_metrics,
        "comparisons": {
            "normalized_text_overlap_rows": len(gen4k_text & gen8k_text),
            "category_jsd_gen4k_vs_gen8k": js_divergence(category_counters["gen4k"], category_counters["gen8k"]),
            "paired_seed_dev_cf1": {
                "gen8k_full": {"seed42": 0.5286, "seed123": 0.5648, "mean": 0.5467},
                "gen8k_cap3000": {"seed42": 0.5123, "seed123": 0.5584, "mean": 0.5353},
                "same_pool_cap3k_minus_full_pp": -1.14
            },
            "ensemble_dev_cf1": {"gen4k_cap3000": 0.6162, "gen8k_full": 0.5858}
        },
        "manual_review": {
            "sample_rows": 100,
            "status": "sample_prepared_not_human_adjudicated",
            "path": "docs/audits/rus_silver_manual_sample_100.jsonl"
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/audits/rus_silver_quality_metrics.json")
    parser.add_argument("--sample-output", default="docs/audits/rus_silver_manual_sample_100.jsonl")
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()
    result = run_analysis(args.sample_size)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    batches = {
        name: load_jsonl(ROOT / result["source_files"][name]) for name in ("gen4k", "gen8k")
    }
    write_manual_sample(ROOT / args.sample_output, batches)
    print(json.dumps({name: {key: result["batches"][name][key] for key in
          ("rows", "invalid_rows", "normalized_duplicate_rows", "category_jsd_vs_gold")}
          for name in ("gen4k", "gen8k")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
