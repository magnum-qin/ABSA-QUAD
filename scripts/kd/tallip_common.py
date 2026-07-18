"""Shared, dependency-light utilities for the TALLIP KD evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/kd/tallip.yaml"
D_MAX = math.sqrt(128.0)


def root_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: str | Path = MANIFEST) -> dict:
    with root_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with root_path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = root_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with root_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Leakage key: NFKC + casefold + Unicode whitespace collapse."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _quad_key(quad: dict, fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(quad.get(field, f"Unknown_{field}")).lower() for field in fields)


def _parse_va(quad: dict) -> tuple[float, float] | None:
    try:
        valence, arousal = str(quad["VA"]).split("#")
        result = float(valence), float(arousal)
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(1.0 <= value <= 9.0 for value in result) else None


def official_row_counts(gold_quads: list[dict], pred_quads: list[dict]) -> dict:
    """Return additive counts matching scripts/official_eval.py Task-3 semantics."""
    fields = ("Aspect", "Opinion", "Category")
    matched_pred = 0
    tp_cat = fn_cat = 0
    ctp = 0.0
    pred_by_key: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for pred in pred_quads:
        pred_by_key[_quad_key(pred, fields)].append(pred)
    va_abs = []
    va_sq = []
    for gold in gold_quads:
        matches = pred_by_key.get(_quad_key(gold, fields), [])
        valid_scores = []
        valid_errors = []
        gold_va = _parse_va(gold)
        if gold_va is not None:
            for pred in matches:
                pred_va = _parse_va(pred)
                if pred_va is None:
                    valid_scores.append(0.0)
                    continue
                distance = math.dist(gold_va, pred_va)
                valid_scores.append(max(0.0, 1.0 - distance / D_MAX))
                valid_errors.append((abs(gold_va[0] - pred_va[0]), abs(gold_va[1] - pred_va[1])))
        if len(valid_scores) == 1:
            matched_pred += 1
            tp_cat += 1
            ctp += valid_scores[0]
            if valid_errors:
                for error in valid_errors[0]:
                    va_abs.append(error)
                    va_sq.append(error * error)
        else:
            fn_cat += 1
    return {
        "ctp": ctp,
        "tp_cat": tp_cat,
        "fp_cat": len(pred_quads) - matched_pred,
        "fn_cat": fn_cat,
        "va_abs_sum": sum(va_abs),
        "va_sq_sum": sum(va_sq),
        "va_n": len(va_abs),
    }


def sum_counts(counts: Iterable[dict]) -> dict:
    result = Counter()
    for item in counts:
        result.update(item)
    return dict(result)


def metrics_from_counts(counts: dict) -> dict:
    tp = counts.get("tp_cat", 0)
    fp = counts.get("fp_cat", 0)
    fn = counts.get("fn_cat", 0)
    ctp = counts.get("ctp", 0.0)
    precision = ctp / (tp + fp) if tp + fp else 0.0
    recall = ctp / (tp + fn) if tp + fn else 0.0
    cf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_p = tp / (tp + fp) if tp + fp else 0.0
    exact_r = tp / (tp + fn) if tp + fn else 0.0
    exact_f1 = 2 * exact_p * exact_r / (exact_p + exact_r) if exact_p + exact_r else 0.0
    va_n = counts.get("va_n", 0)
    return {
        "cPrecision": precision,
        "cRecall": recall,
        "cF1": cf1,
        "exact_precision": exact_p,
        "exact_recall": exact_r,
        "exact_f1": exact_f1,
        "tp_categorical": tp,
        "fp_categorical": fp,
        "fn_categorical": fn,
        "va_mae": counts.get("va_abs_sum", 0.0) / va_n if va_n else None,
        "va_rmse": math.sqrt(counts.get("va_sq_sum", 0.0) / va_n) if va_n else None,
        "va_coordinate_n": va_n,
    }


def _set_counts(gold: set, pred: set) -> dict:
    return {"tp": len(gold & pred), "fp": len(pred - gold), "fn": len(gold - pred)}


def _set_metrics(counts: dict) -> dict:
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_records(gold_rows: list[dict], pred_rows: list[dict], train_rows: list[dict] | None = None) -> dict:
    pred_by_id = {str(row.get("ID")): row for row in pred_rows}
    gold_ids = [str(row.get("ID")) for row in gold_rows]
    missing_ids = [row_id for row_id in gold_ids if row_id not in pred_by_id]
    extra_ids = sorted(set(pred_by_id) - set(gold_ids))
    official_counts = []
    span_counts = {"aspect": Counter(), "opinion": Counter(), "category": Counter()}
    null_counts = Counter()
    train_categories = Counter(
        str(quad.get("Category", "")).lower()
        for row in (train_rows or [])
        for quad in row.get("Quadruplet", [])
    )
    rare = {category for category, count in train_categories.items() if count <= 5}
    rare_counts = Counter()
    for gold in gold_rows:
        pred = pred_by_id.get(str(gold.get("ID")), {})
        gold_quads = gold.get("Quadruplet", []) or []
        pred_quads = pred.get("Quadruplet", []) or []
        official_counts.append(official_row_counts(gold_quads, pred_quads))
        for name, field in (("aspect", "Aspect"), ("opinion", "Opinion"), ("category", "Category")):
            gold_set = {str(q.get(field, "")).lower() for q in gold_quads}
            pred_set = {str(q.get(field, "")).lower() for q in pred_quads}
            if name != "category":
                gold_set.discard("null")
                pred_set.discard("null")
            span_counts[name].update(_set_counts(gold_set, pred_set))
        gold_null = {_quad_key(q, ("Aspect", "Opinion", "Category")) for q in gold_quads
                     if str(q.get("Aspect", "")).lower() == "null" or str(q.get("Opinion", "")).lower() == "null"}
        pred_null = {_quad_key(q, ("Aspect", "Opinion", "Category")) for q in pred_quads
                     if str(q.get("Aspect", "")).lower() == "null" or str(q.get("Opinion", "")).lower() == "null"}
        null_counts.update(_set_counts(gold_null, pred_null))
        if rare:
            gold_rare = {_quad_key(q, ("Aspect", "Opinion", "Category")) for q in gold_quads
                         if str(q.get("Category", "")).lower() in rare}
            pred_rare = {_quad_key(q, ("Aspect", "Opinion", "Category")) for q in pred_quads
                         if str(q.get("Category", "")).lower() in rare}
            rare_counts.update(_set_counts(gold_rare, pred_rare))
    invalid = sum(1 for row in pred_rows if row.get("_err"))
    empty = sum(1 for row in pred_rows if not row.get("Quadruplet") and not row.get("_err"))
    metrics = metrics_from_counts(sum_counts(official_counts))
    metrics.update({f"{name}_span": _set_metrics(dict(counts)) for name, counts in span_counts.items()})
    metrics["null_slice"] = _set_metrics(dict(null_counts))
    metrics["rare_category_slice"] = _set_metrics(dict(rare_counts)) if rare else None
    metrics["coverage"] = {
        "gold_rows": len(gold_rows),
        "prediction_rows": len(pred_rows),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "error_rows": invalid,
        "empty_rows": empty,
        "error_rate": invalid / len(gold_rows) if gold_rows else 0.0,
        "empty_rate": empty / len(gold_rows) if gold_rows else 0.0,
    }
    metrics["_row_counts"] = official_counts
    return metrics


def paired_bootstrap(
    gold_rows: list[dict], pred_a: list[dict], pred_b: list[dict],
    iterations: int = 10000, seed: int = 20260713,
) -> dict:
    import random

    a = evaluate_records(gold_rows, pred_a)["_row_counts"]
    b = evaluate_records(gold_rows, pred_b)["_row_counts"]
    observed_a = metrics_from_counts(sum_counts(a))["cF1"]
    observed_b = metrics_from_counts(sum_counts(b))["cF1"]
    rng = random.Random(seed)
    n = len(gold_rows)
    deltas = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        score_a = metrics_from_counts(sum_counts(a[index] for index in indices))["cF1"]
        score_b = metrics_from_counts(sum_counts(b[index] for index in indices))["cF1"]
        deltas.append(score_a - score_b)
    deltas.sort()

    def percentile(q: float) -> float:
        position = (len(deltas) - 1) * q
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return deltas[low]
        return deltas[low] * (high - position) + deltas[high] * (position - low)

    low, high = percentile(0.025), percentile(0.975)
    return {
        "metric": "official_task3_cF1",
        "iterations": iterations,
        "seed": seed,
        "score_a": observed_a,
        "score_b": observed_b,
        "delta_a_minus_b": observed_a - observed_b,
        "ci95": [low, high],
        "classification": "supported" if low > 0 else "harmful" if high < 0 else "inconclusive",
    }
