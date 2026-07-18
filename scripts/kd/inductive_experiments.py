#!/usr/bin/env python
"""Run and summarize the clean-inductive KD and baseline experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tallip_common import (
    ROOT, evaluate_records, load_manifest, paired_bootstrap, read_jsonl, root_path,
    sha256_file,
)


def select(manifest: dict, names: list[str]) -> list[tuple[str, dict]]:
    if names == ["all"]:
        return list(manifest["datasets"].items())
    unknown = set(names) - set(manifest["datasets"])
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    return [(name, manifest["datasets"][name]) for name in names]


def run_checked(command: list[str], log_path: Path | None = None, dry_run: bool = False) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def train(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["defaults"]["seeds"]
    jobs = []
    for dataset, spec in select(manifest, args.datasets):
        if "kd" in args.systems and spec.get("clean_rerun"):
            jobs.append((dataset, "kd", spec["base_config"], spec["clean_train"], spec["clean_prefix"]))
        if "baseline" in args.systems and spec.get("clean_baseline_rerun"):
            jobs.append((dataset, "baseline", spec["baseline_config"], spec["clean_baseline_train"], spec["baseline_prefix"]))
    if not jobs:
        print("No training jobs selected")
        return 0
    if not args.dry_run:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
            cwd=ROOT,
        )
        if probe.returncode:
            raise SystemExit("CUDA is unavailable; refusing to start GPU training")
    for dataset, system, config, train_file, prefix in jobs:
        command = [
            sys.executable, "scripts/train_multi_seed.py", "--config", config,
            "--seeds", *map(str, seeds), "--save_prefix", prefix,
            "--train_file", train_file, "--resume_existing",
        ]
        run_checked(command, Path(f"/tmp/tallip_{dataset}_{system}_train.log"), args.dry_run)
    return 0


def prediction_root(manifest: dict) -> Path:
    return root_path(manifest["defaults"]["output_root"]) / "inductive_predictions"


def system_spec(spec: dict, system: str) -> dict | None:
    if system == "clean":
        if not spec.get("clean_prefix"):
            return None
        return {"prefix": spec["clean_prefix"], "config": spec["base_config"],
                "vote": spec["vote"], "topk": spec["topk"]}
    if system == "original":
        if not spec.get("original_prefix"):
            return None
        return {"prefix": spec["original_prefix"], "config": spec["base_config"],
                "vote": spec["vote"], "topk": spec["topk"]}
    if system == "baseline":
        if spec.get("baseline_prediction") or spec.get("strongest_baseline_prediction"):
            return None
        prefix = spec.get("baseline_prefix") or spec.get("original_baseline_prefix")
        if not prefix:
            return None
        return {"prefix": prefix, "config": spec.get("baseline_config", spec["base_config"]),
                "vote": spec["baseline_vote"], "topk": spec["baseline_topk"]}
    raise ValueError(system)


def infer(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["defaults"]["seeds"]
    out_root = prediction_root(manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    command_log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_policy": "checkpoint_best.pt only",
        "runs": [],
    }
    for dataset, spec in select(manifest, args.datasets):
        for system in args.systems:
            resolved = system_spec(spec, system)
            if not resolved:
                continue
            seed_dirs = [f"checkpoints/{resolved['prefix']}{seed}" for seed in seeds]
            missing = [directory for directory in seed_dirs if not root_path(directory,).joinpath("checkpoint_best.pt").exists()]
            if missing:
                print(f"SKIP {dataset}/{system}: missing best checkpoints: {missing}", file=sys.stderr)
                continue
            for seed, seed_dir in zip(seeds, seed_dirs):
                output = out_root / f"{dataset}_{system}_seed{seed}.jsonl"
                command = [
                    sys.executable, "scripts/inference.py", "--multi_seed_ensemble",
                    "--seed_dirs", seed_dir, "--config", resolved["config"],
                    "--test_file", spec["dev"], "--output", str(output),
                    "--vote_threshold", "1", "--span_topk_threshold", str(resolved["topk"]),
                    "--batch_size", str(args.batch_size),
                ]
                command_log["runs"].append({"dataset": dataset, "system": system, "seed": seed,
                                             "vote": 1, "topk": resolved["topk"], "output": str(output.relative_to(ROOT))})
                if not output.exists() or args.force:
                    run_checked(command, Path(f"/tmp/tallip_{dataset}_{system}_seed{seed}_inference.log"), args.dry_run)
            output = out_root / f"{dataset}_{system}_ensemble.jsonl"
            command = [
                sys.executable, "scripts/inference.py", "--multi_seed_ensemble",
                "--seed_dirs", *seed_dirs, "--config", resolved["config"],
                "--test_file", spec["dev"], "--output", str(output),
                "--vote_threshold", str(resolved["vote"]),
                "--span_topk_threshold", str(resolved["topk"]), "--batch_size", str(args.batch_size),
            ]
            command_log["runs"].append({"dataset": dataset, "system": system, "seed": "ensemble",
                                         "vote": resolved["vote"], "topk": resolved["topk"],
                                         "output": str(output.relative_to(ROOT))})
            if not output.exists() or args.force:
                run_checked(command, Path(f"/tmp/tallip_{dataset}_{system}_ensemble_inference.log"), args.dry_run)
    if not args.dry_run:
        log_path = root_path("docs/audits/tallip_kd/inference_manifest.json")
        log_path.write_text(json.dumps(command_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def external_baseline(spec: dict) -> Path | None:
    path = spec.get("strongest_baseline_prediction") or spec.get("baseline_prediction")
    return root_path(path) if path else None


def load_prediction(manifest: dict, dataset: str, spec: dict, system: str, seed: int | str = "ensemble") -> tuple[list[dict], str] | None:
    if system == "baseline":
        external = external_baseline(spec)
        if external:
            return read_jsonl(external), str(external.relative_to(ROOT))
    path = prediction_root(manifest) / f"{dataset}_{system}_{'ensemble' if seed == 'ensemble' else f'seed{seed}'}.jsonl"
    return (read_jsonl(path), str(path.relative_to(ROOT))) if path.exists() else None


def collect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    seeds = manifest["defaults"]["seeds"]
    iterations = manifest["defaults"]["bootstrap_iterations"]
    bootstrap_seed = manifest["defaults"]["bootstrap_seed"]
    report = {"bootstrap_iterations": iterations, "bootstrap_seed": bootstrap_seed, "datasets": {}}
    table = [
        "# TALLIP clean-inductive KD results", "",
        "| Dataset | System | Seed mean±sd | Ensemble cF1 | vs baseline Δ [95% CI] | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset, spec in select(manifest, args.datasets):
        if not spec.get("adopted"):
            continue
        gold = read_jsonl(spec["dev"])
        dataset_report = {"dev": spec["dev"], "dev_sha256": sha256_file(spec["dev"]), "systems": {}, "comparisons": {}}
        systems = ["clean", "original"] if spec.get("clean_prefix") else ["original"]
        systems.append("baseline")
        for system in systems:
            ensemble_loaded = load_prediction(manifest, dataset, spec, system)
            if not ensemble_loaded:
                dataset_report["systems"][system] = {"status": "missing"}
                continue
            rows, source = ensemble_loaded
            ensemble_metrics = evaluate_records(gold, rows)
            coverage = ensemble_metrics["coverage"]
            if coverage["missing_ids"] or coverage["extra_ids"] or coverage["error_rows"]:
                raise SystemExit(f"Invalid prediction coverage for {dataset}/{system}: {coverage}")
            ensemble_metrics.pop("_row_counts")
            single_scores = []
            for seed in seeds:
                loaded = load_prediction(manifest, dataset, spec, system, seed)
                if loaded:
                    metrics = evaluate_records(gold, loaded[0])
                    coverage = metrics["coverage"]
                    if coverage["missing_ids"] or coverage["extra_ids"] or coverage["error_rows"]:
                        raise SystemExit(f"Invalid single-seed coverage for {dataset}/{system}/{seed}: {coverage}")
                    single_scores.append({"seed": seed, "cF1": metrics["cF1"]})
            summary = {
                "status": "complete", "source": source, "source_sha256": sha256_file(source),
                "ensemble": ensemble_metrics, "single_seeds": single_scores,
            }
            if single_scores:
                values = [item["cF1"] for item in single_scores]
                summary["single_mean"] = statistics.mean(values)
                summary["single_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            dataset_report["systems"][system] = summary
        comparison_system = "clean" if spec.get("clean_prefix") else "original"
        a = load_prediction(manifest, dataset, spec, comparison_system)
        b = load_prediction(manifest, dataset, spec, "baseline")
        if a and b:
            result = paired_bootstrap(gold, a[0], b[0], iterations, bootstrap_seed)
            dataset_report["comparisons"][f"{comparison_system}_minus_baseline"] = result
        if spec.get("clean_prefix"):
            clean = load_prediction(manifest, dataset, spec, "clean")
            original = load_prediction(manifest, dataset, spec, "original")
            if clean and original:
                dataset_report["comparisons"]["clean_minus_transductive"] = paired_bootstrap(
                    gold, clean[0], original[0], iterations, bootstrap_seed,
                )
        report["datasets"][dataset] = dataset_report
        shown = dataset_report["systems"].get(comparison_system, {})
        comparison = dataset_report["comparisons"].get(f"{comparison_system}_minus_baseline")
        if shown.get("status") == "complete":
            mean_sd = "—"
            if "single_mean" in shown:
                mean_sd = f"{shown['single_mean']:.4f}±{shown['single_sd']:.4f}"
            if comparison:
                delta = comparison["delta_a_minus_b"]
                low, high = comparison["ci95"]
                comparison_text = f"{delta:+.4f} [{low:+.4f}, {high:+.4f}]"
                decision = comparison["classification"]
            else:
                comparison_text, decision = "—", "missing"
            table.append(f"| {dataset} | {comparison_system} | {mean_sd} | {shown['ensemble']['cF1']:.4f} | "
                         f"{comparison_text} | {decision} |")
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table.extend(["", f"Paired bootstrap: {iterations:,} resamples, seed={bootstrap_seed}.", ""])
    markdown = root_path(args.markdown)
    markdown.write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table))
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    failed = False
    for dataset, spec in select(manifest, args.datasets):
        if not spec.get("clean_rerun"):
            continue
        for system, prefix in (("kd", spec["clean_prefix"]),
                               ("baseline", spec.get("baseline_prefix"))):
            if not prefix:
                continue
            for seed in manifest["defaults"]["seeds"]:
                best = root_path(f"checkpoints/{prefix}{seed}/checkpoint_best.pt")
                ok = best.exists()
                print(f"{dataset:<20} {system:<8} seed={seed:<4} best={'OK' if ok else 'MISSING'}")
                failed |= not ok
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="scripts/kd/tallip.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--datasets", nargs="+", default=["all"])
    train_parser.add_argument("--systems", nargs="+", choices=["kd", "baseline"], default=["kd", "baseline"])
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(func=train)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--datasets", nargs="+", default=["all"])
    infer_parser.add_argument("--systems", nargs="+", choices=["clean", "original", "baseline"],
                              default=["clean", "original", "baseline"])
    infer_parser.add_argument("--batch-size", type=int, default=16)
    infer_parser.add_argument("--force", action="store_true")
    infer_parser.add_argument("--dry-run", action="store_true")
    infer_parser.set_defaults(func=infer)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--datasets", nargs="+", default=["all"])
    collect_parser.add_argument("--output", default="docs/audits/tallip_kd/inductive_metrics.json")
    collect_parser.add_argument("--markdown", default="docs/audits/tallip_kd/inductive_table.md")
    collect_parser.set_defaults(func=collect)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--datasets", nargs="+", default=["all"])
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
