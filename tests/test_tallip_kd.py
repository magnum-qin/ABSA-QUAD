import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/kd"))

from tallip_common import (  # noqa: E402
    evaluate_records, load_manifest, normalize_text, paired_bootstrap, read_jsonl,
)


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TallipKDTests(unittest.TestCase):
    def test_cloud_and_local_teacher_prompts_are_identical(self):
        cloud = load_module("silver_cloud", "scripts/llm_silver_label.py")
        local = load_module("silver_local", "scripts/llm_silver_label_openai.py")
        categories = ["FOOD#QUALITY", "SERVICE#GENERAL"]
        self.assertEqual(
            cloud.build_system_prompt(categories, "restaurant"),
            local.build_system_prompt(categories, "restaurant"),
        )

    def test_normalization_matches_registered_cleaning_protocol(self):
        self.assertEqual(normalize_text("  ＬＯＶＥ\tThis  Laptop\n"), "love this laptop")

    def test_teacher_inputs_are_label_free_and_complete(self):
        manifest = load_manifest()
        total = 0
        for dataset, spec in manifest["datasets"].items():
            source = read_jsonl(spec["dev"])
            probe = read_jsonl(
                manifest["defaults"]["output_root"] + f"/teacher_fidelity/inputs/{dataset}.jsonl"
            )
            self.assertEqual([row["ID"] for row in source], [row["ID"] for row in probe])
            self.assertTrue(all("Quadruplet" not in row for row in probe))
            total += len(probe)
        self.assertEqual(total, 1344)

    def test_clean_files_have_expected_counts_and_no_overlap(self):
        manifest = load_manifest()
        for _, spec in manifest["datasets"].items():
            for clean_key, expected_key in (("clean_train", "expected_clean_rows"),
                                            ("clean_baseline_train", "expected_clean_baseline_rows")):
                if clean_key not in spec:
                    continue
                clean = read_jsonl(spec[clean_key])
                self.assertEqual(len(clean), spec[expected_key])
                references = {normalize_text(row["Text"]) for split in ("dev", "test")
                              for row in read_jsonl(spec[split])}
                self.assertFalse({normalize_text(row["Text"]) for row in clean} & references)

    def test_metric_matches_archived_official_scores(self):
        cases = [
            ("data/subtask_3/rus/rus_restaurant_dev_task3.jsonl",
             "predictions/v15_sweep/v15_rus_vote4_topk0.10.jsonl", 0.5267369887665554),
            ("data/subtask_3/tat/tat_restaurant_dev_task3.jsonl",
             "predictions/v15_sweep/v15_tat_vote3_topk0.20.jsonl", 0.4820721598266662),
            ("data/subtask_3/zho/zho_restaurant_dev_task3.jsonl",
             "predictions/v16_missing_sweep/v16_zho_restaurant_vote4_topk0.0.jsonl", 0.5669438101693828),
        ]
        for gold, pred, expected in cases:
            self.assertAlmostEqual(evaluate_records(read_jsonl(gold), read_jsonl(pred))["cF1"], expected)

    def test_bootstrap_is_deterministic(self):
        gold = [{"ID": "1", "Text": "x", "Quadruplet": [
            {"Aspect": "x", "Category": "A#B", "Opinion": "x", "VA": "5#5"}
        ]}]
        good = json.loads(json.dumps(gold))
        bad = [{"ID": "1", "Text": "x", "Quadruplet": []}]
        first = paired_bootstrap(gold, good, bad, iterations=100, seed=7)
        second = paired_bootstrap(gold, good, bad, iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["classification"], "supported")

    def test_inductive_registry_has_frozen_thresholds(self):
        registry = yaml.safe_load((ROOT / "scripts/kd/registry.yaml").read_text())
        expected = {
            "eng_laptop_deepseek_inductive": (4, 0.0),
            "ukr_restaurant_deepseek_inductive": (3, 0.0),
            "zho_restaurant_deepseek_real_inductive": (3, 0.0),
        }
        for name, thresholds in expected.items():
            experiment = registry["experiments"][name]
            self.assertEqual((experiment["eval_vote"], experiment["eval_topk"]), thresholds)
            self.assertTrue(experiment["distill_prebuilt"])
            self.assertIn("inductive", experiment["gold_base"])

    def test_official_test_protocol_is_dev_locked(self):
        protocol = yaml.safe_load((ROOT / "scripts/kd/official_test.yaml").read_text())
        meta = protocol["protocol"]
        self.assertEqual(meta["selection_basis"], "dev_only")
        self.assertTrue(meta["no_test_tuning"])
        self.assertEqual(meta["bootstrap_seed"], 20260713)
        self.assertEqual(meta["bootstrap_iterations"], 10000)
        self.assertEqual(len(protocol["datasets"]), 8)
        generated = set(protocol["inference"])
        referenced_generated = {
            system
            for spec in protocol["datasets"].values()
            for system, source in spec["systems"].items()
            if source == "generated"
        }
        self.assertEqual(generated, referenced_generated)
        for dataset, spec in protocol["datasets"].items():
            self.assertIn(spec["main_system"], spec["systems"], dataset)
        expected_votes = {
            "eng_laptop_clean_kd": 4,
            "ukr_restaurant_clean_kd": 3,
            "zho_restaurant_clean_kd": 3,
            "tat_restaurant_deepseek_matched": 3,
        }
        for system, vote in expected_votes.items():
            self.assertEqual(protocol["inference"][system]["vote"], vote)
            self.assertEqual(protocol["inference"][system]["topk"], 0.0)

    def test_public_conclusion_surface_matches_locked_evidence(self):
        conclusions = (ROOT / "docs/CONCLUSIONS.md").read_text()
        readme = (ROOT / "README.md").read_text()
        ai_readme = (ROOT / "README_AI.md").read_text()

        for dataset, score in {
            "eng_laptop": "0.3532",
            "eng_restaurant": "0.5904",
            "jpn_hotel": "0.4217",
            "rus_restaurant": "0.5189",
            "tat_restaurant": "0.4311",
            "ukr_restaurant": "0.4608",
            "zho_laptop": "0.4309",
            "zho_restaurant": "0.4660",
        }.items():
            self.assertIn(f"| {dataset} |", conclusions)
            self.assertIn(score, conclusions)
            self.assertIn(score, readme)

        self.assertIn("四个域支持 KD 的条件性收益", conclusions)
        self.assertIn("不能写“5/8 KD 成功”", ai_readme)
        self.assertNotIn("test SOTA 对比", readme)
        self.assertNotIn("反超 nchellwig", readme)
        self.assertNotIn("第 5 个经 KD 改善", readme)

    def test_tallip_novelty_is_separated_from_iconip_v20(self):
        boundary = (ROOT / "docs/TALLIP_VS_ICONIP_V20.md").read_text()
        conclusions = (ROOT / "docs/CONCLUSIONS.md").read_text()
        experiments = (ROOT / "docs/EXPERIMENTS.md").read_text()

        inherited = [
            "Rethink",
            "Hungarian matching",
            "language-specific encoder",
            "five-seed ensemble",
        ]
        new_evidence = [
            "8 数据集 × 2 teachers",
            "matched-pool",
            "cap3000",
            "clean-inductive",
            "paired bootstrap",
            "official-test",
        ]
        for phrase in inherited + new_evidence:
            self.assertIn(phrase, boundary)

        self.assertIn("不能再次声称为新贡献", boundary)
        self.assertIn("不声称 SOTA", boundary)
        self.assertIn("不能重复计算 novelty", experiments)
        self.assertIn("TALLIP_VS_ICONIP_V20.md", conclusions)
        self.assertIn("TALLIP_VS_ICONIP_V20.md", (ROOT / "README_AI.md").read_text())

        forbidden_claims = [
            "TALLIP proposes LEADER-DimASQP",
            "TALLIP 提出 Rethink",
            "TALLIP achieves test SOTA",
        ]
        public_surface = conclusions + experiments
        for claim in forbidden_claims:
            self.assertNotIn(claim, public_surface)

    def test_frozen_core_tables_are_reproducible(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/kd/build_tallip_report.py"), "--verify"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("verified 6 frozen tables", result.stdout)

        lock = json.loads(
            (ROOT / "docs/audits/tallip_kd/tallip_core_tables.lock.json").read_text()
        )
        self.assertEqual(lock["status"], "frozen")
        self.assertEqual(lock["table_count"], 6)
        self.assertEqual(len(lock["output"]["sha256"]), 64)
        paths = {item["path"] for item in lock["inputs"]}
        self.assertIn("scripts/kd/official_test.yaml", paths)
        self.assertIn("docs/audits/tallip_kd/ablation_evidence.json", paths)
        self.assertIn("scripts/kd/postlock_v1.yaml", paths)
        self.assertIn("scripts/kd/postlock_v2.yaml", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_metrics.json", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_audit.json", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_v2_metrics.json", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_v2_audit.json", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_complete_metrics.json", paths)
        self.assertIn("docs/audits/tallip_kd/postlock_complete_audit.json", paths)
        self.assertIn("docs/audits/tallip_kd/complementarity_error_analysis.json", paths)
        self.assertIn("docs/audits/tallip_kd/statistical_robustness.json", paths)
        self.assertIn("docs/audits/tallip_kd/cost_efficiency.json", paths)

        tables = (ROOT / lock["output"]["path"]).read_text()
        self.assertEqual(tables.count("## Table "), 6)
        self.assertIn("Inductive-safe KD five-seed", tables)
        self.assertIn("real − generated = +2.3pp", tables)
        self.assertIn("Post-lock dev-only controls", tables)
        self.assertIn("Direct complementarity and error measurements", tables)
        self.assertIn("Multiplicity robustness", tables)
        self.assertIn("Resource and deployment audit", tables)
        self.assertIn("One-shot official Task-3 test evaluation", tables)

    def test_postlock_protocol_is_dev_only_and_repeat_is_deterministic(self):
        module = load_module("postlock_experiments", "scripts/kd/postlock_experiments.py")
        manifest = yaml.safe_load((ROOT / "scripts/kd/postlock_v1.yaml").read_text())
        protocol = manifest["protocol"]
        self.assertEqual(protocol["official_test_evaluation"], "forbidden")
        self.assertEqual(protocol["model_selection_split"], "dev_only")
        self.assertEqual(protocol["publish_root"], "docs/audits/tallip_kd")
        self.assertEqual(protocol["seeds"], [42, 123, 456, 789, 1024])
        self.assertEqual(len(manifest["gold_repeat"]), 5)
        self.assertEqual(set(manifest["self_training"]), {"eng_laptop", "rus_restaurant", "zho_restaurant"})
        for spec in manifest["self_training"].values():
            self.assertNotIn("test", spec)
            self.assertNotIn("official", " ".join(spec))

        rows = [
            {"ID": "a", "Text": "A", "Quadruplet": []},
            {"ID": "b", "Text": "B", "Quadruplet": []},
        ]
        first = module.repeat_gold(rows, 5, 7)
        second = module.repeat_gold(rows, 5, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({row["ID"] for row in first}), 5)

    def test_postlock_v2_completes_self_training_matrix_without_test_selection(self):
        manifest = yaml.safe_load((ROOT / "scripts/kd/postlock_v2.yaml").read_text())
        protocol = manifest["protocol"]
        self.assertEqual(protocol["name"], "postlock_dev_only_v2")
        self.assertEqual(protocol["official_test_evaluation"], "forbidden")
        self.assertEqual(protocol["model_selection_split"], "dev_only")
        self.assertEqual(protocol["threshold_policy"], "frozen_from_postlock_v1")
        self.assertEqual(protocol["artifact_prefix"], "postlock_v2")
        self.assertEqual(protocol["seeds"], [42, 123, 456, 789, 1024])
        self.assertEqual(
            set(manifest["self_training"]),
            {"tat_restaurant", "ukr_restaurant"},
        )
        for dataset, spec in manifest["self_training"].items():
            self.assertNotIn("test", spec)
            self.assertNotIn("official", " ".join(spec))
            self.assertEqual((spec["pseudo_vote"], spec["pseudo_topk"]), (3, 0.0))
            self.assertEqual((spec["eval_vote"], spec["eval_topk"]), (3, 0.0))
            self.assertTrue(
                spec["gold_teacher_prefix"].startswith("postlock_v1_"), dataset
            )

    def test_postlock_analysis_adjustments_and_complementarity_inputs(self):
        module = load_module(
            "tallip_postlock_analysis",
            "scripts/kd/tallip_postlock_analysis.py",
        )
        self.assertEqual(module.holm_adjust([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
        self.assertEqual(module.bh_adjust([0.01, 0.02, 0.5]), [0.03, 0.03, 0.5])
        analysis = module.build_complementarity()
        self.assertEqual(set(analysis["datasets"]), set(module.ADOPTED))
        cross_domain = analysis["cross_domain_summary"]
        self.assertEqual(cross_domain["largest_kd_dev_gain"], "eng_laptop")
        self.assertEqual(cross_domain["smallest_kd_dev_gain"], "zho_restaurant")
        self.assertEqual(
            cross_domain["largest_combined_dev_span_coverage_gain"], "eng_laptop"
        )
        self.assertEqual(cross_domain["highest_teacher_fidelity"], "zho_restaurant")
        self.assertEqual(
            set(cross_domain["silver_category_distribution_closer_to_dev_than_base"]),
            {"eng_laptop", "zho_restaurant"},
        )
        self.assertEqual(
            set(cross_domain["silver_improves_both_dev_aspect_and_opinion_type_coverage"]),
            {"eng_laptop", "ukr_restaurant"},
        )
        for dataset, item in analysis["datasets"].items():
            grain = item["grain"]
            self.assertEqual(
                grain["base_rows"] + grain["silver_rows"],
                grain["llm_train_rows"],
                dataset,
            )
            validity = item["silver_label_validity"]
            self.assertEqual(validity["empty_rows"], 0, dataset)
            self.assertEqual(validity["invalid_explicit_aspect_substrings"], 0, dataset)
            self.assertEqual(validity["invalid_explicit_opinion_substrings"], 0, dataset)
            self.assertEqual(validity["categories_absent_from_gold_train_inventory"], 0, dataset)
            self.assertEqual(validity["invalid_va_format_or_range"], 0, dataset)

    def test_postlock_precomputed_vote_merge_matches_inference_semantics(self):
        module = load_module("postlock_vote_merge", "scripts/kd/postlock_experiments.py")
        predictions = [
            [{"ID": "x", "Quadruplet": [
                {"Aspect": "a", "Category": "C#D", "Opinion": "o", "VA": "3#5"}
            ]}],
            [{"ID": "x", "Quadruplet": [
                {"Aspect": "a", "Category": "C#D", "Opinion": "o", "VA": "5#7"}
            ]}],
            [{"ID": "x", "Quadruplet": []}],
        ]
        merged = module.merge_seed_predictions(predictions, vote_threshold=2)
        self.assertEqual(merged, [{"ID": "x", "Quadruplet": [
            {"Aspect": "a", "Category": "C#D", "Opinion": "o", "VA": "4.00#6.00"}
        ]}])

        predictions[2][0]["ID"] = "wrong"
        with self.assertRaisesRegex(ValueError, "ID/order mismatch"):
            module.merge_seed_predictions(predictions, vote_threshold=2)

    def test_postlock_audit_and_conclusions_are_complete(self):
        audit = json.loads(
            (ROOT / "docs/audits/tallip_kd/postlock_audit.json").read_text()
        )
        summary = audit["summary"]
        self.assertEqual(summary["prediction_files_validated"], 48)
        self.assertEqual(summary["best_checkpoints_validated"], 40)
        self.assertEqual(summary["ensemble_outputs_rebuilt"], 8)
        self.assertEqual(summary["bootstrap_comparisons_repeated"], 11)
        for key in (
            "all_prediction_ids_and_order_match_dev",
            "all_ensemble_rebuilds_exact",
            "all_bootstrap_repeats_exact",
            "all_checkpoint_sources_are_best",
            "retention_policy_compliant",
        ):
            self.assertTrue(summary[key], key)
        self.assertEqual(audit["non_best_checkpoint_files"], [])
        self.assertEqual(audit["scope"]["official_test_evaluation"], "forbidden")
        self.assertEqual(audit["bootstrap"]["iterations"], 10000)
        self.assertEqual(audit["bootstrap"]["seed"], 20260713)

        report = json.loads(
            (ROOT / "docs/audits/tallip_kd/postlock_metrics.json").read_text()
        )
        expected_exposure = {
            "eng_laptop": "supported",
            "rus_restaurant": "supported",
            "tat_restaurant": "inconclusive",
            "ukr_restaurant": "supported",
            "zho_restaurant": "inconclusive",
        }
        for dataset, decision in expected_exposure.items():
            self.assertEqual(
                report["datasets"][dataset]["systems"]["gold_repeat"]
                ["llm_kd_minus_control"]["classification"],
                decision,
            )
        self.assertEqual(
            report["datasets"]["eng_laptop"]["systems"]["self_training"]
            ["llm_kd_minus_control"]["classification"],
            "supported",
        )
        self.assertEqual(
            report["datasets"]["rus_restaurant"]["systems"]["self_training"]
            ["llm_kd_minus_control"]["classification"],
            "supported",
        )
        self.assertEqual(
            report["datasets"]["zho_restaurant"]["systems"]["self_training"]
            ["llm_kd_minus_control"]["classification"],
            "inconclusive",
        )

        registry = yaml.safe_load((ROOT / "scripts/kd/registry.yaml").read_text())
        controls = registry["postlock_controls"]
        self.assertEqual(controls["status"], "complete")
        self.assertEqual(controls["protocol"], "postlock_dev_only_v1+v2")
        self.assertEqual(len(controls["arms"]), 10)
        self.assertEqual(controls["official_test_evaluation"], "forbidden")
        manifests = [
            yaml.safe_load((ROOT / path).read_text()) for path in controls["manifests"]
        ]
        report = json.loads((ROOT / controls["evidence"]).read_text())
        for arm in controls["arms"].values():
            dataset = arm["dataset"]
            if arm["control"] == "exposure_matched_repeated_gold":
                system = "gold_repeat"
                spec = manifests[0]["gold_repeat"][dataset]
                vote, topk = spec["vote"], spec["topk"]
            else:
                system = "self_training"
                manifest = next(
                    item for item in manifests if dataset in item["self_training"]
                )
                spec = manifest["self_training"][dataset]
                vote, topk = spec["eval_vote"], spec["eval_topk"]
            result = report["datasets"][dataset]["systems"][system]
            comparison = result["llm_kd_minus_control"]
            self.assertEqual(arm["train_file"], spec["output_train"])
            self.assertEqual(arm["checkpoint_prefix"], spec["checkpoint_prefix"])
            self.assertEqual(arm["eval"], {"vote": vote, "topk": topk})
            self.assertAlmostEqual(arm["seed_mean"], result["single_mean"])
            self.assertAlmostEqual(arm["seed_sd"], result["single_sd"])
            self.assertAlmostEqual(arm["ensemble_result"], result["ensemble"]["cF1"])
            registered = arm["llm_kd_minus_control"]
            self.assertAlmostEqual(registered["delta"], comparison["delta_a_minus_b"])
            for registered_bound, result_bound in zip(
                registered["ci95"], comparison["ci95"]
            ):
                self.assertAlmostEqual(registered_bound, result_bound)
            self.assertEqual(registered["decision"], comparison["classification"])

    def test_complete_postlock_closure_is_five_domain_and_audited(self):
        audit = json.loads(
            (ROOT / "docs/audits/tallip_kd/postlock_complete_audit.json").read_text()
        )
        summary = audit["summary"]
        self.assertEqual(summary["prediction_files_expected"], 60)
        self.assertEqual(summary["prediction_files_validated"], 60)
        self.assertEqual(summary["best_checkpoints_expected"], 50)
        self.assertEqual(summary["best_checkpoints_validated"], 50)
        self.assertEqual(summary["ensemble_outputs_rebuilt"], 10)
        self.assertEqual(summary["bootstrap_comparisons_repeated"], 15)
        for key in (
            "all_prediction_ids_and_order_match_dev",
            "all_ensemble_rebuilds_exact",
            "all_bootstrap_repeats_exact",
            "all_checkpoint_sources_are_best",
            "retention_policy_compliant",
        ):
            self.assertTrue(summary[key], key)
        self.assertEqual(audit["non_best_checkpoint_files"], [])
        self.assertEqual(audit["scope"]["official_test_evaluation"], "forbidden")

        report = json.loads(
            (ROOT / "docs/audits/tallip_kd/postlock_complete_metrics.json").read_text()
        )
        adopted = {
            "eng_laptop", "rus_restaurant", "tat_restaurant",
            "ukr_restaurant", "zho_restaurant",
        }
        self.assertEqual(set(report["datasets"]), adopted)
        for dataset, item in report["datasets"].items():
            self.assertEqual(set(item["systems"]), {"gold_repeat", "self_training"})
            self.assertIn("self_training_minus_gold_repeat", item["control_comparisons"])
            self.assertEqual(len(item["systems"]["self_training"]["single_seeds"]), 5)

        complementarity = json.loads(
            (ROOT / "docs/audits/tallip_kd/complementarity_error_analysis.json").read_text()
        )
        self.assertEqual(set(complementarity["datasets"]), adopted)
        for dataset, item in complementarity["datasets"].items():
            validity = item["silver_label_validity"]
            self.assertEqual(validity["empty_rows"], 0, dataset)
            self.assertEqual(validity["invalid_explicit_aspect_substrings"], 0, dataset)
            self.assertEqual(validity["invalid_explicit_opinion_substrings"], 0, dataset)
            self.assertEqual(validity["categories_absent_from_gold_train_inventory"], 0, dataset)
            self.assertEqual(validity["invalid_va_format_or_range"], 0, dataset)

        statistics_report = json.loads(
            (ROOT / "docs/audits/tallip_kd/statistical_robustness.json").read_text()
        )
        self.assertEqual(statistics_report["method"]["iterations"], 10000)
        self.assertEqual(statistics_report["method"]["seed"], 20260713)
        self.assertEqual(
            {name: len(items) for name, items in statistics_report["families"].items()},
            {
                "teacher_fidelity_8x2": 8,
                "adopted_kd_vs_baseline_dev": 5,
                "exposure_control_dev": 5,
                "same_text_self_training_dev": 5,
                "frozen_official_test_posthoc": 5,
            },
        )
        for items in statistics_report["families"].values():
            for item in items:
                self.assertIn("holm_adjusted_p", item)
                self.assertIn("bh_adjusted_p", item)
        official = json.loads(
            (ROOT / "docs/audits/tallip_kd/official_test/official_test_metrics.json").read_text()
        )
        archived = {
            item["dataset"]: item
            for item in official["comparisons"]
            if item["role"] == "adopted_kd_vs_baseline"
        }
        posthoc = statistics_report["families"]["frozen_official_test_posthoc"]
        self.assertEqual({item["dataset"] for item in posthoc}, set(archived))
        for item in posthoc:
            prior = archived[item["dataset"]]
            self.assertAlmostEqual(item["delta_a_minus_b"], prior["delta_a_minus_b"])
            self.assertEqual(item["classification"], prior["classification"])

        cost = json.loads(
            (ROOT / "docs/audits/tallip_kd/cost_efficiency.json").read_text()
        )
        self.assertEqual(len(cost["postlock_training"]), 10)
        self.assertEqual(cost["deployment"]["teacher_calls_at_student_inference"], 0)
        for provider in ("deepseek", "qwen"):
            self.assertEqual(cost["teacher_fidelity"][provider]["annotated_rows"], 1344)
            self.assertIsNone(cost["teacher_fidelity"][provider]["token_usage"])
            self.assertIsNone(cost["teacher_fidelity"][provider]["monetary_cost"])
        self.assertEqual(
            cost["teacher_fidelity"]["deepseek"]["timed_processed_rows"], 1344
        )
        self.assertEqual(
            cost["teacher_fidelity"]["qwen"]["timed_processed_rows"], 1264
        )


if __name__ == "__main__":
    unittest.main()
