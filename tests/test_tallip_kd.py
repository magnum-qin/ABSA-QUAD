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
        self.assertIn("verified 5 frozen tables", result.stdout)

        lock = json.loads(
            (ROOT / "docs/audits/tallip_kd/tallip_core_tables.lock.json").read_text()
        )
        self.assertEqual(lock["status"], "frozen")
        self.assertEqual(lock["table_count"], 5)
        self.assertEqual(len(lock["output"]["sha256"]), 64)
        paths = {item["path"] for item in lock["inputs"]}
        self.assertIn("scripts/kd/official_test.yaml", paths)
        self.assertIn("docs/audits/tallip_kd/ablation_evidence.json", paths)

        tables = (ROOT / lock["output"]["path"]).read_text()
        self.assertEqual(tables.count("## Table "), 5)
        self.assertIn("Inductive-safe KD five-seed", tables)
        self.assertIn("real − generated = +2.3pp", tables)

    def test_postlock_protocol_is_dev_only_and_repeat_is_deterministic(self):
        module = load_module("postlock_experiments", "scripts/kd/postlock_experiments.py")
        manifest = yaml.safe_load((ROOT / "scripts/kd/postlock_v1.yaml").read_text())
        protocol = manifest["protocol"]
        self.assertEqual(protocol["official_test_evaluation"], "forbidden")
        self.assertEqual(protocol["model_selection_split"], "dev_only")
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


if __name__ == "__main__":
    unittest.main()
