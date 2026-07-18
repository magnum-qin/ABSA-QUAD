import json
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


class TallipKDTests(unittest.TestCase):
    def test_cloud_and_local_teacher_prompts_are_identical(self):
        def load_module(name, relative):
            spec = importlib.util.spec_from_file_location(name, ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

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


if __name__ == "__main__":
    unittest.main()
