import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train import _atomic_torch_save, _replace_with_hardlink
from src.data.dataset import create_dataloader


class DummyEncoding(dict):
    def char_to_token(self, index):
        return index + 1 if 0 <= index < 6 else None


class DummyTokenizer:
    def __call__(self, text, max_length, **_kwargs):
        ids = [101] + [ord(char) % 97 + 1 for char in text[: max_length - 2]] + [102]
        ids = (ids + [0] * max_length)[:max_length]
        mask = [int(value != 0) for value in ids]
        offsets = [(0, 0)] + [(index, index + 1) for index in range(min(len(text), max_length - 2))]
        offsets = (offsets + [(0, 0)] * max_length)[:max_length]
        return DummyEncoding(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.tensor([mask]),
            offset_mapping=torch.tensor([offsets]),
        )


class TrainingFlowTest(unittest.TestCase):
    def test_atomic_checkpoint_and_hardlink_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, epoch, best = (root / name for name in ("latest.pt", "epoch.pt", "best.pt"))
            _atomic_torch_save({"value": torch.arange(5)}, latest)
            _replace_with_hardlink(latest, epoch)
            _replace_with_hardlink(epoch, best)

            self.assertEqual(os.stat(latest).st_ino, os.stat(epoch).st_ino)
            self.assertEqual(os.stat(epoch).st_ino, os.stat(best).st_ino)
            _atomic_torch_save({"value": torch.arange(7)}, latest)
            self.assertEqual(torch.load(epoch, weights_only=False)["value"].numel(), 5)
            self.assertEqual(torch.load(latest, weights_only=False)["value"].numel(), 7)

    def test_multiworker_loader_preserves_order_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "tiny.jsonl"
            with data.open("w", encoding="utf-8") as handle:
                for index in range(12):
                    handle.write(json.dumps({
                        "ID": str(index), "Text": f"text{index}", "Quadruplet": []
                    }) + "\n")

            common = dict(data_path=str(data), tokenizer=DummyTokenizer(), batch_size=3,
                          max_length=12, shuffle=False)
            single = create_dataloader(**common, num_workers=0)
            multi = create_dataloader(**common, num_workers=2, prefetch_factor=2)
            single_ids = [sample_id for batch in single for sample_id in batch["sample_ids"]]
            multi_ids = [sample_id for batch in multi for sample_id in batch["sample_ids"]]
            self.assertEqual(single_ids, multi_ids)


if __name__ == "__main__":
    unittest.main()
