"""Data split, tokenizer, and batching tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from modular_llm.data import TextData, split_text
from modular_llm.tokenizer import CharTokenizer


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.text = ("The quick brown fox jumps over the lazy dog. " * 100).strip()
        self.path = Path(self.tmp.name) / "tiny.txt"
        self.path.write_text(self.text, encoding="utf-8")
        self.data = TextData.from_file(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_splits_and_hash(self) -> None:
        self.assertAlmostEqual(
            len(self.data.train_ids) / len(self.data.tokenizer.encode(self.text)),
            0.9,
            places=2,
        )
        self.assertEqual(len(self.data.sha256), 64)
        self.assertEqual(self.data.tokenizer.vocab_size, len(set(self.text)) + 1)
        self.assertEqual(self.data.summary()["val_unknown_tokens"], 0)

    def test_batch_shapes_and_targets(self) -> None:
        g = torch.Generator().manual_seed(0)
        x, y = self.data.get_batch("train", 4, 16, g)
        self.assertEqual(x.shape, (4, 16))
        self.assertEqual(y.shape, (4, 16))
        for row in range(4):
            torch.testing.assert_close(x[row, 1:], y[row, :-1])

    def test_batch_reproducible_with_generator(self) -> None:
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        x1, y1 = self.data.get_batch("val", 3, 12, g1)
        x2, y2 = self.data.get_batch("val", 3, 12, g2)
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)

    def test_eval_batches_deterministic(self) -> None:
        a = list(self.data.iter_eval_batches("val", 4, 16))
        b = list(self.data.iter_eval_batches("val", 4, 16))
        self.assertEqual(len(a), len(b))
        for (xa, ya), (xb, yb) in zip(a, b):
            torch.testing.assert_close(xa, xb)
            torch.testing.assert_close(ya, yb)

    def test_split_text_validation(self) -> None:
        with self.assertRaises(ValueError):
            split_text("abc", (0.5, 0.5, 0.5))

    def test_tokenizer_roundtrip_and_unknown(self) -> None:
        tok = CharTokenizer.from_text("abc")
        ids = tok.encode("abcX")
        self.assertEqual(ids[:3], [tok.stoi["a"], tok.stoi["b"], tok.stoi["c"]])
        self.assertEqual(ids[3], tok.unk_id)
        self.assertEqual(tok.decode(ids[:3]), "abc")
        self.assertEqual(tok.decode(ids), "abc<unk>")


if __name__ == "__main__":
    unittest.main()
