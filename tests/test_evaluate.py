"""Evaluation, generation, and latency tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from modular_llm.data import TextData
from modular_llm.evaluate import (
    all_configs,
    benchmark_modular_forward,
    config_label,
    evaluate_modular_configs,
    generate_modular,
    presets,
)
from modular_llm.model import ModularLM
from modular_llm.config import ModelConfig


class EvaluateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        text = ("The quick brown fox jumps over the lazy dog. " * 80).strip()
        path = Path(cls.tmp.name) / "tiny.txt"
        path.write_text(text, encoding="utf-8")
        cls.data = TextData.from_file(path)
        cls.cfg = ModelConfig(
            vocab_size=cls.data.tokenizer.vocab_size,
            block_size=16,
            d_model=32,
            n_heads=4,
            n_shared_layers=1,
            n_optional_blocks=2,
            d_ff=64,
            dropout=0.0,
        )
        cls.model = ModularLM(cls.cfg, block_ids=[0, 1]).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_config_enumeration(self) -> None:
        configs = all_configs(3)
        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0], ())
        self.assertEqual(configs[-1], (0, 1, 2))
        self.assertEqual(config_label(()), "shared")
        self.assertEqual(config_label((0, 2)), "B0+B2")
        self.assertEqual(len(presets(3)["singletons"]), 4)
        self.assertEqual(len(presets(3)["no_all"]), 7)

    def test_evaluate_all_configs(self) -> None:
        configs = all_configs(2)
        results = evaluate_modular_configs(
            self.model,
            self.data,
            self.data.tokenizer,
            configs,
            split="val",
            batch_size=2,
            block_size=16,
        )
        self.assertEqual(set(results), {"shared", "B0", "B1", "B0+B1"})
        for result in results.values():
            self.assertGreater(result["tokens"], 0)
            self.assertGreater(result["loss"], 0)
            self.assertTrue(0 <= result["token_accuracy"] <= 1)
            self.assertIsNotNone(result["per_class"]["lowercase"]["loss"])
        self.assertLess(
            results["shared"]["active_parameters"], results["B0+B1"]["active_parameters"]
        )

    def test_generate(self) -> None:
        text = generate_modular(
            self.model,
            self.data.tokenizer,
            "The ",
            active_blocks=(0, 1),
            max_new_tokens=12,
            temperature=0.8,
            top_k=10,
            seed=3,
        )
        self.assertTrue(text.startswith("The "))
        self.assertGreater(len(text), len("The "))

    def test_benchmark(self) -> None:
        out = benchmark_modular_forward(
            self.model,
            active_blocks=(0,),
            seq_len=16,
            batch_size=1,
            warmup=1,
            iterations=3,
        )
        self.assertEqual(out["tokens_per_run"], 16)
        self.assertGreater(out["median_ms"], 0)
        self.assertGreater(out["tokens_per_second_median"], 0)


if __name__ == "__main__":
    unittest.main()
