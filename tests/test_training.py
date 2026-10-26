"""End-to-end smoke tests for the training loops."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from modular_llm.checkpoint import load_dense_checkpoint, load_modular_checkpoint
from modular_llm.config import ModelConfig
from modular_llm.data import TextData
from modular_llm.train import RunSpec, train_dense, train_modular


class TrainingSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        text = ("The quick brown fox jumps over the lazy dog. " * 60).strip()
        self.data_path = self.root / "tiny.txt"
        self.data_path.write_text(text, encoding="utf-8")
        self.data = TextData.from_file(self.data_path)
        self.cfg = ModelConfig(
            vocab_size=self.data.tokenizer.vocab_size,
            block_size=16,
            d_model=32,
            n_heads=4,
            n_shared_layers=1,
            n_optional_blocks=2,
            d_ff=64,
            dropout=0.0,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _spec(self, name: str, mode: str) -> RunSpec:
        out = self.root / name
        out.mkdir(parents=True, exist_ok=True)
        return RunSpec(
            mode=mode,
            out_dir=out,
            seed=7,
            steps=2,
            batch_size=2,
            block_size=16,
            lr=1e-3,
            weight_decay=0.0,
            warmup=1,
            eval_every=0,
            eval_max_batches=None,
            log_every=0,
            grad_clip=1.0,
            threads=1,
            config_set="all",
            excluded=[],
            selected_blocks=(0, 1),
            model_config=self.cfg,
            note="smoke",
        )

    def test_joint_training_writes_metrics_and_checkpoints(self) -> None:
        spec = self._spec("joint", "joint")
        metrics = train_modular(spec, self.data)
        self.assertTrue((spec.out_dir / "metrics.json").exists())
        self.assertTrue((spec.out_dir / "checkpoints" / "final" / "shared.pt").exists())
        self.assertTrue(
            (spec.out_dir / "checkpoints" / "final" / "blocks" / "block_1.pt").exists()
        )
        self.assertEqual(len(metrics["evaluation"]["test"]), 4)
        self.assertEqual(metrics["parameter_counts"]["n_optional_blocks_total"], 2)
        self.assertEqual(sum(metrics["training"]["config_counts"].values()), 2)
        with (spec.out_dir / "metrics.json").open(encoding="utf-8") as f:
            json.load(f)

    def test_separate_shared_only(self) -> None:
        spec = self._spec("separate_shared", "separate")
        spec.selected_blocks = ()
        metrics = train_modular(spec, self.data)
        self.assertEqual(metrics["parameter_counts"]["loaded_blocks"], [])
        self.assertEqual(len(metrics["evaluation"]["val"]), 1)

    def test_dense_training(self) -> None:
        spec = self._spec("dense", "dense")
        metrics = train_dense(spec, self.data)
        self.assertTrue((spec.out_dir / "checkpoints" / "final.pt").exists())
        self.assertEqual(metrics["parameter_counts"]["n_layers"], 3)
        self.assertIn("dense", metrics["evaluation"]["test"])

    def test_frozen_shared_can_start_from_dense_shared(self) -> None:
        dense_spec = self._spec("dense_shared_base", "dense_shared")
        train_dense(dense_spec, self.data)
        dense_path = dense_spec.out_dir / "checkpoints" / "final.pt"
        self.assertTrue(dense_path.exists())

        spec = self._spec("frozen_from_dense", "joint")
        spec.init_shared_dense = str(dense_path)
        spec.freeze_shared = True
        spec.selected_blocks = (0, 1)
        metrics = train_modular(spec, self.data)
        parameters = metrics["parameter_counts"]
        self.assertEqual(
            parameters["loaded_parameters"],
            parameters["shared_parameters"] + 2 * parameters["per_block_parameters"],
        )
        self.assertEqual(len(metrics["evaluation"]["test"]), 4)

        dense, _, _ = load_dense_checkpoint(dense_path)
        modular, _, _ = load_modular_checkpoint(
            spec.out_dir / "checkpoints" / "final", selected_blocks=[]
        )
        for name, tensor in dense.trunk.state_dict().items():
            torch.testing.assert_close(
                tensor, modular.shared.state_dict()[name], msg=f"shared mismatch: {name}"
            )


if __name__ == "__main__":
    unittest.main()
