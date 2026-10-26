"""Checkpoint tests: separate files, selective loading, and parameter counts."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from modular_llm.checkpoint import (
    _block_param_count,
    _sha256,
    checkpoint_size,
    load_dense_checkpoint,
    load_modular_checkpoint,
    save_dense_checkpoint,
    save_modular_checkpoint,
)
from modular_llm.config import ModelConfig
from modular_llm.model import DenseLM, ModularLM
from modular_llm.tokenizer import CharTokenizer


def tiny_config(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=20,
        block_size=16,
        d_model=32,
        n_heads=4,
        n_shared_layers=1,
        n_optional_blocks=3,
        d_ff=64,
        dropout=0.0,
    )
    base.update(kwargs)
    return ModelConfig(**base)


class CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = tiny_config()
        self.model = ModularLM(self.cfg, block_ids=[0, 1, 2])
        self.tok = CharTokenizer.from_text("hello world, this is a test")
        self.ckpt = self.dir / "ckpt"
        self.manifest = save_modular_checkpoint(
            self.model, self.tok, self.ckpt, metadata={"test": True}
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_files_and_manifest(self) -> None:
        self.assertTrue((self.ckpt / "shared.pt").exists())
        for i in range(3):
            self.assertTrue((self.ckpt / "blocks" / f"block_{i}.pt").exists())
        self.assertEqual(self.manifest["saved_block_ids"], [0, 1, 2])
        self.assertGreater(self.manifest["estimated_full_total_parameter_count"], 0)
        self.assertGreater(checkpoint_size(self.ckpt), 0)
        self.assertEqual(self.manifest["shared"]["sha256"], _sha256(self.ckpt / "shared.pt"))
        actual_block_n = self.model.block_parameter_count(0)
        self.assertEqual(_block_param_count(self.cfg), actual_block_n)
        self.assertEqual(
            self.manifest["estimated_full_total_parameter_count"],
            self.model.shared_parameter_count() + 3 * actual_block_n,
        )

    def test_load_shared_only(self) -> None:
        m, tok, manifest = load_modular_checkpoint(self.ckpt, selected_blocks=[])
        self.assertEqual(m.loaded_block_ids, ())
        self.assertEqual(m.active_parameter_count(()), m.shared_parameter_count())
        self.assertEqual(m.loaded_parameter_count(), m.shared_parameter_count())
        self.assertEqual(tok.vocab_size, self.tok.vocab_size)
        self.assertFalse(any(p.requires_grad for p in m.parameters()))
        x = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.inference_mode():
            self.assertEqual(m(x).shape, (1, 8, self.cfg.vocab_size))
        self.assertIn("config", manifest)

    def test_load_selected_block_reads_only_selected(self) -> None:
        real_load = torch.load
        calls: list[str] = []

        def tracked(*args, **kwargs):
            calls.append(str(args[0]))
            return real_load(*args, **kwargs)

        with patch("torch.load", side_effect=tracked):
            m, _, _ = load_modular_checkpoint(self.ckpt, selected_blocks=[1])
        self.assertEqual(m.loaded_block_ids, (1,))
        self.assertEqual(
            m.loaded_parameter_count(),
            m.shared_parameter_count() + m.block_parameter_count(1),
        )
        joined = "\n".join(calls).lower()
        self.assertIn("shared.pt", joined)
        self.assertIn("block_1.pt", joined)
        self.assertNotIn("block_0.pt", joined)
        self.assertNotIn("block_2.pt", joined)

    def test_unavailable_block_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_modular_checkpoint(self.ckpt, selected_blocks=[7])

    def test_sha256_verification_detects_tampering(self) -> None:
        copy_dir = self.dir / "tampered"
        shutil.copytree(self.ckpt, copy_dir)
        with (copy_dir / "blocks" / "block_1.pt").open("ab") as f:
            f.write(b"corruption")
        with self.assertRaises(ValueError):
            load_modular_checkpoint(copy_dir, selected_blocks=[1], verify_sha256=True)

    def test_dense_checkpoint_roundtrip(self) -> None:
        dense = DenseLM(self.cfg, n_layers=4)
        path = self.dir / "dense.pt"
        info = save_dense_checkpoint(dense, self.tok, path)
        self.assertEqual(info["n_layers"], 4)
        loaded, tok, _ = load_dense_checkpoint(path)
        self.assertEqual(loaded.parameter_count(), dense.parameter_count())
        self.assertEqual(tok.vocab_size, self.tok.vocab_size)
        x = torch.randint(0, self.cfg.vocab_size, (2, 8))
        with torch.inference_mode():
            torch.testing.assert_close(loaded(x), dense.eval()(x))


if __name__ == "__main__":
    unittest.main()
