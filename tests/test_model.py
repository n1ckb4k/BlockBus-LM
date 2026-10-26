"""Unit tests for architecture, optionality, and parameter accounting."""

from __future__ import annotations

import unittest

import torch

from modular_llm.config import ModelConfig
from modular_llm.model import DenseLM, ModularLM


def tiny_config(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=32,
        block_size=16,
        d_model=32,
        n_heads=4,
        n_shared_layers=2,
        n_optional_blocks=2,
        d_ff=64,
        dropout=0.0,
        composition="sequential",
    )
    base.update(kwargs)
    return ModelConfig(**base)


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = tiny_config()
        self.model = ModularLM(self.cfg, block_ids=[0, 1])
        self.x = torch.randint(0, self.cfg.vocab_size, (2, 12))

    def test_output_shape_for_every_subset(self) -> None:
        self.model.eval()
        for cfg in [(), (0,), (1,), (0, 1)]:
            with torch.inference_mode():
                logits = self.model(self.x, active_blocks=cfg)
            self.assertEqual(logits.shape, (2, 12, self.cfg.vocab_size))

    def test_active_parameter_accounting(self) -> None:
        shared = self.model.shared_parameter_count()
        b0 = self.model.block_parameter_count(0)
        b1 = self.model.block_parameter_count(1)
        self.assertEqual(self.model.active_parameter_count(()), shared)
        self.assertEqual(self.model.active_parameter_count((0,)), shared + b0)
        self.assertEqual(self.model.active_parameter_count((0, 1)), shared + b0 + b1)
        self.assertEqual(self.model.loaded_parameter_count(), shared + b0 + b1)

    def test_inactive_block_not_called(self) -> None:
        self.model.eval()

        def fail(_x):
            raise AssertionError("inactive block was executed")

        self.model.blocks["0"].delta = fail
        self.model.blocks["1"].delta = fail
        with torch.inference_mode():
            out_shared = self.model(self.x, active_blocks=())
        self.assertEqual(out_shared.shape, (2, 12, self.cfg.vocab_size))

        # Only block 0 may run.
        original = self.model.blocks["0"].delta
        self.model.blocks["0"].delta = lambda x: torch.zeros_like(x)
        with torch.inference_mode():
            out_b0 = self.model(self.x, active_blocks=(0,))
        self.assertEqual(out_b0.shape, (2, 12, self.cfg.vocab_size))
        self.model.blocks["0"].delta = original

    def test_gradients_only_for_active_blocks(self) -> None:
        self.model.train()
        for p in self.model.parameters():
            p.grad = None
        x = torch.randint(0, self.cfg.vocab_size, (2, 8))
        y = torch.randint(0, self.cfg.vocab_size, (2, 8))
        logits = self.model(x, active_blocks=(0,))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), y.reshape(-1))
        loss.backward()
        self.assertIsNotNone(self.model.blocks["0"].mlp.fc_in.weight.grad)
        self.assertIsNone(self.model.blocks["1"].mlp.fc_in.weight.grad)
        self.assertIsNotNone(self.model.shared.token_emb.weight.grad)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.model(self.x, active_blocks=(0, 0))
        with self.assertRaises(ValueError):
            self.model(self.x, active_blocks=(2,))
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, self.cfg.vocab_size, (1, 17)))
        with self.assertRaises(ValueError):
            ModularLM(self.cfg, block_ids=[0, 0])

    def test_loaded_subset_cannot_run_unloaded_block(self) -> None:
        m = ModularLM(self.cfg, block_ids=[1])
        self.assertEqual(m.loaded_block_ids, (1,))
        m.eval()
        with torch.inference_mode():
            out = m(self.x, active_blocks=(1,))
        self.assertEqual(out.shape, (2, 12, self.cfg.vocab_size))
        with self.assertRaises(ValueError):
            m(self.x, active_blocks=(0,))

    def test_dense_parameter_match_for_sequential(self) -> None:
        dense = DenseLM(self.cfg)
        self.assertEqual(dense.n_layers, 4)
        self.assertEqual(dense.parameter_count(), self.model.loaded_parameter_count())
        self.assertEqual(dense(self.x).shape, (2, 12, self.cfg.vocab_size))

    def test_parallel_composition(self) -> None:
        cfg = tiny_config(composition="parallel")
        m = ModularLM(cfg, block_ids=[0, 1])
        m.eval()
        self.assertEqual(m.blocks["0"].layer_scale.numel(), 1)
        for active in [(), (0,), (1,), (0, 1)]:
            with torch.inference_mode():
                out = m(self.x, active_blocks=active)
            self.assertEqual(out.shape, (2, 12, self.cfg.vocab_size))
            self.assertTrue(torch.isfinite(out).all())

    def test_eval_is_deterministic(self) -> None:
        self.model.eval()
        with torch.inference_mode():
            a = self.model(self.x, active_blocks=(0, 1))
            b = self.model(self.x, active_blocks=(0, 1))
        torch.testing.assert_close(a, b)

    def test_forward_multi_matches_individual_paths(self) -> None:
        self.model.eval()
        with torch.inference_mode():
            shared, active = self.model.forward_multi(self.x, active_blocks=(0, 1))
            expected_shared = self.model(self.x, active_blocks=())
            expected_active = self.model(self.x, active_blocks=(0, 1))
        torch.testing.assert_close(shared, expected_shared)
        torch.testing.assert_close(active, expected_active)
        with torch.inference_mode():
            shared2, active2 = self.model.forward_multi(self.x, active_blocks=())
        torch.testing.assert_close(shared2, active2)


if __name__ == "__main__":
    unittest.main()
