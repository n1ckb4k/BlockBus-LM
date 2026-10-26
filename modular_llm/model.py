"""Causal Transformer components and the modular language model.

The optional blocks are intentionally ordinary Transformer *layers* wrapped in a
defined residual-bus interface.  They are not arbitrary slices of a dense model:
each block owns its LayerNorms, attention, MLP, and (optionally) a layer scale.
A block that is not selected is not called at all.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerLayer(nn.Module):
    """A pre-LN Transformer layer exposing its residual delta.

    `delta(x)` is the update written back to the residual stream; `forward(x)`
    is the conventional `x + delta(x)`.
    """

    def __init__(self, config: ModelConfig, use_layer_scale: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.full((), float(config.layer_scale_init)))
        else:
            self.register_parameter("layer_scale", None)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.ln_1(x))
        m = self.mlp(self.ln_2(x + a))
        return a + m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 if self.layer_scale is None else self.layer_scale
        return x + scale * self.delta(x)


class SharedTrunk(nn.Module):
    """Mandatory language component: embeddings, layers, final norm, LM head."""

    def __init__(self, config: ModelConfig, n_layers: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.n_layers = config.n_shared_layers if n_layers is None else n_layers
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(self.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled initialization for residual output projections.
        residual_layers = max(1, self.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(param, mean=0.0, std=config.init_std / math.sqrt(2 * residual_layers))
        if config.tie_weights:
            self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos)[None, :, :])
        for layer in self.layers:
            x = layer(x)
        return x

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(h))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(idx))


class ModularLM(nn.Module):
    """Shared trunk plus zero or more separately loadable optional blocks.

    Parameters
    ----------
    config:
        Architecture metadata, including the total number of optional blocks
        that exist in the full model.
    block_ids:
        Optional block indices materialized in this instance.  For a full training
        model this is `range(config.n_optional_blocks)`.  For inference it may be
        only the selected blocks, so unselected block parameters are absent.
    """

    def __init__(
        self,
        config: ModelConfig,
        block_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.shared = SharedTrunk(config)
        if block_ids is None:
            block_ids = ()
        block_ids = tuple(int(i) for i in block_ids)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block_ids must be unique")
        for i in block_ids:
            if not 0 <= i < config.n_optional_blocks:
                raise ValueError(f"block id {i} outside [0, {config.n_optional_blocks})")
        use_scale = config.composition == "parallel"
        self.blocks = nn.ModuleDict(
            {
                str(i): TransformerLayer(config, use_layer_scale=use_scale)
                for i in block_ids
            }
        )

        # Initialize optional blocks with the same scheme as shared layers.
        for block in self.blocks.values():
            block.apply(SharedTrunk._init_weights)
            for name, param in block.named_parameters():
                if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                    nn.init.normal_(
                        param,
                        mean=0.0,
                        std=config.init_std
                        / math.sqrt(2 * config.n_total_transformer_layers),
                    )

    @property
    def loaded_block_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(k) for k in self.blocks.keys()))

    def _validate_active(self, active_blocks: Iterable[int] | None) -> tuple[int, ...]:
        if active_blocks is None:
            return self.loaded_block_ids
        active = tuple(int(i) for i in active_blocks)
        if len(set(active)) != len(active):
            raise ValueError("active_blocks contains duplicates")
        for i in active:
            if not 0 <= i < self.config.n_optional_blocks:
                raise ValueError(f"active block {i} outside [0, {self.config.n_optional_blocks})")
            if str(i) not in self.blocks:
                raise ValueError(
                    f"block {i} is selected but not loaded; loaded={self.loaded_block_ids}"
                )
        return tuple(sorted(active))

    def _apply_blocks(self, h: torch.Tensor, active: Sequence[int]) -> torch.Tensor:
        if self.config.composition == "sequential":
            for i in active:
                h = h + self.blocks[str(i)].delta(h)
        else:
            delta = None
            for i in active:
                block = self.blocks[str(i)]
                term = block.layer_scale * block.delta(h)
                delta = term if delta is None else delta + term
            if delta is not None:
                h = h + delta
        return h

    def forward(
        self,
        idx: torch.Tensor,
        active_blocks: Iterable[int] | None = None,
    ) -> torch.Tensor:
        active = self._validate_active(active_blocks)
        h = self.shared.encode(idx)
        h = self._apply_blocks(h, active)
        return self.shared.decode(h)

    def forward_multi(
        self,
        idx: torch.Tensor,
        active_blocks: Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (shared-only logits, selected-config logits) from one encode.

        The shared encode is computed once.  This is used only by the optional
        shared-auxiliary training objective; inference uses `forward`.
        """
        active = self._validate_active(active_blocks)
        h = self.shared.encode(idx)
        logits_shared = self.shared.decode(h)
        if not active:
            return logits_shared, logits_shared
        h_active = self._apply_blocks(h, active)
        return logits_shared, self.shared.decode(h_active)

    def shared_parameter_count(self) -> int:
        return sum(p.numel() for p in self.shared.parameters())

    def block_parameter_count(self, block_id: int) -> int:
        return sum(p.numel() for p in self.blocks[str(block_id)].parameters())

    def loaded_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self, active_blocks: Iterable[int] | None = None) -> int:
        active = self._validate_active(active_blocks)
        total = self.shared_parameter_count()
        for i in active:
            total += self.block_parameter_count(i)
        return total


class DenseLM(nn.Module):
    """Always-on dense Transformer baseline with a configurable depth."""

    def __init__(self, config: ModelConfig, n_layers: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.n_layers = config.n_shared_layers + config.n_optional_blocks if n_layers is None else n_layers
        self.trunk = SharedTrunk(config, n_layers=self.n_layers)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.trunk(idx)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
