"""Model configuration with explicit, serializable defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    The shared component always exists. Optional blocks are separate Transformer
    layers attached to the shared residual bus.
    """

    vocab_size: int = 65
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_shared_layers: int = 4
    n_optional_blocks: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    tie_weights: bool = True
    composition: str = "sequential"  # "sequential" or "parallel"
    layer_scale_init: float = 1.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_shared_layers < 1:
            raise ValueError("n_shared_layers must be at least 1")
        if self.n_optional_blocks < 0:
            raise ValueError("n_optional_blocks must be non-negative")
        if self.composition not in {"sequential", "parallel"}:
            raise ValueError("composition must be 'sequential' or 'parallel'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_total_transformer_layers(self) -> int:
        return self.n_shared_layers + self.n_optional_blocks
