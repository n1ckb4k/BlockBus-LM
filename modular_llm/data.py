"""Dataset loading, deterministic splits, and causal-LM batching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .tokenizer import CharTokenizer


@dataclass
class TextData:
    text: str
    train_text: str
    val_text: str
    test_text: str
    tokenizer: CharTokenizer
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    source_path: str
    sha256: str
    split_fractions: tuple[float, float, float] = (0.90, 0.05, 0.05)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split_fractions: tuple[float, float, float] = (0.90, 0.05, 0.05),
        tokenizer: CharTokenizer | None = None,
    ) -> "TextData":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"empty dataset: {path}")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        train_text, val_text, test_text = split_text(text, split_fractions)
        tok = tokenizer or CharTokenizer.from_text(train_text)

        def encode(s: str) -> torch.Tensor:
            return torch.tensor(tok.encode(s), dtype=torch.long)

        return cls(
            text=text,
            train_text=train_text,
            val_text=val_text,
            test_text=test_text,
            tokenizer=tok,
            train_ids=encode(train_text),
            val_ids=encode(val_text),
            test_ids=encode(test_text),
            source_path=str(path),
            sha256=sha,
            split_fractions=split_fractions,
        )

    def ids_for_split(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_ids
        if split == "val":
            return self.val_ids
        if split == "test":
            return self.test_ids
        raise ValueError(f"unknown split: {split}")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.ids_for_split(split)
        if len(data) <= block_size:
            raise ValueError(f"split {split} is too short for block_size={block_size}")
        starts = torch.randint(
            low=0,
            high=len(data) - block_size,
            size=(batch_size,),
            generator=generator,
        )
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
        return x, y

    def iter_eval_batches(
        self,
        split: str,
        batch_size: int,
        block_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping validation/test windows, batched deterministically."""
        data = self.ids_for_split(split)
        starts = list(range(0, len(data) - block_size, block_size))
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            x = torch.stack([data[i : i + block_size] for i in batch_starts])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in batch_starts])
            yield x, y

    def summary(self) -> dict:
        return {
            "source_path": self.source_path,
            "sha256": self.sha256,
            "chars": len(self.text),
            "vocab_size": self.tokenizer.vocab_size,
            "train_tokens": int(len(self.train_ids)),
            "val_tokens": int(len(self.val_ids)),
            "test_tokens": int(len(self.test_ids)),
            "split_fractions": list(self.split_fractions),
            "train_unknown_tokens": int((self.train_ids == self.tokenizer.unk_id).sum()),
            "val_unknown_tokens": int((self.val_ids == self.tokenizer.unk_id).sum()),
            "test_unknown_tokens": int((self.test_ids == self.tokenizer.unk_id).sum()),
        }


def split_text(
    text: str,
    fractions: tuple[float, float, float] = (0.90, 0.05, 0.05),
) -> tuple[str, str, str]:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    n = len(text)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train = text[:n_train]
    val = text[n_train : n_train + n_val]
    test = text[n_train + n_val :]
    return train, val, test
