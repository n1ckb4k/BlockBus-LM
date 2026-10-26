"""Character-level tokenizer (no external tokenizer dependency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CharTokenizer:
    chars: list[str]
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        if not self.chars:
            raise ValueError("chars must be non-empty")
        if self.unk_token in self.chars:
            raise ValueError("unk_token must not be a corpus character")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = list(self.chars)
        self.unk_id = len(self.chars)
        # `itos` includes the unknown token for decoding by id.
        self.itos.append(self.unk_token)

    @classmethod
    def from_text(cls, text: str, unk_token: str = "<unk>") -> "CharTokenizer":
        return cls(chars=sorted(set(text)), unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int], strip_unk: bool = False) -> str:
        out: list[str] = []
        for i in ids:
            idx = int(i)
            if idx == self.unk_id and strip_unk:
                continue
            if not 0 <= idx < len(self.itos):
                raise ValueError(f"token id out of range: {idx}")
            out.append(self.itos[idx])
        return "".join(out)

    def to_dict(self) -> dict:
        return {"chars": list(self.chars), "unk_token": self.unk_token}

    @classmethod
    def from_dict(cls, values: dict) -> "CharTokenizer":
        return cls(chars=list(values["chars"]), unk_token=values.get("unk_token", "<unk>"))
