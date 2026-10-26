"""Evaluation, generation, and latency measurement."""

from __future__ import annotations

import math
import statistics
import string
import time
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from .data import TextData
from .model import DenseLM, ModularLM
from .tokenizer import CharTokenizer


CHAR_CLASSES = [
    "space",
    "newline",
    "lowercase",
    "uppercase",
    "digit",
    "punctuation",
    "other",
]


def char_class(ch: str) -> str:
    if ch == " ":
        return "space"
    if ch in "\n\r\t":
        return "newline"
    if ch.islower():
        return "lowercase"
    if ch.isupper():
        return "uppercase"
    if ch.isdigit():
        return "digit"
    if ch in string.punctuation:
        return "punctuation"
    return "other"


def class_ids_for_tokenizer(tokenizer: CharTokenizer) -> torch.Tensor:
    mapping = {name: i for i, name in enumerate(CHAR_CLASSES)}
    return torch.tensor(
        [mapping[char_class(ch)] for ch in tokenizer.itos], dtype=torch.long
    )


def config_label(blocks: Iterable[int] | None) -> str:
    blocks = tuple(sorted(int(i) for i in (blocks or ())))
    if not blocks:
        return "shared"
    return "+".join(f"B{i}" for i in blocks)


def all_configs(n_optional: int) -> list[tuple[int, ...]]:
    configs: list[tuple[int, ...]] = [()]
    for bits in range(1, n_optional + 1):
        # combinations of exactly `bits` blocks
        def rec(start: int, remaining: int, prefix: tuple[int, ...]) -> None:
            if remaining == 0:
                configs.append(prefix)
                return
            for i in range(start, n_optional):
                rec(i + 1, remaining - 1, prefix + (i,))

        rec(0, bits, ())
    return configs


def presets(n_optional: int) -> dict[str, list[tuple[int, ...]]]:
    every = all_configs(n_optional)
    return {
        "all": every,
        "singletons": [c for c in every if len(c) <= 1],
        "no_all": [c for c in every if len(c) != n_optional],
        "no_pairs": [c for c in every if len(c) <= 1],
    }


def _evaluate_forward(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    data: TextData,
    tokenizer: CharTokenizer,
    split: str,
    batch_size: int,
    block_size: int,
    max_batches: int | None = None,
) -> dict:
    class_ids = class_ids_for_tokenizer(tokenizer)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    per_class_loss = [0.0] * len(CHAR_CLASSES)
    per_class_count = [0] * len(CHAR_CLASSES)
    n_batches = 0

    with torch.inference_mode():
        for x, y in data.iter_eval_batches(split, batch_size, block_size):
            if max_batches is not None and n_batches >= max_batches:
                break
            logits = forward_fn(x)
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="none",
            )
            flat_y = y.reshape(-1)
            total_loss += float(token_loss.sum().item())
            total_tokens += int(flat_y.numel())
            correct += int((logits.argmax(dim=-1).reshape(-1) == flat_y).sum().item())
            classes = class_ids[flat_y.cpu()]
            for cls in range(len(CHAR_CLASSES)):
                mask = classes == cls
                count = int(mask.sum().item())
                if count:
                    per_class_loss[cls] += float(token_loss.cpu()[mask].sum().item())
                    per_class_count[cls] += count
            n_batches += 1

    if total_tokens == 0:
        raise ValueError("no evaluation tokens")
    loss = total_loss / total_tokens
    return {
        "split": split,
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "bits_per_char": loss / math.log(2.0),
        "token_accuracy": correct / total_tokens,
        "tokens": total_tokens,
        "batches": n_batches,
        "per_class": {
            name: {
                "loss": per_class_loss[i] / per_class_count[i]
                if per_class_count[i]
                else None,
                "tokens": per_class_count[i],
            }
            for i, name in enumerate(CHAR_CLASSES)
        },
    }


def evaluate_modular(
    model: ModularLM,
    data: TextData,
    tokenizer: CharTokenizer,
    active_blocks: Iterable[int] | None,
    split: str = "val",
    batch_size: int = 32,
    block_size: int | None = None,
    max_batches: int | None = None,
) -> dict:
    block_size = block_size or model.config.block_size
    active = tuple(sorted(int(i) for i in (active_blocks or ())))
    was_training = model.training
    model.eval()

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x, active_blocks=active)

    try:
        out = _evaluate_forward(
            forward, data, tokenizer, split, batch_size, block_size, max_batches
        )
    finally:
        model.train(was_training)
    out.update(
        {
            "config": list(active),
            "config_label": config_label(active),
            "active_parameters": model.active_parameter_count(active),
        }
    )
    return out


def evaluate_dense(
    model: DenseLM,
    data: TextData,
    tokenizer: CharTokenizer,
    split: str = "val",
    batch_size: int = 32,
    block_size: int | None = None,
    max_batches: int | None = None,
) -> dict:
    block_size = block_size or model.config.block_size
    was_training = model.training
    model.eval()
    try:
        out = _evaluate_forward(
            model, data, tokenizer, split, batch_size, block_size, max_batches
        )
    finally:
        model.train(was_training)
    out.update(
        {
            "config": "dense",
            "config_label": "dense",
            "active_parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    return out


def evaluate_modular_configs(
    model: ModularLM,
    data: TextData,
    tokenizer: CharTokenizer,
    configs: Sequence[Sequence[int]],
    split: str = "val",
    batch_size: int = 32,
    block_size: int | None = None,
    max_batches: int | None = None,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for cfg in configs:
        cfg = tuple(sorted(int(i) for i in cfg))
        results[config_label(cfg)] = evaluate_modular(
            model,
            data,
            tokenizer,
            cfg,
            split=split,
            batch_size=batch_size,
            block_size=block_size,
            max_batches=max_batches,
        )
    return results


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None, generator: torch.Generator) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if top_k is not None and 0 < top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        cutoff = values[-1]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1).cpu()
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def generate(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: CharTokenizer,
    prompt: str,
    block_size: int,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 0,
) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.encode(" ")[0]]
    idx = torch.tensor([ids], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = forward_fn(idx_cond)[:, -1, :][0]
            next_id = _sample_next(logits, temperature, top_k, generator)
            idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
    return tokenizer.decode(idx[0].tolist())


def generate_modular(
    model: ModularLM,
    tokenizer: CharTokenizer,
    prompt: str,
    active_blocks: Iterable[int] | None,
    **kwargs,
) -> str:
    active = tuple(sorted(int(i) for i in (active_blocks or ())))
    was_training = model.training
    model.eval()
    try:
        return generate(
            lambda x: model(x, active_blocks=active),
            tokenizer,
            prompt,
            model.config.block_size,
            **kwargs,
        )
    finally:
        model.train(was_training)


def generate_dense(
    model: DenseLM,
    tokenizer: CharTokenizer,
    prompt: str,
    **kwargs,
) -> str:
    was_training = model.training
    model.eval()
    try:
        return generate(model, tokenizer, prompt, model.config.block_size, **kwargs)
    finally:
        model.train(was_training)


def _latency_stats(seconds: list[float], tokens_per_run: int) -> dict:
    ordered = sorted(seconds)
    median = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "runs": len(seconds),
        "tokens_per_run": tokens_per_run,
        "mean_ms": 1000.0 * statistics.fmean(seconds),
        "median_ms": 1000.0 * median,
        "p95_ms": 1000.0 * p95,
        "min_ms": 1000.0 * ordered[0],
        "max_ms": 1000.0 * ordered[-1],
        "tokens_per_second_median": tokens_per_run / median if median else None,
    }


def benchmark_modular_forward(
    model: ModularLM,
    active_blocks: Iterable[int] | None,
    seq_len: int = 128,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 50,
    seed: int = 0,
) -> dict:
    active = tuple(sorted(int(i) for i in (active_blocks or ())))
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), generator=g)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, active_blocks=active)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(x, active_blocks=active)
            times.append(time.perf_counter() - t0)
    out = _latency_stats(times, batch_size * seq_len)
    out.update(
        {
            "config": list(active),
            "config_label": config_label(active),
            "active_parameters": model.active_parameter_count(active),
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    )
    return out


def benchmark_dense_forward(
    model: DenseLM,
    seq_len: int = 128,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 50,
    seed: int = 0,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), generator=g)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    out = _latency_stats(times, batch_size * seq_len)
    out.update(
        {
            "config": "dense",
            "config_label": "dense",
            "active_parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    )
    return out
