"""Training entry point for joint, separately trained, and dense baselines.

Examples
--------
Joint model, all 8 subsets (shared + 3 optional blocks):
    python -m modular_llm.train --mode joint --out runs/joint_all --steps 3000

Held-out composition ablation (empty + singleton subsets only):
    python -m modular_llm.train --mode joint --config-set singletons --out runs/joint_singletons

Separately trained all-active configuration:
    python -m modular_llm.train --mode separate --blocks 0,1,2 --out runs/separate_012 --steps 375

Dense, seven always-on layers:
    python -m modular_llm.train --mode dense --out runs/dense_7 --steps 3000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .checkpoint import (
    checkpoint_size,
    load_dense_checkpoint,
    load_modular_checkpoint,
    save_dense_checkpoint,
    save_modular_checkpoint,
)
from .config import ModelConfig
from .data import TextData
from .evaluate import (
    all_configs,
    config_label,
    evaluate_dense,
    evaluate_modular_configs,
    presets,
)
from .metrics import collect_metrics, format_bytes, gpu_metrics, set_threads, timer
from .model import DenseLM, ModularLM
from .tokenizer import CharTokenizer
from .utils import append_jsonl, environment_summary, set_seed, write_json


@dataclass
class RunSpec:
    mode: str
    out_dir: Path
    seed: int
    steps: int
    batch_size: int
    block_size: int
    lr: float
    weight_decay: float
    warmup: int
    eval_every: int
    eval_max_batches: int | None
    log_every: int
    grad_clip: float
    threads: int
    config_set: str
    excluded: list[tuple[int, ...]]
    selected_blocks: tuple[int, ...]
    model_config: ModelConfig
    note: str
    full_bias: int = 0
    empty_bias: int = 0
    full_warmup_steps: int = 0
    block_lr_mult: float = 1.0
    shared_aux_weight: float = 0.0
    init_modular: str | None = None
    init_shared_dense: str | None = None
    freeze_shared: bool = False


class ConfigSampler:
    """Balanced shuffled cycles over configurations, with optional weighting.

    `full_bias` and `empty_bias` add extra copies of the all-blocks and
    shared-only configurations to the sampling pool.  The evaluation set remains
    the unique allowed list; only the training distribution changes.
    """

    def __init__(
        self,
        configs: Sequence[Sequence[int]],
        seed: int,
        full_bias: int = 0,
        empty_bias: int = 0,
        n_optional: int | None = None,
    ) -> None:
        if not configs:
            raise ValueError("configs must be non-empty")
        self.configs = [tuple(sorted(int(i) for i in c)) for c in configs]
        self.rng = random.Random(seed)
        self.queue: list[tuple[int, ...]] = []
        self.counts = {config_label(c): 0 for c in self.configs}
        self.total = 0
        self.pool: list[tuple[int, ...]] = []
        for c in self.configs:
            weight = 1
            if n_optional is not None and len(c) == n_optional:
                weight += max(0, int(full_bias))
            if len(c) == 0:
                weight += max(0, int(empty_bias))
            self.pool.extend([c] * weight)
        if not self.pool:
            raise ValueError("sampling pool is empty")

    def record(self, cfg: Sequence[int]) -> None:
        cfg = tuple(sorted(int(i) for i in cfg))
        self.counts[config_label(cfg)] = self.counts.get(config_label(cfg), 0) + 1
        self.total += 1

    def next(self) -> tuple[int, ...]:
        if not self.queue:
            self.queue = list(self.pool)
            self.rng.shuffle(self.queue)
        cfg = self.queue.pop()
        self.record(cfg)
        return cfg


def _parse_blocks(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(sorted(int(p) for p in parts))


def _parse_excluded(values: Sequence[str] | None, n_optional: int) -> list[tuple[int, ...]]:
    out = []
    for value in values or []:
        cfg = _parse_blocks(value)
        if any(i < 0 or i >= n_optional for i in cfg):
            raise ValueError(f"excluded config {value} contains an invalid block id")
        out.append(cfg)
    return out


def _build_allowed(spec: RunSpec) -> list[tuple[int, ...]]:
    if spec.mode == "separate":
        return [spec.selected_blocks]
    base = presets(spec.model_config.n_optional_blocks)[spec.config_set]
    excluded = set(spec.excluded)
    allowed = [c for c in base if c not in excluded]
    if not allowed:
        raise ValueError("no allowed configurations remain")
    return allowed


def _build_eval_configs(spec: RunSpec) -> list[tuple[int, ...]]:
    if spec.mode == "dense" or spec.mode == "dense_shared":
        return [()]
    if spec.mode == "joint":
        return all_configs(spec.model_config.n_optional_blocks)
    # A separately trained model still supports any subset of its loaded blocks;
    # evaluating all of them is a zero-shot omission test, not a trained result.
    selected = spec.selected_blocks
    subsets: list[tuple[int, ...]] = [()]
    for r in range(1, len(selected) + 1):
        for combo in itertools.combinations(selected, r):
            subsets.append(tuple(combo))
    return subsets


def _lr_lambda(step: int, warmup: int, total_steps: int) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    block_lr_mult: float = 1.0,
):
    groups: list[dict] = []
    for prefix, lr_mult in (("shared.", 1.0), ("trunk.", 1.0), ("blocks.", block_lr_mult)):
        decay = []
        no_decay = []
        for name, p in model.named_parameters():
            if not p.requires_grad or not name.startswith(prefix):
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        if decay:
            groups.append(
                {"params": decay, "weight_decay": weight_decay, "lr": lr * lr_mult}
            )
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr * lr_mult})
    if not groups:
        # Fallback for unexpected module names.
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": params, "weight_decay": weight_decay}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def _param_summary(model: torch.nn.Module, spec: RunSpec) -> dict:
    if isinstance(model, ModularLM):
        block_n = sum(p.numel() for p in model.blocks[str(model.loaded_block_ids[0])].parameters()) if model.loaded_block_ids else None
        if block_n is None:
            # Derive from architecture when no block was loaded.
            d = spec.model_config.d_model
            block_n = (
                4 * d * d
                + 2 * d * spec.model_config.d_ff
                + 5 * d
                + spec.model_config.d_ff
                + (1 if spec.model_config.composition == "parallel" else 0)
            )
        shared_n = model.shared_parameter_count()
        full_total = shared_n + spec.model_config.n_optional_blocks * block_n
        active = {}
        for cfg in all_configs(spec.model_config.n_optional_blocks):
            if all(str(i) in model.blocks for i in cfg):
                active[config_label(cfg)] = model.active_parameter_count(cfg)
        return {
            "shared_parameters": shared_n,
            "per_block_parameters": block_n,
            "loaded_parameters": model.loaded_parameter_count(),
            "estimated_full_total_parameters": full_total,
            "active_parameters": active,
            "n_optional_blocks_total": spec.model_config.n_optional_blocks,
            "loaded_blocks": list(model.loaded_block_ids),
        }
    if isinstance(model, DenseLM):
        n = model.parameter_count()
        return {
            "dense_parameters": n,
            "active_parameters": {"dense": n},
            "n_layers": model.n_layers,
        }
    raise TypeError(type(model))


def _save_best_modular(model: ModularLM, tokenizer: CharTokenizer, out: Path, spec: RunSpec, step: int) -> None:
    save_modular_checkpoint(
        model,
        tokenizer,
        out / "checkpoints" / "best",
        metadata={"run": out.name, "seed": spec.seed, "mode": spec.mode, "step": step, "note": spec.note},
    )


def _save_final_modular(model: ModularLM, tokenizer: CharTokenizer, out: Path, spec: RunSpec, step: int) -> None:
    save_modular_checkpoint(
        model,
        tokenizer,
        out / "checkpoints" / "final",
        metadata={"run": out.name, "seed": spec.seed, "mode": spec.mode, "step": step, "note": spec.note},
    )


def train_modular(spec: RunSpec, data: TextData) -> dict:
    torch.manual_seed(spec.seed)
    set_seed(spec.seed)
    use_all_blocks = spec.mode == "joint"
    block_ids = tuple(range(spec.model_config.n_optional_blocks)) if use_all_blocks else spec.selected_blocks
    model = ModularLM(spec.model_config, block_ids=block_ids)
    if spec.init_modular:
        init_model, init_tok, _ = load_modular_checkpoint(
            spec.init_modular,
            selected_blocks=range(spec.model_config.n_optional_blocks),
            freeze=False,
        )
        if init_tok.vocab_size != data.tokenizer.vocab_size:
            raise ValueError("initial checkpoint tokenizer vocabulary does not match data")
        if init_model.config.d_model != spec.model_config.d_model:
            raise ValueError("initial checkpoint architecture does not match model config")
        model.shared.load_state_dict(init_model.shared.state_dict(), strict=True)
        for i in model.loaded_block_ids:
            model.blocks[str(i)].load_state_dict(
                init_model.blocks[str(i)].state_dict(), strict=True
            )
        print(f"[{spec.mode}] initialized from {spec.init_modular}", flush=True)
    if spec.init_shared_dense:
        init_dense, init_tok, _ = load_dense_checkpoint(spec.init_shared_dense)
        if init_tok.vocab_size != data.tokenizer.vocab_size:
            raise ValueError("initial dense tokenizer vocabulary does not match data")
        if init_dense.config.d_model != spec.model_config.d_model:
            raise ValueError("initial dense architecture does not match model config")
        if init_dense.n_layers != spec.model_config.n_shared_layers:
            raise ValueError(
                "initial dense checkpoint must have n_shared_layers layers; "
                f"got {init_dense.n_layers}"
            )
        state = {
            k.removeprefix("trunk."): v
            for k, v in init_dense.state_dict().items()
            if k.startswith("trunk.")
        }
        model.shared.load_state_dict(state, strict=True)
        print(f"[{spec.mode}] shared initialized from {spec.init_shared_dense}", flush=True)
    if spec.freeze_shared:
        if not (spec.init_modular or spec.init_shared_dense):
            raise ValueError("freeze_shared requires init_modular or init_shared_dense")
        for p in model.shared.parameters():
            p.requires_grad_(False)
        print(f"[{spec.mode}] shared trunk frozen", flush=True)
    model.train()

    allowed = _build_allowed(spec)
    train_allowed = allowed
    if spec.freeze_shared:
        train_allowed = [c for c in allowed if c]
        if not train_allowed:
            raise ValueError("freeze_shared leaves no trainable configuration")
    sampler = (
        ConfigSampler(
            train_allowed,
            seed=spec.seed + 17,
            full_bias=spec.full_bias,
            empty_bias=spec.empty_bias,
            n_optional=spec.model_config.n_optional_blocks,
        )
        if spec.mode == "joint"
        else None
    )
    eval_configs = _build_eval_configs(spec)
    full_config = tuple(range(spec.model_config.n_optional_blocks))
    if spec.full_warmup_steps > 0 and spec.mode == "joint":
        if full_config not in train_allowed:
            raise ValueError("full_warmup_steps requested but full config is not allowed")
    aux_weight = 0.0 if spec.freeze_shared else spec.shared_aux_weight

    optimizer = _make_optimizer(model, spec.lr, spec.weight_decay, spec.block_lr_mult)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, spec.warmup, spec.steps)
    )
    data_gen = torch.Generator().manual_seed(spec.seed + 1)

    history: list[dict] = []
    best_val = float("inf")
    best_step: int | None = None
    train_loss_ema: float | None = None
    t0 = time.perf_counter()
    print(f"[{spec.mode}] model config: {spec.model_config}", flush=True)
    print(f"[{spec.mode}] allowed configs: {[config_label(c) for c in allowed]}", flush=True)
    print(f"[{spec.mode}] parameters: {_param_summary(model, spec)}", flush=True)

    for step in range(1, spec.steps + 1):
        model.train()
        if spec.mode == "joint":
            if step <= spec.full_warmup_steps:
                cfg = full_config
                assert sampler is not None
                sampler.record(cfg)
            else:
                assert sampler is not None
                cfg = sampler.next()
        else:
            cfg = spec.selected_blocks
        x, y = data.get_batch("train", spec.batch_size, spec.block_size, data_gen)
        if aux_weight > 0 and cfg:
            logits_shared, logits = model.forward_multi(x, active_blocks=cfg)
            active_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            shared_ce = F.cross_entropy(
                logits_shared.reshape(-1, logits_shared.shape[-1]), y.reshape(-1)
            )
            loss = active_ce + aux_weight * shared_ce
            aux_active_ce = float(active_ce.detach())
            aux_shared_ce = float(shared_ce.detach())
        else:
            logits = model(x, active_blocks=cfg)
            active_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
            loss = active_ce
            aux_active_ce = float(active_ce.detach())
            aux_shared_ce = None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if spec.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        train_loss_ema = loss_value if train_loss_ema is None else 0.98 * train_loss_ema + 0.02 * loss_value

        if spec.log_every > 0 and (step % spec.log_every == 0 or step == 1):
            elapsed = time.perf_counter() - t0
            aux_text = (
                f" active_ce={aux_active_ce:.4f} shared_ce={aux_shared_ce:.4f}"
                if aux_shared_ce is not None
                else ""
            )
            print(
                f"[{spec.mode}] step {step}/{spec.steps} "
                f"cfg={config_label(cfg)} loss={loss_value:.4f} ema={train_loss_ema:.4f}{aux_text} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s",
                flush=True,
            )

        should_eval = spec.eval_every > 0 and (step % spec.eval_every == 0 or step == spec.steps)
        if should_eval:
            val = evaluate_modular_configs(
                model,
                data,
                data.tokenizer,
                allowed,
                split="val",
                batch_size=spec.batch_size,
                block_size=spec.block_size,
                max_batches=spec.eval_max_batches,
            )
            avg_val = sum(v["loss"] for v in val.values()) / len(val)
            if avg_val < best_val:
                best_val = avg_val
                best_step = step
                _save_best_modular(model, data.tokenizer, spec.out_dir, spec, step)
            record = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - t0,
                "train_loss": loss_value,
                "train_loss_ema": train_loss_ema,
                "active_ce": aux_active_ce,
                "shared_ce": aux_shared_ce,
                "avg_allowed_val_loss": avg_val,
                "val_losses": {k: v["loss"] for k, v in val.items()},
            }
            history.append(record)
            append_jsonl(spec.out_dir / "history.jsonl", record)
            print(
                f"[{spec.mode}] eval step {step}: avg_allowed_val={avg_val:.4f} "
                f"best={best_val:.4f}@{best_step}",
                flush=True,
            )

    train_seconds = time.perf_counter() - t0
    _save_final_modular(model, data.tokenizer, spec.out_dir, spec, spec.steps)
    best_dir = spec.out_dir / "checkpoints" / "best"
    if not (best_dir / "manifest.json").exists():
        best_dir = spec.out_dir / "checkpoints" / "final"
    best_model, tokenizer, best_manifest = load_modular_checkpoint(
        best_dir,
        selected_blocks=model.loaded_block_ids,
        verify_sha256=False,
    )
    print(f"[{spec.mode}] evaluating best checkpoint at {best_dir}", flush=True)
    t_eval = time.perf_counter()
    val_all = evaluate_modular_configs(
        best_model,
        data,
        tokenizer,
        eval_configs,
        split="val",
        batch_size=spec.batch_size,
        block_size=spec.block_size,
        max_batches=None,
    )
    test_all = evaluate_modular_configs(
        best_model,
        data,
        tokenizer,
        eval_configs,
        split="test",
        batch_size=spec.batch_size,
        block_size=spec.block_size,
        max_batches=None,
    )
    eval_seconds = time.perf_counter() - t_eval

    final_size = checkpoint_size(spec.out_dir / "checkpoints" / "final")
    best_size = checkpoint_size(best_dir)
    metrics = {
        "run": {
            "name": spec.out_dir.name,
            "mode": spec.mode,
            "note": spec.note,
            "seed": spec.seed,
            "steps": spec.steps,
            "batch_size": spec.batch_size,
            "block_size": spec.block_size,
            "lr": spec.lr,
            "weight_decay": spec.weight_decay,
            "warmup": spec.warmup,
            "config_set": spec.config_set,
            "excluded": [list(c) for c in spec.excluded],
            "full_bias": spec.full_bias,
            "empty_bias": spec.empty_bias,
            "full_warmup_steps": spec.full_warmup_steps,
            "block_lr_mult": spec.block_lr_mult,
            "shared_aux_weight": spec.shared_aux_weight,
            "init_modular": spec.init_modular,
            "init_shared_dense": spec.init_shared_dense,
            "freeze_shared": spec.freeze_shared,
            "selected_blocks": list(spec.selected_blocks),
            "allowed_configs": [list(c) for c in allowed],
            "eval_configs": [list(c) for c in eval_configs],
        },
        "environment": environment_summary(),
        "data": data.summary(),
        "model_config": spec.model_config.to_dict(),
        "parameter_counts": _param_summary(best_model, spec),
        "training": {
            "seconds": train_seconds,
            "steps_per_second": spec.steps / train_seconds if train_seconds else None,
            "tokens_processed_total": spec.steps * spec.batch_size * spec.block_size,
            "tokens_per_config_average": (
                spec.steps * spec.batch_size * spec.block_size / len(allowed)
                if spec.mode == "joint"
                else spec.steps * spec.batch_size * spec.block_size
            ),
            "final_train_loss": loss_value,
            "final_train_loss_ema": train_loss_ema,
            "final_active_ce": aux_active_ce,
            "final_shared_ce": aux_shared_ce,
            "best_val_avg_allowed": best_val if best_step is not None else None,
            "best_step": best_step,
            "config_counts": sampler.counts if sampler is not None else {config_label(spec.selected_blocks): spec.steps},
            "peak_memory": collect_metrics(),
            "gpu": gpu_metrics(),
            "history": history,
        },
        "evaluation": {
            "seconds": eval_seconds,
            "val": val_all,
            "test": test_all,
        },
        "checkpoints": {
            "best_dir": str(best_dir),
            "best_bytes": best_size,
            "final_dir": str(spec.out_dir / "checkpoints" / "final"),
            "final_bytes": final_size,
            "manifest": best_manifest,
        },
    }
    write_json(spec.out_dir / "metrics.json", metrics)
    print(
        f"[{spec.mode}] done: train={train_seconds:.1f}s eval={eval_seconds:.1f}s "
        f"best={best_val:.4f}@{best_step} final_size={format_bytes(final_size)}",
        flush=True,
    )
    return metrics


def train_dense(spec: RunSpec, data: TextData) -> dict:
    torch.manual_seed(spec.seed)
    set_seed(spec.seed)
    n_layers = (
        spec.model_config.n_shared_layers
        if spec.mode == "dense_shared"
        else spec.model_config.n_shared_layers + spec.model_config.n_optional_blocks
    )
    model = DenseLM(spec.model_config, n_layers=n_layers)
    model.train()
    optimizer = _make_optimizer(model, spec.lr, spec.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, spec.warmup, spec.steps)
    )
    data_gen = torch.Generator().manual_seed(spec.seed + 1)

    history: list[dict] = []
    best_val = float("inf")
    best_step: int | None = None
    train_loss_ema: float | None = None
    t0 = time.perf_counter()
    print(f"[{spec.mode}] dense layers={n_layers} params={model.parameter_count()}", flush=True)

    for step in range(1, spec.steps + 1):
        model.train()
        x, y = data.get_batch("train", spec.batch_size, spec.block_size, data_gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if spec.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        train_loss_ema = loss_value if train_loss_ema is None else 0.98 * train_loss_ema + 0.02 * loss_value
        if spec.log_every > 0 and (step % spec.log_every == 0 or step == 1):
            print(
                f"[{spec.mode}] step {step}/{spec.steps} loss={loss_value:.4f} "
                f"ema={train_loss_ema:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

        if spec.eval_every > 0 and (step % spec.eval_every == 0 or step == spec.steps):
            val = evaluate_dense(
                model,
                data,
                data.tokenizer,
                split="val",
                batch_size=spec.batch_size,
                block_size=spec.block_size,
                max_batches=spec.eval_max_batches,
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                best_step = step
                save_dense_checkpoint(
                    model,
                    data.tokenizer,
                    spec.out_dir / "checkpoints" / "best.pt",
                    metadata={"run": spec.out_dir.name, "seed": spec.seed, "mode": spec.mode, "step": step, "note": spec.note},
                )
            record = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - t0,
                "train_loss": loss_value,
                "train_loss_ema": train_loss_ema,
                "val_loss": val["loss"],
            }
            history.append(record)
            append_jsonl(spec.out_dir / "history.jsonl", record)
            print(f"[{spec.mode}] eval step {step}: val={val['loss']:.4f} best={best_val:.4f}@{best_step}", flush=True)

    train_seconds = time.perf_counter() - t0
    save_dense_checkpoint(
        model,
        data.tokenizer,
        spec.out_dir / "checkpoints" / "final.pt",
        metadata={"run": spec.out_dir.name, "seed": spec.seed, "mode": spec.mode, "step": spec.steps, "note": spec.note},
    )
    best_path = spec.out_dir / "checkpoints" / "best.pt"
    if not best_path.exists():
        best_path = spec.out_dir / "checkpoints" / "final.pt"
    best_model, tokenizer, _ = load_dense_checkpoint(best_path)
    t_eval = time.perf_counter()
    val = evaluate_dense(best_model, data, tokenizer, split="val", batch_size=spec.batch_size, block_size=spec.block_size)
    test = evaluate_dense(best_model, data, tokenizer, split="test", batch_size=spec.batch_size, block_size=spec.block_size)
    eval_seconds = time.perf_counter() - t_eval
    final_size = (spec.out_dir / "checkpoints" / "final.pt").stat().st_size
    best_size = best_path.stat().st_size
    metrics = {
        "run": {
            "name": spec.out_dir.name,
            "mode": spec.mode,
            "note": spec.note,
            "seed": spec.seed,
            "steps": spec.steps,
            "batch_size": spec.batch_size,
            "block_size": spec.block_size,
            "lr": spec.lr,
            "weight_decay": spec.weight_decay,
            "warmup": spec.warmup,
        },
        "environment": environment_summary(),
        "data": data.summary(),
        "model_config": spec.model_config.to_dict(),
        "parameter_counts": _param_summary(best_model, spec),
        "training": {
            "seconds": train_seconds,
            "steps_per_second": spec.steps / train_seconds if train_seconds else None,
            "tokens_processed_total": spec.steps * spec.batch_size * spec.block_size,
            "final_train_loss": loss_value,
            "final_train_loss_ema": train_loss_ema,
            "best_val": best_val if best_step is not None else None,
            "best_step": best_step,
            "peak_memory": collect_metrics(),
            "gpu": gpu_metrics(),
            "history": history,
        },
        "evaluation": {"seconds": eval_seconds, "val": {"dense": val}, "test": {"dense": test}},
        "checkpoints": {
            "best_path": str(best_path),
            "best_bytes": best_size,
            "final_path": str(spec.out_dir / "checkpoints" / "final.pt"),
            "final_bytes": final_size,
        },
    }
    write_json(spec.out_dir / "metrics.json", metrics)
    print(
        f"[{spec.mode}] done: train={train_seconds:.1f}s eval={eval_seconds:.1f}s "
        f"best={best_val:.4f}@{best_step} size={format_bytes(final_size)}",
        flush=True,
    )
    return metrics


def build_spec(args: argparse.Namespace, data: TextData) -> RunSpec:
    model_config = ModelConfig(
        vocab_size=data.tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_shared_layers=args.n_shared_layers,
        n_optional_blocks=args.n_blocks,
        d_ff=args.d_ff,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
        composition=args.composition,
        layer_scale_init=args.layer_scale_init,
    )
    excluded = _parse_excluded(args.exclude, args.n_blocks)
    if args.mode == "separate":
        selected = _parse_blocks(args.blocks)
        for i in selected:
            if not 0 <= i < args.n_blocks:
                raise ValueError(f"block {i} outside [0, {args.n_blocks})")
    else:
        selected = tuple(range(args.n_blocks)) if args.mode == "joint" else ()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    return RunSpec(
        mode=args.mode,
        out_dir=out_dir,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches if args.eval_max_batches > 0 else None,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        threads=args.threads,
        config_set=args.config_set,
        excluded=excluded,
        selected_blocks=selected,
        model_config=model_config,
        note=args.note,
        full_bias=args.full_bias,
        empty_bias=args.empty_bias,
        full_warmup_steps=args.full_warmup_steps,
        block_lr_mult=args.block_lr_mult,
        shared_aux_weight=args.shared_aux_weight,
        init_modular=args.init_modular,
        init_shared_dense=args.init_shared_dense,
        freeze_shared=args.freeze_shared,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["joint", "separate", "dense", "dense_shared"], required=True)
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-max-batches", type=int, default=10, help="set <=0 for full validation during training")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--config-set", choices=["all", "singletons", "no_all", "no_pairs"], default="all")
    p.add_argument("--exclude", action="append", default=[], help="comma-separated block ids to exclude from allowed configs")
    p.add_argument("--full-bias", type=int, default=0, help="extra copies of the all-blocks config in the sampling pool")
    p.add_argument("--empty-bias", type=int, default=0, help="extra copies of the shared-only config in the sampling pool")
    p.add_argument("--full-warmup-steps", type=int, default=0, help="first N joint steps always use all blocks")
    p.add_argument("--block-lr-mult", type=float, default=1.0, help="learning-rate multiplier for optional blocks")
    p.add_argument("--shared-aux-weight", type=float, default=0.0, help="weight of the shared-only auxiliary CE loss during joint training")
    p.add_argument("--init-modular", default=None, help="initialize shared+blocks from an existing modular checkpoint")
    p.add_argument("--init-shared-dense", default=None, help="initialize the shared trunk from a dense shared-only checkpoint")
    p.add_argument("--freeze-shared", action="store_true", help="freeze the initialized shared trunk while training blocks")
    p.add_argument("--blocks", default="", help="for --mode separate, comma-separated block ids (empty means shared-only)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-shared-layers", type=int, default=4)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--composition", choices=["sequential", "parallel"], default="sequential")
    p.add_argument("--layer-scale-init", type=float, default=1.0)
    p.add_argument("--no-tie-weights", action="store_true")
    p.add_argument("--note", default="")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_threads(args.threads)
    if not Path(args.data).exists():
        raise SystemExit(
            f"data file not found: {args.data}\nRun: python scripts/download_data.py"
        )
    data = TextData.from_file(args.data)
    spec = build_spec(args, data)
    write_json(
        spec.out_dir / "run_config.json",
        {
            "args": vars(args),
            "environment": environment_summary(),
            "data": data.summary(),
        },
    )
    print(f"[{spec.mode}] data: {json.dumps(data.summary(), indent=2)}", flush=True)
    if spec.mode in {"joint", "separate"}:
        train_modular(spec, data)
    else:
        train_dense(spec, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
