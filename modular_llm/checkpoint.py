"""Separate checkpointing for the shared trunk and optional blocks.

A modular checkpoint directory contains:

    shared.pt
    blocks/block_0.pt
    blocks/block_1.pt
    ...
    manifest.json

The loader materializes only the requested optional blocks and therefore reads only
their files.  Unselected blocks are absent from the computation graph and from the
loaded parameter set.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import ModelConfig
from .model import DenseLM, ModularLM, TransformerLayer
from .tokenizer import CharTokenizer
from .utils import now_utc, write_json


FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_torch(path: str | Path) -> Any:
    path = str(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Checkpoints are local artifacts produced by this repository and contain only
        # tensors plus plain Python metadata.  Older torch versions may not support
        # weights_only for every metadata type.
        return torch.load(path, map_location="cpu", weights_only=False)


def _block_param_count(config: ModelConfig) -> int:
    """Exact parameter count of one optional Transformer layer."""
    d = config.d_model
    # QKV + output projection (no biases), both MLP projections and biases,
    # two LayerNorms (weight + bias), plus an optional layer-scale scalar.
    per = 4 * d * d + 2 * d * config.d_ff + 5 * d + config.d_ff
    if config.composition == "parallel":
        per += 1
    return per


def save_modular_checkpoint(
    model: ModularLM,
    tokenizer: CharTokenizer,
    out_dir: str | Path,
    metadata: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    config = model.config
    shared_payload = {
        "format_version": FORMAT_VERSION,
        "kind": "blockbus_shared",
        "architecture": "blockbus",
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "n_optional_blocks": config.n_optional_blocks,
        "state_dict": model.shared.state_dict(),
        "metadata": metadata or {},
        "saved_utc": now_utc(),
    }
    shared_path = out_dir / "shared.pt"
    torch.save(shared_payload, shared_path)

    block_entries = []
    for block_id in model.loaded_block_ids:
        block = model.blocks[str(block_id)]
        payload = {
            "format_version": FORMAT_VERSION,
            "kind": "blockbus_block",
            "architecture": "blockbus",
            "config": config.to_dict(),
            "block_id": int(block_id),
            "state_dict": block.state_dict(),
            "metadata": metadata or {},
            "saved_utc": now_utc(),
        }
        rel = f"blocks/block_{block_id}.pt"
        path = out_dir / rel
        torch.save(payload, path)
        block_entries.append(
            {
                "block_id": int(block_id),
                "file": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "parameter_count": sum(p.numel() for p in block.parameters()),
            }
        )

    shared_n = sum(p.numel() for p in model.shared.parameters())
    block_n = _block_param_count(config)
    manifest = {
        "format_version": FORMAT_VERSION,
        "architecture": "blockbus",
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "shared": {
            "file": "shared.pt",
            "bytes": shared_path.stat().st_size,
            "sha256": _sha256(shared_path),
            "parameter_count": shared_n,
        },
        "blocks": block_entries,
        "saved_block_ids": [b["block_id"] for b in block_entries],
        "n_optional_blocks": config.n_optional_blocks,
        "loaded_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "estimated_full_total_parameter_count": int(
            shared_n + config.n_optional_blocks * block_n
        ),
        "metadata": metadata or {},
        "saved_utc": now_utc(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_modular_checkpoint(
    checkpoint_dir: str | Path,
    selected_blocks: Iterable[int] | None = None,
    device: str | torch.device = "cpu",
    verify_sha256: bool = False,
    freeze: bool = True,
) -> tuple[ModularLM, CharTokenizer, dict]:
    """Load the shared trunk plus only the requested block files.

    `selected_blocks=None` loads every block present in the manifest.  An empty
    iterable loads the shared language component alone.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(manifest["config"])
    available = {int(b["block_id"]): b for b in manifest["blocks"]}
    if selected_blocks is None:
        selected = sorted(available)
    else:
        selected = sorted(int(i) for i in selected_blocks)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_blocks contains duplicates")
    for i in selected:
        if i not in available:
            raise ValueError(
                f"block {i} is not present in this checkpoint; available={sorted(available)}"
            )

    shared_info = manifest["shared"]
    shared_path = checkpoint_dir / shared_info["file"]
    if verify_sha256:
        actual = _sha256(shared_path)
        if actual != shared_info["sha256"]:
            raise ValueError(f"sha256 mismatch for {shared_path}")
    shared_payload = _load_torch(shared_path)
    model = ModularLM(config, block_ids=selected)
    model.shared.load_state_dict(shared_payload["state_dict"], strict=True)

    for block_id in selected:
        info = available[block_id]
        path = checkpoint_dir / info["file"]
        if verify_sha256:
            actual = _sha256(path)
            if actual != info["sha256"]:
                raise ValueError(f"sha256 mismatch for {path}")
        payload = _load_torch(path)
        if int(payload["block_id"]) != block_id:
            raise ValueError(f"block id mismatch in {path}")
        model.blocks[str(block_id)].load_state_dict(payload["state_dict"], strict=True)

    model.to(device)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    tokenizer = CharTokenizer.from_dict(shared_payload["tokenizer"])
    return model, tokenizer, manifest


def save_dense_checkpoint(
    model: DenseLM,
    tokenizer: CharTokenizer,
    path: str | Path,
    metadata: dict | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "dense",
        "architecture": "dense",
        "config": model.config.to_dict(),
        "n_layers": model.n_layers,
        "tokenizer": tokenizer.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "saved_utc": now_utc(),
    }
    torch.save(payload, path)
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "n_layers": model.n_layers,
    }


def load_dense_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    verify_sha256: bool = False,
) -> tuple[DenseLM, CharTokenizer, dict]:
    path = Path(path)
    payload = _load_torch(path)
    config = ModelConfig.from_dict(payload["config"])
    model = DenseLM(config, n_layers=int(payload["n_layers"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


def checkpoint_size(checkpoint_dir: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    return sum(p.stat().st_size for p in checkpoint_dir.rglob("*") if p.is_file())


def file_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def block_parameter_count_from_config(config: ModelConfig) -> int:
    return _block_param_count(config)


def describe_loaded(model: ModularLM, selected: Iterable[int] | None = None) -> dict:
    if selected is None:
        selected = model.loaded_block_ids
    selected = tuple(sorted(int(i) for i in selected))
    shared = model.shared_parameter_count()
    block_counts = {i: model.block_parameter_count(i) for i in selected}
    return {
        "selected_blocks": list(selected),
        "shared_parameters": shared,
        "block_parameters": block_counts,
        "active_parameters": model.active_parameter_count(selected),
        "loaded_parameters": model.loaded_parameter_count(),
        "total_if_all_blocks_loaded": model.shared_parameter_count()
        + model.config.n_optional_blocks * block_parameter_count_from_config(model.config),
    }
