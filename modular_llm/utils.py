"""Small shared utilities."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit(path: str | Path = ".") -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def environment_summary() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
    }
