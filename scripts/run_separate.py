"""Run the separately trained configuration baselines sequentially.

Primary baseline: every configuration gets `--steps` optimizer steps.  The final
joint pipeline uses 3,200 steps, so the default 8 * 400 = 3,200 is a
matched-total-optimizer-step budget (not matched wall-clock: separate models do
not share the shared trunk).

A full-budget all-active model is also trained as an upper bound.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "shared": "",
    "b0": "0",
    "b1": "1",
    "b2": "2",
    "b0_b1": "0,1",
    "b0_b2": "0,2",
    "b1_b2": "1,2",
    "b0_b1_b2": "0,1,2",
}


def run(name: str, blocks: str, steps: int, out_root: Path, log_root: Path, force: bool) -> None:
    out = out_root / name
    if (out / "metrics.json").exists() and not force:
        print(f"SKIP {name} (already exists)", flush=True)
        return
    log = log_root / f"{name}.log"
    log_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "modular_llm.train",
        "--mode",
        "separate",
        "--blocks",
        blocks,
        "--out",
        str(out),
        "--steps",
        str(steps),
        "--batch-size",
        "16",
        "--block-size",
        "96",
        "--eval-every",
        str(max(100, steps // 2)),
        "--eval-max-batches",
        "10",
        "--log-every",
        str(max(100, steps // 2)),
        "--threads",
        "4",
        "--seed",
        "1337",
        "--note",
        f"separately trained config [{blocks}] steps={steps}",
    ]
    print(f"RUN {name} steps={steps} blocks='{blocks}'", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    tail = (proc.stdout.strip().splitlines() or ["<no stdout>"])[-1]
    print(f"  exit={proc.returncode} {tail}", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], flush=True)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400, help="steps per separately trained config")
    parser.add_argument("--full-steps", type=int, default=1600, help="steps for the full-budget all-active upper bound")
    parser.add_argument("--out", default="runs/final/separate")
    parser.add_argument("--logs", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="skip the 1600-step all-active upper bound (e.g. when reusing one already trained)",
    )
    args = parser.parse_args()
    out_root = ROOT / args.out
    log_root = ROOT / (args.logs or (args.out + "_logs"))
    for name, blocks in CONFIGS.items():
        run(name, blocks, args.steps, out_root, log_root, args.force)
    if not args.skip_full:
        run("b0_b1_b2_full", "0,1,2", args.full_steps, out_root, log_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
