"""Reproduce the final training pipeline and analyses.

This is intentionally a thin orchestrator: each training run is the same CLI
documented in the README. On the reported 8-thread CPU it takes roughly two
hours for training plus a few minutes for the benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "final"
LOGS = RUNS / "logs"


def run(cmd: list[str], name: str, force: bool) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    tail = (proc.stdout.strip().splitlines() or ["<no stdout>"])[-1]
    print(f"exit={proc.returncode} {tail}", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], flush=True)
        raise SystemExit(proc.returncode)


def train(name: str, args: list[str], force: bool) -> None:
    if (RUNS / name / "metrics.json").exists() and not force:
        print(f"SKIP {name} (metrics exists; use --force)")
        return
    cmd = [
        sys.executable,
        "-m",
        "modular_llm.train",
        "--out",
        str(RUNS / name),
        "--seed",
        "1337",
        "--batch-size",
        "16",
        "--block-size",
        "96",
        "--eval-every",
        "200",
        "--eval-max-batches",
        "10",
        "--log-every",
        "200",
        "--threads",
        "4",
        *args,
    ]
    run(cmd, name, force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-variants", action="store_true", help="skip the training-scheme ablations")
    parser.add_argument("--skip-separate", action="store_true")
    args = parser.parse_args()

    if not (ROOT / "data" / "tinyshakespeare.txt").exists():
        run([sys.executable, "scripts/download_data.py"], "download", args.force)

    # Stage 1: joint subset training with the shared-only auxiliary loss.
    train(
        "joint_aux",
        [
            "--mode", "joint",
            "--steps", "1600",
            "--composition", "sequential",
            "--block-lr-mult", "2.0",
            "--shared-aux-weight", "1.0",
            "--note", "stage 1: joint uniform subsets + shared aux + block LR x2",
        ],
        args.force,
    )

    # Stage 2: freeze the shared trunk and specialize the blocks.
    train(
        "joint_frozen_long",
        [
            "--mode", "joint",
            "--steps", "1600",
            "--composition", "sequential",
            "--full-bias", "3",
            "--block-lr-mult", "2.0",
            "--freeze-shared",
            "--init-modular", str(RUNS / "joint_aux" / "checkpoints" / "best"),
            "--note", "stage 2: frozen shared, specialize blocks, full-bias 3",
        ],
        args.force,
    )

    # Baselines.
    train(
        "dense_7",
        [
            "--mode", "dense",
            "--steps", "1600",
            "--note", "dense 7-layer baseline, same parameter count as all-active",
        ],
        args.force,
    )
    train(
        "dense_shared",
        [
            "--mode", "dense_shared",
            "--steps", "1600",
            "--note", "dense shared-only 4-layer baseline",
        ],
        args.force,
    )

    # Clean pipeline check: an independently trained shared trunk, then blocks
    # from scratch. This separates the shared-language component from the joint
    # subset-training procedure.
    train(
        "joint_pretrained_shared",
        [
            "--mode", "joint",
            "--steps", "1600",
            "--composition", "sequential",
            "--full-bias", "3",
            "--block-lr-mult", "2.0",
            "--freeze-shared",
            "--init-shared-dense", str(RUNS / "dense_shared" / "checkpoints" / "best.pt"),
            "--note", "clean two-stage: dense shared frozen, blocks from scratch",
        ],
        args.force,
    )

    # Held-out composition: clean dense shared trunk frozen, blocks initialized
    # from scratch, only singleton configurations trained. Pairs and the triple
    # are genuinely unseen in this model.
    train(
        "joint_frozen_singletons_clean",
        [
            "--mode", "joint",
            "--steps", "1600",
            "--composition", "sequential",
            "--config-set", "singletons",
            "--block-lr-mult", "2.0",
            "--freeze-shared",
            "--init-shared-dense", str(RUNS / "dense_shared" / "checkpoints" / "best.pt"),
            "--note", "held-out composition: dense shared frozen, blocks from scratch, singletons only",
        ],
        args.force,
    )

    if not args.skip_variants:
        train(
            "joint_all",
            [
                "--mode", "joint",
                "--steps", "1600",
                "--composition", "sequential",
                "--note", "ablation: uniform subsets, no shared aux",
            ],
            args.force,
        )
        train(
            "joint_biased",
            [
                "--mode", "joint",
                "--steps", "1600",
                "--composition", "sequential",
                "--full-bias", "7",
                "--block-lr-mult", "2.0",
                "--note", "ablation: full-config bias, block LR x2",
            ],
            args.force,
        )
        train(
            "joint_frozen",
            [
                "--mode", "joint",
                "--steps", "800",
                "--composition", "sequential",
                "--full-bias", "3",
                "--block-lr-mult", "2.0",
                "--freeze-shared",
                "--init-modular", str(RUNS / "joint_aux" / "checkpoints" / "best"),
                "--note", "ablation: shorter frozen-shared stage",
            ],
            args.force,
        )

    if not args.skip_separate:
        # Exploratory budget (8 x 300 = 2,400 optimizer steps). The full-budget
        # all-active reference is trained once by the 400-step invocation below.
        run(
            [
                sys.executable,
                "scripts/run_separate.py",
                "--steps", "300",
                "--skip-full",
                "--out", "runs/final/separate_300",
                "--logs", "runs/final/separate_300_logs",
                *(["--force"] if args.force else []),
            ],
            "separate_300",
            args.force,
        )
        # Matched total compute for the 3,200-step main pipeline, plus the
        # 1,600-step independently trained all-active upper-budget reference.
        run(
            [
                sys.executable,
                "scripts/run_separate.py",
                "--steps", "400",
                "--out", "runs/final/separate_400",
                "--logs", "runs/final/separate_400_logs",
                *(["--force"] if args.force else []),
            ],
            "separate_400",
            args.force,
        )

    run([sys.executable, "scripts/benchmark.py"], "benchmark", args.force)
    run([sys.executable, "scripts/generate_samples.py"], "samples", args.force)
    run([sys.executable, "scripts/analyze_results.py"], "analysis", args.force)
    print("\nDone. See results/tables.md and results/summary.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
