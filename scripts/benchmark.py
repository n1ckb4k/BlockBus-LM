"""Benchmark selective loading, disk footprint, and CPU inference.

Run after training.  The loading benchmark reports both logical checkpoint bytes
and Windows process read-transfer deltas.  The OS file cache can make repeated
loads nearly free; both first-load and warm-load numbers are reported rather than
pretending every load reads the SSD.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from modular_llm.checkpoint import (
    checkpoint_size,
    load_dense_checkpoint,
    load_modular_checkpoint,
)
from modular_llm.evaluate import (
    benchmark_dense_forward,
    benchmark_modular_forward,
    generate_dense,
    generate_modular,
)
from modular_llm.metrics import collect_metrics, format_bytes, set_threads
from modular_llm.utils import write_json

MAIN = ROOT / "runs" / "final" / "joint_frozen_long" / "checkpoints" / "best"
DENSE7 = ROOT / "runs" / "final" / "dense_7" / "checkpoints" / "best.pt"
DENSE_SHARED = ROOT / "runs" / "final" / "dense_shared" / "checkpoints" / "best.pt"
OUT = ROOT / "results" / "benchmark.json"


def _read_bytes() -> int | None:
    m = collect_metrics()
    return m.get("read_transfer_bytes")


def load_modular_repeated(selected: list[int], repeats: int = 5) -> dict:
    rows = []
    for rep in range(repeats):
        gc.collect()
        before = _read_bytes()
        t0 = time.perf_counter()
        model, tokenizer, manifest = load_modular_checkpoint(
            MAIN, selected_blocks=selected, verify_sha256=False
        )
        elapsed = time.perf_counter() - t0
        after = _read_bytes()
        rows.append(
            {
                "repeat": rep,
                "seconds": elapsed,
                "read_transfer_bytes_delta": (
                    after - before if before is not None and after is not None else None
                ),
                "loaded_parameter_count": model.loaded_parameter_count(),
                "active_parameter_count": model.active_parameter_count(selected),
                "loaded_blocks": list(model.loaded_block_ids),
            }
        )
        del model
    manifest = json.loads((MAIN / "manifest.json").read_text(encoding="utf-8"))
    available = {b["block_id"]: b for b in manifest["blocks"]}
    logical = manifest["shared"]["bytes"] + sum(
        available[i]["bytes"] for i in selected
    )
    times = [r["seconds"] for r in rows]
    reads = [r["read_transfer_bytes_delta"] for r in rows if r["read_transfer_bytes_delta"] is not None]
    return {
        "selected_blocks": selected,
        "logical_checkpoint_bytes_excluding_manifest": logical,
        "manifest_bytes": (MAIN / "manifest.json").stat().st_size,
        "first_load_ms": 1000.0 * rows[0]["seconds"],
        "warm_median_load_ms": 1000.0 * statistics.median(times[1:]) if len(times) > 1 else None,
        "all_median_load_ms": 1000.0 * statistics.median(times),
        "first_read_transfer_bytes": rows[0]["read_transfer_bytes_delta"],
        "warm_median_read_transfer_bytes": (
            statistics.median(reads[1:]) if len(reads) > 1 else None
        ),
        "runs": rows,
        "loaded_parameter_count": rows[0]["loaded_parameter_count"],
        "active_parameter_count": rows[0]["active_parameter_count"],
    }


def bench_generation(loader, kind: str) -> dict:
    if kind == "modular":
        model, tokenizer, _ = loader
        out = {}
        for label, cfg in [("shared", []), ("all", [0, 1, 2])]:
            times = []
            text = ""
            for rep in range(3):
                t0 = time.perf_counter()
                text = generate_modular(
                    model,
                    tokenizer,
                    "ROMEO: ",
                    active_blocks=cfg,
                    max_new_tokens=32,
                    temperature=0.8,
                    top_k=40,
                    seed=rep,
                )
                times.append(time.perf_counter() - t0)
            out[label] = {
                "generated_tokens": 32,
                "median_seconds": statistics.median(times),
                "tokens_per_second": 32 / statistics.median(times),
                "sample": text,
            }
        return out
    model, tokenizer, _ = loader
    times = []
    text = ""
    for rep in range(3):
        t0 = time.perf_counter()
        text = generate_dense(
            model,
            tokenizer,
            "ROMEO: ",
            max_new_tokens=32,
            temperature=0.8,
            top_k=40,
            seed=rep,
        )
        times.append(time.perf_counter() - t0)
    return {
        "dense": {
            "generated_tokens": 32,
            "median_seconds": statistics.median(times),
            "tokens_per_second": 32 / statistics.median(times),
            "sample": text,
        }
    }


def main() -> int:
    set_threads(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Sizes:")
    main_bytes = checkpoint_size(MAIN)
    dense7_bytes = DENSE7.stat().st_size
    dense_shared_bytes = DENSE_SHARED.stat().st_size
    print(f"  modular full: {format_bytes(main_bytes)}")
    print(f"  dense 7:      {format_bytes(dense7_bytes)}")
    print(f"  dense shared: {format_bytes(dense_shared_bytes)}")

    print("Selective loading...")
    load_results = {}
    for label, selected in [
        ("shared", []),
        ("b0", [0]),
        ("b2", [2]),
        ("b0_b1", [0, 1]),
        ("all", [0, 1, 2]),
    ]:
        res = load_modular_repeated(selected)
        load_results[label] = res
        print(
            f"  {label}: logical={format_bytes(res['logical_checkpoint_bytes_excluding_manifest'])} "
            f"first={res['first_load_ms']:.1f}ms warm={res['warm_median_load_ms']:.1f}ms "
            f"first_read={res['first_read_transfer_bytes']}"
        )

    print("Loading dense baselines...")
    dense_load = {}
    for label, path in [("dense_7", DENSE7), ("dense_shared", DENSE_SHARED)]:
        rows = []
        for _ in range(3):
            gc.collect()
            before = _read_bytes()
            t0 = time.perf_counter()
            model, tokenizer, _ = load_dense_checkpoint(path)
            elapsed = time.perf_counter() - t0
            after = _read_bytes()
            rows.append(
                {
                    "seconds": elapsed,
                    "read_transfer_bytes_delta": (
                        after - before if before is not None and after is not None else None
                    ),
                    "parameter_count": model.parameter_count(),
                }
            )
            del model
        reads = [
            r["read_transfer_bytes_delta"]
            for r in rows
            if r["read_transfer_bytes_delta"] is not None
        ]
        dense_load[label] = {
            "bytes": path.stat().st_size,
            "parameter_count": rows[0]["parameter_count"],
            "first_load_ms": 1000.0 * rows[0]["seconds"],
            "warm_median_load_ms": 1000.0 * statistics.median([r["seconds"] for r in rows[1:]]),
            "first_read_transfer_bytes": rows[0]["read_transfer_bytes_delta"],
            "warm_median_read_transfer_bytes": (
                statistics.median(reads[1:]) if len(reads) > 1 else None
            ),
            "runs": rows,
        }
        print(f"  {label}: {format_bytes(path.stat().st_size)} first={dense_load[label]['first_load_ms']:.1f}ms")

    print("Inference latency...")
    model, tokenizer, _ = load_modular_checkpoint(MAIN, selected_blocks=[0, 1, 2])
    modular_latency = {}
    for label, cfg in [
        ("shared", []),
        ("b0", [0]),
        ("b0_b1", [0, 1]),
        ("all", [0, 1, 2]),
    ]:
        one = benchmark_modular_forward(model, cfg, seq_len=96, batch_size=1, warmup=10, iterations=60)
        batch = benchmark_modular_forward(model, cfg, seq_len=96, batch_size=16, warmup=3, iterations=20)
        modular_latency[label] = {"batch_1": one, "batch_16": batch}
        print(
            f"  {label}: B1 {one['median_ms']:.2f}ms {one['tokens_per_second_median']:.0f} tok/s | "
            f"B16 {batch['tokens_per_second_median']:.0f} tok/s"
        )
    gen_mod = bench_generation((model, tokenizer, None), "modular")

    dense_models = {}
    for label, path in [("dense_7", DENSE7), ("dense_shared", DENSE_SHARED)]:
        dmodel, dtok, _ = load_dense_checkpoint(path)
        dense_models[label] = (dmodel, dtok)
    dense_latency = {}
    for label, (dmodel, _) in dense_models.items():
        one = benchmark_dense_forward(dmodel, seq_len=96, batch_size=1, warmup=10, iterations=60)
        batch = benchmark_dense_forward(dmodel, seq_len=96, batch_size=16, warmup=3, iterations=20)
        dense_latency[label] = {"batch_1": one, "batch_16": batch}
        print(
            f"  {label}: B1 {one['median_ms']:.2f}ms {one['tokens_per_second_median']:.0f} tok/s | "
            f"B16 {batch['tokens_per_second_median']:.0f} tok/s"
        )
    gen_dense7 = bench_generation((*dense_models["dense_7"], None), "dense")
    gen_dense_shared = bench_generation((*dense_models["dense_shared"], None), "dense")

    payload = {
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
            "seq_len": 96,
            "batch_sizes": [1, 16],
        },
        "disk": {
            "modular_full_dir_bytes": main_bytes,
            "modular_full_dir_human": format_bytes(main_bytes),
            "modular_shared_file_bytes": (MAIN / "shared.pt").stat().st_size,
            "modular_manifest_bytes": (MAIN / "manifest.json").stat().st_size,
            "modular_block_bytes": {
                b["block_id"]: b["bytes"] for b in json.loads((MAIN / "manifest.json").read_text(encoding="utf-8"))["blocks"]
            },
            "dense_7_bytes": dense7_bytes,
            "dense_shared_bytes": dense_shared_bytes,
        },
        "loading": {"modular": load_results, "dense": dense_load},
        "latency": {
            "modular": modular_latency,
            "dense": dense_latency,
            "generation_modular": gen_mod,
            "generation_dense_7": gen_dense7,
            "generation_dense_shared": gen_dense_shared,
        },
        "process": collect_metrics(),
    }
    write_json(OUT, payload)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
