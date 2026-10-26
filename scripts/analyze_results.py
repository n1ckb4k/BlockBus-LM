"""Build result tables, figures, and a machine-readable summary from run metrics.

The script is intentionally read-only with respect to `runs/`: it consumes the
metrics and benchmark JSON produced by training and `scripts/benchmark.py`, then
writes human-readable tables plus a compact JSON summary under `results/`.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = ROOT / "runs" / "final"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MAIN = RUNS / "joint_frozen_long"
CLEAN = RUNS / "joint_pretrained_shared"
DENSE7 = RUNS / "dense_7"
DENSE_SHARED = RUNS / "dense_shared"
HELDOUT = RUNS / "joint_frozen_singletons_clean"
HELDOUT_MIXED = RUNS / "joint_frozen_singletons"

VARIANTS = [
    ("joint_all", "Uniform subsets, no shared auxiliary loss"),
    ("joint_biased", "Full-config bias 7 + block LR x2"),
    ("joint_aux", "Stage 1: uniform subsets + shared auxiliary loss"),
    ("joint_frozen", "Stage 2: frozen shared, 800 steps"),
    ("joint_frozen_long", "Stage 2: frozen shared, 1,600 steps (main)"),
    (
        "joint_pretrained_shared",
        "Clean pipeline: independent dense shared + blocks from scratch",
    ),
    (
        "joint_frozen_singletons_clean",
        "Held-out composition: clean shared + blocks from scratch, singletons only",
    ),
    (
        "joint_frozen_singletons",
        "Diagnostic: singleton fine-tune of mixed-config stage-1 blocks (not held out)",
    ),
]

CONFIG_ORDER = [
    "shared",
    "B0",
    "B1",
    "B2",
    "B0+B1",
    "B0+B2",
    "B1+B2",
    "B0+B1+B2",
]

# Runs whose reported pipeline cost includes an earlier training stage.
PIPELINE_STAGE = {
    "joint_frozen": "joint_aux",
    "joint_frozen_long": "joint_aux",
    "joint_frozen_singletons": "joint_aux",
    "joint_pretrained_shared": "dense_shared",
    "joint_frozen_singletons_clean": "dense_shared",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict:
    return load_json(path / "metrics.json")


def fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def human_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def config_label(blocks) -> str:
    if not blocks:
        return "shared"
    return "+".join(f"B{i}" for i in blocks)


def text_baselines() -> dict:
    """Uniform and add-one unigram next-character losses on the fixed splits."""
    from modular_llm.data import TextData

    data = TextData.from_file(ROOT / "data" / "tinyshakespeare.txt")
    vocab = data.tokenizer.vocab_size
    train_ids = data.train_ids.tolist()
    counts = [1] * vocab
    for token in train_ids:
        counts[token] += 1
    denominator = len(train_ids) + vocab
    log_probs = [math.log(c / denominator) for c in counts]

    def split_loss(split: str) -> float:
        targets = data.ids_for_split(split).tolist()[1:]
        return -sum(log_probs[t] for t in targets) / len(targets)

    return {
        "uniform_parameters": 0,
        "uniform_val_loss": math.log(vocab),
        "uniform_test_loss": math.log(vocab),
        "unigram_parameters": vocab,
        "unigram_val_loss": split_loss("val"),
        "unigram_test_loss": split_loss("test"),
    }


def config_rows(metrics: dict) -> list[dict]:
    val = metrics["evaluation"]["val"]
    test = metrics["evaluation"]["test"]
    shared_val = val["shared"]["loss"]
    rows = []
    for label in CONFIG_ORDER:
        if label not in val:
            continue
        v, t = val[label], test[label]
        rows.append(
            {
                "config": label,
                "n_blocks": 0 if label == "shared" else label.count("B"),
                "active_parameters": v["active_parameters"],
                "val_loss": v["loss"],
                "val_perplexity": v["perplexity"],
                "val_bits_per_char": v["bits_per_char"],
                "val_token_accuracy": v["token_accuracy"],
                "test_loss": t["loss"],
                "test_perplexity": t["perplexity"],
                "test_bits_per_char": t["bits_per_char"],
                "test_token_accuracy": t["token_accuracy"],
                "delta_val_vs_shared": v["loss"] - shared_val,
            }
        )
    return rows


def selected_label(metrics: dict) -> str:
    return config_label(metrics["run"].get("selected_blocks", []))


def variant_rows() -> list[dict]:
    rows = []
    for name, method in VARIANTS:
        path = RUNS / name
        if not path.exists() or not (path / "metrics.json").exists():
            continue
        m = load_metrics(path)
        val = m["evaluation"]["val"]
        test = m["evaluation"]["test"]
        allowed = m["run"].get("allowed_configs", [])
        allowed_labels = [config_label(c) for c in allowed]
        shared_val = val["shared"]["loss"]
        all_val = val.get("B0+B1+B2", {}).get("loss")
        all_test = test.get("B0+B1+B2", {}).get("loss")
        avg_allowed = (
            sum(val[label]["loss"] for label in allowed_labels) / len(allowed_labels)
            if allowed_labels
            else None
        )
        stage_seconds = 0.0
        stage_steps = 0
        stage_peak = 0
        stage_name = PIPELINE_STAGE.get(name)
        if stage_name and (RUNS / stage_name / "metrics.json").exists():
            stage_m = load_metrics(RUNS / stage_name)
            stage_seconds = stage_m["training"]["seconds"]
            stage_steps = stage_m["run"]["steps"]
            stage_peak = stage_m["training"]["peak_memory"].get(
                "peak_working_set_bytes"
            ) or 0
        total_seconds = m["training"]["seconds"] + stage_seconds
        total_steps = m["run"]["steps"] + stage_steps
        run_peak = m["training"]["peak_memory"].get("peak_working_set_bytes") or 0
        rows.append(
            {
                "run": name,
                "method": method,
                "stage1_run": stage_name if stage_seconds else None,
                "total_optimizer_steps": total_steps,
                "train_seconds": total_seconds,
                "shared_val_loss": shared_val,
                "all_active_val_loss": all_val,
                "all_active_test_loss": all_test,
                "avg_allowed_val_loss": avg_allowed,
                "avg_allowed_test_loss": (
                    sum(test[label]["loss"] for label in allowed_labels) / len(allowed_labels)
                    if allowed_labels
                    else None
                ),
                "active_parameters_all": val.get("B0+B1+B2", {}).get(
                    "active_parameters"
                ),
                "peak_working_set_bytes": max(run_peak, stage_peak) or None,
            }
        )
    return rows


def separate_rows() -> list[dict]:
    rows = []
    seen_full = False
    for dirname, expected_steps, label in [
        ("separate_300", 300, "300 (exploratory)"),
        ("separate_400", 400, "400 (matched)"),
    ]:
        root = RUNS / dirname
        if not root.exists():
            continue
        for path in sorted(
            p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists()
        ):
            m = load_metrics(path)
            if path.name == "b0_b1_b2_full":
                if not seen_full:
                    sel = selected_label(m)
                    val = m["evaluation"]["val"][sel]
                    test = m["evaluation"]["test"][sel]
                    rows.append(
                        {
                            "budget": "full",
                            "run": path.name,
                            "selected_config": sel,
                            "steps": m["run"]["steps"],
                            "active_parameters": val["active_parameters"],
                            "val_loss": val["loss"],
                            "test_loss": test["loss"],
                            "train_seconds": m["training"]["seconds"],
                            "peak_working_set_bytes": m["training"]["peak_memory"].get(
                                "peak_working_set_bytes"
                            ),
                        }
                    )
                    seen_full = True
                continue
            if m["run"]["steps"] != expected_steps:
                continue
            sel = selected_label(m)
            val = m["evaluation"]["val"]
            test = m["evaluation"]["test"]
            if sel not in val:
                continue
            rows.append(
                {
                    "budget": label,
                    "run": path.name,
                    "selected_config": sel,
                    "steps": m["run"]["steps"],
                    "active_parameters": val[sel]["active_parameters"],
                    "val_loss": val[sel]["loss"],
                    "test_loss": test[sel]["loss"],
                    "train_seconds": m["training"]["seconds"],
                    "peak_working_set_bytes": m["training"]["peak_memory"].get(
                        "peak_working_set_bytes"
                    ),
                }
            )
    order = {"300 (exploratory)": 0, "400 (matched)": 1, "full": 2}
    rows.sort(
        key=lambda r: (
            order.get(r["budget"], 9),
            CONFIG_ORDER.index(r["selected_config"])
            if r["selected_config"] in CONFIG_ORDER
            else 99,
        )
    )
    return rows


def find_separate(srows: list[dict], budget_prefix: str, config: str) -> dict | None:
    for row in srows:
        if row["budget"].startswith(budget_prefix) and row["selected_config"] == config:
            return row
    return None


def loading_table(bench: dict) -> list[list[str]]:
    if not bench:
        return []
    loading = bench["loading"]["modular"]
    rows = []
    for label, config in [
        ("shared", "shared"),
        ("B0", "b0"),
        ("B0+B1", "b0_b1"),
        ("B0+B1+B2 (all loaded)", "all"),
    ]:
        load = loading[config]
        rows.append(
            [
                f"Modular {label}",
                f"{load['active_parameter_count']:,}",
                human_bytes(load["logical_checkpoint_bytes_excluding_manifest"]),
                f"{load['first_load_ms']:.0f}",
                f"{load['warm_median_load_ms']:.1f}",
                human_bytes(load["first_read_transfer_bytes"]),
                human_bytes(load["warm_median_read_transfer_bytes"]),
            ]
        )
    for label, display in (
        ("dense_shared", "Dense shared-only (4 layers)"),
        ("dense_7", "Dense 7-layer"),
    ):
        load = bench["loading"]["dense"][label]
        rows.append(
            [
                display,
                f"{load['parameter_count']:,}",
                human_bytes(load["bytes"]),
                f"{load['first_load_ms']:.0f}",
                f"{load['warm_median_load_ms']:.1f}",
                human_bytes(load["first_read_transfer_bytes"]),
                human_bytes(load["warm_median_read_transfer_bytes"]),
            ]
        )
    return rows


def inference_table(bench: dict) -> list[list[str]]:
    if not bench:
        return []
    rows = []
    for label, config in [
        ("Modular shared", "shared"),
        ("Modular B0", "b0"),
        ("Modular B0+B1", "b0_b1"),
        ("Modular B0+B1+B2", "all"),
    ]:
        lat = bench["latency"]["modular"][config]
        rows.append(
            [
                label,
                f"{lat['batch_1']['active_parameters']:,}",
                f"{lat['batch_1']['tokens_per_second_median']:.0f}",
                f"{lat['batch_16']['tokens_per_second_median']:.0f}",
            ]
        )
    for label, display in (
        ("dense_shared", "Dense shared-only (4 layers)"),
        ("dense_7", "Dense 7-layer"),
    ):
        lat = bench["latency"]["dense"][label]
        rows.append(
            [
                display,
                f"{lat['batch_1']['active_parameters']:,}",
                f"{lat['batch_1']['tokens_per_second_median']:.0f}",
                f"{lat['batch_16']['tokens_per_second_median']:.0f}",
            ]
        )
    return rows


def main() -> int:
    main_m = load_metrics(MAIN)
    dense7_m = load_metrics(DENSE7)
    dense_shared_m = load_metrics(DENSE_SHARED)
    heldout_m = load_metrics(HELDOUT)
    heldout_mixed_m = (
        load_metrics(HELDOUT_MIXED)
        if (HELDOUT_MIXED / "metrics.json").exists()
        else None
    )
    clean_m = load_metrics(CLEAN) if (CLEAN / "metrics.json").exists() else None

    main_configs = config_rows(main_m)
    heldout_configs = config_rows(heldout_m)
    clean_configs = config_rows(clean_m) if clean_m else []

    # CSV: main config matrix
    with (OUT / "config_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(main_configs[0].keys()))
        w.writeheader()
        w.writerows(main_configs)

    main_all = main_m["evaluation"]["val"]["B0+B1+B2"]
    main_test_all = main_m["evaluation"]["test"]["B0+B1+B2"]
    dense7_val = dense7_m["evaluation"]["val"]["dense"]
    dense7_test = dense7_m["evaluation"]["test"]["dense"]
    dense_shared_val = dense_shared_m["evaluation"]["val"]["dense"]
    dense_shared_test = dense_shared_m["evaluation"]["test"]["dense"]
    heldout_all = heldout_m["evaluation"]["val"]["B0+B1+B2"]
    heldout_test_all = heldout_m["evaluation"]["test"]["B0+B1+B2"]

    srows = separate_rows()
    text_base = text_baselines()
    baseline_rows = [
        (
            "BlockBus joint all-active (main)",
            "shared + B0+B1+B2",
            main_all["active_parameters"],
            main_all["loss"],
            main_test_all["loss"],
        ),
        (
            "BlockBus held-out all-active",
            "shared + B0+B1+B2; pairs/triple unseen",
            heldout_all["active_parameters"],
            heldout_all["loss"],
            heldout_test_all["loss"],
        ),
    ]
    if clean_m:
        clean_all_val = clean_m["evaluation"]["val"]["B0+B1+B2"]
        clean_all_test = clean_m["evaluation"]["test"]["B0+B1+B2"]
        baseline_rows.append(
            (
                "BlockBus clean-pipeline all-active",
                "independent dense shared + B0+B1+B2",
                clean_all_val["active_parameters"],
                clean_all_val["loss"],
                clean_all_test["loss"],
            )
        )
    baseline_rows += [
        (
            "Dense 7-layer (parameter matched)",
            "7 always-on layers",
            dense7_val["active_parameters"],
            dense7_val["loss"],
            dense7_test["loss"],
        ),
        (
            "Dense shared-only",
            "4 always-on layers",
            dense_shared_val["active_parameters"],
            dense_shared_val["loss"],
            dense_shared_test["loss"],
        ),
        (
            "BlockBus shared-only (main)",
            "shared",
            main_m["evaluation"]["val"]["shared"]["active_parameters"],
            main_m["evaluation"]["val"]["shared"]["loss"],
            main_m["evaluation"]["test"]["shared"]["loss"],
        ),
        (
            "Uniform random",
            "non-parametric",
            text_base["uniform_parameters"],
            text_base["uniform_val_loss"],
            text_base["uniform_test_loss"],
        ),
        (
            "Unigram (add-one)",
            "non-parametric",
            text_base["unigram_parameters"],
            text_base["unigram_val_loss"],
            text_base["unigram_test_loss"],
        ),
    ]
    for cfg in ("shared", "B0+B1+B2"):
        row = find_separate(srows, "400", cfg)
        if row:
            baseline_rows.append(
                (
                    f"Separate {cfg} ({row['steps']} steps)",
                    "independently trained from scratch",
                    row["active_parameters"],
                    row["val_loss"],
                    row["test_loss"],
                )
            )
    full_row = next((r for r in srows if r["budget"] == "full"), None)
    if full_row:
        baseline_rows.append(
            (
                "Separate all-active (full budget)",
                f"independently trained, {full_row['steps']} steps",
                full_row["active_parameters"],
                full_row["val_loss"],
                full_row["test_loss"],
            )
        )
    with (OUT / "baselines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "composition", "active_parameters", "val_loss", "test_loss"])
        w.writerows(baseline_rows)

    vrows = variant_rows()
    with (OUT / "training_variants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(vrows[0].keys()))
        w.writeheader()
        w.writerows(vrows)

    if srows:
        with (OUT / "separate_baselines.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
            w.writeheader()
            w.writerows(srows)

    bench_path = OUT / "benchmark.json"
    bench = load_json(bench_path) if bench_path.exists() else {}
    load_rows = loading_table(bench)
    infer_rows = inference_table(bench)

    # Markdown tables
    main_avg = sum(r["val_loss"] for r in main_configs) / len(main_configs)
    heldout_avg = sum(r["val_loss"] for r in heldout_configs) / len(heldout_configs)
    lines = [
        "# Results tables",
        "",
        "Generated by `scripts/analyze_results.py` from the JSON metrics in `runs/` "
        "and `results/benchmark.json`. All losses are teacher-forced next-character "
        "cross-entropy in nats per character unless labeled otherwise. Test was not "
        "used for model selection. Lower is better.",
        "",
        "The main model is `joint_frozen_long`: a 1,600-step joint subset-dropout stage "
        "followed by a 1,600-step frozen-shared block-specialization stage. The "
        "clean-pipeline check is `joint_pretrained_shared`, which starts from an "
        "independently trained dense 4-layer shared trunk and trains the three blocks "
        "from scratch for 1,600 steps.",
        "",
        "## Main modular model: every subset",
        "",
        table(
            [
                "config",
                "active blocks",
                "active params",
                "val loss",
                "val bpc",
                "val acc",
                "test loss",
                "test bpc",
                "test acc",
                "Δ val vs shared",
            ],
            [
                [
                    r["config"],
                    r["n_blocks"],
                    f"{r['active_parameters']:,}",
                    fmt(r["val_loss"]),
                    fmt(r["val_bits_per_char"]),
                    fmt(r["val_token_accuracy"]),
                    fmt(r["test_loss"]),
                    fmt(r["test_bits_per_char"]),
                    fmt(r["test_token_accuracy"]),
                    fmt(r["delta_val_vs_shared"], 3),
                ]
                for r in main_configs
            ],
        ),
        "",
        f"Mean validation loss over all eight configurations: **{main_avg:.4f}**.",
        "",
        "## Baselines",
        "",
        table(
            ["model", "composition", "active params", "val loss", "test loss"],
            [
                [a, b, f"{c:,}", fmt(d), fmt(e)]
                for a, b, c, d, e in baseline_rows
            ],
        ),
        "",
        "The separate 400-step rows spend the same total optimizer-step budget as the "
        "main two-stage pipeline (8 × 400 = 3,200), but train a fresh shared trunk for "
        "every configuration. The 1,600-step separately trained all-active run is an "
        "upper-budget reference, not a compute match.",
        "",
        "## Training variants / ablations",
        "",
        table(
            [
                "run",
                "method",
                "total steps",
                "train s",
                "shared val",
                "all-active val",
                "all-active test",
                "avg allowed val",
                "peak WS MiB",
            ],
            [
                [
                    r["run"],
                    r["method"],
                    r["total_optimizer_steps"],
                    fmt(r["train_seconds"], 0),
                    fmt(r["shared_val_loss"]),
                    fmt(r["all_active_val_loss"]),
                    fmt(r["all_active_test_loss"]),
                    fmt(r["avg_allowed_val_loss"]),
                    fmt(
                        r["peak_working_set_bytes"] / (1024**2)
                        if r["peak_working_set_bytes"]
                        else None,
                        0,
                    ),
                ]
                for r in vrows
            ],
        ),
        "",
        "## Held-out composition",
        "",
        "The held-out model uses the independently trained dense shared trunk, frozen, "
        "and three randomly initialized blocks trained only on the three singleton "
        "configurations. The frozen shared path is evaluated but receives no updates. "
        "Pairs and the triple were never trained in this model or in an earlier stage. "
        "This is the strict composition test. An earlier diagnostic used blocks "
        "initialized from the mixed-configuration stage-1 model, so its pairs and triple "
        "were not truly unseen; it is retained only under training variants.",
        "",
        table(
            ["config", "active blocks", "val loss", "test loss", "Δ val vs shared"],
            [
                [
                    r["config"],
                    r["n_blocks"],
                    fmt(r["val_loss"]),
                    fmt(r["test_loss"]),
                    fmt(r["delta_val_vs_shared"], 3),
                ]
                for r in heldout_configs
            ],
        ),
        "",
        f"Mean validation loss over all eight configurations: **{heldout_avg:.4f}**.",
        "",
    ]

    if clean_configs:
        lines += [
            "## Clean-pipeline check",
            "",
            "This run does not reuse a jointly trained shared trunk. The mandatory "
            "component is the independently trained dense 4-layer baseline; only the "
            "optional blocks are trained with subset dropout and frozen shared weights.",
            "",
            table(
                ["config", "val loss", "test loss", "Δ val vs dense shared"],
                [
                    [
                        r["config"],
                        fmt(r["val_loss"]),
                        fmt(r["test_loss"]),
                        fmt(r["val_loss"] - clean_configs[0]["val_loss"], 3),
                    ]
                    for r in clean_configs
                ],
            ),
            "",
        ]

    if srows:
        lines += [
            "## Separately trained configurations",
            "",
            "Each row below is a different model trained from scratch on only that "
            "configuration with a fresh shared trunk. The 400-step budget is the "
            "matched-total-compute baseline for the main two-stage pipeline; the "
            "300-step rows are an earlier exploratory budget and are not used for the "
            "main conclusions.",
            "",
            table(
                [
                    "budget",
                    "run",
                    "selected config",
                    "steps",
                    "active params",
                    "val loss",
                    "test loss",
                    "train s",
                    "peak WS MiB",
                ],
                [
                    [
                        r["budget"],
                        r["run"],
                        r["selected_config"],
                        r["steps"],
                        f"{r['active_parameters']:,}",
                        fmt(r["val_loss"]),
                        fmt(r["test_loss"]),
                        fmt(r["train_seconds"], 0),
                        fmt(
                            r["peak_working_set_bytes"] / (1024**2)
                            if r["peak_working_set_bytes"]
                            else None,
                            0,
                        ),
                    ]
                    for r in srows
                ],
            ),
            "",
            "With the same seed, the separately trained one-block models are identical up "
            "to block relabeling, and the two-block models are likewise identical; the "
            "repeated rows document that each was trained as a separate model. The "
            "all-active 400-step row is the matched-budget comparison. These small "
            "models each receive only 400 updates, so they are much weaker than the "
            "3,200-step joint pipeline; this is a budget-allocation result, not evidence "
            "that separate training is inherently inferior.",
            "",
        ]

    if load_rows:
        lines += [
            "## Selective loading efficiency",
            "",
            "Checkpoint bytes include the shared file plus only the selected block files; "
            "the manifest and Python/PyTorch file overhead are excluded from the logical "
            "sum but included in the on-disk directory size in `benchmark.json`. Read "
            "transfer is the Windows process I/O counter, which includes some runtime "
            "and metadata reads in addition to the checkpoint files. Load latency is "
            "CPU-only and the OS cache had already seen the files, so these are not cold "
            "SSD reads. Read counters did not fall after the first repeat in the "
            "reported run, so logical checkpoint bytes are the cleaner measure of "
            "selective loading. See the benchmark JSON for raw counters and generation "
            "samples.",
            "",
            table(
                [
                    "loaded model",
                    "active params",
                    "checkpoint bytes",
                    "first load ms",
                    "warm load ms",
                    "first read",
                    "warm read",
                ],
                load_rows,
            ),
            "",
        ]
    if infer_rows:
        lines += [
            "## Inference throughput",
            "",
            "Batch sizes 1 and 16, sequence length 96, CPU inference mode. These are "
            "teacher-forced forward passes, not autoregressive generation.",
            "",
            table(
                [
                    "loaded model",
                    "active params",
                    "batch-1 tok/s",
                    "batch-16 tok/s",
                ],
                infer_rows,
            ),
            "",
        ]

    if bench:
        gen = bench["latency"]["generation_modular"]
        lines += [
            "## Generation throughput",
            "",
            table(
                ["modular configuration", "32-token generation tok/s"],
                [
                    [label, f"{gen[label]['tokens_per_second']:.1f}"]
                    for label in ("shared", "all")
                ],
            ),
            "",
        ]

    (OUT / "tables.md").write_text("\n".join(lines), encoding="utf-8")

    # Figure 1: loss by number of active blocks, retaining individual configurations.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for rows, marker, name, color in [
        (main_configs, "o", "Joint model (all subsets trained)", "tab:blue"),
        (heldout_configs, "s", "Held-out singleton-only model", "tab:orange"),
    ]:
        xs = np.array([r["n_blocks"] for r in rows], dtype=float)
        ys = np.array([r["val_loss"] for r in rows], dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(xs)) if len(xs) > 1 else np.zeros(1)
        ax.scatter(xs + jitter, ys, marker=marker, alpha=0.55, color=color, s=22)
        means = []
        for n in range(4):
            vals = [r["val_loss"] for r in rows if r["n_blocks"] == n]
            means.append(sum(vals) / len(vals) if vals else np.nan)
        ax.plot(range(4), means, marker=marker, color=color, label=f"{name} (mean)")
    ax.axhline(dense7_val["loss"], color="black", linestyle=":", label="Dense 7-layer")
    ax.axhline(
        dense_shared_val["loss"],
        color="gray",
        linestyle=":",
        label="Dense shared-only (4-layer)",
    )
    ax.set_xlabel("Number of active optional blocks")
    ax.set_ylabel("Validation cross-entropy (nats/char)")
    ax.set_xticks(range(4))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss_vs_blocks.png", dpi=160)
    plt.close(fig)

    # Figure 2: per-configuration gain over the shared-only model.
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    labels = [r["config"] for r in main_configs]
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(
        x - width / 2,
        [-r["delta_val_vs_shared"] for r in main_configs],
        width,
        label="Joint model",
    )
    heldout_by_label = {r["config"]: r for r in heldout_configs}
    ax.bar(
        x + width / 2,
        [
            -heldout_by_label[label]["delta_val_vs_shared"]
            if label in heldout_by_label
            else 0.0
            for label in labels
        ],
        width,
        label="Held-out singleton-only",
    )
    ax.set_ylabel("Validation loss improvement vs shared")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gain_vs_shared.png", dpi=160)
    plt.close(fig)

    # Figure 3: all-active training curves. Every series is measured with the same
    # 10-batch periodic validation protocol used during training. Final full-validation
    # losses are reported in the tables; they are not mixed into this figure.
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for name, label, color, linestyle in [
        ("joint_aux", "Stage 1 (joint, shared aux)", "tab:blue", "-"),
        ("joint_frozen_long", "Stage 2 (frozen shared)", "tab:orange", "-"),
        ("joint_pretrained_shared", "Clean pipeline (blocks from scratch)", "tab:green", "-"),
        ("dense_7", "Dense 7-layer baseline", "black", ":"),
    ]:
        path = RUNS / name / "metrics.json"
        if not path.exists():
            continue
        m = load_json(path)
        hist = m["training"]["history"]
        offset = 0 if name != "joint_frozen_long" else load_metrics(RUNS / "joint_aux")["run"]["steps"]
        if name == "dense_7":
            ys = [h["val_loss"] for h in hist]
        else:
            ys = [h["val_losses"]["B0+B1+B2"] for h in hist]
        ax.plot(
            [offset + h["step"] for h in hist],
            ys,
            marker="o" if linestyle == "-" else "s",
            linestyle=linestyle,
            markersize=3,
            label=label,
            color=color,
        )
    ax.set_xlabel("Optimizer step (stage 2 offset added where applicable)")
    ax.set_ylabel("All-active validation loss (nats/char)")
    ax.text(
        0.01,
        0.02,
        "10-batch periodic evaluation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.png", dpi=160)
    plt.close(fig)

    # Figure 4: CPU throughput from the benchmark.
    if bench:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        xs = list(range(4))
        ys = [
            bench["latency"]["modular"][label]["batch_16"]["tokens_per_second_median"]
            for label in ("shared", "b0", "b0_b1", "all")
        ]
        ax.plot(xs, ys, "o-", label="Modular model (batch 16)")
        ax.axhline(
            bench["latency"]["dense"]["dense_7"]["batch_16"][
                "tokens_per_second_median"
            ],
            color="black",
            linestyle=":",
            label="Dense 7-layer",
        )
        ax.axhline(
            bench["latency"]["dense"]["dense_shared"]["batch_16"][
                "tokens_per_second_median"
            ],
            color="gray",
            linestyle=":",
            label="Dense shared-only",
        )
        ax.set_xlabel("Number of active optional blocks")
        ax.set_ylabel("Tokens/s (batch 16, seq len 96)")
        ax.set_xticks(xs)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_cpu_throughput.png", dpi=160)
        plt.close(fig)

    summary = {
        "main_model": {
            "run": main_m["run"]["name"],
            "note": main_m["run"]["note"],
            "shared_val_loss": main_m["evaluation"]["val"]["shared"]["loss"],
            "shared_test_loss": main_m["evaluation"]["test"]["shared"]["loss"],
            "all_active_val_loss": main_all["loss"],
            "all_active_test_loss": main_test_all["loss"],
            "all_active_parameters": main_all["active_parameters"],
            "shared_parameters": main_m["evaluation"]["val"]["shared"][
                "active_parameters"
            ],
            "avg_config_val_loss": main_avg,
            "stage1_run": "joint_aux",
            "stage1_seconds": load_metrics(RUNS / "joint_aux")["training"]["seconds"],
            "stage2_seconds": main_m["training"]["seconds"],
            "total_pipeline_seconds": load_metrics(RUNS / "joint_aux")[
                "training"
            ]["seconds"]
            + main_m["training"]["seconds"],
            "peak_working_set_bytes": max(
                main_m["training"]["peak_memory"].get("peak_working_set_bytes") or 0,
                load_metrics(RUNS / "joint_aux")["training"]["peak_memory"].get(
                    "peak_working_set_bytes"
                )
                or 0,
            ),
        },
        "dense_baselines": {
            "text_baselines": text_base,
            "dense7_val_loss": dense7_val["loss"],
            "dense7_test_loss": dense7_test["loss"],
            "dense7_parameters": dense7_val["active_parameters"],
            "dense7_seconds": dense7_m["training"]["seconds"],
            "dense_shared_val_loss": dense_shared_val["loss"],
            "dense_shared_test_loss": dense_shared_test["loss"],
            "dense_shared_seconds": dense_shared_m["training"]["seconds"],
        },
        "heldout_model": {
            "run": heldout_m["run"]["name"],
            "init": "dense_shared (frozen); randomly initialized blocks",
            "training_configs": ["B0", "B1", "B2"],
            "shared_path": "frozen dense_shared; evaluated but not trained",
            "all_active_val_loss": heldout_all["loss"],
            "all_active_test_loss": heldout_test_all["loss"],
            "avg_config_val_loss": heldout_avg,
            "stage2_seconds": heldout_m["training"]["seconds"],
        },
        "heldout_mixed_init_diagnostic": (
            {
                "run": heldout_mixed_m["run"]["name"],
                "all_active_val_loss": heldout_mixed_m["evaluation"]["val"][
                    "B0+B1+B2"
                ]["loss"],
                "all_active_test_loss": heldout_mixed_m["evaluation"]["test"][
                    "B0+B1+B2"
                ]["loss"],
                "warning": "Pair/triple configurations appeared during stage-1 initialization; not held out.",
            }
            if heldout_mixed_m
            else None
        ),
        "clean_pipeline": (
            {
                "run": clean_m["run"]["name"],
                "shared_val_loss": clean_m["evaluation"]["val"]["shared"]["loss"],
                "all_active_val_loss": clean_m["evaluation"]["val"]["B0+B1+B2"][
                    "loss"
                ],
                "all_active_test_loss": clean_m["evaluation"]["test"]["B0+B1+B2"][
                    "loss"
                ],
                "block_training_seconds": clean_m["training"]["seconds"],
            }
            if clean_m
            else None
        ),
        "main_configs": main_configs,
        "heldout_configs": heldout_configs,
        "variants": vrows,
        "separate": srows,
        "baselines": [
            {"model": a, "composition": b, "active_parameters": c, "val_loss": d, "test_loss": e}
            for a, b, c, d, e in baseline_rows
        ],
        "benchmark": bench,
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT/'tables.md'}, CSVs, figures, and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
