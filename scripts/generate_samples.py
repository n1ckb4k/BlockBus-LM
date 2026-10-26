"""Generate qualitative samples from the final modular and dense checkpoints.

These are next-character samples, not evidence of reasoning or factual ability.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modular_llm.checkpoint import load_dense_checkpoint, load_modular_checkpoint
from modular_llm.evaluate import generate_dense, generate_modular
from modular_llm.metrics import set_threads


PROMPT = "ROMEO: "
SEED = 7
MAX_NEW = 220


def main() -> int:
    set_threads(4)
    main_dir = ROOT / "runs" / "final" / "joint_frozen_long" / "checkpoints" / "best"
    heldout_dir = (
        ROOT / "runs" / "final" / "joint_frozen_singletons_clean" / "checkpoints" / "best"
    )
    model, tok, _ = load_modular_checkpoint(main_dir, selected_blocks=[0, 1, 2])
    held, _, _ = load_modular_checkpoint(heldout_dir, selected_blocks=[0, 1, 2])
    dense, dtok, _ = load_dense_checkpoint(ROOT / "runs" / "final" / "dense_7" / "checkpoints" / "best.pt")
    assert tok.vocab_size == dtok.vocab_size

    samples = []
    for label, cfg in [
        ("shared", []),
        ("B0", [0]),
        ("B0+B1", [0, 1]),
        ("B0+B1+B2 (trained)", [0, 1, 2]),
    ]:
        text = generate_modular(
            model, tok, PROMPT, active_blocks=cfg,
            max_new_tokens=MAX_NEW, temperature=0.8, top_k=40, seed=SEED,
        )
        samples.append((f"Modular joint: {label}", text))
    text = generate_modular(
        held, tok, PROMPT, active_blocks=[0, 1, 2],
        max_new_tokens=MAX_NEW, temperature=0.8, top_k=40, seed=SEED,
    )
    samples.append(("Modular held-out: B0+B1+B2 (never jointly trained)", text))
    text = generate_dense(
        dense, dtok, PROMPT,
        max_new_tokens=MAX_NEW, temperature=0.8, top_k=40, seed=SEED,
    )
    samples.append(("Dense 7-layer baseline", text))

    lines = [
        "# Qualitative samples",
        "",
        "Prompt: `ROMEO: `; temperature 0.8, top-k 40, seed 7, 220 characters. "
        "These are character-level next-token samples. They do not establish reasoning, "
        "factual knowledge, or instruction following.",
        "",
    ]
    for title, text in samples:
        lines += [f"## {title}", "", "```text", text, "```", ""]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote results/samples.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
