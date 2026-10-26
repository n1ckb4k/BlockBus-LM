# BlockBus-LM

**Repository:** [github.com/n1ckb4k/BlockBus-LM](https://github.com/n1ckb4k/BlockBus-LM)

A CPU-scale research prototype for a modular character-level language model with one
mandatory shared Transformer and three optional parameter blocks that can be selected,
skipped, saved, and loaded independently.

This is **not** a general-purpose chatbot or a claim about reasoning, factual knowledge,
or instruction following. It is a controlled experiment on a 1.1 MB public-domain
Shakespeare corpus, trained from scratch on a laptop CPU. The repository contains the
implementation, tests, experiment scripts, raw benchmark, and result tables.

## Scope and headline result

The model has 548,864 mandatory shared parameters and three optional blocks of 131,968
parameters each. It can run the shared LM alone, any one block, any pair, or all three
blocks in a fixed sequential residual order. A block that is not selected is not called
and its checkpoint file is not read.

With one seed and this small corpus, the main all-active configuration reaches **1.9310
validation / 1.9576 test** cross-entropy (nats per character). A parameter-matched 7-layer
dense baseline trained for 1,600 steps reaches **1.9341 / 1.9575**. The losses are
descriptively tied; the tiny test difference is not evidence of a reliable advantage.

The main model is a two-stage, 3,200-step procedure, while the dense baseline is 1,600
steps. This comparison is parameter-matched, not compute-matched. A separately trained
all-active model at 1,600 steps reaches 1.9369 / 1.9626, close to the dense baseline; the
main model's all-active configuration receives 840 direct updates (200 in stage 1 and 640
in stage 2), plus updates through pair and singleton contexts.

| Model | Active params | Val loss | Test loss | Training note |
|---|---:|---:|---:|---|
| BlockBus main, all blocks | 944,768 | 1.9310 | 1.9576 | two-stage, 3,200 total steps |
| Dense 7-layer (parameter matched) | 944,768 | 1.9341 | 1.9575 | 1,600 steps |
| BlockBus main, shared only | 548,864 | 1.9991 | 2.0188 | stage 1 shared path |
| Dense shared-only (4 layers) | 548,864 | 1.9808 | 2.0030 | 1,600 steps |
| BlockBus clean-pipeline, all blocks | 944,768 | 1.9576 | 1.9793 | frozen dense shared + blocks, 3,200 total |
| Strict held-out triple (never trained) | 944,768 | 1.9763 | 1.9991 | blocks trained only as singletons |
| Separate all-active, 400 steps | 944,768 | 2.3316 | 2.3419 | fresh model, matched total 8 × 400 |
| Separate all-active, 1,600 steps | 944,768 | 1.9369 | 1.9626 | fresh model, upper-budget reference |
| Separate shared, 400 steps | 548,864 | 2.3729 | 2.3793 | fresh model, matched total 8 × 400 |
| Unigram (add-one) | 66 | 3.3326 | 3.3620 | non-parametric |
| Uniform random | 0 | 4.1897 | 4.1897 | non-parametric |

The more informative test is **strict held-out composition**. Starting from the
independently trained dense shared trunk, the three blocks are initialized from scratch and
trained only as singletons. Unseen pairs improve validation loss by about 0.010–0.012 nats
over shared, but the unseen triple improves by only 0.0045 and is worse than every pair and
every singleton. The blocks remain usable when composed without direct training, but
composition is not additive in this experiment.

### What worked

- All eight subsets of the main model produce finite, non-trivial next-character losses.
  More blocks improve the main model's validation loss monotonically.
- Unselected blocks are genuinely absent from the graph. Active parameter counts, logical
  checkpoint bytes, and inference work change with the subset.
- The full-budget all-active configuration is close to a dense 7-layer Transformer with the
  same parameter count. The full-budget separately trained all-active model is also close
  (1.9369 / 1.9626).
- The strict held-out pairs show a small positive transfer signal when blocks are composed
  without ever training that pair.

### What did not work, or needs caution

- The main model's shared-only path is worse than the independently trained dense
  shared-only baseline (1.9991 vs 1.9808 validation). The joint objective trades shared
  accuracy for block gains.
- The three singleton blocks are close substitutes, not clearly specialized experts.
- Strict held-out composition is weak and non-additive. The unseen triple is worse than
  every unseen pair and every singleton, so a potential three-block configuration is not a
  proven functional model.
- Full-config-biased early training improves the all-active configuration but degrades the
  shared-only path severely.
- Separately trained per-configuration models under a matched total budget are much weaker
  than the joint model in these runs; each needs its own shared trunk and gets only 400
  updates. This is a statement about budget allocation, not a universal property of
  separate models.
- With identical seeding, one-block separate models are identical up to block relabeling,
  as are two-block models. The repeated rows document the baseline, not a claim that block
  indices differ.
- The earlier singleton diagnostic that initialized from mixed-config blocks was
  contaminated for the held-out question and is retained only as a diagnostic.

All numbers are one seed (`1337`). Small differences are descriptive, not significance
tests. See [`results/tables.md`](results/tables.md) for the complete table and
[`docs/experiments.md`](docs/experiments.md) for the run matrix and caveats.

## Architecture

```text
characters
  -> shared token embedding + learned position embedding
  -> 4 mandatory pre-LN Transformer layers
  -> residual bus
  -> selected optional blocks in increasing index order
  -> shared final LayerNorm
  -> tied shared LM head
  -> next-character logits
```

For a selected subset `S`:

```text
h = SharedEmbedding(tokens)
h = SharedLayers(h)
for i in sorted(S):
    h = h + Block_i.delta(h)
logits = TiedHead(SharedFinalNorm(h))
```

`Block_i.delta` is a complete pre-LN Transformer-layer update: causal self-attention +
MLP, with its own LayerNorms, written back to the residual stream. It is not an arbitrary
slice of a dense checkpoint. Blocks execute in a fixed canonical order; arbitrary
permutations are not claimed to be valid.

| Component | Count |
|---|---:|
| Shared parameters | 548,864 |
| Parameters per optional block | 131,968 |
| All-active parameters | 944,768 |
| Context length | 96 characters |
| Width / heads | 128 / 4 |
| Shared layers | 4 pre-LN Transformer layers |
| Optional blocks | 3 one-layer Transformer blocks |
| Feed-forward width | 256 |

Why sequential rather than parallel? Both were implemented. In a 500-step pilot, sequential
had all-active validation loss 2.3302 vs 2.3339 for parallel. The difference is tiny and
one-seed; sequential was chosen and the parallel path is retained for auditability.

## Training

Both stages use AdamW, learning rate `1e-3`, 100 warmup steps, cosine decay to 10%, weight
decay 0.1, gradient clipping 1.0, batch size 16, context length 96, dropout 0.1, and
4 CPU threads.

**Stage 1 (1,600 steps, `joint_aux`).** Sample one of the eight subsets in balanced
shuffled cycles. Run only that subset and optimize next-character cross-entropy. On
non-empty subsets, add a shared-only auxiliary cross-entropy loss with weight 1.0. Each
subset receives 200 updates. The shared trunk receives gradients on every step; each block
receives gradients only when selected.

**Stage 2 (1,600 steps, `joint_frozen_long`, main model).** Load the stage-1 model, freeze
the shared trunk, and sample only non-empty subsets with the all-blocks configuration
oversampled (`full_bias=3`). The sampling pool is 3 singleton + 3 pair + 4 triple entries.
Over 1,600 steps the triple receives 640 updates and every other non-empty subset receives
160. Blocks use twice the shared learning rate. Checkpoint selection uses mean validation
loss over the eight configurations; test is not used for selection.

**Strict held-out composition (`joint_frozen_singletons_clean`).** Start from the dense
shared-only checkpoint, freeze it, initialize three blocks from scratch, and train only the
three singleton configurations for 1,600 steps. The frozen shared path is evaluated but
receives no updates. Pairs and the triple are never trained in this model or any earlier
stage.

The clean-pipeline check (`joint_pretrained_shared`) uses the same frozen dense shared
trunk and the same stage-2 sampling as the main model, but starts the blocks from scratch.
It is a more controlled comparison for whether optional blocks can be trained on top of a
fixed language component.

## Every subset of the main model

| Configuration | Active blocks | Active params | Val loss | Test loss |
|---|---:|---:|---:|---:|
| shared | 0 | 548,864 | 1.9991 | 2.0188 |
| B0 | 1 | 680,832 | 1.9687 | 1.9897 |
| B1 | 1 | 680,832 | 1.9668 | 1.9893 |
| B2 | 1 | 680,832 | 1.9654 | 1.9871 |
| B0+B1 | 2 | 812,800 | 1.9467 | 1.9711 |
| B0+B2 | 2 | 812,800 | 1.9439 | 1.9675 |
| B1+B2 | 2 | 812,800 | 1.9452 | 1.9702 |
| B0+B1+B2 | 3 | 944,768 | 1.9310 | 1.9576 |

The gains over shared are real within this run, but the three singleton blocks are nearly
interchangeable (range 0.0033 nats). The blocks add useful capacity in combination, but the
experiment does not establish distinct per-block specialization.

## Held-out composition

The strict held-out model freezes the dense shared trunk, initializes the blocks from
scratch, and trains only singleton configurations. The table below is never trained on
pairs or the triple.

| Configuration | Active blocks | Active params | Val loss | Test loss | Δ val vs shared |
|---|---:|---:|---:|---:|---:|
| shared | 0 | 548,864 | 1.9808 | 2.0030 | — |
| B0 | 1 | 680,832 | 1.9716 | 1.9937 | −0.0092 |
| B1 | 1 | 680,832 | 1.9704 | 1.9929 | −0.0104 |
| B2 | 1 | 680,832 | 1.9715 | 1.9941 | −0.0094 |
| B0+B1 | 2 | 812,800 | 1.9699 | 1.9922 | −0.0110 |
| B0+B2 | 2 | 812,800 | 1.9705 | 1.9928 | −0.0103 |
| B1+B2 | 2 | 812,800 | 1.9691 | 1.9922 | −0.0118 |
| B0+B1+B2 | 3 | 944,768 | 1.9763 | 1.9991 | −0.0045 |

The unseen pairs are slightly better than shared, but they improve by only about 0.010–0.012
nats, roughly the same as a single trained block. The unseen triple is worse than every pair
and every singleton and improves on shared by only 0.0045 nats. Adding the third block in a
composition that was never trained therefore hurts relative to the two-block compositions.
The experiment does not establish a mechanism; one plausible interpretation is interference
between redundant residual updates, but that is not isolated here.

The clean-pipeline run provides a same-trunk, same-budget contrast. It also starts from the
frozen dense shared checkpoint and trains blocks from scratch for 1,600 steps, but trains
all subsets rather than only singletons. Its all-active loss is **1.9576 val / 1.9793
test**, versus **1.9763 / 1.9991** for the strict held-out model. The 0.0187 val and 0.0198
test gaps isolate the cost of leaving pairs and the triple untrained: the optional blocks
can be used unseen, but they do not combine as well as blocks that received combination
training. The main model is not directly comparable here because its shared trunk was
trained jointly with the blocks.

## Baselines and efficiency

All losses are in the table at the top. The main pipeline takes **940.3 s** of training
(547.8 s stage 1 + 392.5 s stage 2) and peaks at about **505 MB** working set. The dense
7-layer baseline takes 668.0 s and peaks at 505 MB; dense shared-only takes 300.7 s and
440 MB; the clean pipeline takes 809.4 s total; the strict held-out model takes 609.3 s
total; the separate full-budget all-active model takes 533.0 s and 505 MB. The 400-step
separate runs are much cheaper individually but collectively train eight fresh trunks.

Selective loading is measured on the main checkpoint. Logical bytes are the shared file
plus only the selected block files; the manifest and Python/PyTorch overhead are excluded.

| Loaded model | Active params | Checkpoint bytes | First load | Warm load | Batch-16 tok/s |
|---|---:|---:|---:|---:|---:|
| Modular shared | 548,864 | 2.11 MiB | 27 ms | 30 ms | 56,940 |
| Modular B0 | 680,832 | 2.62 MiB | 44 ms | 45 ms | 43,510 |
| Modular B0+B1 | 812,800 | 3.12 MiB | 68 ms | 76 ms | 36,592 |
| Modular B0+B1+B2 | 944,768 | 3.63 MiB | 68 ms | 83 ms | 31,192 |
| Dense shared-only | 548,864 | 2.11 MiB | 33 ms | 32 ms | 55,530 |
| Dense 7-layer | 944,768 | 3.63 MiB | 73 ms | 76 ms | 32,244 |

The load measurements are CPU-only and cache-assisted: the OS had already seen the files,
so they are not cold SSD reads. Load timing varied noticeably across repeated benchmark
invocations; the deterministic logical byte counts are the cleaner evidence of selective
loading. The benchmark's Windows process read counter also includes runtime and metadata
reads, and did not fall after cache warmup in this run. Raw counters are in
[`results/benchmark.json`](results/benchmark.json).

Teacher-forced batch-1 throughput is 26,609 tok/s for modular shared, 18,223 for B0, 15,637
for B0+B1, and 12,733 for all blocks; dense 7 is 14,578. Autoregressive 32-token generation
is slower (322 tok/s shared, 202 tok/s all-active, 194 tok/s dense 7) because generation is
sequential.

## Reproduce

Requires Python 3.10+ (tested on 3.14.4), PyTorch CPU, and about 16 GB RAM. The reported
machine is a 4-core / 8-thread Intel Core i7-10510U with 16 GB RAM and no CUDA.

```powershell
git clone https://github.com/n1ckb4k/BlockBus-LM.git
cd BlockBus-LM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python scripts/download_data.py
python -m unittest discover -s tests -v
python scripts/run_all.py
```

`run_all.py` trains the main model, dense baselines, clean-pipeline check, strict held-out
model, training ablations, eight independent 300-step configs, eight independent 400-step
configs, one 1,600-step all-active reference, then runs the benchmark, sample generation,
and analysis. The training runs total about 1.8 hours on the reported CPU; end-to-end time
is roughly 2–2.5 hours including evaluation and benchmarking. Training times vary with
thermal state and background load.

The final artifacts are written to:

- `results/tables.md`, `results/config_matrix.csv`, `results/baselines.csv`,
  `results/separate_baselines.csv`, `results/training_variants.csv`
- `results/summary.json`, `results/benchmark.json`
- `results/fig_loss_vs_blocks.png`, `results/fig_gain_vs_shared.png`,
  `results/fig_training_curves.png`, `results/fig_cpu_throughput.png`
- `results/samples.md`

`runs/` and checkpoint files are gitignored. The committed `results/` directory contains
the distilled metrics and benchmark; regenerating the checkpoints requires rerunning
training.

### Load only selected blocks

```python
from modular_llm.checkpoint import load_modular_checkpoint

# Reads manifest + shared.pt + only the selected block files.
model, tokenizer, manifest = load_modular_checkpoint(
    "runs/final/joint_frozen_long/checkpoints/best",
    selected_blocks=[0, 2],
)
logits = model(input_ids, active_blocks=[0, 2])
```

The loader constructs no module for an unselected block. A test patches `torch.load` and
checks that unselected block files are never opened.

## Repository layout

```text
modular_llm/       model, checkpointing, data, training, evaluation
scripts/           data download, training orchestration, benchmark, samples, analysis
tests/             unit and smoke tests
docs/architecture.md
docs/experiments.md
docs/literature.md
data/              Tiny Shakespeare and its provenance
results/           generated tables, figures, benchmark, summary
```

## Scientific status and limitations

- **One seed, one corpus, one scale.** This is not evidence of scaling or generality.
- **Character-level next-token prediction only.** No reasoning, instruction following,
  factual recall, or dialogue claims are made. Perplexity, bits/char, and next-character
  accuracy are the task metrics.
- **Small effect sizes.** The all-active/dense difference is a few thousandths of a nat;
  treat it as descriptive.
- **Parameter-matched is not compute-matched.** The main two-stage model uses 3,200 total
  steps; the dense baseline uses 1,600. The separate full-budget all-active run is the
  closer same-step reference.
- **Shared-path tradeoff.** The main joint model's shared-only loss is worse than the
  independently trained dense shared baseline.
- **Held-out composition is weak and non-additive.** The strict held-out pairs improve only
  slightly over shared, and the unseen triple is worse than every pair and every singleton.
- **Block specialization is not established.** Aggregate losses show the blocks behave as
  similar capacity additions in these runs.
- **Separate baselines are budget-limited.** The 400-step models each receive only 400
  updates; the repeated singleton/pair rows are initialization-symmetry artifacts.
- **No router or arbitrary permutations.** The caller selects a subset; execution order is
  fixed. This is not a token-level MoE.
- **Timing caveats.** CPU numbers include validation during training and are affected by
  OS file cache, thread count, and thermal state.
- **Not novel as a learning algorithm.** See [`docs/literature.md`](docs/literature.md);
  this is a controlled combination and engineering study of established ideas.

## Literature

The closest work is stochastic depth/LayerDrop, residual adapters, conditional
computation/MoE, MatFormer/elastic transformers, and modular deep learning. The repository
does not claim a new fundamental learning algorithm. It studies a specific residual-bus
design with separate checkpoints, balanced subset dropout, a frozen-shared specialization
stage, and a held-out composition test at CPU scale. The detailed comparison, with
citations and an explicit novelty statement, is in [`docs/literature.md`](docs/literature.md).

## Data and license

The corpus is the public-domain Tiny Shakespeare text from Andrej Karpathy's `char-rnn`
examples. Its SHA-256 and 90/5/5 contiguous split are documented in [`DATA.md`](DATA.md).
Code is MIT licensed; see [`LICENSE`](LICENSE).
