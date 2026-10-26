# Experiment log and interpretation notes

All reported training runs use seed 1337, batch size 16, context length 96, AdamW with
learning rate `1e-3`, 100 warmup steps, cosine decay to 10%, weight decay 0.1, gradient
clipping 1.0, and dropout 0.1. The optional blocks use twice the shared learning rate in
joint and specialization stages. No run uses CUDA.

The final numbers are in [`../results/tables.md`](../results/tables.md). Raw per-run
metrics are produced under `runs/final/<run>/metrics.json` and are intentionally
gitignored because checkpoints and logs are large; the distilled tables and summary are
committed.

## Design-selection pilot

Both residual-bus composition modes were implemented and tested for 500 steps:

| pilot | all-active val loss | all-active test loss |
|---|---:|---:|
| sequential | 2.3302 | 2.3371 |
| parallel | 2.3339 | 2.3400 |

Sequential composition was selected before the main runs. The difference is tiny and is a
one-seed design choice, not evidence that sequential composition is generally superior.
The main experiments use sequential composition only; the parallel path remains implemented
and unit-tested.

## Final run matrix

| run | training purpose | init | configurations trained | steps |
|---|---|---|---|---:|
| `joint_all` | early ablation: uniform subsets, no auxiliary loss | scratch | all 8 | 1,600 |
| `joint_biased` | early ablation: full-config bias 7, block LR x2 | scratch | all 8, weighted | 1,600 |
| `joint_aux` | stage 1 of main pipeline | scratch | all 8, shared aux | 1,600 |
| `joint_frozen` | shorter stage-2 ablation | stage-1 checkpoint | 7 non-empty, full bias 3 | 800 |
| `joint_frozen_long` | **main model**, stage 2 | `joint_aux` | 7 non-empty, full bias 3 | 1,600 |
| `dense_7` | parameter-matched dense baseline | scratch | always-on 7 layers | 1,600 |
| `dense_shared` | parameter-matched shared-only baseline | scratch | always-on 4 layers | 1,600 |
| `joint_pretrained_shared` | clean-pipeline check | `dense_shared`, frozen | 7 non-empty, full bias 3; blocks from scratch | 1,600 |
| `joint_frozen_singletons_clean` | strict held-out composition test | `dense_shared`, frozen | 3 singletons; blocks from scratch | 1,600 |
| `joint_frozen_singletons` | discarded diagnostic: singleton fine-tune of mixed-init blocks | `joint_aux` | 3 singletons (shared frozen) | 1,600 |
| `separate_300/*` | exploratory independent-model baseline | scratch | one configuration per model | 300 each |
| `separate_400/*` | matched-total-compute independent baseline | scratch | one configuration per model | 400 each |
| `separate_400/b0_b1_b2_full` | full-budget all-active reference | scratch | all three blocks | 1,600 |

## Why the main pipeline has two stages

Early end-to-end joint runs (`joint_all`, `joint_biased`) showed a recurring failure mode:
the shared trunk absorbed most of the useful computation, while subset losses improved much
less than a dense model of the same parameter count. `joint_aux` added a shared-only
auxiliary objective so the mandatory path remained a usable LM and doubled the block
learning rate. A second stage then froze the shared trunk and oversampled the all-blocks
configuration to force the blocks to specialize. The 800-step and 1,600-step stage-2 runs
show that longer block specialization continued to improve the all-active configuration
under this recipe.

This is a two-stage optimization procedure, not a claim that a single gradient update
trained all subsets. No separate model is trained per configuration in the main pipeline.
The clean-pipeline check deliberately separates the shared language-model training from
the block training to test whether the main result depends on the stage-1 joint model.

## Budget definitions

- Main two-stage pipeline: 1,600 + 1,600 = **3,200 optimizer steps**.
- Independent per-configuration baseline: 8 configurations × 400 = **3,200 optimizer
  steps total**, but each model has its own shared trunk and sees only 400 steps.
- Independent full-budget all-active reference: **1,600 optimizer steps** on one model.
- Dense baseline: **1,600 optimizer steps** on one model.
- The main all-active configuration itself receives 840 all-active updates (200 in stage 1
  plus 640 in stage 2), but each block also receives updates through pair and singleton
  contexts. Its total training wall time includes the multi-configuration stage-1 cost.

These are different ways of spending compute. The reported table labels each comparison so
that "matched total budget" is not confused with "matched per-configuration budget".

## Strict held-out composition result

`joint_frozen_singletons_clean` starts from the independently trained dense 4-layer shared
checkpoint, freezes it, initializes all three blocks from scratch, and trains only the
three singleton configurations for 1,600 steps. The frozen shared path is evaluated but
receives no updates. No pair or triple appears in this run or in the checkpoint it starts
from.

| configuration | blocks | val loss | test loss | Δ val vs shared |
|---|---:|---:|---:|---:|
| shared | 0 | 1.9808 | 2.0030 | — |
| B0 | 1 | 1.9716 | 1.9937 | −0.0092 |
| B1 | 1 | 1.9704 | 1.9929 | −0.0104 |
| B2 | 1 | 1.9715 | 1.9941 | −0.0094 |
| B0+B1 | 2 | 1.9699 | 1.9922 | −0.0110 |
| B0+B2 | 2 | 1.9705 | 1.9928 | −0.0103 |
| B1+B2 | 2 | 1.9691 | 1.9922 | −0.0118 |
| B0+B1+B2 | 3 | 1.9763 | 1.9991 | −0.0045 |

The unseen pairs remain slightly better than the shared-only model, but they improve by only
about 0.010–0.012 nats, roughly the same as a single trained block. The unseen triple is
**worse than every pair and every singleton** and improves on the shared model by only
0.0045 nats. Adding the third block in a composition that was never trained therefore
hurts relative to the two-block compositions. Composition is not additive in this
experiment.

The clean-pipeline run `joint_pretrained_shared` provides a same-trunk, same-budget
contrast: it also starts from the frozen dense shared checkpoint and trains blocks from
scratch for 1,600 steps, but trains all subsets rather than only singletons. Its all-active
loss is 1.9576 val / 1.9793 test, versus 1.9763 / 1.9991 for the strict held-out model.
The 0.0187 val and 0.0198 test gaps therefore isolate the cost of leaving pairs and the
triple untrained: the optional blocks can be used unseen, but they do not combine as well
as blocks that received combination training.

## Separate-model symmetry

The independent 300- and 400-step baselines use the same seed and deterministic data order.
A model trained with only block 0, only block 1, or only block 2 is therefore identical up
to block relabeling: each run creates one block from the same initialization sequence and
trains it on the same batches. Two-block runs are likewise identical up to relabeling.
This is why the singleton rows within a budget have identical losses, and why the pair rows
do too. The all-active row is the meaningful matched-budget comparison; the repeated rows
are retained to document that each configuration really was trained as a separate model.
It also means these baselines should not be read as evidence about which block index is
better.

## Discarded diagnostic

The first singleton-only composition diagnostic (`joint_frozen_singletons`) initialized
its blocks from `joint_aux`, which had already trained every pair and the triple. Its
unseen-combination evaluation was therefore contaminated, and it is not used as a held-out
result. The corrected `joint_frozen_singletons_clean` run freezes the independent dense
shared trunk, initializes the blocks from scratch, and trains only the three singleton
configurations. That corrected run is the one used in the README and results tables.

## Interpretation cautions

- The test split is evaluated once for each trained model but is shared by all variants.
  The comparisons are descriptive, not a pre-registered significance test.
- Every training result is from one seed. Small differences (a few thousandths of a nat)
  should not be treated as reliable effects.
- The main shared-only path is weaker than the independently trained dense shared-only
  baseline. The joint procedure trades some shared-only loss for optional-block gains.
- The three optional blocks are close substitutes in aggregate: singleton and pair losses
  differ only slightly. No clear per-block specialization is established.
- The strict held-out composition test is deliberately harsh: only singleton contexts are
  trained, then the model is asked to run two and three blocks in sequence.
- Timings are from one Windows laptop CPU. They include validation during training and
  will vary with thread count, thermal state, and background load.
- Checkpoint load timings after the first read are OS-cache assisted. The benchmark
  reports first and warm loads plus process read-transfer deltas rather than claiming
  every warm load reads the SSD again.
