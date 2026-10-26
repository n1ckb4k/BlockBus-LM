# Architecture: BlockBus-LM

**Repository:** [github.com/n1ckb4k/BlockBus-LM](https://github.com/n1ckb4k/BlockBus-LM)

Status: implemented and evaluated. This document records the final design, not the
initial proposal. Results are in [`results/tables.md`](../results/tables.md) and the
machine-readable summary is [`results/summary.json`](../results/summary.json).

## Research question

Can one small language-model training procedure produce a mandatory shared language
component plus a bank of optional parameter blocks, such that many subsets of the optional
blocks are useful language models without training a separate model for every subset?

With one mandatory component and `n` optional blocks there are `2^n` subsets. These are
*possible* configurations, not `2^n` proven models. The experiment measures which subsets
actually work and what is lost when a composition was never trained.

## Fixed constraints

- CPU-only PyTorch, Windows 11, 16 GB RAM, no CUDA assumption.
- Tiny Shakespeare character-level corpus, so tokenization is dependency-free.
- Small enough to train the joint model, ablations, dense baselines, and independent
  per-configuration baselines on the target laptop CPU.

## Final architecture

```
characters
  -> shared token embedding + learned position embedding
  -> 4 mandatory pre-LN Transformer layers
  -> residual bus
  -> selected optional 1-layer Transformer blocks, in increasing block-index order
  -> shared final LayerNorm
  -> shared LM head (weights tied to the token embedding)
  -> next-character logits
```

The **shared component** is mandatory: embeddings, four pre-LN Transformer layers, the
final LayerNorm, and the language-model head. It is a complete causal LM when no optional
block is selected.

Each **optional block** is a self-contained pre-LN Transformer layer with its own two
LayerNorms, causal self-attention, MLP, and residual write. It is not a slice of a dense
checkpoint: its parameters have a defined residual-bus read/write interface and are
trained in many different bus contexts. The shared trunk is the only place where the
language model can exist independently of the blocks.

For selected subset `S` in canonical increasing order:

```text
h = SharedEmbedding(tokens)
h = SharedLayers(h)
for i in sorted(S):
    h = h + Block_i.delta(h)
h = SharedFinalNorm(h)
logits = TiedHead(h)
```

where

```text
Block_i.delta(h) =
    attn_i(LN_1_i(h))
  + mlp_i(LN_2_i(h + attn_i(LN_1_i(h))))
```

The block writes a residual update; it does not replace the shared stream. Sequential
composition was selected over parallel composition after a 500-step pilot: sequential had
a marginally lower all-active validation loss (2.3302 vs 2.3339 nats/char). The difference
is small and is not presented as a general finding.

Key properties:

- If `S = {}`, no optional block is called and the shared LM runs alone.
- If a block is not selected, its module is not called. There is no dense forward pass with
  a multiplicative mask.
- When loading from disk, unselected block files are not read and no corresponding module
  is constructed. `shared.pt` and `blocks/block_i.pt` are separate files with SHA-256
  entries in `manifest.json`.
- Blocks execute in fixed canonical order. Arbitrary permutations are not claimed to be
  valid: with both blocks 0 and 1 selected, block 1 always reads the output after block 0.
- The number of possible configurations is not the number of proven functional LLMs.

## Final configuration

| Item | Value |
|---|---|
| Corpus | Tiny Shakespeare, character level |
| Vocabulary | 66 tokens (65 characters + `<unk>`) |
| Context length | 96 characters |
| Width | 128 |
| Heads | 4 |
| Shared layers | 4 pre-LN Transformer layers |
| Optional blocks | 3 one-layer Transformer blocks |
| Feed-forward width | 256 |
| Dropout | 0.1 |
| Composition | sequential residual bus |
| Shared parameters | 548,864 |
| Parameters per optional block | 131,968 |
| All-active parameters | 944,768 |

The 7-layer dense baseline has exactly the same parameter count (944,768) and width. The
4-layer dense shared baseline has exactly the shared-only parameter count (548,864). The
dense and modular all-active models also have the same number of Transformer layers for
the all-active configuration; only the parameterization and subset-training procedure
differ.

## Training procedure

Both stages use AdamW, learning rate `1e-3` with 100 warmup steps and cosine decay to
10%, weight decay 0.1, gradient clipping 1.0, batch size 16, and context length 96. The
optional blocks use twice the shared learning rate in the joint and specialization
stages.

### Stage 1: joint subset training (1,600 steps)

At every step, sample one of the eight subsets in balanced shuffled cycles, execute only
that subset, and minimize next-character cross-entropy. The shared-only auxiliary loss is
added with weight 1.0 on non-empty subset steps so the mandatory path remains a usable LM:

```text
L = CE(logits_S, y) + 1.0 * CE(logits_shared, y)     for |S| > 0
L = CE(logits_shared, y)                              for S = {}
```

Every subset receives 200 optimizer updates. The shared component receives gradients on
every step; each optional block receives gradients only where selected. There is no
router, per-configuration loss table, or independently trained configuration model.

### Stage 2: frozen-shared specialization (1,600 steps)

The stage-1 shared trunk and blocks are loaded. The shared trunk is frozen. The blocks are
trained with the shared-only configuration removed and with the all-blocks configuration
oversampled: each singleton and pair appears once per sampling pool, while the triple
appears four times (`full_bias=3`). This gives each block more updates in the contexts
where it is expected to be useful without training a separate model per subset.

The stage-2 pool contains 3 singleton + 3 pair + 4 triple entries. Over 1,600 steps the
triple receives 640 updates and every other non-empty configuration receives 160. The
empty configuration is evaluated but is not trained in this stage because the shared
parameters are frozen.

This two-stage recipe was introduced after early end-to-end joint runs showed the shared
trunk absorbing most of the useful computation while the optional blocks remained
under-specialized. The main reported model uses this recipe. A cleaner alternative,
`joint_pretrained_shared`, starts from an independently trained dense 4-layer shared trunk
and trains the blocks from scratch with the same second-stage procedure; it is reported as
a robustness check, not hidden.

## Baselines

1. **Dense 7-layer Transformer** with the same width and total parameter count as the
   all-active modular model (944,768), trained for 1,600 steps.
2. **Dense shared-only 4-layer Transformer** with the same parameter count as the shared
   component (548,864), trained for 1,600 steps.
3. **Separate per-configuration models**: one fresh model for every subset, each trained
   for 400 steps. Total budget 8 × 400 = 3,200 optimizer steps, matching the total of the
   main two-stage pipeline. The 400-step independent models are therefore the
   matched-total-compute baseline; they do not share a trunk.
4. **Full-budget all-active separate model**: 1,600 steps on the all-blocks configuration,
   as an upper-budget reference rather than a compute match.
5. **Held-out-composition model**: the independently trained dense shared trunk is
   frozen, the three blocks are initialized from scratch, and only the singleton
   configurations are trained. Pairs and the triple are genuinely unseen, then evaluated
   after training. (An earlier diagnostic initialized blocks from the mixed-config
   stage-1 model and is not treated as held out.)

## Evaluation protocol

- Training, validation, and test splits are contiguous 90/5/5 character splits with a
  fixed seed. The character vocabulary is built from the training split only.
- Model selection uses validation loss only. The test split is evaluated once for the
  selected checkpoint and is not used for checkpoint selection.
- Loss is teacher-forced next-character cross-entropy in nats/char, with perplexity,
  bits/char, and next-character accuracy reported.
- Per-character-class losses are included in each run's metrics JSON.
- The main model evaluates all eight subsets. The held-out model evaluates all eight
  subsets even though only the three singleton configurations were trained; the frozen
  shared path, unseen pairs, and the unseen triple are evaluation-only.
- Parameter counts distinguish **shared**, **per-block**, **loaded**, and
  **active** parameters. A configuration's active count is the shared count plus only
  the selected block counts.
- Training wall-clock time, process peak working set, checkpoint bytes, first/warm load
  latency, process read-transfer deltas, inference latency, and tokens/s are measured on
  the target CPU. The OS file cache is explicitly allowed to affect warm-load numbers;
  first-load and repeated-load values are both reported.

## What is established vs. new

Established prior work includes stochastic depth/LayerDrop, residual adapters, conditional
computation, Mixture-of-Experts, MatFormer/elastic transformers, and modular deep learning
(see [`literature.md`](literature.md)). This prototype is a controlled combination of
those ideas: a residual block bus, balanced configuration dropout, separate block
checkpoints, a frozen-shared specialization stage, and a held-out composition test.

It is **not** a new fundamental learning algorithm. No novelty claim is made beyond the
specific engineering design and the empirical characterization reported here. The model is
a character-level next-token predictor; the experiments do not establish reasoning,
instruction following, factual knowledge, or behavior at larger scale.

## Success criteria and falsification

A useful positive result would be: the joint model's shared-only and several subset
configurations are clearly better than an untrained/random baseline; the all-active model
approaches a parameter-matched dense baseline; selective loading and active computation
change with the subset; and the held-out model improves on unseen pairs and the unseen
triple.

A negative result is reported if subsets collapse, if untrained compositions fail, or if
the modular model is substantially worse than the dense baseline. The README reports the
observed outcomes, including the small margins, the weakened shared trunk in some joint
variants, and the gap between seen and unseen compositions.

The observed held-out result is mixed. The unseen pairs improve validation loss by about
0.010–0.012 nats over the shared-only path, but the unseen triple improves by only 0.0045
and is worse than every pair and every singleton. The optional blocks remain functional
when composed without direct training, but this experiment does not show additive or
synergistic composition. With the same frozen shared trunk and the same block-training
budget, training all subsets reaches 1.9576 val / 1.9793 test on the triple, versus
1.9763 / 1.9991 for the singleton-only held-out model. This gap is the measured cost of
not training the evaluated combination.
