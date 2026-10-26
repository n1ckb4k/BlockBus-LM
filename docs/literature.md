# Literature comparison and novelty statement

This project is a controlled CPU-scale prototype, not a claim of a new fundamental
learning algorithm. The ingredients below are established. The specific combination —
a mandatory shared Transformer plus separately checkpointed optional Transformer layers
on a residual bus, trained with balanced configuration dropout and a frozen-shared
specialization stage — is the engineering recipe evaluated here.

## Closest established lines of work

### Conditional computation and Mixture-of-Experts

- Jacobs, Jordan, Nowlan & Hinton (1991), *Adaptive Mixtures of Local Experts*,
  Neural Computation. Early learned mixture of local expert networks.
- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer*, ICLR. Sparse gating of experts; conditional compute.
- Lepikhin et al. (2021), *GShard: Scaling Giant Models with Conditional Computation
  and Automatic Sharding*, ICLR. Large-scale sharded MoE.
- Fedus, Zoph & Shazeer (2022), *Switch Transformers*, JMLR. Top-1 routing and
  trillion-parameter sparse models.

**Relation.** MoE also activates a subset of parameters conditionally. In standard MoE
a learned router chooses experts per token, experts are usually trained end-to-end, and
the full expert set is part of the serving system. BlockBus uses an externally supplied
configuration (no router), a single shared trunk that is always active, and optional
full Transformer layers that can be absent from disk and from the computation graph.
We do not reproduce MoE's load-balancing or token-routing behavior.

### Depth elasticity, layer dropping, and adaptive depth

- Huang et al. (2016), *Deep Networks with Stochastic Depth*, ECCV. Randomly drops
  residual blocks during training to regularize and shorten networks.
- Fan, Grave & Joulin (2020), *Reducing Transformer Depth on Demand with Structured
  Dropout* (LayerDrop), ICLR. Randomly drops entire Transformer layers during training,
  enabling shallower inference sub-networks.
- Raposo et al. (2024), *Mixture-of-Depths: Dynamically Allocating Compute in
  Transformer-Based Language Models*, arXiv:2404.02258. Routes tokens through a variable
  number of blocks.
- MatFormer (2023), *MatFormer: Nested Transformer for Elastic Inference*,
  arXiv:2310.07707. Nested elastic submodels within one Transformer.

**Relation.** LayerDrop/stochastic depth is the closest training idea: train with layers
randomly removed so sub-networks remain usable. BlockBus also uses subset dropout, but
treats each optional layer as a named, independently saved block with a defined residual
read/write interface, and evaluates every subset. Unlike LayerDrop, the optional blocks
are not merely slices of one dense checkpoint; unlike MatFormer, the blocks are separate
files and can be loaded alone. We do not claim that variable-depth training itself is new.

### Adapters, low-rank modules, and parameter-efficient composition

- Rebuffi, Bilen & Vedaldi (2017), *Learning Multiple Visual Domains with Residual
  Adapters*, NeurIPS. Small residual adapters for multi-domain specialization.
- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, ICML.
  Bottleneck adapters inserted into a frozen pretrained Transformer.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR.
  Low-rank residual weight updates.
- Pfeiffer et al. (2021), *AdapterFusion*, ACL. Composing adapters with a fusion module.
- Pfeiffer, Ruder, Vulić & Ponti (2024), *Modular Deep Learning*, TMLR
  (arXiv:2302.11529). Survey of modular architectures and composition.

**Relation.** Residual adapters are the conceptual ancestor of the optional-block write
path. BlockBus blocks are larger (full pre-LN attention + MLP layers), and the evaluation
targets language-model subsets rather than task adapters. The shared-frozen specialization
stage is adapter-like, but the blocks are full layers and are evaluated alone and in
combinations.

### Modular ensembles and separately trained components

- Rusu et al. (2016), *Progressive Neural Networks*, arXiv:1606.04671. Adds frozen
  columns for new tasks; preserves old capabilities.
- Li et al. (2022), *Branch-Train-Merge: Embarrassingly Parallel Training of Expert
  Language Models*, arXiv:2208.03306. Trains domain experts separately and combines
  them.
- Wortsman et al. (2022), *Model Soups*, ICML. Averages fine-tuned weights into one
  model; no conditional compute at inference.

**Relation.** These works show that separately trained components can be combined, but
they either grow the model per task, train full experts separately, or merge weights into
a single dense model. BlockBus instead trains one shared trunk and a bank of optional
blocks for many subsets in one procedure, with selective loading.

## What this project does differently

1. **Explicit residual-bus interface.** Optional blocks are self-contained pre-LN
   Transformer layers that read and write the shared residual stream at a fixed point.
   They are not arbitrary chunks of a dense checkpoint.
2. **Manual subset selection, no router.** The caller chooses the active blocks. The
   shared component is mandatory; the optional blocks are genuinely absent when not
   selected.
3. **Separate block files.** `shared.pt` and `blocks/block_i.pt` are independent files;
   the selective loader materializes and reads only requested blocks.
4. **Two-stage training recipe.** Stage 1 jointly samples all subsets with a shared-only
   auxiliary loss; stage 2 freezes the shared trunk and specializes the optional blocks
   with configuration dropout. This stage was added after the initial end-to-end joint
   model showed the shared trunk absorbing the useful computation, leaving the blocks
   under-trained.
5. **Held-out composition test.** A controlled model trains blocks only as singletons
   and evaluates unseen pairs and the unseen triple.

Items 1–5 are design choices and empirical results in this codebase. We have not
established that the recipe is novel in the literature, nor that it scales beyond a
~0.95M-parameter character-level model. A proper novelty claim would require a broader
systematic search and multi-seed, multi-scale experiments.

## What the experiments do and do not establish

**Established (within this prototype):**

- All eight subsets of the three optional blocks produce finite, better-than-uniform
  next-character predictions and can be loaded from disk.
- On one seed, the trained all-active configuration is essentially tied with a dense
  7-layer Transformer of the same parameter count on validation/test loss, while the
  shared component alone is weaker.
- A strict held-out-composition model was trained with blocks initialized from scratch
  and only singleton configurations. Its unseen pairs improve validation loss by about
  0.010–0.012 nats over shared, while the unseen triple improves by only 0.0045 and is
  worse than every pair. Blocks remain usable when composed without direct training, but
  the composition is not additive. The same-trunk clean-pipeline model, which does train
  all subsets, reaches 1.9576 val / 1.9793 test with the triple versus 1.9763 / 1.9991
  for the strict held-out model; see [`experiments.md`](experiments.md).
- Optional blocks are skipped in the computation graph: active parameter count, latency,
  and selective-loading behavior change with the selected subset.

**Not established:**

- That the model reasons, follows instructions, or has factual knowledge. It is a
  character-level next-token model trained on Shakespeare.
- That all eight subsets are distinct or equally useful LLMs. They are configurations of
  one small prototype; losses differ slightly and some subsets are near-duplicates.
- That the approach scales to large models, many blocks, long contexts, or general text.
- That the specific architecture is novel relative to LayerDrop, adapters, MoE, or
  modular deep learning.
- That the small validation/test differences are statistically significant; only one
  training seed is reported.
