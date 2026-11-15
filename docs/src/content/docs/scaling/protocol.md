---
title: 2K scaling protocol
description: Common training recipe, T3 size rules, cost accounting and forecast evaluation.
---

The public design is `nextt-scale-2k-v1`. T3 assigns **1.0 × 10²⁰ theoretical training FLOPs per cell**. The four sizes use 16, 20, 28 and 32 executed layers. Width is 64 times depth; attention head dimension is 128. Four families × four sizes give 16 planned runs. Start with Dense T3-L20 as the shared data, evaluation and cost check.

| Setting | T3 specification |
|---|---|
| Data | Pinned ClimbMix and tokenizer, vocab 32,768, BOS best-fit packing; identical token-stream prefixes, dev 06541, validation 06542 |
| Sequence and batch | Context 2,048 for training and evaluation; 1,048,576 input tokens per update |
| Seed and shell | Seed 42 throughout, standard residual stream, no value embeddings or VE gates |
| Optimizer | Muon on block matrices, AdamW on embeddings, head and scalars, width-scaled learning rates; 40-step warmup, final 65% linear warmdown to 5% of peak |
| Evaluation | Validation every 1,000 updates, full CORE-22 every 20,000 and at the endpoint, independent final BPB over 20,971,520 tokens |

### Family definitions

Dense uses causal MHA and a ReLU² FFN of width 4d. MoE retains that first block and subsequently uses 256 routed experts (top 8) plus one shared expert, each of width d/4. Freeze routing and optimization details using dev pilots. Hybrid repeats KDA, KDA, KDA, DSA with a dense FFN in every layer. Its token-level index uses top-256 and 4 × 128 index heads with 64 rotary dimensions. Run 512 dense burn-in updates and account for the indexer, teacher, selected edges and backward paths. The published KDA + MLA hybrid has a different fourth layer.

Fixed Recursive uses p independent input blocks, c shared core blocks executed twice and p output blocks. Set p = floor(L/8 + 0.5) and c = L/2 − p. Count unique weights for capacity and both executions for compute, including full backpropagation.

### Budgets and reporting

Round the planned token count to whole updates. For Dense the leading training FLOPs per token are 72Ld² + 12SLd + 6dV (S context, V vocabulary). MoE charges active expert and router work. Confirm actual cumulative costs against the target (within 1%). Hybrid needs separate burn-in and sparse-phase proofs before its token counts can be fixed. The [public specification](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json) describes configurations for planning; the archived dense launcher reproduces the earlier 4K run. Matching 2K implementations and cost proofs are prerequisites for reporting formal results.

The observatory estimates GPU-hours from target FLOPs divided by effective per-device throughput and 3,600 seconds per hour, plus 15% operating overhead. Eight same-model GPUs for one day use 192 GPU-hours. Pilot runs and optional extra sizes have separate budgets.

Fit validation BPB as a function of model capacity and training tokens, with shared exponents and family-specific coefficients. Track CORE-22 at the chosen sizes. If a minimum sits on a declining boundary, consider a neighboring size. Forecasts are conditional on seed 42. Pre-register all four T5 predictions before observing T5 outcomes, score them, then include those observations in the final fit. Freeze all four C200 predictions before any final run begins.
