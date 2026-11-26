---
title: MoE protocol
description: The d768 and d1024 May and 240 TPP budgets, covering the Marin recipe, budgets, compute, throughput and MoE-CORE.
---

The MoE track asks the NextTransformer question for sparse models: **when the training budget is fixed, how should
a Mixture-of-Experts language model spend its computation?**

Every method trains on the same tokens, in the same order, at the same theoretical training FLOPs per token as the
track's reference baseline. Every method is scored with the same throughput harness and the same MoE-CORE suite.

This page is normative. The machine-readable values are in [`spec.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/spec.py). Every configuration, run and record
carries the protocol and track fingerprints.

## Lifecycle

```text
 implement          calibrate              admit                 train                   evaluate               audit + export
 a plugin under ─▶  FLOPs/token within ─▶  config + cost   ─▶   all updates, seed 42 ─▶ val BPB every 500,  ─▶  ledger within 2%,
 moe/methods/       1% of the anchor       record match         --formal                MoE-CORE x 4 + final   record + throughput
                                           the source                                                          → pull request
```

This is the dense protocol's lifecycle. Only the shapes, budgets, recipe, data and evaluation differ.

## Two tracks, reproducing Marin

Both tracks are Marin's "May Recipe" compute-optimal Grug MoE cells: architecture, optimizer, schedule, balancing,
context, batch and number of updates. Sources:
- [Marin README](https://github.com/marin-community/marin/blob/cd0e932c780e0171c5612658faa0f180a851569e/experiments/grug/moe/README.md)
- [issue #6074](https://github.com/marin-community/marin/issues/6074)
- the launch commit `d259a97068`, which started `moe_may_compute_opt_d{768,1024}_ep1`

| | **d768** | **d1024** |
|---|---|---|
| Marin cell | `moe_may_compute_opt_d768_ep1` | `moe_may_compute_opt_d1024_ep1` |
| Width / layers | 768 / 8 | 1024 / 11 |
| Attention | 6 query heads, 1 KV head, head dim 128 | 8 query heads, 2 KV heads, head dim 128 |
| Long (full-attention) layers | 3, 7 | 3, 7, 10 |
| Routed experts | 256 SwiGLU, width 384, top 4 | 256 SwiGLU, width 512, top 4 |
| Shared expert | 1 SwiGLU, width 768 | 1 SwiGLU, width 1024 |
| Parameters, total | 1,996,408,320 | 4,707,434,496 |
| Parameters, active per token | 212,780,544 | 347,455,488 |
| Active, excluding embedding and head | 58,639,872 (Marin: 58,641,920 incl. router bias) | 141,934,592 (Marin: 141,937,408) |
| Context | 4,096 tokens | 4,096 tokens |
| Rows × tokens per update | 64 × 4,096 = 262,144 | 128 × 4,096 = 524,288 |
| Updates (warmup) | 16,875 (168) | 16,080 (160) |
| **Training tokens** | **4,423,680,000** | **8,430,551,040** |
| **Training FLOPs per token** (the anchor) | **936,884,736** | **1,694,558,208** |
| Training FLOPs | 4.1445 × 10¹⁸ | 1.4286 × 10¹⁹ |
| Same budget in Marin's formula | 2.797 × 10¹⁸ (Marin: 2.81 × 10¹⁸) | 1.152 × 10¹⁹ (Marin: 1.16 × 10¹⁹) |
| Peak LR: MuonH / Adam | 0.008373 / 0.001932 | 0.008791 / 0.002029 |
| Adam β₂ / ε | 0.998001 / 1.26e-15 | 0.996006 / 1.23e-15 |
| QB groups per update | 16 × 4 rows | 16 × 8 rows |
| MoE-CORE events | 4,000 / 8,000 / 12,000 / 16,875 | 4,000 / 8,000 / 12,000 / 16,080 |
| Validation events | 35 (0, every 500, 16,875) | 34 (0, every 500, 16,080) |

Marin's budget formula covers the attention, router and expert matrices plus non-causal attention. It leaves out the
head, the GatedNorms and the gates. With that formula the two shapes land within 0.7% of Marin's published budgets.
This protocol counts every matmul and the exact attention edges, so its anchor is larger.

## 240 TPP comparison

NextTransformer also defines a longer budget for each May backbone. TPP here is the number of input tokens per
**active baseline parameter excluding both the token embedding and the output head**. Each method at a given
width uses the same reference parameter count and the same prefix of the training stream.

| Quantity | `d768-240tpp` | `d1024-240tpp` |
|---|---:|---:|
| Active baseline parameters for TPP | 58,639,872 | 141,934,592 |
| Rows × sequence length | 64 × 4,096 | 128 × 4,096 |
| Input tokens per update | 262,144 | 524,288 |
| Updates / warmup steps | 53,686 / 536 | 64,973 / 649 |
| **Training input tokens** | **14,073,462,784** | **34,064,564,224** |
| Peak LR, MuonH and AdamH / Adam | 0.005300844 / 0.001223272 | 0.005063817 / 0.001168573 |
| MoE-CORE steps | 12,726 / 25,451 / 38,177 / 53,686 | 16,162 / 32,325 / 48,487 / 64,973 |
| Eight-device layout, micro-batch × accumulation per device | 8 × 1 | 8 × 2 |
| Admitted methods | baseline, HC, mHC, SimpleHC, iHC | baseline |

Round `240 × active baseline parameters / tokens per update` to the nearest whole update, with halves rounded up.
The d1024 target is exactly 64,972.5 updates. CORE events retain their fractions of the original track and
validation runs at step zero, every 500 updates and the final step. The schedule derives peak rates, warmup
and Adam epsilon from the new budget. Both tracks use the shared data order, seed, quantile-balancing groups
and FLOPs admission windows of this protocol.

For the d1024 baseline on eight 80 GiB A800s, `--offload-activations` stores large BF16 and FP32 tensors
in pinned host memory during training. It preserves saved values, tensor layouts and the optimizer update.
Record this setting when comparing throughput.

NextTransformer defines these budgets on the May backbones. Marin's MuonH learning-rate sweep lists
240 TPP at widths 768 and 1024, with global batches of 64 and 128 rows respectively.
[Marin issue #5951](https://github.com/marin-community/marin/issues/5951)

The Marin reference measurements above belong to the original cells. In `moe.compute report`, divide `total_tokens` by `parameters.active_non_embedding`
to recover this TPP convention. The separate `tokens_per_active_parameter` field includes the embedding and head.

## Reference baseline: Marin's Grug MoE (`method = "baseline"`)

[`model.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/model.py) is a line-by-line port of Marin's run-time code.

**Model structure**
```text
h = embed_gated(embed_norm(E[tok]))
block:  x += Attn(attn_gated(attn_norm(x)));  x += MoE(m) + Shared(m),  m = mlp_gated(mlp_norm(x))
logits = final_gated(final_norm(x)) @ head        (untied embedding and head)
```

**Norms and gates**
- **RMSNorm:** fp32 statistics, eps 1e-5, learned gain.
- **GatedNorm:** `y = x · sigmoid(silu(x W_down) W_up)`, rank 128.

**Attention**
- GQA without biases.
- **PKO** on long layers: the rope-free half of each key comes from the previous token of the same document, and is zero at document starts.
- QK RMS norm without weights, eps 1e-6.
- **Half-RoPE** on the first 64 of the 128 dims, θ = 10,000.
- q × 1.3, softmax scale 1/√128.
- Mask: causal and same-document. Short layers add a 2,048-key window; long layers are every fourth layer plus the last.
- **XSA:** subtract the output's component along the token's own value.
- Head-wise gate `2·sigmoid(a G)`, with G initialized to zero.

**MoE**
- 256 routed SwiGLU experts (top 4) plus one always-on SwiGLU shared expert of width d.
- Dropless.

**Router (QB)**
- bf16 logits, top-4 on logits plus a balancing bias.
- Combine weights are sigmoid(logits), renormalized to sum to 2.5.
- The bias is set by Marin's quantile balancing:
  - Each group of rows (one Marin device's share of the batch; 16 groups per update) records, per expert, the q-th largest `logit − α`. Here α is the token's 5th-largest biased logit and q = tokens·4/256.
  - After the update, `bias = −β̄ + mean(β̄)`.
- No auxiliary loss, no z-loss.

**Initialization**
- Every matrix: truncated normal (±3σ) with σ = 0.5/√d. This includes the embedding, head, router, GatedNorms and experts.
- Head-wise gates: zero. Norm gains: one.

**Optimizer** ([`optim.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/optim.py), Marin's MuonH):

| Group | Parameters | Update |
|---|---|---|
| MuonH | All other matrices, including the GatedNorms and each expert separately | Nesterov momentum 0.95, five quintic Newton–Schulz steps per matrix, then the Frobenius-hyperball update `W ← (W − lr·U·‖W‖/‖U‖)·‖W‖/‖W'‖`, which keeps every matrix at its initial norm |
| AdamH | The head | The same hyperball update applied to the Adam step |
| Adam | Embedding, routers, head-wise gates, gains | Standard Adam (β₁ 0.9062, β₂ and ε per track) |

- **Learning rates:** `adam_lr = 0.06602 · T^-0.395 · d^-0.150 · √B` and `muonh_lr = 13/3 · adam_lr`, where T is the training tokens, d the width and B the tokens per update.
- **Schedule:** linear warmup from zero over `int(1%)` of the updates, then linear decay to zero.
- No weight decay, no gradient clipping.

**The only intended differences from Marin's runs:**
- **Tokenizer:** dolma2, vocabulary 100,278 padded to 100,352. Marin used Llama 3 (128,256).
- **Data:** Dolma 3 HQ instead of Nemotron-CC.
- **Document boundaries:** a document begins at its `<|endoftext|>` separator. Marin uses BOS/EOS.

**Implementation detail:** training on CUDA uses compiled FlexAttention and `torch._grouped_mm`. Evaluation uses
SDPA with an explicit mask, because evaluation prompts vary in length and would otherwise recompile the kernel per
length. The CPU path computes the same function; the CPU tests and a GPU equivalence check verify this.

## What is controlled

| Control | Value |
|---|---|
| Data | [`xjxyys/dolma3_hq_300B_shuffled_clean`](https://huggingface.co/datasets/xjxyys/dolma3_hq_300B_shuffled_clean) at revision `128a6856…`. Training reads shards 00000, 00001, … in order. Shard 00255 is the development split. The six `validation/*.parquet` files are reserved |
| Tokenizer | `allenai/dolma2-tokenizer`, revision `5292e5d6…`, `tokenizer.json` SHA-256 pinned. Reproduced id for id with tiktoken |
| Packing | Every document is prefixed with `<|endoftext|>`, and documents are concatenated in file order. Row k is tokens [4096k, 4096k + 4097). Attention is masked to the document |
| Data order | Update u reads rows [uR, (u+1)R). Ranks take contiguous slices, and micro-batches must hold whole QB groups. Any world size that divides R gives the same update |
| Budget | The track's tokens, including warmup |
| Randomness | Seed 42 |
| Compute | Training FLOPs per token within **1%** of the track anchor at admission, and within **2%** cumulatively |
| Recipe | Marin's MuonH / AdamH / Adam groups, learning rates, β, ε and schedule (above). Methods cannot change them |
| Precision | bf16 parameters and compute, with fp32 master weights and gradient reduction (FSDP2) |
| Evaluation | Validation BPB at step 0, every 500 updates and the last update. MoE-CORE at the four evaluation steps and at the end |

Parameters, memory and throughput are **outcomes**.

## Compute accounting

Training FLOPs per input token = **measured matmul FLOPs** + **attention edges charged by rule**.

- **Matmul FLOPs are measured.** Each block, the embedding/final GatedNorms and the head run one forward and backward
  pass at full size on CPU under `FlopCounterMode`, one block at a time. Dropless routing makes the matmul cost
  independent of which experts are chosen. For the baseline the measurement equals the closed form exactly.
- **Softmax attention is charged by rule.** It goes through `moe.model.attention`, which is charged
  3 × 4 × heads × head_dim × (causal or windowed key edges per query) at the 4,096-token context. The document mask is
  not credited.
- **Optimizer arithmetic is not charged.** This follows Marin's budget, and every method shares the same optimizer.

`python -m moe.compute calibrate --method <name> --track <track>` measures a method and solves its declared knob for
the anchor. The default knob is `expert_width`; `shared_expert_width`, `layers` or `none` can be declared instead.
It writes:
- the configuration `moe/configs/<track>/<name>.json`;
- the cost record `moe/evidence/costs/<track>/<name>.json`, which fingerprints `moe/{model,optim,plugins,compute,spec}.py` and the method's files.

## Throughput standard

`python -m moe.throughput` runs the trainer's own update on the real training rows:
- 10 warm-up updates, then 30 timed updates, dropping the fastest and slowest 10%.
- The method and the baseline run back to back on the same node.
- It reports seconds per update, tokens/s, tokens/s per device, MFU (against 989.4 TFLOP/s dense bf16 per device) and peak memory.

The published comparison is the **relative throughput**: the method's tokens/s divided by the baseline's, measured
on the same node in the same invocation.

Measured baseline (8 devices of 141 GiB, torch 2.9.1+cu128; full numbers in [`references.json`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/references.json)):

| | d768 | d1024 |
|---|---|---|
| Micro-batch | 8 (1 gradient accumulation step) | 8 (2 steps) |
| Tokens/s (per device) | 686,005 (85,751) | 536,017 (67,002) |
| MFU | 8.1% | 11.5% |
| Wall clock for the full budget | 1.8 h (14.3 device-hours) | 4.4 h (35.0 device-hours) |

A micro-batch of 16 does not fit d1024 on 141 GiB devices.

## Evaluation: MoE-CORE

One score: the **mean centered accuracy over 14 entries**, where `centered = (acc − chance)/(1 − chance)` as in
DCLM's CORE, and every entry weighs the same. Each task comes from published work, with its published format and
scoring.

| Entry | Source and format | Scoring |
|---|---|---|
| ARC-E, ARC-C, HellaSwag, PIQA, WinoGrande, CSQA, OBQA, LAMBADA, SQuAD | The dense track's pinned DCLM bundle, CORE formats and few-shot counts. These are the cloze/multiple-choice core shared by OLMoE, OLMES, OLMo 3's small-model suite and DataDecide | Multiple choice: lowest mean continuation loss. Short answers: every token is the greedy argmax |
| TriviaQA | DCLM's `triviaqa` file, 3-shot. One of the highest-SNR tasks at small scale (Signal and Noise) | Teacher-forced exact match |
| RULER S-NIAH-1 at **2,048** and at **4,096** tokens (two entries) | RULER (Hsieh et al. 2024) as lm-evaluation-harness generates `niah_single_1`: repeated-noise haystack, one adjective-noun key with a 7-digit value, 500 prompts per length, haystack sized with the dolma2 tokenizer | Greedy generation; the value must appear in the prediction (`string_match_all`). The cap is 32 new tokens; RULER allows 128 |
| FDA, SWDE | Based / JRT recall tasks as lm-evaluation-harness defines `fda` and `swde`. FDA documents run to about 2.2K tokens and SWDE to about 1.7K | Greedy generation until a newline (≤ 48 tokens); the prediction must contain the gold value |

- **Long-context probe:** RULER S-NIAH-1 at 8,192 tokens is reported as a length-extrapolation probe and is not scored.
- **Short-task prompts** are left-truncated to 4,096 tokens.
- **Deliberately excluded:**
  - MMLU and SIQA: at chance at this scale, and the bundle's SIQA is a letter format.
  - RULER multi-key/value/variable tasks: no signal below about 340M parameters (arXiv 2507.06457).
  - LongBench and HELMET: generation or a judge, and large models only.

## Formal and research modes

`python -m moe.train` runs in research mode by default. Research runs are never published.

`--formal` requires all of the following:
- an admitted configuration whose cost record matches the current source files and PyTorch version (2.9.1);
- a complete, verified data manifest;
- the full schedule and full evaluations;
- formal mode in every resumed segment.

The audit (`python -m moe.audit`) checks every event, the ledger and the throughput measurement before `--export`
writes the record.

## Versioning

Changing any value above creates a new protocol version, and records of different versions are never ranked
together. Larger budgets or sizes are added as new tracks; existing tracks never change.
