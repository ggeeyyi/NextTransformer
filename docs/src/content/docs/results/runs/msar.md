---
title: MSAR
description: Multi-Stream Attention Residuals and its formal NextTransformer benchmark result.
---

[Multi-Stream Attention Residuals (MSAR)](https://arxiv.org/abs/2610.13630), by
Xiaohan Qin et al., is a residual architecture based on multi-stream depth
attention, temporal source augmentation and adaptive stream mixing. This page
reports its evaluation under the NextTransformer protocol.

## Benchmark result

At update 46,875, after 49.152B input tokens, MSAR reached **0.661044 periodic
validation BPB**, **0.658278 independent final validation BPB**, and **36.7452
CORE x100**, at seed 42. The model has 23 layers, hidden width 1,536, FFN width
6,144, and 1,358,888,937 parameters.

This experiment uses four temporal streams, three-layer depth blocks, softmax depth reads,
scalar-per-stream HPre, two-phase reads, direct embedding input to the first
attention, and the terminal MLP input-plus-output readout. It retains the shared
full-attention shell, ReLU-squared FFN, value embeddings, smear and training recipe.

The cumulative audit verifies 189 scheduled validation events and six complete
CORE-22 evaluations. It reports `matched`, with no findings or budget anomalies.
The charge is **4.716253378 GFLOPs per input token**, **+0.938598%** relative to
the compute anchor, totaling **2.3181328602510056e20 training FLOPs**.

The formal run used 32 ranks, microbatch eight and two accumulation steps,
resuming from updates 500 and 1,000. Training used a frozen snapshot of
`538653b` plus the MSAR implementation, with SDPA attention. The release uses
updated upstream code and a simplified configuration schema. Source hashes,
numerical comparisons and checkpoint loading checks are recorded in the
[source notes](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/evidence/runs/msar/README.md).

[Run record](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/results/runs/msar.json) · [Cumulative audit](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/evidence/audits/msar.json)
