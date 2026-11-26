---
title: MoE track
description: A compute-controlled comparison platform for Mixture-of-Experts language models, at two Marin-derived sizes.
---

The MoE track applies the NextTransformer contract to sparse models. Every method uses the same data, in the same
order, at the same training FLOPs per token, with the same evaluation. Parameters and speed are outcomes.

The two reference tracks reproduce Marin's compute-optimal Grug MoE cells: model, MuonH optimizer,
schedule, quantile-balanced routing, context, batch and updates.

| Track | Shape | Context | Tokens | FLOPs / token | Total / active params |
|---|---|---|---|---|---|
| **d768** | 8 × 768, GQA 6/1, 256 experts × 384 top-4 + shared 768 | 4,096 | 4.42B | 9.369 × 10⁸ | 2.00B / 213M |
| **d1024** | 11 × 1024, GQA 8/2, 256 experts × 512 top-4 + shared 1024 | 4,096 | 8.43B | 1.695 × 10⁹ | 4.71B / 347M |

The [240 TPP comparison](../protocol/#240-tpp-comparison) runs these backbones for longer. The d768
track admits baseline, HC, mHC, SimpleHC and iHC at 14,073,462,784 input tokens. The d1024 track admits the
baseline at 34,064,564,224 input tokens. Both use the same global batch as their May reference track.

- **Data.** Dolma 3 HQ (300B tokens, shuffled and cleaned), tokenized with the dolma2 tokenizer. Shards are read in a
  fixed order.
- **Score.** **MoE-CORE** is the mean centered accuracy over 14 published tasks:
  - 10 short: ARC-E/C, HellaSwag, PIQA, WinoGrande, CSQA, OBQA, LAMBADA, SQuAD, TriviaQA, in DCLM formats.
  - 4 long: RULER S-NIAH-1 at 2,048 and 4,096 tokens, FDA, SWDE, in lm-evaluation-harness formats.
- **Throughput.** `moe.throughput` measures every method back to back with the baseline on the same node. The
  relative throughput is published with the result.
- **Compute.** Matmul FLOPs are measured block by block, and attention edges are charged by rule. A method is admitted
  within 1% of the track anchor and audited within 2% cumulatively.

Explore the [MoE leaderboard](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp) and [training curves and final task scores](https://interactive-active-ai.github.io/NextTransformer/#results?track=d768-240tpp) in the observatory.

The [protocol](../protocol/) gives every value. The
[MoE README](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/README.md#submitting-a-method)
gives the submission flow, which is the dense track's flow with `moe.*` entry points: plugin, calibrate, formal
training, throughput, audit and export, pull request.
