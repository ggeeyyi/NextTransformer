---
title: Scaling at fixed compute
description: Four architecture families, a fixed training budget, and preregistered performance forecasts.
---

How should each architecture divide a fixed training budget between model size and tokens? The 2K study starts at **T3**, with four execution depths for each of four families. Every planned run gets **10²⁰ training FLOPs**.

| Family | What changes |
|---|---|
| Vanilla Dense | Independent full-attention blocks and dense FFNs |
| Vanilla MoE | Full attention with conditional expert capacity |
| Hybrid | Linear and sparse-retrieval attention interleaved |
| Looped | Weights reused across depth |

A family names a design axis, and T3 studies each one through a single instance. Hybrid is studied as KDA–DSA, three KDA layers per DSA layer; Looped is studied as the fixed recursive shell, a shared core executed twice for every token. Naming the axis keeps room for other instances at later stages without renaming the family.

Three of the four families share the Vanilla MoE feed-forward recipe: a dense first layer, then 256 routed experts of width d/4 with eight selected per token and one shared expert. Only Vanilla Dense keeps a dense FFN in every layer. Holding the feed-forward side fixed is what lets a difference between Vanilla MoE, Hybrid and Looped be read as a difference in attention or weight sharing rather than in expert capacity.

Use the [observatory](https://interactive-active-ai.github.io/NextTransformer/#scaling) to inspect model shapes and switch between hardware planning estimates. The [T3 configuration](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json) is the machine-readable design. It distinguishes estimated token counts from counts awaiting a cost proof.

## Research sequence

T3 samples the first size curve. T4 refines the fit. T5 checks predictions registered using the smaller stages. After scoring T5, freeze the final forecasts for all families and evaluate them independently at C200. Details for later stages will be released progressively.

The historical [4K Dense run](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/scaling/runs/dense_200b.json) supplies a compute reference. Its quality scores and the 49B leaderboard remain outside the 2K fit. See the [protocol](../protocol/) and [contribution guide](../contribute/).
