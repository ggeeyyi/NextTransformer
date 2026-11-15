---
title: Scaling at fixed compute
description: Four architecture families, a fixed training budget, and preregistered performance forecasts.
---

How should each architecture divide a fixed training budget between model size and tokens? The 2K study starts at **T3**, with four execution depths for each of four families. Every planned run gets **10²⁰ training FLOPs**.

| Family | What changes |
|---|---|
| Vanilla Dense | Independent full-attention blocks and dense FFNs |
| Vanilla MoE | Full attention with conditional expert capacity |
| Dense KDA–DSA | Three KDA layers per DSA layer, dense FFNs throughout |
| Fixed Recursive | A shared core executed twice for every token |

Use the [observatory](https://interactive-active-ai.github.io/NextTransformer/#scaling) to inspect model shapes and switch between hardware planning estimates. The [T3 configuration](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json) is the machine-readable design. It distinguishes estimated token counts from counts awaiting a cost proof.

## Research sequence

T3 samples the first size curve. T4 refines the fit. T5 checks predictions registered using the smaller stages. After scoring T5, freeze the final forecasts for all families and evaluate them independently at C200. Details for later stages will be released progressively.

The historical [4K Dense run](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/scaling/runs/dense_200b.json) supplies a compute reference. Its quality scores and the 49B leaderboard remain outside the 2K fit. See the [protocol](../protocol/) and [contribution guide](../contribute/).
