---
title: Scaling overview
description: The scaling stage, where architecture, parameters and training tokens are chosen together at three training budgets.
---

The leaderboard answers one question: at a fixed training budget, with the model size calibrated to it and the token
count fixed, which architecture learns best? The scaling stage asks the broader one. **At a fixed training budget,
which combination of architecture family, parameter count and training tokens reaches the lowest validation BPB and the
highest CORE?** Only the compute is fixed; each family may trade parameters against tokens.

Follow it on the observatory's [Scaling page](https://interactive-active-ai.github.io/NextTransformer/#scaling): a 3D
view of every finished run (parameters, training tokens and CORE or validation BPB, coloured by family) and the
ladder of planned runs.

## Three budgets

| Tier | Training FLOPs | Batch | Role |
|---|---|---|---|
| T3 | 1.0 × 10²⁰ | 2²⁰ tokens per update | Fit: five sizes per family |
| T4 | 2.3 × 10²⁰ | 2²⁰ tokens per update | Fit: five sizes per family; the budget of the 49B comparison protocol |
| C200 | 4.88 × 10²¹ | 2²¹ tokens per update | Test: set by the Dense 200B-token reference run |

Budgets count theoretical training FLOPs per input token, times the input tokens ([accounting](../protocol/#budget)).
At T3 and T4 each family trains five sizes, from many tokens per parameter to few, so each budget traces a curve with a
minimum. From these runs we predict each family's best size and its validation BPB and CORE at C200, register the
prediction, and then train the predicted optimum there.

## Families

| Family | Size rule | State |
|---|---|---|
| Dense | Width = 64 × layers, head dimension 128, ReLU² FFN of 4 × width, value embeddings: the backbone of the reference run | Open for claims |
| MoE | The Dense backbone with a dense first layer and 256 routed experts (top 8) plus one shared expert, each of width/4, in every other layer | Waits for its recipe to pass the [stability gate](../protocol/#families) |

Further families join through [contributions](../contribute/#add-an-architecture-family).

## What is reported

- **Per budget:** final validation BPB and CORE against parameters, one curve per family.
- **Per family:** the compute-optimal parameters, tokens and loss as functions of the budget.
- **Across families:** the compute-equivalent gain, the Dense compute that would reach a family's result, divided by
  the compute the family used.
- **At C200:** the registered forecasts against the measured runs.

## Where things live

| Path | Contents |
|---|---|
| `results/scaling/ladder.json` | Budgets, families and every cell with its size, token count, status and claim |
| `results/scaling/runs/` | One record per finished run, written by `python -m nexttransformer.scaling.export` |
| `evidence/forecasts/` | Registered forecasts and their outcomes |
| `configs/scaling/` | The configuration of each trained cell |
| `nexttransformer/scaling/`, `launch/scaling/` | The trainer for nodes without a shared filesystem, and its launchers |

The Dense reference run, 40 layers and width 2,560 trained on 200B tokens at a 4,096-token context, is the first C200
point. Scaling records are not leaderboard entries: they follow their own protocol, described in the
[scaling protocol](../protocol/).
