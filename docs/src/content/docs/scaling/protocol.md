---
title: Scaling protocol
description: Budget accounting, the frozen recipe, family size rules, evaluation, fitting and forecasts of the scaling stage.
---

This is the scaling protocol `nextt-scale-4k-v1`. It is separate from the frozen 49B
[reference protocol](../../protocol/overview/), whose training contract it does not change. Machine-readable values
live in `results/scaling/ladder.json` and `nexttransformer/scaling/config.py`.

## Budget

A run's budget is its theoretical training FLOPs: FLOPs per input token from
`nexttransformer.models.native.model.theoretical_training_flops`, times the input tokens. For the Dense backbone with
`L` layers, width `d`, context `T = 4096` and vocabulary `V = 32768`, that is `72·d²·L + 12·T·d·L + 6·d·V` per token,
with every causal attention pair counted. An MoE counts its active parameters in place of `12·d²·L`.

A cell's token count is its tier budget divided by its FLOPs per token, rounded to whole updates; the result is within
1% of the budget. The ladder file lists every count, and the trainer's `--dry-run` prints the plan for a configuration.

| Tier | Training FLOPs | Tokens per update |
|---|---|---|
| T3 | 1.0 × 10²⁰ | 1,048,576 |
| T4 | 2.3 × 10²⁰ | 1,048,576 |
| C200 | 4.88 × 10²¹ | 2,097,152 |

## Recipe

Every run uses the recipe of the Dense reference run, `configs/scaling/dense_200b.json`. Only the model size, the token
count and the tier's batch change.

| Item | Value |
|---|---|
| Data | Pinned ClimbMix revision, the shared train and validation split, BOS best-fit packing |
| Tokenizer | The pinned 32,768-token tokenizer |
| Context | 4,096 tokens |
| Optimizer | Muon for matrices, AdamW for embeddings, the output head and scalars; learning rates scaled by √(batch / 524,288), and those of the embeddings and the output head also by √(768 / width); Muon weight decay by nanochat's rule for the trained model |
| Schedule | 40 warmup updates, constant, linear warmdown over the last 65% to 5% of the peak |
| Seed | 42 for a cell's first run |
| Evaluation | Validation BPB every 1,000 updates and at the end; full 22-task CORE every 20,000 updates and at the end; a final BPB pass on 20,971,520 tokens |

A change to the recipe is a new protocol version; cells trained under an older version are not mixed into the same fit.

## Families

A family is a backbone with one size knob, the number of layers, and a rule that derives every other dimension from it.

- **Dense:** width = 64 × layers, head dimension 128 (heads = width / 128), ReLU² FFN of 4 × width, value embeddings
  on alternating layers.
- **MoE:** the Dense backbone; the first layer keeps a dense FFN, every other layer has 256 routed experts, top-8
  routing, and one shared expert, each of width / 4.

A family's cells open only after its recipe passes the **stability gate**: at least 1,500 updates at T3 and at the C200
scale without load collapse (the busiest expert of every layer below 12 times its fair share, and the per-layer mean
below 3) and without divergence. MoE is gated today.

## Evaluation

The plotted and fitted quantities are a run's final validation BPB from the final BPB pass, at the run's 4,096-token
context, and its final CORE. Validation BPB from the 49B protocol is measured at 2,048 tokens and is not compared with
scaling runs.

## Fitting

1. For each family, fit `L(N, D) = E + A / N^α + B / D^β` to every run of that family at T3 and T4 and to its recorded
   C200 runs, with `N` the block parameters (active block parameters for MoE) and `D` the training tokens.
2. Check the fit per tier: within T3 and within T4, a parabola in `log N` locates the minimum, which must lie inside the
   five sizes.
3. Derive the compute-optimal `N`, `D` and loss at C200 under the budget accounting above, with bootstrap intervals
   over runs.
4. Report the compute-equivalent gain of each family over Dense at T3, T4 and C200.

## Forecasts and the C200 test

Before the first C200 run of a family starts, at least one forecast for it is merged into `evidence/forecasts/`
([format](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/forecasts)): the records it
uses, the method, the predicted optimum and the predicted validation BPB and CORE with 90% intervals. Anyone may
register one. The maintainers then train the optimum of their own forecast, rounded to the nearest size the family rule
allows, and publish the outcome next to every forecast, whether it held or not.

## Records

`python -m nexttransformer.scaling.export` writes `results/scaling/runs/<id>.json` from the run directory. It refuses a
run that did not complete, that was trained with a different configuration, or that misses a scheduled validation or a
full CORE evaluation. A record keeps scientific quantities only; it never names hosts, paths, devices or commands.
Records are immutable.
