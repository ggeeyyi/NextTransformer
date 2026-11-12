# Scaling-stage records

Runs trained with a scaling-stage configuration (`configs/scaling/`), outside the frozen 49B comparison protocol.
They are not leaderboard entries: each one names its own protocol version, token budget, context length and batch.

| Path | Contents |
|---|---|
| `runs/<id>.json` | One record per finished run, written by `python -m nexttransformer.scaling.export` from the run directory |
| `ladder.json` | The scaling ladder: budgets, family size rules, and every planned cell with its size, token count, status and claim |

A record holds the configuration and its hash, parameter counts, the training budget and measured cost, the source,
data and software fingerprints, every scheduled validation BPB and CORE evaluation, the final BPB and per-task CORE,
and links to the W&B run and the released checkpoint. It describes the protocol, not the machines: no host names,
paths, device models or commands. Records are immutable; `python -m nexttransformer.scaling.export --check`
validates all of them.

| Record | Model | Tokens | Context | Notes |
|---|---|---|---|---|
| [`dense_200b`](runs/dense_200b.json) | Dense, 40 layers, width 2,560 | 200B | 4,096 | Reference run of the scaling stage ([config](../../configs/scaling/dense_200b.json)) |
