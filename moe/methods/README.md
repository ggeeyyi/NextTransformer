# MoE method plugins

Each contributed MoE method is one directory here, `moe/methods/<name>/`, with a `method.json` declaration
and a Python package that implements its hooks (`router`, `ffn`, `mixer`, `block`, `model`, `init`, `check`).
The registry discovers it; a plugin never edits `moe/model.py`, the trainer, the data stream, the ledger or the
evaluation, so it cannot change another method.

The hook contract is in [`moe/plugins.py`](../plugins.py) and the end-to-end submission flow (calibrate,
train `--formal`, evaluate, audit, export, pull request) is in [`moe/README.md`](../README.md#submitting-a-method).
Test plugins in [`tests/moe_plugins/`](../../tests/moe_plugins) show a router and a whole-FFN replacement.

## 240 TPP residual comparison (`d768-240tpp` and `d1024-240tpp`)

The [d768-240tpp](../configs/d768-240tpp/) and [d1024-240tpp](../configs/d1024-240tpp/) configurations share the
same attention, experts and routing within each track. The four plugins each carry four residual streams, and
each plugin's declaration links to the paper and fixes its options.

| Method | Residual computation | `d768-240tpp` FLOPs/tok vs baseline | `d1024-240tpp` FLOPs/tok vs baseline |
|---|---|---:|---:|
| [baseline](../model.py) | The reference model adds every attention and FFN output to one residual stream. | 0.0000% | 0.0000% |
| [HC](hc/) | Dynamic read, residual mixing and write weights for each attention and FFN sublayer. | +0.3463% | +0.3510% |
| [mHC](mhc/) | Constrained mixing with 20 Sinkhorn iterations per sublayer. | +0.9129% | +0.9253% |
| [SimpleHC](simplehc/) | One group's read and write gates are computed at block entry for its two sublayers. | +0.3148% | +0.3191% |
| [iHC](ihc/) | Each sublayer reads and writes four streams with scalar gates and identity residual transport. A learned read merges the final streams. | +0.2931% | +0.2944% |

The [d768-240tpp](../evidence/costs/d768-240tpp/) and [d1024-240tpp](../evidence/costs/d1024-240tpp/) cost
records contain each measured configuration and its source hash.
