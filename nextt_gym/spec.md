# NextT-Gym model contract (`nextt-gym-model-v1`)

The research task: **lower validation BPB at matched training FLOPs per input token.**
Every submission is compared under the upstream [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/): same data and order,
32768-token vocabulary, 2048-token context, 1,048,576 tokens per update, fixed Muon/AdamW recipe,
seed 42. The architecture is free; these are not.

## Files

| File | Required | Content |
|---|---|---|
| `model.py` | yes | `build(config) -> torch.nn.Module` and optional hooks |
| `config.json` | yes | model fields and declarations (below) |
| `kernels.py` | no | optional fast implementation; must match the reference path numerically |
| `NOTES.md` | no | hypothesis and what changed relative to the parent |

Total size ≤ 256 KB. Submissions are immutable; a change is a new attempt with `parent_ids`.

## `model.py`

```python
def build(config: dict) -> torch.nn.Module            # required; fully initialized on the current default device
def stage_for_step(config, step, total_steps) -> str   # optional; default "joint"
def set_stage(model, stage, *, audit, step, total_steps) -> str | None   # optional
```

The module's `forward(idx, targets=None, loss_reduction="mean")` (extra optional arguments such as an unused
`kv_cache=None` are fine):

| Call | Returns |
|---|---|
| `model(idx)` | logits `[B, T, >= vocab]` |
| `model(idx, targets)` | scalar training objective (may include auxiliary losses) |
| `model(idx, targets, loss_reduction="none")` | per-token cross-entropy `[B, T]`, no auxiliary losses (used for BPB) |

`model.eval()` selects evaluation behaviour. Optional methods `set_step(step)` and `set_row_offset(row)`
are called by the trainer when present (step-keyed randomness, exact resume). Inputs are strictly causal:
logits at position *t* may depend only on tokens ≤ *t*.

**Stages.** If compute changes during training (dense burn-in, sampled depth, gate-only phases),
`stage_for_step` names the stage of every optimizer step and `set_stage` prepares the model for it.
The audit charges each stage for exactly the steps it occupies at each scale.

**Parameter roles.** Each parameter gets one role; the optimizer hyperparameters per role are fixed
(those of nexttransformer/training/optimizer.py). Default roles follow upstream names (`lm_head.*` unembedding,
`transformer.wte.*` embedding, `value_embeds.*` value_embedding, `resid_lambdas`, `x0_lambdas`,
smear/backout gates, upstream auxiliary names → aux, other 2D → `matrix` (Muon), other → `vector`).
Override with a model method `param_roles() -> {name: role}`. Roles: `unembedding, embedding,
value_embedding, resid_scalar, x0_scalar, gate_scalar, aux, vector, matrix`.

## `config.json`

```json
{
  "model": { "...": "passed to build()" },
  "model_dim": 1536,               // width used for the upstream LR scaling
  "model_dim_field": "width",      // optional: model field that model_dim follows during calibration
  "aux_loss": false,               // true if the training objective adds terms beyond token CE
  "tunable": {"n_layer": {"min": 8, "max": 48, "step": 1}},   // integer model fields calibrate may change
  "audit": {"mode": "direct"},     // or {"mode": "affine", "probe_lengths": [16,32,48]} for recurrent scans,
                                   // {"pair_block": 64, "cross_check_lengths": [192,256,320]} for chunk tiles
  "debug": {"model": {"...": "small overrides"}, "vocab_size": 64, "sequence_len": 32}
}
```

## Rules for a trustworthy cost audit

- The audit runs `model.py` under FakeTensor at formal size and counts every aten operator
  (nexttransformer/protocol/compute.py, `reference-aten-math-v2`). It cannot see inside custom CUDA/Triton kernels,
  so `model.py` must be a pure-PyTorch reference path; `kernels.py` may only accelerate it.
- Attention: call `nanochat.flash_attention.flash_attn.flash_attn_func` through the module attribute
  (exact causal-edge rule), or use explicit matmul/softmax (charged in full). Fused operators without a
  counting rule fail with `AUDIT_UNKNOWN_OP`.
- Allowed imports: `torch`, `math`, `functools`, `itertools`, `collections`, `dataclasses`, `typing`,
  `numbers`, `enum`, `einops`, `nexttransformer.models.*`, `nanochat.flash_attention`, `nanochat.common`
  (and `triton` in `kernels.py`). No file/network I/O, subprocesses, `eval`/`exec`, reseeding,
  `__torch_function__`/`__torch_dispatch__`, `torch.ops`/`torch.library`, or assignment to attributes of
  imported modules. Module level may contain only imports, definitions and assignments.

## Verification levels

| Level | Checks |
|---|---|
| V0 | files, size, static rules above, config fields |
| V1 | builds on the meta device, forward signature, roles and optimizer grouping |
| V2 | debug-size: logits shape, determinism, causality (future tokens never change past logits), finite loss and gradients, `"none"` matches the mean unless `aux_loss` |
| V3 | FLOPs audit per stage; every scale within ±1% of 4.67239832544785 GFLOPs/token |
| V4 | at run start on the target GPUs: throughput, memory, micro-batch preflight |

Every run is charged the audited per-update cost of the formal model times the updates it completes, plus its
evaluations. Smoke backends (synthetic data, debug-size model, a few updates) are charged and quoted the same way,
for those few updates.

Start from `nextt_gym/templates/` (a plain Transformer that passes V0–V2 and reaches V3 after
`calibrate(["mlp_width"])`), or from an imported upstream method.
