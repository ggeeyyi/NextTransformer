# Exclusive Self Attention

This plugin implements Section 3, Equation 2 and Algorithm 1 of Shuangfei Zhai's
[Exclusive Self Attention](https://arxiv.org/abs/2603.09078v1).

For each token and attention head, ordinary causal attention first produces `y`.
The plugin normalizes that token's value vector `v` to `u` and returns
`z = y - sum(y * u) * u` before concatenating heads and applying the output projection.
The self position remains in softmax. Removing the attention diagonal would implement
a different function: XSA also removes contextual components parallel to the self value
and retains the self score's contribution to the normalization.

`F.normalize` uses the paper pseudocode's default epsilon, `1e-12`. Gradients flow
through both the attention result and the exclusion direction. For zero `v`, the
projection returns `y`; for a first token with a nonzero value norm above epsilon,
the attention branch is zero up to numerical roundoff. Normalization and projection
use FP32 for FP16/BF16 inputs, then cast back before `c_proj`; FP32/FP64 stay unchanged.
This precision policy is an explicit numerical adaptation of the paper pseudocode.

## Integration with the fixed protocol

The plugin reuses the repository's FA3/SDPA interface and preserves RoPE, QK norm,
attention scaling, gated value embeddings, residual paths and parameter names.
The direction removed is the effective value supplied to attention, including the
gated value embedding. The paper does not use this value-embedding shell; this is
the adaptation used for the controlled comparison. No parameters or custom kernels
are added. The protocol uses MHA; the mixer also maps grouped query heads to their
corresponding value head when called with a GQA configuration.

The mixer supports the existing KV-cache interface. The shared `ReferenceLM` still
supports only full-sequence training/evaluation and rejects cached model generation;
this plugin does not change that shared API.

The paper's nanoGPT/FineWeb/AdamW recipe is not imported. Training retains the shared
ClimbMix data and tokenizer, 2,048-token context, seed 42, Muon + AdamW schedule,
49.152B tokens, 46,875 updates and the fixed evaluation schedule. New projection
operations are included in the existing forward/backward cost audit. Architecture
selection uses the registered calibration grid, never held-out scores.

## Checks and calibration

```bash
source launch/env.sh --site-config site.local.json
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 \
  python -m pytest -q tests/test_xsa.py tests/test_reference.py -k xsa
# Choose a new cache and proof-set name for each future code revision.
CUDA_VISIBLE_DEVICES= python -m nexttransformer.protocol.calibrate \
  --methods xsa --cache .agent-work/xsa-cost-cache-new \
  --output configs --results evidence/costs/xsa-new
CUDA_VISIBLE_DEVICES= python -m nexttransformer.protocol.fingerprint
```

The tests compare outputs and gradients against an explicit attention/projection
oracle, check zero and small values, head isolation, causal prefixes, checkpoint
round trips, mixer prefill/decode, and accounting of the extra forward/backward work.
GPU parity tests run only when CUDA and the repository's FA3 backend are available.
Calibration establishes theoretical admission; it is not a completed leaderboard run
or a measured GPU performance result.

The admitted configuration is `configs/others/xsa.json`: 23 layers, width 1,536,
12 heads of dimension 128, FFN width 6,144 and context 2,048. It has 1,355,810,568
parameters and costs 4.675544617 GFLOPs per input token, +0.06734% versus the SSSL
anchor. The complete registered-grid search and operator accounting are recorded in
`evidence/costs/xsa-20261114/`. At the same shape as `dense_full`, the projection
adds 882,924 FLOPs per input token (about 0.01889%) and no parameters.

After environment setup and bounded GPU checks, a formal run uses the normal launcher:

```bash
source launch/env.sh --site-config site.local.json
NANOCHAT_DTYPE=bfloat16 launch/train.sh --config configs/others/xsa.json \
  --site-config site.local.json --output runs/xsa-2k --micro-batch 8 --formal
```

This command starts the full protocol run. Publishing a result requires a completed
run, a passing cumulative audit and exported results.
