# Dynamic-KV ShortConv · MLP output

This plugin adapts the head-shared, input-conditioned short-convolution mechanism
in the cached primary Inkling 2 implementation to the Static ShortConv comparison. It is not
an exact reproduction of Inkling 2 or its complete architecture.

Every full-attention Dense layer has four kernel-four convolution sites: dynamic
K after projection and before RoPE/QK normalization; dynamic V after projection
and before the existing gated value-embedding addition; static attention output
after the output projection; static MLP output after the down projection.
Ordinary shell residuals, RoPE, QK normalization, value embeddings and their gate,
squared-ReLU MLP, optimizer, and schedule are unchanged. No Q convolution,
AttnRes, XSA, REL, extra gate, activation, or residual scale is introduced.

For each K/V branch, x is the normalized hidden input to attention, z its projected
K/V, and a[t] = P(x[t]) has shape (heads, 4). Each independent, bias-free P maps
width to heads * 4. The operator is

    y[t,h,c] = z[t,h,c] + sum_lag (W[h,c,lag] + a[t,h,lag]) * z[t-lag,h,c]

Negative indices are zero. Dynamic taps use the output position t, shared across
channels only within each head. K/V generators and static taps are independent.
Static taps retain Static ShortConv shape (width, 1, 4), with stored index 3-lag addressing
lag. Generator output index lag directly addresses that lag. The shared optimizer
assigns static taps to non-matrix AdamW and generator matrices to Muon.

Both taps and generators start at zero. ZeroLinear construction and resets consume
no random draws, preserving all shared parameters at equal shape and seed.
The original Static ShortConv conv1d plus residual path is retained verbatim. Dynamic mixing
is accumulated separately in FP32 for BF16/FP16 input (FP64 remains FP64 for the
oracle), cast back, and added to that static result. Zero dynamic weights reduce
to the exact static values; nonzero low-precision arithmetic can differ from a
single fused sum by rounding. CPU double forward/backward checks use 1e-12;
GPU BF16 forward tolerances are atol=0.04, rtol=0.025; gradient relative L2
error must be below 0.03. Exact-speedup model outputs and gradients must be
bit-identical using math SDPA to isolate FA3 backward nondeterminism.
All arithmetic, including generator projection, dynamic products and sums,
backpropagation and shared optimizer work, is counted by the existing ledger.

Packed rows and calls reset state. BOS within a row does not reset state. Training
and full-sequence evaluation are supported; incremental KV-cache decoding raises
NotImplementedError. These boundary choices, zero initialization, lack of the
reference 1/sqrt(2*k+1) scale, separate static arithmetic, full-attention shell,
and static branch outputs are deliberate comparison adaptations.

Primary source: Transformers modeling_inkling2.py, class
Inkling2DynamicShortConvolution, cached from
https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/inkling2/modeling_inkling2.py.
The original cache did not pin a Git revision; the immutable content hashes are:

- `inkling2-model.txt` SHA-256: `f245732ca6825d16cc16d4f162985d5042fdc436bc177e7c8cd8fc35ebce146d`
- `inkling2-config.txt` SHA-256: `58211cb34baf5e71ef4cc12c2f17466775a0bfb2c23d5abae9ecc3bd44f770fb`
- `sources.json` SHA-256: `47636c1862ea30936636b473f13e8eba0248541cb64a647f71ce5da9650ae241`

Configuration selection uses only the official structural FLOPs calibration;
training data is used for bounded preflight. Validation and CORE-22 are reserved
for the formal 49.152B-token run. The complete formal run passed the cumulative audit. Compare with the existing Dense and Static ShortConv runs
at matching stages; node-count differences prevent attributing throughput changes
solely to the dynamic operator.

Official calibration selects the same shape as Static ShortConv: 23 layers, width 1536,
12 heads, FFN width 6144, context 2048. There are 1,359,767,304 parameters.
Training cost is 4,700,459,139.729808 FLOPs/input token, +0.600565541% versus
the SSSL anchor, including forward, backward and optimizer. No floating operator
is unaccounted for. The independent proof is
`evidence/costs/sconv-dynamic-out-20261119/sconv_dynamic_out.json`.
