# Inkling REL in the shared Dense shell

This plugin replaces RoPE with Inkling's content-dependent relative position bias. It is an independent
method; it does not include XSA, Inkling's MoE, short convolutions, GQA, local/global layer pattern,
learned QK norm, or Inkling's different attention scaling.

For a layer's input x and head h:

```text
r[b,i,h,:] = x[b,i,:] W_r[h]
bias[b,h,i,j] = Σ_a r[b,i,h,a] E[a,i-j]
score[b,h,i,j] = dot(q[b,i,h,:], k[b,j,h,:]) / sqrt(head_dim) + bias[b,h,i,j]
y[b,i,h,:] = Σ_{j≤i} softmax_j(score[b,h,i,j]) v[b,j,h,:]
```

W_r produces a separate 16-dimensional vector per query head. E has shape `[16, 2048]`, is trained,
and is shared across heads within each layer. Distance zero is self attention; the final slot covers
distance 2047. The plugin requires `rel_extent >= sequence_len` and full causal attention in every layer.
There is no clipping, bucketing or zero-bias region inside the 2,048-token context.

The existing parameter-free QK RMS norm, 1.2 gain, `1/sqrt(head_dim)` score scale, gated value embeddings,
residual stream, FFN and Muon + AdamW grouping rules remain intact. Both new matrices receive ordinary
Muon ownership through the existing matrix rule; no optimizer exception is introduced. New REL matrices
use zero-mean normal initialization with standard deviation 0.02, matching the public Transformers
Inkling initialization. This is an explicit adaptation, not a claim about Inkling's private pretraining recipe.

## Implementation

`reference.py` packs causal edges and uses differentiable gathers/scatters. It is the CPU correctness and
shape-only cost reference. Gather backward reductions are explicit and charged. The reference uses
dense masked softmax storage, whose padded bookkeeping is conservatively counted by the existing ledger.
The independent dense oracle computes the full matrix expression for tests.
The cost path includes both trained REL projections and their backward/optimizer work; it never
replaces REL with ordinary attention during calibration.

`kernel.py` is a Triton training implementation:

- Compute distance logits with a matrix multiplication in the activation dtype.
- Add the query/distance bias inside a tiled causal online softmax, using FP32 accumulators.
- Save Q/K/V, the small relative branch, table, output and row log-normalizers; do not retain NxN probabilities
  or per-layer distance-logit buffers.
- Recompute distance logits during backward. Separate query-owned and key-owned kernels avoid atomics for
  attention gradients. Compute the relative-branch and distance-table gradients with matrix multiplications.

The first row has a singleton softmax and its score gradients are set to exact zero.
BF16/FP16 score probabilities and intermediate score gradients are rounded for tensor-core products.
Tests compare output and all five input gradients against the FP32/FP64 mathematical reference.
The public interface currently supports equal-length full-sequence MHA. Incremental KV-cache decoding
is rejected explicitly; the repository's training, BPB and CORE paths use full sequences.
The CUDA kernel supports head dimensions 16, 32, 64, 128 and 256 in BF16, FP16 or FP32. The admitted
configuration uses dimension 128. The default tiny debug configuration uses dimension 8 and is intended
for the CPU reference; CUDA mixer checks should select a supported head dimension.

## Protocol configuration

`configs/others/inkling_rel.json` is calibrated against the fixed SSSL anchor using the shared
structural search. It has 22 layers, width 1536, 12 heads of dimension 128, FFN width 6272 and
1,293,026,934 trainable parameters. The audited cost is 4,629,592,729.874441 training FLOPs per input
token, or -0.91614% versus the anchor, within the 1% admission bound. The proof is in
`evidence/costs/inkling-rel-20261114/`; the kernel, reference and plugin source are fingerprinted.
Calibration uses no data or held-out scores. The configuration retains every fixed protocol field.

The CUDA implementation is selected automatically by this plugin. The shared trainer's
`native_attention` software field describes its installed native attention backend; REL itself
executes `kernel.py`, whose identity is covered by the method source hash.

## Verification

`tests/test_inkling_rel.py` checks FP64 forward/gradient parity, gradcheck, causality, distance
orientation, self attention, full context coverage, RoPE removal, initialization, optimizer ownership,
checkpoint round trips and cost coverage. CUDA tests cover BF16 lengths 1, 17, 65, 257 and 2048, plus
noncontiguous FP32 inputs and exact zero gradients for unused distance entries.

On a CUDA device, the admitted model completed three training-data-only updates in BF16 with micro-batch 8 and
context 2048. Every trainable parameter had a finite gradient and the two REL factors had nonzero
gradients after the initially zero output projection was updated. Peak allocated memory was 38.71 GiB.
This was a bounded single-GPU integration check with a reduced global batch, not a formal run or a
learning-quality comparison. The fixed protocol and cumulative cost audit still apply to a full run.

The unchanged shared trainer also completed a bounded eight-device integration check with admission enabled:
two full 1,048,576-token updates (`8 ranks × 8 rows × 8 accumulation steps × 2048 tokens`), with
checkpoint save, process exit and resume between updates. Loss, gradients, token counters, training-only
BPB and all eight optimizer checkpoint shards passed validation. Rank zero's peak allocated memory
across the two recorded updates was 41.75 GiB. All BPB checks used training rows; no validation shard or CORE
scores were used. This checks distributed integration, not convergence or steady-state throughput.

After activating the repository environment:

```bash
CUDA_VISIBLE_DEVICES= NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_inkling_rel.py
# On an available CUDA device:
NANOCHAT_DTYPE=bfloat16 python -m pytest -q tests/test_inkling_rel.py -k cuda
python -m nexttransformer.protocol.fingerprint --all
```

## Sources

- Inkling release: https://huggingface.co/thinkingmachines/Inkling
- Configuration revision: `828496eeae4c243ff1a22f7f28ff83694f2f7bc9` (`d_rel=16`, original global `rel_extent=1024`).
- Transformers `models/inkling/modular_inkling.py`: `InklingRelativeLogits`, `InklingAttention`, `_init_weights`.
- vLLM revision `43c1db8c9f30526fe0342c700e99d2c2648d92e4`, `models/inkling/nvidia/attention.py` and `ops/fa4_rel_attention.py`: head-shared projection and distance indexing.

The experiment extends the released distance extent to 2048 as requested. The implementation is original;
ForgeKernel's environment-check tooling and Hopper attention design notes inform the local validation workflow.
