# Training compute: accounting, calibration, and admission

The controlled quantity is **average theoretical training FLOPs per input token, weighted across all training stages**, as defined by the [protocol](../../PROTOCOL.md). It is not parameter count, inference depth, wall time, or one forward pass.

The [evidence index](../../evidence/README.md#current-proof-set) identifies the current source-aware proof set: `plugin-interface-20261031`, re-audited after the method-plugin interface ([record 055](../records/EXPERIMENTS.md#record-055)); its counts equal `method-fingerprints-20261031`, which introduced per-method source fingerprints ([record 054](../records/EXPERIMENTS.md#record-054)) with counts identical to `mhc-logsinkhorn-20261028`, which followed mHC's log-space Sinkhorn-Knopp ([record 051](../records/EXPERIMENTS.md#record-051)). The historical [gsa-removal-20261027](../../evidence/costs/gsa-removal-20261027/summary.json) set re-audited the source with identical counts after GSA's removal ([record 040](../records/EXPERIMENTS.md#record-040)); it, the earlier [layout-20261024](../../evidence/costs/layout-20261024/summary.json), the [merge proofs](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/merged-20261022/summary.json), and prior records remain immutable. The historical layout re-audit retained every selected structure and the same mathematical counting policy.

## Denominator and execution

```text
N = 49,152,000,000 input tokens
G = 1,048,576 input tokens / update
S = 46,875 optimizer updates
F_step = F_forward + F_backward + F_optimizer
C_train = sum(F_step) / N
```

Repeated loops, teachers, and readouts increase the numerator, not the input denominator. Validation/CORE costs are separate. `compute.py` uses FakeTensor at formal dimensions and a logical **512 × 2048** global batch, executing forward, autograd, custom first-order backward, and the unfused mathematical optimizer body. It allocates no model storage and reads no training data.

During counting, backbone and gathered attention are temporarily replaced by shape operators with explicit edge rules. Numerical forward/backward correctness is tested separately. No-gradient forward paths have no invented backward charge.

## `reference-aten-math-v2`

FMA=2. Floating math is counted consistently across dtypes. Casts, movement, indexing, integer comparisons/top-k, and communication are outside the FLOP account. Nonlinear unit-operation conventions are reproducible accounting choices, not equal runtime costs.

| Operation | Count |
|---|---|
| matmul / bmm | `2 × batch × M × N × K`; backward counts actual multiplications |
| Conv1d | `2 × output_numel × input_channels_per_group × kernel_size`; dX/dW charged when needed |
| add/sub/mul/div, exp/log/sqrt/rsqrt, sin/cos/tanh, pow, clamp/ReLU | One per output element; nonunit add alpha adds a multiplication |
| sigmoid / SiLU / softplus / ELU | 4 / 5 / 4 / 3 per element |
| sigmoid / tanh / SiLU / softplus / ELU backward | 3 / 3 / 8 / 5 / 3 per element |
| sum / mean / max reduction | Input count minus output count; mean also divides each output |
| softmax / log-softmax | Forward 5 per score; backward 4 |
| RMSNorm | Actual expanded pow/mean/rsqrt/mul and backward; no double counting |
| Fused LayerNorm | Forward `7N+5R`; dX `10N`; each optional affine gradient `2N` |
| L2 norm | `2N + output_numel` |
| Embedding | Lookup zero; dense-gradient scatter one accumulation per gradient element |
| scatter_add | One per floating source element |
| NLL / BCE-with-logits | NLL forward/backward 2 per target, mean adds 1; BCE forward 10 per logit, actual backward |
| cumsum | `n − n/L` additions per scan |
| Triangular solve | Unit lower triangle `n(n−1)` per column; nonunit adds n divisions |
| tril / triu / eye | Zero floating operations |

v2 retains v1's rules and adds cumsum, triangular solves, zero-cost structural operators, and chunk causal-pair accounting. `matrix_total` exposes the matrix-only subtotal. Unregistered floating operations populate `unknown` and fail calibration; never silently charge them zero.

## Attention and optimizer

A left window `w` includes the current key, allowing at most `w+1` keys. `causal_edges(T,w)` handles triangular boundaries exactly: SSSL short=512 allows at most 513 keys. For C required edges, Q/K dimension d, V dimension v: forward matrices cost `2C(d+v)`; backward with all Q/K/V gradients costs `4C(d+v)`, plus scale/softmax. Frozen inputs remove only the actual absent gradient paths.

DSA counts `min(K,i+1)`; QSA counts selected complete blocks plus the unfinished tail; MoBA includes the mandatory current block in K; NSA counts compression, selection, and local branches separately. Tests compare analytical edge counts with actual masks.

Indexer/teacher and NSA compression score matrices are charged at their actual full shapes before masks. Sparse primary attention gathers selected K/V. The optional FlexAttention/fused-teacher backend is numerically checked independently. `padded_envelope` reports reference attention padding/full-score envelopes, **not complete hardware instruction counts or MFU evidence**.

Optimizer counting executes the pinned AdamW/Muon math, including momentum, five Polar Express iterations, MuonEq/Muon+, variance reduction, and weight decay. Count one logical parameter update, with real trainable groups; DDP/shard padding and extra microbatch reductions are execution overhead. Ouro gate-only preserves existing gate moments and schedule.

## Scan and chunk backends

Scan counts are checked at T=16/32/48: every operator and count field must satisfy the affine relation before extrapolation to T=2048. A prior quadratic slice-backward path was fixed with one `unbind`; failed affine traces were not accepted.

`fla_chunk64` uses FLA 0.5.2 for KDA, RetNet, and GDN. All three ELU variants retain that backend name but execute PyTorch IEEE FP32 chunks. The cost model counts the **chunk training algorithm**, not the recurrent scan or actual fused-kernel instructions.

- The pure-torch `chunk_reference` in `nexttransformer/models/linear.py` implements scalar-decay simple-GLA and WY/UT gated delta algorithms. Autograd supplies the actual mathematical backward. GDN uses scalar-gate pairwise decay.
- A C×C chunk tile counts only causal pairs `i≥j`, or `C(C+1)/2`; dense padding goes to the envelope. Triangular solves are already triangular and are not scaled twice. Skip zero-state readout at the first chunk and unused state update after the last.
- Audit T=2048 directly and cross-check every operator against 3/4/5 chunks (192/256/320 tokens). The structure is affine only from three chunks onward. Reject dimensions that make tile classification ambiguous.
- Kernel-internal replay, tile padding, and BF16/TF32 arithmetic are execution-account details. Required algorithmic recomputation is still charged. KDA/GDN use `disable_recompute=True` to retain w/u/h.
- GDN's scalar algorithm is audited directly. Its CUDA KDA kernel broadcasts a scalar gate over 128 channels; this generic-kernel redundancy is an execution overhead (about 516,021 extra FLOPs/token across 23 layers, approximately 0.011% of budget), not budget filler.
- Fused causal convolution+SiLU and GDN fused QK normalization/output gated RMSNorm preserve the mathematical operations counted by their torch equivalents.
- ELU's `_ELUChunkFP32` is the actual CPU/CUDA/audit graph: stable feature mapping, batched causal chunk matrices, cumsum state propagation, and full backward, with TF32 disabled. Denominator/output normalization and gains are method-specific charges. The 2026-10-20 correction reduced each original ELU row by 116,400 FLOPs/token; older proofs remain available.
- KDA's plain chunk algebra can overflow FP32 for trained strong gates (block cumulative decay below roughly −88). It remains a cost representation and a bounded-input oracle; use FP64 recurrent checks for trained inputs. FLA uses subchunk reference points with nonpositive exponents. GDN's pairwise decay also keeps exponents nonpositive.

Precision is explicit: KDA/RetNet/GDN use FP32 carry/accumulation, BF16 operands and boundary states; KDA/GDN triangular inversion uses TF32. ELU feature-to-attention-to-division/output-normalization paths use IEEE FP32 before the output projection. See [source notes](../methods/SOURCES.md) for exact norms and fused backend choices.

Residual-family maps are charged as executed: HC/mHC stream norms, dynamic projections, tanh/sigmoid, all 20 Sinkhorn-Knopp iterations and their backward, stream mixing, and the final stream sum; AttnRes key norms, depth logits, softmax, and weighted sums. Casts to coefficient precision are free structural operations.

## Stages, selection, and admission

ACT charges all candidates. Huginn enumerates the frozen depth sample for all 46,875 steps and charges early no-gradient forwards plus the final two backward loops. Ouro includes each coda/head and its final 1,024 gate-only updates. DSA/QSA/Seer and H05 include 512 dense-burn-in and 46,363 sparse updates, with detached teacher/indexer inputs and indexer-only auxiliary gradients. H05 applies these paths only at its QSA positions; H06 has one joint stage. Hybrid attention edge accounting uses the executed layer kind, not the whole-model method.

Existing configs preserve their legacy cost-only selection: begin at registered candidates, search stack depth 16–32 or shared-core depth 1–16 in the budget direction when outside 1%, then use FFN multiples of 128 in [3d,5d] if needed. A finite-difference slope proposes a candidate, which is fully re-audited. The v2 candidate tie-break policy is separately preregistered in the [protocol](../../PROTOCOL.md#training-compute-and-calibration); it is not yet a claim about the legacy search implementation. The five v2 rows (`dense_full`, R00–R03) were selected by the legacy search, and their structures also satisfy that tie-break ([record 028](../records/EXPERIMENTS.md#record-028)).

`protocol.admit_config` rejects stale/unmatched proofs, different PyTorch, changed model/protocol hashes, unknown operations, missing stages, and preflight error above 1%; formal runs (`nexttransformer.train --formal`) require it, while research mode records these findings. `BudgetState` tracks optimizer steps, tokens, cumulative compute, and completed evaluation events; incomplete training/evaluation cannot be `matched`. Final cumulative error must be ≤2%.

Each proof fingerprints the source its method runs ([`protocol/fingerprint.py`](../../nexttransformer/protocol/fingerprint.py)). Calibration records the repository files executed while the method's cost is audited, plus the repository modules those files import (not followed from the hubs `lm.py` and the models package, nor from the core); the protocol, cost ledger, calibration, fingerprint and optimizer sources are always included. Files are hashed without comments, docstrings or layout, and module-level data tables of model files keep only the entries of the method and its layer mixers. Registry families are display metadata and are not fingerprinted. A plugin's whole package belongs to its method's fingerprint, and the plugin interface (`models/plugins.py`) counts only for the methods that run it ([plugin guide](../guide/PLUGINS.md)). A change to another method's module or table entries, or to comments only, keeps a proof valid; a change to anything the method runs requires a new audit (`protocol.calibrate --methods <method>`). `python -m nexttransformer.protocol.fingerprint` compares every configuration with the current source, and CI fails when a published method no longer matches its proof. The global `source_hash` remains in proofs as provenance only.

All current configuration proofs have `cumulative_run_audit=not_run`: this field is a preflight artifact, not a claim about separate frozen runs. Never combine historical native matrix-only estimates with the full accounting table as if they were the same budget.
