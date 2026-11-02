# Sources and native adaptations

These implementations study individual mechanisms in a common nanochat shell. Method names do not imply reproduction of a paper's full model, dataset, recipe, or results. Existing sources were reviewed on 2026-10-18, with GDN and merge updates on 2026-10-22; full-attention Dense and Residual sources were reviewed on 2026-10-23. This English consolidation preserves those recorded adaptations; the [v2 plan](../plans/V2.md) separately lists unimplemented extensions.

## Pinned references and attribution

| Source | Recorded version |
|---|---|
| nanochat | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; [file manifest](../../configs/native/upstream_sources.json) |
| [DeepLoop](https://github.com/lszshu/DeepLoop/tree/70c170866019ed956ef6a87af7752c149ff6a5e5) | `70c170866019ed956ef6a87af7752c149ff6a5e5` |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | `87e509a2e5a100d221c97df52c6e8be7835f0057` |
| [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526) | `69885871a64393807d988b27b1b5e380e8f28526` |
| [SeerAttention](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) | `aba03e3f2caefd0ccd21e576670aa830b748c84e` |
| [MoBA](https://github.com/MoonshotAI/MoBA/tree/b5d58363311d3ca946f1ec444182727c15e338b5) | `b5d58363311d3ca946f1ec444182727c15e338b5` |
| [FLA](https://github.com/fla-org/flash-linear-attention/tree/c885377e65a36f612bd10f57b7d1daa55224c31b) | `c885377e65a36f612bd10f57b7d1daa55224c31b`; layer and naive-recurrence references |
| [FLA GatedDeltaNet layer](https://pypi.org/project/flash-linear-attention/0.5.2/) | Version 0.5.2, reference wheel SHA256 `dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400` |
| [Hyper-Connections](https://arxiv.org/html/2409.19606v3) | arXiv v3: Eqs. 10–14, Algorithms 2–3 |
| [mHC](https://arxiv.org/html/2512.24880v2) | arXiv v2: Eqs. 7–9, Appendix Table 5 (n=4, gate scale 0.01, 20 iterations) |
| [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | arXiv v1: §3, Fig. 2; [official repository](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6) `85e22310fe5ee860b4a023de312d791de8a5a5e6` holds the same pseudocode and no training code |
| [lucidrains/hyper-connections](https://github.com/lucidrains/hyper-connections/blob/e89e30357d1f79945b0d3558e6721451ff68789a/hyper_connections/manifold_constrained_hyper_connections.py) | `e89e30357d1f79945b0d3558e6721451ff68789a`; consulted only for mHC static-bias initialization |

The readable mathematical references are independently implemented. FLA/FlashMLA/Seer CUDA/Triton sources were not copied into these references. CUDA linear training calls the external MIT-licensed `fla-core==0.5.2`, hash-pinned with einops in [requirements-linear.txt](../../requirements/linear.txt). The full flash-linear-attention wheel above was inspected as a reference, not added as a dependency. Preserve the root MIT license, [upstream README](../upstream/NANOCHAT_UPSTREAM_README.md), and [NextT reference license](../upstream/NEXTT_REFERENCE_LICENSE.txt).

## GSA + QK-norm

A shared-shell [Gated Slot Attention](https://arxiv.org/abs/2409.07146) variant with
per-head QK RMSNorm × 1.2 before SiLU, 64 slots and chunk128 FP64 recurrence.
[Completed run](../../results/runs/gsa_qknorm.json).

## Common shell

[`nexttransformer/models/lm.py`](../../nexttransformer/models/lm.py) extends the pinned GPT-derived [backbone](../../nexttransformer/models/backbone.py). Historical `dense` delegates to its original forward and is checked against logits and every gradient. Other rows reuse untied input/output embeddings, VE, smear, backout, ReLU² MLP, RMSNorm, and learned residual/x0 coefficients except documented overrides. VE is attached to physical blocks and shared with shared weights. `dense_full` uses the reference execution path with window `(2048, 0)` in every layer and is checked against the native forward with `window_pattern=L` for logits and every gradient. The four Residual rows use the same full-attention shell and replace the native residual/x0 coefficients and backout with their own residual path; those scalars stay registered, frozen, and unused.

Shared cores schedule SSSL by unfolded position, forcing the final layer long. ACT/Huginn/Ouro use prelude for backout; Fixed Recursive/DeepLoop use the unfolded midpoint. Linear methods use their own feature/position rules; sparse methods use their own selections rather than an extra SSSL mask.

## Looped methods

| ID | Source / retained mechanism | Local implementation and limits |
|---|---|---|
| `ut_act` | [Universal Transformer v3](https://arxiv.org/html/1807.03819v3): coordinates, accumulated halting mass, remainder, ponder cost | Add token/depth coordinates and gate before each core traversal; mix after it. Freeze halted states and consume remaining mass at the maximum loop. All candidates still execute and are charged; not an efficient sparse ACT backend. |
| `fixed_recursive` | Project sharing control | Prelude once, shared core twice, coda once; no extra gate/conditioner. |
| `huginn` | [Recurrent depth v2](https://arxiv.org/html/2502.05171v2): sampled depth/state, input injection, truncated backpropagation | One clipped Poisson-lognormal depth per update, maximum 4 loops, final 2 backpropagated; key by routing seed/step. Row state noise is keyed by seed/global row. Concatenate state/prelude, project, and use sandwich RMSNorm. Not the complete published Huginn model. |
| `ouro` | [Ouro v1](https://arxiv.org/html/2510.25741v1): multidepth objective, entropy reward, dedicated gate phase | Coda/head at every loop. Joint exit-probability-weighted CE minus 0.1×entropy; final 1,024 updates train only the gate using detached next-loop CE improvement. Evaluation selects logits at cumulative exit probability 0.5; reference execution still computes all candidates. |
| `deeploop` | [DeepLoop v2](https://arxiv.org/abs/2607.13491v2): shared-depth alpha/beta stabilization | Two full-stack traversals; each attention/MLP sublayer uses `RMSNorm(alpha*x + F(RMSNorm(x)))`, alpha=sqrt(2N). V/O and both FFN matrices use beta=1/sqrt(8N) Xavier gain; Q/K retain native initialization. Disable native residual/x0 mixing. Scale the extra native VE value path by beta too. Required normalization controls remain separate experiments. |

Ouro gate logits mean **exit**. Gate t uses `CE(t)−CE(t+1)` so greater improvement favors continuation; the final loop receives remaining mass. Preserve gate Adam moments, update index, data cursor, and DDP semantics at the phase boundary. No-gradient coda/head work is not free.

## Sparse attention

[`sparse.py`](../../nexttransformer/models/sparse.py) gathers selected K/V before primary attention, with an independent first-order backward. Indexer and internal-teacher full-history scans remain explicitly charged. The optional [`sparse_kernels.py`](../../nexttransformer/models/sparse_kernels.py) backend uses FlexAttention over selected 128×128 tiles and a fused Triton teacher without materializing the full score tensor. Output, auxiliary loss, and gradients are compared against the reference. Partial-tile padding belongs to execution overhead; speedups are not assumed.

| ID | Source | Local adaptation |
|---|---|---|
| `dsa` | [DeepSeek report and indexer](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | Multihead ReLU indexer with learned signed head weights, key LayerNorm, partial RoPE; causal top-256 tokens. Adapted to native MHA, without MLA/MoE/FP8 indexer. |
| `qsa` | [Qwen report §2.1.2](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | Mean-compress four-token keys, then RMSNorm/partial RoPE; four query heads and shared key, summed ReLU scores. Select complete blocks and append unfinished tail; no forced tail at complete boundaries. Teacher uses normalized head mean followed by max pooling; sparse KL renormalizes both sides over selected complete blocks. No GDN hybrid, MoE, GR, MTP, or 256K CPT. |
| `nsa` | [NSA v1](https://arxiv.org/html/2502.11089v1) | Learned compression, selection, local branches, each with per-head sigmoid gates. Compression/selection blocks both 32 and nonoverlapping, making importance mapping identity; local window 512; historical selection plus current tail. A constrained variant, not a general overlap/block-size implementation. |
| `seer` | [SeerAttention v1](https://arxiv.org/html/2410.13276v1) | **SeerAttention-style** token-query to pooled-64-token-key gate with internal teacher KL. From-scratch adaptation, not the original pooled-query AttnGate or full recipe. |
| `moba` | [Official MoBA](https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/moba_efficient.py) | Per-head pooled-K block selection; K includes the forced current block. Complete historical blocks and causal current block; no teacher. Block 128, token budget 256. Independent gather/reference and local kernel backends, not the official fused kernel. |

DSA/QSA/Seer train from random initialization with 512 dense-burn-in updates and then sparse LM, all within 46,875 steps. Auxiliary KL weight is 1 per layer, summed. Teacher and indexer input detach, so auxiliary gradients update only the indexer. DSA explicitly describes this separation; Seer originally trains the gate over a frozen backbone; QSA's matching detach policy is a local adaptation. Changing weights/stages/gradient semantics requires registration and recalibration.

## Linear attention

State resets at each packed row, not at internal BOS. Preserve Q/K/V/output projections and native VE. Let S be the K×V state, g a forget gate, and beta write strength. Independent history-expansion oracles and gradient tests accompany the code.

| ID | Source | State mechanism / adaptation |
|---|---|---|
| `deltanet` | [DeltaNet](https://arxiv.org/abs/2406.06484) | `S <- S + k[beta(v-k^T S)]^T`, `y=q^T S`; causal conv+SiLU, L2 QK, sigmoid beta. |
| `gla` | [Gated Linear Attention](https://arxiv.org/abs/2312.06635) | `S <- diag(g)S + kv^T`; low-rank forget gate and SiLU output gate; no extra softmax RoPE/QK normalization. |
| `kda` | [Kimi Delta Attention](https://arxiv.org/abs/2510.26692) | Feature-wise decay before delta update; short conv, L2 QK, beta, learned decay bias. Custom first-order BPTT stores one complete state history; FP64 gradcheck covers all inputs/initial state. |
| `retnet` | [RetNet](https://arxiv.org/abs/2307.08621) | Fixed head-specific decay, rotary positions, retention update, output gate. Published GroupNorm replaced with native per-head RMSNorm. |
| `elu_linear` | [Linear Transformer](https://arxiv.org/abs/2006.16236) | phi=ELU+1; `S=sum(phi(k)v^T)`, `z=sum(phi(k))`, output `phi(q)^T S / (phi(q)^T z + epsilon)`. Positive-feature control. |
| `gdn` | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | `S <- alpha(I-beta kk^T)S + beta kv^T`; scalar per-head decay before delta update. `log alpha=-exp(A_log)*softplus(W_a x+dt_bias)`; A uniform (0,16), dt log-uniform [1e-3,1e-1]. Conv+SiLU, L2 QK, sigmoid beta, output RMSNorm × gain × SiLU gate. K/V dimension 128; native RMSNorm. KDA's existing constant gate initialization remains separate. |

`elu_norm` (A06-prime) follows the NormAttention idea (Qin et al., 2022): remove the denominator and apply per-head output RMSNorm with learned gain. `elu_qknorm` (A06-double-prime) additionally applies parameter-free FP32 RMSNorm to q/k **before** phi and multiplies by fixed tau=8, chosen from the earlier pilot's stable scales. This is related to pre-kernel normalization in [ReBased](https://arxiv.org/abs/2402.10644), without its learned affine map or quadratic kernel. Failed earlier variants remain recorded.

## Full-attention Dense and Residual family

| ID | Source / retained mechanism | Local implementation and limits |
|---|---|---|
| `dense_full` | v2 baseline: full causal MHA in every layer | Native shell, VE, smear, backout, and learned residual/x0 coefficients unchanged; only depth was calibrated. |
| `residual_standard` | Single-stream Pre-Norm residual control (v2 R00) | `h + F(norm(h))` for every attention and FFN sublayer; VE, smear, embeddings, and ReLU² FFN retained. Checked against native `Block` execution for logits and gradients. |
| `hc` | [Hyper-Connections](https://arxiv.org/html/2409.19606v3): dynamic HC, expansion 4 | One map per attention and FFN sublayer; sublayer index k selects the static read stream `k mod 4`. Static `B=1`, `A_m=e_(k mod n)`, `A_r=I`; dynamic projections zero; `s_α=s_β=0.01`. Embedding replicated into four streams; streams summed before the final norm. Branches stay 1536 wide. The dynamic-mapping norm is the native parameter-free RMSNorm instead of LayerNorm. The paper's √n output-projection scale is a no-op because native output projections start at zero. The registered auxiliary AdamW (no weight decay) replaces the paper's weight decay on dynamic projections. At initialization it reproduces `residual_standard`. |
| `mhc` | [mHC](https://arxiv.org/html/2512.24880v2) Eqs. 7–9: n=4, gate scales 0.01, 20 Sinkhorn-Knopp iterations | Flattened 4×1536 stream, parameter-free RMSNorm (its weight is absorbed into the projection, as in the paper's kernels), one 6144×24 projection for pre/post/residual logits; `H_pre=σ`, `H_post=2σ`, `H_res` = 20 column-then-row normalizations of `exp`, computed in log space: each normalization subtracts a log-sum-exp, so a row whose entries all underflow under FP32 `exp` never divides by a vanishing sum ([record 051](../records/EXPERIMENTS.md#record-051)). Projection initialized to zero (HC's rule for dynamic parameters). The paper does not state static-bias initialization; biases start at HC's Eq. 14 values (`e_(k mod n)`, ones, identity) before the constraints, as in the public reference implementation. Coefficients use at least FP32. Each log-sum-exp subtracts a detached maximum, whose gradient cancels. Twenty iterations are approximate: rows sum to one, columns only approximately. Calibration narrowed the FFN to 6016. |
| `attnres` | [Attention Residuals](https://arxiv.org/abs/2603.15031v1) Block AttnRes, about 8 groups | Depth is counted in attention and FFN sublayers; separate zero-initialized pseudo-queries for every sublayer and the final readout. Keys use the native parameter-free RMSNorm (a learnable key-norm weight would be absorbed by the query). Whole transformer blocks form groups, larger first and remainder last: 23 blocks give 3,3,3,3,3,3,3,2. Sources are the embedding, completed group sums, and the current partial sum; the boundary precedes the first attention sublayer of a group, as in Fig. 2. The final readout attends over the embedding and all eight group sums. |

All four Residual rows share the full-MHA shell with `dense_full`; report paired R00 differences separately from overall Dense differences. Published gains come from other scales and MoE backbones and do not predict results here.

## DeltaNet hybrid (v2 H01)

`deltanet_hybrid` retains the registered [DeltaNet global hybrid](https://arxiv.org/html/2406.06484v1)
layout: full causal MHA at one-based layers **2 and N/2+1**, DeltaNet elsewhere, even N. At N=24 the
full layers are 2 and 13; layer 24 is DeltaNet. This is the native-MHA adaptation of the global hybrid,
not the paper's separate local-MQA variant. Sources and layout constraints are in the [hybrid plan](../plans/V2-linear-hybrids.md#21-h01-deltanet_hybrid--confirmed).

DeltaNet reuses A01's projections, causal convolution, beta, q/k normalization and output normalization.
`fla_chunk64` uses FLA `chunk_delta_rule`, with q/k normalized and scaled before the call, `scale=1`,
no incoming/final state, and no in-kernel second normalization. The audited torch path reuses the GDN
WY solve and state update with decay absent; it does not charge fictitious GDN gates. Pure A01 keeps
its historical scan configuration. Chunk and scan implement the same recurrence but have separately
counted training algorithms. Native MHA, ReLU-squared FFN, smear, backout and residual/x0 are retained.
VE follows the existing shell: at N=24, MHA layer 2 has VE, MHA layer 13 does not, and even DeltaNet
layers receive the projected-V adaptation. There is one joint CE stage and no auxiliary loss.

H01 uses the same string-valued layer layout, model builder, direct full mixed-graph audit, calibration
grid, admission and trainer as H02/H03. Its minimum debug depth is four; linear state resets per packed
row, not at internal BOS. The shared serialization gets a new fingerprint and fresh proof. Existing
runs on [the original FLA implementation](https://github.com/Selen-Suyue/NextTransformer/commit/c5ccfd42a42eae68f2318c020b6b28fb435a5ef8)
must resume from their original frozen source, environment and proof; their
[historical deployment guide](https://github.com/Selen-Suyue/NextTransformer/blob/f85599117737c737b671a336133c9a65e66e5df0/reference/H01_CLUSTER_RUNBOOK.md)
remains available with that snapshot.

## GDN hybrids (v2 H02, H03)

Both rows combine this repository's `gdn` mixer (A07, `fla_chunk64`) with `dense_full`'s native attention: 12 heads of 128, RoPE, QK normalization, VE where the shell places it, and unchanged smear, backout, and residual/x0 coefficients. The per-layer layout is derived from the method and depth, serialized as `layer_mixers`, and verified on load. Sources were checked on 2026-10-24; see the [hybrid plan](../plans/V2-linear-hybrids.md#2-source-verification).

| ID | Source / retained layout | Local implementation and limits |
|---|---|---|
| `gdn_hybrid` | Qwen3-Next 3:1: HF model card *"12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))"*, `config.json` `full_attention_interval: 4` (the registered blog link renders no text) | GDN, GDN, GDN, full attention repeated; attention at one-based layers 4, 8, …, N. Qwen's gated attention (sigmoid output gate), 16Q/2KV heads of 256, 25% partial RoPE, zero-centered weight-decayed norms, and MoE are not reproduced. |
| `gdn_hybrid_swa` | [Gated DeltaNet](https://arxiv.org/html/2412.06464v1) H1, §3.3 (v1; §3.4 in v3): GDN + SWA with a 2K window. Order from the official [lit_gpt](https://github.com/NVlabs/GatedDeltaNet/tree/main/lit_gpt) `Block` with `gated_delta_per_layer=2` | GDN at one-based odd layers, sliding-window attention at even ones. The window holds 2048 keys including the current token (left window 2047, as the official `(local_window-1, 0)`); our SDPA fallback also applies it, unlike the official one. **At the 2048-token context the window covers every causal key, so this row computes a 1:1 GDN/full-attention hybrid** (tested). The official SWA has no QK norm or VE; ours follows `dense_full`. Mamba2-based H2 is not reproduced. |

The shell's alternating VE rule (`has_ve`: one-based even layers when N is even) puts VE on every attention layer of both rows; in `gdn_hybrid` one GDN layer in three also has VE, in `gdn_hybrid_swa` none does. VE enters GDN layers through the existing projected-V adaptation. Report these placements with the paired comparisons against A07 and `dense_full`.

## QSA and MoBA hybrids (v2 H05, H06)

H05 reuses the [registered GDN/GDN/GDN/QSA layout](../plans/V2-linear-hybrids.md#25-h05-qsa_hybrid--confirmed-sparsity-comes-from-our-s02-adaptation), A07's GDN/FLA adaptation and S02's QSA implementation. Its 256-token budget, compression 4, indexer and detach policy are the local S02 choices above. Dense burn-in affects only QSA positions for 512 updates, followed by joint sparse training within the common budget. Auxiliary terms sum over QSA layers; GDN stays unchanged. GR, n-grams, MoE, MTP and mature-model 256K continued pretraining are not reproduced.

H06 reuses [V2's last-three full-attention rule](../plans/V2.md#3-hybrids-with-registered-source-evidence), S05's MoBA (block 128, budget 256 including the current block) and B01's native MHA. Calibration preserves exactly three final MHA layers: at N=25 their proportion is 12%, compared with 3/32 = 9.375% in the registered source layout. It has no burn-in or auxiliary loss. Both rows preserve the native VE placement and shell; `--sparse-backend kernel` selects the existing local FlexAttention/teacher kernels, not the upstream MoBA fused implementation. No optimized generation-cache implementation is claimed. [Calibration and bounded evidence](../records/EXPERIMENTS.md#record-046).

## KDA / MLA-NoPE hybrid (v2 H04) and VE-off controls

`kda_hybrid` repeats KDA, KDA, KDA, full causal MLA-NoPE, with N divisible by four.
It follows V2's O03-N definition (decision D1-a), shared with `mla_nope`: Q latent rank 768,
joint KV latent rank 256, 12 heads, QK content/V dimensions 128/128, and zero RoPE dimensions.
`others.MLAAttention` is reused from [@xiongxuyuan's implementation, 8cba0a3](https://github.com/Selen-Suyue/NextTransformer/commit/8cba0a3b268263ec553862025db65a1159521158),
with its import relocated. Both latent norms are the native parameter-free RMSNorm. Training reconstructs
K/V explicitly and uses the common FA3/SDPA interface; it has no reconstructed-head QK norm or output gate.
Query/KV down-projections use native initialization, up-projections use their own fan-in, and output
projections start at zero. MLA matrices use the existing Muon groups; no optimizer policy changes.

H04 reuses A04's KDA, including its constant 0/-3 decay initialization, rank-16 decay gate,
full-rank SiLU output gate, causal convolution and `chunk_kda(disable_recompute=True)` backend.
All VE tables and VE gates are removed from H04 and its independently calibrated controls
`kda_no_ve`, `mla_nope`, and the shared full-MHA control `mha_no_ve`. FFN, smear, backout and residual/x0
stay native; every row has one joint next-token CE stage. Linear state resets at packed rows, not BOS.

This is a registered native adaptation, not the released Kimi model: Kimi has no Q compression,
KV rank 512, 32 heads and an additional shared 64-dimensional unrotated key component even in NoPE,
and different KDA gate ranks/initialization/output gating; see [the source comparison](../plans/V2-linear-hybrids.md#24-h04-kda_hybrid--registered-o03-n-adaptation).
The collaborator's [later record](https://github.com/Selen-Suyue/NextTransformer/blob/77046daf9754de2dbabf3ff194bcbe4ce17d209c/reference/EXPERIMENTS.md#record-039)
documents logit explosion in pure unnormalized MLA under Muon and separately registers QK-normalized rows.
H04 retains the registered original definition. Bounded tests do not establish long-run stability;
the separate pilot and formal acceptance remain required.

## Backend and numerical limits

Readable recurrent scans are correctness references, not FLA chunked-training kernels. KDA calls `chunk_kda(disable_recompute=True)`; RetNet calls `chunk_simple_gla` with head log decay. CUDA causal convolutions use FLA `causal_conv1d(activation="silu")`; CPU uses equivalent torch operations. Explicit `scale=1` and `FLA_DISABLE_BACKEND_DISPATCH=1` fix the registered interpretation/path.

FLA 0.5.2's GDN backward is guarded on Hopper/Triton 3.4.0–3.7.0 because of issue #640; the registered environment uses Triton 3.5.1. GDN therefore calls the validated KDA kernel with scalar log decay broadcast across key channels. QK L2 normalization/q scaling run inside the kernel; output gated RMSNorm uses FLA `rms_norm_gated`, epsilon matching the framework's FP32 approximately 1.19e-7. CPU/audit uses equivalent torch normalization and gates. Record this as a backend adaptation.

KDA/RetNet/GDN use FP32 carry and accumulation, BF16 operands and boundary states; KDA/GDN triangular inversion uses TF32. GDN's torch chunk oracle uses `exp(b_i-b_j)` over causal pairs so exponents are nonpositive. KDA's whole-chunk algebra can overflow FP32 for trained strong gates; use the FP64 recurrent oracle for trained-input checks. There are no second-order derivative or optimized generation-cache guarantees.

All ELU variants use a stable FP32 piecewise `x+1 / exp(x)` feature map, clamping the unused exponential branch to nonpositive inputs. `_ELUChunkFP32` keeps feature/attention/denominator/output-norm arithmetic in IEEE FP32, disables TF32 in forward and backward, and does not recompute. `elu_norm`/`elu_qknorm` use explicit epsilon 0.0078125 and BF16-quantized gains, then convert before output projection. This differs from the earlier mistaken description of framework BF16 epsilon: the original framework promotes BF16 norm input and actually uses FP32 epsilon. Trained-input ELU checks begin at raw Q/K projections and use actual LM upstream gradients; post-feature checks alone cannot validate phi or normalization.
