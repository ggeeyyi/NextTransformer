---
title: Source notes
description: What each implementation retains from its original method, and every adaptation to the shared shell.
---

These implementations study individual mechanisms in one nanochat shell. A method name does not imply a reproduction
of the paper's full model, dataset, recipe or results. Each section lists the source a method was checked against and
the local adaptations.

## Pinned references

| Source | Recorded version |
|---|---|
| nanochat | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; file manifest in `configs/native/upstream_sources.json` |
| [DeepLoop](https://github.com/lszshu/DeepLoop/tree/70c170866019ed956ef6a87af7752c149ff6a5e5) | `70c170866019ed956ef6a87af7752c149ff6a5e5` |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | `87e509a2e5a100d221c97df52c6e8be7835f0057` |
| [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526) | `69885871a64393807d988b27b1b5e380e8f28526` |
| [SeerAttention](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) | `aba03e3f2caefd0ccd21e576670aa830b748c84e` |
| [MoBA](https://github.com/MoonshotAI/MoBA/tree/b5d58363311d3ca946f1ec444182727c15e338b5) | `b5d58363311d3ca946f1ec444182727c15e338b5` |
| [FLA](https://github.com/fla-org/flash-linear-attention/tree/c885377e65a36f612bd10f57b7d1daa55224c31b) | `c885377e65a36f612bd10f57b7d1daa55224c31b`; layer and naive-recurrence references |
| [FLA GatedDeltaNet layer](https://pypi.org/project/flash-linear-attention/0.5.2/) | Version 0.5.2 |
| [Hyper-Connections](https://arxiv.org/html/2409.19606v3) | arXiv v3: Eqs. 10–14, Algorithms 2–3 |
| [mHC](https://arxiv.org/html/2512.24880v2) | arXiv v2: Eqs. 7–9, Appendix Table 5 (n = 4, gate scale 0.01, 20 iterations) |
| [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | arXiv v1: §3, Fig. 2; [official repository](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6) |
| [lucidrains/hyper-connections](https://github.com/lucidrains/hyper-connections/blob/e89e30357d1f79945b0d3558e6721451ff68789a/hyper_connections/manifold_constrained_hyper_connections.py) | Consulted only for mHC static-bias initialization |

The readable mathematical references are implemented independently, except for the attributed MSAR kernel dependency
described below. CUDA linear training calls the external MIT-licensed `fla-core==0.5.2`, hash-pinned in `requirements/linear.txt`.

## Common shell

`nexttransformer/models/lm.py` extends the pinned GPT-derived backbone. The historical `dense` (SSSL) delegates to its
original forward and is checked against logits and every gradient. Other methods reuse untied input and output
embeddings, value embeddings (VE), smear, backout, the ReLU² MLP, RMSNorm and learned residual and x0 coefficients,
except documented overrides. VE is attached to physical blocks and shared with shared weights. `dense_full` uses the
reference execution path with window `(2048, 0)` in every layer and is checked against the native forward with
`window_pattern=L`.

Shared cores schedule SSSL windows by unfolded position, forcing the final layer long. ACT, Huginn and Ouro use the
prelude for backout; Fixed Recursive and DeepLoop use the unfolded midpoint. Linear methods use their own feature and
position rules; sparse methods use their own selections rather than an extra SSSL mask.

## Looped methods

| ID | Source and retained mechanism | Local implementation and limits |
|---|---|---|
| `ut_act` | [Universal Transformer v3](https://arxiv.org/html/1807.03819v3): coordinates, accumulated halting mass, remainder, ponder cost | Token and depth coordinates and a gate before each core traversal; mixing after it. Halted states are frozen and the remaining mass is consumed at the maximum loop. All candidates still execute and are charged; not an efficient sparse ACT backend. |
| `fixed_recursive` | Weight-sharing control | Prelude once, shared core twice, coda once; no extra gate or conditioner. |
| `huginn` | [Recurrent depth v2](https://arxiv.org/html/2502.05171v2): sampled depth and state, input injection, truncated backpropagation | One clipped Poisson-lognormal depth per update, at most 4 loops, the final 2 backpropagated; keyed by routing seed and step. Row state noise is keyed by seed and global row. State and prelude are concatenated, projected, with sandwich RMSNorm. Not the complete published Huginn model. |
| `ouro` | [Ouro v1](https://arxiv.org/html/2510.25741v1): multi-depth objective, entropy reward, dedicated gate phase | Coda and head at every loop. Joint exit-probability-weighted CE minus 0.1 × entropy; the final 1,024 updates train only the gate using the detached next-loop CE improvement. Evaluation selects logits at cumulative exit probability 0.5; reference execution still computes all candidates. |
| `deeploop` | [DeepLoop v2](https://arxiv.org/abs/2607.13491v2): shared-depth alpha/beta stabilization | Two full-stack traversals; each attention and MLP sublayer uses `RMSNorm(alpha·x + F(RMSNorm(x)))`, alpha = √(2N). V/O and both FFN matrices use the beta = 1/√(8N) Xavier gain; Q/K keep native initialization. Native residual/x0 mixing is disabled; the extra VE value path is also scaled by beta. |

Ouro gate logits mean **exit**. Gate t uses `CE(t) − CE(t+1)`, so a greater improvement favours continuation; the final
loop receives the remaining mass. Gate Adam moments, the update index, data cursor and DDP semantics are preserved at
the phase boundary. No-gradient coda and head work is not free.

## Sparse attention

`sparse.py` gathers the selected K/V before primary attention, with an independent first-order backward. Indexer and
internal-teacher full-history scans are charged explicitly. The optional `sparse_kernels.py` backend uses FlexAttention
over selected 128 × 128 tiles and a fused Triton teacher without materializing the full score tensor; its output,
auxiliary loss and gradients are compared with the reference. Speed-ups are not assumed.

| ID | Source | Local adaptation |
|---|---|---|
| `dsa` | [DeepSeek report and indexer](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | Multi-head ReLU indexer with learned signed head weights, key LayerNorm, partial RoPE; causal top-256 tokens. Adapted to native MHA, without MLA, MoE or the FP8 indexer. |
| `qsa` | [Qwen report §2.1.2](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | Mean-compressed four-token keys, then RMSNorm and partial RoPE; four query heads and a shared key, summed ReLU scores. Complete blocks are selected and the unfinished tail appended. The teacher uses the normalized head mean followed by max pooling; the sparse KL renormalizes both sides over the selected complete blocks. |
| `nsa` | [NSA v1](https://arxiv.org/html/2502.11089v1) | Learned compression, selection and local branches, each with per-head sigmoid gates. Compression and selection blocks are both 32 and non-overlapping; local window 512; historical selection plus the current tail. A constrained variant. |
| `seer` | [SeerAttention v1](https://arxiv.org/html/2410.13276v1) | **SeerAttention-style** gate from token queries to pooled 64-token keys, with an internal teacher KL. A from-scratch adaptation, not the original pooled-query AttnGate or recipe. |
| `moba` | [Official MoBA](https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/moba_efficient.py) | Per-head pooled-K block selection; K includes the forced current block. Block 128, token budget 256; no teacher. Independent gather and local kernel backends, not the official fused kernel. |

DSA, QSA and Seer train from random initialization with 512 dense burn-in updates and then sparse language modelling,
all within 46,875 updates. The auxiliary KL weight is 1 per layer, summed. Teacher and indexer inputs are detached, so
auxiliary gradients update only the indexer.

## Linear attention

State resets at each packed row, not at internal BOS. Q/K/V and output projections and native VE are preserved. Let S be
the K × V state, g a forget gate and beta the write strength. Independent history-expansion oracles and gradient tests
accompany the code.

| ID | Source | State mechanism and adaptation |
|---|---|---|
| `deltanet` | [DeltaNet](https://arxiv.org/abs/2406.06484) | `S ← S + k[β(v − kᵀS)]ᵀ`, `y = qᵀS`; causal conv + SiLU, L2-normalized Q/K, sigmoid β. |
| `gla` | [Gated Linear Attention](https://arxiv.org/abs/2312.06635) | `S ← diag(g)S + kvᵀ`; low-rank forget gate and SiLU output gate. |
| `kda` | [Kimi Delta Attention](https://arxiv.org/abs/2510.26692) | Feature-wise decay before the delta update; short conv, L2 Q/K, β, learned decay bias. A custom first-order backward stores one complete state history; FP64 gradcheck covers all inputs and the initial state. |
| `retnet` | [RetNet](https://arxiv.org/abs/2307.08621) | Fixed head-specific decay, rotary positions, retention update, output gate. The published GroupNorm is replaced with the native per-head RMSNorm. |
| `elu_qknorm` (ELU-Norm) | [Linear Transformer](https://arxiv.org/abs/2006.16236) | Per-head RMSNorm and a fixed scale of 8 precede the positive feature map. The output uses per-head RMSNorm on the accumulated state readout. |
| `gdn` | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | `S ← α(I − βkkᵀ)S + βkvᵀ`; scalar per-head decay before the delta update; output RMSNorm × gain × SiLU gate. K/V dimension 128. |

`elu_norm` follows the NormAttention idea: remove the denominator and apply a per-head output RMSNorm with a learned
gain. `elu_qknorm` additionally applies a parameter-free FP32 RMSNorm to q and k **before** φ and multiplies by a fixed
τ = 8, chosen from an earlier pilot's stable scales, related to pre-kernel normalization in
[ReBased](https://arxiv.org/abs/2402.10644) without its learned affine map or quadratic kernel.

## Full-attention Dense and Residual family

| ID | Source and retained mechanism | Local implementation and limits |
|---|---|---|
| `dense_full` | Full causal MHA in every layer | Native shell, VE, smear, backout and learned residual/x0 coefficients unchanged; only depth was calibrated. |
| `residual_standard` | Single-stream Pre-Norm residual control | `h + F(norm(h))` for every attention and FFN sublayer; VE, smear, embeddings and the ReLU² FFN retained. Checked against native block execution for logits and gradients. |
| `hc` | [Hyper-Connections](https://arxiv.org/html/2409.19606v3): dynamic HC, expansion 4 | One map per attention and FFN sublayer; sublayer k reads stream `k mod 4` at initialization. Static `B = 1`, `A_m = e_(k mod n)`, `A_r = I`; dynamic projections zero; `s_α = s_β = 0.01`. The embedding is replicated into four streams; streams are summed before the final norm. Branches stay 1536 wide. The dynamic-mapping norm is the native parameter-free RMSNorm instead of LayerNorm, and the auxiliary AdamW group (no weight decay) replaces the paper's weight decay on dynamic projections. At initialization it reproduces `residual_standard`. |
| `mhc` | [mHC](https://arxiv.org/html/2512.24880v2) Eqs. 7–9: n = 4, gate scales 0.01, 20 Sinkhorn-Knopp iterations | Flattened 4 × 1536 stream, parameter-free RMSNorm, one 6144 × 24 projection for read, write and residual logits; `H_pre = σ`, `H_post = 2σ`, `H_res` = 20 column-then-row normalizations of `exp`, computed in log space so that a row whose entries underflow never divides by a vanishing sum. The projection starts at zero; static biases start at Hyper-Connections' initialization. Coefficients use at least FP32. Twenty iterations are approximate: rows sum to one, columns only approximately. Calibration narrowed the FFN to 6016. |
| `attnres` | [Attention Residuals](https://arxiv.org/abs/2603.15031v1), Block AttnRes, about 8 groups | Depth counted in attention and FFN sublayers; zero-initialized pseudo-queries for every sublayer and the final readout. Keys use the native parameter-free RMSNorm. 23 blocks form groups of 3,3,3,3,3,3,3,2. Sources are the embedding, completed group sums and the current partial sum. |
| `attnres_full_canon_abcd` | Full AttnRes ([Attention Residuals](https://arxiv.org/abs/2603.15031v1), one group per block) with [Canon layers](https://arxiv.org/abs/2512.17351) at all four positions (A–D); found with NextT-Gym in arena v1 | Every block is its own group, so each sublayer reads the embedding and every earlier block's output separately instead of group sums. A Canon layer is a causal depthwise convolution over the last four positions, added to its input; its weights start at zero, so the model starts as Full AttnRes. Canon-A/C act on every sublayer input after the depth read, before the sublayer's RMSNorm; Canon-B on q, k and v before rotary embedding and QK-norm; Canon-D on the MLP hidden activations before the squared ReLU. The convolution weights train with AdamW (the shell's group for non-matrix parameters). Plugin `nexttransformer/methods/attnres_full_canon_abcd`; its Canon AttnRes code is in the helper module `nexttransformer/methods/attnres_canon_abcd`. |
| `ngram_canon_attnres` | The AttnRes-Canon trunk with Canon layers at more sites and a hashed bigram memory (after the modded-nanogpt bigram hash embedding and [Engram](https://arxiv.org/abs/2601.07372)); found with NextT-Gym in arena v1 | As `attnres_full_canon_abcd`, and: each position hashes its (previous, current) token pair with two multiplicative hashes into two zero-initialized tables of 320,000 rows × 768, and one bit of each hash sets the sign of its lookup. The two lookups are concatenated to the model width, scaled by 0.1 and added to the token embedding; the first position pairs its token with a padding id. The tables sit with the value embeddings (the shell's embedding group, AdamW) and add no matrix arithmetic. New Canon layers filter every completed block output before it becomes a depth source and the embedding source after the memory is added (kernel 12), and the final depth read (kernel 8). Canon-B on the values acts after the gated value embedding is added, and the A–D sites use kernel 8 instead of 4. Every addition starts at zero, so the model starts as Full AttnRes. Plugin `nexttransformer/methods/ngram_canon_attnres`. |
| `simplehc` | [SimpleHC](https://arxiv.org/abs/2611.05984) §3.1–3.2, Eqs. 3–6: identity residual transitions (`H_res = I`), mHC full-state read/write gates, shared gate inputs over groups of `K = 4` sublayers | Four 1536-wide streams (`n = 4`); the 46 attention and FFN sublayers form 11 groups of 4 and a final group of 2. Each group normalizes its flattened `4 × 1536` entry state once with parameter-free RMSNorm and projects to `H_pre = σ` and `H_post = 2σ` (`N(0, 0.006)` weights, gate scale `0.01`, zero bias, so read starts at `0.5` and write at `1`, Appendix A.1). Within a group, earlier sublayer outputs enter later inputs through scalar corrections `d_{j,k} = h_pre_j · h_post_k` (Eq. 4) and write back once at the group boundary (Eq. 5). PyTorch algebra without the paper's fused DeepGEMM/Triton kernels; gate parameters train in the auxiliary AdamW group without weight decay. Plugin `nexttransformer/methods/simplehc`. |
| `msar` | [Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630v1): multi-stream depth attention, temporal source augmentation and adaptive stream mixing | Four streams from the raw state and zero-initialized causal depthwise convolutions of widths 4/8/12. Within each three-layer block, sublayer output deltas accumulate into a single state before temporal expansion. Per-stream softmax depth reads combine completed sources and the current partial through two-phase reads; HPre mixes the streams with one sigmoid scalar per stream. The first attention reads the embedding directly; the terminal MLP returns its input plus output. Sublayer pre-normalization is retained. Queries and HPre use auxiliary AdamW; temporal taps use nonmatrix AdamW. Plugin `nexttransformer/methods/msar`; temporal kernel attribution and license are in `_temporal_kernels.py`. |

All Residual methods share the full-MHA shell with `dense_full`; report their differences from Standard Residual
separately from overall Dense differences. Published gains come from other scales and MoE backbones and do not predict
results here.

## Short convolutions in the Dense shell

Both methods retain the full-attention Dense shell and place kernel-four causal
convolutions on projected K/V, attention output, and MLP output after the down
projection. They retain the shell's residuals, RoPE, QK normalization, value
embeddings and ReLU² MLP. Their MLP placement is inspired by
[Inkling](https://thinkingmachines.ai/news/introducing-inkling/) and
[Marin's architecture report](https://marin.community/blog/2026/11/09/marin-535b-architecture/).

| ID | Display name | Retained mechanism and adaptation |
|---|---|---|
| `sconv_out` | Static ShortConv · MLP output | Four independent per-channel filters, `x + DWConv(x)`, with zero-initialized taps and the shared non-matrix AdamW group. No Q convolution or AttnRes. |
| `sconv_dynamic_out` | Dynamic-KV ShortConv · MLP output | Independent input-conditioned K/V tap generators, shared across channels within each head, added to the static per-channel filters. Generators use the normalized attention input and shared Muon; attention-output and MLP-output convolutions stay static. Adapted from the Transformers Inkling 2 implementation; cached source hashes are recorded in the plugin README. |

Packed rows reset state; internal BOS tokens do not. Both omit the released
convolution output scaling to keep identity initialization. The dynamic method
accumulates its additional term in FP32 for BF16 input and retains the static
arithmetic path. These are fixed-budget adaptations, not complete Inkling/Marin
reproductions. Both completed one seed-42 formal run; the original 2,048-token
left-cropped CORE policy is declared in each record. Incremental KV-cache
decoding is unsupported. See the plugin READMEs for equations and the run notes
for failed attempts and interventions.

## Hybrids

Hybrids serialize their per-layer layout as `layer_mixers`, verified on load, and share one model builder, audit,
calibration grid, admission and trainer.

| ID | Source and retained layout | Local implementation and limits |
|---|---|---|
| `deltanet_hybrid` | [DeltaNet](https://arxiv.org/html/2406.06484v1) global hybrid | Full causal MHA at one-based layers 2 and N/2 + 1, DeltaNet elsewhere (layers 2 and 13 at N = 24). The native-MHA adaptation, not the paper's local-MQA variant. Trained with FLA `chunk_delta_rule`. |
| `gdn_hybrid` | Qwen3-Next 3:1 (`full_attention_interval: 4`) | GDN, GDN, GDN, full attention repeated. Qwen's gated attention, 16Q/2KV heads of 256, 25% partial RoPE, zero-centered norms and MoE are not reproduced. |
| `gdn_hybrid_swa` | [Gated DeltaNet](https://arxiv.org/html/2412.06464v1) H1: GDN + SWA with a 2K window, order from the official `lit_gpt` block | GDN at odd layers, sliding-window attention at even ones; the window holds 2,048 keys including the current one. **At the 2,048-token context the window covers every causal key, so this method is a 1:1 GDN/full-attention hybrid.** The attention layers keep QK normalization and VE, unlike the official SWA. |
| `kda_hybrid` | Kimi Linear's KDA with a registered MLA-NoPE adaptation | KDA, KDA, KDA, full causal MLA-NoPE repeated (N divisible by four). MLA-NoPE: query latent 768, joint KV latent 256, 12 heads, QK and V dimensions 128, no RoPE. All VE tables and gates removed. Kimi uses no query compression, a 512-rank KV latent, 32 heads and different KDA gating. |
| `qsa_hybrid` | The registered GDN/GDN/GDN/QSA layout | GDN with the QSA adaptation above at every fourth layer: 256-token budget, compression 4. Dense burn-in affects only QSA positions for 512 updates. |
| `moba_hybrid` | MoBA with the last three layers full attention | MoBA (block 128, budget 256 including the current block) followed by exactly three final MHA layers; at N = 25 their share is 12%. No burn-in or auxiliary loss. |

The shell's alternating VE rule places VE on every attention layer of `gdn_hybrid` and `gdn_hybrid_swa`; in `gdn_hybrid`
one GDN layer in three also has VE. VE enters GDN layers through the projected-V adaptation.

## Others

| ID | Mechanism | Notes |
|---|---|---|
| `dense` (SSSL) | 24 blocks: 18 short-window (512) and 6 global layers | The historical model and the frozen compute anchor; listed under Others because only full-attention Dense is a baseline. |
| `mqa`, `gqa` | Native attention with fewer key/value heads (`n_kv_head`), sharing RoPE and QK normalization | Trained from a frozen source revision; the mainline port is pending. |
| `mla_qknorm`, `mla_nope_qknorm` | Multi-head latent attention with QK normalization, with and without rotary dimensions | Trained from a frozen source revision; the mainline port is pending. |
| `gated_attention`, `gated_attention_headwise` | [Gated Attention](https://arxiv.org/abs/2505.06708): each head's SDPA output is multiplied by `σ(X W_θ)` before `c_proj`, with X the block's pre-normalized input; one score per channel (elementwise, the paper's best row and Qwen3-Next's layout) or per head (headwise); [official code](https://github.com/qiuzh20/gated_attention) | The Dense baseline's attention otherwise (VE, RoPE, QK norm, full attention in every layer). The official code fuses W_θ into `q_proj` and trains every weight with AdamW; here W_θ is the separate bias-free layer `output_gate`, initialized with `c_q`'s uniform rule and trained by AdamW in the auxiliary group, like the linear-attention output gates, while `c_q` stays with Muon. The elementwise gate costs 7.0% of the budget at 23 layers, so calibration reduced the depth to 21 layers and widened the FFN to 6272; the headwise gate keeps Dense's 23 layers and FFN 6144. Plugins in `nexttransformer/methods/`. |

MLA reconstructs K/V explicitly for training and uses the common attention interface; it has no reconstructed-head QK
norm or output gate. Unnormalized MLA can show logit growth under Muon; the QK-normalized rows are registered separately.

## Historical native track

SSSL, Mixture-of-Recursions, LoopFormer-Fixed2 and Circle Loop were trained in an earlier native track
(`nexttransformer/models/native/`) and are reported with the historical matrix-FLOPs estimate (4.775 GFLOPs per token for
SSSL). Their implementations, run identities and curves are preserved; entering them into the strict matched
comparison would require independent full-cost evidence. Circle Loop's design history is told in the
[arena example](../../arena/circle-loop/).

## Backend and numerical limits

Readable recurrent scans are correctness references, not FLA chunked-training kernels. KDA calls
`chunk_kda(disable_recompute=True)`; RetNet calls `chunk_simple_gla` with head log decay. CUDA causal convolutions use
FLA `causal_conv1d(activation="silu")`; CPU uses equivalent PyTorch operations. Explicit `scale=1` and
`FLA_DISABLE_BACKEND_DISPATCH=1` fix the registered path.

FLA 0.5.2's GDN backward is guarded for some GPU architectures with Triton 3.4.0–3.7.0 (FLA issue #640); the registered
environment uses Triton 3.5.1, so GDN calls the validated KDA kernel with the scalar log decay broadcast across key
channels. The output gated RMSNorm uses FLA `rms_norm_gated`; CPU and the audit use equivalent PyTorch operations.

KDA, RetNet and GDN use FP32 carry and accumulation with BF16 operands and boundary states; KDA and GDN triangular
inversion uses TF32. KDA's whole-chunk algebra can overflow FP32 for trained strong gates; use the FP64 recurrent oracle
for trained-input checks. There are no second-order derivative or optimized generation-cache guarantees. All ELU
variants use a stable FP32 feature map, keep their arithmetic in IEEE FP32 with TF32 disabled, and do not recompute.

## XSA and Inkling REL

| ID | Source and retained mechanism | Shared-shell adaptation and limits |
|---|---|---|
| `xsa` | [Exclusive Self Attention v1](https://arxiv.org/abs/2603.09078v1), Section 3, Equation 2 and Algorithm 1: per-head orthogonal projection against the current value | Self remains in causal softmax. Projection uses the effective value including gated VE; normalization/projection use FP32 for FP16/BF16. Retains shared RoPE, QK norm/gain, scaling and optimizer; no extra parameters or attention kernel. |
| `inkling_rel` | [Inkling release](https://huggingface.co/thinkingmachines/Inkling), configuration revision `828496eeae4c243ff1a22f7f28ff83694f2f7bc9`; Transformers `InklingRelativeLogits`; vLLM revision `43c1db8c9f30526fe0342c700e99d2c2648d92e4`, NVIDIA attention and REL operators | Independent content-dependent relative bias with d_rel 16. Extends the released global distance extent from 1024 to 2048 and replaces RoPE, retaining shared parameter-free QK norm/gain, scaling, VE, dense FFN and optimizer. Head-shared table per layer; original Triton kernel. Does not import Inkling MoE, convolutions, GQA, local/global pattern, learned QK norm or distinct scale. Full-sequence MHA only. |

Both method READMEs document the equations, numerical choices, cost evidence and tests. Their run notes retain original CORE scores and describe post-run BOS sensitivity; restored diagnostic scores are not replacement leaderboard results.
