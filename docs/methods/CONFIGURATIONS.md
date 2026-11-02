# Registered configurations and parameters

Updated 2026-10-28. **`dense` is the historical SSSL configuration**, not full-attention Dense. The rebased registry contains **33 configurations**: main's 26 after GSA removal, plus H01/H04/H05/H06 and the three VE-off controls. `dense_full` is the accepted full-attention baseline ([record 038](../records/EXPERIMENTS.md#record-038)); R00 `residual_standard` and R03 `attnres` have complete formal runs ([record 050](../records/EXPERIMENTS.md#record-050)); Hybrid formal acceptance remains separate.

The current [proof set](../../evidence/costs/plugin-interface-20261031/summary.json) re-audits every structure after the method-plugin interface ([record 055](../records/EXPERIMENTS.md#record-055)); like [method-fingerprints-20261031](../../evidence/costs/method-fingerprints-20261031/summary.json), which introduced per-method source fingerprints, it keeps every count of the preceding [mhc-logsinkhorn-20261028](../../evidence/costs/mhc-logsinkhorn-20261028/summary.json) set ([record 054](../records/EXPERIMENTS.md#record-054)); SSSL's configuration is `configs/others/dense.json`. The preceding set uses `reference-aten-math-v2` and PyTorch `2.9.1+cu128`. It re-audits all 33 structures after mHC's log-space Sinkhorn-Knopp ([record 051](../records/EXPERIMENTS.md#record-051)): against the previous [hybrid-main-20261027](../../evidence/costs/hybrid-main-20261027/summary.json) set, 32 keep every stage and operator count and mHC rises by 6,072 FLOPs per input token (4.674363 GFLOPs, +0.042%). Historical proofs remain immutable; current admission requires the current source hash.

The shared controls are defined once in [PROTOCOL.md](../../PROTOCOL.md): 49.152B tokens, 1,048,576 tokens/update, 46,875 updates, seed 42, context 2048, vocabulary 32768, common data/optimizer/evaluation, and a frozen **4.67239832544785 GFLOPs/input-token** SSSL anchor. Every JSON repeats and fingerprints the protocol. Local paths and credentials are not scientific configuration.

This table reports **stage-weighted theoretical training compute**, not inference or GPU instruction counts. All proof artifacts retain `cumulative_run_audit=not_run`; final run acceptance is separate.

## Selected structures

| Method ID | Physical blocks | FFN width | Independent parameters | Training GFLOPs/input token | Deviation from SSSL anchor |
|---|---:|---:|---:|---:|---:|
| `dense` (SSSL) | 24 | 6144 | 1,384,122,122 | 4.672398 | +0.000% |
| `ut_act` | 12 | 6144 | 742,394,259 | 4.660085 | -0.264% |
| `fixed_recursive` | 15 | 6144 | 927,990,968 | 4.663054 | -0.200% |
| `huginn` | 14 | 6528 | 870,581,286 | 4.649327 | -0.494% |
| `ouro` | 9 | 6144 | 607,127,805 | 4.658210 | -0.304% |
| `deeploop` | 12 | 6144 | 742,392,722 | 4.660529 | -0.254% |
| `dsa` | 22 | 6400 | 1,316,236,406 | 4.650934 | -0.459% |
| `qsa` | 23 | 6144 | 1,378,420,488 | 4.628037 | -0.949% |
| `nsa` | 23 | 6144 | 1,381,293,832 | 4.638686 | -0.722% |
| `seer` | 23 | 6272 | 1,387,464,456 | 4.659564 | -0.275% |
| `moba` | 25 | 6144 | 1,462,765,468 | 4.673521 | +0.024% |
| `deltanet` | 25 | 6144 | 1,463,690,268 | 4.674979 | +0.055% |
| `gla` | 23 | 6144 | 1,411,207,816 | 4.642048 | -0.650% |
| `kda` | 23 | 6144 | 1,412,126,344 | 4.692443 | +0.429% |
| `retnet` | 23 | 6144 | 1,410,077,320 | 4.630060 | -0.906% |
| `elu_linear` | 25 | 6144 | 1,462,765,468 | 4.651241 | -0.453% |
| `elu_norm` | 25 | 6144 | 1,462,768,668 | 4.651242 | -0.453% |
| `elu_qknorm` | 25 | 6144 | 1,462,768,668 | 4.652243 | -0.431% |
| `gdn` | 23 | 6144 | 1,411,349,680 | 4.687197 | +0.317% |
| `dense_full` | 23 | 6144 | 1,355,810,568 | 4.674662 | +0.048% |
| `residual_standard` | 23 | 6144 | 1,355,810,568 | 4.674298 | +0.041% |
| `hc` | 23 | 6144 | 1,356,235,700 | 4.698483 | +0.558% |
| `mhc` | 23 | 6016 | 1,353,550,818 | 4.674357 | +0.042% |
| `attnres` | 23 | 6144 | 1,355,882,760 | 4.679857 | +0.160% |
| `deltanet_hybrid` | 24 | 6400 | 1,403,810,314 | 4.675326 | +0.063% |
| `gdn_hybrid` | 24 | 5760 | 1,399,275,962 | 4.703800 | +0.672% |
| `gdn_hybrid_swa` | 22 | 6528 | 1,329,681,662 | 4.647087 | -0.542% |
| `kda_hybrid` | 24 | 6016 | 793,542,986 | 4.692309 | +0.426% |
| `kda_no_ve` | 23 | 6144 | 808,144,840 | 4.692295 | +0.426% |
| `mla_nope` | 26 | 6144 | 744,751,182 | 4.684529 | +0.260% |
| `mha_no_ve` | 23 | 6144 | 751,829,064 | 4.674513 | +0.045% |
| `qsa_hybrid` | 24 | 5760 | 1,405,174,202 | 4.691637 | +0.412% |
| `moba_hybrid` | 25 | 6016 | 1,452,935,068 | 4.659989 | -0.266% |

Physical blocks count distinct weights. For recurrent models use `ModelConfig.n_layer`, not the unrelated ordinary-stack `layers` field. Shared calls still consume compute. Calibration does not maximize parameter count and never reads val/CORE scores.

## Looped execution

| Method | Selected execution | Training semantics |
|---|---|---|
| UT-ACT | 3 pre + 6 core × at most 3 loops + 3 coda | All 24 block candidates plus coordinates, gates, mixing, and ponder cost are charged |
| Fixed Recursive | 3 + 9 × 2 + 3 | 15 physical blocks, 24 calls; full BPTT, LM loss only |
| Huginn | 3 + 8 × sampled 1–4 loops + 3; evaluation uses 3 loops | Last at most 2 loops backpropagate; earlier forwards, injection, and sandwich norms are charged |
| Ouro | 3 pre + 3 core × at most 3 loops; each loop has its own 3 coda calls and head readout | 45,851 joint and 1,024 gate-only updates; joint graph has 21 block calls and 3 head calls |
| DeepLoop | 12 shared blocks × 2; no separate pre/coda | Post-norm and alpha/beta initialization; native residual/x0 parameters frozen; VE values also beta-scaled |

Huginn's `mean=3, sigma=0.5` are distribution parameters, not its clipped expected depth. Routing seed 1729 over steps 0–46,874 yields **9,276 / 12,599 / 10,239 / 14,761** updates at depths 1/2/3/4, mean **2.6503467** loops. Costs use this exact frozen sequence. Row-keyed state noise is restorable and independent of microbatch splitting.

ACT epsilon/ponder are 0.01/0.01. Ouro uses exit-distribution-weighted CE minus 0.1×entropy during joint training and detached next-depth CE improvement during gate-only training; evaluation uses cumulative exit probability 0.5. All candidates remain charged.

## Sparse stages and selections

DSA/QSA/Seer use **512 dense-burn-in + 46,363 sparse updates**, including the internal teacher. Indexer input and teacher are detached; per-layer index loss weight 1 is summed across layers and updates only the indexer. QSA's detach policy is a disclosed local adaptation.

| Method | Selection policy |
|---|---|
| DSA | Top 256 tokens; indexer 4×128, partial RoPE 64; native MHA adaptation |
| QSA | Four-token compressed blocks; up to 64 complete selected blocks plus at most 3 unfinished tail tokens; no mandatory tail at a complete-block boundary |
| NSA | Nonoverlapping compression/selection blocks of 32; per-head three-branch gates; historical selection budget 256 plus current tail, local-512 and compression branches |
| SeerAttention-style | Blocks of 64; historical budget 256 plus tail; not a complete pooled-query AttnGate reproduction |
| MoBA | Blocks of 128; total budget 256 includes the mandatory current block, leaving at most one historical block |

## Linear dimensions and precision

K/V dimension is 128; GLA/KDA gate rank is 16; DeltaNet/KDA/GDN use causal convolution width 4. GDN uses one scalar forget gate per head, with d→12 projection and 12-element A_log/dt_bias initialized from the registered FLA distributions.

DeltaNet/GLA use the readable reference scan. KDA/RetNet/GDN use `fla_chunk64`: FP32 carry/accumulation, BF16 operands and boundary states; KDA/GDN triangular inversion uses TF32. The three ELU variants retain the backend name but use IEEE FP32 torch chunks. ELU features, division/output normalization stay FP32 until the output projection; `elu_norm`/`elu_qknorm` use epsilon 0.0078125 and BF16 gains. `elu_qknorm` adds pre-feature FP32 q/k RMSNorm × **tau=8**, with its pilot-derived origin disclosed.

GDN fuses QK L2 normalization and gated output RMSNorm in CUDA; its framework epsilon is the FP32 value (approximately 1.19e-7). See [SOURCES.md](SOURCES.md) for the backend rationale and exact adaptations.

## Full-attention Dense and Residual rows

All five rows use full causal MHA in every layer (`window_pattern=L`, window `(2048, 0)`), 12 Q / 12 KV heads, head dimension 128, VE, smear, and the common ReLU² FFN. Each was calibrated from 24 layers / FFN 6144; the selected structures also satisfy the v2 tie-break (FFN closest to 6144, then smallest error, then depth closest to 24).

| Method | Residual path | Extra parameters | Notes |
|---|---|---:|---|
| `dense_full` | Native learned residual/x0 coefficients and backout | 0 | B01 baseline; only depth changed |
| `residual_standard` | `h + F(norm(h))` for every attention and FFN sublayer | 0 | R00 control; the 47 native residual/x0/backout scalars stay frozen and unused but are counted |
| `hc` | Dynamic Hyper-Connections, 4 streams, one map per sublayer (46 maps) | 425,132 | Embedding replicated into streams; streams summed before the final norm |
| `mhc` | mHC, 4 streams, 20 Sinkhorn-Knopp iterations, gate scale 0.01 (46 maps) | 6,784,218 | FFN 6016: FFN 6144 was +1.21% and 22 layers −2.91% |
| `attnres` | Block AttnRes, 8 groups of 3,3,3,3,3,3,3,2 blocks, 47 zero-initialized pseudo-queries | 72,192 | Final readout attends over the embedding and all group sums |

Residual maps and pseudo-queries use the registered auxiliary AdamW group (base LR 0.0006 with batch scaling, betas 0.9/0.95, no weight decay) and never enter Muon. Frozen native scalars have no optimizer owner. On A800 with microbatch 8 × accumulation 8, trainer smoke peaked at 45.6 GiB (`dense_full`), 44.6 GiB (R00), 49.0 GiB (AttnRes), 66.2 GiB (mHC), and 72.6 GiB (HC) per GPU; HC at microbatch 4 × accumulation 16 peaked at 44.5 GiB and is the recommended formal setting ([record 028](../records/EXPERIMENTS.md#record-028)).

## DeltaNet hybrid (H01)

`deltanet_hybrid` has 22 DeltaNet layers and two full causal MHA layers at one-based positions **2 and 13**. Layer 24 remains DeltaNet. It uses the common hybrid layout, trainer, operator counters and admission. Its ungated DeltaNet CUDA path calls FLA `chunk_delta_rule`; the pure A01 configuration retains its reference scan. The registered even-depth/FFN grid selects 24 / 6400 with 1,403,810,314 parameters and **4.675325779958443 GFLOPs/input token**, +0.0626542% versus SSSL. The single `joint` stage totals **229,801,612,736,517,375,000 FLOPs** over 46,875 updates. See the [source and adaptation notes](SOURCES.md#deltanet-hybrid-v2-h01) and [record 044](../records/EXPERIMENTS.md#record-044) for evidence and pending deployment checks.

## GDN hybrids

Both rows mix A07's GDN mixer (`fla_chunk64`) with `dense_full`'s attention (12 Q / 12 KV heads of 128, QK norm, RoPE, VE where the shell places it); the per-layer layout is serialized as `layer_mixers` and verified on load. `gdn_hybrid` (H02) repeats GDN, GDN, GDN, full attention, so 24 layers hold 18 GDN and 6 attention layers (one-based 4, 8, …, 24). `gdn_hybrid_swa` (H03) alternates GDN and sliding-window attention, GDN first, so 22 layers hold 11 of each; the window is 2048 keys including the current token, `(2047, 0)`, which at the 2048-token context equals full causal attention.

Selection follows the v2 tie-break over the locked repeating unit: every allowed depth in 16–32 (multiples of 4 for H02, even for H03) is audited at FFN 6144, every FFN on the 128 grid is estimated with the exact FFN slope, and candidates are fully audited in tie-break order; each proof publishes the whole table. At FFN 6144, 24 layers cost +4.33% (H02) and +4.26% (H03), 20 layers −11.98% and 22 layers −3.89%. H02 takes 24 / 5760 (+0.672%) over 24 / 5632 (−0.546%) because 5760 is closer to 6144; for H03, 22 / 6528 (−0.542%) and 24 / 5760 (+0.602%) are equally far from 6144 and the smaller error decides.

## KDA / MLA-NoPE hybrid and controls

H04 repeats `[KDA, KDA, KDA, MLA-NoPE]` six times: MLA at one-based 4/8/12/16/20/24.
All four new rows remove every VE table and gate, while keeping the native FFN, smear, backout and residual/x0.
MLA uses Q/KV ranks 768/256, 12 heads and content/V dimensions 128/128 with no RoPE or per-head QK norm;
KDA uses A04's existing FLA backend. See the [adaptations](SOURCES.md#kda--mla-nope-hybrid-v2-h04-and-ve-off-controls).

The public V2 grid independently selects each row: depths 16–32 (multiples of four for H04), FFN 4608–7680
in steps of 128, preferring FFN closest to 6144, then absolute cost error, distance from depth 24, then shallower.
H04 at 24/6144 exceeds 1%; 24/6016 is the first admitted candidate. Its direct `[512,2048]` mixed graph,
including backward and one optimizer update, totals **230,636,373,707,947,921,875 FLOPs** over 46,875 steps.
No pure-row proof is averaged to obtain that cost. Controls also have independent proofs.

## Evidence boundaries

The 2026-10-22 merged implementation has **180 CPU tests, 21 GPU tests, 20 full-size [1,2048] BF16 architecture checks, and 12 bounded eight-rank updates**. The synchronized updates use small model dimensions and synthetic data; they are not full formal-model training results. See [record 026](../records/EXPERIMENTS.md#record-026).

The 2026-10-24 reorganization moved the code into the `nexttransformer` package and re-audited all 27 configurations: model hashes, parameters, stage weights, and every operator count are unchanged; only source hashes and proof paths differ. The current 34-configuration re-audit and bounded evidence are indexed in [evidence/README.md](../../evidence/README.md); historical records retain their original scope.

Reference scans can require roughly 44–49 GiB for one full-size row. They are correctness references, not optimized training/decode implementations. No optimized generation state/KV-cache claim is made. Compare changed backends on outputs, gradients, state boundaries, costs, memory, and throughput before using them. Historical native matrix estimates must not be merged into this table as if they used the same formula.

## QSA and MoBA hybrids (H05/H06)

H05 repeats `[GDN, GDN, GDN, QSA]` six times. Only the six QSA positions use dense burn-in
for updates 0–511; GDN stays unchanged. The shared QSA indexer and detached teacher operate
in both training stages; auxiliary losses are summed across QSA layers. The 512 burn-in updates
cost 4.777911770 GFLOPs/input token, followed by 46,363 sparse updates at 4.690684175.
The stage-weighted average is 4.691636932 (+0.411750%), totaling 230,603,338,504,308,997,890 FLOPs.

H06 has 22 MoBA layers followed by three full causal MHA layers (23/24/25), so full attention
occupies 3/25 = 12% of the calibrated stack. It reuses the
registered 128-token blocks and 256-token budget including the current block, with no burn-in
or auxiliary loss. Its single joint stage totals 229,047,779,618,553,093,750 FLOPs.

Both retain the native VE placement, FFN, smear, backout and residual/x0 shell. H05 uses GDN's
existing FLA backend; both support the existing `--sparse-backend kernel` and public trainer,
with the reference sparse path retained for variable-length CORE inputs. Calibration audits
the complete mixed graphs at logical `[512,2048]`, including backward, teacher/indexer and
optimizer costs. H05 searches depths 16/20/24/28/32; H06 searches every depth 16–32, always
regenerating the last-three layout. Both use the registered FFN grid and tie-break. H06 at 25 / 6144 is just outside the
1% admission threshold (+1.003352%), so the selected FFN is 6016.
See [record 046](../records/EXPERIMENTS.md#record-046) for bounded evidence and remaining acceptance.
