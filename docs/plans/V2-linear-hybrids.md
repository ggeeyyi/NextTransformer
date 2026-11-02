# v2 linear hybrids: source check and implementation plan

This is the **source-review and acceptance reference**, originally drafted on 2026-10-24 and consolidated
on 2026-10-29. Current status, owners, evidence and remaining work are maintained only in
[PLAN.md](../../PLAN.md#hybrid-family-and-its-controls). Registered source adaptations are maintained in
[SOURCES.md](../methods/SOURCES.md); selected structures and audited costs are maintained in
[CONFIGURATIONS.md](../methods/CONFIGURATIONS.md).

It covers H01 `deltanet_hybrid`, H02 `gdn_hybrid`, H03 `gdn_hybrid_swa`, H04 `kda_hybrid`, H05
`qsa_hybrid`, and H04's independently calibrated VE-off controls. H06 `moba_hybrid` is sparse-only:
its layout is defined in [v2 §3](V2.md#3-hybrids-with-registered-source-evidence), while it shares the
infrastructure and applicable acceptance requirements here. The [protocol](../../PROTOCOL.md) and
[v2 design constraints](V2.md) apply unchanged. This document does not authorize a new method identity,
changed layout or formal training run. Original H01 frozen-run sources remain separate from the shared
mainline implementation, as recorded in [SOURCES.md](../methods/SOURCES.md#deltanet-hybrid-v2-h01).

## 1. Summary

| Row | Preregistered layout | Distinct acceptance concern |
|---|---|---|
| H02 `gdn_hybrid` | GDN, GDN, GDN, MHA repeated; N divisible by 4 | Record and validate the native attention backend on the selected hardware |
| H03 `gdn_hybrid_swa` | GDN, SWA alternating; window 2048 including the current token; even N | At 2K this equals a 1:1 GDN/full-MHA hybrid, not a sparse-locality experiment (§2.3) |
| H01 `deltanet_hybrid` | Full MHA at one-based layers 2 and N/2+1; DeltaNet elsewhere; even N | Chunked DeltaNet and pure A01's scan implement the same recurrence but have separate audited training algorithms |
| H05 `qsa_hybrid` | GDN, GDN, GDN, QSA repeated; N divisible by 4 | Per-position burn-in, indexer/teacher, kernel equivalence and stage-weighted costs |
| H04 `kda_hybrid` | KDA, KDA, KDA, MLA-NoPE repeated; N divisible by 4; no VE | O03-N differs from Kimi's release (§2.4); verify independent VE-off controls and long-run stability |

Freeze ratios and positions before calibration. The detailed layout definitions in [v2 §3](V2.md#3-hybrids-with-registered-source-evidence)
and the source analysis below remain scientific constraints; they are not a separate progress checklist.

## 2. Source verification

Each subsection compares the preregistered row with the primary sources read on 2026-10-24. "Verbatim" means read directly from the paper's TeX, HTML, or official code. Values marked *summary* came from a truncated page via a model-extracted summary and must be re-read before they are cited in a record.

### 2.1 H01 `deltanet_hybrid` — confirmed

- **Layout.** DeltaNet v6 (camera ready), §3.4 "Hybrid Models", verbatim from the TeX source: *"We follow [Fu et al.] to replace only two layers with global attention: the second layer and the (N/2+1)-th layer, where N is total number of layers."* This matches the preregistration: one-based layers 2 and N/2+1, i.e. 0-based {1, N/2}, with N even. The cited H3 paper writes its own hybrid as *"layer 2+N/2"*, one layer later; we follow DeltaNet's text, not H3's.
- **The other variant** (not registered): DeltaNet layers interleaved with sliding-window attention every other layer, following Griffin/Samba. v1 calls these "sliding multi-query local attention" layers (*summary*). The paper does not state the window size, head count, or RoPE of either attention variant.
- **DeltaNet layer.** Recurrence `S_t = S_{t-1} − β_t(S_{t-1}k_t − v_t)k_tᵀ` with `β_t = σ(W_β x_t)` (verbatim §2.2); SiLU feature map, L2-normalized q/k, kernel-4 short convolutions after the q/k/v projections, normalization before the output projection, head dimension 128 (§3.3, Appendix A.1, *summary*). This repo's `deltanet` mixer has the same elements and no output gate, as in the paper.
- **Reported results** (verbatim, table `lm_harness`, 1.3B / 100B SlimPajama tokens): DeltaNet Wiki ppl 16.87, average accuracy 51.6, recall tasks SWDE/SQuAD/FDA 49.5/37.4/17.2. With 2 global attention layers: 16.55, 51.8, 71.0/43.0/29.8. Transformer++: 16.85, 50.9, 66.6/31.5/27.4. The gain is concentrated in the recall tasks.

<a id="22-h02-gdn_hybrid--confirmed-source-citation-to-be-replaced"></a>

### 2.2 H02 `gdn_hybrid` — confirmed

- The Qwen3-Next blog page (`qwen.ai/blog?id=qwen3-next`) renders client-side and returned no readable text, so the preregistered link cannot serve as evidence. The HF model card for `Qwen/Qwen3-Next-80B-A3B-Instruct` states *"Hybrid Layout: 12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))"*, and `config.json` has `"full_attention_interval": 4` with 48 layers: attention is the 4th layer of each group of four, as preregistered.
- Not reproduced, by registration: Qwen's gated attention (sigmoid output gate on every head channel), 16 query / 2 KV heads with head dimension 256, partial RoPE on 25% of dimensions (64 dims), zero-centered RMSNorm with weight decay, and MoE. H02 keeps this repo's GDN (A07) and B01's native MHA. List each difference in the adaptation card.
- External evidence for the ratio (QSA paper §2.1.1, Table 1; 25B-A3B MoE, 28 layers, 400B tokens at 4K + 80B at 32K): average 49.87 for full attention, 51.15 for an SWA hybrid, and 53.81 for the GDN hybrid, all with one attention layer in four. No other ratio was ablated there.

### 2.3 H03 `gdn_hybrid_swa` — confirmed, but at 2K context it is a 1:1 global hybrid

- The paper text (Gated DeltaNet v1 §3.3; §3.4 in v3, the ICLR camera-ready) says only that H1 combines Gated DeltaNet with SWA, with *"a sliding window size of 2K"*. **Order and ratio come from the official code**, not the paper. In NVlabs `lit_gpt/model.py`, `Block` uses Gated DeltaNet when `layer_idx % gated_delta_per_layer == 0`, and both H1 configs set `gated_delta_per_layer=2`. So GDN sits at one-based odd positions and SWA at even ones, and the last layer is SWA for even N (configs `GatedDeltaNet_H1_0.4B`: 12 layers, `GatedDeltaNet_H1_1.3B`: 18 layers).
- **Window.** The official code passes `window_size=(local_window-1, 0)` with `local_window=2048` to flash-attn: 2047 previous keys plus the current one. Its SDPA fallback silently drops the window. Our `nanochat.flash_attention` applies the window on both FA3 and SDPA, so that gap does not carry over. Register the window as left=2047.
- **Consequence.** Training, validation, and CORE inputs are at most 2,048 tokens, so a 2,048-key window covers every causal key. **Under this protocol H03 computes exactly the same function as a 1:1 GDN/full-MHA hybrid**, and its attention cost equals full causal attention. V2 forbids shrinking the window to manufacture sparsity, so H03 stays as registered. Report it as "SWA-2048 (equivalent to global at 2K)". Its contrast with H02 is the ratio (1:1 vs 3:1) and positions, not locality.
- The official SWA layers are plain MHA with full RoPE, no QK norm, and no output gate. Ours follow B01: QK norm and VE, which is a native adaptation. H2 (Mamba2 + GDN + SWA) is not in the public code and is not planned.

### 2.4 H04 `kda_hybrid` — registered O03-N adaptation

- **Layout.** Kimi Linear's released config (`moonshotai/Kimi-Linear-48B-A3B-Instruct`, `configuration_kimi.py` uses one-based `kda_layers`): 27 layers, `full_attn_layers` [4, 8, 12, 16, 20, 24, 27]. Layers 1–24 are groups of [KDA, KDA, KDA, MLA]; the 27-layer release ends with a truncated [KDA, KDA, MLA]. The paper states *"a uniform 3:1 ratio"*. With N divisible by 4, our last layer is MLA, as in the release.
- **MLA in the release** (verbatim config and code): no Q compression (`q_lora_rank: null`, asserted), `kv_lora_rank` 512, `qk_nope_head_dim` 128 plus `qk_rope_head_dim` 64, `v_head_dim` 128, 32 heads, `mla_use_nope: true` (asserted), one latent RMSNorm (`kv_a_layernorm`), no output gate. **NoPE there means the 64 "rope" channels are never rotated, but they are kept**: a shared, MQA-style, unrotated key part, giving a 192-dimensional QK head.
- **Our O03-N `mla_nope`** (V2 §5): Q rank 768, KV rank 256, QK content 128, RoPE 0, V 128, 12 heads. It differs from the release in Q compression, KV rank, and the absent 64 shared dimensions (head count and width differ by scale anyway). See decision D1.
- **KDA.** The release initializes `A_log = log U(1,16)`, uses low-rank (rank = head dim) decay and output gates, a sigmoid-gated output RMSNorm, and per-head β. This repo's A04 KDA uses constant 0/−3 gate initialization, `gate_rank` 16, and a full-rank SiLU output gate. H04 reuses the repo's KDA (as registered for A04) and discloses this.
- Reported ablations (*summary*, verify before citing): hybrid ratio at 16 layers, same FLOPs — validation ppl 5.65 (3:1), 5.66 (1:1), 5.70 (7:1), 5.82 (15:1), 5.77 (0:1). NoPE vs RoPE at 1.4T tokens: RULER@128K 84.3 vs 78.8.

### 2.5 H05 `qsa_hybrid` — confirmed; sparsity comes from our S02 adaptation

- QSA paper v1 (31 Aug 2026) §2.1.1, verbatim: *"We place one such full-attention layer in every four layers, with the other three layers using GDN."* §2.1.2 replaces all of those full-attention layers (backbone and MTP) with QSA during continued pretraining. The layout matches the preregistration.
- **Paper QSA:** MQA indexer with 4 query heads and 1 key head, partial RoPE on 64 of 128 dimensions, keys mean-pooled into blocks of r = 4, a token budget K = 2048 (512 blocks) plus the unfinished tail, and ReLU block scores. Training: an indexer-only dense distillation stage (1,000 steps, about 2B tokens) followed by joint sparse training (8,000 steps, about 200B tokens), both at 256K context during CPT.
- **At our 2K context, the paper's K = 2048 would select every block.** H05 therefore reuses S02's registered adaptation: token budget `sparse_topk` = 256 (64 blocks of 4 plus the tail), 4 index heads, index dimension 128, RoPE 64, and a 512-update dense burn-in with the indexer teacher. V2 registers the burn-in at QSA positions only. "GR" in the paper is Gated Residual, a residual-stream design (§2.2), and is correctly excluded with n-grams, MoE, and MTP.

<a id="3-proposed-preregistration-amendments-approval-required-before-calibration"></a>

## 3. Preregistered source clarifications

The original amendment heading is retained as a compatibility anchor. These clarifications explain the
registered identities; maintained adaptation cards are in [SOURCES.md](../methods/SOURCES.md).
A different scientific choice still requires a separately approved preregistration before calibration.

1. **H03 source.** GDN §3.3 (v1) / §3.4 (v3) establishes GDN + SWA with a 2K window; NVlabs
   `gated_delta_per_layer=2` establishes GDN at odd positions and SWA at even ones. Preserve left window 2047
   and disclose equivalence to a global 1:1 hybrid at this context length.
2. **H02 source.** Use the model card and `config.json` (`full_attention_interval: 4`) as readable primary
   evidence for the layout; QSA §2.1.1 supplies related evidence. The client-rendered blog alone is insufficient.
3. **H01 source.** Preserve DeltaNet v6 §3.4's layer positions; H3's different `2+N/2` text is not substituted.
4. **H04 MLA.** Retain the O03-N choice recorded in D1 and disclose all differences from Kimi's release.
5. **H05 sparsity.** K = 256 and r = 4 come from S02: the paper's K = 2048 would be dense at 2K. Its mature-model
   continued-pretraining recipe is not reproduced.

## 4. Shared implementation

Changes to fingerprinted scientific source require a fresh source-aware proof set for every registered
configuration. Preserve each frozen run's source, environment and proof; use a reviewed current main as the
base for new implementation work. Current integration evidence is linked from [PLAN.md](../../PLAN.md#hybrid-family-and-its-controls).

- **Configuration.** Preserve existing method order and identities. Serialize an explicit per-layer mixer list
  (`layer_mixers`) and per-layer arguments such as windows. Generate it from one registered method/depth layout
  function, validate it on load, scope hybrid-only fields to hybrids, and check existing configuration hashes.
- **Model.** Select each layer's linear, sparse or native causal-attention implementation from its registered
  mixer kind; apply the per-layer window. Shared MLA lives in `nexttransformer/models/others.py`. Trace every
  layer's mixer, effective window, VE placement and backend.
- **Value Embeddings.** H01/H02/H03/H05 retain native VE placement and the existing projected-V adaptation in
  linear layers. H04 and its controls remove VE entirely, consistent with v2 §5.
- **Optimizer groups.** Assign every parameter exactly once, including MLA latent norms and projections;
  preserve existing rows' auxiliary/matrix assignments.
- **Trainer.** Use `nexttransformer.train` with the registered 8/16/32-rank layouts, FLA and `--sparse-backend`.
  Configurations with learned-sparse layers use `has_sparse_burnin` to select burn-in/joint stage costs. CORE
  keeps the reference sparse path because the kernel path requires T % 128 == 0.
- **Cost audit.** Trace the actual mixed graph and weight every stage; never average pure-row proofs. H01's
  chunked DeltaNet algorithm is defined in §5. Verify that the QSA kernel computes the audited reference
  function at the registered hybrid positions.
- **Attention backend.** Stage the exact FA3 kernel when required, or record and validate the intended SDPA
  path. Offline settings must not silently change the accepted backend. Check windows and causality on the
  path actually used by the run.
- **Tests.** Verify layout/hash stability, per-layer causality and state reset, loss ownership, FP64 mixer
  oracles, complete distributed updates, bitwise resume on the intended registered rank layout, and trace
  fields for global positions, effective windows, mixer parameter/FLOP shares, state and KV memory.

## 5. Per-row work

**H02 `gdn_hybrid`.** Layout plus existing GDN (`fla_chunk64`) and B01 MHA. Check the H20 attention backend, full-size `[1|8, 2048]` forward/backward and causality, then a single-GPU update with throughput and memory. This row proves the shared infrastructure.

**H03 `gdn_hybrid_swa`.** Same as H02 plus a per-layer left window of 2047 (2048 keys including the current token). Test bitwise equality with full causal attention for T ≤ 2048, and a real difference at T > 2048 on a debug model, so the window is shown to be applied rather than assumed.

**H01 `deltanet_hybrid`.** DeltaNet is registered in the chunked backends. The audited algorithm is the WY/UT delta rule without decay: a new `chunk_reference` kind, not GDN with a zero gate, which would charge useless decay work. The CUDA path retains FLA `chunk_delta_rule` from the original H01 implementation; it is checked against the FP64 oracle through the common FLA checker. Shared admission and the complete mixed-graph audit replace the earlier H01-specific validation and cost adapters. Disclose that the pure A01 DeltaNet configuration retains the scan backend: both compute the same function, but their audited training algorithms differ.

**H05 `qsa_hybrid`.** Apply the shared trainer requirements in §4. Run the dense burn-in and indexer teacher only at QSA positions for the first 512 updates; GDN layers stay GDN throughout. Sum auxiliary losses over QSA positions with S02's policy. Price `burnin` and `joint` as separate stages. Check the QSA kernel against the reference function on H20, and report actual selected edges and density per QSA layer.

**H04 `kda_hybrid`.** Reuses O03-N `mla_nope` (V2 §5): latent compression/norms/up-projections, content-only QK and V at 128, and the shared attention backend. Require the VE-off shell and independently calibrated `kda_no_ve`, `mla_nope`, and `mha_no_ve` controls, with explicit reconstruction and gradient oracles. Pilot and formal runs remain separate; every formal run starts from scratch.

## 6. Calibration expectations (not claims)

Search N in 16–32 under the locked layout: H02/H04/H05 use N in {16, 20, 24, 28, 32}; H01/H03 use even N.
Adjust FFN in [4608, 7680] in steps of 128 under the protocol tie-break: closest to 6144, then smallest
absolute cost error, then depth closest to 24, then shallower. A pure mixer's selected depth or cost does not
predict the hybrid's admitted structure. Use the complete audited mixed graph to select a candidate.

[CONFIGURATIONS.md](../methods/CONFIGURATIONS.md) maintains the selected rows and their proofs. H01's
chunked mixed graph has its own proof; pure A01's scan-accounted proof cannot be substituted for it. H06
uses the v2 grid with its last-three layout regenerated at every candidate depth.

## 7. Validation and admission per row

Follow the path used for A07: config registration → ≤1% preflight and admission → CPU regression → H20 single-GPU full-size checks and backend numerics against FP64 for every mixer kind → eight-rank bounded check (L1 bitwise resume) → preregistered 512-update pilot (P1–P5) → multi-node B1–B4 if run on 16/32 ranks → formal 49.152B run **only with explicit authorization** → cumulative audit. Pilots are accounted separately and never seed formal runs.

## 8. Paired comparisons this enables

| Comparison | Isolates |
|---|---|
| H02 vs A07 GDN | Replacing one GDN layer in four with full attention |
| H05 vs H02 | QSA vs full attention at identical positions and ratio |
| H05 vs S02 QSA | GDN vs QSA in three of four layers |
| H03 vs H02 | 1:1 vs 3:1 GDN/global ratio (H03 is global at 2K) |
| H01 vs A01 DeltaNet | Two global layers in a DeltaNet stack (backend difference disclosed) |
| H04 vs `kda_no_ve`, `mla_nope` | The hybrid against each of its components, all without VE |
| Every row vs B01 `dense_full` | Overall difference from the accepted full-attention Dense baseline |

## 9. Resources (rough)

The original 2026-10-24 estimate assumed five linear-based hybrids plus two H04 component controls:
seven 49.152B-token runs. Using completed H20 linear runs at 614–804 training GPU-hours each (RetNet 613.6,
KDA v2 734.5, ELU-QKNorm 803.8), it estimated roughly 4,500–6,000 H20 GPU-hours, plus pilots, bounded
checks and evaluation (3–6 wall-hours per run).

This is a historical sizing assumption, not a current run-count commitment or hardware allocation. It does
not budget every shared control or H06. Each approved row needs its own owner, hardware, measured
throughput/memory and run manifest; track these in [PLAN.md](../../PLAN.md#hybrid-family-and-its-controls).
Measure attention/backend costs on the target hardware before scheduling; fixed theoretical FLOPs do not
imply fixed GPU-hours.

## 10. Decisions for the owner

The scientific choices below preserve the registered design. Current ownership, allocations and optional
extensions are tracked in [PLAN.md](../../PLAN.md#hybrid-family-and-its-controls).

- **D1 — H04 MLA definition.** Retain O03-N for H04 and its independent `mla_nope` control, derived from
  `xxy/others-attention` commit `8cba0a3`. Preserve the Kimi differences and known Muon/MLA stability risk in
  SOURCES. A Kimi-faithful or QK-normalized variant requires its own approved identity and control.
- **D2 — H01 CUDA kernel.** Retain FLA `chunk_delta_rule`, with fresh oracle checks for source changes;
  do not introduce GDN decay or an output gate.
- **D3 — H03.** Keep its registered 2048-key window, measuring a 1:1 global ratio at 2K. Any smaller-window
  adaptation is separately named, approved and calibrated; it does not replace H03.
- **D4 — Ownership and hardware.** Record an owner, source, hardware allocation and acceptance evidence for
  each experiment in the consolidated plan and its run manifest. Historical integration ownership or bounded
  checks on one hardware class do not allocate a future formal run.
- **D5 — Trainer.** Use `nexttransformer.train` for H05/H06, including sparse kernel selection, FLA,
  checkpointing and registered 8/16/32-rank data layouts. Four-rank smoke evidence remains separate from
  formal-layout acceptance.
