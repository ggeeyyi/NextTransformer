# Experiment protocol

This is the scientific contract for the reference comparison and all v2 methods and paired controls. Machine-readable values live in [`nexttransformer.protocol.spec.Protocol`](nexttransformer/protocol/spec.py), with fingerprints in each [configuration](configs). The protocol remains `nextt-reference-50b-v1`; the accounting formula remains `reference-aten-math-v2`. Reorganizing documentation does not revise either contract.

## Baseline identity

- **SSSL** is the historical `method=dense` model: 24 blocks, 18 short-window and 6 global layers. Preserve its configuration ID, checkpoint keys, run IDs, and historical scores. The display layer maps it to `sssl`.
- **Dense** is the `dense_full` model: full causal MHA in every layer, with its own configuration, proof, run, and checkpoint. Its formal run was accepted as the Dense baseline on 2026-10-26 ([record 038](docs/records/EXPERIMENTS.md#record-038)).
- SSSL defines the **frozen compute anchor**. Full Dense must fit that same budget. Its greater attention cost does not authorize a larger budget.
- Main-table Dense deltas use only an accepted full-Dense run: complete training and evaluation, and cumulative cost acceptance. Do not substitute SSSL or a control row.

## Fixed inputs and evaluation

| Control | Required value / behavior |
|---|---|
| Upstream source | nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; exact project commit and source fingerprints per run |
| Data | ClimbMix revision `915333b4f8b8684f39aeaafea600fea6f43fb703`, identical files and content |
| Split | Train excludes `shard_06542.parquet`; validation reads only that shard |
| Data order | Same sorted Parquet files, row groups, document stream, global row IDs, and eight logical streams |
| Tokenizer | Same prepared 32,768-vocabulary tokenizer and byte table; joint SHA256 |
| Context | 2,048 tokens; identical shifted targets, label mask, padding, and BOS handling |
| Packing | Native BOS best-fit; 1,000-document buffer; encode 128 documents at a time |
| Boundaries | Attention may cross BOS inside a packed row; linear state resets at the row boundary, not each BOS |
| Input budget | **49,152,000,000 tokens**, including warmup, burn-in, gate training, and all stages |
| Global batch | **1,048,576 tokens = 512 × 2,048-token rows per optimizer update** |
| Updates | **46,875**; gradient-accumulation microsteps are not optimizer steps |
| Randomness | **Single seed 42**; method-specific routing RNG is documented, independent, and restorable |
| Objective | Same next-token objective; auxiliary losses separately named and costed |
| Precision | BF16 training with the common master-parameter/reduction policy; method-required accumulation exceptions documented; no per-method FP8 switch |
| Periodic validation | Step 0, every 250 steps, and final 46,875: **189 events**, **41,943,040 tokens** each |
| CORE | Steps **8,000 / 16,000 / 24,000 / 32,000 / 40,000 / 46,875**: **6 events**, all **22 tasks**, all questions (`core_max_per_task=-1`); no step-0 CORE |
| Checkpoints | Every 5,000 updates and final; report the **46,875** checkpoint, not the best retrospectively selected checkpoint |
| Independent final BPB | Train and val each **20,971,520 tokens**; name separately from periodic validation |
| Logging axis | Optimizer step in local logs and W&B |

Verify data manifests, tokenizer/byte-table hashes, validation-shard content, evaluation resources, source, and config. Existing startup checks cover filenames/sizes and the validation SHA256; complete training-shard content checks remain an explicit verification task. A directory name alone is not evidence of identical data.

A repeated internal loop or teacher pass over the same update consumes no additional *input* tokens, but every computation is charged. Reading input again for an additional optimizer update consumes tokens again. Pilots cannot supply pretrained weights to a run labelled “from scratch.”

## Shared model shell and allowed variables

Default external width is 1536, query heads 12, head dimension 128, vocabulary 32768, and FFN width 6144. Preserve untied input/output embeddings, ReLU² FFN, the native RMSNorm/QK-normalization/RoPE policy where applicable, Value Embeddings (VE), smear, backout, and learned residual/x0 coefficients unless a registered mechanism requires an explicit adaptation.

Layer count, shared-core size, loop distribution, sparse selection, state/latent dimensions, required gates/convolutions/normalization, and registered FFN width are architecture variables. Freeze method semantics before cost calibration. Document every necessary shell change and provide paired controls; see [v2](docs/plans/V2.md).

SSSL's causal short-window tuple `(512, 0)` includes the current token, so its maximum visible span is **513 tokens**. Looped models schedule windows by unfolded execution position. Pure sparse/linear mixers replace attention in every layer; a leftover `window_pattern=SSSL` field does not mean those mixers use SSSL. Hybrids must serialize the actual per-layer layout.

Independent parameters, inference cost, memory, throughput, and GPU-hours are reported outcomes. Do not assume a sparse model must have more parameters or be faster at a 2K context.

## Optimizer, schedule, and distributed data

Use Muon for ordinary block matrices and the existing AdamW groups for embeddings, output head, VE, and native scalars. Explicitly register auxiliary gates/indexers/halters/residual maps and pseudo-queries; each trainable parameter must appear exactly once.

Base learning rates: Muon **0.02**, embedding **0.3**, unembedding **0.008**, scalar **0.5**, auxiliary AdamW **0.0006**. Apply the existing square-root global-batch scaling and native width scaling; these are base values, not necessarily final group rates. Warmup is **40** updates; the final **65%** of training linearly warms down to **5%** of peak LR. Weight decay is **0.28** with the common d24 scaling and native Muon schedules. No default global gradient clipping; preregister method-specific auxiliary LR/clip exceptions before pilots.

| World size | Fixed logical streams | Ranks per stream | Per-rank microbatch × accumulation |
|---:|---:|---:|---:|
| 8 | 8 | 1 | 64 |
| 16 | 8 | 2 | 32 |
| 32 | 8 | 4 | 16 |

Adjacent ranks sharing a stream pack all its rows and retain their contiguous share of each update. The global row set, ordering, IDs, and update-boundary stream state must match eight-rank execution. Validation uses the same sharing; CORE distributes question IDs without changing the question set. Unregistered world sizes are smoke-only. `nexttransformer.train` provides the registered 8-, 16-, and 32-rank layouts. Do not confuse four independent eight-GPU jobs with one 32-GPU job.

A800, H200, and H20 may change microbatch and accumulation after preflight without changing global tokens. H200 settings are candidates until tested. Record hardware, kernels, BF16 and reduction-order differences. Microbatch-dependent nonlinear auxiliary losses require tests and disclosure; averaging squared microbatch means is not the square of a global mean.

## Training compute and calibration

```text
N_input = 49,152,000,000
C_ref   = 4,672,398,325.44785 FLOPs / input token
        = 4.67239832544785 GFLOPs / input token
F_ref   = C_ref × N_input ≈ 2.2965772249241273 × 10^20 FLOPs

F_train = sum(forward + backward + auxiliary + explicit_recompute + optimizer)
C_train = F_train / N_input
relative_error = C_train / C_ref - 1
```

The anchor is the audited SSSL row in the [frozen merge proof](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/merged-20261022/dense.json). The historical matrix-only estimate **4.775225616 GFLOPs/token** is a different accounting scheme.

Use FMA=2 and one published operator-coverage policy. Include projections, FFN, head, normalization, activations, convolutions, state operations, gates, indexers/teachers, softmax/Sinkhorn, actual backward, and optimizer updates. Unknown expensive floating operators block admission. Integer sorting/selection, communication, and data loading have separately reported time/bytes. Evaluation is separate from training compute.

- A candidate computed fully and masked afterward is still charged fully. Shared weights do not make repeated calls free. No-gradient forwards and multi-exit heads are charged.
- Do not replace method-specific backward accounting with `6ND`, parameter count, inference exit depth, or a universal `3 × forward`.
- Weight stages by their actual input tokens. DSA/QSA/Seer use 512 dense-burn-in updates and 46,363 sparse updates; Ouro uses 45,851 joint and 1,024 gate-only updates.
- Keep algorithmic compute separate from kernel padding, tiles, and redundant execution. A dense masked kernel does not demonstrate sparse training savings. Necessary algorithmic recomputation is charged.
- Activation checkpointing is currently off. Changing it, model semantics, source, software, or kernels requires a new audit. Preserve prior proofs and frozen runs.

Admission thresholds are **≤1% preflight**, **≤2% final cumulative error**, and a recorded anomaly when rolling deviation exceeds **3%**. Freeze these thresholds before runs; do not relax them for one method.

Admission binds formal runs (`nexttransformer.train --formal`). Research mode, the trainer's default, records stale or out-of-budget cost evidence instead of refusing it; its runs are never results.

For new v2 stacks, start at 24 layers / FFN 6144 and search layers **16–32**, subject to the registered layout. If needed, adjust FFN within **[3d, 5d]** in multiples of **128**. Among candidates within 1%, prefer FFN closest to 6144, then smallest absolute cost error, then depth closest to 24, then shallower depth. Publish a full candidate table if monotonic assumptions fail. Existing configurations retain their recorded selection provenance.

No padding with useless computation, fewer input tokens, early stopping, seed shopping, or formal CORE-based selection. If the grid fails, record `calibration_pending` and preregister a revised grid. Report parameters by embedding/VE, mixer, FFN, router/residual components; physical blocks and unfolded calls; forward/backward/auxiliary/optimizer costs and stage weights.

## Validation, pilots, and resume

Before formal admission, require:

1. Fingerprints and manifests for source, model, protocol, cost formula, data, tokenizer, and evaluation resources.
2. Small numerical oracles, forward/backward checks, causality, boundaries, and method-specific semantic checks.
3. Full-size BF16 checks and a complete global optimizer update on the intended backend; verify parameter grouping and distributed synchronization.
4. Identical data rows and scheduled val/CORE events across supported batch layouts.
5. Resume checks covering model, optimizer, scheduler, all RNG, stream buffers/cursors, stage, budget controller, and cumulative compute. Do not reset counters or replay a data prefix as continuous training.

Pilots allow at most **two preregistered candidates × 512 updates per method**, the same prefix and seed (up to approximately 1.074B input tokens). Publish pilot cost separately. Use them for stability, mechanism, cost, and resources; never tune on formal CORE. Record all attempts, including initialization/LR/clip fixes, and reinitialize the formal run.

## Result admission and release

Publish final validation BPB and CORE separately, all 22 CORE task scores and six event points, token count, average/cumulative theoretical compute and deviation, parameters, layout, VE/residual policies, controls, hardware, throughput, memory, GPU-hours, and fingerprints. Missing measurements are null, not zero.

Distinguish `planned`, `implementation_pending`, `calibration_pending`, `preflight_passed`, `running`, `matched`, `budget_mismatch`, `protocol_mismatch`, `incomplete_eval`, and `failed`. Only a complete accepted run is `matched`. Preserve historical SSSL results, failed runs, and prior formulas without rewriting their scores or fingerprints.
