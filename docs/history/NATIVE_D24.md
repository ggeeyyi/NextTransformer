# Historical native d24 experiments

> Historical record. Paths, module names, and commands are those in use at the time; the 2026-10-24
> reorganization maps them as described in [the repository guide](../guide/REPOSITORY.md).

This guide preserves the original four-model study. Its internal `dense` model is now displayed as **SSSL**. These runs use historical **matrix-only FLOPs estimates**, which must not be confused with the [current full-cost protocol](../../PROTOCOL.md). The new full-attention `dense_full` baseline is a separate planned experiment.

Pinned upstream: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; original file hashes are in [upstream_sources.json](../../configs/native/upstream_sources.json). Earlier GPT-2 and native implementations remain in Git history and frozen runtime snapshots.

## Architecture and budget

All four use width 1536, 12 Q/KV heads, head dimension 128, FFN 6144, context 2048, vocabulary 32768, seed 42, BF16, the same data/tokenizer/packing, and the native optimizer recipe.

| Display name / internal ID | Execution | Independent parameters | Historical training GFLOPs/input token |
|---|---|---:|---:|
| SSSL / `dense` | 24 independent blocks | 1,384,122,122 | 4.775226 |
| MoR / `mor` | 3 pre + 9 shared core, at most 3 loops + 3 coda | 927,994,042 | 4.774791 at target routing |
| LoopFormer-Fixed2 / `loopformer` | 3 + 9×2 + 3 | 1,018,436,792 | 4.775756 |
| Circle Sparse / `circle` | 3 pre + 9–27 core steps + 3 coda; mean target 17.5 | 927,992,523 | 4.779192 at reference routing |

Shared physical blocks also share VE and residual/x0 coefficients. Untied input/output embeddings, smear, backout, RoPE, QK normalization, RMSNorm, and ReLU² remain. Windows follow unfolded position, with the final layer long; short-window parameter 512 includes the current key. Compacted paths preserve original token positions for RoPE and causal/window distances.

The input budget is `49,152,000,000 / 1,048,576 = 46,875` updates, with 512 rows per update. Internal recurrence does not multiply input tokens. At eight ranks, microbatch × accumulation=64.

The historical cost convention estimates matrix forward ×3, excluding embedding lookup, elementwise work, optimizer, communication, and data processing. Actual route/padding estimates are logged separately and accumulated. SDPA masking may still execute full rectangles; equal estimates are not equal GPU instructions or wall time. MoR/Circle actual depth can depart from their target; do not claim strict matched compute from the design table.

## Routing and optimization

MoR uses the same per-token causal router and threshold in train/eval, with no sequence top-k or reentry. Target survival is 1/.64/.32, mean 1.96 loops or 17.64 core blocks. A selected loop updates `h+q*(F(h)-h)`. The ST budget loss uses `sum(dual*e + 0.5*e^2)` with e=hard survival−target; after each global update, `dual <- clip(dual+0.05*global_e,-10,10)`. The quadratic loss is microbatch-local, so changing partitioning can change that regularizer's gradient.

LoopFormer uses two complete traversals, no teacher/short path, and FP32 normalized time/step conditioning. Modulation is `1+0.5*tanh(raw)` in [0.5,1.5], initialized at one. Conditioning uses separate AdamW and norm-1 clipping. See [shared core](COMMON_CORE_README.md).

Circle gathers only surviving Q/K/V and MLP paths; exited tokens no longer supply later core K/V. It uses a shared halter with 18 depth biases, deterministic threshold 0.5, a 17.5-step budget, 256 random warmup and 768 transition updates. See [the full mechanism](CIRCLE_SPARSE_README.md). Incremental generation caches are not implemented for these sparse routes.

Ordinary matrices use Muon; embedding/head/VE/scalars use native AdamW groups. Base LR is embedding .3, unembedding .008, scalar .5, Muon .02, router .0006, with sqrt(global batch/524288) and native width scaling. Use the common d24 weight-decay baseline, warmup 40, final 65% linear warmdown to 5%, native Muon momentum .85→.97→.90 and cosine weight decay. No main-model global clip. Circle halter and LoopFormer conditioner each use 0.5×router LR (peak approximately .000424264) and independent norm-1 clipping. Optimizer compile-cache limit 64 covers extra AdamW shapes without changing fused math.

## Resources, launch, and evaluation

Follow [local configuration](../guide/CONFIGURATION.md) to prepare the same ClimbMix revision, tokenizer, byte table, and CORE bundle. Training excludes `shard_06542.parquet`; validation uses it alone. The tokenizer preparation uses 2B characters, at most 10K characters/document, vocabulary 32768. Best-fit packing buffers 1000 documents, tokenizing 128 at a time; complete buffers/cursors are checkpointed. File/size and validation-SHA checks do not imply all train shards are rehashed at every launch.

```bash
bash runs/train_native_a800.sh mor --site-config site.native.local.json --dry-run
bash runs/train_native_a800.sh circle --site-config site.native.local.json --preflight
# Start only after the chosen batch passes the relevant acceptance checks.
bash runs/train_native_a800.sh circle --site-config site.native.local.json --micro-batch 8
```

A800's registered starting batch is 8×accumulation8; H200's candidate is 16×4 and requires actual preflight. Preflight exercises maximum-depth routes and optimizer updates, halves batch on OOM, writes evidence, and exits. It does not automatically launch formal training.

Periodic BPB uses the native evaluator: step 0, every 250, final 46,875, 189 events ×41,943,040 tokens. CORE uses the pinned 22-task bundle, all questions, at 8K/16K/24K/32K/40K/46,875; no step-0 CORE and no duplicated final event. Independent final train/val BPB each use 20,971,520 tokens with separate names. `--skip-final-core` skips only final CORE and makes a formal result incomplete.

The upstream d24 speedrun's approximately 5.838B tokens and FP8 recipe are not this study's 49.152B/BF16 conditions. Checkpoint recovery validates scientific config, world size, resources, source, optimizer shards, all RNG, route controller, and the entire data stream. Old snapshots cannot be resumed by weakening those checks. Replaying an earlier checkpoint requires a new output directory when later checkpoints already exist.

Bounded correctness, memory, fresh-process resume, BPB, and CORE checks are recorded in [validation](NATIVE_VALIDATION.md). Short checks establish their stated scope, not 49B convergence or superiority.
