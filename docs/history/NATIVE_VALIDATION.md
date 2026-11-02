# Historical native acceptance: 2026-10-13

> Historical record. Paths, module names, and commands are those in use at the time; the 2026-10-24
> reorganization maps them as described in [the repository guide](../guide/REPOSITORY.md).

This page records the **3+9+3** native implementation and its [compact evidence](../../evidence/checks/native-v3-evidence.json). It is historical bounded validation, not current reference-cost admission or a completed long-run comparison. The former “Dense” row is now displayed as **SSSL**. Earlier GPT-2, four-loop MoR/LoopFormer, and full-candidate Circle-ST results remain in Git history.

## Source and CPU checks

Final delivered source fingerprint: `20dfff8d26b7ba0060e8a60801a0d51ecbd20c58326d89647392f76ca319e7e6`. GPU validation used the separately recorded fingerprint; afterward, five files received docstring/unused-import edits only and the complete CPU suite was rerun.

**116 passed, 14 skipped in 17.50 s**. Skips were ten upstream FlashAttention3 comparisons and four CUDA optimizer cases; actual eight-GPU training covered GPU optimizer updates. Coverage included upstream logits/loss/gradient/update equivalence, causality, batch independence, routing determinism, sparse-versus-masked Circle gradients, empty/padded routing, budget gradients, LM-to-halter gradients, packing/BPB, exact RNG/data/checkpoint recovery, local configuration/tracking, and LoopFormer bounds/initialization/AdamW/clip.

All eight native A800/H200 launcher dry-runs passed: global batch 1,048,576; A800 8×8, H200 16×4. The offline lock check and pinned upstream hashes passed; dependencies were unchanged.

## Eight-A800 deepest-path preflight

Full width, context 2048, real data, BF16, microbatch 8, accumulation 8. All tokens were forced through the deepest path. Two warmups and two timed optimizer updates included cross-rank synchronization; one evaluation microbatch was also checked.

| Model | Max allocated GiB | Max reserved GiB | Tokens/s | Forced path |
|---|---:|---:|---:|---|
| MoR | 68.705 | 70.605 | 110,966 | Three core loops, 33 blocks |
| LoopFormer-Fixed2 | 58.213 | 59.268 | 156,639 | Two core loops, 24 blocks |
| Circle Sparse | 75.878 | 77.672 | 101,733 | 27 ring blocks, 33 total blocks |

Timing excludes compilation, evaluation, CORE, and checkpoints. MoR/Circle numbers are deepest-path stress tests, not normal target-depth throughput. Two measurements do not establish long-term variance. Reserved memory includes allocator cache; neither memory measure represents SM utilization. H200 was not tested.

## Training, fresh-process resume, and evaluation

Each run checked eight optimizer shards and the model file, token cursor, finite loss/gradients, train/val BPB over 131,072 tokens each, and all 22 CORE interfaces with at most 64 examples per task. These short-run scores validate the pipeline and must not rank model quality. The arrow indicates fresh-process checkpoint recovery.

| Model | Update before → after resume | Train BPB | Val BPB | CORE interfaces |
|---|---:|---:|---:|---|
| SSSL | 2→3 | 3.091523 | 3.098088 | Passed |
| Circle | 2→3 | 3.296688 | 3.268422 | Passed |
| MoR | 64→65 | 1.485939 | 1.500802 | Passed |
| LoopFormer | 256→257 | 1.026517 | 1.065655 | Passed |

Circle used a smoke-only curriculum offset here. MoR update 65 had survival 65.03%/31.71% and mean 1.9674 loops, near the 64%/32% and 1.96 targets. Actual routing/padding produced a matrix-only estimate of **4.9088 GFLOPs/input token**; the ideal design-table gap below 0.1% does not replace this measured gap.

LoopFormer's last 32 updates averaged loss 3.2876; maximum gradient norm was 5.268 and largest adjacent post-warmup loss increase 0.0676. Modulation offsets stayed within 0.5. Saturation peaked at 91.85% at update 128 and fell to 0.135% at update 257. This did not reach the earlier failure near update 2710; long-run stability remained unproven.

## Circle algorithm pilot

Before the shared native cleanup, the same new Circle algorithm completed 128 updates and resumed to 129: **135,266,304 input tokens**. BPB and small-sample CORE paths passed. Its separate source hash is preserved in the JSON.

The pilot compressed the curriculum to 8 random warmup + 24 transition updates; the formal schedule remained 256+768. From update 96, logged mean ring depth averaged 17.773 and cap fraction 20.0%; per-update mean depth ranged 13.957–19.590. Maximum gradient norm was 1.853; final matrix-only cost was 4.825 GFLOPs/input token. Budget oscillation remained visible; this was neither a quality result nor evidence of a uniform depth distribution.

## Schedule and ModelArts follow-up

The same day, all four formal configurations adopted `core_every=8000`, `core_max_per_task=-1`, and the six-event schedule ending at 46875. Architecture, token budget, global batch, and BPB cadence stayed fixed.

The expanded CPU suite passed **130 tests, 14 skipped in 23.99 s**. Real small-model updates tested divisible/nondivisible final steps, skipping final CORE, and smoke suppression, with expensive evaluation replaced only to record calls; these mocks are not evaluation results.

ModelArts dry-runs covered nodes 0–3, four independent eight-rank groups, injected outer 32-rank variables, output isolation, accumulation 8, 46,875 steps, and six CORE events. Invalid/missing node settings and nested torchrun were rejected. Actual four-node training was not submitted by this acceptance task.

## Reproduction boundary

Use the frozen historical source for these checkpoints; current model paths and source hashes have changed. The [cleanup record](CLEANUP.md) lists removed historical implementations. Frozen training source, original data, tokenizer, and old checkpoints were retained. Compact evidence replaced temporary test checkpoints and duplicate logs; no new 49.152B run was started for that cleanup. Current protocol and model status are in [PROTOCOL.md](../../PROTOCOL.md) and [PLAN.md](../../PLAN.md).
