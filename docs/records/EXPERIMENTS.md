# Implementation and experiment history

> Historical record. Paths, module names, and commands are those in use at the time; the 2026-10-24
> reorganization maps them as described in [the repository guide](../guide/REPOSITORY.md).

This is the maintained English account of records 001–027. The [complete original pre-cleanup log](https://github.com/Selen-Suyue/NextTransformer/blob/867d0930fec102ddb6f8ac74fce39fe8a3f34a42/reference/EXPERIMENTS.md) preserves the original wording, deployment details, preregistrations, and timestamps. Existing JSON evidence is unchanged. The summaries below retain failures, corrections, and deviations rather than turning historical progress into a completion claim.

Records 029–032 were first written on branch `linear-fla-h20` as records 026–029 and were renumbered when that branch merged main; the [original branch log](https://github.com/Selen-Suyue/NextTransformer/blob/2134daf2d1cfd936c9104adb3a39f032937b9436/reference/EXPERIMENTS.md) keeps their original wording, job names, and timestamps. JSON evidence, code history, and commit messages from that branch keep the original numbers: branch 026 → 029, 027 → 030, 028 → 031, 029 → 032. Records 033–034 (the GDN hybrids H02/H03) are kept on branch `linear-fla-h20` until that work is merged; records 035–037 follow them here.

Records 059–061 (arena v1) are kept on branch `gy/arena-v1` until that work is merged. They follow main's record 058 (`58a0854`), merged into the branch on 2026-11-02; if main adds records before the branch merges, these move after them.

“Dense” in records 001–026 means historical **SSSL**, API `dense`. Full-attention `dense_full` is separate; its formal run is [record 038](#record-038). Historical running states are observations at the stated date, not live status. Current requirements are in [PROTOCOL.md](../../PROTOCOL.md); current implementation status and new work are in [PLAN.md](../../PLAN.md) and [v2](../plans/V2.md).

## Record navigation

This index adds navigation only. Record bodies and their historical status, identities, failures and
limitations are preserved. Use [PLAN.md](../../PLAN.md) for current status.

| Topic | Starting records |
|---|---|
| Framework, accounting and initial checks | [001](#record-001), [007](#record-007), [013](#record-013) |
| Distributed recovery and linear formal results | [029](#record-029), [031](#record-031), [035](#record-035), [037](#record-037) |
| Dense baseline and frozen linear publication | [038](#record-038), [039](#record-039) |
| Others results and runtime scoring | [041](#record-041), [042](#record-042), [043](#record-043), [049](#record-049) |
| Hybrid implementation, bounded checks and H02/H03 result submissions | [044](#record-044), [045](#record-045), [046](#record-046), [052](#record-052), [053](#record-053) |
| Residual publication, interruption, source transition and family comparison | [050](#record-050), [051](#record-051), [057](#record-057), [058](#record-058) |
| Research mode, per-method cost fingerprints, SSSL in Others and method plugins | [054](#record-054), [055](#record-055) |
| Arena v1 research with NextT-Gym on Koala H20 | [059](#record-059), [060](#record-060), [061](#record-061) |

All records present in this file: [001](#record-001), [002](#record-002), [003](#record-003), [004](#record-004), [005](#record-005), [006](#record-006), [007](#record-007), [008](#record-008), [009](#record-009), [010](#record-010), [011](#record-011), [012](#record-012), [013](#record-013), [014](#record-014), [015](#record-015), [016](#record-016), [017](#record-017), [018](#record-018), [019](#record-019), [020](#record-020), [021](#record-021), [022](#record-022), [023](#record-023), [024](#record-024), [025](#record-025), [026](#record-026), [027](#record-027), [028](#record-028), [029](#record-029), [030](#record-030), [031](#record-031), [032](#record-032), [033](#record-033), [034](#record-034), [035](#record-035), [036](#record-036), [037](#record-037), [038](#record-038), [039](#record-039), [040](#record-040), [041](#record-041), [042](#record-042), [043](#record-043), [044](#record-044), [045](#record-045), [046](#record-046), [047](#record-047), [048](#record-048), [049](#record-049), [050](#record-050), [051](#record-051), [052](#record-052), [053](#record-053), [054](#record-054), [055](#record-055), [056](#record-056), [057](#record-057), [058](#record-058), [059](#record-059), [060](#record-060), [061](#record-061).

## Record 001

**Initial verification, 2026-10-18.** Started from `69a253e` with eight idle A800 GPUs. Added reference methods separately from the original four native experiments. Verified source, resource, and configuration boundaries before bounded checks; no formal run was started.

## Record 002

**Mathematical references and CPU checks.** Twenty-nine checks passed on width-48, two-head, short-context models. Independent FP64 references checked outputs and gradients. Small-model checks established implementation behavior, not full-size memory or training quality.

## Record 003

**Eight-A800 BF16 checks.** The initial 17 methods passed independent small-model and target-width `[1,64]` forward/backward and causality checks. GPUs divided the methods; this was not distributed training or a global optimizer update.

## Record 004

**2K context, KDA memory, and QSA correction.** Full-context KDA autograd exceeded memory; custom recurrent backward reduced peak allocation to about 48.85 GiB and was checked for all input gradients. QSA teacher pooling, tail blocks, and KL details were corrected against the source method. Failures and fixes preceded final validation.

## Record 005

**Eight-rank updates.** All 17 methods completed real synchronized updates with width 48, vocabulary 64, context 2048, and 1,048,576 global tokens (`micro 32 × accumulation 2 × 8 ranks`). Routing noise was keyed to global row identity. CPU regression reached 98 checks; CPU compilation was disabled after a missing-setuptools environment failure.

## Record 006

**Real data and evaluation interfaces.** Verified resources, row packing, exact data recovery, 65,536-token BPB, and CORE-22 interfaces on representative small models with the real 32K vocabulary. CORE used at most 32 examples/task; an initial one-example limit failed the ten-shot sampling requirement and was corrected. These were integration scores, not formal quality comparisons. [Evidence](../../evidence/checks/data-eval-integration.json).

## Record 007

**Complete cost calibration.** FakeTensor traced actual forward/backward/optimizer operations at the full batch. Fixed a slicing-induced quadratic scan accounting issue by unbinding once and corrected optimizer grouping for auxiliary gates. All 17 configurations met the 1% threshold; the largest absolute deviation was RetNet at 0.986794%. SSSL established the frozen **4.67239832544785 GFLOPs/input-token** anchor. Unknown floating operators block admission.

## Record 008

**Final CPU and full-size checks.** 113 CPU checks passed. All 17 formal-size models passed BF16 `[1,2048]` forward/backward and had zero causal-prefix error. Scan models used roughly 44–49 GiB. These checks excluded optimizer state and updates. [CPU](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/cpu-regression.json), [architectures](../../evidence/checks/a800-full-context.json).

## Record 009

**Synchronized optimizer and special stages.** All 17 small models passed eight-rank global updates; Ouro gate-only and DSA/QSA/Seer burn-in were checked separately. Full-size SSSL completed one eight-rank synthetic update with peak allocation 43.54 GiB. The other 16 global-update tests used small dimensions. [Joint updates](../../evidence/checks/ddp-global-update.json), [gate-only](../../evidence/checks/ddp-gate-only.json), [burn-in](../../evidence/checks/ddp-burnin.json), [full SSSL](../../evidence/checks/ddp-dense-full.json).

## Record 010

**DeepLoop initialization review.** Rechecking the source identified omitted initialization scaling for V and the FFN input. Corrected V/O, both FFN matrices, and VE using the registered beta. Costs were unchanged, but source-aware proofs, CPU tests, full-size checks, and 17-method DDP checks were regenerated. Earlier unaffected data/CORE and special-stage evidence retained its original hashes.

## Record 011

**Chunked linear training backend.** Introduced `fla_chunk64` and math-v2 accounting without changing selected architecture shapes. Chunk lengths 1/63/65/2048 were checked against FP64 recurrent output/gradient references; FP32 compute and BF16 boundaries were explicit. Direct full-context operator counts and affine cross-checks were reconciled. [CPU evidence](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/cpu-regression-fla.json).

## Record 012

**H20 resources and single-GPU checks.** Verified resources and the 6,543-shard dataset metadata, FLA numerical behavior, and full-size KDA/RetNet/ELU checks. Selected microbatch 8 × accumulation 8 for subsequent eight-rank checks. Single-rank timing and memory do not establish distributed throughput or long-run stability. [Numerics](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/h20-fla-numerics.json), [full context](../../evidence/checks/h20-full-context.json), [preflight](../../evidence/checks/h20-microbatch-preflight.json).

## Record 013

**Eight-H20 bounded acceptance.** Formal-size models completed 12 smoke updates, full global batch, evaluation wiring, and bitwise fresh-process resume. CPU checks: 218 passed, 14 skipped. Median update times were approximately 8.49/5.85/6.14 seconds for KDA/RetNet/ELU in this setup. [Evidence](../../evidence/checks/h20-ddp-bounded-check.json).

## Record 014

**512-update pilots and trained-input numerics.** Each method consumed 536,870,912 pilot tokens with finite loss/gradients. CORE was used for timing only. Checked real checkpoint-layer inputs against FP64, including 4×/16× KDA gate stress. Pure Torch chunk KDA overflowed on strong gates; the adopted FLA path passed. Pilot scores and weights were excluded from formal results. [Pilot](../../evidence/checks/h20-pilot.json), [trained-input checks](../../evidence/checks/h20-fla-trained-numerics.json).

## Record 015

**Formal launch, 2026-10-19.** KDA, RetNet, and ELU formal jobs were launched on eight H20 GPUs each with frozen source, local JSON records, and offline tracking mirrored separately. This records launch, not completion. Local training artifacts remained authoritative when mirror state lagged.

## Record 016

**KDA implementation optimization and fresh restart.** Adopted fused convolution, disabled explicit recomputation, and changed microbatch/accumulation to 16×4 after numerical and bounded checks. Update time improved from about 8.456 to 7.041 seconds. The old run was preserved; the new formal run started from scratch. [Evidence](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/h20-kda-speed-ab.json).

## Record 017

**ELU numerical failure and IEEE FP32 attention.** The first ELU formal attempt stopped at update 5370 as invalid numerics. Trained-input checks exposed q/k-gradient cancellation in BF16 attention. IEEE FP32 attention passed bounded numerical/resume checks and a fresh attempt began; the damaged run was retained and its weights were not repaired or reused. [Diagnosis](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/h20-elu-numerics.json).

## Record 018

**Second ELU failure and `elu_norm`.** A second attempt diverged near update 1660. Registered `elu_norm`, removing the denominator and adding output RMS normalization; CPU checks reached 233 passed / 16 skipped. A historical 2000→4000-update stability pilot followed. Early claims excluding BF16 as a cause were superseded by record 019. This longer historical pilot is not relabelled as compliant with the current v2 two-by-512 pilot allowance.

## Record 019

**ELU feature precision correction, 2026-10-20.** BF16 `ELU(x)+1` lost the negative tail through cancellation. Replaced it with a stable piecewise FP32 feature map and checked the entire path from raw q/k projections through the LM VJP; previous post-feature-only oracles missed this failure. CPU: 246 passed / 16 skipped. ELU cost fell by 116,400 FLOPs/token. An explicit epsilon of 0.0078125 was introduced for the ELU output norm; the claim that it preserved native epsilon was later corrected in record 025. Full-run stability was not established. [Precision evidence](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/h20-elu-precision-fix.json).

## Record 020

**Cost of the ELU precision fix, 2026-10-20.** Single-H20, one-rank-share timing with optimizer: old 8×8 was 5.642 s / 48.83 GiB, corrected 8×8 was 7.259 s / 82.51 GiB, and old 16×4 was 5.540 s / 82.27 GiB. The same-shape fix was 28.65% slower. This excluded DDP, evaluation, and data-read time. [Evidence](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/h20-elu-precision-speed.json).

## Record 021

**`elu_norm` collapse, 2026-10-20.** Stopped the formal attempt at update 7110; the last checkpoint was 7000. Frozen-checkpoint diagnosis on two validation rows linked q/k scale growth to increasingly rank-one attention and cancellation. This supports a mechanism hypothesis, not an attribution of every training failure. [Evidence](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/elu-norm-collapse-diagnosis.json).

## Record 022

**`elu_qknorm` registration and pilot, 2026-10-20–21.** Added per-head FP32 RMS normalization before the feature map and fixed temperature 8. The registered shape was 25 layers, 1,462,768,668 parameters, and 4.652242983 GFLOPs/token. IEEE FP32 attention passed output ≤1% / gradient ≤3% gates, including diluted/rank-one and trained-input cases; BF16/TF32 alternatives failed relevant stress cases. [Checks](../../evidence/checks/h20-elu-qknorm-checks.json).

The historical 4000-update pilot passed P1–P4: finite loss/gradients, bounded loss jump and gradient norm, and improving validation (3.1689 / 0.8723 / 0.8458 at 0/2000/4000). **P5 was missing as preregistered:** update-4000 train/val numerical checks passed, but the required update-2000 checkpoint had already been pruned. The driver stopped without starting formal training.

The pruning bug used a negative slice when retention exceeded the existing checkpoint count. Clamping the deletion count at zero fixed it; retention 0/2/3/8/20 was covered, with 275 CPU passes / 16 skips. Default retention 2 was unaffected.

Post-hoc checks at 2500/3000/3500 passed (maximum output error 0.168%, q/k/v gradients ≤0.191%, gain gradient ≤0.205%). The project owner explicitly accepted those as a substitute for update 2000 and authorized formal launch. **This is a documented post-hoc deviation; the original P5 check was never performed.** These historical long-pilot conditions remain distinct from current v2 pilot limits.

## Record 023

**Sixteen-H20 data layout and launch, 2026-10-21.** Replaced rank-dependent stream partitioning with eight fixed logical streams, each shared by two adjacent ranks. Training rows, IDs, update-boundary stream state, and validation rows matched eight-rank execution. Actual CPU 8/16-rank tests and H20 bounded checks validated the layout.

B1–B4 passed: bitwise same-layout resume, equal stream states, step-0 validation agreement within 1e-5, and first-update loss agreement within 1e-5. Later update differences were reported, not treated as exact cross-layout equality. Sixteen-rank median update time was 3.81 s versus 7.43 s on eight, about 97.5% scaling efficiency. [Evidence](../../evidence/checks/h20-two-node-checks.json).

Formal `elu_qknorm` launched at 2026-10-21 09:09Z on 16 H20 GPUs, microbatch 8 × accumulation 4, after the explicit P5 substitution above. Quota-driven 36.5-hour segments preserved the schedule/checkpoint state. This is a dated launch observation, not proof of final completion.

## Record 024

**GDN registration, 2026-10-22.** Added scalar decay, method-specific initialization and optimizer groups: 23 layers, 1,411,349,680 parameters, 4.687197103 GFLOPs/token (+0.317% versus SSSL). On Hopper/Triton 3.5.1, FLA's dedicated GDN kernel was rejected by guard #640; the adopted backend uses the mathematically equivalent broadcast scalar gate through `chunk_kda`, rather than disabling the production guard. CPU: 291 passed / 18 skipped; targeted GPU: 74 passed. Full-size `[1|8,2048]` and single-rank update checks passed. [Evidence](../../evidence/checks/h20-gdn-checks.json).

## Record 025

**GDN kernel review and deployment checks, 2026-10-22.** Fused q/k and output RMS paths were checked against native and FP64 references. An erroneous BF16 epsilon assumption caused up to 96% mismatch; using FP32 compute-type epsilon (about 1.19e-7) restored native semantics. GDN uses native epsilon; ELU variants retain their separately registered 0.0078125 epsilon. Correcting the ELU description does not alter its running model. CPU: 292 passed / 20 skipped; GPU: 77 passed. [Kernel evidence](../../evidence/checks/h20-gdn-kernel.json).

Eight-/sixteen-H20 bounded acceptance and the 512-update sixteen-rank pilot passed all registered conditions. Pilot val BPB reached 0.9167; all 23 layers passed trained-input FP64 checks (max output error 0.39%, gradient error 0.96%). Sixteen-rank updates took about 3.33 s, with 96.4% scaling efficiency. [Deployment evidence](../../evidence/checks/h20-gdn-deployment.json).

The sixteen-rank formal job submitted at 2026-10-22 13:22Z stopped after step-0 validation when the owner selected 32 GPUs. **It performed no training update and wrote no checkpoint.** A 24-rank layout was rejected because 512 global rows and 64 rows/stream do not divide evenly. The registered 32-rank layout uses four ranks/stream and microbatch 16 × accumulation 1.

Four-node B1–B4 (same-layout exact resume, matching eight-stream state, step-0 val and first loss within 1e-5) were preregistered. The first four-node check passed B2–B4 but failed B1, so no formal run started; records 029 and 031 diagnose and fix the cause, and record 032 records the 32-rank launch. Do not infer four-node acceptance from the earlier sixteen-rank pilot.

## Record 026

**Looped/sparse merge, 2026-10-22.** Integrated main `1972eb3` with collaborator branch `xxy/h800-looped-sparse` at `26f4ad1`, retaining the branch and both histories. Kept the collaborator's Huginn SandwichBlock, DeepLoop initialization, Ouro post-coda gate, indexer detach policies, NSA per-head gates, FlexAttention/fused teacher, and burn-in state; retained FLA/GDN and multi-node stream/resume support. The shared native-style reference trainer now explicitly rejects non-eight-rank formal execution.

All 20 selected configurations were re-audited and admitted under PyTorch 2.9.1+cu128 and math-v2. SSSL retained 4.67239832544785 GFLOPs/token; maximum absolute deviation was QSA at 0.949432%. Seer retained FFN 6272. Fifteen costs matched prior main exactly; Huginn/DSA/QSA/NSA/Seer changes are in the [cost comparison](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/merge-20261022/cost-changes.json). [New proof set](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/merged-20261022/summary.json).

Ninety-five source configuration/proof files were preserved and byte-checked against their original commits in [historical snapshots](https://github.com/Selen-Suyue/NextTransformer/tree/evidence-archive-20261024/reference/results/merge-20261022/history). Existing checkpoints, frozen packages, run directories, and historical average costs were not rewritten. Changed NSA shapes and method semantics require the original code for old runs.

Validation: **180 CPU passes / 20 CUDA-only skips; 21 GPU passes; 20 full-size BF16 `[1,2048]` checks with zero causal-prefix error; 12 eight-rank small-model global updates** (eight joint, three burn-in, one gate-only). Independent full-size checks are not distributed full-size training. A missing-setuptools failure was resolved with a task-local dependency, leaving the shared environment intact. [Summary](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/merge-20261022/validation.json).

No formal run or repeated pilot was launched; cumulative audits remained `not_run`. Full-attention Dense remained v2 work with separate identity and evidence.

## Record 027

**Public-facing repository organization, 2026-10-23.** Consolidated implementations into `models/backbone.py`, `models/native/`, and `models/reference/`; retained legacy module aliases and tensor names. Updated active imports, package contents, source hashing, and original upstream-resource path resolution without changing pinned bytes or mathematical implementations. Changed the inaccurate ELU epsilon comment to match records 019/025; numerical behavior is unchanged.

Reworked README around the research framework, translated maintained guides, separated the scientific protocol from the status index and v2 plan, and retained necessary regression suites. The machine-specific website smoke helper is now ignored; curated proofs and historical evidence remain versioned.

Re-audited all 20 configurations against the relocated implementation. Selected architectures, parameter counts, theoretical totals, and per-operator counts match the preceding merge proofs. New proofs reside in [layout-20261023](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/layout-20261023/summary.json); old proofs are unchanged. Validation passed 327 CPU tests (34 skipped), 22 targeted GPU tests, and 17 website tests. All 20 configurations also passed admission from the built wheel, and the README model example ran successfully. Import/resource compatibility, source invalidation inside isolated checkouts, and packaging are recorded in the [validation summary](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/layout-20261023/validation.json).

No model quality result, new formal run, repository visibility change, or frozen-run migration was part of this cleanup.

## Record 028

**Full-attention Dense and Residual family, 2026-10-23 (owner @selen).** Registered `dense_full` and Residual rows R00 `residual_standard`, R01 `hc`, R02 `mhc`, and R03 `attnres`. All use full causal MHA in every layer, VE, and the native shell; the Residual rows replace the native residual/x0 coefficients and backout, which stay frozen and unused. New scoped configuration fields (`residual_streams`, `residual_gate_init`, `sinkhorn_iterations`, `attnres_groups`) are serialized and fingerprinted only for the methods that read them, so every pre-v2 configuration keeps its file contents and hash. Implementation follows Hyper-Connections Algorithm 2, mHC Eqs. 7–9, and Block AttnRes Fig. 2; adaptations are listed in [sources](../methods/SOURCES.md#full-attention-dense-and-residual-family).

Calibration started every row at 24 layers / FFN 6144 and kept the legacy search: `dense_full` 23 / 6144 (+0.048%), R00 23 / 6144 (+0.041%), HC 23 / 6144 (+0.558%), mHC 23 / 6016 (+0.042%), AttnRes 23 / 6144 (+0.160%). For mHC, FFN 6144 gave +1.21% at 23 layers and −2.91% at 22. Using the audited per-layer and FFN slopes, 23 / 6016 is the only within-1% candidate 128 from 6144, and the other four rows have a unique within-1% depth at FFN 6144, so the selections also satisfy the v2 tie-break. All 25 configurations passed admission; the 20 pre-v2 proofs match [layout-20261023](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/layout-20261023/summary.json) in model hashes, parameters, stage weights, and every operator count. [Proof set](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/v2-selen-20261023/summary.json).

Validation: **363 CPU passes / 34 CUDA-only skips**, including paper-oracle, FP64 operator/gradcheck, initialization, frozen-scalar, optimizer-ownership, and launcher tests; **22 GPU passes**; five full-size BF16 `[1,2048]` checks with zero causal-prefix error; five eight-rank small-model and five full-size eight-rank synchronized global updates (1,048,576 tokens, microbatch 8 × accumulation 8). Trainer smoke ran through `runs/train_reference_modelarts.py` on real ClimbMix rows: two updates, resume to four, and an uninterrupted four-update run per method; resumed losses matched within 1e-5. Steady eight-A800 updates took 4.49 s (`dense_full`), 4.29 s (R00), 9.24 s (HC), 8.95 s (mHC), and 5.76 s (AttnRes); peak allocation was 45.6, 44.6, 72.6, 66.2, and 49.0 GiB. HC at microbatch 4 × accumulation 16 took 9.70 s with 44.5 GiB and is the recommended formal HC setting. [Validation summary](../../evidence/checks/v2-selen-20261023/validation.json).

The first calibration pass forced coefficients to FP32 and was discarded before commit when FP64 checks required "at least FP32" precision; arithmetic counts were unchanged. No pilot or formal run was launched, and cumulative audits remain `not_run`. The ModelArts launcher runs one admitted configuration per node, so the four Residual rows are four independent eight-GPU experiments, not one 32-rank run.


## Record 029

**Four-node resume diagnosis, part 1: DDP bucket warm-up, 2026-10-22.** The first four-node GDN bounded check (bundle `49cc41a8`) passed B2–B4 but failed B1, so under the preregistered rule no formal run started; the owner chose to diagnose and fix first. A resumed segment matched the continuous run bit for bit at updates 1–2 (loss, gradient norm), then differed by 3.8e-8 relative at update 3, growing to 8e-6 by update 11. Stream states, step-0 validation, and update-1 loss matched eight-rank execution exactly. Thirty-two ranks took 1.78 s/update, 90.1% scaling efficiency versus eight. [Evidence](../../evidence/checks/h20-four-node-determinism.json).

NCCL 2.27.5 logs showed four-node DDP bucket all-reduces mostly on RING (LL128/SIMPLE), whereas one and two nodes reduce inside NVSwitch with NVLS/NVLS_TREE. Each process's first backward uses DDP's initial buckets (one 1.3 GB and one 3.0 GB all-reduce); later backwards use buckets rebuilt in gradient-ready order (28.5–201 MB), identical from the second iteration on. A four-node probe (two independent launches; three DDP forward/backward passes on a fixed batch, one raw all-reduce, and one Muon+AdamW update on fixed synthetic gradients; per-rank bit fingerprints) was bitwise identical across launches, but within a launch iteration 1 differed from iteration 2 in 48/376 parameter gradients on every rank, while iterations 2 and 3 matched.

**Cause:** a resumed segment's first update used the initial buckets while the continuous run's same update used the rebuilt ones; under four-node RING the summation order of an element depends on its bucket position, so the last bits differed. NVLS reductions are layout-independent, which is why eight- and sixteen-rank B1 passed. **Fix (execution only, `reference/train.py`):** before each process's first update, one discarded DDP forward/backward on an all-zero dummy batch rebuilds the buckets; gradients are cleared and all RNG, including routing RNG, is restored. It reads no data and changes no optimizer state, budget, or theoretical cost; it costs about one microbatch forward/backward. The science hash `b0a4afc1…` and cost proofs were unchanged. A staged method resumed after a stage switch would rebuild from that stage's gradient-ready order; GDN has no stages. The new eight-gloo-rank test `test_resumed_segment_is_bitwise_with_eight_ddp_ranks` fails without the warm-up and passes with it (gloo's ring is also layout-dependent). CPU: 293 passed / 20 CUDA-only skipped.

The repeated four-node check used the record 025 criteria and unchanged eight-rank reference (bundle `79c3ff91`). B2–B4 passed; **B1 still failed**: updates 1–2 were bitwise, then 1.2e-8 relative from update 3. The warm-up remains necessary but was not the only cause; see record 031.

## Record 030

**RetNet formal run complete, cumulative audit `matched`, 2026-10-22.** The RetNet run launched in record 015 finished at 14:47Z in two segments (updates 0–25,736 and 25,736–46,875), both on 8×H20-3e with microbatch 8 × accumulation 8 and bundle `eed81b3e…`. Every segment's provenance matched: model, protocol, data, and tokenizer fingerprints, parameter count, configuration, cost proof, validation source `ae7a22b5…`, and science hash `178345f7…`.

The run completed 46,875 optimizer updates and 49,152,000,000 input tokens. Theoretical training compute was 2.2758×10²⁰ FLOPs, cumulatively −0.906% versus the SSSL anchor, equal to its configuration proof and within the 2% threshold, with no budget anomaly; the trainer recorded `final_status=matched`. Evaluation was complete: 189 periodic validations (41,943,040 tokens each) and six CORE events with all 22 tasks.

Results use the final step-46,875 checkpoint without selection: periodic val BPB **0.6825** (also the lowest of the 189), independent final BPB val **0.6797** / train **0.6820** (20,971,520 tokens per split), and CORE **0.2936**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7989/0.7835/0.7570/0.7296/0.7022 and CORE 0.1850/0.1892/0.2236/0.2349/0.2799. All losses and gradient norms were finite; the largest gradient norm after update 500 was 0.61.

Resources are outcomes, not controls: median 5.87 s/update, theoretical MFU 69.9%, 76.7 h training wall time (613.6 GPU-hours), and 5.1 h evaluation wall time. Evaluation forward compute, 1.22×10¹⁹ FLOPs, is reported separately and excluded from the training budget. The W&B mirror recorded all 4,885 events (4,689 train, 189 val, one final BPB, six CORE). A read-only audit of metrics, val/CORE files, budget, run status, and segment provenance found no problems. [Evidence](../../evidence/checks/h20-retnet-formal.json).

This is a single-seed result. The KDA v2 (8×H20), `elu_qknorm` (16×H20), and GDN (32×H20) runs use other rank layouts, which change floating-point reduction order; disclose this in cross-run comparisons. No method ranking is made before the other methods finish.

## Record 031

**Four-node resume diagnosis, part 2: optimizer kernel configurations, 2026-10-23.** The mismatch left after record 029 came from per-process Inductor configuration choices in the fused optimizer step. After the fix (bundle `b41faaab…`, commit `c77bf36`), four-node bounded acceptance L1 and B1–B4 passed, and an additional eight-segment resume chain was bitwise. All diagnostics were bounded jobs that kept no training state. [Evidence](../../evidence/checks/h20-four-node-determinism.json).

Localization:

- Per-rank bit fingerprints in the real trainer showed that the resumed process's first update had identical gradients, optimizer state, and pre-update parameters. After the update exactly one parameter differed, identically on all ranks: the `[1536, 6144]` Muon matrix `h.18.mlp.c_proj.weight` owned by rank 18.
- With per-rank kernel caches the mismatch remained, but now the continuous run's own first update differed from other processes, so the problem was not resume-specific.
- A four-node optimizer probe repeated the same step 30 times from one snapshot of the real model and optimizer in five independent launches (default twice, cuBLAS workspace, waiting for all_gather, eager). Each launch was bitwise self-consistent, and the default, workspace, and wait launches agreed with one another: the step is deterministic and unrelated to NCCL overlap. Repeating each trainer update three times also matched.
- Per-rank Inductor logs showed that in the resumed process rank 5 compiled one Muon reduction with `R0_BLOCK` 1024 where the first launch used 2048. Even with pointwise autotuning and coalescing analysis disabled, Inductor shrinks `R0_BLOCK` 2048 to 1024 at runtime by register use and times both (0.1409 / 0.1404 ms, within 0.4%). Every launch made 23 such choices, one per rank holding a `[1536, 6144]` Muon matrix.

**Cause:** the compiled AdamW/Muon fused step benchmarks some reduction-kernel configurations in each process. Near-equal timings let processes choose differently, and cached choices are not guaranteed to be reused. A different block size changes the summation order and the last bits of the same update. Gradients were always bitwise reproducible; one- and two-node B1 passed only because this case was not triggered.

**Fix (execution only):**

- Each local rank uses its own Triton and Inductor cache directories, and the Inductor cache is stored inside the per-node archived Triton cache (`reference/train.py`, `run.sh`, commit `f5ca6e3`). FLA's Triton autotuning also chooses differently per rank on a first launch; those choices had no numerical effect in this diagnosis, and each rank now reuses its own on resume. Each node's archive is about 1.3–1.6 GB.
- `ExperimentOptimizer.step` compiles with Inductor `triton.autotune_pointwise`, `triton.coalesce_tiling_analysis`, and `dynamic_scale_rblock` disabled (`nanochat/experiment/optimizer.py`, commit `c77bf36`). Each kernel keeps one heuristic configuration, no process benchmarks, and the mathematics equal the compiled version.
- The science hash `b0a4afc1…` and cost proofs were unchanged. CPU: 295 passed / 20 CUDA-only skipped, including new tests for per-rank cache directories and fixed optimizer configurations.
- Two intermediate attempts were not adopted: waiting for all_gather before Muon updates (`8cd832d`) was not the cause and was removed; eager fused steps (`d9110b9`) were bitwise but less precise (below) and were replaced.

Acceptance used the record 025 criteria and the unchanged eight-rank reference. Three instrumented launches performed zero runtime block shrinks and zero multi-configuration timings, and all fingerprints over four updates matched on 32 ranks. L1 and B1–B4 passed: step-0 val 3.1687824620648284 and update-1 loss equal eight-rank execution bit for bit, and stream states match. An eight-update continuous run and eight single-update resumed segments (each starting from an empty cache directory and restoring only its node's archive, as on a new pod) had bitwise identical losses, gradient norms, and final weights. Thirty-two ranks took 1.813 s/update (88.6% scaling efficiency versus eight) with 104.0 GiB peak memory; over 12 updates the largest relative loss difference from eight ranks was 2.3e-5.

Precision (single GPU, each group's per-rank Muon blocks and AdamW slices at 32 ranks, update relative L2 error versus float64 of the same algorithm): fixed-configuration and default Inductor Muon agree at the reported digits — `[12,12]` 3.854%, 3×`[1536,1536]` 1.420%, `[1536,6144]` 2.056%, `[6144,1536]` 2.129%. Eager gives 4.102% / 2.185% / 3.272% / 3.093% because it rounds every BF16 operation, while the compiled step fuses the Polar Express elementwise parts in FP32. AdamW is identical at BF16 resolution for all three; with FP32 parameters compiled and eager differ by 3e-6 (the reference is effectively FP32 because of the in-function `.float()`).

**Disclosure:** RetNet, KDA v2, and `elu_qknorm` ran the optimizer with Inductor's per-process benchmark choices, re-timed on a new pod in every segment. Their resumes restored optimizer, RNG, data cursor, and cost accumulators as the protocol requires, but segment boundaries are only rounding-level, not bitwise, equal to a continuous run. GDN's fixed heuristic configuration has the same compiled mathematics; reduction block sizes may differ, a rounding-level difference to disclose in cross-run comparisons.

## Record 032

**GDN formal launch on 4×8 H20, 2026-10-23.** All conditions preregistered in record 025 were met: eight-rank L1, two-node B1–B4 and pilot P1–P5, and four-node B1–B4 (record 031). With the owner's authorization, the formal run was submitted at 04:27 CST on 32 H20-3e GPUs with bundle `b41faaab…` (commit `c77bf36`), microbatch 16 × accumulation 1 per rank, evaluation batch 16, and 45.5-hour `chain.sh` segments. The 2026-10-22 sixteen-rank attempt had completed only step-0 validation and no update; the trainer moved its records aside, and the formal run starts from random initialization.

Provenance: science hash `b0a4afc1…`, validation source `6ba2fbd0…` (the launch commit), 4.687197 GFLOPs/token (+0.317% versus SSSL), and 1,411,349,680 parameters. Launch checks: step-0 val 3.1687824620648284 equals the eight-rank, two-node, and four-node checks bit for bit; update-1 and update-10 loss and gradient norm equal the four-node acceptance's continuous run bit for bit; update-250 val BPB was 1.0048; after update 100 the largest gradient norm was 0.44, all values were finite, and no budget anomaly occurred. Steady updates took 1.81 s with 57.3% theoretical MFU and 104.0 GiB peak memory; the per-update cost ledger stayed at +0.317%. A read-only W&B mirror was started.

Training was expected to take about 23.5 h, or about 26 h with 189 validations, six CORE events, and checkpoints, ending around 2026-10-24 07:00 CST within one segment. **At this record the run is in progress;** final results, the cumulative audit, and `final_status` require a later record.

## Record 033

**GDN hybrids H02/H03 registered, CPU evidence only, 2026-10-24.** Added per-layer mixer layouts for v2 hybrids: a `HYBRID` method set whose `layer_mixers` are derived from the method and depth, serialized into the fingerprint, and verified on load. Registered `gdn_hybrid` (H02: GDN, GDN, GDN, full attention, repeated) and `gdn_hybrid_swa` (H03: GDN and sliding-window attention alternating, GDN first, 2048 keys including the current token). GDN layers are A07's mixer on `fla_chunk64`; attention layers are `dense_full`'s native attention, with VE where the shell places it. Sources were checked against primary material ([plan](../plans/V2-linear-hybrids.md#2-source-verification)): H02's 3:1 order comes from the Qwen3-Next model card and `config.json` because the registered blog page renders no text, and H03's order comes from the official `lit_gpt` code (`gated_delta_per_layer=2`). At the 2048-token context, H03's window covers every causal key, so it computes a 1:1 GDN/full-attention hybrid; a test confirms the equivalence. Adaptations are listed in [sources](../methods/SOURCES.md#gdn-hybrids-v2-h02-h03).

`reference.calibrate` now applies the PROTOCOL v2 tie-break to hybrids. It audits every allowed depth in 16–32 at FFN 6144 with exact FFN slopes, fully audits the estimated-within-1% candidates in tie-break order, and publishes the table. H02 selected **24 layers / FFN 5760, +0.672%, 1,399,275,962 parameters**; the legacy nearest-width search would have taken 24 / 5632 (−0.546%), which the tie-break ranks second. H03 selected **22 / 6528, −0.542%, 1,329,681,662 parameters**, tied with 24 / 5760 on FFN distance and chosen by the smaller error. In the new [proof set](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/hybrids-20261024/summary.json), all 27 configurations pass admission. The 25 earlier rows equal [v2-selen-20261023](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/v2-selen-20261023/summary.json) in model hash, parameters, stage weights, and every operator count; only the source hash changed, because the configuration, model, linear-mixer, and calibration sources were edited. The SSSL anchor is unchanged at 4.67239832544785 GFLOPs/token.

Validation: **383 CPU passes / 34 CUDA-only skips**. This includes 13 hybrid tests (registered positions; serialization and rejection of an edited layout; per-layer mixer types and windows; trace; an SWA receptive field of exactly w keys; a covering window equal to full attention; attention edges charged only at attention layers, including the SWA window; optimizer ownership; tie-break selection) and the existing all-method backward, causality, and operator-coverage tests for both rows. `reference.fla_check` checkpoint mode reports physical layer numbers for hybrids.

Limits: there is no GPU evidence yet. The H20 attention backend (FA3 from the offline `kernels` cache, or the SDPA fallback), full-size BF16 `[1|8,2048]` checks, single-GPU update throughput and memory, an eight-rank bounded check with bitwise resume, and the pilot are all pending; formal runs need explicit authorization. No training was started.

## Record 034

**H02/H03 single-H20 checks, deterministic attention, and preregistered 4×8 acceptance, 2026-10-24.** Two bounded one-GPU jobs ran on bundle `b7f4f7a7…` (commit `da3f7b3`, record 033) on an H20-3e with PyTorch 2.9.1+cu128. FA3 did not load from the `kernels` hub, either offline (as `run.sh` runs) or with network access, so native attention used PyTorch SDPA. The 36 targeted GPU tests passed. Full-size BF16 `[1,2048]` and `[8,2048]` forward/backward passes were finite with zero causal-prefix error for both rows.

Throughput for one rank's 4×8 share (16 rows) with the unsharded optimizer and real ClimbMix rows:

| Row | Attention path | Microbatch 16 × 1 | Microbatch 8 × 2 |
|---|---|---|---|
| H02 | SDPA flash (full causal) | 1.788 s, MFU 58.3%, 99.7 GiB | 1.820 s, 61.4 GiB |
| H02 | SDPA without flash (efficient kernel) | 1.839 s, MFU 56.6%, 99.7 GiB | — |
| H03 | Masked SDPA (left window 2047) | 1.945 s, MFU 52.9%, 87.7 GiB | 1.974 s, 54.9 GiB |

GPU utilization was 100% in every case. **Run-to-run determinism** was tested with two identical full-size forward/backward passes, comparing every gradient bitwise:

- H02 with flash: 327/337 gradients differed at microbatch 4 (max 5.2e-6), and 326/337 at microbatch 16.
- H02 without flash: bitwise identical at microbatch 4 and 16.
- H03 on its masked path: identical at microbatch 4, but 249/260 differed at microbatch 16 (max 9.5e-7).

Forward losses were always equal. The bitwise resume criteria (L1/B1) that A04–A07 met therefore cannot hold with SDPA's default kernels. The owner chose to keep the bitwise criteria rather than switch to a tolerance.

**Changes (execution only; model semantics unchanged):**

- `nanochat.flash_attention` now takes SDPA's unmasked causal path when the left window is at least T−1. That window already reaches the first key from the last query, so the result is identical; H03's 2047 window at T=2048 previously built an explicit mask. A test checks the dispatch and exact equality.
- `reference.train --deterministic-attention` disables SDPA's flash kernel, and rejects FA3, whose backward is also nondeterministic. The segment identity records the native attention kernel (`fa3`, `sdpa`, or `sdpa-noflash`), so a resumed segment must load the same one.
- The science hash changed with `nanochat/flash_attention.py`. In the re-audited [proof set](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/sdpa-window-20261024/summary.json), all 27 configurations pass admission with every count identical to [hybrids-20261024](https://github.com/Selen-Suyue/NextTransformer/blob/evidence-archive-20261024/reference/results/costs/hybrids-20261024/summary.json). CPU: 385 passed / 34 skipped.

**Preregistration (written before these jobs run; not adjusted afterwards).** The owner chose four H20 nodes (32 GPUs) for each hybrid's training. Every job below uses the bundle published from this record's commit, with `--deterministic-attention`, microbatch 16, and evaluation batch 16; A07's thresholds (record 025) apply unchanged. For each of `gdn_hybrid` and `gdn_hybrid_swa`:

1. One-GPU confirmation (informational): determinism at microbatch 16 and throughput through the corrected path.
2. Eight-GPU bounded check (`bounded_check.sh`, 12 updates, `--smoke-val-tokens 41943040`). **L1:** after a resumed segment, losses and final weights equal the continuous run bitwise.
3. Four-node bounded check (`NEXTT_NODES=4`, the same options, `NEXTT_LAYOUT_REFERENCE` = step 2).
   - **B1:** bitwise resume.
   - **B2:** at update 12, the 32 ranks' data-stream states equal the corresponding eight-GPU streams.
   - **B3:** step-0 val BPB (41,943,040 tokens) within 1e-5 relative of eight GPUs.
   - **B4:** update-1 loss within 1e-5 relative of eight GPUs.
4. Pilot on 4×8 H20: smoke, accounted separately, formal schedule, 512 updates, full validation at 0 and 512.
   - **P1:** all losses and gradient norms finite, and the phase is `smoke_complete` at 512.
   - **P2:** after update 100, every logged loss is at most 0.25 above the median of the logged losses in the preceding 200 updates.
   - **P3:** after update 100, gradient norm ≤ 5.
   - **P4:** val BPB at 512 ≤ 1.05.
   - **P5:** `fla_check --checkpoint` at update 512 (one val row, every GDN layer, production FLA path against the FP64 recurrent oracle) has output error ≤1% and every input-gradient error ≤3%.

   Attention layers have no separate oracle; step 1 and record 033 cover their causality and finiteness.
5. A formal 4×8 run (`chain.sh`, `NEXTT_NODES=4`, 45.5-hour segments) starts only if every condition passes **and the owner explicitly approves the launch**. Any failure stops that method and is recorded. Pilot weights never enter a formal run.

## Record 035

**KDA v2 formal run complete, cumulative audit `matched`, 2026-10-23.** The KDA v2 run restarted in record 016 finished at 12:25Z in three segments (updates 0–21,681, 21,681–32,000, and 32,000–46,875), all on 8×H20-3e with microbatch 16 × accumulation 4 and bundle `db1fbcfa…` (commit `82bbe58`). Every segment's provenance matched: model, protocol, data, and tokenizer fingerprints, parameter count (1,412,126,344), configuration, cost proof, validation source `303ec29b…`, and science hash `f77d313e…`. The first KDA attempt, aborted at update 2340, is not part of this run.

The run completed 46,875 optimizer updates and 49,152,000,000 input tokens. Theoretical training compute was 2.3064×10²⁰ FLOPs, cumulatively **+0.429%** versus the SSSL anchor, equal to its configuration proof and within the 2% threshold, with no budget anomaly; the trainer recorded `final_status=matched`. Evaluation was complete: 189 periodic validations and six CORE events with all 22 tasks.

Results use the final step-46,875 checkpoint without selection: periodic val BPB **0.6671** (also the lowest of the 189), independent final BPB val **0.6644** / train **0.6666**, and CORE **0.3106**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7797/0.7639/0.7385/0.7127/0.6867 and CORE 0.2156/0.2252/0.2494/0.2689/0.2986. All losses and gradient norms were finite; the largest gradient norm after update 500 was 0.61.

Resources: median 7.02 s/update, theoretical MFU 59.2%, 91.8 h training wall time (734.5 GPU-hours), and 5.8 h evaluation wall time; evaluation forward compute (1.24×10¹⁹ FLOPs) is excluded from the training budget. The read-only audit found no problems. [Evidence](../../evidence/checks/h20-kda-formal.json).

## Record 036

**`elu_qknorm` (A06″) formal run complete, cumulative audit `matched`, 2026-10-24.** The run launched in record 023 finished at 16:24Z (2026-10-23) in two segments (updates 0–31,315 and 31,315–46,875) on 2×8 H20-3e (16 ranks sharing the eight frozen streams) with microbatch 8 × accumulation 4 and bundle `2235165e…` (commit `a3a9fde`). Segment provenance matched (1,462,768,668 parameters, validation source `dd3171c0…`, science hash `534914e0…`).

The run completed 46,875 updates and 49,152,000,000 input tokens with 2.2867×10²⁰ FLOPs, cumulatively **−0.431%** versus the SSSL anchor, no budget anomaly, and `final_status=matched`; all 189 validations and six full CORE events are present. Final step-46,875 results: periodic val BPB **0.6998** (also the lowest), independent final BPB val **0.6970** / train **0.6990**, CORE **0.2304**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.8289/0.8138/0.7833/0.7518/0.7213 and CORE 0.1428/0.1604/0.1782/0.2017/0.2256. All values were finite; the largest gradient norm after update 500 was 1.29.

Resources: median 3.82 s/update on 16 GPUs, theoretical MFU 53.9%, 50.2 h training wall time (803.8 GPU-hours), and 3.4 h evaluation. [Evidence](../../evidence/checks/h20-elu-qknorm-formal.json). **Report this as A06″**: it was registered after the original ELU+1 runs and `elu_norm` failed (records 017–022), and its pilot's P5 check was a post-hoc substitution (record 022); those failures are part of the A06 result.

## Record 037

**GDN formal run complete, cumulative audit `matched`, 2026-10-24.** The run launched in record 032 finished at 23:48Z (2026-10-23) in one segment on 4×8 H20-3e (32 ranks) with microbatch 16 × accumulation 1 and bundle `b41faaab…` (commit `c77bf36`); 1,411,349,680 parameters, validation source `6ba2fbd0…`, science hash `b0a4afc1…`.

The run completed 46,875 updates and 49,152,000,000 input tokens with 2.3039×10²⁰ FLOPs, cumulatively **+0.317%** versus the SSSL anchor, no budget anomaly, and `final_status=matched`; all 189 validations and six full CORE events are present. Final step-46,875 results: periodic val BPB **0.6700** (also the lowest), independent final BPB val **0.6673** / train **0.6694**, CORE **0.3055**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7826/0.7670/0.7419/0.7158/0.6896 and CORE 0.2021/0.2036/0.2545/0.2674/0.2942. All values were finite; the largest gradient norm after update 500 was 0.58.

Resources: median 1.80 s/update on 32 GPUs, theoretical MFU 57.6%, 24.1 h training wall time (772.5 GPU-hours), and 1.5 h evaluation. [Evidence](../../evidence/checks/h20-gdn-formal.json).

With records 030 and 035–037, the four H20 linear runs are complete: final independent val BPB KDA v2 0.6644, GDN 0.6673, RetNet 0.6797, A06″ 0.6970; final CORE 0.3106, 0.3055, 0.2936, 0.2304. These are single-seed runs with no variance estimate, on different rank counts (8/32/8/16), whose floating-point reduction order differs; segment boundaries of the three multi-segment runs are rounding-level rather than bitwise equal to continuous runs (record 031). The ordering is an observation of these runs, not a statistical result.

## Record 038

**Full-attention Dense (B01 `dense_full`) formal run complete, cumulative audit `matched`, accepted as the Dense baseline, 2026-10-26.** The run launched on 2026-10-23 (record 028) from the frozen checkout `asoul/NextT` at commit `4dd6ebb` finished at 10:43 CST in one segment on 1×8 A800-SXM4-80GB (ModelArts) with microbatch 8 × accumulation 8 and SDPA native attention; 1,355,810,568 parameters, 23 layers, FFN 6144, validation source `2130db89…`, science hash `e6e8b847…`, PyTorch 2.9.1+cu128. The preregistered pilot of v2 B01 was skipped by the owner's decision; the formal run started from random initialization and is the only `dense_full` run.

The run completed 46,875 updates and 49,152,000,000 input tokens with 2.2977×10²⁰ FLOPs, cumulatively **+0.048%** versus the SSSL anchor and equal to its configuration proof on every update, with no budget anomaly and `final_status=matched`; all 189 validations and six full CORE events are present. Final step-46,875 results: periodic val BPB **0.6672** (also the lowest of the 189), independent final BPB val **0.6645** / train **0.6666**, CORE **0.3310**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7825/0.7697/0.7441/0.7159/0.6878 and CORE 0.2157/0.2250/0.2401/0.2765/0.3058. All values were finite; the largest gradient norm after update 500 was 1.09.

Resources: median 4.51 s/update, theoretical MFU 43.5%, 45.7 GiB peak memory, 59.0 h training wall time (472.4 GPU-hours), and 4.0 h evaluation; evaluation forward compute (1.24×10¹⁹ FLOPs) is excluded from the training budget. A read-only audit of metrics, val/CORE files, budget, run status, and segment provenance found no problems, and the tracked run's summary equals the run directory. [Evidence](../../evidence/checks/a800-dense-full-formal.json).

The record was exported with `export_run --accept-baseline`, so the observatory and the README compute Dense-relative differences against this run. At the final checkpoint, 3 of the 18 other finished runs have a lower periodic val BPB or a higher CORE (SSSL -0.00002 BPB and +0.69 CORE; NSA +0.00062 BPB and +0.69 CORE; KDA -0.00005 BPB and -2.03 CORE); the other 15 trail on both. These are single-seed observations on different hardware and rank counts, not statistical rankings. SSSL keeps its own results, and its native-v3 matrix-FLOPs accounting is not comparable with this run's ledger.

## Record 039

**Xiangyue's DeltaNet and GLA complete, 2026-10-27.** DeltaNet and GLA each finished 46,875 updates / 49.152B tokens, 189 full validations, six full CORE-22 events and independent final BPB. Independent val BPB: **0.675966 / 0.666320**; CORE: **0.283357 / 0.311091**. Records: [DeltaNet](../../results/runs/deltanet.json), [GLA](../../results/runs/gla.json); [cumulative audits and historical source patch](../../evidence/checks/xiangyue-a01-a03-20261027). All nine final payloads per run passed shared-storage readback and both archive validators. Their frozen `reference-aten-math-v2-all-backends-replay` accounting uses a 4.71661506444785 theoretical SSSL anchor; the results use the standard final/replay rankings and measured Dense-relative BPB/CORE differences while retaining their original compute accounting. The source patch reconstructs the 89 executed files from `f417862`; it does not change main's active trainer. GLA's originally empty Git commit is preserved with its source SHA. Single seed.

## Record 040

**Cost re-audit after removing GSA, 2026-10-27.** Removing GSA from `registry.py` and `models/linear.py` changed the science hash to `a2bff1f3…`, so every configuration's proof became stale and all 26 failed admission in CI. All 26 were re-audited with `calibrate --reuse-configs` under PyTorch 2.9.1+cu128 and `reference-aten-math-v2`, keeping each selected structure. Relative to [layout-20261024](../../evidence/costs/layout-20261024/summary.json), every model hash, protocol hash, parameter count, stage weight, and per-operator count is identical; only source hashes, proof paths, and audit timings differ. The SSSL anchor stays at 4.67239832544785 GFLOPs per input token. New proofs: [gsa-removal-20261027](../../evidence/costs/gsa-removal-20261027/summary.json). Formal runs keep the proofs they were admitted with.


## Record 041

**Others results consolidated; both QK-normalized MLA runs complete on H20 and audited, 2026-10-28.** The [progress report](PROGRESS-20261028.md) consolidates the 21 finished records already on main and four finished Others runs from `gy/others-attention` at `fe8089e`. It distinguishes accepted results, tracking-reported completions awaiting independent audits, and historical native accounting. The model port and observatory exports are separate work.

A read-only audit of the original `mla_qknorm` and `mla_nope_qknorm` H20 run directories passes for both: 46,875 updates, 49.152B input tokens, 189 formal validations, six full CORE-22 events, final BPB and checkpoint completion, cumulative compute and segment provenance. Additional checks reproduce all CORE centering/aggregates and confirm the canonical task set, final BPB consistency, pinned tokenizer/protocol and recorded deterministic mode. Checkpoint file inventories are included; tensor payloads were not loaded or rehashed. [Audits and provenance](../../evidence/checks/others-20261028/README.md).

| Row | Independent final val BPB | Final CORE | Cumulative cost vs SSSL | Training GPU-hours |
|---|---:|---:|---:|---:|
| `mla_qknorm` | 0.673241 | 0.325330 | −0.0415% | 685.0 |
| `mla_nope_qknorm` | 0.682963 | 0.276536 | +0.2819% | 664.2 |

Both were trained from update 0 in one segment on 32 H20 GPUs, microbatch 16 × accumulation 1, with strict deterministic algorithms and compiled flash SDPA. The original provenance identifies bundle `c149799c…`; its Git commit field is null and remains null. The preregistration/launch record associates that bundle with `a0c795c`. Finishing times were 2026-10-28 02:15:51 / 01:57:27 UTC+8. There were no logged loss jumps greater than 1.0 above the running minimum after update 1,000 and no adjacent validation increases above 0.005; maximum gradients after update 500 were 1.185 / 0.759.

The two original H800/H200 audits are copied byte-for-byte: MQA final BPB/CORE 0.674317/0.319617 and GQA 0.671447/0.326245. Their branch-local record numbers refer to the frozen source in `sources.json`, not main's numbering. The QK-normalized rows were registered after the first `mla` and `mla_nope` runs diverged; by the maintainer's decision those runs, and two H800 attempts of the QK-normalized rows that stopped for capacity, are not reported here, and their evidence stays on `gy/others-attention` at `fe8089e`. All four rows disable VE; the missing O00 control prevents isolating KV parameterization from VE policy. No new model admission, training launch or result export is implied by this report.

## Record 042

**First GPU parity check of the scoring image: `dense_full` scored bit-identically on A800, 2026-10-28.** At the maintainer's request, the PR #8 scoring image scored the final checkpoint of the accepted Dense baseline (record 038, trained at `4dd6ebb` on 1×8 A800 with SDPA). The first host evaluation under the scorer's environment failed: forced weights-only loading rejected `torch.torch_version.TorchVersion`, the type of the `torch.__version__` string every trainer checkpoint stores. `640ceb5` allows only that `str` subclass in the preflight inspector and the evaluator; the science hash is unchanged (`a2bff1f3…`), and the CI Python job passed on Linux (383 passed, 38 skipped).

A reference interpreter was built on the GPU host from the image recipe; its 107-package inventory and interpreter binary equal the image's. The training environment shares every package version but adds development tools and the project's own package and lacks the linear overlay, so the preflight's exact inventory check excludes it. `environments.eval.capture` recorded the reference (SDPA, microbatch 8, empty FA3 snapshot, all 6,543 shard hashes); tolerances were registered before either evaluation (BPB ±1e-4, CORE ±1e-3, each task ±5e-3). The image was built from `640ceb5` with `build.sh`, with the apt mirror switched to Huawei after Tsinghua returned HTTP 403 (image `sha256:a1cffa18…`). The GPU host has no Docker daemon, so `environments.eval.run` ran under udocker 1.3.17 (fakechroot F3, host NVIDIA libraries) with `launch.py`'s mounts, environment and entry point. Input preflight, eight-rank GPU preflight, evaluation and validation completed in 19 minutes.

| Source | Train BPB | Val BPB | CORE |
|---|---:|---:|---:|
| Training-final record (record 038) | 0.6665835136673783 | 0.6644601775656058 | 0.33095152996093863 |
| Host reference, `640ceb5` | 0.6665835136673783 | 0.6644601775656058 | 0.33095152996093863 |
| Scoring image, `640ceb5` | 0.6665835136673783 | 0.6644601775656058 | 0.33095152996093863 |

All 22 task scores are identical too, and `environments.eval.compare` passed with largest difference 0.0. This is evidence for one registered method on one hardware class. udocker does not enforce the launcher's network, read-only, capability or device isolation; the Docker launcher, Harbor GPU trial, FA3 kernels, sparse and linear methods, a second machine and image publication remain untested. [Evidence](../../evidence/checks/scoring-dense-full-a800-20261028/README.md).


## Record 043

**H20 Others run records exported for the observatory, 2026-10-28.** Following [issue #9](https://github.com/Selen-Suyue/NextTransformer/issues/9), the standard `nexttransformer.export_run` exporter read the original metrics and provenance for `mla_qknorm_h20_seed42` and `mla_nope_qknorm_h20_seed42`, copied read-only from persistent storage with their file timestamps. The resulting [MLA + QK-norm](../../results/runs/mla_qknorm.json) and [MLA-NoPE + QK-norm](../../results/runs/mla_nope_qknorm.json) records each contain 189 validation points, six complete CORE-22 events with raw and centered task scores, independent final BPB, cumulative compute and source fingerprints. Their final values agree with the [record 041 audits](../../evidence/checks/others-20261028/README.md). The exporter uses source commit `a0c795c` from the launch record, preserves bundle `c149799c…`, and includes the frozen-branch provenance note; current model admission is unchanged.

The existing Atlas figures for these two MLA variants now separate the key and value paths: QK normalization applies to queries and keys, values bypass it into causal attention, and the shared rotary key appears only in the RoPE variant. MQA/GQA retain their existing figures and model cards. Their original run directories are not available on the H20 mount, so their record exports remain with their owner under issue #9. No training or evaluation was rerun.


Hybrid branch records are numbered 044–048 below to retain main's existing identifiers. Historical JSON evidence and source hashes remain unchanged.

## Record 044

**H01 shared-framework integration, 2026-10-25 (owner @kris): implemented, theoretical preflight passed, bounded checks passed; full deployment acceptance pending.** H01 now uses H02/H03's `hybrid_layout`, `ReferenceLM`, public trainer, full-graph auditor and admission. The only model changes are registration/configuration and adding ungated DeltaNet to the shared linear FLA dispatch/oracle. CUDA calls `chunk_delta_rule`; GDN retains its existing KDA-kernel adaptation. No H01-specific trainer, admission identity, audit extrapolator or optimizer was added. See the [source and adaptation notes](../methods/SOURCES.md#deltanet-hybrid-v2-h01) and [shared training guide](../guide/METHODS_FRAMEWORK.md).

The registered grid selects **24 layers, FFN 6400, 1,403,810,314 parameters**, with full causal MHA at one-based layers **2 and 13**, DeltaNet elsewhere. Direct audit of the complete `[512, 2048]` mixed graph reports **4.675325779958443 GFLOPs/input token**, **+0.0626542%** versus SSSL, and **229,801,612,736,517,375,000 FLOPs** over the one `joint` stage's 46,875 updates. Unknown floating operators: zero. The [28-configuration proof set](../../evidence/costs/h01-shared-20261025/summary.json), science hash `657db174fe6d87a0f4af6772ce474a19631779d22c46be16484a9a8a39d3e738`, preserves all 27 earlier configurations' model hashes, stage weights and operator counts. H01's counts also exactly match its original FLA proof; [comparison](../../evidence/checks/h01-shared-20261025/cost-comparison.json). All fixed Protocol values and tolerances remain unchanged.

**Validation:** 419 CPU tests passed, 37 CUDA-only tests skipped (full suite plus separate configuration/admission tests); five targeted CUDA regressions passed. CPU comparison against the original Hybrid implementation is bitwise equal for initialization, logits, loss, every gradient and optimizer ownership on both scan and chunk paths, with nonzero output projections. On one **H100 80GB**, PyTorch 2.9.1+cu128 / Triton 3.5.1 / FLA 0.5.2, the public FLA check passes DeltaNet and GDN at T=65/2048, batch 2, seeds 0/1. DeltaNet's maximum relative output error is **0.7269%**, maximum input-gradient error **0.9009%**, within the existing 1%/3% limits. FLA causality and independent-row checks pass.

The full selected H01 configuration passes synthetic BF16 `[1, 2048]` forward/backward and causal-prefix checks (error zero), with peak allocated memory **10,785,426,432 bytes**. The native attention dispatch was SDPA; this was not the deterministic-attention resume check. It performs no complete global optimizer update. [Machine-readable validation](../../evidence/checks/h01-shared-20261025/validation.json), [FLA details](../../evidence/checks/h01-shared-20261025/fla.json), [full-size details](../../evidence/checks/h01-shared-20261025/full-size.json).

**Remaining acceptance:** new-source eight-rank complete updates and real-data fresh-process resume, target-cluster resource identity and deterministic attention, and hardware checks for any intended 16/32-rank deployment. Other jobs occupied four GPUs, so no eight-GPU test was run here. No pilot or formal training was launched by this integration; no new `matched` claim is made. Earlier H01 code and evidence on `Hybrid` remain immutable. The shared layout serialization changes H01's model fingerprint, so existing formal runs must resume from their original frozen source/environment/proof.

## Record 045

**H04 and VE-off controls, 2026-10-26 (owner @kris): implemented and cost-admitted; bounded H100 and bitwise resume checks passed; pilot and formal runs pending.** H04 repeats KDA/KDA/KDA/MLA-NoPE, reusing A04's FLA mixer and the collaborator's O03-N MLA implementation (`8cba0a3`). MLA has Q/KV ranks 768/256, 12 heads, content/V dimensions 128/128 and no RoPE or reconstructed-head QK norm. All VE tables and gates are removed. The public model builder, optimizer, full-graph auditor, admission and trainer are reused; no H04-specific trainer or admission path was added. [Source adaptations and known MLA/Muon stability risk](../methods/SOURCES.md#kda--mla-nope-hybrid-v2-h04-and-ve-off-controls).

The registered grid selects **24 layers / FFN 6016, 793,542,986 parameters**, with MLA at one-based 4/8/12/16/20/24. The direct `[512,2048]` mixed graph costs **4.692309035399331 GFLOPs/input token (+0.42613%)**: forward/backward/optimizer are 1,632,110,029,336,083 / 3,263,947,478,872,351 / 24,185,130,894,455 FLOPs per update. Independent controls are `kda_no_ve` 23/6144 (+0.42583%), `mla_nope` 26/6144 (+0.25962%), and `mha_no_ve` 23/6144 (+0.04526%). All 32 configurations pass current-source admission; the prior 28 retain every operator count, parameter count and model/Protocol fingerprint. The SSSL anchor is unchanged and historical proofs are preserved. [Proof set](../../evidence/costs/h04-shared-20261026/summary.json).

**Checks:** 476 CPU regression passes (39 CUDA skips), including H04/control resume and deterministic-option cases. Old 28-model initialization, logits, losses and gradients are bitwise equal to the parent commit, and an existing H01 checkpoint loads identically. MLA FP64 reconstruction/absorption and CUDA BF16 output/all-gradient oracles pass; KDA's public FLA checks pass at T=64/65/1000/2048. All four new rows pass full-size `[1,2048]` BF16 forward/backward and causality, and an eight-rank synthetic update of 1,048,576 tokens with finite gradients and identical updated parameters on every rank. H04 uses microbatch 8 × accumulation 8; the pure KDA control uses 4 × 16. Logical 16/32-rank data mapping is checked on CPU, not corresponding hardware.

All 6,543 ClimbMix files (599,758,882,644 bytes) match the pinned revision's content manifest; the tokenizer/byte-table identity matches `06258dc0…7ec54`. On **8×H100 80GB**, the public trainer completes 12 real-data updates. FA3+FLA measures **376,030 tokens/s** over updates 4–12 (mean **2.789 s/update**, peak **67,455,391,232 bytes/rank 0**); this performance smoke uses short validation and no CORE. The separate SDPA-noflash resume check performs full 41,943,040-token validation at steps 0/12 and a 22-task CORE smoke capped at 32 examples/task. None is a formal run or a quality result.

**Initial failure, diagnosis and correction:** the first SDPA-noflash segmented check failed bitwise acceptance (maximum loss difference 0.0037431, maximum parameter difference 0.2421875). Instrumentation confirmed exact restored parameters and optimizer state on all eight ranks; differences first appeared in update-2 gradients. Full-model single-GPU repetition on the same real rows reproduced differences in MLA query gradients and their upstream gradients. Fixed cuBLAS workspace alone did not resolve them. Requesting deterministic PyTorch algorithms did: two fresh processes each repeated the full backward three times with identical losses and every gradient. The existing public `--deterministic-attention` option now also enables that mode and defaults the cuBLAS workspace to `:4096:8`. FA3 remains excluded by that option. This execution fix changes no science hash, operator counts or Protocol constants; no model-specific check was added.

The **uninstrumented public trainer** then passed continuous 12 updates versus 1 update, checkpoint, fresh-process resume to 12: every loss, final weight and all eight ranks' optimizer/RNG/data states are bitwise equal; token/FLOPs counters are exact. Deterministic SDPA+FLA measures **335,140 tokens/s**, mean **3.129 s/update** over updates 4–12, separate from the FA3 measurement. Full validation and the CORE smoke also pass. The initial failure remains in the evidence. A stability pilot, every eventual formal training/evaluation event, final cumulative acceptance and any 16/32-rank hardware deployment checks remain required. No formal run or `matched` result is claimed. [Machine-readable checks and limits](../../evidence/checks/h04-shared-20261026/validation.json).

## Record 046

**H05/H06 shared integration, 2026-10-26 (owner @kris): implemented and cost-admitted; bounded checks only.** H05 `qsa_hybrid` repeats GDN/GDN/GDN/QSA. H06 `moba_hybrid` uses MoBA except for exactly the last three full causal MHA layers. Both retain the native VE, FFN, smear, backout and residual/x0 shell. The only model additions are method/layout registration and an explicit layer kind for the existing `SparseMixer`. H05 reuses GDN's FLA path and S02's QSA; H06 reuses S05's MoBA and B01's MHA. Both use the existing sparse kernel backend, optimizer, trainer, checkpoint and admission. The trainer and auditor select sparse stages from actual layer kinds; there is no model-specific trainer or admission check.

The registered depth/FFN grid and tie-break select:

| Row | Layers / FFN | Parameters | GFLOPs/input token | Error vs SSSL | Total training FLOPs |
|---|---|---|---|---|---|
| H05 | 24 / 5760 | 1,405,174,202 | 4.691636932460713 | +0.411750% | 230,603,338,504,308,997,890 |
| H06 | 25 / 6016 | 1,452,935,068 | 4.659989005911318 | -0.265588% | 229,047,779,618,553,093,750 |

H05 has QSA at 4/8/12/16/20/24. Its first 512 updates apply dense burn-in and the indexer teacher only at those positions; all 18 GDN layers stay GDN. The remaining 46,363 updates use sparse QSA. Auxiliary losses follow S02 and sum across QSA positions. Burn-in costs 4.777911769516393 GFLOPs/token and joint training 4.690684174516393; both stages fit **inside** the same 46,875 updates / 49.152B tokens. H06 has 22 MoBA layers and full MHA at 23/24/25 (12% of layers), with one joint stage and no auxiliary loss. Its 25/6144 candidate is rejected at +1.003352%; the threshold is unchanged.

The complete mixed `[512,2048]` FakeTensor graphs include forward, backward, indexer/teacher, state operations and optimizer work, with no unknown floating operators. [Proofs and candidate tables](../../evidence/costs/h05-h06-20261026/summary.json) use science hash `95e60748722ec5d6d7b8f5fa6f52c671d6ab7098bbc15f829192d41f4aa24074`. All 34 configs pass common admission. The previous 32 retain their model/Protocol hashes, parameter counts, stage weights and every operator count; SSSL remains 4.67239832544785 GFLOPs/token. Historical proofs are preserved. [Cost comparison](../../evidence/checks/h05-h06-20261026/cost-comparison.json).

**Regressions and GPU checks:** 541 CPU passes / 43 CUDA-only skips, plus four CUDA sparse-kernel/reference output and gradient comparisons. The previous 32 models' initialization, logits, losses and gradients are bitwise equal to parent `3113eb8`, and an existing H01 checkpoint loads identically. Tests cover layouts and serialization, stage boundaries 0/511/512/46,874, teacher and selected-edge costs, summed auxiliary losses, optimizer ownership, causality, budget/events and CPU public-trainer resume across a debug burn-in boundary. On H100 with PyTorch 2.9.1+cu128, FLA 0.5.2 and Triton 3.5.1, all three full-size BF16 `[1,2048]` cases (H05 burn-in/joint and H06 joint) pass finite-gradient and causal-prefix checks with nonzero output projections; separate optimizer checks update every mixer output projection.

The same three cases pass a **four-rank synthetic complete update** on GPU 4/5/6/7: microbatch 4 × accumulation 32 × 4 ranks × 2048 = **1,048,576 input tokens**. Every gradient is finite and all four ranks have identical changed parameters. Peak rank-0 allocated memory is 38.28 GB for H05 joint, 38.26 GB for H05 burn-in and 32.76 GB for H06. Timings include cold compilation and are not stable throughput measurements. The shared distributed checker permits non-eight-rank checks only under an explicit `--smoke` flag and still requires the complete global batch.

The **uninstrumented public trainer**, with deterministic attention and the existing sparse kernel backend, also passes two real-ClimbMix updates versus one update, save, fresh-process resume for each model. Final weights, every loss and all four ranks' optimizer/RNG/packing states are bitwise equal; next-row hashes and token/FLOPs counters match exactly (2,097,152 tokens per trajectory). Tokenizer/byte-table identity remains `06258dc0…7ec54`. Validation at steps 0/2 uses 8,192 tokens per event, and CORE is not run in this smoke. H05 stays in burn-in during these two updates; its GPU joint path and CPU shortened-boundary resume are separate checks. These two updates are not a stable throughput benchmark. [Resume details](../../evidence/checks/h05-h06-20261026/resume.json).

Final packaging removed ten macOS AppleDouble metadata files accidentally carried by file transfer. Every executable Python file is byte-identical to the checked GPU source. All 34 cost proofs were regenerated against the clean checkout; raw GPU evidence keeps its original source hashes, with the cleanup identity mapping in `validation.json`. No historical proof was overwritten.

**Remaining acceptance:** registered eight-stream 8-rank updates/resume and deployment-specific 16/32-rank checks, the stability pilot (including H05's real 512-update transition), all formal training/evaluation events and final cumulative acceptance. Four-rank smoke uses four data streams, so it does not prove the registered eight-stream layout. No pilot, formal training or `matched` result is claimed. [Machine-readable checks and scope](../../evidence/checks/h05-h06-20261026/validation.json).

## Record 047

**Hybrid rebase and Gym interface adaptation, 2026-10-27: bounded validation passed.** H01/H04/H05/H06 and the existing controls are replayed on main `1a47b0b`, preserving main's GSA removal. The public Gym adapter now uses `has_sparse_burnin` and validates supplied Hybrid layouts: H05 retains 512 burn-in and 46,363 joint updates; its S scale is the first 512 updates. All 33 registered models retain their configuration/Protocol hashes, parameters and every audited operator count in the [new proof set](../../evidence/costs/hybrid-main-20261027/summary.json). No fixed budget or tolerance changes.

Validation: 553 CPU passes / 58 CUDA-only skips, 95 Gym passes including exact full-size public/generic audit agreement for all six Hybrids, bitwise initialization/logits/loss/gradient/optimizer-ownership equivalence for 33 models across 38 stage cases, and an existing H01 checkpoint loaded with identical weights. Result/Arena checks, 26 website tests, type checking and production build pass. [Evidence and limits](../../evidence/checks/hybrid-main-20261027/validation.json). This does not constitute new GPU, multi-rank or formal-training acceptance; frozen runs keep their original snapshots.

## Record 048

**Hybrid PR integration on latest main, 2026-10-28: bounded compatibility checks passed.** Rebase onto `31489f1` preserves main's scoring interfaces and immutable records 041–042; the prior Hybrid records are now 044–047. All 33 configs pass public admission with byte-identical configs/proofs and unchanged science hash `70606cd1…`; no further cost re-audit is needed for this rebase. Validation: 588 CPU passes / 58 CUDA skips, 80 Gym passes (15 unchanged expensive audit cases retain record 047's passing evidence), and 26 website tests plus type check/build. All six Hybrid debug checkpoints load with exact weights/logits through restricted scoring; BPB/CORE backend routing is tested with mocked metrics. An existing 1,403,810,314-parameter H01 checkpoint at step 12 also passes read-only restricted loading with exact weights. Result/Arena checks and documentation links pass. [Evidence and limits](../../evidence/checks/hybrid-pr-20261028/validation.json). No new GPU, distributed, scoring-image parity or formal-run acceptance is claimed. On merge after main's record 043 (H20 Others exports), these Hybrid records moved to 044–048; `evidence/checks/hybrid-pr-20261028/validation.json` cites their earlier numbers 043–046.


## Record 049

**MQA and GQA run records exported for the observatory, 2026-10-28.** The two H800/H200 runs (`f3_mqa`, `f3_gqa`) were trained by the Others branch's eight-rank trainer, which writes one `final_bpb` event per split, `status` in `run_status.json`, no `core_final.json` and a flat provenance. `nexttransformer.export_frozen_8rank` maps those files onto the standard record, requires identical identity fields in every resume segment, and checks the record against the [record 041 audits](../../evidence/checks/others-20261028/README.md). The [MQA](../../results/runs/mqa.json) and [GQA](../../results/runs/gqa.json) records each contain 189 validation points and six complete CORE-22 events; final BPB, every centered task score, cumulative FLOPs and fingerprints equal the audits. The source commit is `8cba0a3` (frozen tree `wt_a96982a_827b56d023`), with the frozen-branch note. With record 043, all four completed Others runs are on the observatory, in its Others family. No training or evaluation was rerun.

## Record 050

**Residual R00 and R03 formal runs complete and published; HC and mHC interrupted, 2026-10-28.** The four-node ModelArts job launched on 2026-10-23 (record 028) from the frozen checkout `asoul/NextT` at commit `4dd6ebb` trained R00–R03 as four independent 8×A800 runs (`runs/train_reference_modelarts.py`, prefix `nextt-v2-residual-20261023`). R00 `residual_standard` finished at 18:59 CST on 2026-10-26 and R03 `attnres` at 15:42 CST on 2026-10-27, each in one segment with microbatch 8 × accumulation 8: 46,875 updates, 49,152,000,000 input tokens, all 189 validations and six full CORE events, no budget anomaly and `final_status=matched`. Cumulative compute is +0.041% (R00) and +0.160% (R03) versus the SSSL anchor, equal to their configuration proofs. Read-only audits of both run directories found no problems, and each tracked run's summary equals its run directory ([evidence](../../evidence/checks/residual-20261028/README.md)). The standard `export_run` wrote the [Standard Residual](../../results/runs/residual_standard.json) and [Block AttnRes](../../results/runs/attnres.json) records; both join the observatory in a Residual family with new model cards and figures.

Final step-46,875 results (periodic val BPB / independent final val BPB / CORE): R00 **0.6740 / 0.6712 / 0.3195**, R03 **0.6630 / 0.6602 / 0.3400**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7911/0.7763/0.7500/0.7223/0.6944 (R00) and 0.7780/0.7648/0.7394/0.7115/0.6834 (R03); CORE was 0.1963/0.2127/0.2456/0.2720/0.2993 and 0.2269/0.2336/0.2678/0.2759/0.3238. Paired against R00, Block AttnRes is −0.0110 BPB and +2.05 CORE. Against the Dense Baseline, R00 is +0.0068 BPB and −1.14 CORE, and R03 −0.0042 BPB and +0.90 CORE; R00 differs from Dense only in replacing nanochat's learned residual/x0 scalars and backout with plain Pre-Norm addition. R03 has the lowest periodic val BPB and the highest CORE of the 27 published runs. All values were finite; the largest gradient norm after update 500 was 2.89 (R00) and 4.92 (R03). These are single-seed observations, not statistical rankings.

Resources: R00 ran at a median 4.32 s/update, theoretical MFU 45.5%, 44.6 GiB peak memory, 56.5 h training (452 GPU-hours) and 3.9 h evaluation; R03 at 5.81 s/update, 33.8%, 49.0 GiB, 76.0 h (608 GPU-hours) and 5.1 h. The reference residual implementations are eager PyTorch without fused kernels, so AttnRes's extra time measures this implementation, not the method.

HC (`hc`, microbatch 4 × accumulation 16) and mHC (`mhc`, 8 × 8) stopped at 18:04 CST on 2026-10-27, at updates 29,300 and 31,750. Within the first updates after mHC's update-31,750 validation, all eight ranks raised `RuntimeError` from `clip_grad_norm_(..., error_if_nonfinite=True)`: the total gradient norm was non-finite, after finite norms of 0.21–0.31 over updates 31,670–31,750 and no OOM, Xid, ECC or NCCL warning. ModelArts then ended the job (`TerminateJob` at 18:04:55) and sent SIGTERM to HC, which had no error of its own. Their newest complete checkpoints are 29,000 and 31,500. Both runs are to resume in their original output directories from the same frozen checkout; the trainer records a changed micro-batch in `micro_batch_history`. Which computation produced the non-finite gradient is not yet known, and the partial results are not published.

## Record 051

**mHC Sinkhorn-Knopp in log space; mHC resumes from update 31,500 across a registered source transition; all configurations re-audited, 2026-10-28.** A replay of the interrupted mHC run (record 050) from checkpoint 31,500 on 8×A800, instrumented at runtime without changing any source file, recorded the Sinkhorn denominators of every sublayer. Over updates 31,500–31,578 the column logit spread reached 180 and the smallest row sum fell to 2.8×10⁻³¹. FP32 `exp` underflows beyond a spread of about 104, so a row whose entries all underflow divides 0 by 0, and a vanishing row sum makes the backward overflow; on such inputs the multiplicative iteration returns NaN outputs and gradients. The replay stopped at update 31,578 without a non-finite norm. Training is not bitwise reproducible across resumes (HC's first resumed logged loss differed from the interrupted run by 0.0013), so the failing update does not replay exactly, but the denominators were already at the edge.

`sinkhorn_knopp` now keeps the iterates as logarithms. Each column or row normalization subtracts a log-sum-exp built from `amax`, `exp`, `sum` and `log`, the operators the cost ledger counts; the detached maximum's gradient cancels. The function is unchanged: it agrees with the multiplicative form in FP64 to 1e-12, and a new test covers rows that underflow under FP32 `exp`, where it matches FP64 to 1e-5 with finite gradients.

For the formal run, branch [`mhc-logsinkhorn-20261028`](https://github.com/Selen-Suyue/NextTransformer/tree/mhc-logsinkhorn-20261028) (`51f4560`, from the frozen `4dd6ebb`) applies the same change and re-audits the mHC and Dense-anchor proofs: mHC costs 4.674363 GFLOPs per input token, +0.042% versus SSSL and 6,072 FLOPs more than its earlier proof; the anchor is unchanged. It also adds `reference.train --source-transition`, which admits a resume across a source change only if every changed fingerprint (science, validation-source and native-source hashes) matches a registered JSON, and records the change in the resume provenance and run state. A GPU check resumed checkpoint 31,500 with the branch on 8×A800 and trained through update 31,770, past the original failure: no non-finite value, logged losses within 0.0002 of the original run at every common update, validation BPB 0.70982 at update 31,750 (original 0.70996), and the same speed and memory. The formal run then resumed in its original output directory on ModelArts (`mhc-resume-31500`); its record will carry two source segments, with the transition registered at update 31,500. HC resumed unchanged from checkpoint 29,000 (`hc-resume-29000`). Both continuations are tracked as new W&B runs with the interrupted runs as parents, because each resumes from a checkpoint earlier than its last logged update.

On main, `nexttransformer/models/residual.py` takes the same change. All 33 configurations were re-audited under PyTorch 2.9.1+cu128 ([proofs](../../evidence/costs/mhc-logsinkhorn-20261028/summary.json)): 32 keep every model and protocol hash, parameter count, stage weight and per-operator count, with only source hashes and audit timings changed, and mHC changes as on the branch. The SSSL anchor stays at 4.67239832544785 GFLOPs per input token. Validation: 517 CPU passes, 131 skips.


## Record 052

**H02 GDN + MHA formal run completed on 8×B300; metadata audit passes, prerequisite acceptance pending, 2026-10-30 (owner @kris).** Run `h02-b300-formal-20261027T0035` trained from frozen `fab8c032e9ebc91cc14933d8e7e0275f6c60224f`, using the original `hybrid-kernels-20261026-release/gdn_hybrid.json` cost proof. All three recorded science/reference/native-source hashes reproduce from that Git tree, although the deployed package recorded `git_commit=null`. The proof is byte-identical to its frozen version; no current-source re-audit is substituted. The run has 1,399,275,962 parameters, 24 independent blocks, `[GDN, GDN, GDN, full causal MHA] × 6`, width 1536, FFN 5760, native VE and one `joint` stage. It uses the registered Qwen3-Next-inspired ratio with this repository's GDN/MHA shell, without Qwen's attention output gate, GQA, partial RoPE, zero-centered RMSNorm or MoE.

The supplied run metadata reports `training_complete` / `matched`: 46,875 updates, 49,152,000,000 input tokens, microbatch 8 × accumulation 8 × eight ranks at context 2048, seed 42, BF16, Muon+AdamW, and the pinned tokenizer/Protocol. All 189 validation files agree with their logged events; all six complete CORE-22 events have the exact task set, centering and aggregates (maximum arithmetic error 2.78e-17). Independent final train/val BPB each uses 20,971,520 tokens. Every logged cumulative token/FLOPs counter matches its update number and the frozen one-stage proof; total training compute is **231,201,171,450,565,312,500 FLOPs**, **4.703799874889431 GFLOPs/token**, **+0.6720649%** versus SSSL, **0%** versus the registered H02 proof, with zero logged budget anomalies. [Audit and reproduction](../../evidence/checks/h02-b300-20261030/README.md).

| Metric (final update 46,875) | H02 | Dense `dense_full` | Pure GDN `gdn` | H02 − Dense | H02 − GDN |
|---|---:|---:|---:|---:|---:|
| Independent final val BPB | 0.6619626273 | 0.6644601776 | 0.6672715641 | −0.0024975503 | −0.0053089368 |
| CORE ×100 | 33.37156882 | 33.09515300 | 30.54997382 | +0.27641582 | +2.82159500 |

H02's independent final **train** BPB is **0.6640584744**; its separate periodic final **val** BPB is **0.6647055545**. The table compares independent final passes, not periodic validation against final passes. These are single-seed observations across different hardware, not statistical superiority claims. The [exported record](../../results/runs/gdn_hybrid.json) preserves every supplied curve/task score, fingerprint and cumulative counter; the [model card](../../results/methods/gdn_hybrid.json) uses the registry's `hybrid` family. The public exporter regenerates the schema-compatible frozen result and supplies the actual audit URL; no manual score construction or new training is involved.

**Timing and resources.** The owner reports start **2026-10-28 18:19:27** and finish **2026-10-30 01:57:20**, recorded as UTC+8: 31h 37m 53s elapsed. The run id is a label, not its actual start. The device reports **NVIDIA B30Z**; **B300** is the owner-supplied label. PyTorch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5 and FLA 0.5.2; GDN uses FLA and MHA uses deterministic `sdpa-noflash`. Logged rank-0 timing after update 10 gives median **2.05965 s/update**, **509,104 input tokens/s**, and maximum logged allocated memory **59.2588 GiB**. Training consumes **27.4034 h / 219.2274 GPU-hours**; evaluation **2.7900 h** and **12,431,315,494,456,880,400 forward FLOPs**, reported separately. Multiplying the owner's elapsed interval by eight gives **253.0511 allocated GPU-hours**, conditional on continuous eight-GPU allocation; the remaining **1.4379 h** is unclassified overhead. No hardware MFU was reported.

**Boundaries and outstanding evidence.** There is one supplied provenance segment, `--resume never`, no resume records, and a final checkpoint completion marker listing the model and eight rank payloads; tensor files were not supplied or rehashed. Original data/tokenizer/evaluation bytes were not supplied for independent content verification. The owner recalls prechecks before transferring the code. Located H02 evidence includes the historical H20 checks (034) and H100 numerical/throughput checks (046); the latter uses fresh weights at schedule offset 512 for three warmup and five measured updates, not a separate 512-update pilot. A qualifying H02 pilot and target-B300 distributed/data/update/fresh-process resume report have not been located. Earlier failed attempts are not established by this single-run package. These gaps remain pending, rather than being inferred from a completed run or another method's tests; final acceptance requires maintainer review. The default result export's `complete=true` and `reportedAudit=matched` preserve the trainer's report, not a new acceptance decision. No weights, private runtime paths, model/runtime changes or website code are submitted. Following the current consolidated roadmap, status is updated in PLAN.md; V2 design documents remain unchanged.

**Submission validation:** 517 CPU tests passed, 131 environment/CUDA-dependent tests skipped; export and Arena checks pass. All 30 website tests, TypeScript checking and production build pass. The regeneration preserves every supplied metric, identity and budget; only publication metadata changes. [Validation evidence](../../evidence/checks/h02-b300-20261030/validation.json). No new GPU check is claimed.

## Record 053

**H03 GDN + SWA formal run completed on 8×B300; metadata audit passes, prerequisite acceptance pending, 2026-10-31 (owner @kris).** Run `h03-b300-fix-20261028T1845` trained from the configuration and cost proof of frozen `fab8c032e9ebc91cc14933d8e7e0275f6c60224f` (tag [`archive/hybrid-source-20261026`](https://github.com/Selen-Suyue/NextTransformer/tree/archive/hybrid-source-20261026)), using the original `hybrid-kernels-20261026-release/gdn_hybrid_swa.json` cost proof. The model hash (`f9fd81f8…`), protocol hash (`f8cd1b56…`), parameter count, and every stage/operator cost figure match `fab8c03` byte-for-byte, and the proof is byte-identical to its frozen version; no current-source re-audit is substituted. The deployed B300 package (`git_commit=null`) recorded cluster-patched software source hashes (`science_hash` `e1b5a94c…`, `reference_source_hash` `9a3aecef…`, `native_source_sha256` `2822cb47…`); the source diff between `fab8c03` and the cluster fix package was not included in the metadata-only archive and is disclosed below. The run has 1,329,681,662 parameters, 22 independent blocks, `[GDN, SWA] × 11`, width 1536, FFN 6528, `sliding_window=2048`, native VE at one-based even layers (all 11 SWA layers and no GDN layer) and one `joint` stage. At the 2048-token training context, the registered 2048-key window (left window 2047 plus the current token) covers every causal key, so SWA layers compute full causal attention and this run evaluates a 1:1 GDN/global-attention alternation at 2K context.

The supplied run metadata reports `training_complete` / `matched`: 46,875 updates, 49,152,000,000 input tokens, microbatch 8 × accumulation 8 × eight ranks at context 2048, seed 42, BF16, Muon+AdamW, and the pinned tokenizer/Protocol. All 189 validation files agree with their logged events; all six complete CORE-22 events have the exact task set, centering and aggregates (maximum arithmetic error 2.78e-17). Independent final train/val BPB each uses 20,971,520 tokens. Every logged cumulative token/FLOPs counter matches its update number and the frozen one-stage proof; total training compute is **228,413,623,152,778,546,875 FLOPs**, **4.647087059586152 GFLOPs/token**, **−0.5417189%** versus SSSL, **0%** versus the registered H03 proof, with zero logged budget anomalies. [Audit and reproduction](../../evidence/checks/h03-b300-20261031/README.md).

| Metric (final update 46,875) | H03 | Dense `dense_full` | Pure GDN `gdn` | H02 `gdn_hybrid` | H03 − Dense | H03 − GDN | H03 − H02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent final val BPB | 0.6629074499 | 0.6644601776 | 0.6672715641 | 0.6619626273 | −0.0015527277 | −0.0043641142 | +0.0009448226 |
| CORE ×100 | 33.66143915 | 33.09515300 | 30.54997382 | 33.37156882 | +0.56628615 | +3.11146533 | +0.28987033 |

H03's independent final **train** BPB is **0.6649872953**; its separate periodic final **val** BPB is **0.6656266908**. At updates 8,000/16,000/24,000/32,000/40,000, periodic val BPB was 0.7794/0.7646/0.7393/0.7131/0.6865 and CORE ×100 was 21.83/21.45/26.41/28.88/31.60. The table compares independent final passes, not periodic validation against final passes. These are single-seed observations across different hardware, not statistical superiority claims. The [exported record](../../results/runs/gdn_hybrid_swa.json) preserves every supplied curve/task score, fingerprint and cumulative counter; the [model card](../../results/methods/gdn_hybrid_swa.json) uses the registry's `hybrid` family. The public exporter generates the schema-compatible frozen result and supplies the actual audit URL; no manual score construction or new training is involved.

**Timing and resources.** The owner reports start **2026-10-30 01:58:03** and finish **2026-10-31 06:40:30**, recorded as UTC+8: 28h 42m 27s elapsed. The run id is a label, not its actual start. The device reports **NVIDIA B30Z**; **B300** is the owner-supplied label. PyTorch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5 and FLA 0.5.2; GDN uses FLA and SWA uses deterministic `sdpa-noflash`. Logged rank-0 timing after update 10 gives median **1.85030 s/update**, **566,705 input tokens/s**, and maximum logged allocated memory **53.1706 GiB**. Training consumes **24.6133 h / 196.9066 GPU-hours**; evaluation **2.3737 h** and **12,283,683,987,055,256,000 forward FLOPs**, reported separately. Multiplying the owner's elapsed interval by eight gives **229.6600 allocated GPU-hours**, conditional on continuous eight-GPU allocation; the remaining **1.7205 h** is unclassified overhead. No hardware MFU was reported.

**Boundaries and outstanding evidence.** There is one supplied provenance segment, `--resume never`, no resume records, and a final checkpoint completion marker listing the model and eight rank payloads; tensor files were not supplied or rehashed. Original data/tokenizer/evaluation bytes were not supplied for independent content verification. Located H03 evidence includes the historical H20 checks (034) and H100 numerical/throughput checks (046); the latter uses fresh weights at schedule offset 512 for three warmup and five measured updates, not a separate 512-update pilot. A qualifying H03 pilot, target-B300 distributed/data/update/fresh-process resume report, and the exact source diff of the B300 cluster fix package against `fab8c03` have not been located. Earlier failed attempts are not established by this single-run package beyond the `fix` label in the run id. These gaps remain pending, rather than being inferred from a completed run or another method's tests; final acceptance requires maintainer review. The default result export's `complete=true` and `reportedAudit=matched` preserve the trainer's report, not a new acceptance decision. No weights, private runtime paths, model/runtime changes or website code are submitted. Following the current consolidated roadmap, status is updated in PLAN.md; V2 design documents remain unchanged.

**Submission validation:** 517 CPU tests passed, 131 environment/CUDA-dependent tests skipped; export and Arena checks pass. All 32 website tests, TypeScript checking and production build pass. The export preserves every supplied metric, identity and budget. [Validation evidence](../../evidence/checks/h03-b300-20261031/validation.json). No new GPU check is claimed.

## Record 054

**Research mode by default, per-method cost fingerprints and SSSL in Others; all 33 configurations re-audited with unchanged counts, 2026-10-31.** Training defaults to research mode: stale or out-of-budget cost evidence (changed source, another PyTorch build, a cost beyond the 1% preflight) is printed and recorded in provenance, checkpoints and run status, and `export_run` refuses a run with any research segment. `--formal` keeps the fail-closed admission, a run keeps its mode on resume, and `launch/koala` and `launch/modelarts.py` pass `--formal`.

Proofs now fingerprint the source each method runs instead of one repository-wide hash ([fingerprint](../../nexttransformer/protocol/fingerprint.py)). Calibration traces the repository files executed while a method's cost is audited and adds the repository modules they import, without following the hubs (`lm.py`, the models package) or the core; the protocol, cost ledger, calibration, fingerprint and optimizer sources are always included. Files are hashed without comments, docstrings or layout, and module-level data tables of model files keep only the entries of the method and its layer mixers; registry families are display metadata and are excluded. The normalized tokens are identical on Python 3.9 to 3.12 for all 74 package files. Admission compares a configuration's recorded fingerprint with the current source; `python -m nexttransformer.protocol.fingerprint` reports every configuration, and a test fails when a published method no longer matches its proof. On a copy of the source, a comment change in `lm.py` and SSSL's registry move changed no fingerprint, an edit of one hybrid's table entry changed only that hybrid, and a code edit in `linear.py` changed GDN but not SSSL, QSA or the MoBA hybrid.

SSSL (`dense`) moves from the baseline family to Others in the registry, its model card, the site, the Gym and the documentation; full-attention `dense_full` is the only baseline. SSSL keeps its historical id, run identities and role as the 4.67239832544785 GFLOPs/input-token cost anchor; its configuration moves to `configs/others/dense.json`, and its model card sorts first among Others.

All 33 configurations were re-audited on daxiao under PyTorch 2.9.1+cu128 (CPU only, 16 processes, about four minutes, reusing every structure): [proofs](../../evidence/costs/method-fingerprints-20261031/summary.json). Against [mhc-logsinkhorn-20261028](../../evidence/costs/mhc-logsinkhorn-20261028/summary.json), every model and protocol hash, parameter count, stage weight, total and operator count is identical; only the source fingerprints are new. In that environment all 33 configurations pass strict admission and the 75 configuration and fingerprint tests pass. Beyond the shared backbone, configuration and model assembly, the audited files are `linear.py` for the twelve linear and linear-hybrid methods, `sparse.py` with its kernels for the five sparse methods and the MoBA hybrid, both for the QSA hybrid, `looped.py` for UT-ACT, Huginn, Ouro and DeepLoop, `residual.py` for HC, mHC and AttnRes, and `others.py` for MLA-NoPE and the KDA/MLA-NoPE hybrid; SSSL, Dense, Fixed Recursive and the standard-residual and MHA-noVE controls run only the shared files. Published results and historical proofs are unchanged. Deferred: carrying a proof forward automatically when a shared-code change leaves every count unchanged.

## Record 055

**Method plugins; all 33 configurations re-audited with unchanged counts and audited files, 2026-10-31.** A new method is now a package under `nexttransformer/methods/<name>/`: a `method.json` declaration read without importing code (display family, options with defaults, debug options, shell overrides, layer multiple) and hooks that build it without editing shared files: `mixer`, `ffn`, `block`, a whole-model subclass of `ReferenceLM`, `frozen` shell parameters, `init`, contiguous training `stages` and an option `check` ([plugin guide](../guide/PLUGINS.md), [interface](../../nexttransformer/models/plugins.py)). The registry appends discovered plugins after the built-in methods in name order; the model rules in `config.py` stay built-in only. `ModelConfig` gains an `options` field that only plugin methods serialize, so built-in model hashes are unchanged. Plugins calibrate on the registered v2 grid; calibration and the trainer charge plugin stages by contiguous update ranges; `calibrate` writes the SSSL anchor proof beside a single-method proof set. A plugin's whole package belongs to its fingerprint, and the plugin interface counts only for methods that run it. The arena scope check now also classifies pull requests that add a plugin: its own files pass, edits to shared or other methods' files need review or are blocked.

Built-in behaviour is unchanged. At debug size on CPU, all 33 built-in methods give bitwise-identical losses, parameters, gradients, parameter names and debug/candidate model hashes before and after the change. With three test plugins registered and built first, every built-in file list and fingerprint is identical to the run without plugins. Calibrating a test plugin from scratch on CPU took 51 s and selected 18 layers and FFN 6016 (+0.280% versus SSSL), with the anchor proof written beside it and no admission finding.

All 33 configurations were re-audited on daxiao under PyTorch 2.9.1+cu128 (CPU only, 16 processes, about four minutes): [proofs](../../evidence/costs/plugin-interface-20261031/summary.json). Against [method-fingerprints-20261031](../../evidence/costs/method-fingerprints-20261031/summary.json), every model and protocol hash, parameter count, stage weight, total, operator count and audited file list is identical. All 33 pass strict admission there, and the configuration, fingerprint and plugin tests pass (84), including the plugin-mode suite in which every published proof stays valid with the test plugins registered. A first audit attempt showed that a function-level import in `config.py` added `plugins.py` to every built-in method's file set, and that a module-level cache in it made fingerprints depend on runtime state; the interface now counts only when executed, the cache moved into `functools.lru_cache`, and empty tables project to one form. Built-in methods stay in `lm.py`; moving them into plugins is optional.

## Record 056

**GSA + QK-norm complete, 2026-10-30.** 49.152B tokens; periodic val BPB **0.685242**,
independent final val BPB **0.682638**, CORE **0.281720**. [Run](../../results/runs/gsa_qknorm.json)
and [cumulative audit](../../evidence/checks/gsa-qknorm-formal.json).

## Record 057

**mHC finished on 2026-10-31.** Four segments form one seed-42 trajectory over 46,875 updates and 49.152B tokens. Periodic validation BPB was **0.661665**, independent final validation BPB **0.658908**, and CORE ×100 **35.3278**. Against Standard Residual this is -0.01229 periodic BPB and +3.38 CORE points; against full-attention Dense it is -0.00551 periodic BPB and +2.23 CORE points. The calibrated mHC FFN width was 6016 (controls: 6144).

| Starting checkpoint | Source | Tracking run | Last logged update before rollback or completion |
|---:|---|---|---:|
| 0 | `4dd6ebb` | `3777cc25` | 31,750 |
| 31,500 | `51f4560` | `3501a113` | 35,500 |
| 35,000 | `51f4560` | `67c82dad` | 45,490 |
| 45,000 | `51f4560` | `08144d8e` | 46,875 |

The first continuation used the [registered numerical change](#record-051). Later continuations retained its source, optimizer recipe and batch layout (8 ranks × microbatch 8 × accumulation 8). The canonical metrics exclude the three superseded tails. Each recorded cumulative charge equals the sum of the original per-update proof through 31,500 and the log-space proof thereafter. The total is 229,754,105,811,196,491,375 FLOPs (+0.04197% versus SSSL), with zero budget anomalies, 189 validation events and six full CORE-22 events. The final tracking summary independently agrees on the step and CORE. [Run](../../results/runs/mhc.json) · [Audit](../../evidence/checks/mhc-formal.json) · [Source and cost registration](../../evidence/registrations/mhc-sinkhorn.json).

## Record 058

**HC finished on 2026-11-01; the Residual family is complete.** Four segments form one seed-42 trajectory over 46,875 updates and 49.152B tokens. Periodic validation BPB was **0.664400**, independent final validation BPB **0.661683**, and CORE ×100 **35.0943**, second of the 32 published runs on CORE (after mHC) and third on periodic BPB. Against Standard Residual this is -0.00956 periodic BPB and +3.14 CORE points; against full-attention Dense it is -0.00277 and +2.00.

| Starting checkpoint | Source | Tracking run | Last logged update before rollback or completion |
|---:|---|---|---:|
| 0 | `4dd6ebb` | `799efb64` | 29,300 |
| 29,000 | `4dd6ebb` | `23a550bc` | 33,170 |
| 33,000 | `4dd6ebb` | `7b41b59e` | 38,580 |
| 38,500 | `4dd6ebb` | `df90fe6d` | 46,875 |

The first segment stopped when ModelArts ended the shared job after mHC's non-finite gradient ([record 050](#record-050)); two later interruptions restored the newest complete checkpoint. Every segment kept the source, configuration, cost proof, optimizer recipe and batch layout (8 ranks × microbatch 4 × accumulation 16). The canonical metrics exclude the three superseded tails (29,010–29,300, 33,010–33,170 and 38,510–38,580), which remain under `replayed_events/`. The cumulative charge is 230,939,827,919,624,578,125 FLOPs (+0.558% versus SSSL), equal to the proof at every logged update, with zero budget anomalies, 189 validation events and six full CORE-22 events. The final tracking summary (`df90fe6d`, synced from offline mode) agrees with the run directory on the step, both validation BPBs and CORE. [Run](../../results/runs/hc.json) · [Audit](../../evidence/checks/hc-formal.json).

Gradient norms were finite throughout but larger and spikier than Standard Residual's: after update 500 the median was 0.90 (Standard Residual 0.25, maximum 2.89). One isolated spike reached 1,824 at update 1,900 and transient spikes of 30–72 occurred around updates 24,400–24,750, with no change in the loss trend. The median rose gradually from 0.65 in the first segment to 4.2 in the last, starting before the second resume. Resources: median 9.73 s/update, theoretical MFU 20.3%, 44.5 GiB peak memory, 127.0 h training (1,016 GPU-hours) and 8.8 h evaluation across the four segments.

Residual family at update 46,875 (seed 42). Δ columns are periodic BPB / CORE points.

| Run | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs R00 | Δ vs B01 | Training GFLOPs / token |
|---|---:|---:|---:|---:|---:|---:|
| Standard Residual (R00) | 0.673959 | 0.671214 | 31.9513 | — | +0.00679 / -1.14 | 4.674298 |
| HC (R01) | 0.664400 | 0.661683 | 35.0943 | -0.00956 / +3.14 | -0.00277 / +2.00 | 4.698483 |
| mHC (R02) | 0.661665 | 0.658908 | 35.3278 | -0.01229 / +3.38 | -0.00551 / +2.23 | 4.674359 |
| Block AttnRes (R03) | 0.662963 | 0.660217 | 33.9971 | -0.01100 / +2.05 | -0.00421 / +0.90 | 4.679857 |
| Dense Baseline (B01) | 0.667172 | 0.664460 | 33.0952 | -0.00679 / +1.14 | — | 4.674662 |

All three residual mechanisms improve on Standard Residual, by 0.0096–0.0123 periodic BPB and 2.05–3.38 CORE points. mHC is best on both; HC is second on CORE and Block AttnRes second on BPB. mHC's FFN is 6016 wide versus 6144 for the others. These are single-seed observations, not statistical rankings.

## Record 059

**Arena v1 Stage 0: NextT-Gym accepted end to end on real data on Koala H20, 2026-10-31 (bounded-validated, unmerged on `gy/arena-v1`).** Under the [arena v1 research plan](../plans/ARENA_V1_RESEARCH.md), the Gym ran on real data for the first time (AR-01). Four S-prefix runs of imported methods went through the Gym MCP tools: 512 updates and 536,870,912 input tokens each (1.09% of a formal run), seed 42, one 8×H20 Koala job per segment. They were `gdn` (`run_000021`), `dense_full` in three planned segments (`run_000022`), `kda` (`run_000023`), and `dense_full` stopped with `koala stop` at update 300 (`run_000024`). The Gym restarted `run_000024` from its update-250 checkpoint in a fresh job.

All four acceptance checks pass ([evidence](../../evidence/checks/arena-v1-stage0-20261031/README.md)):
- **A1, trainer agreement.** The step-1 losses of `gdn` and `kda` equal their formal H20 runs'. The mean |Δ| over updates 300–510 is at most 0.0039, against a limit of 0.015.
- **A2, resume.** Tokens and FLOPs are continuous across segments and across the restart. `run_000024` settles at exactly `run_000022`'s FLOPs; their final dev BPB is 0.887221 and 0.887830.
- **A3, ledger.** Per-update, training, evaluation and settled FLOPs equal the audited formal costs.
- **A4, sealing and recovery.** No official or CORE value reaches the agent, including through imported reference runs, and a fresh server rebuilds its state from the store.

The first attempt (`run_000018`–`run_000020`) failed at the update-0 dev evaluation, on a race between ranks over the dev-shard link. It cost 7.33 platform GPU-hours.

The Gym changes are on the branch:
- the Koala backend now runs end to end (`ef6c696`, `6f2c422`, `9d2fb94`);
- dev links are created on rank 0 only, and a blank Koala state no longer ends a run (`785a65f`);
- the dev shard's SHA256 is pinned (`45301a3`);
- imported official results are sealed, and a failed segment restarts from the latest checkpoint (`93983dd`);
- each dev evaluation is announced once (`e524c1f`).

Stage 0 used 41.57 platform GPU-hours. This is tooling evidence; its dev values do not rank methods.

## Record 060

**Arena v1 Stage 1: the compressed 1% scale only screens, 2026-11-01 (bounded-validated, unmerged on `gy/arena-v1`).** Stage 1 (AR-03, D1) asked whether scale C ranks methods by their published final BPB. C is 512 updates under the formal schedule code compressed to 512 steps: warmup 40, then linear decay to 5% over the last 333 updates. The rule was preregistered in `e4c42f2` and amended in `3a2e383` before any run started.

Thirteen 8×H20 runs (`run_000025`–`run_000037`, bundles from `e4c42f2`) covered eleven imported methods and two repeats of `dense_full`. All pass the recomputed checks ([evidence](../../evidence/checks/arena-v1-stage1-20261101/README.md)):
- **Ranking.** On the eleven rows, C gives Spearman 0.40 and pair accuracy 0.674. The rule needs 0.85 / 0.9; the constant-LR prefix at update 500 gives 0.718 / 0.826.
- **Screening.** To keep the three rows within 0.005 of the best final, C needs a threshold of 0.0301 above `dense_full`, and it keeps `deeploop`, whose final is worst. The prefix needs 0.0261 and rejects `deeploop`.

Under the rule, Stage 2 screens on S with a threshold of 0.030 above `dense_full`. Ranking survivors needs the 2,000-update rung, which meets the rule in the backtest on published curves (0.918 / 0.935).

The repeats share one initialization, because the seed offset does not change an imported method's initialization. Their 0.00009 spread at update 512 is therefore run-to-run non-determinism. Stage 1 used 110.44 platform GPU-hours; arena v1 then stood at 152.0.

## Record 061

**Arena v1 Stage 2: four `attnres` variants beat both anchors at the 2,000-update rung, 2026-11-02 (research evidence, unmerged on `gy/arena-v1`); Stage 3 not started.** The user approved Stage 2 on 96 H20 (twelve concurrent 8-GPU runs). Claude acted as the researcher through the Gym MCP tools and as the operator.

The rules were preregistered in `e06e4e6` before the first job; the tooling is `15d2fca`:
- **Scale M.** The first 2,000 updates of the formal schedule (2,097,152,000 tokens, 4.27%), evaluated on the dev split every 256 updates.
- **Screen.** At update 512, against `dense_full` + 0.030.
- **Decision.** A candidate beats the best anchor if it ends below the lower anchor by more than max(2δ, 0.001).
- **Stop and cap.** Three rounds, two rounds without a winner, or 600 platform GPU-hours.

The anchors ended at 0.80135 (`dense_full`) and 0.80031 (`attnres`). δ is 0.00041, so the line was 0.79931.

Seventeen runs (`run_000038`–`run_000054`) all pass the recomputed checks ([evidence](../../evidence/checks/arena-v1-stage2-20261102/README.md)). No candidate came near the screen.

| Round | Candidate | Change | Dev BPB at 2,000 | vs `attnres` |
|---|---|---|---:|---:|
| 3 | `attnres_full_canon_abcd` | Full AttnRes added to `attnres_canon_abcd` | **0.79651** | **−0.00380** |
| 2 | `attnres_canon_abcd` | Canon-A/B/C/D | 0.79819 | −0.00212 |
| 2 | `attnres_full_canon` | Full AttnRes with Canon-A/C | 0.79849 | −0.00182 |
| 2 | `attnres_full_gate` | Full AttnRes with a head-wise output gate | 0.79930 | −0.00101 |

Candidates that did not win:
- **Round 1.** `attnres_full` (−0.00080) and `attnres_canon` (−0.00059) came closest. The gate, value embeddings on every layer and a deeper layout did not help, and `dense_full_gate` was cancelled while queued.
- **Round 2.** `attnres_gate_canon`, the 8-expert MoE FFN and the 16-expert MoE FFN did not win. The 16-expert MoE first ran out of memory at micro batch 16 (`run_000051`) and was relaunched at micro batch 8 (`run_000053`).

Two limits apply:
- These are single-seed values at a constant-LR prefix, and the gaps may change after the decay of a full run.
- The Gym trainer runs without `torch.compile`, and in it the best candidate takes 11.33 s per update against 6.82 s for `attnres`.

Stage 2 used 549.72 platform GPU-hours in 8-GPU jobs, plus 0.8–1.7 in single-GPU debug pods, over 24.1 h. Arena v1 has used 701.7 platform GPU-hours in 8-GPU jobs.

**State on 2026-11-02.** By the preregistered rule, `attnres_full_canon_abcd` and `attnres_canon_abcd` (Gym attempts `att_000050` and `att_000043`) are the Stage 3 candidates. Stage 3 is not started and needs the user's approval. Its work is to:
- port them into `nexttransformer/models/` with calibration, proofs, tests, a model card and a disclosure card;
- enter them through the arena's two pull requests.

A formal 49.152B run needs its own authorization and quota. The branch commits are local and not pushed. No Koala GPU job is running for arena v1.
