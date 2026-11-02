# NextTransformer: consolidated plan and status

Updated **2026-10-29 (Asia/Hong_Kong)**. Evidence baseline: `main@5cb15f3`, the checked-in experiment records through **051**, current registered methods, and [issue #13](https://github.com/Selen-Suyue/NextTransformer/issues/13).

This is the single index of existing project work: completed deliverables, unfinished acceptance, remaining experiments, and decisions still needed. The design documents retain technical definitions and rationale: [V2](docs/plans/V2.md), [hybrids](docs/plans/V2-linear-hybrids.md), [Arena](docs/plans/ARENA.md), and the [Gym runtime](docs/plans/REPRODUCIBLE_EVAL.md). The scientific rules remain in [PROTOCOL.md](PROTOCOL.md), and Arena rules in [ARENA.md](ARENA.md). The maintainer approved the nine roadmap recommendations AR-01 through AR-07 and Q-01/Q-02 on 2026-10-29; their scope and acceptance are recorded below. This documentation change does not launch experiments, publish the repository, or change either scientific protocol. Newly raised MCP interface proposals are not included in that approval.

Historical reports describe their named source and date. In particular, [PROGRESS-20261028.md](docs/records/PROGRESS-20261028.md) is an earlier snapshot, not a second live status index. A recorded launch or resume is not a claim that a remote process is running now.

## Completion criteria

| Status | Meaning |
|---|---|
| **Complete** | The named deliverable has the evidence listed. Completion of code, a bounded check, a published result, and a whole research workstream are separate claims. |
| **Partial** | Some deliverables exist; the remaining work is stated explicitly. |
| **Pending** | Planned work or a decision has no completion evidence yet. This does not prove nobody has started it elsewhere. |
| **Deferred / optional** | Preserve the original trigger or optional scope; it is not an obligation to spend compute now. |
| **Failed; retained** | The attempt ended unsuccessfully and its evidence remains part of the record. Success of a different variant does not erase it. |

A new matched formal result requires documented sources/adaptations, registered semantics and controls, numerical/causality/backend/data/update/resume checks, cost admission, and a complete **49.152B-token / 46,875-update** run with **189 validation events, six full CORE-22 events, and accepted cumulative compute**. Use its final checkpoint and preserve all failures and source transitions. Initial cost error is at most 1%; final cumulative error at most 2%, under the unchanged protocol.

SSSL is the **compute anchor**, 4.67239832544785 GFLOPs/input token; `dense_full` is the accepted **quality baseline**. Historical accounting definitions stay attached to their runs. Single-seed, hardware, rank-count, and resume limitations are disclosures, not an automatic instruction to rerun all experiments.

## Completed foundations

| Deliverable | Status and evidence | Boundary |
|---|---|---|
| Shared model/runtime framework | **Complete:** nanochat shell, common tokenizer/packing, BPB and CORE; looped/sparse/linear implementations including pure GDN; common model API and checkpoints. [Source notes](docs/methods/SOURCES.md) | Shared-shell adaptations, not every paper's complete recipe. |
| Compute audit and admission | **Complete:** operator-level forward/backward/auxiliary/recomputation/optimizer ledger, stage-weighted calibration, fail-closed admission and cumulative budgets. The trainer defaults to research mode (findings recorded, never a result); `--formal` enforces admission and `export_run` publishes only formal runs. [Cost specification](docs/protocol/COSTS.md) | Each changed method/source still needs its own valid proof for formal runs. |
| Current registered proof set | **Complete:** 33 configurations re-audited at record 055 after the method-plugin interface, every count unchanged from records 054 and 051 (where mHC added 6,072 FLOPs/input token); record 054 introduced per-method source fingerprints. A change now invalidates only the proofs of methods whose code it touches; CI fails when a published method's code changed. [Evidence index](evidence/README.md), [record 055](docs/records/EXPERIMENTS.md#record-055) | Registration and proof are not formal training completion. Deferred: carrying a proof forward automatically when shared code changes without changing any count. |
| Method plugins | **Complete:** a new method is a package under `nexttransformer/methods/` with mixer, FFN, block, whole-model, frozen, init, stages and option hooks; it changes no shared file, its proof covers only its package and the shared core, and CI's scope check accepts a pull request that adds only its own files. Three test plugins exercise every hook. [Plugin guide](docs/guide/PLUGINS.md), [record 055](docs/records/EXPERIMENTS.md#record-055) | Built-in methods stay in `lm.py`; moving them into plugins is optional and would need bitwise checks and a re-audit. |
| Data layouts and bounded recovery validation | **Complete within tested scope:** eight-stream 8/16/32-rank layouts and bounded multi-node segmented-resume checks. [Records 029–031](docs/records/EXPERIMENTS.md#record-029) | Earlier H20 formal segments were rounding-level consistent; actual Residual continuations are not bitwise identical (records 037, 051). Do not generalize bounded results to every run. |
| Repository and command-line integration | **Complete:** 2026-10-24 reorganization, `nexttransformer` package, unified reference trainer, evaluation/generation/export, tokenizer verification, CI and migration checks. [Repository guide](docs/guide/REPOSITORY.md) | Frozen experiments continue from their original sources. |
| V2-0 baseline identity and website gate | **Complete:** historical `dense` displayed as SSSL; independent `dense_full`; explicit baseline acceptance before Dense-relative claims. [Record 038](docs/records/EXPERIMENTS.md#record-038) | No substitution of SSSL measurements for full Dense. |

## Evidence and remaining limits

### Baseline and existing families

| Row / deliverable | Status | Completed evidence | Remaining work or retained limitation |
|---|---|---|---|
| B01 `dense_full` (owner @selen) | **Complete** | Implementation, calibration, bounded A800 checks, formal training, independent audit, export and accepted website baseline; cumulative +0.048%. [Records 028](docs/records/EXPERIMENTS.md#record-028), [038](docs/records/EXPERIMENTS.md#record-038) | Single seed; pilot skipped by the owner's decision. Full-layout looped controls are a separate item below. |
| Looped: UT-ACT, Fixed Recursive, Huginn, Ouro, DeepLoop | **Partial** | Five H800 runs reported complete training and scheduled evaluations, with curves and trainer-reported matched costs. [Run records](results/runs) | Independent read-only audits of the original run directories. All five exported records retain `complete: false` until evidence supports acceptance. DeepLoop controls are separate below. |
| Sparse: DSA, QSA, NSA, Seer, MoBA | **Partial** | Five H800 runs reported complete training and scheduled evaluations, with curves and trainer-reported matched costs. [Run records](results/runs) | Independent run-directory audits; actual backend/density disclosure and teacher/indexer/stage verification. All five exported records retain `complete: false`. |
| H20 linear: KDA v2, RetNet, A06″ `elu_qknorm`, GDN | **Complete** | Full training, scheduled evaluations, cumulative acceptance and independent read-only evidence. [Records 030](docs/records/EXPERIMENTS.md#record-030), [035–037](docs/records/EXPERIMENTS.md#record-035) | Preserve single seed, differing ranks, earlier failures and numerical/resume limitations. |
| DeltaNet and GLA (Xiangyue's frozen runs) | **Complete** | Training, evaluations, audits, export, standard final/replay rankings and Dense quality comparisons. [Record 039](docs/records/EXPERIMENTS.md#record-039) | Keep their frozen all-backends-replay accounting; do not relabel it as current accounting. |
| GSA + QK-norm | **Complete** | [Run and cumulative audit](results/runs/gsa_qknorm.json); periodic val BPB 0.68524, CORE 28.172. | Single seed; frozen-source run. |
| Original `elu_linear` / `elu_norm` attempts | **Failed; retained** | Failures are recorded; the successful q/k-normalized A06″ has its own identity. [Record 036](docs/records/EXPERIMENTS.md#record-036) and its referenced attempts | No existing commitment to rerun both failed variants to success. |
| Historical native SSSL, MoR, LoopFormer, Circle | **Complete as historical results; new-protocol admission conditional** | Original implementations, run identities, curves and matrix-cost evidence preserved. [Native history](docs/history/NATIVE_D24.md) | If entered into the new strict matched comparison, supply independent full-cost and protocol evidence. Do not convert old estimates by relabeling. |

### Residual family

Owner of R00–R03: **@selen**. Definitions and paired-control policy remain in [V2 §4](docs/plans/V2.md#4-residual-a-separate-family).

| Row | Status | Completed evidence | Remaining work |
|---|---|---|---|
| R00 `residual_standard` | **Complete** | Full training, 189 val / six CORE events, matched +0.041%, independent audit, model card and website publication. [Record 050](docs/records/EXPERIMENTS.md#record-050) | Full-family comparison in [record 058](docs/records/EXPERIMENTS.md#record-058). |
| R03 `attnres` | **Complete** | Full training and evaluation, matched +0.160%, independent audit and publication; R00/Dense differences recorded. [Record 050](docs/records/EXPERIMENTS.md#record-050) | Full-family comparison in [record 058](docs/records/EXPERIMENTS.md#record-058); `attnres_full` is optional, not required for this completed row. |
| R01 `hc` | **Complete** | [Record 058](docs/records/EXPERIMENTS.md#record-058) covers all four segments, the cumulative audit (+0.558%) and the full-family comparison. | Single seed. |
| R02 `mhc` | **Complete** | [Record 057](docs/records/EXPERIMENTS.md#record-057) covers all four segments, the registered source change, paired results, and the cumulative audit (+0.042%). | Single seed. |

The resumes are recorded events, not final completion or live-process status. Record 051 explicitly observes non-bitwise continuation; bounded replay evidence does not remove that limitation.

### Hybrid family and its controls

All six hybrid methods are implemented and registered in [the registry](nexttransformer/models/registry.py). Exact layouts, source choices and acceptance remain in [V2 §3](docs/plans/V2.md#3-hybrids-with-registered-source-evidence) and the [hybrid design](docs/plans/V2-linear-hybrids.md). Model code or a cost proof does not substitute for a formal result.

| Row | Status / existing owner | Completed evidence | Remaining work |
|---|---|---|---|
| H01 `deltanet_hybrid` | **Partial**; @kris / @Xiangnan-Wu | Shared implementation, independent full-graph calibration, CPU checks and bounded full-size H100 BF16/FLA checks. [Record 044](docs/records/EXPERIMENTS.md#record-044) | Current-source eight-rank complete update and real-data fresh-process resume; target-environment acceptance, pilot/formal work and final audit. Preserve any older-source run with its original proofs. |
| H02 `gdn_hybrid` | **Formal run published; acceptance review pending**; @kris | Frozen `fab8c03` (preserved by tag [`archive/hybrid-source-20261026`](https://github.com/Selen-Suyue/NextTransformer/tree/archive/hybrid-source-20261026)), 8×B300 (device reports B30Z): 46,875 updates, 49.152B tokens, 189 val / six CORE-22 events; read-only metadata audit passes at +0.672% compute. Export and model card in [record 052](docs/records/EXPERIMENTS.md#record-052). | Locate H02 pilot and target-cluster distributed/data/update/fresh-process resume evidence; maintainer acceptance. Existing H100 timing at schedule offset 512 is not a pilot. |
| H03 `gdn_hybrid_swa` | **Formal run published; acceptance review pending**; @kris | Frozen base `fab8c03` (tag [`archive/hybrid-source-20261026`](https://github.com/Selen-Suyue/NextTransformer/tree/archive/hybrid-source-20261026)) with B300 cluster fix package, 8×B300 (device reports B30Z): 46,875 updates, 49.152B tokens, 189 val / six CORE-22 events; read-only metadata audit passes at −0.542% compute; registered window 2048 spans the full 2K context (1:1 GDN/global-attention alternation). Export and model card in [record 053](docs/records/EXPERIMENTS.md#record-053). | Locate H03 pilot, target-cluster distributed/data/update/fresh-process resume evidence, and cluster fix source diff; maintainer acceptance. Existing H100 timing at schedule offset 512 is not a pilot. |
| H04 `kda_hybrid` | **Partial**; @kris / @Xiangnan-Wu | Implementation/calibration, real MLA-NoPE, full-size checks, eight-rank complete update and real-data bitwise fresh-process resume. [Record 045](docs/records/EXPERIMENTS.md#record-045) | Stability pilot, full formal training/evaluation and cumulative acceptance; any different formal hardware/layout requires its applicable checks. |
| H05 `qsa_hybrid` | **Partial**; @kris / @Xiangnan-Wu | Implementation/calibration, BF16 checks, four-rank global-batch update and real-data bitwise resume. [Record 046](docs/records/EXPERIMENTS.md#record-046) | Registered eight-stream update/resume acceptance, actual 512-step stage transition, pilot, formal training/evaluation and cumulative acceptance. |
| H06 `moba_hybrid` | **Partial**; @kris / @Xiangnan-Wu | Implementation/calibration, BF16 checks, four-rank global-batch update and real-data bitwise resume. [Record 046](docs/records/EXPERIMENTS.md#record-046) | Registered eight-stream update/resume acceptance, pilot, formal training/evaluation and cumulative acceptance. |
| `kda_no_ve`, `mla_nope`, O00 `mha_no_ve` | **Partial**; delivery requested in issue #13 | Implemented and independently calibrated/admitted, with bounded checks. [Record 045](docs/records/EXPERIMENTS.md#record-045) | Separate formal runs, final audits, paired comparisons and model/result cards. O00 is shared with Others, not an extra duplicate run. |
| Formal hybrid publication | **H02 and H03 published; other rows pending** | H02 export, model card, cumulative metadata audit and paired results merged in PR #14: [record 052](docs/records/EXPERIMENTS.md#record-052). H03 export, model card, cumulative metadata audit and paired results merged in PR #15: [record 053](docs/records/EXPERIMENTS.md#record-053). The README table and the site's Hybrid family include both. [Issue #13](https://github.com/Selen-Suyue/NextTransformer/issues/13) defines per-row delivery. | Resolve H02/H03 prerequisite evidence and acceptance; publish other completed rows independently. Maintainer owns website presentation and summary regeneration. |

Hybrid decisions **D1** (H04 uses O03-N MLA-NoPE), **D2** (H01 uses `chunk_delta_rule`) and **D5** (existing common trainer) are **resolved**. Remaining formal-run owners/hardware allocations (D4) must be registered. A smaller-window H03 or Kimi-faithful/QK-normalized H04 is a separately named optional adaptation, not an implicit change to the current rows.

### Others: completed results versus mainline support

| Deliverable | Status | Completed evidence | Remaining work |
|---|---|---|---|
| Frozen-branch MQA, GQA, `mla_qknorm`, `mla_nope_qknorm` results | **Complete** | Four formal runs, cumulative audits, model cards, full curves and publication under Others. Run owners: @xiongxuyuan / @geyi. [Records 041](docs/records/EXPERIMENTS.md#record-041), [043](docs/records/EXPERIMENTS.md#record-043), [049](docs/records/EXPERIMENTS.md#record-049) | No duplicate result-export task. Mainline implementation is separate below. Earlier failed MLA attempts remain in their frozen evidence; no rerun or republication is implied. |
| Mainline integration of those four methods | **Partial** | Reusable MLA components exist for H04; current Others registrations are SSSL (`dense`, reclassified from Baseline at record 054) and the controls `mla_nope` and `mha_no_ve`. [Registry](nexttransformer/models/registry.py), [implementation](nexttransformer/models/others.py) | Port/register the frozen methods, fresh cost admission and applicable numerical/backend/update/resume checks. Integration owner still to be assigned; old run ownership alone does not assign it. |
| O00 noVE paired comparison | **Partial** | Control implemented and admitted. | The shared O00 formal run above, then separate VE-policy effects from attention/KV changes. |
| MLA decoding/cache evidence | **Pending** | Explicit reconstructed-attention components are present. | Supported projection-absorption/latent-cache interfaces, equivalence checks and measurements before claiming measured decoding or cache savings. [V2 §5](docs/plans/V2.md#5-others-mqa-gqa-and-mla) |

### Scientific controls and interpretation

| Existing item | Status | Remaining work / scope |
|---|---|---|
| Full-layout looped controls | **Pending** | Separate recurrence from the historical SSSL/window policy; register and independently admit each control. |
| DeepLoop controls | **Pending** | Nonshared same-normalization and shared unscaled controls. |
| Residual paired interpretation | **Complete** | All four residual variants are reported against R00 and B01 in [record 058](docs/records/EXPERIMENTS.md#record-058), including mHC's calibrated narrower FFN. |
| Hybrid paired interpretation | **H02 and H03 reported; acceptance pending** | H02 vs GDN and B01: [record 052](docs/records/EXPERIMENTS.md#record-052). H03 vs H02, GDN and B01: [record 053](docs/records/EXPERIMENTS.md#record-053). Remaining pairs: H01 vs DeltaNet; H05 vs H02 and QSA; H04 vs its noVE components; all vs B01. [Comparison definitions](docs/plans/V2-linear-hybrids.md#8-paired-comparisons-this-enables) |
| Others paired interpretation | **Partial** | Frozen results are published; add O00 comparison and separate VE, ranks, KV projections and calibration changes. |
| Optional extra ablations | **Deferred / optional** | `attnres_full`, a separately named smaller-window H03, and alternative H04 identities require separate decisions/configurations. |
| Statistical and historical comparability | **Retained limitation** | Current runs are single-seed and use different hardware/ranks/accounting histories. This plan does not mandate all-seed, all-hardware or historical-native reruns. |

## Arena

This section consolidates the Arena, Gym and fixed-runtime work without counting **G3 and P6** twice. Ordinary training, evaluation and pull-request contributions, including Arena PR entries, do **not** require Docker. Current Arena research is unrestricted in budget but disclosed; a controlled-budget comparison belongs to future G4.

### Delivered components and remaining Arena work

| ID | Status | Already delivered | Remaining work / dependency | Existing owner |
|---|---|---|---|---|
| A0 — initial Arena | **Partial** | Draft rules/task, disclosure format/template/checker, PR scope CI, Arena page, provenance badges/filter and Circle Loop example. | Team review/finalization of protocol and task text. Circle stays **legacy L3**, not a ranked L0 example. | @Selen-Suyue |
| A1 - research tooling | **Partial** | Gym dev scoring, S/L research runs, FLOPs ledger, archive and interfaces. | Generate cards with G1; complete the trusted real-data end-to-end milestone AR-01. The dev-shard SHA256 is pinned on `gy/arena-v1`. | Unassigned |
| A2 - S proxy fidelity | **Partial; first stage bounded-validated on `gy/arena-v1`, unmerged (AR-03; [Stage 1 evidence](evidence/checks/arena-v1-stage1-20261101/README.md); Stage 2 search at the 2,000-update rung: [Stage 2 evidence](evidence/checks/arena-v1-stage2-20261102/README.md))** | Evaluation question and existing full-run reference results. | Preregister a representative baseline/linear/looped/sparse or staged subset, correlation measures and expansion/longer-prefix triggers; run that first stage before deciding on a full-method sweep. Report small-sample limits. | Unassigned |
| A3 - first L0 entry | **Pending** | Agent task statement exists. | Complete AR-01 and G3 before unattended operation, decide the full-run budget, then execute admission/result flow with no human research input. This is not a prerequisite for opening the ordinary PR track. | Unassigned |
| A4 - public opening | **Partial; PR-first sequence approved (AR-06)** | Local observatory and repository-backed contributions; generic PR template includes Arena disclosure/check items. | Team review, artifact-hosting decision, static Pages build/deployment, public-repository decision and announcement. The ordinary PR track may open before complete Gym/L0 support; disclose Gym limitations. | @Selen-Suyue |
| A5 — reference scaffold | **Deferred (later)** | Gym interfaces for an outer loop. | Minimal reproducible agent loop for comparisons with the scaffold fixed. | Unassigned |
| A6 — verification upgrades | **Deferred; trigger-based** | Trust-based protocol, source/data identities and published evidence framework. | Prefix replay, seed-noise estimate or private held-out only when submission scale, inconsistent claims or close margins justify them. | Unassigned |
| G1 — Gym attempt to entry | **Pending** | Attempts, archive and metered research history. | Produce module/registry/configuration/proofs plus a disclosure card for an admission PR. | Chengxing Xie **proposed** |
| G2 - one training path | **Pending; two acceptance gates approved (AR-02)** | Separate Gym and formal trainers exist. | G2a: initialization/data/loss/optimizer/resume equivalence. G2b: hash-frozen candidate loading through the formal trainer, full evaluation/compute/provenance/export eligibility. Gym L remains research-only until the second gate and scoring-policy decision pass. | Unassigned |
| G3 / P6 - isolation | **Partial overall; integration pending (AR-04/05)** | Static checks, API sealing, candidate runtime and scoring tools. | On one selected supported hardware/launcher path, prove training cannot read reserved data or modify scoring infrastructure; freeze model/checkpoint handoff and independently score it. Include training/update/resume and negative boundary tests. Required before unattended agents. | Chengxing Xie **proposed** |
| G4 - metered track | **Pending; disclosure first (AR-07)** | Server-side FLOPs ledger. | Add metering source, ledger snapshot, failed/verification attempts, GPU-hours, wall time and LLM tokens/API cost (unknown values remain null). Keep descriptive displays until enough comparable episodes support a separately defined researcher-ranking rule. Research remains unrestricted for the ordinary PR track. | Unassigned |

Existing assignments marked proposed are not confirmed commitments. Details and motivations remain in the [Arena design](docs/plans/ARENA.md).

### Gym implementation and validation evidence

| Deliverable | Status | Evidence / remaining boundary |
|---|---|---|
| Research service, verification and accounting | **Complete as implementation** | PR #3 merged; 22 MCP/HTTP/Python operations, blocking evaluation adapter, candidate validation/calibration, durable store/events/idempotency, budget reserve/charge/settle and archive. [Gym README](nextt_gym/README.md) |
| Arena alignment and CI | **Complete as integration** | S defined as research rather than a protocol pilot; dev naming, sealed official fields, protocol seed at L, importer/path fixes, protected Gym scope and CI; 72 Gym tests passed at integration. |
| Agent/GPU smoke | **Complete within smoke scope** | [PR #3 author evidence](https://github.com/Selen-Suyue/NextTransformer/pull/3): four A100 GPUs, six synthetic-token Dense updates with DDP/BF16 and ledger settlement; Claude Code/MCP iterations including parent-to-child. This is not a real-data research experiment. |
| Real-data and cluster acceptance | **Pending; AR-01/05** | First validate a trusted/reviewed candidate with real tokenizer/data, S, fresh-process resume, budget settlement, cross-session recovery and an inspectable archive/disclosure draft. Then accept the selected production path; additional remote backends remain outside the first release. |
| Formal-result eligibility | **Pending; G1/G2** | Gym results do not rank on their own. Current path remains registered/admitted method, `nexttransformer.train`, `export_run` and disclosure. |
| Enforcement boundary | **Partial; G3** | API sealing exists; static checks are not a sandbox. Current training can access official data, and Gym L evaluates CORE only at the end rather than the six formal events. |

### Fixed-runtime subplan for G3

The [runtime design](docs/plans/REPRODUCIBLE_EVAL.md) uses one pinned runtime in separate training/scoring containers. The [scoring guide](environments/eval/SCORING.md) holds the detailed open acceptance checklist.

| ID | Status | Completed | Remaining acceptance |
|---|---|---|---|
| P0 — reference capture | **Partial** | Capture tools and Dense/A800 reference evidence. | Accepted reference scope, source/software/kernel/input/settings identities and unresolved alignment items. |
| P1 - runtime build | **Partial** | Dockerfile/build tooling and a built candidate with inventory. | Select one actually available hardware/launcher combination, define its reference/kernel manifests and support limits, then validate distribution. Broader hardware/kernel support is later work. |
| P2 - direct scoring | **Partial** | Launcher, preflight, restricted checkpoint loading, result validator and comparison tooling. | Validate the selected real GPU launch/isolation path, applicable precision/compiler/cache settings and failure handling. |
| P3 — numerical parity | **Partial** | Dense final checkpoint on 8×A800/SDPA matched reference/training-final BPB, CORE and all 22 task scores bit for bit. [Record 042](docs/records/EXPERIMENTS.md#record-042), [evidence](evidence/checks/scoring-dense-full-a800-20261028/README.md) | This used udocker, not Docker isolation. Other claimed models/kernels/hardware require their own evidence. |
| P4 - release | **Pending** | No supported scoring image published. | Publish immutable identity/manifests and support limits for the selected first path; pass a clean second-machine consumer test. |
| P5 - Harbor | **Partial; deferred beyond the first supported path** | Pinned 0.9.0 generator/controller and CPU transport validation. | Optional real GPU trial and direct/Harbor agreement after the core path works; not a prerequisite for opening Arena. |
| P6 — Gym training/scoring isolation | **Pending integration** | See G3 above; the same item, not another task. | Training/update/resume acceptance and tested model/data/scoring boundary. |

### Existing decisions still open

| Decision | What needs agreement |
|---|---|
| Arena protocol review | Editable surface, dev split, two-PR process and L0–L3 classification clarity. |
| Checkpoint hosting | Organization-hosted or entrant-hosted artifacts and accessible evidence for complete entries. |
| Research execution | Owners/hardware for A1/A2 and AR-01, the first supported Gym hardware/launcher combination, instance operators and budget caps. Broader backends and eventual G4 tiers are later decisions. |
| First L0 funding and opening sequence | The ordinary PR track may open first (AR-06). Choose its publication date and separately decide when to fund A3; a complete L0 example is not required to open the PR track. |
| Gym-scored formal results | If G2 replaces a separate formal run: who operates scoring, and whether its score replaces or must agree with training-final evaluation. Put any new rule in Arena v2; ordinary PR entries are unaffected. |

## Approved roadmap refinements

Approved by the maintainer on **2026-10-29**. These rows refine the existing work above; they are not a second count of A/G/P work. Each future execution still needs its named inputs, resources and scientific configuration. New MCP proposals raised during the documentation review remain outside this approved set.

| Approval ID | Integrated work | Status | Acceptance / scope |
|---|---|---|---|
| AR-01 | A1, before A3: trusted real-data end-to-end milestone | **Partial; bounded-validated on Koala H20, unmerged** ([Stage 0 evidence](evidence/checks/arena-v1-stage0-20261031/README.md): S runs, segment and abrupt-stop resume, settlement, sealing including imported runs, and recovery pass; the disclosure draft remains) | Reviewed candidate, real tokenizer/data, S run, fresh-process resume, budget settlement, cross-session recovery, inspectable archive and disclosure draft. This is bounded acceptance, not a full L0 run. |
| AR-02 | G2a/G2b: one trainer, two gates | **Pending** | First compare initialization, data, losses, optimizer and recovery; then verify frozen-candidate identity, 189 val/six CORE events, cumulative accounting and formal export eligibility. Prefer reuse of the formal trainer. Any new leaderboard rule still belongs to Arena v2. |
| AR-03 | A2: representative subset before full sweep | **First stage bounded-validated on Koala H20, unmerged** ([Stage 1 evidence](evidence/checks/arena-v1-stage1-20261101/README.md): the compressed 512-update scale ranks eleven rows worse than the constant-LR prefix and screens no better, so the 1% rung only screens; ranking needs the 2,000-update rung, which the backtest supports. [Stage 2 evidence](evidence/checks/arena-v1-stage2-20261102/README.md) ran that rung in the Gym for three search rounds: four `attnres` variants beat both anchors by more than 0.001 at update 2,000, single seed, not a formal result) | Select representative families/stages and preregister correlation, expansion and longer-prefix criteria. Expand only from that evidence; do not claim a small subset establishes all-method rankings. |
| AR-04 | G3/P6: enforce and test the real boundary | **Pending** | Training cannot access official data or alter scoring; independent verifier and immutable model/checkpoint handoff; test prohibited access as well as the successful path. Ordinary PR workflows remain container-free. |
| AR-05 | P1-P4: narrow first supported release | **Pending** | Choose one available hardware/launcher path; verify real GPU execution, isolation and a clean consumer. Publish unsupported paths explicitly. Defer Harbor and extra backends/platforms. |
| AR-06 | A4: open the ordinary PR track first | **Partial** | Ordinary contribution tooling exists. Finish protocol/team review, artifact hosting and static publication; opening need not wait for G3/A3. Publishing remains a separate maintainer action. |
| AR-07 | G4: cost disclosure before ranking | **Pending** | Expose measured research compute and provenance, failures/verification, GPU-hours, wall time and LLM/API use; unknown is null. Keep researcher views descriptive until comparable episodes and a ranking rule exist. |
| Q-01 | Recovery guarantee matrix and targeted checks | **Partial** | The [recovery guide](docs/guide/METHODS_FRAMEWORK.md#recovery-guarantees) distinguishes bounded bitwise checks from real numerical continuations. Remaining: targeted checks of actual source/kernel/rank/optimizer paths and explicit guarantee labels. No automatic rerun of historical experiments. |
| Q-02 | Lightweight documentation consistency checks | **Pending** | Automate deterministic checks of registry/config/result/evidence references, counts and paths. Do not infer training completion from code or unchecked checkboxes. Current link/fact review is manual task validation, not a completed permanent CI guard. |

Implementation order: finish the public documentation/contribution path in parallel with scientific run acceptance; validate AR-01 and the first supported isolated path before unattended A3; use A2's first-stage evidence to choose further research spend. G4 starts with disclosure. No broad hardware/seed sweep or optional ablation is implied.

## Observatory and public release

| Deliverable | Status | Completed / remaining |
|---|---|---|
| Current observatory | **Complete as local implementation** | Repository-backed records, curves, CORE views, exact-step leaderboard/replay, diagrams, contribution guidance, Arena view, Others and Residual categories. [Website guide](website/README.md) |
| Existing results publication | **Complete for accepted supplied records** | Dense, R00/AttnRes, four Others and published linear results are already on the site. H800 independent acceptance and historical accounting limits remain explicit. |
| Hybrid presentation | **Partial** | Hybrid family filter, leaderboard and chart colors, and H02/H03 figures and previews are on the site. Add each further card/result as issue #13 deliveries arrive. Maintainer owns website changes and summary regeneration. |
| Static public site and repository opening | **Partial; same work as A4** | Local build/start works; static GitHub Pages build is not configured and repository is private at review. Configure/validate static deployment and obtain publication decision. Earlier Sites deployment is retired. |

## v2 work, in dependency order

Preserve the existing roadmap IDs while using the detailed rows above for acceptance.

| ID | Consolidated status | Next existing deliverable |
|---|---|---|
| V2-0 | **Complete** | Baseline naming/gate delivered. |
| V2-1 | **Complete** | B01 full Dense accepted. |
| V2-2 | **Partial** | Finish H01–H03/H05/H06 applicable checks and formal experiments; allocate owners/hardware for unassigned work. |
| V2-3 | **Partial** | H04/noVE control formal runs; mainline Others integration and O00 paired work. Published frozen Others results stay complete. |
| V2-4 | **Complete** | R00–R03 are complete; the full Residual analysis is in [record 058](docs/records/EXPERIMENTS.md#record-058). |
| V2-5 | **Partial** | Independent H800 audits, remaining normalization/layout controls, method-specific final audits and publication. |

There is no fixed "17 total training runs" commitment. The older hybrid estimate of seven runs / 4,500–6,000 H20 GPU-hours covers its original subset, not this whole consolidated plan. Additional controls, adaptations and formal launches require their own configuration, provenance, resources and authorization.

## Evidence and document responsibilities

- **Current work and status:** this file.
- **Binding scientific / Arena rules:** [PROTOCOL.md](PROTOCOL.md) and [ARENA.md](ARENA.md).
- **Method definitions, provenance and admitted configuration details:** [SOURCES.md](docs/methods/SOURCES.md) and [CONFIGURATIONS.md](docs/methods/CONFIGURATIONS.md); design rationale remains in the linked plan documents.
- **Current cost evidence:** [evidence/README.md](evidence/README.md) and the configurations' referenced proof sets.
- **Operations and contribution format:** [guides](docs/guide/REPOSITORY.md), [CONTRIBUTING.md](CONTRIBUTING.md), component READMEs and interface specifications.
- **Dated evidence:** [EXPERIMENTS.md](docs/records/EXPERIMENTS.md), other records, history and evidence files. Preserve source identities, old failures and immutable results; do not rewrite them into current-status summaries.
