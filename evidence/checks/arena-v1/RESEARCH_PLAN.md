# Arena v1 research plan: 1% screening on Koala

**Status: Stages 0–2 done; Stage 3 awaits approval.** Drafted on 2026-10-30 on the private development branch
`gy/arena-v1`; published here with the edits listed in the [overview](README.md#publication-edits).
- **Stage 0.** The user approved it; its runs finished on 2026-10-31
  ([Stage 0 evidence](../arena-v1-stage0-20261031/README.md)).
- **Stage 1 (D1).** The user approved it on 2026-10-31 on 64 GPUs; its runs finished on 2026-11-01
  ([Stage 1 evidence](../arena-v1-stage1-20261101/README.md)). The compressed 1% scale only
  screens, and no better than the prefix.
- **Stage 2.** The user approved it on 2026-11-01 on 96 GPUs; its rules and the D2–D5 choices are under *Amendments
  before round 1*. Its runs finished on 2026-11-02 ([Stage 2 evidence](../arena-v1-stage2-20261102/README.md)). Four candidates beat the
  best anchor; the best is `attnres_full_canon_abcd`.
- **Not launched.** Stage 3.
This plan runs NextT-Gym for the first time on real data, then uses it for one supervised research episode. It
changes neither the [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/) nor the [arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/). It implements
the approved items AR-01 (real-data acceptance) and AR-03 (proxy subset first) of the
[arena and gym plan](../../../docs/src/content/docs/arena/plan.md). A formal run of a
selected candidate needs a separate authorization.

## 1. Starting state (checked 2026-10-30)

| Component | State | Consequence for v1 |
|---|---|---|
| Gym service, 22 operations | CPU suite passes (96 tests, `main@0d19bff`). The `mcp` package is not installed in `.venv`, so no MCP session has run here | Install `nextt_gym/requirements.txt`; the first MCP session is part of Stage 0 |
| Koala backend (`backends/koala.py`) | `fetch()` raises `NotImplementedError`, and `_refresh()` calls it on every poll, so a Koala run never updates or settles in the store. The default `env_setup` runs `uv sync` in a directory without `pyproject.toml`. Single node only | Fix `fetch` (code change) and the bootstrap (configuration) before any real run; 8 ranks per run |
| Gym trainer | Separate from `nexttransformer.train` (G2 pending); no `torch.compile`; no `--sparse-backend`; Ouro untrainable; micro batch must come from the run spec or the site JSON | Stage 0 compares it with the formal runs; sparse rows stay out of v1 |
| S scale | 512 updates = 1.09% of the formal tokens, the **prefix** of the 46,875-update schedule, so the LR is still at its peak at update 512 | See §2 |
| Baselines | Imported upstream methods carry no dev-split score | Every anchor must run at the research scale |
| Data | `shard_06541.parquet` (dev) and `shard_06542.parquet` are on the cluster's shared storage next to the training data; the dev shard's SHA256 is not pinned (A1) | Pin it in Stage 0 |
| Isolation (G3/P6) | Not accepted. The trainer's `dev_data` directory symlinks the validation shard, and training processes can read it | Trusted agent only, operator-supervised; not an unattended L0 run |
| Koala | Enough GPU quota for the research; each job is limited to 48 h; the authorization had a fixed expiry | Research fits before expiry; a formal run likely needs renewal |

## 2. What a 1% run can tell us

Published `reference-v1` curves give the prefix variant for free: validation BPB at update 500 against final BPB.

| Rows | Update 500 (1.1%): Spearman / pair accuracy | Update 2,000 (4.3%) |
|---|---|---|
| All 21 finished rows | 0.45 / 0.72 | 0.72 / 0.86 |
| Without the staged sparse rows (QSA, Seer burn-in) | 0.67 / 0.84 | 0.84 / 0.93 |
| The nine Stage 1 rows below | 0.72 / 0.89 (27 pairs) | 0.87 / 0.93 |

Pair accuracy counts pairs whose final gap exceeds 0.005 BPB. The best four rows (AttnRes 0.6602, KDA 0.6644,
Dense 0.6645, NSA 0.6651) finish within 0.005, below the resolution of any early signal. A constant-LR prefix can
therefore **screen** out clearly worse candidates, but it cannot rank close ones. The decay at the end of training
also reorders close rows, as an earlier internal study found.

A **compressed** 1% run, with the formal schedule shape (warmup 40, last 65% warmdown to 5%) squeezed into 512
updates, measures after a decay. Whether it ranks better has not been tested here. Stage 1 tests it before it is used.

## 3. Flow

```
Stage 0  tooling acceptance (AR-01)          ->  Gym trusted on real data on Koala
Stage 1  proxy qualification (AR-03)         ->  1% is a ranker, or a screen only
Stage 2  search rounds (researcher: Claude)  ->  2-3 finalists by dev BPB
Stage 3  selection and entry                 ->  admission PR; formal run on separate approval
```

### Stage 0: tooling acceptance

Operator work (not research). It finishes before Stage 2 starts and its code is frozen by commit.

1. Code: implement `KoalaBackend.fetch()` through a configurable reader command, because the cluster's shared storage is not mounted on the
   dev host. The reader is `koala exec` on a CPU debug pod and copies only `RESULT_FILES` and `val_*.json`.
   Configuration: an upload wrapper around `koala code publish --json -y`, and a bootstrap that mirrors
   `launch/koala/run.sh` (unpack the whole bundle on `/local-ssd`, `uv sync --frozen` plus `requirements/linear.txt`,
   `NANOCHAT_DTYPE=bfloat16`, caches on `/local-ssd`). Add a site JSON with `micro_batch`. Pin the dev shard's SHA256.
2. One single-GPU debug check: the bundle unpacks, resources verify, the dev evaluation runs, and GPU utilization is
   measured by the team rule (median ≥ 80%, or a recorded reason).
3. Three S-prefix runs on 8 GPUs each through the MCP tools: `dense_full`, `gdn`, `kda`. Preregistered checks:
   - **A1, trainer agreement:** `gdn` and `kda` training loss against the first 512 updates of their formal
     runs (same global rows under the eight-stream layout): step-1 loss within 1e-3 and mean |Δloss| over updates
     300–510 at most 0.015 (record 041's C3′ bound). This checks the initialization, data order, loss and optimizer
     equivalence that G2a asks for, at the 512-update horizon only.
   - **A2, resume:** cancel `dense_full` near update 256, relaunch it in a fresh job, and confirm that it finishes
     with continuous tokens, budget and loss.
   - **A3, ledger:** the settled FLOPs equal 512 × the audited per-update cost plus the evaluated tokens × the forward
     cost; the GPU-hours are recorded.
   - **A4, sealing and recovery:** no official-split value appears in agent-visible outputs. A new session recovers
     the state from the store alone.

Failure stops the plan at Stage 0 with a written record; it does not lower a threshold.

**Result (2026-10-31).** A1–A4 pass; no threshold was changed.
- A2 passes on planned 0.4 h segment boundaries (`run_000022`). It also passes after an external `koala stop` at
  update 300 (`run_000024`): the Gym restarted the run from its update-250 checkpoint in a fresh job, a mechanism
  added at the operator's request (`max_restarts`). `cancel_run` stays terminal, so the stop replaced the planned
  cancel.
- The two `dense_full` runs differ by 0.0006 in final dev BPB and by up to 0.0027 at intermediate evaluations. This
  is GPU non-determinism at one seed.
- A4 covers the imported reference runs as well as agent runs, as the operator decided.
- An S run used 8.3–8.9 platform GPU-hours. The first attempt failed on a dev-link race that the synthetic debug
  check (item 2) did not exercise; it was fixed before the reruns.
- A defect in `run.eval` events was fixed.

See the [evidence](../arena-v1-stage0-20261031/README.md).

### Stage 1: proxy qualification

Add a compressed 512-update scale to the Gym (`prefix_of=None`, so the trainer's `Schedule` uses 512 steps). Stages
scale proportionally, as `upstream_model._scaled` already does. Run nine rows with known finals that the Gym can
train: `dense_full`, `attnres`, `residual_standard`, `kda`, `gdn`, `retnet`, `elu_qknorm`, `fixed_recursive`,
`huginn`. Add two seed offsets of `dense_full` for the noise floor.

Preregistered rule: the compressed scale **ranks** if Spearman ≥ 0.85 and pair accuracy ≥ 0.9 against the final BPB
(the prefix gives 0.72 / 0.89 on the same rows); otherwise it only **screens**. With nine rows the estimate is
coarse; report it with that caveat. Published finals are used only to qualify the proxy, never to choose a candidate.

**Amendments before launch (2026-10-31).** The user approved exploring D1 on 64 GPUs (eight concurrent 8-GPU runs).
These amendments were fixed before any Stage 1 run.
- *Scale.* Scale `C` in `nextt_gym/task.py` keeps the formal schedule code over 512 updates:
  - warmup 40, then the last 333 updates decay linearly to 5%, and the Muon weight decay follows a cosine;
  - the Muon momentum ramp keeps its fixed 400 updates, so it overlaps the decay, and momentum drops from 0.970 to
    0.924 at update 400;
  - every candidate gets the same schedule.
- *Rows.* Thirteen rows, the nine above plus `gla`, `deltanet`, `ut_act` and `deeploop`. They now also have
  finals under the same protocol, a matched audit within 1% of the anchor, and Gym attempts. All thirteen are
  within the FLOPs tolerance at C. The rule is decided on the thirteen rows (62 pairs with a final gap above 0.005)
  and also reported on the nine.
- *Comparator.* Published validation BPB at update 500 gives 0.703 / 0.823 on the thirteen rows and 0.717 / 0.889
  on the nine. The backtest script is committed with the Stage 1 evidence.
- *Measure.* Each row's dev BPB at update 512 of its C run, seed 42.
- *Uncertainty.* A 90% bootstrap interval over rows for Spearman and pair accuracy, and the share of bootstrap
  samples in which C beats the prefix on the same rows.
- *Noise.* The spread of three `dense_full` C runs (seed offsets 0, 1 and 2). The screen threshold for Stage 2 is
  derived from it and from the backtest after Stage 1, before any candidate runs. Stage 1 reports how many of the
  rows within 0.005 of the best final the threshold would reject.
- *Prefix.* The prefix S scale stays as implemented. If C neither ranks nor screens better than the prefix, Stage 2
  screens on S.
- *Rows, second amendment (21:40; the first eight runs were queued, none had started).*
  - `gla` and `deltanet` are dropped. Their configs on this branch train on the PyTorch reference recurrence
    (`backend: reference`), which is not the kernel path behind their published finals; it also ran out of memory
    on one GPU at micro batch 16.
  - The rule is decided on the remaining eleven rows (46 pairs), where the prefix at update 500 gives 0.718 / 0.826,
    and is also reported on the nine.
  - `elu_qknorm` needs micro batch 8 on this hardware for its FP32 chunk path; that changes memory only and is recorded with
    its run.
  - The single-GPU debug check passed for the eleven rows: median GPU utilization 99.5–100% and peak memory at most
    89 GB (`elu_qknorm` at micro batch 8).

**Result (2026-11-01).** C does not rank, and it screens no better than the prefix. Under the rule above, Stage 2
screens on S. No threshold was changed.
- **Ranking.** On the eleven rows, C gives Spearman 0.40 and pair accuracy 0.674. The prefix at update 500 gives
  0.718 / 0.826, and C beats it in 14% (Spearman) and 0% (pair accuracy) of bootstrap draws. On the nine rows, C gives
  0.717 / 0.852 against the prefix's 0.717 / 0.889.
- **Misranked rows.**
  - `deeploop` ranks third at C and last at the final.
  - `kda` and `gdn` rank below `fixed_recursive` and `ut_act`, though they finish 0.008–0.022 ahead.
  - The likely cause: the short decay helps the looped rows most.
- **Screening.** Keeping the three rows within 0.005 of the best final needs a threshold of 0.0301 above
  `dense_full` at C, which keeps eight rows including `deeploop`. The prefix needs 0.0261, keeps seven rows and
  rejects `deeploop`.
- **Noise.** The three `dense_full` runs differ by 0.00009 at update 512. The seed offsets do not change an imported method's initialization, so this is run-to-run non-determinism (corrected 2026-11-01).
- **Proposed Stage 2 threshold:** 0.030 above `dense_full` on S. It comes from the update-500 backtest, the Stage 0 S
  runs and their 0.0006 noise. It is fixed if Stage 2 is approved, before any candidate runs.
- **Ranking survivors.** S only screens, so ranking survivors needs the 2,000-update rung (D2). That rung meets the
  rule in the backtest on the eleven rows (0.918 / 0.935) but has not been run in the Gym.
- **Compute.** 110.44 platform GPU-hours in thirteen 8-GPU jobs, mostly spent waiting in the queue: 17 h from the
  first launch to the last result.

### Stage 2: search rounds

- **Researcher:** Claude through the Gym MCP tools, following `research_iteration`. The researcher cannot edit
  `nextt_gym/`, the protocol, or the anchors' runs.
- **Round:** up to 6 candidates. Each makes one primary change against a named parent (an imported method or an
  earlier candidate), passes V0–V3, and is calibrated to ±1% at the research scale. All candidates run at the scale
  qualified in Stage 1, with seed 42, on 8 GPUs, under one frozen Gym commit per round.
- **Reject** if the loss or gradients are non-finite, the run diverges, or the dev-BPB gap to the `dense_full`
  anchor exceeds a screen threshold. That threshold is set from the Stage 1 noise and backtest before round 1, and
  never afterwards.
- **Rank** by final dev BPB only if Stage 1 qualified the scale as a ranker; otherwise rank survivors only through
  the optional 4% rung (decision D2).
- **Stop** after 3 rounds, at the compute cap, or after 2 rounds without a candidate beating the best anchor.

**Amendments before round 1 (2026-11-01).** The user approved Stage 2 on 96 GPUs (twelve concurrent 8-GPU runs) and
asked for the shortest wall time. These rules were fixed before any Stage 2 run.
- *Screen.* Stage 1 did not qualify C, so the screen is S. Reject a candidate if any of these holds:
  - its dev BPB at update 512 exceeds the `dense_full` anchor's value at update 512 by more than **0.030**
    (derivation in the [Stage 1 evidence](../arena-v1-stage1-20261101/README.md));
  - its loss or gradients are non-finite;
  - it diverges.
- *Rung (D2).* Survivors are ranked by dev BPB at update 2,000 of the formal schedule. This is the new Gym scale `M`:
  a prefix of L, evaluated every 256 updates. On the published curves this rung meets the ranking rule on the eleven
  Stage 1 rows (0.918 / 0.935).
- *One job per candidate.* Every candidate is launched at M.
  - M evaluates at update 512 like S, so that evaluation is the screen.
  - A candidate that fails the screen is cancelled there; the others continue to update 2,000.
  - This avoids a second queue wait (4–11 h in Stage 1). The cost is that every survivor reaches the rung, instead of
    at most three finalists as D2 recommended.
- *Anchors.* `dense_full` and `attnres` run once at M, in round 1. A second `dense_full` M run (seed offset 1) measures
  run-to-run noise δ at M.
  - The seed offset does not change an imported method's initialization; see the Stage 1 correction.
  - Later rounds reuse these anchors unless the trainer, the data path or the scale code changes.
- *Decision.* A candidate **beats the best anchor** if its dev BPB at update 2,000 is lower than the lower of the two
  anchors' by more than max(2δ, 0.001). The best one or two such candidates go to Stage 3, which needs its own
  approval. If no candidate beats the best anchor, Stage 2 reports a negative result.
- *Rounds.*
  - A round has up to six candidates. They are submitted, verified, debug-checked on one GPU (median utilization ≥ 80%)
    and launched under one Gym commit.
  - Round r+1 may launch once round r's candidates have their update-512 values.
  - The stop rule above applies, counting a round as finished when its last candidate is cancelled or reaches
    update 2,000.
- *Compute (D4).* The cap is 600 platform GPU-hours for Stage 2, including anchors and debug pods. The Gym policy
  allows twelve concurrent runs.
- *Tooling (D3).* Tooling changes (scale `M` and the concurrency limit) are separate commits on this branch, for a
  maintainer-reviewed tooling pull request later.
- *Involvement (D5).* The level is L1: the user approves stage gates and compute.
  - Claude designs, chooses and runs the candidates (researcher) and operates the Gym (operator).
  - The researcher has read the public results in `results/runs/`. Candidates are chosen with dev-split values only.
  - The disclosure card records both facts.

**Result (2026-11-02).** Four candidates beat the best anchor, and Stage 2 stopped after its third round
([Stage 2 evidence](../arena-v1-stage2-20261102/README.md)). No threshold was changed.
- **Anchors and noise.** At update 2,000, `dense_full` ends at 0.80135 and `attnres` at 0.80031. δ is 0.00041, so the
  margin is 0.001 and the line is 0.79931.
- **Round 1.** No candidate won. `attnres_full` (−0.00080 against `attnres`) and `attnres_canon` (−0.00059) came
  closest; the gate, value embeddings on every layer and a deeper layout did not help.
- **Round 2.** Three combinations won: `attnres_canon_abcd` (−0.00212), `attnres_full_canon` (−0.00182) and
  `attnres_full_gate` (−0.00101). The two MoE FFNs and `attnres_gate_canon` did not.
- **Round 3.** `attnres_full_canon_abcd`, Full AttnRes added to `attnres_canon_abcd`, ended at 0.79651 (−0.00380).
- **Screen.** No candidate came near the 0.030 screen; the largest gap at update 512 was +0.0100.
- **Stage 3.** By the rule, `attnres_full_canon_abcd` and `attnres_canon_abcd` go to Stage 3.
  - These are single-seed values at a constant-LR prefix, and the gaps may change after the decay.
  - In the Gym trainer, which runs without `torch.compile`, the best candidate takes 1.66× the step time of `attnres`.
- **Compute.** 549.72 platform GPU-hours in seventeen 8-GPU jobs, plus 0.8–1.7 in single-GPU debug pods. It took
  24.1 h from the first submission to the last result.

### Stage 3: selection and entry

Port the selected one or two candidates into `nexttransformer/models/` with a registry entry, calibration, proofs,
tests, a model card and a disclosure card. This is G1, done by hand in v1. They enter through the
[two pull requests](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/). The formal 49.152B run uses `nexttransformer.train`,
never research weights, and needs its own authorization and quota.

## 4. Controlled variables at the research scale

| Fixed for all candidates and anchors | Allowed to differ |
|---|---|
| Tokenizer, byte table, packing, 2,048 context, data order from the first shard, 1,048,576 tokens per update, 512 updates | Architecture, calibrated to ±1% of 4.67239832544785 GFLOPs/token |
| Muon+AdamW by parameter role; one schedule variant per comparison; seed 42 (offsets only for the noise estimate) | Micro batch (memory only; recorded) |
| 8 GPUs, BF16, one Gym commit and bundle per round; the dev split with the same tokens and steps | Kernels with an audited reference path |

## 5. Roles, disclosure and budget

- **Operator:** the user and Claude acting for them: tooling, launches, quota. **Researcher:** Claude through the
  MCP tools. Human involvement is L1 if people only approve and operate, or L2 if they choose among proposals.
- Record every candidate, run, failure, GPU-hour and intervention in the Gym archive and the disclosure card. Unknown
  LLM token counts are `null`.
- Expected compute: one S run takes about 10–13 GPU-hours, to be measured in Stage 0. Stage 0 needs about 40,
  Stage 1 about 130, and Stage 2 about 150–230 (18 candidates). That totals about 320–400 GPU-hours, roughly half a
  formal run. The optional 4% rung adds about 45 per finalist.

## 6. Decisions requested

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Research schedule: S prefix (as implemented), compressed 1%, or both | Compressed, qualified in Stage 1; the prefix stays as Stage 0's acceptance check |
| D2 | A 4% (2,000-update) rung for finalists if 1% only screens | Yes, for at most 3 finalists |
| D3 | Route for tooling changes (`nextt_gym/` is maintainer tooling) | Separate commits on this branch, then a tooling PR reviewed by the maintainer before the research results are used |
| D4 | Compute cap and concurrency | 400 GPU-hours for Stages 0–2; Gym policy of 4 concurrent runs (32 GPUs) |
| D5 | Human involvement level | L1: the user approves stage gates and does not pick candidates |
