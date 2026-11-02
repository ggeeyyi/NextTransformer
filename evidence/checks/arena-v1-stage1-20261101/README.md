# Arena v1 Stage 1: proxy qualification of the compressed 1% scale on Koala H20 — 2026-11-01

Stage 1 of the [arena v1 research plan](../../../docs/plans/ARENA_V1_RESEARCH.md#stage-1-proxy-qualification)
asked whether a 1% run with its own compressed schedule ranks methods by their final BPB. Scale C trains 512 updates
(536,870,912 input tokens, 1.09% of a formal run) under the formal schedule code over 512 steps: warmup 40, linear
decay to 5% over the last 333 updates, cosine Muon weight decay. The Muon momentum ramp keeps its fixed 400 updates.

- **Runs:** eleven imported methods with published finals, plus two repeats of `dense_full` (seed offsets 1 and 2, which do not change its initialization), one 8×H20 Koala job
  each, launched through the Gym MCP tools.
- **Code:** every bundle comes from commit `e4c42f2` on `gy/arena-v1`.
- **Rule:** preregistered in `e4c42f2` and amended in `3a2e383` (written before any run started).
- **Status:** this is tooling evidence, not a formal result.
- **Data:** the runs evaluate only the dev split `shard_06541.parquet`. Published finals qualify the proxy here; they
  never choose a candidate.

**Outcome: C neither ranks nor screens better than the prefix. Under the preregistered rule, Stage 2 screens on S.**
- **Ranking.** On the eleven decision rows, C reaches Spearman 0.40 and pair accuracy 0.674 (31 of 46 pairs).
  - The rule needs 0.85 / 0.9.
  - The constant-LR prefix at update 500 gives 0.718 / 0.826 on the same rows.
  - Over bootstrap draws of rows, C beats the prefix in 14% of draws on Spearman and in none on pair accuracy.
- **Screening.** Three rows finish within 0.005 of the best final: `attnres`, `kda` and `dense_full`.
  - To keep all three, C needs a threshold of 0.0301 above `dense_full`. That keeps eight rows, including
    `deeploop`, whose final is the worst.
  - The prefix needs 0.0261. It keeps seven rows and rejects `deeploop`.
- **Noise.** The three `dense_full` runs (one initialization; see *Noise*) differ by only 0.00009 at update 512.
- **Run checks.** All thirteen runs pass. None restarted.

## Runs

| Run | Method | Micro batch | Dev BPB at 512 | Published final | s/update | Peak memory | Cost vs anchor | Gym GPU-h | Platform GPU-h | Queue |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_000025` | `dense_full` | 16 | 0.8554 | 0.6645 | 6.05 | 75 GiB | +0.05% | 7.13 | 7.50 | 9.3 h |
| `run_000026` | `gdn` | 16 | 0.8769 | 0.6673 | 6.39 | 108 GiB | +0.32% | 7.50 | 8.29 | 10.1 h |
| `run_000027` | `kda` | 16 | 0.8855 | 0.6644 | 6.95 | 123 GiB | +0.43% | 8.15 | 9.15 | 10.3 h |
| `run_000028` | `dense_full` (offset 1) | 16 | 0.8555 | 0.6645 | 6.05 | 75 GiB | +0.05% | 7.10 | 7.52 | 10.7 h |
| `run_000029` | `dense_full` (offset 2) | 16 | 0.8555 | 0.6645 | 6.04 | 75 GiB | +0.05% | 7.08 | 7.55 | 11.0 h |
| `run_000030` | `huginn` | 16 | 0.8934 | 0.7009 | 8.24 | 74 GiB | +0.67% | 9.11 | 9.51 | 11.1 h |
| `run_000031` | `ut_act` | 16 | 0.8702 | 0.6866 | 7.93 | 72 GiB | −0.26% | 9.32 | 9.59 | 11.3 h |
| `run_000032` | `deeploop` | 16 | 0.8604 | 0.7145 | 7.78 | 73 GiB | −0.25% | 9.10 | 9.49 | 11.4 h |
| `run_000033` | `fixed_recursive` | 16 | 0.8637 | 0.6757 | 7.85 | 72 GiB | −0.20% | 9.20 | 9.61 | 4.8 h |
| `run_000034` | `attnres` | 16 | 0.8517 | 0.6602 | 6.80 | 82 GiB | +0.16% | 7.97 | 8.43 | 3.9 h |
| `run_000035` | `residual_standard` | 16 | 0.8669 | 0.6712 | 5.88 | 73 GiB | +0.04% | 6.91 | 7.26 | 3.7 h |
| `run_000036` | `retnet` | 16 | 0.8944 | 0.6797 | 5.73 | 82 GiB | −0.91% | 6.73 | 7.36 | 3.5 h |
| `run_000037` | `elu_qknorm` | 8 | 0.9475 | 0.6970 | 7.41 | 90 GiB | −0.43% | 8.70 | 9.18 | 3.7 h |

- **Common setup.**
  - Seed 42 unless noted. One segment per run, BF16, 8 ranks.
  - Accumulation 4, except 8 for `elu_qknorm`.
- **Cost vs anchor.** This is the mean audited cost per update against 4.67239832544785 GFLOPs/token. For `huginn`
  it is the mean over its sampled depth stages: 82, 161, 107 and 162 updates at depths 1–4.
- **Platform GPU-hours.** Pod time from admission to the end state, multiplied by 8. This is the same convention as
  Stage 0 ([koala_jobs.json](koala_jobs.json)).
- **Queue.** Time from job creation to admission.

## Checks

[check.py](check.py) recomputes the following for each run, from the files under `runs/`. The output is
[checks.json](checks.json).
- The scale is C, with 512 steps on a 512-step schedule, and the seed is 42 + offset.
- Only the dev split is evaluated.
- Logged updates have continuous tokens and cumulative cost; for `huginn`, the cost follows the stage sequence of its
  Gym submission, which matches every logged stage.
- The run settles exactly at its train plus evaluation FLOPs, within its reservation.
- The mean cost is within 1% of the anchor.
- The final dev BPB is the evaluation at update 512.

All thirteen pass. The settled total is 33,424,300,702,081,761,408 FLOPs.

## Ranking against the published finals

[analyze.py](analyze.py) is the preregistered analysis, and [analysis.json](analysis.json) is its output. It reads
the dev BPB at update 512 of each run's final evaluation. [backtest.py](backtest.py) computes the prefix comparators
from the published curves in `results/runs/` ([backtest.json](backtest.json)).

| Row | C at 512 (rank) | Prefix at 500 | Prefix at 2,000 | Final (rank) |
|---|---|---:|---:|---|
| `attnres` | 0.8517 (1) | 0.9016 | 0.8107 | 0.6602 (1) |
| `dense_full` | 0.8554 (2) | 0.9009 | 0.8125 | 0.6645 (3) |
| `deeploop` | 0.8604 (3) | 0.9479 | 0.9250 | 0.7145 (11) |
| `fixed_recursive` | 0.8637 (4) | 0.9042 | 0.8173 | 0.6757 (6) |
| `residual_standard` | 0.8669 (5) | 0.9166 | 0.8234 | 0.6712 (5) |
| `ut_act` | 0.8702 (6) | 0.9167 | 0.8257 | 0.6866 (8) |
| `gdn` | 0.8769 (7) | 0.9193 | 0.8224 | 0.6673 (4) |
| `kda` | 0.8855 (8) | 0.9270 | 0.8214 | 0.6644 (2) |
| `huginn` | 0.8934 (9) | 0.9652 | 0.8722 | 0.7009 (10) |
| `retnet` | 0.8944 (10) | 0.9387 | 0.8381 | 0.6797 (7) |
| `elu_qknorm` | 0.9475 (11) | 0.9921 | 0.8696 | 0.6970 (9) |

| Proxy | Rows | Spearman | Pair accuracy | Bootstrap 90%: Spearman / pair accuracy |
|---|---:|---:|---:|---|
| C at 512 | eleven (decision) | 0.400 | 0.674 (31/46) | −0.17 to 0.84 / 0.41 to 0.92 |
| Prefix at 500 | eleven | 0.718 | 0.826 (38/46) | |
| C at 512 | nine (original plan) | 0.717 | 0.852 (23/27) | 0.23 to 0.95 / 0.64 to 1.00 |
| Prefix at 500 | nine | 0.717 | 0.889 (24/27) | |
| Prefix at 2,000 (backtest only) | eleven | 0.918 | 0.935 (43/46) | |

Pairs count only when the final gap exceeds 0.005. The bootstrap resamples rows with replacement in 10,000 draws.
Draws with fewer than four distinct rows are skipped, leaving 9,999 on the eleven rows and 9,975 on the nine.

**C misranks in two ways:**
- *Looped rows.* `deeploop` takes 8 of C's 15 wrong pairs. It ranks third at C and last at the final.
- *Linear rows.* `kda` and `gdn` rank below `fixed_recursive` and `ut_act`, although they finish 0.008–0.022 ahead.

**Likely cause: the short decay helps some rows more than others.**
- *Size of the drop.* Going from the published prefix at 500 to C at 512, `deeploop` falls by 0.088 and `huginn` by
  0.072; the other rows fall by 0.040–0.050. The two values come from different splits, but that offset is common to
  every row.
- *Ranking is unaffected.* The final ranking does not reward this sensitivity: the 46,875-update run decays from a
  far later point.

## Screening and the proposed Stage 2 threshold

The Stage 2 reject rule compares each candidate's dev BPB with the `dense_full` anchor. For each proxy,
[screen.py](screen.py) finds the smallest threshold that keeps every row within 0.005 of the best final, and reports
which rows that threshold rejects. Its output is [screen.json](screen.json).

| Proxy | Threshold keeping `attnres`, `kda`, `dense_full` | Kept | Rejected | Worst final kept |
|---|---:|---:|---|---|
| C at 512 | 0.0301 | 8 of 11 | `huginn`, `elu_qknorm`, `retnet` | `deeploop` (0.7145) |
| Prefix at 500 | 0.0261 | 7 of 11 | `deeploop`, `huginn`, `elu_qknorm`, `retnet` | `ut_act` (0.6866) |
| Prefix at 2,000 (backtest only) | 0.0089 | 4 of 11 | the seven others | `fixed_recursive` (0.6757) |

**Proposed S threshold for Stage 2:** reject a candidate whose dev BPB at update 512 of S exceeds `dense_full` by
more than **0.030**.
- **Backtest gap.** In the backtest, `kda` needs 0.0261 at update 500.
- **Stage 0 S runs.** Gym dev values on S agree: `kda` is 0.0234–0.0240 above the two `dense_full` S runs, against
  0.0261 on the published curves, and `gdn` is 0.0186–0.0192, against 0.0184.
- **Noise.** The two `dense_full` S runs differ by 0.0006.
- **Margin.** The remainder up to 0.030 covers this noise and the dev-versus-validation difference.

In the backtest, 0.030 rejects none of the three best rows and the same four rows as the table above. The threshold
is fixed when Stage 2 is approved, before any candidate runs, and is not changed afterwards.

S only screens: it keeps rows that finish up to 0.026 behind the best. Ranking the survivors needs the 2,000-update
prefix rung (decision D2). It meets the rule in the backtest on the eleven rows (0.918 / 0.935), but it has not been
run in the Gym. At the S costs measured here, a 2,000-update run would take about 28–38 platform GPU-hours.

## Noise

The three `dense_full` runs repeat one run: they start from the same initialization and read the same data.
- **Why.** The run seed is 42 + offset, and the trainer seeds the generator with it. But an imported method builds its
  model through `nextt_gym/upstream_model.build`, which reseeds at the protocol seed 42. The offsets therefore change
  no input of these runs.
- **Step 1.** Their step-1 loss (10.39759) and update-0 dev BPB (3.16206) agree.
- **Divergence.** Their losses separate from update 10, through non-deterministic GPU kernels. Their dev BPB spreads
  by 0.0045 at update 128, 0.0008 at 256 and 0.0002 at 384.
- **Update 512.** After the decay the spread is 0.00009 (sd 0.00005). For comparison, the two S runs of Stage 0, at a
  constant LR, differ by 0.0006.
- **What this measures.** The spread is H20 run-to-run non-determinism, not seed-to-seed variation.

*Correction (2026-11-01).* The first version of this record said that the offsets changed the initialization. They do
not, as explained above. The values are unchanged.

## Deviations and notes

- **Rows.** `gla` and `deltanet` were dropped before any Stage 1 run had started (second amendment, `3a2e383`). On
  this branch, their configs train on the PyTorch reference recurrence, which is not the kernel path behind their
  finals, and they ran out of memory at micro batch 16. The rule is decided on the eleven remaining rows.
- **Micro batch.** `elu_qknorm` ran at micro batch 8 through the operator backend `koala-h20-mb8`, because its FP32
  chunk path runs out of memory at 16. This changes memory only.
- **Debug check.** One single-GPU debug pod checked the eight rows that had not run in the Gym before, from 21:06 to
  21:35 on 2026-10-31.
  - Median GPU utilization was 99.5–100%.
  - Peak memory was at most 89 GB, for `elu_qknorm` at micro batch 8.
  - `dense_full`, `gdn` and `kda` had already run at S in Stage 0.
  - The first attempt was killed by the 300 s default limit of `koala exec --detach` and rerun with `--timeout 0`.
  - The platform no longer reports this pod's runtime.
- **Interim look.** After the first eight runs finished, the operator computed the analysis on the six finished
  rows. Six wrong pairs already capped pair accuracy at 40/46 = 0.87. No run, row or rule changed afterwards; the
  remaining five ran as launched.
- **Schedule of launches.**
  - The Gym policy allowed eight concurrent runs (64 H20, as the operator approved).
  - The first eight runs queued for 9.3–11.4 h; the other five were launched as slots freed and queued for
    3.5–4.8 h.
  - The round took 17 h from the first launch (21:30) to the last result (14:23), mostly in the queue.
- **GPU-hours.** The Gym's GPU-hours are 0.3–1.0 lower per run than platform time, because it counts training and
  evaluation only.

## GPU-hours

Stage 1 used 110.44 platform GPU-hours in thirteen 8-GPU jobs, plus one single-GPU debug pod. Together with
Stage 0's 41.57, arena v1 has used 152.0 platform GPU-hours in 8-GPU jobs.

## Files

| File | Content |
|---|---|
| [check.py](check.py), [checks.json](checks.json) | Per-run checks and their output |
| [analyze.py](analyze.py), [analysis.json](analysis.json) | Preregistered analysis (reads `runs/` by default) and its output |
| [screen.py](screen.py), [screen.json](screen.json) | Screen comparison under the Stage 2 reject rule |
| [backtest.py](backtest.py), [backtest.json](backtest.json) | Prefix backtest on the published curves |
| [koala_jobs.json](koala_jobs.json) | Koala job, queue and pod times per run |
| `runs/<run>/` | `metrics.jsonl`, `budget.json`, `run_status.json`, `run_spec.json`, `chain.json` and a reduced `provenance.json` |
| `submissions/huginn/` | Copy of the Gym attempt `att_000020`, used by `check.py` for the depth-stage sequence |

## Reproduce

From the repository root with its environment:

```
.venv/bin/python evidence/checks/arena-v1-stage1-20261101/check.py
.venv/bin/python evidence/checks/arena-v1-stage1-20261101/analyze.py
.venv/bin/python evidence/checks/arena-v1-stage1-20261101/screen.py
.venv/bin/python evidence/checks/arena-v1-stage1-20261101/backtest.py   # also rewrites backtest.json
```
