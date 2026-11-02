# Arena v1 Stage 0: NextT-Gym tooling acceptance on Koala — 2026-10-31

Stage 0 of the [arena v1 research plan](../arena-v1/RESEARCH_PLAN.md#stage-0-tooling-acceptance) ran
four S-prefix runs through the Gym MCP tools: 512 updates and 536,870,912 input tokens each (1.09% of a formal
run), seed 42, on one 8-GPU Koala job per segment. The Gym and bundle commit was `785a65f` on `gy/arena-v1` for
runs 021–023 and `93983dd` for run 024. The runs were imported upstream methods. This is tooling evidence, not a
formal result, and the dev BPB values below do not rank methods.

**Outcome:** A1–A4 pass. No threshold was changed.
- A2 passes twice:
  - On planned segment boundaries (`run_000022`).
  - After an abrupt stop of the Koala job at update 300, from which the Gym restarted the run from its update-250
    checkpoint in a fresh job (`run_000024`).
- The preregistered cancel is replaced by that external stop, because the Gym's `cancel_run` is terminal by design
  (see *Deviations*).
- A4's scope covers the imported reference runs as well as agent runs, as the operator set it on 2026-10-31.
- The checks found a Gym defect in the `run.eval` events, which commit `e524c1f` fixes.

| Run | Method | Backend | Koala jobs | Dev BPB at 512 | s/update | Peak memory | Gym GPU-h | Platform GPU-h |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `run_000021` | `gdn` | `koala` | 1 | 0.906398 | 6.40 | 108 GiB | 7.50 | 8.34 |
| `run_000022` | `dense_full` | `koala-seg` (0.4 h segments) | 3 | 0.887830 | 6.05 | 76 GiB | 7.16 | 8.36 |
| `run_000023` | `kda` | `koala` | 1 | 0.911209 | 6.95 | 123 GiB | 8.14 | 8.92 |
| `run_000024` | `dense_full` | `koala` (`max_restarts = 2`), stopped at 300 | 2 | 0.887221 | 6.05 | 76 GiB | 7.13 | 8.63 |

Every run used microbatch 16 × accumulation 4, BF16, torch 2.9.1+cu128, and the dev shard `shard_06541.parquet`.
Platform GPU-hours are pod time from Running to the end state, multiplied by 8 ([koala_jobs.json](koala_jobs.json)).

## Checks

[check.py](check.py) recomputes A1–A3 from the files in this directory; its output is
[acceptance.json](acceptance.json).

- **A1, trainer agreement (pass).** Training loss is compared with the first 512 updates of the formal runs, on the same GPU model
  ([formal/](formal)), over 52 shared logged updates.

  | Method | Step-1 difference (limit 1e-3) | Mean \|Δ\|, updates 300–510, 22 points (limit 0.015) | Max \|Δ\| |
  |---|---:|---:|---:|
  | `gdn` | 0 | 0.00386 | 0.0165 |
  | `kda` | 0 | 0.00337 | 0.0158 |

- **A2, resume (pass).** Both runs below meet the same conditions:
  - All 53 logged updates are unique and in order.
  - Input tokens equal step × 1,048,576 at every logged update.
  - Cumulative FLOPs equal step × the per-update cost plus the evaluations so far at every logged update.
  - The trainer restores model, optimizer, data-stream state, all RNG, and the cumulative cost and timing state. The
    CPU suite's `test_resume_matches_continuous_run` shows bitwise-equal weights after resume on synthetic data.

  *Planned segments.* `run_000022` ran as three Koala jobs. The first two ended cleanly at the 0.4 h wall limit, at
  updates 230 and 458 (`segment_complete`), and each next job resumed from that checkpoint. The loss shows no
  discontinuity at either boundary: 3.339 / 3.358 / 3.297 at updates 220 / 230 / 240, and 3.026 / 2.994 at updates
  450 / 460.

  *Abrupt stop.* `run_000024` ran on `koala` with `max_restarts = 2` (commit `93983dd`).
  - A watcher script ran `koala stop` on the job at 15:12:30, once update 250's checkpoint was complete and update
    300 had been logged. This is an external stop, not a Gym cancel.
  - The job ended without the trainer's final status. The chain driver recorded the restart
    `{segment 1, failed_at_step 300, resumed_from 250}`; Koala created the fresh job at 15:12:48, 18 s after the stop.
  - `get_run` showed the restart under `extra.restarts`, and `wait_events` delivered `run.restarted`.
  - The new job queued for 4.1 h, resumed with `--resume auto` from update 250 (`resume_00000250.json`) and finished
    at update 512.
  - The trainer archived the stopped job's events after update 250 (`run/replayed_events/`,
    [replayed_metrics.jsonl](runs/run_000024/replayed_metrics.jsonl)).
  - At those six events (the update-256 dev evaluation and updates 260–300), input tokens and cumulative FLOPs equal
    the replay's to the last digit.
  - Values differ only by GPU non-determinism: the update-256 dev BPB by 9e-6, and the loss by 0.0002–0.0062.
  - The loss runs 3.308 / 3.280 / 3.199 at updates 240 / 250 / 260.
  - Against `run_000022`, which has the same method, seed and data:

    | Measure | Difference |
    |---|---:|
    | Update-1 loss | 0 |
    | Mean \|Δloss\|, updates 300–510 | 0.0035 (A1's bound is 0.015) |
    | Max \|Δloss\| | 0.019 |
    | Dev BPB at 128 / 256 / 384 / 512 | +0.0020 / +0.0027 / −0.0006 / −0.0006 |

    This is the noise floor of two equal S runs on this hardware.
- **A3, ledger (pass).** For each run:
  - The per-update cost equals the audited formal cost: `gdn` 4,914,882,389,683,629, `dense_full`
    4,901,738,059,886,504 and `kda` 4,920,383,344,138,811 FLOPs. The `gdn` and `kda` values are update 1 of their
    formal runs; the `dense_full` value is the [formal run's audit](../../audits/dense_full.json)'s total / 46,875.
  - Training FLOPs equal 512 × that cost.
  - Evaluation FLOPs equal the sum of int(forward cost × tokens) over five dev evaluations (41,943,040 tokens).
  - Settled FLOPs are 1/1.02 of the reservation, which includes the 2% cumulative tolerance.
  - The Gym budget sums the four runs to 10,315,594,928,120,901,136 FLOPs, with nothing left reserved.
  - `run_000024` settled at 2,574,741,487,928,393,768 FLOPs, the same as `run_000022` to the last digit. The 50
    updates and one evaluation replayed after the restart stay outside the ledger, because the trainer restores the
    cumulative cost from the checkpoint. That is the formal trajectory convention; `extra.restarts` records the replay.
- **A4, sealing and recovery (pass after `93983dd`).**
  - Agent S runs carry no official or CORE field ([a4.json](a4.json)).
  - `get_run_curve` and `get_leaderboard` with the official split at S return `POLICY_DENIED`.
  - A log search for official, CORE or sealed lines returns nothing.
  - Every MCP call in this check ran in a fresh server process that rebuilt its state from the store.
  - The operator session also ended while runs 018–020 were active. Their chain drivers and the store recorded the
    failures, and a later session read them through `get_run`.
  - Scope. At `f8f5d05` the Gym sealed only agent runs. Imported reference runs, such as `run_000002`, showed
    their published official BPB, official curve and CORE. On 2026-10-31 the operator set A4's scope, tentatively,
    to the historical results (the imported reference runs) as well as the agent's own runs.
  - Commit `93983dd` implements that scope:
    - Every run at a scale with official evaluation is sealed until its attempt is revealed.
    - `latest` also drops the official value.
    - The experiment record (`read_doc("experiments")`, `nextt://upstream/experiments`) and the failure knowledge
      are served without the lines that report results.
  - [a4_sweep.py](a4_sweep.py) calls every agent-visible tool, resource and the prompt over one MCP session: 281
    calls covering 33 attempts and 24 runs. It searches the outputs for 347 published values, which are the final
    BPB, CORE events and last validation points of `results/runs/*.json` to four decimals
    ([a4_sweep.json](a4_sweep.json)).
    - At `f8f5d05`, 83 outputs contained 168 of the values.
    - At `93983dd`, 3 outputs contained 9 of them. All 9 are `grad_norm` values in the agent S runs' training logs
      whose first four decimals coincide with a published value.
  - The sealing covers Gym outputs only. A researcher with repository access can still read `results/` and the
    records: isolation is G3, which is not accepted.

## First attempt and fixes

Runs `run_000018`–`020` (`gdn`, `dense_full`, `kda`) all failed at the step-0 dev evaluation
([failed_runs.json](failed_runs.json)):
- Every rank symlinked the dev shard into the run's `dev_data` directory, and the ranks that lost the race raised
  `FileExistsError`.
- The pods ran for 1,556 s, 786 s and 955 s, which is 7.33 platform GPU-hours. The Gym recorded 0 FLOPs and 0 GPU-hours.

The chain driver also ended chains about 25 s after admission. Koala's `get --json` reports a blank status while it
admits a queued job, and the driver read any status that was not active as an end.

Commit `785a65f` fixed both, with regression tests:
- Rank 0 creates the links, and the other ranks wait at the existing barrier.
- A chain ends only on an ended Koala state. At most 60 unknown polls in a row are tolerated.

Commit `45301a3` then pinned the dev shard's SHA256 (`51da1a9b…`), which plan item 1 asked for. The pinned data
inventory records only names and sizes. The cluster's copy hashed to this value on 2026-10-31, as does the scoring
contract of the Dense reference. The cluster file's modification time (2026-10-19) predates every run here.

## Event defect found during A2

- Since the first Gym commit (`9417261`), the service compared the latest dev evaluation's step with
  `latest.at_step`, which training events also advance.
- Every poll therefore announced the last evaluation again. Runs 021–023 received 46, 179 and 50 `run.eval` events
  for five evaluations each.
- An evaluation first seen after its update's training event was never announced: update 512 of `run_000022` and
  `run_000023`.
- Commit `e524c1f` announces each dev evaluation once, using the stored dev curve, so an evaluation that a restart
  replays is not announced again. Its regression test is `test_each_dev_evaluation_is_announced_once`.
- The monitor was restarted on the fix at 16:03, while `run_000024` waited for its second job. After that, its
  updates 384 and 512 were announced once each, and the replayed update-256 evaluation was not announced.
- Events already in the store were left as written.

## Deviations from the plan

- **Debug check (plan item 2).** The single-GPU debug check measured GPU utilization: the median of non-zero
  samples was 100% for each of `dense_full`, `gdn` and `kda`. It used synthetic data on one rank, so it did not
  exercise the real dev shard or the multi-rank link, and the race above reached the 8-GPU runs. The successful runs
  now exercise both.
- **A2 mechanism.** The plan says to cancel `dense_full` near update 256 and relaunch it in a fresh job.
  - In the Gym, `cancel_run` is terminal by design: an agent's cancel ends the run.
  - At `785a65f`, a Koala segment that failed also ended its run as failed, so A2 first ran with 0.4 h segments
    (`run_000022`).
  - On 2026-10-31 the operator asked for restarts from the latest checkpoint after a failure.
  - Commit `93983dd` added them for backends that support them (Koala), behind the backend option `max_restarts`
    (default 0), at most once from each checkpoint.
  - `run_000024` then exercised the plan's case with an external `koala stop` at update 300, which stands in for an
    eviction or a node failure. The checkpoint it resumed from was update 250, not 256, because rolling checkpoints
    come every 250 updates.

## Findings not fixed here

- The Gym records 0 GPU-hours for failed runs. For successful runs it counts training and evaluation time only:
  - Platform time was 0.8 GPU-h higher per single-job run and 1.2 GPU-h higher for the three-segment run.
  - The gap is environment setup (2–23 min per job: `uv sync` on `/local-ssd`), model build and compilation.
  - After a restart, the Gym's GPU-hours come from the checkpoint like its FLOPs, so the replay is not counted
    either. For `run_000024` the gap is 1.5 GPU-h: two job setups, plus about 0.7 GPU-h of replayed updates 251–300
    and the update-256 evaluation.
- Queueing dominated wall time.
  - The three rerun jobs waited 72 minutes; the failed attempt's jobs had waited 186–196 minutes.
  - The second `dense_full` segment waited another 137 minutes, so the segmented run took 4.6 h from launch to end.
    The single-job runs took 2.2–2.3 h.
  - The restarted job of `run_000024` waited 4.1 h, so that run took 5.6 h from launch to end. A restart costs a
    full queue wait, which is far longer than the replay itself.
- `koala stop` reclaims the Pod log at once. Koala then keeps no start time for the job, and its state reads
  `Deleting` while the stop completes, which is what the chain recorded. The stopped job's start in
  [koala_jobs.json](koala_jobs.json) is the first Gym poll that saw it Running.
- While a restart replays, `get_run` shows `progress` and `consumed` back at the checkpoint. The budget keeps the
  highest charge until the run settles. `nextt_gym/INTERFACE.md` now says so.
- Two `run.poll_error` events (`unknown backend 'koala'`) at 07:50:36 UTC came from an operator MCP call that
  lacked `NEXTT_GYM_BACKENDS`. They did not affect the run.
- `provenance.json` has `source_commit: null` for bundle runs; the commit appears only in the bundle name
  `bundle-785a65f71384-…`. For a segmented run it also keeps the first segment's host only.
- No `dense_full` anchor is registered at S, so the leaderboard's `delta_vs_anchor` is null (`cli anchor`).
- AR-01 also asks for an inspectable archive and a disclosure draft. The archive can be read through the MCP tools.
  No disclosure draft was generated, because card generation is G1 work.
- In the operator host's `.venv`, `setuptools` is missing, so CPU `torch.compile` tests fail unless
  `TORCH_COMPILE_DISABLE=1` is set. Pods are not affected.

Stage 0 used 41.57 platform GPU-hours in 8-GPU jobs, plus one single-GPU debug pod. Of these, 7.33 went to the failed
attempt and 8.63 to `run_000024`.

## Files

| File | Contents |
|---|---|
| [check.py](check.py), [acceptance.json](acceptance.json) | A1–A3 auditor and its output; A2 includes the restart's replayed events and the comparison of the two `dense_full` runs |
| [a4.json](a4.json) | Agent-visible fields of the three runs, the `POLICY_DENIED` responses and the imported reference example, as seen at `f8f5d05` |
| [a4_sweep.py](a4_sweep.py), [a4_sweep.json](a4_sweep.json) | Sweep of every agent-visible MCP output for published official values, before and after `93983dd` |
| [runs/](runs) | Per run: `metrics.jsonl`, `budget.json`, `run_status.json`, `run_spec.json` and Koala `chain.json` as fetched; `provenance.json` reduced to fields without host paths; for `run_000024` also `replayed_metrics.jsonl`, the events the stopped job logged after update 250 |
| [formal/](formal) | Training events up to update 512 of the formal `gdn` and `kda` runs |
| [koala_jobs.json](koala_jobs.json) | All ten Koala jobs of the S runs: state, queue time and pod time |
| [failed_runs.json](failed_runs.json) | The first attempt's Gym records and first log error, with paths replaced by `DATA`, `WORKDIR` and `RUN` |
