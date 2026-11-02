# Arena v1 Stage 2: search rounds at the 2,000-update rung on Koala — 2026-11-02

Stage 2 of the [arena v1 research plan](../arena-v1/RESEARCH_PLAN.md#stage-2-search-rounds) searched for
changes to the imported methods that beat both anchors at the research rung. Claude acted as the researcher through the
Gym MCP tools and as the operator for the user.

- **Scale.** Every run trains at scale M: the first 2,000 updates of the formal 46,875-update schedule.
  - That is 2,097,152,000 input tokens, 4.27% of a formal run.
  - After the 40-update warmup the LR stays at its peak, so M is a constant-LR prefix.
  - Each run evaluates the dev split every 256 updates and at update 2,000.
- **Runs.** Each run is one 8-GPU Koala job, launched through the Gym MCP tools at seed 42. There are 17 in all:
  - anchors `dense_full` and `attnres`, plus a repeat of `dense_full` (seed offset 1) for noise;
  - five round-1 candidates and one cancelled before it started;
  - six round-2 candidates, one of them relaunched after running out of memory;
  - one round-3 candidate.
- **Code.** Every bundle comes from commit `e06e4e6` on `gy/arena-v1`.
- **Rule.** The rules are preregistered in `e06e4e6` (*Amendments before round 1*). That commit is dated 16:26:08 on
  2026-11-01; the first Stage 2 job was submitted at 16:26:53. No threshold changed afterwards.
- **Status.** This is research evidence, not a formal result.
- **Data.** The runs evaluate only the dev split `shard_06541.parquet`. Candidates were chosen from dev values only.

**Outcome: four candidates beat the best anchor.** The line is 0.79931, 0.001 below `attnres` (0.80031).
- **Best.** Round 3's `attnres_full_canon_abcd` (Full AttnRes with Canon-A/B/C/D) ends at **0.79651**, 0.00380 below
  `attnres`. It led every candidate from update 768 on; at update 512 it was second, 0.0006 behind
  `attnres_gate_canon`. Its lead over `attnres` grew from 0.0029 at update 1,024 to 0.0038 at update 2,000.
- **Others.** The round-2 winners `attnres_canon_abcd` (0.79819, −0.00212), `attnres_full_canon` (0.79849, −0.00182)
  and `attnres_full_gate` (0.79930, −0.00101, by 0.000014).
- **Stop.** Stage 2 stopped after its third round, at 549.7 platform GPU-hours in Koala jobs (cap 600). It took 24.1 h
  from the first submission to the last result.
- **Stage 3.** By the preregistered rule, the best two, `attnres_full_canon_abcd` and its parent `attnres_canon_abcd`,
  go to Stage 3. Stage 3 needs its own approval.
- **Limits.**
  - These are single-seed values at a constant-LR prefix (4.27% of a formal run). The gaps may shrink or reverse
    after the decay of a full run.
  - The Gym trainer runs without `torch.compile`. Here `attnres_full_canon_abcd` took 11.33 s per update against
    6.82 s for `attnres` (1.66×), at audited costs of +0.48% and +0.15% against the anchor. Speed is an outcome under the protocol, not a constraint,
    but it sets the GPU-hours of a formal run.

## Rule

[analyze.py](analyze.py) applies the preregistered rule; [analysis.json](analysis.json) is its output.
- **Screen.** Cancel a candidate whose dev BPB at update 512 exceeds the `dense_full` anchor's (0.8875) by more than
  0.030. No candidate came near: the largest gap was +0.0100 (`attnres_deep`).
- **Noise.** δ = |0.80135 − 0.80176| = 0.00041, the gap at update 2,000 between the two `dense_full` runs. As in
  Stage 1, the seed offset does not change an imported method's initialization. δ therefore measures GPU run-to-run
  non-determinism, not seed-to-seed variation.
- **Decision.** A candidate beats the best anchor if it ends below the lower anchor by more than max(2δ, 0.001) =
  0.001. The lower anchor is `attnres` at 0.80031, so the line is **0.79931**.
- **Stop.** Stop after three rounds, at the 600 platform GPU-hour cap, or after two rounds without a winner.

## Runs

| Run | Round | Method | Micro batch | Dev BPB 512 | Dev BPB 2,000 | vs `attnres` | Beats | s/update | Peak memory | Cost vs anchor | Gym GPU-h | Platform GPU-h | Queue |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `run_000038` | anchor | `dense_full` | 16 | 0.8875 | 0.80135 | +0.00104 | — | 6.05 | 75.0 GiB | +0.04% | 27.23 | 27.96 | 3.6 h |
| `run_000039` | anchor | `attnres` | 16 | 0.8900 | 0.80031 | +0.00000 | — | 6.82 | 81.7 GiB | +0.15% | 30.71 | 31.38 | 3.6 h |
| `run_000040` | noise | `dense_full (offset 1)` | 16 | 0.8880 | 0.80176 | +0.00145 | — | 6.07 | 75.0 GiB | +0.04% | 27.31 | 28.03 | 3.7 h |
| `run_000041` | 1 | `attnres_gate` | 16 | 0.8873 | 0.80091 | +0.00060 | no | 6.91 | 83.9 GiB | +0.21% | 31.09 | 31.78 | 3.4 h |
| `run_000042` | 1 | `attnres_canon` | 16 | 0.8893 | 0.79972 | −0.00059 | no | 7.53 | 86.1 GiB | +0.19% | 33.91 | 34.68 | 5.4 h |
| `run_000043` | 1 | `attnres_full` | 16 | 0.8889 | 0.79951 | −0.00080 | no | 8.10 | 81.8 GiB | +0.30% | 36.47 | 37.24 | 5.4 h |
| `run_000044` | 1 | `attnres_deep` | 16 | 0.8975 | 0.80299 | +0.00268 | no | 7.17 | 85.8 GiB | +0.00% | 32.25 | 32.98 | 5.4 h |
| `run_000045` | 1 | `attnres_ve_all` | 16 | 0.8922 | 0.80101 | +0.00070 | no | 6.89 | 86.1 GiB | +0.16% | 31.04 | 31.76 | 5.4 h |
| `run_000046` | 1 | `dense_full_gate` | — | — | — | — | cancelled | — | — | — | — | 0.00 | — |
| `run_000047` | 2 | `attnres_moe` | 16 | 0.8955 | 0.80662 | +0.00631 | no | 7.71 | 115.7 GiB | +0.21% | 34.69 | 35.64 | 5.1 h |
| `run_000048` | 2 | `attnres_gate_canon` | 16 | 0.8862 | 0.80020 | −0.00011 | no | 7.64 | 88.2 GiB | +0.25% | 34.44 | 35.16 | 5.2 h |
| `run_000049` | 2 | `attnres_full_gate` | 16 | 0.8875 | 0.79930 | −0.00101 | **yes** | 8.18 | 84.0 GiB | +0.36% | 36.85 | 37.58 | 5.3 h |
| `run_000050` | 2 | `attnres_full_canon` | 16 | 0.8879 | 0.79849 | −0.00182 | **yes** | 8.83 | 86.1 GiB | +0.34% | 39.73 | 40.46 | 5.3 h |
| `run_000051` | 2 | `attnres_moe16` | 16 | — | — | — | cancelled | 8.39 | 130.9 GiB | +0.26% | 5.73 | 6.21 | 5.2 h |
| `run_000052` | 2 | `attnres_canon_abcd` | 16 | 0.8882 | 0.79819 | −0.00212 | **yes** | 10.04 | 101.2 GiB | +0.33% | 45.21 | 45.94 | 5.2 h |
| `run_000053` | 2 | `attnres_moe16` | 8 | 0.8907 | 0.80195 | +0.00164 | no | 8.94 | 82.1 GiB | +0.26% | 40.26 | 41.21 | 1.4 h |
| `run_000054` | 3 | `attnres_full_canon_abcd` | 16 | 0.8868 | 0.79651 | −0.00380 | **yes** | 11.33 | 101.3 GiB | +0.48% | 51.01 | 51.72 | 0.0 h |

- **Common setup.** BF16 on 8 ranks, one segment per run except `run_000051`.
  - Micro batch 16 with accumulation 4, except `run_000053` (8 and 8).
  - Every candidate is calibrated to within 1% of the anchor cost.
- **Cost vs anchor.** This is the audited cost per update against 4.67239832544785 GFLOPs/token.
- **Platform GPU-hours.** Pod time from admission to the end state, multiplied by 8
  ([koala_jobs.json](koala_jobs.json)).
- **Queue.** Time from job creation to admission.
- **Gym GPU-hours.** These count training and evaluation only, so they are 0.7–1.0 lower per run than platform
  time (0.5 for the failed segment of `run_000051`).

## Rounds

Each candidate makes one primary change, named below against the method or candidate it extends. The Gym records
`attnres` as the parent of every round-1 and round-2 `attnres` attempt, including the combinations. Values are dev BPB.

**Round 1** was launched with the anchors at 16:47–16:49 on 2026-11-01. Each candidate changes an imported method.

| Candidate | Change | Update 512 | Update 2,000 | Result |
|---|---|---:|---:|---|
| `attnres_gate` | Head-wise sigmoid gate on the attention output (Gated Attention) | 0.8873 | 0.80091 | no |
| `attnres_canon` | Canon-A/C: causal depthwise convolutions on every sublayer input | 0.8893 | 0.79972 | no (−0.00059) |
| `attnres_full` | Full AttnRes: `attnres_groups` 8 → 23, one depth source per block | 0.8889 | 0.79951 | no (−0.00080) |
| `attnres_deep` | 27 layers with FFN 4608, recalibrated to the anchor cost | 0.8975 | 0.80299 | no |
| `attnres_ve_all` | Value embeddings on every layer | 0.8922 | 0.80101 | no |
| `dense_full_gate` | The gate of `attnres_gate` on `dense_full` | — | — | cancelled while queued |

No round-1 candidate wins. `attnres_full` and `attnres_canon` end below `attnres` but not by 0.001.

**Round 2** had a pool of ten candidates, each combining or extending round-1 changes.
- **Pool.** Eight were written and verified at 20:17, before any round-1 candidate had an update-512 value. The two
  MoE candidates were added at 23:00–23:28. By then, `attnres_gate` had shown no gain over `attnres` at update 1,536.
- **Choice.** Six were chosen at 23:31 from the round-1 values at update 512. They were launched at 23:32–23:40,
  queued for 5.1–5.3 h, and started at 04:42–04:54 on 2026-11-02.

| Candidate | Change | Update 512 | Update 2,000 | Result |
|---|---|---:|---:|---|
| `attnres_canon_abcd` | `attnres_canon` plus Canon-B on q, k, v and Canon-D on the MLP hidden | 0.8882 | 0.79819 | **beats** (−0.00212) |
| `attnres_full_canon` | Canon-A/C of `attnres_canon` added to `attnres_full` | 0.8879 | 0.79849 | **beats** (−0.00182) |
| `attnres_full_gate` | The gate of `attnres_gate` added to `attnres_full` | 0.8875 | 0.79930 | **beats** (−0.00101) |
| `attnres_gate_canon` | The gate of `attnres_gate` added to `attnres_canon` | 0.8862 | 0.80020 | no |
| `attnres_moe16` | 16-expert top-4 MoE FFN at the same compute and parameters | 0.8907 | 0.80195 | no |
| `attnres_moe` | 8-expert top-2 MoE FFN (capacity 1.25, load-balancing loss 0.01) | 0.8955 | 0.80662 | no |

- **`attnres_full_gate`.** It clears the line by only 0.000014.
- **`attnres_moe16`.** Its values are from `run_000053`; see *Deviations*.
- **Not chosen.**
  - `attnres_deep_gate` and `attnres_deep_canon`: `attnres_deep` was 0.0075 behind `attnres` at update 512.
  - `attnres_gate_elem`: it displaces a layer and FFN width to pay for its gate, and the head-wise gate's gain had
    faded by update 1,536.
  - `attnres_full_gate_canon`: it makes two changes against every earlier candidate.

**Round 3** opened at 08:35 on 2026-11-02, when the last round-2 candidate had its update-512 value.
- **Choice.** One candidate was chosen from round-2 values at updates 1,280–1,536. On `attnres_canon` at update 1,536,
  Full AttnRes and Canon-B/D each lowered dev BPB by about 0.002 (0.8114 → 0.8093 and 0.8094).
- **Launch.** It was verified and debug-checked, then launched at 10:07, after round 2 already had winners. It started
  without queueing.
- **Size.** Round 3 has one candidate because of the compute cap; see *Deviations*.

| Candidate | Change | Update 512 | Update 2,000 | Result |
|---|---|---:|---:|---|
| `attnres_full_canon_abcd` | Full AttnRes added to `attnres_canon_abcd` | 0.8868 | 0.79651 | **beats** (−0.00380) |

## Checks

[check.py](check.py) recomputes the following for each run, from the files under `runs/`. The output is
[checks.json](checks.json).
- The scale is M: 2,000 updates of the 46,875-update schedule, evaluated every 256 updates, at seed 42 + offset.
- Only the dev split is evaluated, at updates 0, 256, …, 1,792 and 2,000.
- Logged updates have continuous tokens and the cumulative cost of the audited per-update cost plus the evaluations
  before them.
- The run settles exactly at its train plus evaluation FLOPs, within its reservation.
- The cost is within 1% of the anchor.
- The final dev BPB is the evaluation at update 2,000.

Cancelled runs are checked up to their last logged update. All 17 runs pass. Together they settled
150,574,109,362,689,828,884 FLOPs (1.506 × 10²⁰) and 537.95 Gym GPU-hours.

## Deviations, interventions and notes

- **Verification re-run (operator).**
  - The first verifications of the six round-1 attempts (`ver_000100`–`ver_000105`) failed V0, because the attempt
    directories held a `__pycache__`.
  - The operator requested them again as `ver_000106`–`ver_000111`, and all passed.
  - Later submissions went through a client that disables bytecode writing and keeps the MCP session open until
    verification ends. The one-shot client used first ends the server process, and with it the verification thread.
- **Gym tests.** Before launch, the Gym suite passed 118 tests on the dev host, with one deselected.
  `test_adapter.py::test_evaluate_scores_and_reattaches` runs a local training, and torch inductor fails there
  because the host environment lacks `setuptools`.
- **Before submission.** These failed `estimate_cost`, so the Gym has no attempt for them:
  - `attnres_deeper` failed V1, because its FFN of 3072 is below the formal grid (3d–5d). It was dropped from the
    round-2 pool.
  - The first `attnres_moe` failed V3 on `aten.scatter.src` (the backward of `topk`). It was fixed by gathering the
    routing weights.
  - `attnres_gate_elem` measured +7.19% at 23 layers and was recalibrated to 22 layers and FFN 5760 (−0.54%).
- **Duplicate refusals.**
  - The Gym refused three pool members (`attnres_full_canon`, `attnres_full_gate_canon`, `attnres_deep_canon`) as
    `DUPLICATE_ATTEMPT`. They shared a configurable model file with other members, with the same config and
    similarity 0.971–0.974.
  - They were resubmitted with only the round-1 components each uses. The code changed, not the semantics.
- **`run_000046`.** `dense_full_gate` was cancelled while queued at 23:31, so round 2 could launch within the
  twelve-run limit. It was the dense control of `attnres_gate`, whose gain over `attnres` had vanished by
  update 1,536, and every round-2 candidate builds on `attnres`. It never ran.
- **`run_000051` → `run_000053` (operator).**
  - **Failure.** `attnres_moe16` at micro batch 16 peaked at 130.9 of 139.8 GiB. Rank 6 ran out of memory in the
    backward of update ~301, with 8.7 GiB reserved but unallocated.
  - **Gym restart.** The Gym queued a restart from the update-250 checkpoint.
  - **Intervention.** The operator cancelled that restart before admission and relaunched the attempt from scratch
    on `koala-mb8` (micro batch 8, accumulation 8). Stage 1 used the same backend for `elu_qknorm`; it changes
    memory only.
  - **Accounting.** The failed segment's 6.21 platform GPU-hours count toward the cap.
- **Compute cap and round 3.**
  - Through round 2, Stage 2 used 498.0 platform GPU-hours, plus at most about 1.7 in debug pods.
  - `attnres_full_canon_abcd` was estimated at 51 more.
  - A second candidate would have brought the total to about 595 of 600. The obvious one was the verified
    `attnres_full_gate_canon`, whose gate was neutral at update 2,000 (+0.0005 on `attnres_canon`, −0.0002 on
    `attnres_full`), so round 3 has one candidate.
- **Debug checks.** Five single-GPU debug pods checked 17 candidates before launch, with median utilization 100%.
  - The checks run 40 updates on one GPU and peak lower than the formal runs, so they did not flag the memory risk of
    `attnres_moe16`: it peaked at 116.9 GiB there, against 130.9 GiB in `run_000051`.
  - The platform no longer reports the pods' runtime. [debug_pods.json](debug_pods.json) bounds it from creation to
    last output.
- **Disclosure (D5).**
  - The researcher has read the public results in `results/runs/`.
  - Candidates were chosen with dev values only: round 2 from round-1 values at updates 512 and 1,536, and round 3
    from round-2 values at updates 1,280–1,536.

## GPU-hours

| Part | Runs | Platform GPU-hours |
|---|---|---:|
| Anchors and noise | `run_000038`–`run_000040` | 87.37 |
| Round 1 | `run_000041`–`run_000046` | 168.43 |
| Round 2 | `run_000047`–`run_000053`, with the failed segment of `run_000051` (6.21) | 242.19 |
| Round 3 | `run_000054` | 51.72 |
| **Koala jobs** | 17 | **549.72** |
| Debug pods | five single-GPU pods | 0.77–1.7 |

- **Debug pods.** 0.77 is from creation to last output in [debug_pods.json](debug_pods.json). Only round 3's stop time
  is recorded, and the budget estimate before round 3 bounded the pods at about 1.7.
- **Stage 2 total.** 550.5–551.4 platform GPU-hours, within the 600 cap.
- **Arena v1 so far.** With Stages 0 and 1 (152.0), arena v1 has used 701.7 platform GPU-hours in 8-GPU jobs, plus
  single-GPU debug pods.

## Files

| File | Content |
|---|---|
| [check.py](check.py), [checks.json](checks.json) | Per-run checks and their output |
| [analyze.py](analyze.py), [analysis.json](analysis.json) | Preregistered screen, noise and decision, and its output |
| [screen_decisions.json](screen_decisions.json), [screen.log](screen.log) | The live screen's decision at update 512 for each candidate, as it ran |
| [rounds.json](rounds.json) | Pools, debug checks and selections of rounds 2 and 3, written when each was made; round 1's hypotheses are in each candidate's `NOTES.md` |
| [koala_jobs.json](koala_jobs.json), [debug_pods.json](debug_pods.json) | Koala job, queue and pod times per run, and the single-GPU debug pods |
| `runs/<run>/` | `metrics.jsonl`, `budget.json`, `run_status.json`, `run_spec.json`, `chain.json` and a reduced `provenance.json` |
| `submissions/<candidate>/` | `model.py`, `config.json` and `NOTES.md` of every Stage 2 attempt (`att_000034`–`att_000050`) |

## Reproduce

From the repository root with its environment:

```
.venv/bin/python evidence/checks/arena-v1-stage2-20261102/check.py
.venv/bin/python evidence/checks/arena-v1-stage2-20261102/analyze.py
```
