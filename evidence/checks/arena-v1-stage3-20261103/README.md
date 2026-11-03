# Arena v1 Stage 3: the winners as method plugins, and the checks that carry gym research to the formal path — 2026-11-03

Stage 3 of the [arena v1 research plan](../arena-v1/RESEARCH_PLAN.md) takes the two candidates that won Stage 2 out of
the gym and onto the formal path, and checks the parts of the [arena and gym plan](../../../docs/src/content/docs/arena/plan.md)
that this needs: G1 (a gym attempt becomes a method and an entry), G2a and G2b (the gym trainer and the formal trainer
train the same method identically), G4 (metered research compute) and the disclosure draft of AR-01. No formal run was
made: a 49.152B-token run needs its own authorization. Every evaluation here reads the dev split
`shard_06541.parquet`; no run read the validation shard or CORE.

**Outcome: all checks pass.**
- **Port (G1).** `nexttransformer/methods/attnres_canon_abcd/` and `attnres_full_canon_abcd/` start from the Stage 2
  attempts' initial weights at seed 42 and compute their loss and every gradient bit for bit
  (`nextt_gym/tests/test_port.py`). Calibrated on the registered grid: 23 layers / FFN 6144, +0.337% and +0.486% of the
  shared budget, the per-update costs the gym audited in Stage 2 to nine digits
  ([proofs](../../costs/arena-v1-stage3-20261103/summary.json)).
- **G2a (CPU).** Both trainers' `main()`, fed the same rows, give equal losses at every update and equal final weights;
  representative methods take bitwise equal updates; the prefix scales S and M follow the formal schedule
  (`nextt_gym/tests/test_g2a.py`).
- **G2b (GPU, real data).** Each method trained 512 updates in the gym (an S run) and in the formal trainer (a smoke
  run). The first two updates are bit for bit equal in both pairs; afterwards GPU non-determinism separates them, and
  the mean |Δloss| over updates 300–510 is 0.0033 (`dense_full`) and 0.0041 (`attnres_full_canon_abcd`), within Stage 0's
  limit of 0.015 and the size of Stage 0's gym-versus-formal gaps (0.0034–0.0039). Both trainers take the same time per
  update.
- **This code base.** The gym's `dense_full` S run reaches dev BPB 0.887448 at update 512; the Stage 0 S runs on the
  earlier code reached 0.887830 and 0.887221, which differ by about the same amount from each other.
- **G4.** Platform GPU-hours are metered for every Koala run; backfilled over the arena v1 archive they equal the
  Stage 0–2 evidence (701.72 GPU-hours, all 37 runs). The disclosure cards drafted from the archive pass the arena
  card check.

## Checks before any GPU job

- **CPU tests.** The gym suite (`nextt_gym/tests`) and the repository suite pass, with the new tests above, the plugins'
  own tests (`tests/test_attnres_canon_abcd.py`) and the shared per-method tests for the two plugins.
- **Independent review.** Codex (`gpt-6-astra`, reasoning effort xhigh) reviewed the changes and the operator scripts in
  four rounds before any GPU job. It found 20 problems, all fixed before submission; among them the plugins' initial
  weights (the plugin build drew the embeddings once more), the formal trainer's smoke evaluations reading the
  validation shard (fixed with [formal_dev.py](formal_dev.py) below), and metering that kept a provisional reading over
  the final one. The last round's verdict was "OK to submit GPU checks".
- **One-GPU debug pod.** The formal trainer on real data (12 smoke updates of each method) and the gym trainer with the
  plugin (12 full-size synthetic updates), all at micro batch 16: every part ran, with a median GPU utilization of 100%
  and peak memory of 104.9 GiB (plugin, formal trainer), 74.6 GiB (`dense_full`) and 100.9 GiB (plugin, gym)
  ([debug_check.json](debug_check.json)).

## Runs

One 8-GPU Koala job per run, seed 42, micro batch 16, accumulation 4, BF16. The gym runs come from commit `25c553d`
through the gym's own bundles; the formal runs from one code bundle of the same commit.

| Run | Trainer | Method | Updates | s/update | Dev BPB at 512 | Platform GPU-h | Trainer GPU-h |
|---|---|---|---:|---:|---:|---:|---:|
| `run_000060` | gym, S | `dense_full` | 512 | 6.02 | 0.887448 | 7.55 | 7.08 |
| `run_000061` | gym, S | `attnres_full_canon_abcd` | 512 | 11.29 | 0.886702 | 13.72 | 13.26 |
| `formal-dense_full` | formal, smoke | `dense_full` | 512 | 6.04 | (one batch) | 7.35 | — |
| `formal-attnres_full_canon_abcd` | formal, smoke | `attnres_full_canon_abcd` | 512 | 11.30 | (one batch) | 13.30 | — |

- **Dev BPB.** The gym evaluates 8,388,608 dev tokens; the formal smoke run one distributed batch (262,144 tokens),
  which is too small to compare. The plugin's gym value matches Stage 2's `run_000054` at the same update (0.8868), which
  trained from the same initial weights on the same schedule prefix.
- **Platform GPU-hours.** Pod time from running to the end, times eight: the gym's metering for the gym runs, the job
  records for the formal runs ([koala_jobs.json](koala_jobs.json)). The debug pod used about 0.2 more (one GPU for about
  ten minutes; its record was cleared when it was stopped). Stage 3 used about 42.1 platform GPU-hours.

## G2b

[compare.py](compare.py) recomputes [compare.json](compare.json) from the runs' training events
(`python compare.py pairs.json`). Both runs of a pair must log all 512 updates, or the comparison is incomplete.

| Pair | Updates equal bit for bit | Mean \|Δloss\|, updates 300–510 | Largest \|Δloss\| | Final loss (gym / formal) |
|---|---:|---:|---|---|
| `dense_full` | 1–2 | 0.0033 | 0.045 at update 137 | 2.9597 / 2.9618 |
| `attnres_full_canon_abcd` | 1–2 | 0.0041 | 0.170 at update 19 | 2.9576 / 2.9626 |

Equal first updates rule out a difference in initialization, data, loss, optimizer groups or schedule: any of those
would separate the losses by update 2. From update 3 the non-deterministic order of GPU reductions makes the weights
differ in the last bits, and the runs drift apart most while the loss still falls fast (updates 1–150), as Stage 0's
gym and formal runs did. A deterministic comparison would need deterministic attention kernels in both trainers.

## The formal trainer on the dev split

A smoke run of `nexttransformer.train` still evaluates the validation shard at its first and last update, and that
shard is reserved for formal runs. [formal_dev.py](formal_dev.py) runs the formal trainer unchanged (its own start,
its data and inventory checks on the full dataset, its training) and replaces only the stream an evaluation reads,
with the gym's dev directory: the evaluations read the dev shard alone. The trainer still labels those events `val`;
in `runs/formal-*/metrics.jsonl` they are dev-split values. It refuses anything but a smoke run without CORE.
[formal_job.sh](formal_job.sh) is the pod entry (environment on the pod-local disk, the dev shard checked against its
pinned SHA-256), and [debug_check.sh](debug_check.sh) the debug pod's script.

## Publication edits

The CPFS user directory in the scripts reads `<user>`, the site file `site.gym.json`, and job names begin with `user-`.
Nothing else differs from what ran.
