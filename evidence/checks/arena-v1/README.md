# Arena v1: the first research episode with NextT-Gym — 2026-10-30 to 2026-11-02

Arena v1 ran the research gym (`nextt_gym/`) on real data for the first time, then used it for one supervised research
episode. It carries out two approved items of the [arena and gym plan](../../../docs/src/content/docs/arena/plan.md):
AR-01, the real-data acceptance of the gym (Stage 0), and AR-03, a representative proxy study before any wider search
(Stages 1 and 2). Every run trained through the gym's MCP tools on Koala, one 8-GPU job per run or segment, and
evaluated only the arena's dev split `shard_06541.parquet`. This is research evidence, not a formal result: nothing
here ranks on the leaderboard, and no threshold was changed after a stage started.

| Path | Contents |
|---|---|
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | The plan, with each stage's preregistered checks and the amendments made before its runs |
| [`../arena-v1-stage0-20261031/`](../arena-v1-stage0-20261031/README.md) | Stage 0: tooling acceptance (A1–A4) |
| [`../arena-v1-stage1-20261101/`](../arena-v1-stage1-20261101/README.md) | Stage 1: does a compressed 1% run rank methods? |
| [`../arena-v1-stage2-20261102/`](../arena-v1-stage2-20261102/README.md) | Stage 2: three search rounds at the 2,000-update rung |
| [bundles.json](bundles.json) | The code bundle of every run: commit, SHA-256, submission and source hashes |

Each stage directory holds its run files (`runs/`), the scripts that recompute its checks and analysis, and their
outputs. Run them from the repository root's environment, for example `python evidence/checks/arena-v1-stage2-20261102/check.py`.

## Results

**Stage 0 (AR-01): the gym is accepted on real data.** Four S-prefix runs (512 updates, 1.09% of a formal run) of
imported methods passed all four preregistered checks:
- **A1, trainer agreement.** The step-1 losses of `gdn` and `kda` equal their formal runs'; the mean |Δ| over updates
  300–510 is at most 0.0039 (limit 0.015).
- **A2, resume.** Tokens and FLOPs stay continuous across planned segments and across an abrupt `koala stop` at update
  300, after which the gym restarted the run from its update-250 checkpoint. The restarted run settles at exactly the
  FLOPs of an uninterrupted one; their final dev BPB is 0.887221 and 0.887830.
- **A3, ledger.** Per-update, training, evaluation and settled FLOPs equal the audited formal costs.
- **A4, sealing and recovery.** No official or CORE value reaches the agent, including through imported reference
  runs, and a fresh server rebuilds its state from the store.

A first attempt (`run_000018`–`run_000020`) failed at the update-0 dev evaluation on a race between ranks; the gym
fixes from this stage are part of `nextt_gym/`. Stage 0 used 41.57 platform GPU-hours.

**Stage 1 (AR-03): a compressed 1% run only screens.** Scale C trains 512 updates under the formal schedule compressed
to 512 steps. On eleven imported methods with published finals it reaches Spearman 0.40 and pair accuracy 0.674,
against the preregistered 0.85 / 0.9 and the constant-LR prefix's 0.718 / 0.826 at update 500. To keep the three rows
within 0.005 of the best final, C needs a threshold of 0.0301 above `dense_full` and keeps `deeploop`, whose final is
the worst; the prefix needs 0.0261 and rejects it. Ranking needs the 2,000-update prefix, which meets the rule in the
backtest on published curves (0.918 / 0.935). Two repeats of `dense_full` differ by 0.00009. Stage 1 used 110.44
platform GPU-hours.

**Stage 2 (AR-03): four `attnres` variants beat both anchors at the 2,000-update rung.** Claude acted as the
researcher through the gym's MCP tools, under rules preregistered before the first job: scale M (the first 2,000
updates, 4.27% of a formal run), a screen at update 512, and a candidate wins if it ends below the better anchor by
more than max(2δ, 0.001). The anchors ended at 0.80135 (`dense_full`) and 0.80031 (`attnres`); δ, the gap between two
`dense_full` runs, is 0.00041, so the line was 0.79931.

| Round | Candidate | Change | Dev BPB at 2,000 | vs `attnres` |
|---|---|---|---:|---:|
| 3 | `attnres_full_canon_abcd` | Full AttnRes added to `attnres_canon_abcd` | **0.79651** | **−0.00380** |
| 2 | `attnres_canon_abcd` | Canon-A/B/C/D | 0.79819 | −0.00212 |
| 2 | `attnres_full_canon` | Full AttnRes with Canon-A/C | 0.79849 | −0.00182 |
| 2 | `attnres_full_gate` | Full AttnRes with a head-wise output gate | 0.79930 | −0.00101 |

No single change of round 1 won: `attnres_full` (−0.00080) and `attnres_canon` (−0.00059) came closest, and a gate,
value embeddings on every layer and a deeper layout did not help. The 8- and 16-expert MoE FFNs did not win. These are
single-seed values at a constant-LR prefix; the gaps may change after the decay of a full run. The gym trainer runs
without `torch.compile`, and in it the best candidate takes 1.66× the time per update of `attnres`. Stage 2 used
549.72 platform GPU-hours in 8-GPU jobs over 24.1 h, which brings arena v1 to 701.7.

**Next.** By the preregistered rule, `attnres_full_canon_abcd` and `attnres_canon_abcd` are the Stage 3 candidates:
port them into `nexttransformer/methods/` with calibration, proofs, tests, a model card and a disclosure card, and enter
them through the arena's two pull requests. The port and its checks are done
([Stage 3](../arena-v1-stage3-20261103/README.md)); a formal run needs its own authorization.

## Code provenance

The runs were made on the private development branch `gy/arena-v1`. Its commits, and the records, branches and
commits that the stage READMEs cite, belong to that history and do not resolve in this repository. What identifies the
exact code of a run is its **code bundle**: the gym packs the repository and the submission into one archive, records
the commit inside it (`BUNDLE.json`), and every run verifies the archive's SHA-256 before it starts. [bundles.json](bundles.json)
lists, for each run, that SHA-256, the commit and the submission hash, which matches the run's own `provenance.json`,
with the upstream source hash from that file. The operator keeps the archives.

| Stage | Runs | Commit | Distinct bundles |
|---:|---|---|---:|
| 0 | `run_000018`–`run_000020` (failed first attempt) | `6f2c42291027` | 3 |
| 0 | `run_000021`–`run_000023` | `785a65f71384` | 3 |
| 0 | `run_000024` | `93983dd15f65` | 1 |
| 1 | `run_000025`–`run_000037` | `e4c42f2c105d` | 11 |
| 2 | `run_000038`–`run_000054` | `e06e4e654cc6` | 15 |

Repeated runs of one method share a bundle. On 2026-11-03 the gym at `e06e4e6` and the gym in this repository trained
the same seven submissions (the template, both anchors and the four Stage 2 winners) for six synthetic CPU steps; the
losses, dev BPB, final weights and FLOPs were bit for bit identical. The data pipeline, the optimizer and the locked
runtime dependencies are unchanged; the other upstream changes add the method-plugin interface, which built-in methods
do not run, and rename settings. That check needs the private tree, so it cannot be rerun from this repository alone.

## Publication edits

The evidence was copied from the private branch at `7324f9c` with these edits and no others. Numbers, metrics and run
files are otherwise unchanged, and every script here reproduces its stored output in this repository.
- The GPU model is withheld: `device` fields read `withheld`, the backends are named `koala`, `koala-seg` and
  `koala-mb8`, and the text says "8-GPU" or "GPU" instead of the model. Two formal reference files were renamed to
  `formal/<method>_seed42_head.json`.
- The platform user name in job names is replaced by `user`.
- The Stage 0 check reads the `dense_full` cost from [`evidence/audits/dense_full.json`](../../audits/dense_full.json),
  whose `theory_flops` equals that of the private audit file it used.
- The Stage 1 backtest is pinned to the 23 runs published when Stage 1 ran, so that later results do not change it.
- Links to the plan point to [RESEARCH_PLAN.md](RESEARCH_PLAN.md), which carries the same edits plus links to the
  public protocol, rules and plan pages.
