# Arena v2 agent research workflow: draft roadmap

**Status:** draft v0.1 for discussion, 2026-11-05.

Summaries: [what has been explored](arena-v2-explored.md) and the [roadmap](arena-v2-roadmap.md).

**Scope.** This document covers the research loop that an agent runs inside one Arena v2 session: how it reads
history, proposes, tests and selects one architecture, submits it, and what it records along the way. The environment
the loop runs in is specified elsewhere:
- the Harbor task and the Gym API in the [roadmap](../src/content/docs/roadmap.md) (#24);
- the draft Gym in [#25](https://github.com/Interactive-Active-AI/NextTransformer/pull/25).

This document lists only what the workflow needs from that environment (§9).

## 1. Requirements

| # | Requirement | Where this draft meets it |
|---|---|---|
| R1 | Build on the Arena v1 research workflow | §2 summarizes v1. §5 keeps its stages, preregistered rules and evidence, and replaces the human operator's steps with Gym operations and a frozen self-preregistration |
| R2 | The agent may read all historical results and runs and use the internet. It optimizes BPB and never sees the CORE-22 score of its own work during research | §3 |
| R3 | New architectures account for training and inference efficiency ([issue #19](https://github.com/Interactive-Active-AI/NextTransformer/issues/19)) | §6 |
| R4 | The complete trajectory of every session is recorded | §7 |

**Objective.** The objective is the lowest validation BPB at update 46,875 of the formal run, under the unchanged
protocol:
- the same data, data order and tokenizer;
- a 2,048-token context, 49.152B input tokens and seed 42;
- the fixed optimizer and schedule;
- training FLOPs per input token within ±1% of the anchor.

CORE-22 stays a held-out test, reported after the session. This differs from the merged roadmap, in which the final
score comes only from CORE-22 (decision D2).

## 2. Starting point: the Arena v1 workflow

Arena v1 (2026-10-30 to 2026-11-05) was one supervised research episode. Claude was the researcher, working through the
research gym's tools, and a person approved each stage's plan, rules and compute. Evidence:
[`evidence/checks/arena-v1/`](https://github.com/ggeeyyi/NextTransformer/tree/25c553d9e03e43334c7fdf3171736b5a8bf42413/evidence/checks/arena-v1).

| Stage | Question | Rule and outcome | Platform GPU-h |
|---|---|---|---:|
| 0. Tooling acceptance | Does the gym train real data like the formal trainer, and does it resume, meter and seal correctly? | All four preregistered checks (A1–A4) pass | 41.6 |
| 1. Proxy qualification | Does a cheap run rank methods by their final BPB? | **Compressed 512-update schedule:** Spearman 0.40 and pair accuracy 0.674, against the required 0.85 / 0.9, so it only screens.<br>**2,000-update constant-LR prefix:** 0.918 / 0.935 in a backtest on published curves | 110.4 |
| 2. Search | Which change beats both anchors? | **Runs:** 17 at 2,000 updates, in three rounds, with one primary change per candidate.<br>**Screen at 512:** cancel a candidate that is more than 0.030 above `dense_full`.<br>**Win:** end below the best anchor by more than max(2δ, 0.001), with δ = 0.00041.<br>**Stop:** after three rounds, at 600 GPU-h, or after two rounds without a winner.<br>**Result:** four winners. The best, `attnres_full_canon_abcd`, is −0.0038 against `attnres` | 549.7 |
| 3. Port and check | Does the winner train the same way as a method plugin? | Ported to plugins by hand. The first updates are bitwise equal, and mean \|Δloss\| over updates 300–510 is 0.0033–0.0041 | 42.1 |
| Formal run | Does the lead hold over 46,875 updates? | 32 GPUs, two segments, in progress.<br>**At update 39,500:** val BPB 0.6819, first of 34 runs.<br>**vs `mhc`:** −0.0014, stable since update 10k.<br>**vs `attnres`:** −0.0032, narrowing during the decay, just as `mhc`'s lead over `attnres` narrows.<br>**CORE at 32k:** 0.292, against 0.293 for `mhc` | ≈1,400 (projected) |

Lessons that shape v2:
1. **Screen early, rank late.** A 512-update run can reject clearly worse candidates but cannot rank them. The
   2,000-update prefix ranked well on published curves, but the learning-rate decay reorders close rows. Our lead over
   `attnres` shrank from −0.0057 (mean over updates 10k–22k) to −0.0032 during the decay.
2. **The noise estimate measured the wrong thing.** δ measured GPU non-determinism, because the seed offset did not
   change the initialization. Seed-to-seed variance is still unknown.
3. **Selection ignored efficiency.** In the gym, the winner took 1.66× the time per update of `attnres` (11.33 s
   against 6.82 s, without `torch.compile`), at +0.48% audited FLOPs. The protocol does not score speed, but speed set
   the formal run's GPU-hours, and inference cost was never measured.
4. **BPB and CORE disagree by mechanism.** Over the 23 complete published runs, final validation BPB and CORE rank 83%
   of pairs the same way (Spearman 0.85). Fixed-state mixers lose on tasks that read answers from the context. The v1
   winner leads `mhc` on BPB but ties it on CORE.
5. **A person did steps that v2 must automate or record:** plan approval, rule preregistration, compute caps, stopping
   decisions, porting and operating the formal run.
6. **Porting cost a whole stage.** v2 candidates are native plugins (#25), so the code that was researched is the code
   that is submitted.
7. **The record was incomplete.** Runs, the FLOPs ledger and the archive were kept. LLM turns, tokens, cost and web
   access were not metered; the disclosure card says so.

## 3. Information access

| Source | Contents | Agent access | Notes |
|---|---|---|---|
| Published results | For each run in `results/runs/`: 189 validation points, 6 CORE points, final BPB and training compute | Read | Historical context, historical CORE included |
| Evidence | Audits, cost proofs, model cards, method source and admitted configurations | Read | Through the Gym's history and reference operations (`list_experiments`, `read_experiment`, `read_reference` in #25) |
| Research history | Arena v1 research runs (curves at 512 and 2,000 updates), stage reports, failure records, and the trajectories of earlier published sessions | Read | Gives the agent proxy-fidelity evidence without repeating Stage 1 |
| Internet | Papers, code, discussions | Allowed and logged (§7) | Public CORE data and leaderboard pages are reachable, so the safeguard is logging, not blocking |
| Session runs | Training loss, research BPB, efficiency metrics, checkpoints | Read | Visible only to the session that owns them |
| CORE-22 of session candidates | — | **Never** during the session | Research profiles never evaluate CORE. The formal run of the submitted candidate evaluates CORE at six points; those values stay out of the agent's tools, logs and reach until the session is closed |

**Leaks to close for CORE**, given that the agent has internet access:
- The formal run's experiment tracking (for example W&B) is private, and its credentials are not in the agent
  container.
- No result pull request, leaderboard update or public log of the final run appears before the session closes.
- Research job logs are filtered to the metrics the profile exposes.
- History is frozen at session start (a cohort snapshot), so a result published during the session is not new
  feedback.

**Which BPB the agent sees during research (decision D1):**
- **(a) Recommended.** The dev split `shard_06541`, as in v1. The validation shard `shard_06542` stays untouched until
  the final run, so the final BPB is a held-out number too. We recommend this because BPB is the objective.
- **(b)** The validation shard, as in #24 and #25. The final BPB is then not held out, and CORE is the only held-out
  signal.

## 4. Roles

| Role | Owns |
|---|---|
| Agent | Hypotheses and candidate code (native plugins); experiment lengths, continuation and stopping; budget allocation across experiments; its own preregistered rules; the final choice |
| Gym | Data, trainer, evaluation, FLOPs accounting, efficiency measurement, CORE, final training, trajectory evidence |
| Operator (a person) | The cohort settings (budget, efficiency gates, history snapshot); starting and closing the session. Any intervention is a logged action and sets the involvement level in the disclosure |

## 5. The workflow

One session is one Harbor trial. The phases follow v1's stages, and each phase ends with an artifact recorded in the
trajectory (§7).

**P0. Orientation** (CPU only, no research FLOPs)
- Read `describe_space`, `get_resources` and `get_budget`.
- Build a **bank**: every historical run as one table, with its mechanism tags, final BPB, curve shape, decay gain,
  FLOPs breakdown and efficiency metrics.
- Diagnose the frontier: which mechanisms lead, where the curves cross, and the efficiency position of each run.
- Exit artifact: a research plan.

**P1. Preregistration** (CPU only)
- Before any training, write and freeze:
  - the proxy rungs and the screen and rank rules;
  - the noise plan;
  - the efficiency gates (the cohort's plus the agent's own);
  - the budget split, the stopping rules and the final selection rule.
- Backtest every proxy rule on the published curves first. The v1 evidence is the default: screen at 512 updates and
  rank at 2,000 or more updates of the formal prefix.
- Store the rules through the Gym, server-timestamped and hashed. An amendment is allowed only before the runs it
  affects start, and must give a reason (v1 practice).
- This replaces v1's human approval with a commitment that can be audited.

**P2. Anchors and noise** (about 15% of the budget)
- Run the anchors at the ranking rung: the best published run by BPB (`mhc` today) and the method the agent builds on,
  if different.
- Run one repeat with a different initialization seed, unlike v1, to estimate seed noise δ_seed. v1's
  GPU-non-determinism figure, 0.00041, stays as a lower bound.

**P3. Generation** (CPU only)
- Use the operators from the earlier study-loop design: complement, transplant, repair, reallocate, ablate, and import
  from the literature or the web, citing the source and version.
- Each candidate makes one primary change against a named parent and comes with a written forecast: its gap at the
  rung and the direction of each efficiency change.
- Before any spend:
  - `inspect_cost` checks FLOPs per token, and `calibrate_candidate` brings it within ±1%;
  - the static efficiency metrics of §6 are computed;
  - the candidate is checked for novelty against registered methods;
  - candidates that violate a gate are rejected.

**P4. Screen and rank by successive halving** (about 60% of the budget)
1. Train each candidate to update 512 and apply the screen.
2. Continue the survivors from their checkpoints (continuation in #25) to update 2,000, and rank them.
3. Continue the top few to a longer rung (for example 8,000), or through a short decay branch (decision D5), to check
   whether the ranking changes in the decay.
4. Measure the training efficiency of every run (§6).

The next round's pool comes from diagnosing this round. In v1, combining near-winners was what won in rounds 2 and 3.

**P5. Confirmation** (about 20% of the budget)
- Rerun the finalists with a second seed at the ranking rung, or continue them further.
- **Winner rule:** the mean gap beats the best anchor by more than max(2·δ_seed, 0.001), and the candidate passes every
  efficiency gate. Candidates within noise of each other go to the more efficient one.
- Report how many candidates were screened, as a measure of the winner's curse.

**P6. Finalization**
- Run `validate_candidate` on the final profile (admission within ±1%), then `finalize`, which returns the receipt. The
  session ends.
- The Gym trains the candidate from scratch. Its CORE events are withheld, and the session is closed before any result
  becomes visible.

**P7. Reveal and reflection** (after the final run)
- The platform publishes the final BPB, CORE, efficiency metrics and the trajectory.
- **Cross-session self-improvement** (the RSI part): the research layer may change between sessions, meaning the
  generator heuristics, proxy rungs and priority rules. A change is adopted only if, under the same budget, it finds
  better architectures than the frozen previous version on a sealed cohort. Lower forecast error alone does not count.
  The evaluation layer never changes.

**Budget.** The unit is formal-run equivalents (FRE): 1 FRE = 46,875 updates ≈ 2.3×10²⁰ training FLOPs. v1's
Stages 0–2 used about 0.9 FRE of updates. The proposed default for one session is **1 FRE** (decision D4), split as:

| Share | Phase |
|---:|---|
| 15% | P2 anchors and noise |
| 60% | P4 search, including the charged checks |
| 20% | P5 confirmation |
| 5% | Reserve |

## 6. Training and inference efficiency

Each metric is labelled measured, estimated or unknown. Measured values state the hardware, world size, micro batch,
compile setting and software versions, as the maintainers proposed in issue #19.

| Kind | Metric | Source |
|---|---|---|
| Training | Training FLOPs per token (fixed within ±1%) | Ledger, `inspect_cost` |
| Training | Step time, tokens/s per GPU at the formal batch, MFU | Measured on the research runs; `nexttransformer.native.benchmark` |
| Training | Peak memory; the largest micro batch that fits | Measured |
| Inference | Parameters (total, and active per token for MoE); weight bytes | Static |
| Inference | Prefill FLOPs at 2k; decode FLOPs per token at 2k, 8k and 32k context | Static: the backbone estimators, or HuggingArch ([#21](https://github.com/Interactive-Active-AI/NextTransformer/pull/21)) where a spec exists |
| Inference | KV-cache or state bytes per token and per sequence at 2k and 32k; bytes read per decoded token (weights plus cache), as the memory-bandwidth proxy | Static |
| Inference (optional) | Decode tokens/s at batch 1 and 64; prefill latency at 2k | Measured on the reference hardware |

**How efficiency enters decisions:**
- **Hard gates**, set per cohort (decision D3) and checked before training. Illustrative values:
  - step time ≤ 1.5× `dense_full` at equal settings;
  - peak memory within the device at the formal micro batch;
  - KV or state bytes per token at 32k ≤ `dense_full`;
  - decode FLOPs at 8k ≤ 1.2× `dense_full`.
- **Tie-break:** among candidates within noise of each other, choose the more efficient one.
- **Reporting:** Pareto plots of BPB against training tokens/s and of BPB against decode bytes per token, covering
  every candidate and every published run. The final report compares the winner's efficiency with `dense_full` and
  with the best published run.
- **Portfolio:** keep candidates alive in several inference-cost classes, so that the search does not collapse onto
  the most expensive one. The classes are full attention, sparse or local attention, linear state, hybrids, and
  cross-layer KV sharing (for example YOCO, from issue #19).
- **v1 check:** `attnres_full_canon_abcd` would have to pass the step-time gate, since it took 1.66× `attnres`'s time
  per update in the gym.

## 7. Trajectory recording

Every session produces one **trajectory bundle**. The bundle is append-only and hash-chained; the platform keeps it,
and publishes it with the result after redaction.

| Layer | What is recorded | Collected by |
|---|---|---|
| Agent | Every model call: input, output, tool calls and results, model ID, tokens, cost, timestamps | Harbor's native trajectory (#25) |
| Workspace | Shell commands, files written, candidate drafts; container runtime snapshots | Harbor runtime snapshots, and an archive of the final workspace |
| Web | Every search query and fetched URL, with time and content hash | A logging proxy on the agent container's outbound traffic |
| Gym | Every API call and response, candidate packages, quotes, jobs, results, the ledger, efficiency measurements | The Gym store and signed evidence (#25) |
| Decisions | The plan, preregistered rules and amendments; hypotheses with their forecasts; each selection, with the rule applied and the evidence IDs | A new Gym operation `record_decision`, server-timestamped and immutable |
| People | Every operator action, with its reason | Operator log |

**Uses:**
- **Disclosure card:** generated from the bundle, including LLM tokens and cost, which closes v1's gap.
- **Anti-hack review** (roadmap "Later"): attempts to reach CORE, fetches of sealed pages or of the session's own result
  pages, attempts to change the evaluation, seed shopping, and claims without evidence.
- **Evolution dashboard** (roadmap "Later"): the Arena v1 progress page, with its lineage tree, curves and compute per
  stage, is a prototype.
- **Self-improvement:** the bundle is the input to the cross-session self-improvement step (P7).

## 8. Guardrails

| Risk | Guardrail |
|---|---|
| The agent changes what is measured | The protocol and evaluation layer are outside its write scope. Candidates are plugins, and admission checks cost and fingerprints |
| CORE becomes a development signal | CORE is never computed in research. The final run's CORE is withheld until the session closes, tracking is private and logs are filtered. Web access is logged and reviewed |
| Rules tuned after seeing results | Preregistration in P1; amendments logged before the runs they affect |
| Unpaid compute | The Gym's budget covers checks, failures and cancellations (#25) |
| Winner's curse and seed luck | A second seed for finalists, the number of screened candidates reported, final seed fixed at 42, no research weights in the final run |
| Efficiency hidden behind BPB | Gates before training, Pareto reporting, a portfolio across inference-cost classes |

## 9. What the workflow needs from the Gym

| Need | #25 today | Gap |
|---|---|---|
| Formal-dimension research runs on GPUs, with several at a time | Local CPU worker, one slot | A GPU cluster adapter. The Arena v1 cluster backend (segments, restart from checkpoint, GPU-hour metering) could supply it |
| Agent-chosen run length and continuation | Yes (`target_update`, `checkpoint_id`) | — |
| A short-decay branch from a checkpoint | No | A research profile for it (D5) |
| Efficiency metrics from `inspect_cost` and `get_result` | Cost report only | Add the §6 metrics with their labels |
| Research history | Published runs only | Add the Arena v1 research runs and earlier published trajectories |
| Decision log | No | `record_decision` |
| CORE withheld from the final run's observers | The final adapter runs formal training | Keep its CORE events, logs and tracking away from the session |
| Web log | No | An outbound logging proxy for the agent container |

## 10. Milestones

| Milestone | Target | Deliverable |
|---|---|---|
| M0 | 2026-11-07 | Decisions D1–D8 taken; this workflow frozen as v0 |
| M1 | 2026-11-10 | The Gym additions from §9: efficiency metrics, `record_decision`, research history, the trajectory bundle with the web log |
| M2 | 2026-11-12 | A dry run of the whole workflow on the CPU fixture, in Harbor with Claude Code; the trajectory bundle reviewed |
| M3 | 2026-11-14 | A bounded GPU session on real data (about 0.1 FRE); workflow v0 published with the Arena v2 release |
| M4 | After 11-14 | The first full session (1 FRE) and its final run; the reveal; the first cross-session revision |

## 11. Decisions needed

| ID | Decision |
|---|---|
| D1 | Research BPB on the dev split (recommended) or on the validation shard |
| D2 | Final score: BPB (this draft) or CORE-22 (#24); the other is reported as a held-out check |
| D3 | Efficiency gates, their values and the reference hardware |
| D4 | Research budget per session and concurrency |
| D5 | A short-decay branch in the research profiles |
| D6 | Trajectory publication: what is public, redaction, retention; the web logging proxy |
| D7 | History snapshot: frozen at session start, or live |
| D8 | Seed noise: initialization seeds for the noise runs (research only; the final run keeps seed 42) |
