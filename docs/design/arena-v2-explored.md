# Arena v2: what has been explored

Status as of 2026-11-05. This page records what Arena v1 and the work around it have established for Arena v2. The
remaining work is in the [roadmap](arena-v2-roadmap.md). The detailed workflow design is in
[arena-v2-agent-workflow.md](arena-v2-agent-workflow.md). Evidence is in
[`evidence/checks/arena-v1/`](https://github.com/ggeeyyi/NextTransformer/tree/25c553d9e03e43334c7fdf3171736b5a8bf42413/evidence/checks/arena-v1) and the stage directories beside it.

## Arena v1: one supervised research episode (2026-10-30 to 2026-11-05)

1. **The research gym on real data (Stage 0).**
   - The gym trains like the formal trainer: step-1 losses are equal, and the mean \|Δloss\| over updates 300–510 is
     at most 0.0039.
   - Runs resume across planned segments and after an abrupt stop. FLOPs settle exactly, and the final dev BPB is
     0.887221 against 0.887830 for an uninterrupted run.
   - The ledger equals the audited costs. Validation and CORE values never reach the agent, and a fresh server rebuilds
     its state from the store.
   - Cost: 41.6 platform GPU-hours.
2. **Proxy fidelity (Stage 1).**
   - A compressed 512-update schedule only screens. It reaches Spearman 0.40 and pair accuracy 0.674, against the
     required 0.85 / 0.9.
   - The 2,000-update constant-LR prefix ranks: 0.918 / 0.935 in a backtest on published curves.
   - Repeats with the same initialization differ by 0.00009 at update 512.
   - Cost: 110.4 GPU-hours.
3. **Candidate search (Stage 2).**
   - Claude proposed, wrote and selected candidates through the gym's tools, under rules preregistered before the first
     job:
     - screen at update 512: cancel a candidate more than 0.030 above `dense_full`;
     - rank at update 2,000;
     - win: end below the best anchor by more than max(2δ, 0.001), with δ = 0.00041.
   - 17 runs over three rounds. No single change won in round 1; combinations of the near-winners won in rounds 2 and 3.
   - Best: `attnres_full_canon_abcd`, −0.0038 against `attnres`. A gate, a deeper layout, value embeddings on every
     layer and MoE FFNs did not help.
   - Cost: 549.7 GPU-hours, over 24.1 h of wall time.
4. **Port and checks (Stage 3).**
   - The two best candidates were ported by hand to method plugins and calibrated (+0.337% and +0.486% against the
     anchor), with cost proofs and tests.
   - The gym and the formal trainer give bitwise-equal first updates; the mean \|Δloss\| over updates 300–510 is
     0.0033–0.0041.
   - Cost: 42.1 GPU-hours.
5. **Formal run of the winner (in progress).**
   - A bounded 32-GPU check passed first: step-1 loss equal to the 8-GPU run, 2.98 s/update (95% scaling).
   - Segment 1 ended at its time limit at update 39,500. Segment 2 resumes from the update-39,000 checkpoint.
   - At update 39,500 the run is first of 34 runs, with val BPB 0.6819:
     - −0.0014 against `mhc`, stable since update 10k;
     - −0.0032 against `attnres`, narrowing during the decay, as `mhc`'s lead over `attnres` also narrows.
   - CORE: 0.232 / 0.240 / 0.279 / 0.292 at updates 8k–32k. At 32k it ties `mhc` and is 0.016 above `attnres`.
   - Projected final val BPB: 0.660–0.661. The best published run, `mhc`, ends at 0.6617.

## What the evidence means for v2

1. **Screen early, rank late.** A 512-update run rejects clearly worse candidates but cannot rank them. The decay
   reorders close rows: our lead over `attnres` shrank from −0.0057 (mean over updates 10k–22k) to −0.0032.
2. **Seed noise is unknown.** δ measured GPU non-determinism only, because the seed offset did not change the
   initialization.
3. **BPB and CORE diverge by mechanism.** Over the 23 complete published runs, they agree on 83% of pairs
   (Spearman 0.85). Fixed-state mixers lose on tasks that read answers from the context. The v1 winner ties `mhc` on
   CORE while leading it on BPB.
4. **Efficiency was not part of selection.**
   - Training: in the gym (without `torch.compile`), the winner took 1.66× the time per update of `attnres`, at +0.48%
     audited FLOPs.
   - The formal run takes 3.01 s/update on 32 GPUs, with an MFU of about 35% and a peak of 97.9 GiB.
   - Inference cost was never measured.
5. **People did these steps:** plan approval, preregistration review, compute caps, stopping decisions, porting and
   operating the formal run.
6. **The record had gaps.** Runs, the ledger and the archive were kept. LLM tokens, cost and web access were not
   metered.
7. **Operations.**
   - On several nodes, the trainer's per-rank wall-clock limit can stop ranks at different updates, so segments end at
     the job's time limit instead.
   - Finalization cannot be resumed, so a job's time limit must not fall in the final evaluation.
   - Queueing dominated wall time: 3.4–5.4 h per Stage 2 run.

## Tooling built

| Area | State |
|---|---|
| Research gym (`nextt_gym/`) | 22 operations over MCP, HTTP and Python; four scales (512 updates on the formal prefix and on a compressed schedule, 2,000 updates, full length); a FLOPs ledger with platform GPU-hours; archive, lineage and leaderboards; sealing; an `evaluate()` adapter for evolutionary frameworks |
| Cluster backend | Segmented jobs, restart from a checkpoint after a failure, GPU-hour metering; used for 37 runs and about 700 GPU-hours |
| Formal-run tooling | Multi-node launch with rendezvous; segment submission with a finalization guard; a watcher |
| Disclosure | Card drafts generated from the archive |
| Progress page | Per-stage charts, a lineage tree and formal-run tracking; a prototype for the evolution dashboard |

## Upstream state

| Item | State |
|---|---|
| Roadmap [#24](https://github.com/Interactive-Active-AI/NextTransformer/pull/24) (merged) | Harbor tasks; three splits (training, the last shard for validation, CORE-22 as the test, which alone gives the final score); a single-file submission; the Gym interface list. Later: anti-hack checks and an evolution dashboard |
| Gym [#25](https://github.com/Interactive-Active-AI/NextTransformer/pull/25) (draft) | Native-plugin candidates; an HTTP SDK; a per-session FLOPs ledger that also charges checks; sandboxed workers; checkpoint continuation; a `finalize` receipt with a separate verifier; final-training dispatch. Validated on a CPU fixture only; no GPU or cluster execution |
| HuggingArch [#21](https://github.com/Interactive-Active-AI/NextTransformer/pull/21) (open) | Static analysis for 33 methods: parameters, prefill FLOPs, weight sizes; cache layouts as estimates |
| Issue [#19](https://github.com/Interactive-Active-AI/NextTransformer/issues/19) | A request to report training tokens/s and memory bandwidth. The maintainers agreed to report measured throughput with its setup, and estimates labelled as estimates |
| Agent workflow | Not specified in any pull request: the Gym design leaves the search policy to the RSI system |
