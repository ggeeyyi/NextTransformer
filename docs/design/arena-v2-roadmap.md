# Arena v2 agent workflow: roadmap

This roadmap covers what remains to explore and build for the Arena v2 agent research workflow. What is already known
is in [arena-v2-explored.md](arena-v2-explored.md). The detailed design is in
[arena-v2-agent-workflow.md](arena-v2-agent-workflow.md). The environment (the Harbor task and the Gym API) follows
the project [roadmap](../src/content/docs/roadmap.md) and the draft Gym in
[#25](https://github.com/Interactive-Active-AI/NextTransformer/pull/25).

## By November 14, 2026: an agent workflow for Arena v2

1. **Agree the objective and the access rules.**
   - The agent optimizes validation BPB under the unchanged protocol. CORE-22 is a held-out test, reported after the
     session.
   - The agent may read every published result and run, the Arena v1 research runs and stage reports, and the
     internet. Web access is logged.
   - The CORE-22 score of the agent's own candidates is never visible during research. Research profiles do not
     evaluate it, and the final run's CORE events, tracking and logs stay sealed until the session closes.
   - History is frozen at session start, so a result published during a session is not new feedback.
2. **A workflow of preregistered phases.**
   - Orientation, preregistration, anchors and noise, generation, successive halving (512 → 2,000 → longer),
     confirmation, finalization, reveal.
   - Before any training, the agent freezes its rules through the Gym: proxy rungs, screen and rank thresholds, the
     noise plan, efficiency gates, budget split and stopping rules. An amendment is recorded before the runs it
     affects.
   - Each candidate makes one primary change against a named parent, with a written forecast.
   - Default budget: one formal-run equivalent (46,875 updates, about 2.3×10²⁰ training FLOPs), split 15% anchors and
     noise, 60% search, 20% confirmation, 5% reserve.
3. **Training and inference efficiency.**
   - **Training:** step time, tokens/s per GPU, MFU and peak memory, measured on every research run together with its
     hardware and settings.
   - **Inference:** parameters, prefill and decode FLOPs, KV or state bytes per token, and bytes read per decoded
     token, as a memory-bandwidth proxy. They come from the backbone estimators or from HuggingArch
     ([#21](https://github.com/Interactive-Active-AI/NextTransformer/pull/21)).
   - **Use:** hard gates per cohort, checked before training; ties within noise go to the more efficient candidate;
     Pareto plots against the published runs; candidates kept alive in several inference-cost classes.
4. **Complete trajectory recording.**
   - Record model calls (Harbor's trajectory), workspace snapshots, a web log through an outbound proxy, every Gym
     API call, decisions through a new `record_decision` operation, and operator actions.
   - Keep everything in one append-only, hash-chained bundle per session. Generate the disclosure card from it,
     LLM tokens and cost included.
5. **Run it.**
   - A dry run of the whole workflow on the CPU fixture, in Harbor with Claude Code.
   - A bounded GPU session on real data (about 0.1 formal-run equivalents) through a GPU cluster adapter.

Later:

- **First full session:** one formal-run equivalent of research, then the final training. Reveal BPB, CORE and
  efficiency.
- **Cross-session self-improvement:** change the research layer (generator, proxy rungs, priorities) only when the new
  version beats the frozen previous one on a sealed cohort under the same budget.
- **Anti-hack review** of trajectories, and the **evolution dashboard**, both built from the bundle.

## Open research questions

| Question | What we know | How to explore | Cost |
|---|---|---|---|
| Does a short decay after the 2,000-update prefix rank close candidates better? | The decay reorders close rows. Our lead over `attnres` shrank from −0.0057 to −0.0032 | Branch a short decay from the 2,000-update checkpoints of the anchors and the Stage 2 winners, then compare with their finals | A few short runs |
| How large is the seed noise at the ranking rung? | Only GPU non-determinism was measured (δ = 0.00041) | Two more initialization seeds each of `dense_full` and `attnres` at 2,000 updates; Stage 2 already gives the first | About 0.17 formal-run equivalents |
| Does selecting on BPB keep CORE? | Spearman 0.85; fixed-state mixers lose on in-context tasks; the v1 winner ties `mhc` on CORE | Compare the v1 formal run's final values. If BPB and CORE keep diverging, add a development capability suite built from non-test splits | Evaluation only |
| Which efficiency gate values? | The v1 winner took 1.66× `attnres`'s time per update; no inference figures exist yet | Measure step time and inference metrics for all published methods, and set the gates against that distribution | Static analysis plus short benchmarks |
| Do the proxy rules hold for every family? | Curve shapes differ by family: linear mixers start slow, looped models start fast | Backtest the 512- and 2,000-update rules per family on the published curves | CPU only |

## Current implementation and remaining work

| Area | Current state | Required change |
|---|---|---|
| Agent workflow | Arena v1's workflow, with a person approving each stage; the v2 design draft | Agree D1–D8; freeze workflow v0 |
| Candidate format | Local gym: its own model contract, with hand porting. #25: native plugins | Use native plugins and retire hand porting |
| Execution | Local gym: a GPU cluster backend. #25: one local CPU worker | A GPU cluster adapter for the #25 Gym, built from the v1 backend |
| Run length and continuation | Local gym: fixed scales. #25: an agent-chosen target update with checkpoint continuation | Use #25's; add a short-decay research profile |
| Budget | Local gym: FLOPs ledger, checks free. #25: checks charged, exact integer ledger | Use #25's ledger |
| CORE sealing | Local gym: sealed at the API. #25: research never evaluates CORE | Also seal the final run's CORE events, tracking and logs from the session |
| Research history | Local gym: imported methods and its archive. #25: published runs | Add the v1 research runs and earlier trajectories; freeze them per session |
| Efficiency | Step time and memory in run logs; static training FLOPs | Inference metrics, measured/estimated labels, gates and Pareto reports |
| Trajectory | Runs, ledger and archive; no LLM or web metering. #25: Harbor trajectory and signed Gym evidence | A web proxy, `record_decision`, the hash-chained bundle, disclosure generation |
| Isolation | Local gym: static checks. #25: sandboxed workers | Keep #25's; GPU device isolation per deployment |
| Final training | Local: manual admission and operator scripts. #25: `finalize` and separate single-node dispatch | Multi-node dispatch, built from the v1 formal-run tooling |

## Decisions needed

| ID | Decision |
|---|---|
| D1 | Research BPB on the dev split, which keeps the final BPB held out (recommended), or on the validation shard (#24) |
| D2 | Final score: BPB with CORE-22 as a held-out check (this roadmap), or CORE-22 only (#24) |
| D3 | Efficiency gates, their values and the reference hardware |
| D4 | Research budget per session, and concurrency |
| D5 | A short-decay branch in the research profiles |
| D6 | Trajectory publication: what is public, redaction, retention; the web logging proxy |
| D7 | History snapshot: frozen at session start, or live |
| D8 | Initialization seeds for the noise runs (research only; the final run keeps seed 42) |
