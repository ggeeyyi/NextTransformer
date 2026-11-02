# Auto-research arena: plan

This is the arena's design reference. **[PLAN.md — Arena](../../PLAN.md#arena) owns the current
completion states, remaining deliverables, owners, dependencies and decisions.** The binding rules
are [ARENA.md](../../ARENA.md); the scientific training contract is [PROTOCOL.md](../../PROTOCOL.md).
A design note or implementation milestone does not amend either protocol.

## Why

NextTransformer combines an architecture framework, a fixed-budget benchmark and a common set of
baselines. The arena lets an LLM, with or without a research framework, propose an architecture and
have its final artifact judged under the same training contract as human-designed methods.

The existing operator ledger, admission checks, frozen configurations and exported run records
provide the scientific foundation. The arena adds research provenance: who made the decisions,
what candidates were tried, how the candidate was selected and what resources the research used.

## Design principles

1. **One training contract.** The arena governs research and disclosure; the formal model run follows
   the same data, compute, initialization and evaluation requirements as other methods.
2. **Free research, full disclosure.** The current protocol does not cap research compute or candidate
   count. It requires disclosure of models, frameworks, human involvement, selection signals and cost.
3. **Trust first, verifiable always.** Entrants run their own experiments. Maintainers do not repeat
   every run by default; frozen inputs, checkpoints, logs and provenance support later verification.
4. **One model leaderboard.** Provenance badges and filters accompany the ordinary model ranking.
   Researcher comparisons remain descriptive while their budgets are not controlled.
5. **Separate responsibilities.** The agent or RSI system chooses a research strategy; Gym executes
   and meters experiments; Arena defines admission, disclosure and publication. Gym is optional.
6. **Evidence at the claimed level.** A unit test, synthetic GPU smoke, real-data research run and
   accepted formal result establish different things. Interface-level sealing is not data isolation.

## What the first version contains

| Responsibility | Authoritative reference |
|---|---|
| Entry rules, human-involvement levels, two-PR flow and versioning | [ARENA.md](../../ARENA.md) |
| Instructions given to a research agent | [TASK.md](../arena/TASK.md) |
| Disclosure schema and starter card | [Card fields](../../results/arena/README.md) · [template](../arena/card.template.json) |
| Card validation and changed-file classification | [arena.py](../../nexttransformer/arena.py) · [tests](../../tests/test_arena.py) |
| Research tools, state, budget and execution | [NextT-Gym](../../nextt_gym/README.md) |
| Runtime separation and environment acceptance | [GPU runtime design](REPRODUCIBLE_EVAL.md) |
| Website behavior and data flow | [Observatory guide](../../website/README.md) |
| Current delivery and acceptance state | [Consolidated plan](../../PLAN.md#arena) |

[Circle Loop's card](../../results/arena/circle-loop.json) and
[research history](../arena/circle-loop.md) demonstrate disclosure of human-led, LLM-assisted work.
It remains `legacy`, L3: the research predates this protocol, used validation and CORE for selection,
and its historical matrix-cost accounting is not the current operator-level admission evidence.
A historical example does not establish a new L0 entry or a competitive researcher ranking.

## Work, in dependency order

The A0–A6, G1–G4 and GPU P0–P6 work items are tracked only in
[PLAN.md](../../PLAN.md#arena). Their dependencies follow these boundaries:

- Ordinary PR entries use the existing admission and result flow. They do not require Gym or Docker.
- A Gym candidate must become a frozen admitted method before its formal result can enter the
  leaderboard. Research weights cannot initialize that formal run.
- Short-run selection needs evidence that the chosen proxy is informative. Full-length research is
  not automatically formal training: schedules, evaluations, accounting and export must also comply.
- Training and verification require an effective data and execution boundary before unattended
  agents use Gym. G3 and runtime P6 describe this same boundary, not two separate deliverables.
- An outer-loop scaffold and a metered researcher comparison build on usable research tools and
  traceable experiments. Neither is supplied merely by exposing an MCP server.

Detailed acceptance, the chosen implementation order and any approved scope changes belong in the
consolidated plan; this page does not maintain a second status or owner table.

## When to add verification

The first-version trust model does not require routine independent reruns. A6 is triggered by
numerous submissions, consequential prizes or paper claims, inconsistent results, or margins too
small to interpret with one seed. Start with the least expensive check that addresses the concern.

A deterministic prefix replay can compare early training losses without repeating the entire run.
A seed-noise estimate addresses ranking uncertainty. A held-out evaluation can use published
checkpoints without retraining. These checks answer different questions and should not become a
blanket prerequisite for every ordinary contribution.

## Decisions for the team

Approval state, owners, compute allocations, publication choices and unresolved decisions are
recorded in [PLAN.md](../../PLAN.md#arena). Operational guides describe how to execute an approved
choice; they do not grant permission to launch a full run or publish the repository.

Changing a binding entry rule belongs in [ARENA.md](../../ARENA.md#versioning), with the required
protocol version change. In particular, accepting a Gym-scored run directly would require explicit
rules for scorer ownership and its relationship to the formal training evaluations. Existing cards
retain their entered protocol version.

## Risks

- **Proxy transfer:** short-run rankings can favor methods whose early schedule differs from their
  long-run behavior. Report proxy evidence before relying on it for selection.
- **Accounting gaps:** unfamiliar expensive operators must fail admission until their computation is
  accounted for; a custom kernel needs a reference implementation for accounting.
- **Selection leakage:** formal validation and CORE cannot be used to select research candidates.
  A PR scope check and output filtering do not control arbitrary filesystem access.
- **Execution boundaries:** submitted model code executes during verification and scoring as well
  as training. The runtime design must test those boundaries; static Python checks are not a sandbox.
- **Provenance:** human-involvement levels and unmetered research costs are disclosed claims. Traces
  and explicit unknown values let reviewers interpret them without inventing measurements.
