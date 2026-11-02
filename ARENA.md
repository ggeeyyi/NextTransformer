# Auto-research arena protocol

Version **`nextt-arena-v1`**. The arena opens NextTransformer to researchers that are not only people: any LLM, with or
without an auto-research framework, may search for an architecture and enter it. The entered model is trained and
evaluated under the unchanged training contract [`nextt-reference-50b-v1`](PROTOCOL.md). This document adds only the
rules for the research that precedes that run and what an entry must disclose. Machine-readable values live in
[`nexttransformer/arena.py`](nexttransformer/arena.py). Current delivery and remaining work are in
[PLAN.md](PLAN.md#arena); [the Arena design](docs/plans/ARENA.md) retains rationale.
[CONTRIBUTING.md](CONTRIBUTING.md) owns the contribution workflow and commands; this file defines the Arena-specific
entry, disclosure and two-PR requirements. The [agent task](docs/arena/TASK.md) remains self-contained.

The arena is trust-based in this first version. Entrants run their own research and their own formal training; the
maintainers do not re-run entries by default. Pinned data, a fixed seed, restored training state, published checkpoints and complete disclosure make claims
inspectable. Numerical recovery guarantees are limited to the evidence for each path; see the
[recovery matrix](docs/guide/METHODS_FRAMEWORK.md#recovery-guarantees).

## An entry

An entry is a **researcher** and the **artifact** it produced.

- The researcher is the LLM or LLMs, the framework (or none), and the people involved, with their level of involvement.
- The artifact is one registered method with an admitted configuration and, once trained, one formal run.
- Each entry has one disclosure card, `results/arena/<entry>.json` ([fields](results/arena/README.md)).
- A method version has at most one scored run. Changing the method's semantics after admission makes a new method id.

## What an entry may change

| May add | Must not change |
|---|---|
| A method plugin `nexttransformer/methods/<method>/` ([plugin guide](docs/guide/PLUGINS.md)) | Anything the [training contract](PROTOCOL.md) fixes: data, tokenizer, packing, budget, batch, schedule, seed, evaluation points |
| A calibrated `configs/<family>/<method>.json` and its proof set under `evidence/costs/` | `nexttransformer/{protocol,evaluation,runtime,training,checks,native}/` and the command-line entry points |
| Tests under `tests/` | `nexttransformer/models/backbone.py` and `nanochat/` |
| A model card `results/methods/<method>.json` and a `docs/methods/SOURCES.md` entry | Existing run records, proofs, and acceptance evidence |
| The run record, its audit under `evidence/checks/`, and a `docs/records/EXPERIMENTS.md` record | `PROTOCOL.md`, this file, the arena's tooling (`nexttransformer/arena.py`, `nextt_gym/`), CI, and the website |
| The entry's own disclosure card | Another entry's card |

A plugin needs no edit to shared model files (`registry.py`, `config.py`, `lm.py`, a family module); if a method truly
needs one, the maintainer reviews it, and it invalidates the proofs of every method that runs the edited file. The
plugin's proof fingerprints only its own package and the shared core, so the pull request carries the method's proof set
and leaves every existing configuration admitted with its model unchanged, as [CONTRIBUTING.md](CONTRIBUTING.md)
requires. `python -m nexttransformer.arena --scope origin/main` checks these rules, and
`python -m nexttransformer.protocol.fingerprint` confirms that every published method still matches its proof.

## Research

1. **Any scale, any compute, any number of experiments.** Nothing about the research is limited; all of it is disclosed.
2. **Select with the dev split or training loss.** The dev split is `shard_06541.parquet`, the last training shard in
   sorted order. A formal stream reads shards from the first one and consumes far less than the corpus, so it never
   reaches the dev shard. Research runs read from the start as well; do not train on the dev shard.
3. **The validation shard and CORE-22 are reserved.** `shard_06542.parquet` and the CORE-22 tasks belong to the
   formal run. Do not use them to choose among candidates, settings, or seeds.
4. **Literature and the internet are allowed.** Disclose whether they were used. Rediscovering a published method is a
   valid entry; the card names the closest prior methods and the difference.
5. **Research never supplies weights.** The formal run starts from scratch, as the training contract requires.

The research environment [`nextt_gym/`](nextt_gym/README.md) applies these rules for an agent: it checks a candidate's
cost before training, trains it for the first 512 updates of the formal schedule or at full length, scores it on the
dev split, meters every FLOP, and keeps validation and CORE-22 sealed from the agent. Its runs are research runs, not
pilots and not formal runs. This sealing is an interface boundary; enforced training/scoring isolation has its
own [G3 acceptance](PLAN.md#delivered-components-and-remaining-arena-work). Using the gym is optional; its ledger
and archive record what the card's research fields ask.

## Entering: two pull requests

| Pull request | Contains | After the merge |
|---|---|---|
| **1. Admission** | Method plugin, configuration admitted within ±1%, proof set, tests, model card, disclosure card with `status: admitted` | The entry is listed as *in training*. The merge date is its public preregistration: the method is frozen before any formal result exists. |
| **2. Result** | The run record from `nexttransformer.export_run` (complete, cumulative audit `matched`), its audit, an `EXPERIMENTS.md` record, and the card updated to `status: complete` with checkpoint and log links | The run joins the main leaderboard with its provenance. |

The formal run uses the admitted configuration: its record's `model_hash` must equal the configuration's. Every formal
launch counts toward `formalAttempts`, including launches that failed; failed launches are recorded in
`EXPERIMENTS.md`, as they are for every method.

## Disclosure

The card records, at minimum:

- **Researcher:** each LLM and version, the framework and version (or `none`), the people involved, and the
  involvement level below.
- **Research:** the period, candidates trained, formal attempts, GPU-hours by hardware (`null` when not metered),
  whether the internet was used, and the selection signal.
- **Novelty:** the closest prior methods and the difference.
- **Artifacts:** the final checkpoint and the training logs (required for `complete`); the research trace is encouraged.

Missing measurements are `null`, never zero.

| Level | Human involvement |
|---|---|
| **L0** | Fully autonomous: people start the task and provide compute, then do not intervene until submission |
| **L1** | Operations only: people fix environments, clusters, or launches, and make no research decision |
| **L2** | People choose among the agent's proposals or give directional hints |
| **L3** | People lead the idea or write method code; the agents assist |

## Status and verification

| Status | Meaning |
|---|---|
| `admitted` | Pull request 1 merged; formal run pending |
| `complete` | Formal run published, complete and matched; the entry ranks on the leaderboard |
| `legacy` | Predates this protocol or does not meet it; shown with full disclosure and a reason, never ranked as an arena entry |
| `retracted` | Withdrawn or found invalid; kept visible with the reason, never ranked |

Each card also says who produced the result: `self-reported` (the default), `reproduced` (an independent party
re-ran the formal run, or replayed a prefix of it and matched the logged training loss), or `maintainer`.

## Ranking

Arena entries are ranked on the main leaderboard with the same rules as every other run: measured values at update
46,875, nothing interpolated, incomplete runs unranked. The leaderboard marks each run's provenance and can filter by it.
The arena view groups entries by researcher. Research budgets are disclosed, not controlled, so comparisons between
researchers are descriptive.

## Checks

- `python -m nexttransformer.arena --check` validates every card and the records it names. CI runs it.
- `python -m nexttransformer.arena --scope origin/main` classifies a pull request's changed files as allowed, needing
  review, or blocked. CI runs it on pull requests that add or change a card.
- The existing checks still apply: `export_run --check`, configuration admission, and the test suite, whose causal
  prefix and gradient tests cover every registered method.

## Versioning

Changing any rule here makes `nextt-arena-v2`. Cards keep the version they were entered under, and existing entries are
never rewritten.
