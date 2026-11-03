---
title: Arena and gym plan
description: What exists, what remains and what is still to be decided for Arena v2 and the research gym.
---

This page tracks the work behind [Arena v2](../overview/) and the research gym (`nextt_gym/`): what is delivered,
what remains, in which order, and which decisions are still open. It reports the state of the work, not research
results; the [arena rules](../rules/) bind entries. Updated **2026-11-03**.

| Status | Meaning |
|---|---|
| **Complete** | The named deliverable exists with the evidence stated. Code, a bounded check and a published result are separate claims. |
| **Partial** | Some deliverables exist; the remaining work is stated. |
| **Pending** | Planned work or a decision without completion evidence yet. |
| **Deferred** | Kept with its trigger; not an obligation to spend compute now. |

## The roadmap in terms of this plan

| [Roadmap](../../roadmap/) item for Arena v2 | Work on this page |
|---|---|
| A research gym with MCP tools, scoring on the dev split while validation and CORE stay sealed | A1, AR-01, G1, G2 |
| Metered compute and research traces for every attempt | G4 (AR-07) |
| Self-improving research loops | A5 |
| Leaderboards by researcher | G4: descriptive views first, a ranking rule once comparable episodes exist |

Unattended agents, and with them the first fully agent-made (L0) entry, wait for the isolation work G3.

## Arena work

| ID | Status | Delivered | Remaining |
|---|---|---|---|
| A0 — initial arena | **Partial** | Rules, task, disclosure card format, template and checker, the pull-request scope check, the arena pages, provenance badges and filter, and the [Circle Loop](../circle-loop/) example (legacy, L3) | Team review and finalization of the rules and task text |
| A1 — research tooling | **Partial** | The research gym: dev-split scoring at four scales (S and C at 512 updates, M at the first 2,000, L the formal schedule), a FLOPs ledger with platform GPU-hours, an archive, 22 operations over MCP, HTTP and Python, and disclosure drafts from the archive; the dev shard's SHA-256 is pinned | Turn an attempt into a plugin without hand porting (G1) |
| A2 — proxy fidelity | **Partial** (AR-03) | Arena v1 Stage 1: a 512-update run screens candidates but does not rank them; ranking needs the 2,000-update prefix (M). Stage 2: a three-round candidate search at M. Stage 3: the two winners ported to method plugins and calibrated. Evidence: `evidence/checks/arena-v1/`, `evidence/checks/arena-v1-stage3-20261103/` | A formal run of a selected candidate needs its own authorization; report small-sample limits |
| A3 — first L0 entry | **Pending** | The agent task statement | AR-01 and G3 before unattended operation; a full-run budget; then admission and result with no human research input |
| A4 — public opening | **Partial** (AR-06) | Repository open since 2026-10-31; documentation and observatory on GitHub Pages; pull-request template with disclosure items | Decide checkpoint and artifact hosting |
| A5 — reference scaffold | **Deferred** | Gym interfaces for an outer loop | A minimal reproducible agent loop, for comparisons with the scaffold held fixed |
| A6 — verification upgrades | **Deferred; on trigger** | Trust-based rules, source and data identities, published evidence | Prefix replay, seed-noise estimates or a private held-out split, only when submission scale, inconsistent claims or close margins call for them |
| G1 — gym attempt to entry | **Partial** | The disclosure card drafted from the archive (`nextt_gym.cli disclose`); two attempts ported by hand to plugins whose initial weights and computation equal the attempts' bit for bit, with calibration, proofs, model cards and tests | Generate the plugin, configuration and proofs from an attempt without hand porting |
| G2 — one training path | **Partial** (AR-02) | G2a: both trainers' `main()` give equal losses and weights on the same rows, and the prefix scales follow the formal schedule (CPU). G2b, bounded: the same method trained 512 updates by each trainer on real data is equal for two updates and within GPU non-determinism after (mean \|Δloss\| 0.0033–0.0041 over updates 300–510) | A full-length formal-trainer run of a frozen gym candidate with its 189 validation and six CORE events, cumulative accounting and export eligibility. Gym L runs stay research-only until then and a scoring-policy decision |
| G3 / P6 — isolation | **Partial; integration pending** (AR-04, AR-05) | Static checks and API sealing in the gym; a candidate runtime and scoring tools, not yet in this repository | On one supported hardware and launcher path, show that training cannot read reserved data or change scoring; freeze the model handoff and score it independently; test training, update and resume as well as the prohibited paths. Required before unattended agents |
| G4 — metered track | **Partial; disclosure first** (AR-07) | Server-side FLOPs ledger; per run the trainer's GPU-hours and the cluster's platform GPU-hours (pod time x GPUs, queueing excluded), backfilled over the arena v1 archive; failed and cancelled runs metered; LLM use as the operator records it | Automatic LLM token and API-cost metering; researcher views stay descriptive until comparable episodes and a ranking rule exist |

## Gym status

| Deliverable | Status | Evidence and boundary |
|---|---|---|
| Research service, verification and accounting | **Complete as implementation** | 22 operations over MCP (stdio, Streamable HTTP) and Python, a blocking evaluation adapter, candidate verification and calibration, a durable store with events and idempotency, budget reserve, charge and settlement, and the archive. See the gym's README and `INTERFACE.md` |
| Arena alignment and CI | **Complete** | S is a research run, not a protocol pilot; the dev split; official results sealed for gym and imported runs; the protocol seed at L; arena entries may not change the gym; CI runs the gym's test suite |
| Interface conformance | **Complete** | Tests check the served tools, resources, prompt, event types and task card against `INTERFACE.md`; the cluster backend's tests accept only the command lines its CLI accepts |
| Agent and GPU smoke | **Complete within smoke scope** | Synthetic-token data-parallel BF16 updates with ledger settlement, and agent iterations over MCP including parent-to-child; not a real-data experiment |
| Real data on a cluster | **Complete within AR-01's bounds** | S runs with the real tokenizer and data, segmented and abrupt-stop resume from checkpoints, budget settlement, sealing including imported runs, cross-session recovery, and a disclosure draft from the archive (arena v1 Stages 0 and 3). Accepting a production path and further backends remain |
| Formal-result eligibility | **Partial** (G1, G2) | Gym results never rank by themselves. The path to the leaderboard stays: a registered and admitted method, `nexttransformer.train`, `export_run` and a disclosure card; G2a and a bounded G2b show that the gym and the formal trainer train a registered method alike |
| Enforcement boundary | **Partial** (G3) | API sealing exists, but static checks are not a sandbox: training can still read official data, and a gym L run evaluates CORE only at the end, not at the six formal events |

## Scoring runtime for G3

One pinned runtime in separate training and scoring containers. Ordinary contributions and arena pull requests need
no containers. The runtime and its tools are not yet in this repository.

| ID | Status | Delivered | Remaining |
|---|---|---|---|
| P0 — reference capture | **Partial** | Capture tools and a `dense_full` reference | Accepted reference scope and identities: source, software, kernels, inputs and settings |
| P1 — runtime build | **Partial** | Container build tooling and one built candidate with its inventory | Select one available hardware and launcher path; its manifests and support limits |
| P2 — direct scoring | **Partial** | Launcher, preflight, restricted checkpoint loading, result validator and comparison tools | Validate the real GPU launch and isolation path, precision, compiler and cache settings, and failure handling |
| P3 — numerical parity | **Partial** | The final `dense_full` checkpoint scored bit for bit equal to its training-final BPB, CORE and all 22 task scores, on the reference hardware in a user-space container | Other models, kernels and hardware need their own evidence; isolation at the container level is not yet shown |
| P4 — release | **Pending** | — | An immutable runtime identity, manifests and support limits for the first path; a clean second-machine test |
| P5 — Harbor | **Partial; deferred** | Pinned generator and controller; CPU transport validation | An optional GPU trial after the core path works; not needed to open the arena |
| P6 — gym isolation | **Pending integration** | The same item as G3 | Training, update and resume acceptance and a tested model, data and scoring boundary |

## Approved refinements

Approved by the maintainer on 2026-10-29. They refine the work above; they are not further items.

| ID | Refines | Status | Acceptance and scope |
|---|---|---|---|
| AR-01 | A1, before A3 | **Complete within its bounds** | A reviewed candidate, the real tokenizer and data, an S run, fresh-process resume, budget settlement, cross-session recovery, an inspectable archive and a disclosure draft. Bounded acceptance, not a full L0 run |
| AR-02 | G2 | **Partial** | Initialization, data, losses, optimizer and schedule compared (G2a, and G2b over 512 updates); remaining: the frozen candidate's identity, 189 validation and six CORE events, cumulative accounting and export eligibility in a full run. Prefer reusing the formal trainer; any new leaderboard rule belongs to Arena v2 |
| AR-03 | A2 | **Stages 0–3 done** | Representative families and stages with preregistered correlation, expansion and longer-prefix criteria; expand only from that evidence; never claim all-method rankings from a small subset |
| AR-04 | G3 / P6 | **Pending** | Training cannot reach official data or alter scoring; an independent verifier and an immutable model handoff; tests of prohibited access as well as the successful path. Ordinary pull requests stay container-free |
| AR-05 | P1–P4 | **Pending** | One available hardware and launcher path, real GPU execution, isolation and a clean consumer; unsupported paths are stated; Harbor and extra backends wait |
| AR-06 | A4 | **Partial** | Open the ordinary pull-request track first, without waiting for G3 or A3. The repository is open; artifact hosting remains |
| AR-07 | G4 | **Partial** | Measured research compute (FLOPs, trainer and platform GPU-hours, wall time) per run, failed and cancelled runs included, in the disclosure draft with unknown values null; automatic LLM metering and a ranking rule remain |

## Order

Run acceptance continues alongside the contribution path. AR-01 and the first isolated path (G3, P6) come before any
unattended agent (A3). A2's evidence decides where further research compute goes. G4 starts with disclosure. Nothing
here implies a broad hardware or seed sweep.

## Open decisions

| Decision | What needs agreement |
|---|---|
| Arena rules review | The editable surface, the dev split, the two-pull-request process and the L0–L3 classification |
| Checkpoint hosting | Organization-hosted or entrant-hosted artifacts, and accessible evidence for complete entries |
| Research execution | Owners and hardware for A1 and A2, the first supported gym hardware and launcher path, operators and budget caps |
| First L0 entry | When to fund A3; opening the pull-request track did not wait for it |
| Gym-scored formal results | If G2 replaces a separate formal run: who operates scoring, and whether its score replaces or must agree with the training-final evaluation. A new rule goes into Arena v2; ordinary entries are unaffected |
