---
title: Roadmap
description: What is coming next to NextTransformer.
---

## Arena v2: agent-native research

Arena v2 is live. The research gym ships in [`gym/`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/gym/README.md); the experiment plan below is what it is used for next.

1. **Auto-research tasks in Harbor format.**
   - The repository runs as the gym's sidecar service and exposes its interfaces to the model.
   - The model develops inside a closed container.
   - The agent can read the repository's full history and search the web to find the best architecture.
2. **Separate research from formal evaluation.**
   - Research uses ClimbMix training data and the dev shard, `shard_06541.parquet`.
   - The formal validation shard, `shard_06542.parquet`, and CORE-22 are reserved for formal reporting, not candidate selection.
3. **The submission is a single file.** The gym then runs the final training itself.

4. **Gym interfaces.**
   - **Connecting:** the container receives an endpoint and token in `/run/gym-auth/credential.json`. Each call is a JSON
     `POST /v1/call` with an operation name and parameters; the Python `GymClient` wraps it without importing any
     training code.
   - **Information:** `describe_space`, `list_components`, `read_reference`, `get_resources` for the search space,
     available components, read-only reference code and resource limits.
   - **Pre-check:** `validate_candidate`, `calibrate_candidate`, `debug_candidate`, `inspect_cost` check a candidate's
     loading, FLOPs/token and parameter budget before it spends research compute.
   - **Fixed experiments:** `run_experiment` trains a candidate under a fixed profile and can continue from a checkpoint;
     `get_job`, `get_result`, `cancel_job` track it.
   - **Budget:** `quote_experiment` prices an experiment before it runs, and `get_budget` shows what remains. Every run
     draws on a fixed research allocation; when it runs out, experiments stop.
   - **Submission:** `finalize` freezes one candidate as the final file.

### Gym experiment plan

- **Models and harness:** Opus, GLM and DeepSeek, each researching independently through Claude Code and Gym MCP. Pin model and client versions before launch.
- **Initial settings:** share the same code/reference snapshot, ClimbMix data order, tokenizer, 2,048-token context, seed 42, 1,048,576 tokens/update and original optimizer/learning-rate protocol. Research uses training loss and dev only; formal validation and CORE-22 remain reserved.
- **Research strategy:** the model chooses its initial architecture, controls, candidate count, experiment lengths, search and stopping strategy. Do not prescribe training steps or search rounds.
- **Model calls:** no spending cap; record tokens and actual cost.
- **Research compute cap:** **1.96×10²⁰ FLOPs per model session**, shared across training, evaluation, checks and failed attempts, without per-candidate quotas. One session for each of the three models gives a total research cap of **5.88×10²⁰ FLOPs**. Final formal training is separate.
- **Budget basis:** Arena v1 Stage 2 recorded 17 attempts, including cancellations and retries. Estimate capacity at 20 anchor-equivalent 2,000-update runs: `20 × 2,000 × 1,048,576 × 4.67239832544785e9 ≈ 1.96×10²⁰ FLOPs`. This sizes the allocation; it neither prescribes 20 experiments of 2,000 updates nor reconstructs v1's actual FLOPs ledger.
- **Scheduling reference:** v1 search and transfer checks used `549.7 + 42.1 = 591.8` GPU-hours. With 20% headroom, allow **about 710 GPU-hours per model, or 89 hours on a comparable eight-GPU setup**. Actual time depends on the model's experiments and throughput; GPU-hours are not an additional hard cap.
- **Goal:** compare the architectures each model discovers autonomously under equal research compute, then verify final submissions under the same formal training protocol.

### Arena results example

Start with a [frontend example](../../arena-example.html): model, harness, research budget and status at the top; candidate dev results and baseline improvements in the middle; expandable architecture changes, formal results and evidence below.
Use historical Arena v1 research data for the example. Mark Opus, GLM and DeepSeek as planned and leave formal scores empty until real run records are connected.

## 2K scaling

The T3 sweep compares model size and tokens for Vanilla Dense, Vanilla MoE, Dense KDA–DSA and Fixed Recursive at a fixed training budget, with a 2K window and seed 42. The T3 configurations and budgets are public. T4 refines the fit, T5 checks forecasts, and C200 is the final blind test. See the [scaling overview](../scaling/overview/).

## MoE track

The [MoE track](../moe/overview/) puts sparse models under the same contract, at two Marin-derived sizes. The
[240 TPP comparison](../moe/protocol/#240-tpp-comparison) is published: d768 admits baseline, HC, mHC and SimpleHC
over 14.07B input tokens, and d1024 admits the baseline over 34.06B. Formal runs come next, scored by MoE-CORE and
published through the same [result pull request](../guides/submit-a-result/) as any other contribution. Further sizes
or longer budgets enter as new tracks, each with its own fingerprint.

## Upcoming results and evidence

- **Hybrid results:** GDN + QSA. Its model card is already in the observatory, marked
  "Results coming soon". Each result is published through the same [result pull request](../guides/submit-a-result/)
  as any contribution.
- **Audits:** independent run-directory audits for the looped and sparse runs; pilot and resume acceptance evidence
  for the published hybrids.

## Planned

- **Checkpoints and the tokenizer on the Hugging Face Hub.**
- Port the Others implementation (MQA, GQA, MLA) into the current code.

Have an idea or want to help? Open a [discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions).
