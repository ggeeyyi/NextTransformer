# Architecture research

## Goal

Design one language-model architecture that, trained from scratch under the frozen training contract, reaches a
lower validation bits-per-byte (BPB) or a higher CORE-22 score at update 46,875 than the published runs. Both metrics
are reported.

After this session, your single frozen submission is trained from scratch under that contract, independently of this
session. During the session you work within a fixed research compute allocation and see training loss and
development-split BPB (`dev_bpb`) only. How you explore the architecture space and allocate your compute budget is
entirely up to you.

## What is fixed

| Control | Value |
|---|---|
| Data | ClimbMix at a pinned revision; one tokenizer (32,768 tokens); 2,048-token context; one global sample order |
| Budget | 49.152B input tokens · 1,048,576 tokens per update · 46,875 updates · seed 42 |
| Compute | Training cost within ±1% of 4.67239832544785 GFLOPs per input token, by the operator ledger, including backward and optimizer work |
| Optimizer | Muon for block matrices, AdamW groups for embeddings, head and scalars; fixed learning rates and schedule |
| Evaluation | Validation BPB every 250 updates; full CORE-22 at 8K, 16K, 24K, 32K, 40K and 46,875 updates |

Parameters, memory and speed are not controlled; they are reported. Gym owns the training profiles, optimizer, data
and scoring.

## What you submit

A native ModelConfig and its plugin source. Do not submit machine settings or training commands.

Candidate packages follow the `proposal.json`, `architecture.json`, `plugin/<method>/method.json` and
`plugin/<method>/__init__.py` contract; `/opt/gym/make_candidate.py` is an example. Define native `model`, `mixer`,
`ffn` or `block` hooks, with optional local Python modules. A `model` hook must subclass `ReferenceLM`. The
`plugin-contract` and `reference-model` references give the exact interfaces.

`describe_space.candidate_contract` gives the native ModelConfig JSON schema, defaults and merge order, and each
profile's `model_config_constraints` gives its fixed values. `architecture.json` contains native `ModelConfig` fields
and declared plugin `options`. Changing source or configuration creates a new candidate. For config-only changes,
re-upload unchanged source with a new `architecture.json`, or use `calibrate_candidate.grid`: each item replaces the
entire architecture object, not a patch. Target updates and GPU placement go to `quote_experiment`, never into the
model configuration. Gym does not parse or translate model expressions; research and final workers load the same
source without changes.

## Rules for the research

1. The research allocation (`get_budget`) covers all of this session's Gym numerical work: training, evaluation,
   checks, and failed or cancelled jobs. Within it, any experiment length and any number of experiments.
2. Choose among candidates with training loss or `dev_bpb`. The validation shard and CORE-22 are reserved for the
   formal run and are unavailable during the session.
3. Literature and the internet are allowed. Rediscovering a published method is a valid entry. In the submitted
   candidate's `proposal.json`, `hypothesis` names the closest prior methods and the difference.
4. Research never supplies weights: the formal run starts from scratch. Resume only from checkpoints returned by this
   service.
5. Do not install or execute candidate code in the agent workspace. Gym rejects arbitrary training commands,
   trainer, data or environment overrides and remote shell jobs.

No person intervenes or answers during the session. The session ends when you stop; jobs still running are stopped.

## The Gym

Read `/run/gym-auth/credential.json` for your session-scoped endpoint and token. Use the preconfigured `gym` MCP
tools or import `GymClient` from `nexttransformer_gym`; both use the same HTTP API and session budget, and you may
script them. `describe_space` lists the operations, profiles and contract; `get_resources` reports the GPU inventory,
the configured allocation, the memory allowance and current availability; `list_components` lists the available
components.

`list_experiments(scope="published")` and `read_experiment` show past runs, their results and model configurations;
`source_reference_ids` lead to architecture code through `read_reference`. Published runs report validation-shard BPB
and CORE-22 from their formal runs; your jobs report `dev_bpb` on the dev shard, `shard_06541.parquet`. Gym serves
current reference files, which may differ from the code a published run used; each record states its provenance.
`scope="session"` shows your own jobs, results and candidate source. No repository checkout or Git history is
provided.

`inspect_cost` and `calibrate_candidate` are static cost analysis and use no research compute. `quote_experiment`
prices a job before it runs and takes the absolute `target_update`. A research job uses the whole configured GPU
allocation; `gpu_id` placement applies to single-GPU profiles, and omitted placement is automatic. A job waits while
the allocation is busy; `get_job` reports its state and waiting reason.

## How to submit

1. Choose one candidate and the final profile that Gym provides.
2. Call `validate_candidate` with that `profile_id` and wait for it to finish. It is charged to the research
   allocation. All checks must pass and `admitted` must be true.
3. Call `finalize`.
4. Write `/workspace/submission.json` containing exactly your `session_id`, the server-returned `submission_id` and
   `candidate_hash`.

A separate verifier checks this receipt against authenticated server evidence. Acceptance means a valid submission,
not model performance. The verifier never starts final training. Do not write reward files.
