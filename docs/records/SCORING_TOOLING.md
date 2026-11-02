# Scoring tooling implementation: 2026-10-28

Status: **implemented, CPU-tested, Harbor transport checked; GPU scoring not run**.
At the user's request, further runtime evaluation and image rebuilding were stopped.
Training integration is explicitly deferred. This record does not supersede the earlier
[candidate build evidence](GPU_RUNTIME_BUILD.md) or establish an official scoring release.

## Implementation

- Reference capture and credential-free materialization of an existing HF kernel snapshot.
- Full-content input manifests, checkpoint/source/package checks, and per-rank GPU/backend/NCCL preflight.
- An offline direct Docker launcher with explicit GPU IDs, fresh outputs, timeout cleanup,
  read-only inputs, private caches, and a shared container scoring entry point.
- Scorer-only restricted checkpoint deserialization; ordinary training/resume code is unchanged.
- Evaluator options to check actual attention dispatch and preserve phase-specific sparse backends.
- Strict full-BPB/CORE result validation and comparison using operator-preregistered tolerances.
- A reviewed source-layer build tool; its image integration remains unvalidated.
- A Harbor 0.9.0 generator using explicit Compose GPU IDs and the same scoring entry point,
  with a separate fully hash-locked Python 3.12 controller environment.
- Arena scope protection for `environments/` and `adapters/`.

The initial wrapper deliberately supports registered methods, a final step-46,875 checkpoint,
BF16, and a single eight-GPU node. Its kernel snapshot is a separate read-only, hash-verified
artifact. Native-track and multi-node scoring, training isolation, and intermediate-logit
comparison are not implemented by this wrapper.

## Checks completed before the request to stop runtime work

The bounded CPU suite passed **138 tests**, with **6 CUDA-only skips**:
`test_eval_environment`, `test_models_api`, `test_protocol`, `test_arena`, and `test_reference`.
Coverage includes incomplete/nonfinite/mismatched score rejection, artifact tampering,
restricted pickle rejection (including the historical model API under the scorer's environment),
GPU allocation/configuration checks, kernel snapshot materialization, model round trips,
and existing mathematical/protocol regressions.

The scientific implementation hash is unchanged from main:
`a2bff1f3a7f4afe3215fb965a7c8278357d9a624092fe328cc98a849287e998e`.
No model algorithm, frozen upstream file, dependency lock for the GPU stack, or historical
proof was changed. No historical evidence was relabelled.

Harbor 0.9.0 and 105 controller distributions installed from the Huawei Cloud mirror with
hash verification; `uv pip check` passed. The interpreter was Python 3.12.12 from npmmirror.
The generated production schema passed the installed Harbor `TaskConfig` model and
`docker compose config`. A real **CPU-only transport fixture** using an already-cached
Python 3.12 image completed with one trial, zero exceptions, and reward 1. This fixture
checks container lifecycle, verifier upload, logs, and reward handling; it does not execute
NextTransformer scoring or prove direct/Harbor numerical equivalence.

Local logs: `.agent-work/eval-runtime/tests.log`, `harbor-install.log`, and
`harbor-jobs/2026-10-28__13-45-58/`. Subsequent reference-contract guard and documentation
edits received syntax/static checks only, respecting the user's request not to run more.

## Findings and artifact state

Harbor 0.9.0 requires `--env docker`, and its Docker writer does not translate the task's
GPU count into a device request. The generator therefore writes explicit GPU device IDs.
Its upload operation fails against a read-only root filesystem on this daemon, including
with a `/tests` tmpfs. The adapter uses a non-root user, root-owned runtime files, a writable
container layer, and read-only inputs; the direct Docker launcher retains its read-only root.
The generated verifier script is executable before upload.

Harbor's default cleanup uses `docker compose down --rmi all`. An early transport attempt
removed the earlier local `nextt-runtime:cu128-candidate` image. Commands now require
`--no-delete`, which removes trial containers while retaining images. No unrelated image
was targeted. Rebuilding the updated scoring candidate was started via the declared mirrors,
then stopped at the user's request; **no new scoring image was produced, and the earlier
local image must be rebuilt before reuse**. Historical build identity remains recorded as
an observation, not a promise that the local artifact is still present.

## Remaining release gates

An accepted reference environment, immutable checkpoint/resources/data/kernel snapshots,
and preregistered tolerances are still required. No GPU kernel, NCCL, complete BPB/CORE,
reference parity, source-layer image integration, clean rebuild, or second-machine consumer
test was completed here. No image was published. Apt snapshot pinning and the maintainer's
official-scoring ownership/final-evaluation policy decisions are also outstanding.

Use the [scoring instructions](../../environments/eval/SCORING.md) and
[Harbor instructions](../../adapters/harbor_eval/README.md) when execution is resumed.
