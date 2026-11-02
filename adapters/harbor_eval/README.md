# Harbor adapter for fixed-runtime scoring

This is the operational guide for optional orchestration of the arena Gym scorer.
[PLAN.md](../../PLAN.md#arena) owns acceptance and release scope; the
[runtime design](../../docs/plans/REPRODUCIBLE_EVAL.md) explains the boundary.
Ordinary PR submissions do not require Harbor. The adapter targets **Harbor 0.9.0**, isolated
from the GPU image. Its complete controller
resolution is hash-locked in `requirements.lock` for Python 3.12/Linux. Installation uses
the Huawei Cloud mirror by default and never changes the scientific GPU lock.

## Install the controller

```bash
HARBOR_PYTHON=/absolute/python3.12 HARBOR_VENV=/absolute/controller-venv \
  UV_BIN=/absolute/uv-0.9.5 bash adapters/harbor_eval/setup.sh
```

Python itself can be installed with uv's `UV_PYTHON_INSTALL_MIRROR` pointed at
`https://registry.npmmirror.com/-/binary/python-build-standalone`; keep the installation
under an explicit site directory. The validated controller interpreter was Python 3.12.12.
The lock can be regenerated deliberately with uv 0.9.5 `pip compile --generate-hashes
--only-binary :all:` against `requirements.in` using the same mirror. Review the resulting
changes; running a task must not resolve new packages.

## Generate and run a task

Prepare the same reference/input contract and site configuration used by the
[direct scoring path](../../environments/eval/SCORING.md), then generate a fresh task:

```bash
python3 -m adapters.harbor_eval.generate --config /absolute/scoring.local.json \
  --output /absolute/new-scoring-task
LITELLM_LOCAL_MODEL_COST_MAP=True /absolute/controller-venv/bin/harbor run \
  --path /absolute/new-scoring-task --agent nop --env docker --n-concurrent 1 --no-delete \
  --jobs-dir /absolute/harbor-jobs
```

Use `--env docker` with the pinned version; `--environment-type` is not supported by this
release. The controller may inherit host proxy settings; configure
those locally. `LITELLM_LOCAL_MODEL_COST_MAP=True` avoids a model-cost-map download for this
nop workflow. No LLM service is used.

## GPU allocation and execution boundary

**GPU requests are explicit.** Inspection of the installed 0.9.0 wheel shows its Docker
resource writer handles CPU/RAM, not the newer upstream GPU-count translation. Therefore
the generator writes Compose `driver: nvidia`, eight allocated `device_ids`, and GPU
capabilities itself. `[environment].gpus = 8` alone is not relied upon. The host still needs
the NVIDIA runtime and an actual allocation; Harbor does not reserve exclusive devices.

**Always use `--no-delete`**: Harbor 0.9.0 otherwise runs Compose cleanup with
`--rmi all`, removing the local scoring image. With `--no-delete`, it removes the trial
containers without deleting the image.

The task pins the resolved local image ID and uses read-only inputs,
private `/work` caches, no network, and the caller's UID. Harbor owns the log mounts.
Harbor 0.9.0 uploads `/tests` using `docker cp`; the tested Docker daemon rejects this
on a read-only root filesystem even with a `/tests` tmpfs. This adapter therefore permits
a writable container layer, while running as the host's non-root UID with all capabilities
dropped. The image-owned `/opt` interpreter, packages, and source remain root-owned, and
preflight verifies source/package identity. Inputs remain read-only. This is a bounded
compatibility difference from the direct launcher, not an arbitrary-agent sandbox. Host
bind mounts make this adapter local-Docker-only.

The verifier calls the exact same `environments.eval.run` entry point as direct Docker.
It writes reward `1` only after preflight, full evaluation, and result validation succeed.
Reward means completeness, not BPB/CORE quality. This is a `nop` workflow for reviewed
checkpoint/source artifacts, not an agent training or arbitrary-code isolation environment.

## Evidence and acceptance

Schema and CPU transport checks are distinct from GPU scoring acceptance. A transport test
proves container start, test upload, logs and reward handling only. A supported GPU path needs
a complete trial and direct-versus-Harbor comparison on the same accepted inputs and allocation.
See [PLAN.md](../../PLAN.md#arena) for current coverage and the
[tooling record](../../docs/records/SCORING_TOOLING.md) for the dated CPU transport evidence.
That record's early local-image removal and stopped rebuild do not describe every later image;
the [runtime design](../../docs/plans/REPRODUCIBLE_EVAL.md#8-acceptance-when-can-the-gym-use-the-image)
links the subsequent A800 parity experiment and its limitations.
