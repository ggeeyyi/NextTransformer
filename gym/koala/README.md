# Gym on Koala

`koala_gym.py` runs the unchanged [Gym](../README.md) on a CPU host and executes its numerical jobs on Koala GPU
clusters. It is the only Koala-specific file. Harbor, the agent and verifier containers, the Gym controller, its API,
budget and evidence all stay on the host; each research, check or final job becomes one Koala job.

```text
host (CPU, Docker)
├─ Harbor: agent container (Claude Code or Codex) and verifier container
├─ Gym controller: API, budget, state; static audits in the local bubblewrap sandbox
└─ relay (one per numerical job): publish a code bundle, submit one Koala job, follow its log,
   replay the outcome through the Gym's worker protocol
      ▼
Koala GPU pod: the candidate's training or checks, as `nobody`
```

The agent never reaches Koala. Koala and object-store credentials go to the relay only, and pods cannot connect back
to the host: the pod acknowledges the worker's operations within the job's reservation, and the relay records the
outcome.

## Setup

1. Install the Gym environment (`uv sync --project gym --extra worker`) and a worker Python with the pods' CUDA build of
   torch (`worker_python`). Log in to the `koala` CLI.
2. Build the images from the repository root:

   ```sh
   docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
   docker build -f gym/Dockerfile --target client-codex -t nt-gym-client:local-codex .   # for Codex
   docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .
   ```

3. Copy an example configuration to `.agent-work/` and replace the placeholders:
   [`examples/koala.h20.json`](examples/koala.h20.json) or [`examples/koala.h200.json`](examples/koala.h200.json).
   Unknown keys are rejected; the defaults and their meaning are in `DEFAULTS` at the top of `koala_gym.py`.
4. Export the credentials named in `relay_env` (OSS for H20 code bundles, S3 for H200) in the controller environment.
5. Record the pods' runtime once, and check the configuration's isolation backend on its cluster:

   ```sh
   gym/.venv/bin/python gym/koala/koala_gym.py --koala .agent-work/koala.h200.json probe
   ```

Keep `staging_dir` outside any git work tree: `koala code publish` packs tracked files only.

## Commands

```sh
# A Harbor trial (run_trial.py arguments), with numerical jobs on Koala. The adapter adds
# --execution-config (execution.<CONF name>.json, written next to CONF) and --worker-python from CONF.
gym/.venv/bin/python gym/koala/koala_gym.py --koala CONF trial --config gym/examples/codex.toml \
  --profiles gym/koala/examples/profiles.fixture.koala.json

# Trusted controller commands (nexttransformer_gym.cli arguments): prepare-final, launch-final, reconcile, ...
gym/.venv/bin/python gym/koala/koala_gym.py --koala CONF cli --state STATE --profiles PROFILES --platform . ...

# Koala jobs of a Gym state with GPU-hours; --stop-orphans stops jobs that no live relay follows
gym/.venv/bin/python gym/koala/koala_gym.py --koala CONF jobs --state STATE [--stop-orphans]
```

`relay` and `pod` are internal entries, started by the Gym scheduler and by the Koala job.

## Clusters

| | H20 (`"cluster": "h20"`) | H200 (`"cluster": "h200"`) |
|---|---|---|
| Pod kernel | No Landlock, no user namespaces | Landlock and user namespaces |
| `pod_isolation` | `none`: the worker runs as `nobody`; `pod_network: "deny"` refuses new network sockets with seccomp | `landlock` |
| Storage | CPFS holds `remote_root`, data and checkpoints | No shared filesystem: `remote_root` is an `s3://` prefix, mirrored in `pod_remote_dir`; `data_stage` stages data per job |
| Code bundles | OSS | S3 |
| Final runs | Several nodes, rendezvous over TCP at node 0 | One node |

Under `none`, file permissions protect reserved data. Make the validation shard, CORE-22 data, run outputs and
checkpoints unreadable to `nobody`, and list them under `probe_checks.reserved`: `probe` then fails if the worker can
reach any of them.

A research job stages only its own profile's data and never the validation shard.

## Accounting and recovery

- One Gym operation per Koala job: the intent when the pod reports that the worker starts, the charge (the quoted
  bound) when it reports the outcome. A job that ends before its worker starts is charged nothing.
- An unknown outcome keeps the reservation for the Gym's `reconcile`.
- While Koala refuses submissions because the quota is in use, the relay retries every `quota_wait_seconds` and
  honours `cancel_job`.
- The relay journals every Koala job in `<state>/koala-jobs.jsonl`, with pod start time and GPU-hours.

## Codex

`gym/run_trial.py --agent codex` runs Codex through Harbor. Pin the CLI with `--agent-version` and use the
`client-codex` image; [`gym/examples/codex.toml`](../examples/codex.toml) is a fixture example. Authenticate with
`OPENAI_API_KEY`, or set `CODEX_AUTH_JSON_PATH` to an `auth.json` that Harbor copies into the trial. Codex has no turn
or USD limit in the launcher: only the trial timeout applies, so set a provider limit for a billing cap.

## Tests

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python -m pytest -q gym/koala/tests    # fake Koala CLI and s5cmd, CPU only
```

The other scripts in `tests/` submit real Koala jobs; their docstrings give the commands. `gpu_e2e.py` reads ClimbMix
paths from `KOALA_E2E_DATA` and `KOALA_E2E_RESOURCES`.

## Limits

- Remote dispatch is outside the Gym's agreed scope (one local machine, one concurrent numerical job). This adapter
  keeps that scope: one Koala job at a time per Gym state.
- Research jobs use one GPU and the profile's batch, without gradient accumulation.
- The Gym's final runs start with `--resume never`, and a Koala job has a maximum duration. A full formal run must
  finish inside one job: on one H200 node it does not. `official_smoke` bounds the final path for tests; its results
  are never official.
