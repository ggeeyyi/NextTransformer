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

## Research GPUs

| Setting | Meaning |
|---|---|
| `research_gpus` | GPUs of one research job (default 1). The Gym's execution configuration lists that many virtual cards; the Gym leases them as one group and splits each update across them, and the pods map them onto their own GPUs in order. |
| `research_gpus_per_node` | GPUs per node (default 8). A larger `research_gpus` (a multiple of it) spans `research_gpus / research_gpus_per_node` nodes; see below. |
| `research_mesh_timeout_seconds` | How long a multi-node worker waits for the other nodes at a collective (default 3600: compilation, checkpoint loads). |
| `research_microbatch_size` | The Gym's `research_microbatch_size`: rows per GPU per forward/backward pass (default 1, as in the Gym). An update of the profile's `batch_size` rows accumulates gradients over its passes. |

A profile's `batch_size` is the global rows per update, whatever the GPU count; `512` with `sequence_len` `2048` is
the protocol's 1,048,576 tokens. The Gym's quote-time memory bound decides the largest microbatch it accepts: for the
dense anchor shape on a 128 GiB limit that is 10 rows, and 8 divides the protocol's 512 rows evenly. A check with a final profile uses one GPU, and the Gym runs one
numerical job at a time per state.

### Several nodes

With `research_gpus` 32 and `research_gpus_per_node` 8, one research job is one Koala job of 4 nodes x 8 GPUs. The
Gym splits each update over the 32 cards exactly as on one machine with 32 GPUs (the quote, its FLOPs and memory bound
do not change); node `k` trains cards `8k .. 8k+7`:

1. The nodes meet through the job's directory on shared storage (as final runs do), and each pod runs a trusted NCCL
   check (`pod-mesh-check`) as the worker identity before node 0 reports the job ready. A mesh that does not form
   ends the job before its worker starts, so it is charged nothing. The pod's log records the measured bandwidth.
2. Each node's worker process joins a one-process-per-node NCCL group on its first GPU, runs every collective once
   (`NCCL_RUNTIME_CONNECT=0`), and only then installs the `pod_network` `deny` filter for every thread before it loads
   candidate code: the worker keeps its connections to the job's own nodes and cannot open any other socket.
3. Each node sums its local replicas' gradients with CUDA copies, as on one machine; the node sums are all-reduced, and
   every node applies the same optimizer update. After each update the nodes compare a fingerprint of their state; if it
   ever differs they take node 0's (the result's `execution.cross_node_resyncs` counts it). Dev evaluation is split the
   same way.
4. Node 0 alone reports to the relay and writes the checkpoint. The other nodes leave markers under
   `<remote>/research/`; a node stops its worker when another node failed, and node 0 then settles a known failure.

It needs a shared `remote_root` (H20's CPFS) and `pod_isolation` `none`: the Landlock backend unshares the network
namespace.

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

## Claude Code

`gym/run_trial.py --agent claude-code` installs Claude Code 2.1.327 in the client container unless `agent.version`
pins another. `agent.reasoning_effort` (`low`, `medium`, `high`, `xhigh` or `max`) reaches `claude --effort`. Credentials
come from the controller environment: `ANTHROPIC_API_KEY`, or a subscription token (`CLAUDE_CODE_OAUTH_TOKEN` with
`CLAUDE_FORCE_OAUTH=1`).

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
  keeps one Koala job at a time per Gym state; a research job may span several nodes (above).
- Research workers run in `research_dtype` (float32 by default) and compile only with `research_compile`.
- A Koala Normal job has a maximum duration (48 h for one H20 node, 27 h for four): give research profiles a
  `timeout_sec` below it, and continue longer runs from their checkpoints.
- The Gym's final runs start with `--resume never`, and a Koala job has a maximum duration (`koala quota` shows it).
  A full formal run must finish inside one job. `official_smoke` bounds the final path for tests; its results are
  never official.
