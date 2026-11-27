# NextT-Gym

An environment in which research agents iterate on language-model **architectures** and try to reach a
lower validation BPB at the **same training FLOPs per input token** as the frozen upstream SSSL cost anchor
(4.67239832544785 GFLOPs/token, ±1%), under the frozen data, tokenizer, batch, optimizer and seed of
the [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/). The full-attention `dense_full` model is the quality baseline;
it is distinct from the historical SSSL computation anchor.

This page describes the implemented interface and operating workflow.
The [arena and gym plan](../docs/src/content/docs/arena/plan.md) holds delivery state, remaining acceptance and open decisions;
[INTERFACE.md](INTERFACE.md) specifies the 22 operations and [spec.md](spec.md) the submission contract.

The gym is the research environment of the [auto-research arena](https://interactive-active-ai.github.io/NextTransformer/docs/arena/overview/): it carries out the research that
precedes an arena entry and meters it. It is a separate layer. It imports the upstream modules and changes none of
them. The folder lies outside every upstream source fingerprint (`science_hash`, `source_hash`, `source_manifest`),
so cost proofs and frozen runs stay valid. Results from the gym form their own cohort and never rank by themselves.

## Scales

| Scale | Updates (tokens) | Schedule | Relation to the protocol |
|---|---|---|---|
| S | 512 (≈0.54B) | first 512 updates of the L schedule and data prefix | shaped like a pilot, but a research run: not a protocol pilot, no pilot quota |
| C | 512 (≈0.54B) | S's updates and tokens under their own compressed schedule (warmup 40, the last 65% decaying to 5%), so a run ends after a decay | a research run |
| M | 2,000 (≈2.10B) | first 2,000 updates of the L schedule and data prefix, evaluated every 256 updates | a research run |
| L | 46,875 (49.152B) | the formal schedule | a research run at full length; not a formal result |

S is matched through its parent: the FLOPs budget is the L run's average, so a method whose first updates are
a dense burn-in is still matched at S. By default an L run requires a succeeded S run of the same attempt, and L
runs use the protocol seed (`seed_offset` is for S runs).

## What the agent sees

Agents select with training loss and the **dev split**, `shard_06541.parquet`, the arena's dev split, which no
training stream reaches. L runs also evaluate the official validation shard and CORE-22, but the [arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/) reserve
those for the formal run, so the gym keeps them **sealed**: the tools, events and logs omit them for gym runs, and
the official leaderboard lists only imported upstream results. The operator can reveal an attempt
(`python -m nextt_gym.cli reveal ATTEMPT_ID --reason ...`), for example once it has been entered.

## From the gym to the leaderboard

An attempt enters the leaderboard only through the arena rules: it becomes a registered method with an admitted
configuration (admission pull request), its formal run is trained with `nexttransformer.train` and exported with
`nexttransformer.export_run` (result pull request), and its disclosure card reports the research. The gym's
ledger and archive hold that research record: every attempt, run, and FLOP.

## Loop

```
observe (leaderboards, archive, budget) -> propose model.py + config.json -> verify V0-V3 (calibrate)
  -> quote -> launch on a backend -> events -> result, ledger settlement -> archive -> observe ...
```

| Piece | File |
|---|---|
| Task, scales, visibility, policy | `task.py` |
| Model contract | `spec.md`, `contract.py`, `templates/` |
| Optimizer groups by parameter role (upstream hyperparameters) | `roles.py` |
| FLOPs audit on the upstream `Ledger` (bit-identical with every upstream proof) | `audit.py` |
| V0 static, V1 structure, V2 correctness/causality, V3 audit | `verify.py` |
| Calibration of declared tunable integers | `calibrate.py` |
| Upstream methods as gym attempts | `upstream_model.py`, `importer.py` |
| SQLite store, FLOPs ledger (reserve, charge, settle), archive, leaderboard | `store.py`, `ledger.py`, `archive.py` |
| Run spec, trainer (mirrors `nexttransformer/train.py`), code bundles, remote entry point | `runspec.py`, `trainer.py`, `bundle.py`, `remote.py` |
| Cluster backends: local, ssh, koala (modelarts: command builder only) | `backends/`, `backends.example.toml` |
| Reading run files on another machine (mounted directory, ssh, `koala exec`) | `backends/reader.py` |
| Koala: backend, platform operations (code bundles, job states, stop, exec), segment driver, pod entry | `backends/koala/` |
| The 22 agent tools | `service.py` |
| MCP server (tools, `nextt://` resources, prompt) | `mcp_server.py` |
| Admin CLI (imports, budget, anchors, monitoring, unsealed leaderboard, reveal) | `cli.py` |

## Setup

```bash
uv sync --frozen --extra cpu --group dev        # or --extra gpu --extra upstream for real runs
uv pip install --python .venv/bin/python -r nextt_gym/requirements.txt
export NEXTT_GYM_HOME=/persistent/nextt-gym     # state: SQLite, attempts, run directories
cp nextt_gym/backends.example.toml nextt_gym/backends.local.toml   # edit; ignored by git
python -m nextt_gym.cli import                  # upstream methods + completed formal results
python -m nextt_gym.cli budget                  # default cap: ten L runs (~2.30e21 FLOPs)
```

Each cluster also needs the upstream environment and resources: `launch/setup_environment.sh`,
`python -m nexttransformer.prepare`, an upstream site JSON ([configuration guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/configuration/)),
and the dev-split shard `shard_06541.parquet` next to the training data. Runs ship the **committed** code: `bundle.py` refuses to
build a bundle if `nextt_gym/` has uncommitted changes.

## Connecting an agent

```bash
python -m nextt_gym.mcp_server                       # stdio
python -m nextt_gym.mcp_server --http --port 8780    # Streamable HTTP
```

For Claude Code, see `mcp.example.json`. Keep `python -m nextt_gym.cli monitor` running. It polls the
backends, charges the ledger and emits events. The MCP tools also poll, lazily, whenever runs or events
are read.

## Plugging into an RSI system

The gym is the environment; the outer loop (which agent, how many iterations, when to wake it) belongs to the
RSI system. The service interfaces expose the same 22 operations; the evaluation adapter
combines them into one call per candidate:

| Entry point | Use |
|---|---|
| `python -m nextt_gym.mcp_server` (stdio) | an agent host that spawns tools locally (Claude Code: `mcp.example.json`) |
| `python -m nextt_gym.mcp_server --http --port 8780` | a remote or long-lived orchestrator; endpoint `http://HOST:8780/mcp`, stateless Streamable HTTP |
| `from nextt_gym.service import GymService` | a Python orchestrator in the same environment |
| `from nextt_gym.adapter import evaluate` / `python -m nextt_gym.adapter` | frameworks that call an evaluation function or command (evolutionary / program search); one blocking call per candidate |

Contract for an orchestrator:
- **Expensive actions return immediately.** `submit_attempt` returns `verification_id`, `launch_run` returns
  `run_id`, and both return `events_cursor`. Resume from `wait_events(events_cursor, types)`, or poll
  `get_verification` / `get_run`. A run can take hours to days, so do not hold an agent session open for it:
  end the session after `launch_run` and wake a new one on `run.finished`.
- **State lives in the gym store**, not in the session. A new session reads `get_budget`,
  `get_leaderboard`, `list_attempts` and the attempt notes, so iterations can be independent processes.
- **Writes are idempotent** through `idempotency_key`, so retries are safe.
- **Errors are structured** (`{"error": {code, message, retryable, details, hint}}`). Retry only when
  `retryable` is true.
- **The FLOPs budget is enforced server-side** (reserve, charge, settle). An orchestrator may add its own
  limits, for example iterations or API spend.
- **Run state advances** while any tool reads runs or events, or under `python -m nextt_gym.cli monitor`.
  Keep `monitor` running when no agent is attached.

If the host sets HTTP or SOCKS proxy variables, exclude the gym host from them (`NO_PROXY`). Otherwise MCP HTTP
clients route localhost traffic through the proxy.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 python -m pytest nextt_gym/tests -q
```

CI runs it in the `gym` job. The suite runs on CPU and covers:
- role groups equal upstream for all methods;
- audits reproduce the upstream proofs;
- the verifier rejects causality leaks and escape hatches;
- store, ledger and archive;
- sealed official results, the operator's reveal, and the protocol seed at L;
- trainer smoke with exact resume and budget stop;
- the local backend;
- the Koala backend against a stand-in that accepts only the real CLI's command lines (`tests/_fake_koala.py`;
  compared with the installed `koala --help` where there is one);
- the served tools, resources, prompt, event types and task card against [INTERFACE.md](INTERFACE.md);
- the whole service loop.

The verifier's static checks are not a sandbox. API sealing hides results but does not remove
official data from the training process. Until a data and execution boundary separates submissions
from the official data, use reviewed submissions with trusted agents; the operating commands above
do not provide that separation by themselves.
