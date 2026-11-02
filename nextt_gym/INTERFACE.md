# NextT-Gym interface

The same operations are served over MCP stdio, MCP Streamable HTTP (`/mcp`, stateless), and in Python
(`nextt_gym.service.GymService`). An error is returned as `{"error": {code, message, retryable, details, hint}}`.

## Tools (22)

### Observe (read-only, no FLOPs charged)

| Tool | Input | Output |
|---|---|---|
| `get_task` | none | The task card, described below |
| `get_budget` | none | FLOPs cap, used, reserved and available (floats plus exact strings); breakdown by kind and scale; GPU-hours per GPU type; rate counters |
| `get_leaderboard` | `scale` (S, C, M, L), `split` (dev, official), `limit`, `include_imported` | Best succeeded run per attempt, ranked by validation BPB. The official split lists revealed attempts only: official results of gym and imported runs are sealed. Also `entries_per_split`, `notes` (for example, when there is no anchor yet) and `accepted` on imported results |
| `list_attempts` | filters: `status`, `origin`, `tags`, `ancestor_of`, `descendant_of`; sort: `created`, `best_S`, `best_L`; `limit`, `cursor` | Attempt summaries and `next_cursor` |
| `get_attempt` | `attempt_id`, `include` ⊂ {files, audit, runs, lineage} | Attempt with code, audit, runs, lineage and notes |
| `get_run` | `run_id` | Status, progress, consumed FLOPs, latest metrics (loss, step time, tokens/s, peak memory) and final metrics; `extra.restarts` lists restarts after a failed segment; after a restart, `progress` and `consumed` go back to the checkpoint while the updates since it are replayed (the budget keeps the highest charge until the run settles at its final total). Local paths are omitted |
| `get_run_curve` | `run_id`, `metric` (train_loss, val_bpb), `split`, `max_points` | Points `{step, input_tokens, value}` |
| `get_run_logs` | `run_id`, `tail`, `grep` | Redacted log lines |
| `compare_attempts` | `attempt_ids` (2 to 8), `scale`, `metric` | Curves aligned by trained tokens; final values and deltas |

### Propose and verify

| Tool | Input | Output |
|---|---|---|
| `estimate_cost` | `files` (`model.py`, `config.json`, optional `kernels.py`, `NOTES.md`) | Dry run of V0–V3: `ok`, `audit`, per-level errors. No attempt is created; limited per hour |
| `submit_attempt` | `files`, `title`, `hypothesis`, `parent_ids`, `tags`, `idempotency_key` | `attempt_id`, `verification_id`, `events_cursor`. Verification runs in the background. Near-duplicates (same config and near-identical code) are rejected |
| `get_verification` | `verification_id` | Per-level status, seconds and errors. The audit is included when V3 passed |
| `calibrate` | `attempt_id`, `knobs` (declared tunable integer fields), `target_relative_error` | A new verified child attempt, with `changed`, `audit` and `verification_id` |

### Execute

| Tool | Input | Output |
|---|---|---|
| `list_backends` | `gpu_type` | Backend id, kind, GPU type, allowed world sizes, status, queue estimate and segment limit |
| `quote_run` | `attempt_id`, `scale`, `backend_id`, `world_size` | FLOPs to reserve (train, eval, margin), estimated GPU-hours and wall time, budget after reserve, and `feasible` with `blockers` (for example, an L run without a succeeded S run). Valid for 30 minutes |
| `launch_run` | `quote_id`, `idempotency_key`, `seed_offset` (S only; L runs use the protocol seed) | `run_id`, `events_cursor` and the reserved FLOPs, returned immediately |
| `cancel_run` | `run_id`, `reason` | Final status, consumed FLOPs and refunded FLOPs |

### Events and memory

| Tool | Input | Output |
|---|---|---|
| `wait_events` | `cursor`, `types`, `timeout_s` (≤ 60) | Events after `cursor` and `next_cursor`. Event types: `verification.finished`, `run.status`, `run.eval`, `run.restarted`, `run.finished`, `budget.low`, `budget.settled` |
| `annotate_attempt` | `attempt_id`, `text` (≤ 4 KB), `tags_add` | Stores a research note in the archive |

### Documents (tool versions of the resources and the prompt)

| Tool | Input | Output |
|---|---|---|
| `get_spec` | none | The model contract with the template files, as in `nextt://spec` |
| `read_doc` | `name` (spec, failures, protocol, costs, configurations, sources, experiments, plan, or `method` together with `method=<id>`), `offset`, `max_chars` | One page of the document and `next_offset`. `experiments` leaves out the lines that report results |
| `get_instructions` | `focus`, `scale` | The `research_iteration` text, the list of readable documents, and the task card |

Many MCP clients give the model tools only. These three tools make every resource and the prompt reachable
anyway.

## Resources (read-only)

| URI | Content |
|---|---|
| `nextt://task` | The task card (same as `get_task`) |
| `nextt://spec` | The model contract (`spec.md`) with the template `model.py` and `config.json` |
| `nextt://budget` | Same as `get_budget` |
| `nextt://leaderboard/{scale}` | Same as `get_leaderboard(scale)` |
| `nextt://attempts/{attempt_id}` | An attempt with its audit and runs |
| `nextt://knowledge/failures` | Failure modes from the upstream experiment record (without lines that report results), plus rejected or failed gym attempts |
| `nextt://upstream/{doc}` | `protocol`, `costs`, `configurations`, `sources`, `experiments`, `plan`; `experiments` and `plan` without lines that report results |
| `nextt://upstream/methods/{method}` | Frozen configuration, cost-proof summary and implementation source of a registered method |

Every resource also has a tool: `get_task`, `get_budget`, `get_leaderboard`, `get_attempt`, `get_spec` and
`read_doc`.

## Prompt

`research_iteration(focus="", scale="S")` gives the standard iteration:
1. Observe.
2. Read the strongest attempts and the failures.
3. State one hypothesis.
4. Write the model, estimate its cost, submit it, and calibrate if needed.
5. Run it at S and wait for it.
6. Compare with the parent and annotate.

Promote to L only when the S run is clearly better on the dev split than its parent and the strongest S results.

## Adapter for evaluation-function frameworks

Evolutionary and program-search frameworks call an evaluation function instead of tools. `nextt_gym/adapter.py`
wraps the whole loop into one blocking call:

```python
from nextt_gym.adapter import evaluate          # or Evaluator(service).evaluate(...)
r = evaluate("model.py", config="config.json", scale="S", backend="a100-smoke", calibrate=["mlp_width"])
r["combined_score"]   # -dev-split validation BPB (higher is better); -1e9 on any failure
r["metrics"], r["feedback"], r["error"], r["attempt_id"], r["run_id"]
```

or as a command: `python -m nextt_gym.adapter model.py --config config.json --scale S --backend ID`. The command
prints the JSON result and a last line `combined_score: <value>`.

- **What it does:** submit, then wait for verification, then calibrate the given tunable fields if only FLOPs are
  off, then quote, launch and wait for the run.
- **Idempotent on content:** the same `model.py` + `config.json` re-attaches to the existing attempt and run, so a
  framework's retries or timeouts never launch twice.
- **Failures are scores, not exceptions:** a rejected submission, a policy block or an exhausted budget returns
  `ok=false` with the worst score and a readable `feedback`.
- **Runtime:** an S run takes hours, so give the framework's evaluator timeout at least that long.

## The task card (`nextt://task`, `get_task`)

| Field | Meaning |
|---|---|
| `task_id`, `spec_version` | Task and contract versions (`nextt-iso-flop-arch-v1`, `nextt-gym-model-v1`) |
| `objective` | Minimize validation BPB at matched training FLOPs per input token |
| `controlled` | Fixed variables: vocabulary 32768, context 2048, 1,048,576 tokens per update, the ClimbMix revision, the Muon/AdamW recipe by parameter role, the schedule (warmup 40, last 65% warmdown to 5%), seed 42, BF16 |
| `flops_anchor` | 4,672,398,325 training FLOPs per input token (upstream SSSL Dense), tolerance 1% |
| `scales` | S: 512 updates, a prefix of the L schedule (`origin: gym`; shaped like a pilot, but a research run outside the pilot quota). C: S's 512 updates and tokens under their own compressed schedule (warmup 40, the last 65% decaying to 5%), so a run ends after a decay (`origin: gym`). M: the first 2,000 updates of the L schedule, evaluated every 256 updates (`origin: gym`). L: 46,875 updates (`origin: upstream protocol`). Also validation cadence, which splits are evaluated, CORE and theory FLOPs |
| `visible_splits` | S and L: `dev` (the arena's dev split, `shard_06541`) |
| `sealed`, `sealing` | L also evaluates the official validation split and CORE-22; for gym runs and imported upstream runs they stay sealed until the operator reveals the attempt |
| `upstream_protocol_hash` | Fingerprint of the upstream protocol the gym follows |
| `policy` | An L run requires a succeeded S run and uses the protocol seed; at most 4 concurrent runs; one run may reserve at most 25% of the cap; 30 `estimate_cost` calls per hour; submissions up to 256 KB |

The card is static unless the upstream protocol or the gym policy changes. Read it once per session.
