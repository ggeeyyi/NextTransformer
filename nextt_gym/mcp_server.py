"""MCP adapter: the 19 agent tools, read-only resources and the iteration prompt.

Every tool is a thin call into GymService; errors come back as structured GymError dicts.
    python -m nextt_gym.mcp_server                      # stdio (local agent, e.g. Claude Code)
    python -m nextt_gym.mcp_server --http --port 8780   # Streamable HTTP
"""
import argparse
import functools
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .errors import GymError
from .service import GymService, REPO_ROOT, without_results

INSTRUCTIONS = """NextT-Gym: design language-model architectures that reach lower validation BPB at the same
training FLOPs per input token as the upstream Dense anchor (4.672 GFLOPs/token, +-1%).
Read nextt://task and nextt://spec first. Expensive actions return handles: poll get_run / get_verification
or call wait_events. Every experiment draws on a shared FLOPs budget (get_budget). Select with the dev split and
training loss; official BPB and CORE of gym runs and of imported upstream runs stay sealed until the operator
reveals them."""

server = MCPServer("nextt-gym", instructions=INSTRUCTIONS)
_service = None


def svc():
    global _service
    if _service is None: _service = GymService()
    return _service


def tool(fn):
    """Register `fn` as a tool; GymError becomes a structured error result instead of an exception."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GymError as exc:
            return dict(error=exc.to_dict())
    return server.tool()(wrapper)


# ---- observation -----------------------------------------------------------------
@tool
def get_task() -> dict:
    """The research task: objective, controlled variables, FLOPs anchor and tolerance, scales, visibility."""
    return svc().get_task()


@tool
def get_budget() -> dict:
    """Total FLOPs cap, used, reserved and available; breakdown by kind and scale; GPU-hours per GPU type."""
    return svc().get_budget()


@tool
def get_leaderboard(scale: str, split: str = "dev", limit: int = 20, include_imported: bool = True) -> dict:
    """Best succeeded run per architecture at one scale ("S" research prefix, "C" compressed 512-update
    schedule, "M" 2,000-update prefix, "L" formal schedule), ranked by validation BPB (lower is better).

    Imported upstream results exist only on the "official" split (at L). Official results are sealed: runs appear
    on that split only after the operator reveals their attempt. See entries_per_split."""
    return svc().get_leaderboard(scale, split, limit, include_imported)


@tool
def list_attempts(status: list[str] | None = None, origin: str | None = None, tags: list[str] | None = None,
                  ancestor_of: str | None = None, descendant_of: str | None = None, sort: str = "created",
                  limit: int = 50, cursor: str | None = None) -> dict:
    """List architecture attempts with filters; sort by "created" or "best_S" / "best_L"."""
    return svc().list_attempts(status=status, origin=origin, tags=tags, ancestor_of=ancestor_of,
                               descendant_of=descendant_of, sort=sort, limit=limit, cursor=cursor)


@tool
def get_attempt(attempt_id: str, include: list[str] | None = None) -> dict:
    """One attempt; include any of "files", "audit", "runs", "lineage"."""
    return svc().get_attempt(attempt_id, include or [])


@tool
def get_run(run_id: str) -> dict:
    """Status, progress, consumed FLOPs, latest and final metrics of a training run."""
    return svc().get_run(run_id)


@tool
def get_run_curve(run_id: str, metric: str = "val_bpb", split: str = "dev", max_points: int = 200) -> dict:
    """Curve points [{step, input_tokens, value}] for "train_loss" or "val_bpb" on the "dev" or "official" split."""
    return svc().get_run_curve(run_id, metric, split, max_points)


@tool
def get_run_logs(run_id: str, tail: int = 200, grep: str | None = None) -> dict:
    """Last lines of a run's log (paths, hosts and credentials redacted)."""
    return svc().get_run_logs(run_id, tail, grep)


@tool
def compare_attempts(attempt_ids: list[str], scale: str, metric: str = "val_bpb") -> dict:
    """Curves of 2-8 attempts at one scale, aligned by input tokens, with final values and deltas."""
    return svc().compare_attempts(attempt_ids, scale, metric)


# ---- proposal and verification ---------------------------------------------------------
@tool
def estimate_cost(files: dict[str, str]) -> dict:
    """Dry-run V0-V3 (contract, causality, FLOPs audit) on files {"model.py": ..., "config.json": ...}; no attempt is created.

    If only V3 fails with FLOPS_OUT_OF_TOLERANCE, submit_attempt and then calibrate on a tunable field."""
    return svc().estimate_cost(files)


@tool
def submit_attempt(files: dict[str, str], title: str, hypothesis: str, idempotency_key: str,
                   parent_ids: list[str] | None = None, tags: list[str] | None = None) -> dict:
    """Submit an immutable architecture; verification V0-V3 starts automatically (poll get_verification)."""
    return svc().submit_attempt(files, title=title, hypothesis=hypothesis, parent_ids=parent_ids or [],
                                tags=tags or [], idempotency_key=idempotency_key)


@tool
def get_verification(verification_id: str) -> dict:
    """Per-level verification status and errors; the FLOPs audit report when it passed V3."""
    return svc().get_verification(verification_id)


@tool
def calibrate(attempt_id: str, knobs: list[str], target_relative_error: float = 0.0) -> dict:
    """Adjust declared tunable integer fields until FLOPs/token is within tolerance; creates a new verified attempt."""
    return svc().calibrate(attempt_id, knobs, target_relative_error)


# ---- resources and execution -------------------------------------------------------------
@tool
def list_backends(gpu_type: str | None = None) -> dict:
    """Available clusters: GPU type, allowed world sizes, status, queue estimate."""
    return svc().list_backends(gpu_type)


@tool
def quote_run(attempt_id: str, scale: str, backend_id: str, world_size: int | None = None) -> dict:
    """Price a run: FLOPs to reserve, estimated GPU-hours and wall time, budget after reservation, blockers."""
    return svc().quote_run(attempt_id, scale, backend_id, world_size)


@tool
def launch_run(quote_id: str, idempotency_key: str, seed_offset: int = 0) -> dict:
    """Reserve the quoted FLOPs and start the run; returns run_id immediately. seed_offset is for S runs only."""
    return svc().launch_run(quote_id, idempotency_key, seed_offset)


@tool
def cancel_run(run_id: str, reason: str) -> dict:
    """Stop a run; consumed FLOPs are charged up to the last completed step, the rest is refunded."""
    return svc().cancel_run(run_id, reason)


# ---- events and memory -------------------------------------------------------------------
@tool
def wait_events(cursor: int | None = None, types: list[str] | None = None, timeout_s: int = 30) -> dict:
    """Long-poll for events after `cursor` (verification.finished, run.status, run.eval, run.restarted,
    run.finished, budget.*).

    Pass the `events_cursor` returned by submit_attempt / launch_run; without a cursor only events after this call
    are returned. Use `next_cursor` from the result for the following call."""
    return svc().wait_events(cursor, types, timeout_s)


@tool
def annotate_attempt(attempt_id: str, text: str, tags_add: list[str] | None = None) -> dict:
    """Attach a research note to an attempt (kept in the archive for later iterations)."""
    return svc().annotate_attempt(attempt_id, text, tags_add)


# ---- resources -----------------------------------------------------------------------------
def _json(value): return json.dumps(value, indent=1, default=str)


@server.resource("nextt://task", mime_type="application/json")
def r_task() -> str: return _json(svc().get_task())


def spec_text() -> str:
    template = (REPO_ROOT / "nextt_gym/templates")
    return ((REPO_ROOT / "nextt_gym/spec.md").read_text() + "\n\n## Template model.py\n\n```python\n"
            + (template / "model.py").read_text() + "```\n\n## Template config.json\n\n```json\n"
            + (template / "config.json").read_text() + "```\n")


@server.resource("nextt://spec", mime_type="text/markdown")
def r_spec() -> str: return spec_text()


@server.resource("nextt://budget", mime_type="application/json")
def r_budget() -> str: return _json(svc().get_budget())


@server.resource("nextt://leaderboard/{scale}", mime_type="application/json")
def r_leaderboard(scale: str) -> str: return _json(svc().get_leaderboard(scale))


@server.resource("nextt://attempts/{attempt_id}", mime_type="application/json")
def r_attempt(attempt_id: str) -> str: return _json(svc().get_attempt(attempt_id, ["audit", "runs"]))


@server.resource("nextt://knowledge/failures", mime_type="text/markdown")
def r_failures() -> str: return svc().knowledge_failures()


UPSTREAM_DOCS = {"protocol": "PROTOCOL.md", "costs": "docs/protocol/COSTS.md",
                 "configurations": "docs/methods/CONFIGURATIONS.md", "sources": "docs/methods/SOURCES.md",
                 "experiments": "docs/records/EXPERIMENTS.md", "plan": "PLAN.md"}
# The experiment record and the plan report results of imported runs; their result lines are left out.
RECORD_DOCS = ("experiments", "plan")


def upstream_doc(doc: str) -> str:
    if doc not in UPSTREAM_DOCS:
        raise GymError("NOT_FOUND", f"unknown document {doc!r}", details=dict(available=sorted(UPSTREAM_DOCS)))
    text = (REPO_ROOT / UPSTREAM_DOCS[doc]).read_text()
    if doc not in RECORD_DOCS: return text
    return "(Lines that report results are left out: official results are sealed.)\n\n" + without_results(text)


@server.resource("nextt://upstream/{doc}", mime_type="text/markdown")
def r_upstream(doc: str) -> str: return upstream_doc(doc)


@server.resource("nextt://upstream/methods/{method}", mime_type="text/markdown")
def r_method(method: str) -> str: return svc().upstream_method(method)


# ---- prompt ------------------------------------------------------------------------------------
@server.prompt()
def research_iteration(focus: str = "", scale: str = "S") -> str:
    """One research iteration, following the gym's loop."""
    return instructions_text(focus, scale)


def instructions_text(focus: str = "", scale: str = "S") -> str:
    return f"""Run one research iteration{f' focused on {focus}' if focus else ''}.
1. get_budget; get_leaderboard("{scale}"); list_attempts(sort="best_{scale}").
2. Read the strongest attempts and their lineage (get_attempt with files, lineage); read nextt://knowledge/failures and
   the relevant nextt://upstream/* documents.
3. State one falsifiable hypothesis. Write model.py + config.json (contract: nextt://spec), starting from a parent.
4. estimate_cost until V0-V3 pass (use calibrate for FLOPs); then submit_attempt with parent_ids.
5. quote_run at scale S; launch_run; wait_events(events_cursor, ["run.finished"]). Promote to L only when the S run
   is clearly better on the dev split than its parent and the strongest S results.
6. Compare with the parent (compare_attempts); annotate_attempt with what you learned, including failures."""


# ---- tool versions of the resources and the prompt (many MCP clients show the model tools only) -------------
DOCS = ("spec", "failures", *UPSTREAM_DOCS)


@tool
def get_spec() -> dict:
    """The model contract (what model.py and config.json must provide) with the template files. Same as nextt://spec."""
    return dict(spec=spec_text())


@tool
def read_doc(name: str, method: str | None = None, max_chars: int = 60000, offset: int = 0) -> dict:
    """Read a reference document: "spec", "failures", "protocol", "costs", "configurations", "sources",
    "experiments", "plan", or name="method" with method=<registered method> for its frozen config, cost proof and
    source. Long documents are paged with offset / max_chars."""
    if name == "method":
        if not method: raise GymError("INVALID_ARGUMENT", "name='method' needs method=<id>")
        text = svc().upstream_method(method)
    elif name == "spec": text = spec_text()
    elif name == "failures": text = svc().knowledge_failures()
    else: text = upstream_doc(name)
    max_chars = max(1000, min(int(max_chars), 100000)); offset = max(0, int(offset))
    chunk = text[offset:offset + max_chars]
    return dict(name=name, method=method, offset=offset, chars=len(chunk), total_chars=len(text),
                next_offset=offset + len(chunk) if offset + len(chunk) < len(text) else None, text=chunk)


@tool
def get_instructions(focus: str = "", scale: str = "S") -> dict:
    """The standard research iteration (same as the research_iteration prompt) and the list of readable documents."""
    return dict(instructions=instructions_text(focus, scale), documents=list(DOCS) + ["method"],
                task=svc().get_task())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true"); parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.http:
        # Stateless: every request stands alone (state lives in the gym store and in returned handles).
        server.run("streamable-http", host=args.host, port=args.port, stateless_http=True)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
