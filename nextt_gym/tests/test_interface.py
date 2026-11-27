"""The served interface matches INTERFACE.md: tools and their inputs, resources, the prompt, event types, the task
card's policy, errors, sorting at every scale, the documents, and the operations no other test drives."""
import asyncio
import json
from pathlib import Path
import re

import pytest

from nextt_gym import mcp_server
from nextt_gym.errors import GymError
from nextt_gym.service import GymService
from nextt_gym.store import EVENT_TYPES
from nextt_gym.task import SCALES, TASK

ROOT = Path(__file__).resolve().parents[2]
INTERFACE = (ROOT / "nextt_gym/INTERFACE.md").read_text()
TEMPLATES = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}


def section(title):
    """The text of one `## title` section of INTERFACE.md."""
    return re.search(rf"^## {re.escape(title)}.*?$(.*?)(?=^## |\Z)", INTERFACE, re.S | re.M).group(1)


def rows(text):
    """`(first cell, other cells)` of every table row whose first cell is a code span."""
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| `"): out.append((cells[0].strip("`"), cells[1:]))
    return out


@pytest.fixture
def gym(tmp_path, monkeypatch):
    (tmp_path / "backends.toml").write_text(
        '[cpu]\nkind = "local"\ngpu_type = "CPU"\nallowed_world_sizes = [1]\n'
        '[koala-h20]\nkind = "koala"\ngpu_type = "H20"\nworkdir = "/w"\nmax_segments = 3\n')
    service = GymService(home=tmp_path / "home", backends_path=tmp_path / "backends.toml")
    monkeypatch.setattr(mcp_server, "_service", service)
    yield service
    service.store.close()


def call(tool, /, **arguments):
    """A tool call through the MCP server, as an agent makes it."""
    result = asyncio.run(mcp_server.server.call_tool(tool, arguments))
    return json.loads(result.content[0].text)


def test_tools_and_their_inputs_match_the_interface():
    documented = dict((name, cells[0]) for name, cells in rows(section("Tools")))
    served = {t.name: t for t in asyncio.run(mcp_server.server.list_tools())}
    assert set(served) == set(documented)
    assert f"## Tools ({len(served)})" in INTERFACE and f"the {len(served)} agent tools" in mcp_server.__doc__
    for name, tool in served.items():
        parameters = set(tool.input_schema.get("properties", {}))
        missing = [p for p in parameters if not re.search(rf"\b{p}\b", documented[name])]
        assert not missing, f"{name}: inputs {missing} are not in INTERFACE.md"
        assert parameters or documented[name] == "none", name


def test_resources_and_the_prompt_match_the_interface():
    documented = {uri for uri, _ in rows(section("Resources (read-only)"))}
    served = {str(r.uri) for r in asyncio.run(mcp_server.server.list_resources())}
    served |= {t.uri_template for t in asyncio.run(mcp_server.server.list_resource_templates())}
    assert served == documented
    prompts = asyncio.run(mcp_server.server.list_prompts())
    assert [(p.name, [a.name for a in p.arguments]) for p in prompts] == [("research_iteration", ["focus", "scale"])]
    assert 'research_iteration(focus="", scale="S")' in section("Prompt")


def test_event_types_match_the_interface(gym):
    row = dict(rows(section("Tools")))["wait_events"][1]
    assert set(re.findall(r"`([a-z]+\.[a-z_]+)`", row)) == set(EVENT_TYPES)
    emitted = set()
    for path in (ROOT / "nextt_gym").glob("*.py"):
        emitted |= set(re.findall(r"append_event\(\s*\"([a-z]+\.[a-z_]+)\"", path.read_text()))
    assert emitted <= set(EVENT_TYPES), emitted - set(EVENT_TYPES)
    with pytest.raises(ValueError): gym.store.append_event("run.unknown")
    error = call("wait_events", cursor=0, types=["run.finished", "run.unknown"], timeout_s=0)["error"]
    assert error["code"] == "INVALID_ARGUMENT" and error["details"]["available"] == list(EVENT_TYPES)
    assert call("wait_events", cursor=0, types=list(EVENT_TYPES), timeout_s=0)["events"] == []


def test_task_card_matches_the_interface(gym):
    card, policy = call("get_task"), TASK.policy
    assert card == json.loads(json.dumps(gym.get_task()))
    fields = {name for name, _ in rows(section("The task card (`nextt://task`, `get_task`)"))}
    assert {f.strip() for name in fields for f in name.replace("`", "").split(",")} == set(card)
    text = dict(rows(section("The task card (`nextt://task`, `get_task`)")))["policy"][0]
    assert f"at most {policy.max_concurrent_runs} concurrent runs" in text
    assert f"{policy.estimate_cost_per_hour} `estimate_cost` calls per hour" in text
    assert f"at most {policy.max_run_share_of_cap:.0%} of the cap" in text
    assert f"submissions up to {policy.submission_max_bytes // 1024} KB" in text
    splits = dict(rows(section("The task card (`nextt://task`, `get_task`)")))["visible_splits"][0]
    assert set(card["visible_splits"]) == set(SCALES) and all(s in splits for s in SCALES)


def test_errors_are_structured(gym):
    error = call("get_run", run_id="run_999999")["error"]
    assert error == dict(code="NOT_FOUND", message="run run_999999 does not exist", retryable=False)
    assert call("read_doc", name="nope")["error"]["details"]["available"] == sorted(mcp_server.UPSTREAM_DOCS)


def test_every_scale_sorts_attempts_and_the_prompt_uses_valid_sorts(gym):
    attempts = []
    for title, value in (("worse", 0.95), ("better", 0.90)):
        attempt = gym.archive.put_attempt({**TEMPLATES, "config.json": TEMPLATES["config.json"] + " " * len(title)},
                                          title=title, hypothesis="h")
        run = gym.archive.create_run(attempt, "M")
        gym.archive.update_run(run, status="succeeded", final=dict(val_bpb_dev=value))
        attempts.append(attempt)
    for scale in SCALES:
        sort = f"best_{scale}"
        assert f'sort="{sort}"' in mcp_server.instructions_text(scale=scale)
        assert f"`{sort}`" in dict(rows(section("Tools")))["list_attempts"][0]
        items = call("list_attempts", sort=sort)["items"]
        if scale == "M": assert [i["attempt_id"] for i in items] == attempts[::-1]
    error = call("list_attempts", sort="best_X")["error"]
    assert error["code"] == "INVALID_ARGUMENT" and error["details"]["available"] == ["created"] + [f"best_{s}" for s in SCALES]


def test_documents_and_method_sources_are_readable(gym):
    instructions = call("get_instructions", scale="M")
    assert instructions["documents"] == list(mcp_server.DOCS) + ["method"] and 'get_leaderboard("M")' in instructions["instructions"]
    assert call("get_spec")["spec"] == mcp_server.spec_text()
    for name in mcp_server.DOCS:
        page = call("read_doc", name=name, max_chars=1000)
        assert page["total_chars"] > 0 and page["chars"] <= 1000, name
        if page["next_offset"]: assert call("read_doc", name=name, offset=page["next_offset"])["offset"] == page["next_offset"]
    method = call("read_doc", name="method", method="dense_full")["text"]
    assert "## Frozen configuration" in method and "## Cost proof summary" in method
    contents = asyncio.run(mcp_server.server.read_resource("nextt://upstream/methods/dense_full"))
    assert contents[0].content.startswith("# Upstream method `dense_full`")
    assert call("read_doc", name="method")["error"]["code"] == "INVALID_ARGUMENT"


def test_list_backends_reports_each_backend(gym):
    backends = {b["backend_id"]: b for b in call("list_backends")["backends"]}
    assert set(backends) == {"cpu", "koala-h20"}
    koala = backends["koala-h20"]
    assert koala["kind"] == "koala" and koala["status"] in ("available", "offline")
    assert koala["max_wall_hours_per_segment"] == 45.5 and koala["max_segments"] == 3
    assert "max_segments" not in backends["cpu"]
    assert [b["backend_id"] for b in call("list_backends", gpu_type="H20")["backends"]] == ["koala-h20"]


def test_estimate_cost_runs_the_verifier_without_an_attempt(gym):
    estimate = call("estimate_cost", files=TEMPLATES)
    assert set(estimate) == {"ok", "audit", "levels", "errors"} and [l["level"] for l in estimate["levels"]][:1] == ["V0"]
    assert not estimate["ok"] and any("tolerance" in str(e).lower() or "flops" in str(e).lower() for e in estimate["errors"])
    assert gym.list_attempts()["items"] == []


def test_cancel_run_stops_the_backend_and_refunds(gym):
    class Stub:
        cancelled = []
        def cancel(self, handle): self.cancelled.append(handle)
        def status(self, handle): return dict(state="cancelled", step=10, consumed_flops=str(10 ** 15))
        def fetch(self, handle, dest): Path(dest).mkdir(parents=True, exist_ok=True); return dest
        def logs(self, handle, tail=200): return []
    gym.backends["stub"] = Stub()
    attempt = gym.archive.put_attempt(TEMPLATES, title="mine", hypothesis="h")
    run_id = gym.archive.create_run(attempt, "S", backend_id="stub")
    gym.ledger.reserve(run_id, 10 ** 18, scale="S")
    gym.archive.update_run(run_id, status="running", handle="h-1")
    result = call("cancel_run", run_id=run_id, reason="stop")
    assert Stub.cancelled == ["h-1"] and result["status"] == "cancelled"
    assert result["consumed"]["flops_exact"] == str(10 ** 15) and result["refunded_flops"] == pytest.approx(10 ** 18 - 10 ** 15)
    assert call("cancel_run", run_id=run_id, reason="again")["error"]["code"] == "ALREADY_TERMINAL"
