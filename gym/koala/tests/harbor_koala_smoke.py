"""Harbor + Koala smoke: native Claude Code in Docker against a local model stub, numerical jobs on Koala.

It is gym/smoke.py with two changes for remote execution:
- the trial goes through `koala_gym.py --koala CONF trial`, so every numerical job runs in a Koala pod;
- each Koala job takes minutes, longer than one Claude Code Bash call may block, so the stub starts the
  unchanged research script in the background and then repeats a bounded wait until it finishes.

No external model API is called. Usage (from the repository root):
    gym/.venv/bin/python gym/koala/tests/harbor_koala_smoke.py --koala .agent-work/koala.h200-harbor.json
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
GYM = ROOT / "gym"
MAX_WAITS = 16
LAUNCH = """cat > /workspace/research.py <<'GYM_SMOKE'
{script}
GYM_SMOKE
setsid nohup sh -c 'python /workspace/research.py && touch /workspace/research.ok || touch /workspace/research.failed' \
  > /workspace/research.log 2>&1 < /dev/null &
echo launched"""
WAIT = """timeout 540 sh -c 'until [ -e /workspace/research.ok ] || [ -e /workspace/research.failed ]; do sleep 15; done'
if [ -e /workspace/research.ok ]; then echo GYM_RESEARCH_DONE; elif [ -e /workspace/research.failed ]; then echo GYM_RESEARCH_FAILED; else echo GYM_RESEARCH_RUNNING; fi
tail -n 5 /workspace/research.log"""


def text_of(block):
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def model_server(state, token):
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "listener.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = state / "listener.json"
    port = json.loads(config.read_text())["port"] if config.exists() else 0
    script = (GYM / "tasks/architecture-research/solution/research.py").read_text()
    calls = {
        "messages": 0,
        "research_commands": 0,
        "waits": 0,
        "mcp_resource_calls": 0,
        "mcp_resource_success": False,
        "research_status": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.headers.get("x-api-key") != token:
                self.send_error(403)
                return
            self.connection.settimeout(30)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024:
                self.send_error(413)
                return
            data = json.loads(self.rfile.read(size))
            if self.path.split("?")[0].endswith("/count_tokens"):
                self.reply({"input_tokens": 128})
                return
            if self.path.split("?")[0] != "/v1/messages":
                self.send_error(404)
                return
            calls["messages"] += 1
            tools = {t["name"] for t in data.get("tools", [])}
            results = {
                block["tool_use_id"]: block
                for message in data.get("messages", [])
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if isinstance(block, dict) and block.get("type") == "tool_result"
            }
            resource_result = results.get("toolu_gym_resources")
            if resource_result is not None:
                calls["mcp_resource_success"] = not resource_result.get(
                    "is_error", False
                )
            waits = sorted(k for k in results if k.startswith("toolu_gym_wait_"))
            status = None
            if waits:
                last = text_of(results[waits[-1]])
                status = (
                    "done"
                    if "GYM_RESEARCH_DONE" in last
                    else "failed"
                    if "GYM_RESEARCH_FAILED" in last
                    else "running"
                )
                calls["research_status"] = status
            resource_tool = next(
                (n for n in tools if n == "mcp__gym__get_resources"), None
            )
            block = None
            if resource_tool and resource_result is None:
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_resources",
                    "name": resource_tool,
                    "input": {},
                }
                calls["mcp_resource_calls"] += 1
            elif (
                not resource_tool
                and resource_result is None
                and "ToolSearch" in tools
                and "toolu_gym_search" not in results
            ):
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_search",
                    "name": "ToolSearch",
                    "input": {"query": "+gym get_resources", "max_results": 1},
                }
            elif (
                "Bash" in tools
                and calls["mcp_resource_success"]
                and "toolu_gym_launch" not in results
            ):
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_launch",
                    "name": "Bash",
                    "input": {
                        "command": LAUNCH.format(script=script),
                        "description": "Start the bounded Gym integration workflow",
                        "timeout": 120000,
                    },
                }
                calls["research_commands"] += 1
            elif (
                "toolu_gym_launch" in results
                and status in (None, "running")
                and len(waits) < MAX_WAITS
            ):
                block = {
                    "type": "tool_use",
                    "id": f"toolu_gym_wait_{len(waits):02d}",
                    "name": "Bash",
                    "input": {
                        "command": WAIT,
                        "description": "Wait for the Gym workflow",
                        "timeout": 590000,
                    },
                }
                calls["waits"] += 1
            if block is None:
                block, stop = (
                    {"type": "text", "text": f"Bounded Gym smoke finished: {status}."},
                    "end_turn",
                )
            else:
                stop = "tool_use"
            message = {
                "id": "msg_gym_smoke",
                "type": "message",
                "role": "assistant",
                "model": data.get("model", "smoke"),
                "content": [block],
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 128, "output_tokens": 128},
            }
            if not data.get("stream"):
                self.reply(message)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.event(
                "message_start",
                {"message": {**message, "content": [], "stop_reason": None}},
            )
            if block["type"] == "tool_use":
                start = {**block, "input": {}}
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                }
            else:
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            self.event("content_block_start", {"index": 0, "content_block": start})
            self.event("content_block_delta", {"index": 0, "delta": delta})
            self.event("content_block_stop", {"index": 0})
            self.event(
                "message_delta",
                {
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": 128},
                },
            )
            self.event("message_stop", {})

        def reply(self, value):
            payload = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def event(self, kind, value):
            self.wfile.write(
                (
                    "event: "
                    + kind
                    + "\ndata: "
                    + json.dumps({"type": kind, **value})
                    + "\n\n"
                ).encode()
            )
            self.wfile.flush()

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        server.daemon_threads = True
        temporary = config.with_suffix(".tmp")
        temporary.write_text(json.dumps({"port": server.server_port}))
        temporary.replace(config)
    except BaseException:
        lock.close()
        raise
    return server, lock, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koala", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / ".agent-work" / "koala-gym" / "harbor-smoke",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=HERE / "examples" / "profiles.fixture.koala.json",
    )
    parser.add_argument("--client-image", default="nt-gym-client:local")
    parser.add_argument("--verifier-image", default="nt-gym-verifier:local-js")
    parser.add_argument("--timeout-sec", default="7200")
    args = parser.parse_args()
    token = secrets.token_urlsafe(32)
    server, lock, calls = model_server(args.state, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    jobs = args.state / "jobs"
    before = set(jobs.glob("*.binding.json"))
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_", "AWS_"))
    }
    environment.update(
        ANTHROPIC_API_KEY=token,
        ANTHROPIC_BASE_URL=f"http://host.docker.internal:{server.server_port}",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(HERE / "koala_gym.py"),
                "--koala",
                str(args.koala.resolve()),
                "trial",
                "--agent",
                "claude-code",
                "--model",
                "claude-sonnet-4-5",
                "--max-turns",
                str(MAX_WAITS + 8),
                "--max-budget-usd",
                "1",
                "--timeout-sec",
                args.timeout_sec,
                "--research-flops-limit",
                "1000000000000",
                "--profiles",
                str(args.profiles.resolve()),
                "--jobs-dir",
                str(jobs),
                "--client-image",
                args.client_image,
                "--verifier-image",
                args.verifier_image,
            ],
            env=environment,
            cwd=ROOT,
        )
        bindings = set(jobs.glob("*.binding.json")) - before
        summary = {
            "run_trial_exit": process.returncode,
            "agent": "claude-code",
            "model_backend": "local_stub",
            "external_model_calls": 0,
            **calls,
        }
        if len(bindings) == 1:
            binding = json.loads(bindings.pop().read_text())
            summary["trial_id"] = binding["GYM_TRIAL_ID"]
            files = list((jobs / binding["GYM_TRIAL_ID"]).glob("*/result.json"))
            if len(files) == 1:
                result = json.loads(files[0].read_text())
                summary["exception"] = (result.get("exception_info") or {}).get(
                    "exception_type"
                )
                summary["reward"] = (
                    (result.get("verifier_result") or {}).get("rewards") or {}
                ).get("reward")
        (args.state / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
        ok = (
            process.returncode == 0
            and summary.get("reward") == 1
            and not summary.get("exception")
            and calls["research_status"] == "done"
            and calls["mcp_resource_success"]
        )
        return 0 if ok else 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
