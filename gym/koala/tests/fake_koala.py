#!/usr/bin/env python3
"""A local stand-in for the Koala CLI calls koala_gym.py makes (tests only).

State lives under $HOME/.fake-koala. Jobs run their command once per node as local processes, with
the bundle copied to a per-node directory that replaces the mount path in the command.
Settings ($HOME/.fake-koala/settings.json): {"queue_seconds": float, "fail_submit": bool,
"quota_full_submits": int, "quota_race_submits": int} (the last two refuse the first N submissions).
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ["HOME"]) / ".fake-koala"


def settings():
    path = ROOT / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def digest(directory):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        h.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def publish(args):
    directory = Path(args[0])
    project = args[args.index("--project") + 1]
    sha = digest(directory)
    files = sum(1 for p in directory.rglob("*") if p.is_file())
    value = {
        "bundle_sha256": sha,
        "files": files,
        "bundle_uri": f"fake://{project}/{sha}",
        "dry_run": "--dry-run" in args,
    }
    if "--dry-run" not in args:
        if args[args.index("--expect-sha256") + 1] != sha:
            print("sha mismatch", file=sys.stderr)
            return 2
        target = ROOT / "bundles" / sha
        if target.exists():
            print("目标目录必须不存在", file=sys.stderr)
            return 2
        shutil.copytree(directory, target)
        value["published"] = True
    print("[koala] 集群: fake")
    print(json.dumps(value))
    return 0


def quota_refusal(kind, name):
    """The first N submissions of each kind fail like Koala's quota check (counted in $HOME/.fake-koala)."""
    budget = settings().get(kind, 0)
    counter = ROOT / f"{kind}.count"
    used = int(counter.read_text()) if counter.exists() else 0
    if used >= budget:
        return False
    counter.write_text(str(used + 1))
    if kind == "quota_full_submits":
        print(
            "❌ 常规 GPU 配额不足：本次申请 1 GPU，配额上限 48（已用 48，可用 0）",
            file=sys.stderr,
        )
    else:
        print(f"[normal] {name}")
        print(
            "⚠ 并发提交撞车：本次任务 k1-fake 已自动撤销。你的常规 GPU 配额 48 卡，并发提交后合计声明 49 卡（超 1）。"
        )
    return True


def submit(args):
    if settings().get("fail_submit"):
        print("submission refused", file=sys.stderr)
        return 2
    revoked = f"user-{args[args.index('-j') + 1]}-{uuid.uuid4().hex[:10]}-normal-{time.strftime('%Y%m%d-%H%M%S')}"
    if quota_refusal("quota_full_submits", revoked) or quota_refusal(
        "quota_race_submits", revoked
    ):
        return 1
    nodes = int(args[args.index("-n") + 1])
    gpus = int(args[args.index("-g") + 1])
    prefix = args[args.index("-j") + 1]
    uri, mount = args[args.index("--code-bundle") + 1].rsplit(":", 1)
    command = args[args.index("-c") + 1]
    name = (
        f"user-{prefix}-{uuid.uuid4().hex[:10]}-normal-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    job = ROOT / "jobs" / name
    job.mkdir(parents=True)
    sha = uri.rsplit("/", 1)[1]
    for node in range(nodes):
        code = job / f"node-{node}" / "code"
        shutil.copytree(ROOT / "bundles" / sha, code)
        (job / f"node-{node}" / "command").write_text(command.replace(mount, str(code)))
    (job / "spec.json").write_text(
        json.dumps({"nodes": nodes, "gpus": gpus, "created": time.time()})
    )
    launcher = subprocess.Popen(
        [sys.executable, __file__, "_launch", str(job)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=open(job / "launcher.err", "w"),
    )
    (job / "launcher.pid").write_text(str(launcher.pid))
    print(name)
    return 0


def launch(job):
    job = Path(job)
    time.sleep(float(settings().get("queue_seconds", 0)))
    if (job / "stopped").exists():
        return 0
    spec = json.loads((job / "spec.json").read_text())
    processes = []
    for node in range(spec["nodes"]):
        directory = job / f"node-{node}"
        env = dict(
            os.environ,
            KOALA_GYM_HOST=f"{job.name}-worker-{node}",
            KOALA_GYM_ADDRESS="127.0.0.1",
            KOALA_JOB_NAME=job.name,
        )
        log = open(directory / "log", "w")
        processes.append(
            subprocess.Popen(
                ["bash", "-c", (directory / "command").read_text()],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=directory,
                start_new_session=True,
            )
        )
        (directory / "pid").write_text(str(processes[-1].pid))
    (job / "started").write_text(str(time.time()))
    for node, process in enumerate(processes):
        (job / f"node-{node}" / "exit").write_text(str(process.wait()))
    return 0


def status(job):
    spec = json.loads((job / "spec.json").read_text())
    if (job / "stopped").exists():
        return "Stopped"
    if not (job / "started").exists():
        return "Queuing"
    exits = [job / f"node-{n}" / "exit" for n in range(spec["nodes"])]
    if not all(p.exists() for p in exits):
        return "Running"
    return "Succeeded" if all(p.read_text().strip() == "0" for p in exits) else "Failed"


def main(argv):
    if argv[0] == "_launch":
        return launch(argv[1])
    if argv[0] == "--cluster":
        argv = argv[2:]
    command, args = argv[0], argv[1:]
    if command == "code" and args[0] == "publish":
        return publish(args[1:])
    if command == "submit":
        return submit(args)
    job = ROOT / "jobs" / args[0]
    if command not in ("get", "log", "stop") or not job.exists():
        if command == "stop" and args[0] == "-y":
            job = ROOT / "jobs" / args[1]
            if not job.exists():
                # A job Koala already revoked: record the stop and succeed, as the real CLI does.
                with (ROOT / "stopped-unknown.txt").open("a") as handle:
                    handle.write(args[1] + "\n")
                return 0
        else:
            print(f"unknown job or command {argv}", file=sys.stderr)
            return 2
    if command == "get":
        print(json.dumps({"status": status(job)}))
        return 0
    if command == "log":
        worker = int(args[args.index("--worker") + 1]) if "--worker" in args else 0
        tail = int(args[args.index("-n") + 1]) if "-n" in args else 1000
        path = job / f"node-{worker}" / "log"
        if not path.exists():
            print("pod not started", file=sys.stderr)
            return 1
        print("\n".join(path.read_text().splitlines()[-tail:]))
        return 0
    (job / "stopped").write_text(str(time.time()))
    for pid in [job / "launcher.pid", *job.glob("node-*/pid")]:
        try:
            os.killpg(int(pid.read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
