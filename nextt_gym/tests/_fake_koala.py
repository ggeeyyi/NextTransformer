"""A stand-in for the `koala` CLI (Koala 3.0.3) that the Koala backend tests put on PATH.

It accepts exactly the command lines of the real CLI (GRAMMAR, from `koala <command> --help`, plus the platform
rules the backend relies on) and answers in the real CLI's formats: the cluster banner on stderr, `-q` job names
and `--json` documents on stdout, exit code 1 for an unknown job, and the documented `wait` exit codes. A
command line the real CLI would reject exits 2 and is recorded in `violations.jsonl`, which the test fixture
checks. `test_fake_grammar_matches_the_installed_cli` compares GRAMMAR with the installed CLI when there is one.

Each submitted segment runs instantly: it writes the trainer's status files into the run directory named by
`--output` in the job's command. Environment:
  FAKE_KOALA_STATE   directory for calls.jsonl, violations.jsonl and the job records
  FAKE_PHASES        comma-separated phase of each submitted segment: a trainer phase (segment_complete,
                     training_complete), `crash` (fails after update 300 with a complete checkpoint at 250),
                     `crash-early` (fails before any checkpoint) or `hang` (stays active until stopped)
  FAKE_JOB_STATE     the state of a hanging job (default Running; Queuing and Updating are also active)
  FAKE_CRASH_STATE   the state a crashed job ends in (default Failed)
  FAKE_STOP_STATE    the state after `koala stop` (default Deleting, as observed on H20)
Job times (`get --json` conditions): a job is created when it is submitted; one that runs (every phase but `hang`
under a queued FAKE_JOB_STATE) starts running 1 s later, and a finished one ends 3,600 s after that. A stopped job keeps
no conditions at all, as on H20: its record cannot tell whether it ran.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

# Options of each command: flags take no value, options take one; positionals are (min, max), max None = any.
GRAMMAR = {
    "submit": dict(flags=["-q", "--quiet", "-y", "--yes", "--dry-run", "--no-image-check", "--large-ssd", "--rebuild",
                          "--wait", "--no-follow"],
                   options=["-m", "--mode", "-j", "--job-name", "-n", "--nodes", "-g", "--gpus", "--cpu", "--mem",
                            "-c", "--cmd", "--image", "--code", "--code-bundle", "--mount", "--ssd", "--port", "--lease",
                            "--access", "--timeout"],
                   positionals=(0, 0)),
    "get": dict(flags=["--context", "--json", "--include-command", "--open"], options=[], positionals=(0, 1)),
    "wait": dict(flags=["--json"], options=["--timeout"], positionals=(1, None)),
    "log": dict(flags=["--all", "-f", "--follow"], options=["-n", "--tail", "--container", "--worker"],
                positionals=(1, 1)),
    "stop": dict(flags=["-y", "--yes"], options=["--timeout"], positionals=(1, None)),
    "exec": dict(flags=["--detach", "--gpu", "--cpu", "--stdin", "--json", "-q", "--quiet"],
                 options=["--attach", "--exec-id", "--script", "--cwd", "--timeout"], positionals=(0, 1)),
    "code publish": dict(flags=["--json", "--dry-run", "-y", "--yes"],
                         options=["--project", "--exclude", "--prefix", "--expect-sha256"], positionals=(0, 1)),
}
CLUSTERS = ("h20", "h200", "aws")
MODES = ("debug", "normal", "demo")
ACTIVE = ("Queuing", "Updating", "Running")


class Rejected(Exception):
    pass


def parse(argv):
    """`(cluster, command, options, positionals, rest)` of a command line, or Rejected as the real CLI would."""
    cluster = None
    if argv[:1] == ["--cluster"]:
        if len(argv) < 2 or argv[1] not in CLUSTERS: raise Rejected(f"--cluster must be one of {CLUSTERS}")
        cluster, argv = argv[1], argv[2:]
    if not argv: raise Rejected("a command is required")
    command, argv = (" ".join(argv[:2]), argv[2:]) if argv[:2] == ["code", "publish"] else (argv[0], argv[1:])
    if command not in GRAMMAR: raise Rejected(f"unknown command {command!r}")
    if "--cluster" in argv: raise Rejected("--cluster goes before the command")
    grammar, options, positionals, rest = GRAMMAR[command], {}, [], []
    words = list(argv)
    while words:
        word = words.pop(0)
        if word == "--":
            rest = words; break
        if word in grammar["flags"]: options[word] = True
        elif word in grammar["options"]:
            if not words: raise Rejected(f"{word} needs a value")
            options[word] = words.pop(0)
        elif word.startswith("-"): raise Rejected(f"unrecognized argument {word} for koala {command}")
        else: positionals.append(word)
    low, high = grammar["positionals"]
    if len(positionals) < low or (high is not None and len(positionals) > high):
        raise Rejected(f"koala {command} takes {low}..{high} positional arguments, got {positionals}")
    if command == "submit": check_submit(options)
    if command == "exec" and not rest and not ({"--attach", "--script"} & set(options)):
        raise Rejected("koala exec needs `-- COMMAND`")
    return cluster, command, options, positionals, rest


def check_submit(options):
    value = lambda *names: next((options[n] for n in names if n in options), None)
    mode, command = value("-m", "--mode") or "debug", value("-c", "--cmd")
    if mode not in MODES: raise Rejected(f"--mode must be one of {MODES}")
    if mode in ("normal", "demo") and not command: raise Rejected(f"{mode} jobs need -c")
    if mode == "debug" and command: raise Rejected("debug jobs take no -c")
    if int(value("-g", "--gpus") or 0) > 8: raise Rejected("-g is GPUs per node, at most 8; use -n for more nodes")
    if "--code" in options and "--code-bundle" in options: raise Rejected("--code and --code-bundle exclude each other")
    bundle = value("--code-bundle")
    if bundle and (":" not in bundle.split("://", 1)[-1]): raise Rejected("--code-bundle is URI:container directory")
    if int(value("--ssd") or 0) > 128: raise Rejected("--ssd is at most 128 GiB")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    state = Path(os.environ["FAKE_KOALA_STATE"]); state.mkdir(parents=True, exist_ok=True)
    jobs = state / "jobs"; jobs.mkdir(exist_ok=True)
    if argv in (["--help"], ["--version"]):
        print("usage: koala [--cluster {h20,h200,aws}] <command> ..." if argv == ["--help"] else "koala 3.0.3")
        return 0
    try:
        cluster, command, options, positionals, rest = parse(argv)
    except Rejected as exc:
        with open(state / "violations.jsonl", "a") as log: log.write(json.dumps(dict(argv=argv, error=str(exc))) + "\n")
        print(f"koala: error: {exc}", file=sys.stderr)
        return 2
    with open(state / "calls.jsonl", "a") as log:
        log.write(json.dumps(argv[2:] if argv[:1] == ["--cluster"] else argv) + "\n")
    print(f"[koala] 集群: fake-{cluster or 'h20'}", file=sys.stderr)
    record = lambda job: jobs / f"{job}.json"
    read = lambda job: json.loads(record(job).read_text()) if record(job).exists() else None
    write = lambda job, data: record(job).write_text(json.dumps(dict(dict(created=int(time.time())), **data)))
    missing = lambda job: print(f"未找到任务: {job}；请核对集群，或确认任务是否已回收。", file=sys.stderr)

    if command == "code publish":
        directory = Path(positionals[0] if positionals else ".")
        names = sorted(p.name for p in directory.iterdir())
        digest = hashlib.sha256(b"".join(n.encode() + (directory / n).read_bytes() for n in names)).hexdigest()
        if not options.get("--dry-run"):
            if not (options.get("-y") or options.get("--yes")): print("需要确认（-y）", file=sys.stderr); return 1
            if options.get("--expect-sha256") not in (None, digest): print("代码包与预览不一致", file=sys.stderr); return 1
        project = options.get("--project") or directory.name
        print(json.dumps(dict(schema="koala.code.publish/v1", dry_run=bool(options.get("--dry-run")), files=len(names),
                              file_list=names, bundle_sha256=digest, project=project,
                              bundle_uri=f"s3://fake-code/tester/code/{project}/blobs/{digest}.tar.gz",
                              published=not options.get("--dry-run"))))
        return 0
    if command == "submit":
        n = len(list(jobs.iterdir())) + 1
        mode = options.get("-m") or options.get("--mode") or "debug"
        job = f"tester-{options.get('-j') or options.get('--job-name') or 'job'}-{n:010x}-{mode}-20261102-{n:06d}"
        words = shlex.split((options.get("-c") or options.get("--cmd")).replace("&&", " ").replace(";", " "))
        out = Path(words[words.index("--output") + 1])
        phases = os.environ.get("FAKE_PHASES", "training_complete").split(",")
        phase = phases[min(n, len(phases)) - 1]
        run = out / "run"; run.mkdir(parents=True, exist_ok=True)
        if phase.startswith("crash"):
            (run / "run_status.json").unlink(missing_ok=True)
            (run / "budget.json").write_text(json.dumps(dict(step=300, consumed_flops="300", gpu_hours=0.3)))
            if phase == "crash": (run / "latest.json").write_text(json.dumps(dict(step=250)))
            write(job, dict(state=os.environ.get("FAKE_CRASH_STATE", "Failed"), crashed=True, ran=True))
        elif phase == "hang":
            state = os.environ.get("FAKE_JOB_STATE", "Running")
            write(job, dict(state=state, ran=state not in ("Queuing", "Updating")))
        else:
            final = {"val_bpb_dev": 0.9} if phase != "segment_complete" else None
            (run / "run_status.json").write_text(json.dumps(dict(phase=phase, step=256 * n, consumed_flops=str(1000 * n),
                                                                 final=final)))
            (run / "budget.json").write_text(json.dumps(dict(step=256 * n, consumed_flops=str(1000 * n),
                                                             gpu_hours=0.5 * n)))
            (run / "metrics.jsonl").write_text('{"event": "train", "step": 1, "loss": 10.0}\n')
            (run / f"val_{256 * n:08d}_dev.json").write_text("{}")
            (out / "remote_status.json").write_text(json.dumps(dict(phase="exited", returncode=0)))
            write(job, dict(state="Succeeded", ran=True))
        if not (options.get("-q") or options.get("--quiet")): print(f"已提交 {job}")
        print(job)
        return 0
    if command == "get":
        if not positionals: print(json.dumps(dict(cluster=cluster or "h20"))); return 0
        job = read(positionals[0])
        if job is None: missing(positionals[0]); return 1
        stamp = lambda seconds: datetime.fromtimestamp(job.get("created", 0) + seconds, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        conditions = [] if job.get("wiped") else [dict(type="Created", status="True", reason="MPIJobCreated",
                                                        lastTransitionTime=stamp(0))]
        if job.get("ran") and conditions:
            conditions.append(dict(type="Running", status="True", reason="MPIJobRunning", lastTransitionTime=stamp(1)))
        if job["state"] in ("Succeeded", "Failed") and conditions:
            conditions.append(dict(type=job["state"], status="True", reason=f"MPIJob{job['state']}",
                                   lastTransitionTime=stamp(3601)))
        print(json.dumps(dict(name=positionals[0], resourceName="k1-fake", status=job["state"], reason="fake",
                              createdAt=stamp(0), resources=dict(gpu=8), conditions=conditions)))
        return 0
    if command == "wait":
        time.sleep(0.2)
        states = [(read(name) or {}).get("state") for name in positionals]
        codes = [0 if s == "Succeeded" else 1 if s == "Failed" else 2 for s in states]   # 2: timeout or deleted
        return max(codes)
    if command == "stop":
        for name in positionals:
            job = read(name)
            if job is None: missing(name); return 1
            write(name, dict(job, state=os.environ.get("FAKE_STOP_STATE", "Deleting"), wiped=True))
        return 0
    if command == "log":
        job = read(positionals[0])
        if job is None: missing(positionals[0]); return 1
        if not job.get("crashed"): print("NEXTT_RUN_STATUS training_complete 512 1000")
        return 0
    if command == "exec":
        # Runs COMMAND here, as the debug pod would; with -q only COMMAND's own output reaches stdout.
        if not (options.get("-q") or options.get("--quiet")): print("连接到 fake-debug-pod")
        return subprocess.run(rest, cwd=options.get("--cwd")).returncode
    raise AssertionError(command)


if __name__ == "__main__":
    sys.exit(main())
