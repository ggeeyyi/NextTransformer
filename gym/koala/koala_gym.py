#!/usr/bin/env python3
"""Koala for the Gym's multi-node executor. This is the only Koala-specific file.

It plugs into the Gym only through the Gym's public interfaces (nexttransformer_gym.nodes, .executor, .remote; see
the Gym's DESIGN.md, "Operator executors"). The Gym's controller, API, budget and evidence code run unchanged on this
machine, with or without local GPUs, and the Gym's multi-node executor (`nodes`) runs every job, as it does on local
machines over ssh; this file is its Koala launcher:

* `trial` and `cli` write a Gym execution configuration whose `executor` block names this file's relay, the job's
  cards (virtual identities, the configured GPU name), the pods' recorded runtime (`probe`) and the pods'
  confinement, and then start the Gym's own entry points (run_trial.py, nexttransformer_gym.cli) unmodified;
* the Gym starts the relay for each job it schedules: `nodes.relay` with `KoalaLauncher`, which publishes a code
  bundle, submits one Koala job of the job's nodes (waiting while the Normal quota is full), reads node 0's output
  from the Koala log and stops the job; the Gym's relay speaks the worker protocol, bills ahead of the observed
  work, polls for cancel_job, journals the job and keeps the checkpoint pointer;
* each pod builds the environment, finds its node index and node 0's address on shared storage, fetches what an
  object store holds, and then runs the Gym's node (`nodes.run`): the runtime check, the worker identity,
  environment and confinement, and on a group of nodes the mesh check and mesh worker.

Static audits (`inspect_cost`, calibration grids) run in the Gym's local sandbox (`local_isolation`, bubblewrap by
default).

Isolation. Candidate code trains in the pods, so the executor block reports the pod backend (`pod_isolation`: the
Gym's `bwrap`, `landlock` or `none`, with `pod_network`). Each pod checks that backend's prerequisites as root before
work and fails closed, without a charge, when they are missing. As in the Gym, a root runner (here the pod) runs
every non-official worker as `nobody`; official final runs execute the operator's admitted snapshot without a
sandbox, as the Gym does.

Clusters. One file serves both Koala GPU clusters; a configuration names one (`cluster`).
* H20 (`h20`): kernel 5.10 without Landlock or user namespaces, so `pod_isolation` is `none` (the worker runs as
  `nobody` with `pod_network` `deny`); CPFS holds `remote_root`, data and checkpoints; bundles go to OSS.
* H200 (`h200`): kernel 6.12 with Landlock and user namespaces, so `pod_isolation` can be `landlock`; there is
  no CPFS, so `remote_root` is an `s3://` prefix (the pods' own credentials reach it): a pod works under
  `pod_remote_dir`, uploads the job's checkpoint, result and logs when it ends, and downloads an earlier job's
  checkpoint before continuing from it; `data_stage` maps the pod-local data paths in profiles to S3 prefixes, and
  a research job stages only its own data (never the validation shard). Without a shared directory a final run
  has one node. Bundles go to S3. `relay_env` passes the credentials either store needs.
`probe` confirms a configuration's backend and runtime on its cluster before use.

Research GPUs. A research job uses `research_gpus` GPUs: the executor block lists that many cards, the Gym leases
them as one group and splits each update across them, and the pods map them onto their own GPUs in order. Up to
`research_gpus_per_node` GPUs fit one node; a larger group spans research_gpus / research_gpus_per_node nodes, and
the Gym's split of each update is the same as on one machine with that many GPUs. `research_microbatch_size` is the
Gym's rows per GPU per pass; an update accumulates gradients over its passes. A check with a final profile uses one
GPU.

Multi-node research. The pods meet through shared storage (`<remote>/rendezvous`); the rest is the Gym's
multi-node executor (nodes.research_node): the mesh check before the job is reported ready, the mesh worker on each
node's share of the group (cards node*k .. node*k+k-1), node 0 reporting and alone writing the checkpoint, and
markers under `<remote>/research/` (a Koala job ends when any of its pods ends, so the other nodes wait for node 0).
It needs a shared `remote_root` (the rendezvous) and `pod_isolation` `none`: the Landlock backend unshares the
network namespace.

Commands (controller, with the Gym environment's Python):
  koala_gym.py --koala koala.json probe                    record the pod runtime once
  koala_gym.py --koala koala.json trial <run_trial.py arguments>
  koala_gym.py --koala koala.json cli <nexttransformer_gym.cli arguments>
  koala_gym.py --koala koala.json jobs --state <gym state> [--stop-orphans]
Internal:
  koala_gym.py relay koala.json                            the executor command the Gym starts per job
  koala_gym.py pod job.json                                Koala job entry on every node

Accounting. A pod cannot reach this machine and a normal job has no interactive entry, so node 0 acknowledges the
worker's operations itself, within the job's reservation, and the Gym's relay bills ahead of the work it reads from
the log (executor.Charges): the first window when the pod reports that the worker starts, more as progress arrives,
and exactly the quoted bound when the pod reports success. A job that ends before its worker starts is charged
nothing; a started job that fails or is cancelled is charged its started windows; an unknown outcome keeps the
reservation for the operator's `reconcile`.

Multi-node final runs. Node 0 runs the Gym's official pipeline; its rank spawn is split across the nodes, which
meet over TCP at node 0 (found through shared storage). `official_smoke` bounds that path for tests (trainer smoke
steps, no audit or export); its results are never official. This path still adapts the Gym's official pipeline
in-process (see `install_official`).
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THIS = Path(__file__).resolve()
MARK = "KOALA_GYM "
CHECKPOINT_SCHEMA = "koala_gym.checkpoint.v1"
LABEL = "koala-job"
ENDED = ("Succeeded", "Failed", "Stopped", "Deleting")
CHUNK = 3000
VIRTUAL_UUID = "GPU-4b6f616c-6167-7900-8000-{:012x}"  # a placeholder identity, never a local device
PLATFORM_FILES = ("pyproject.toml", "uv.lock", "README.md", ".python-version")
PLATFORM_DIRS = ("nexttransformer", "nanochat")
# Caches and work files never ship (a code bundle counts its files against the publisher's preview).
IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "node_modules",
    ".agent-work",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
)

DEFAULTS = {
    "schema_version": "koala_gym.v1",
    "cluster": "h20",
    "koala": "koala",
    "poll_seconds": 30,
    "ssd_gib": 64,
    # The node's large local disk (`--large-ssd`, shared by the jobs on a node) instead of `ssd_gib`: final runs
    # that stage the whole dataset from object storage need it.
    "large_ssd": False,
    "code_mount": "/data/work/run_codes",
    "pod_work_dir": "/local-ssd/koala-gym",
    "pod_python": None,
    "pod_env_command": ["uv", "sync", "--frozen", "--extra", "gpu", "--python", "3.12"],
    "pod_isolation": "bwrap",
    "local_isolation": "bwrap",
    # Research workers in a pod without a sandbox ("none"): "deny" refuses new network sockets with seccomp.
    "pod_network": "deny",
    # GPUs of one research job: the Gym's execution configuration lists this many (virtual) cards, and the Gym splits
    # each update across all of them. Checks with a final profile still use one GPU.
    "research_gpus": 1,
    # GPUs per node: a research group larger than this spans research_gpus / research_gpus_per_node nodes.
    "research_gpus_per_node": 8,
    # How long a multi-node research worker waits for the other nodes at a collective (compilation, checkpoints).
    "research_mesh_timeout_seconds": 3600,
    # Rows per GPU per forward/backward pass (the Gym's research_microbatch_size); updates accumulate gradients.
    "research_microbatch_size": 1,
    # Precision and compiler settings for CUDA research/validation workers (static audits stay float32 on CPU).
    "research_dtype": "float32",
    "research_compile": False,
    "final": {"nodes": 1, "gpus_per_node": 8},
    "official_smoke": None,
    "site_config": None,
    "virtual_gpu_offset": 0,
    # The cards' name agents see (get_resources, results).
    "gpu_name": "GPU",
    "gpu_memory_total_bytes": None,
    "gpu_memory_limit_bytes": None,
    "runtime": None,
    "worker_python": None,
    # Code-bundle credentials the relay's `koala code publish` needs: OSS for H20, S3 for H200.
    "relay_env": [
        "OSS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
    ],
    "relay_margin_seconds": 300,
    # While Koala refuses submissions because the Normal quota is in use, the relay retries at this interval
    # until the profile timeout (the Gym already treats a Koala job as queued until its pod starts).
    "quota_wait_seconds": 120,
    "rendezvous_seconds": 1800,
    "spawn_wait_seconds": 7200,
    "master_port": 29531,
    "keep_staging": 20,
    # `probe` confinement checks: {"data": <a Gym data profile>, "forbidden": [paths], "reserved": [paths],
    # "network": [[host, port]]}. `forbidden` paths are outside a sandbox's allowed set; `reserved` paths (files or
    # directories) must stay unreachable to the worker identity under every backend, `none` included, because file
    # permissions protect them: the validation shard, CORE-22 data, other runs' outputs and weights.
    "probe_checks": None,
    # Object storage (an `s3://` remote_root, for clusters without a shared filesystem): the pod-local mirror of
    # remote_root, the s5cmd command, [pod-local prefix, s3:// prefix] pairs for data, and inline site values
    # (written in the pod for final runs, in place of a `site_config` path on shared storage).
    "pod_remote_dir": "/local-ssd/harbor-gym",
    "s5cmd": "s5cmd",
    "data_stage": [],
    "site": None,
}
REQUIRED = ("home", "staging_dir", "remote_root", "code_project", "job_prefix")


class KoalaError(RuntimeError):
    pass


class QuotaFull(KoalaError):
    """Koala refused or revoked a submission because the Normal quota is in use; a later retry can succeed."""


# Koala CLI messages for a full quota: a refused submission, and a job revoked by the non-atomic quota check.
QUOTA_FULL = ("配额不足", "并发提交撞车")


class PodError(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


# ---------------------------------------------------------------------------------------------- config


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if "virtual_gpus" in value:
        raise SystemExit(
            f"koala_gym: virtual_gpus in {path} is now research_gpus (GPUs per research job)"
        )
    unknown = set(value) - set(DEFAULTS) - set(REQUIRED)
    if unknown:
        raise SystemExit(f"koala_gym: unknown settings {sorted(unknown)} in {path}")
    config = dict(DEFAULTS)
    config.update(value)
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        raise SystemExit(f"koala_gym: {path} needs {missing}")
    if config["schema_version"] != "koala_gym.v1":
        raise SystemExit("koala_gym: unsupported schema_version")
    if re.search("debug|normal", config["job_prefix"]):
        raise SystemExit("koala_gym: job_prefix must not contain 'debug' or 'normal'")
    if config["pod_isolation"] not in ("bwrap", "landlock", "none"):
        raise SystemExit(
            "koala_gym: pod_isolation must be 'bwrap', 'landlock' or 'none'"
        )
    # The controller runs as an ordinary user, and the Gym's landlock backend needs a root controller.
    if config["local_isolation"] not in ("bwrap", "none"):
        raise SystemExit("koala_gym: local_isolation must be 'bwrap' or 'none'")
    if config["pod_network"] not in ("deny", "allow"):
        raise SystemExit("koala_gym: pod_network must be 'deny' or 'allow'")
    for key in (
        "research_gpus",
        "research_microbatch_size",
        "research_gpus_per_node",
        "research_mesh_timeout_seconds",
    ):
        if not (type(config[key]) is int and config[key] > 0):
            raise SystemExit(f"koala_gym: {key} must be a positive integer")
    if config["research_gpus"] > config["research_gpus_per_node"]:
        if config["research_gpus"] % config["research_gpus_per_node"]:
            raise SystemExit(
                "koala_gym: research_gpus must be a multiple of research_gpus_per_node to span nodes"
            )
        if object_store(config["remote_root"]):
            raise SystemExit(
                "koala_gym: multi-node research needs a shared remote_root for the nodes' rendezvous"
            )
        if config["pod_isolation"] != "none":
            raise SystemExit(
                "koala_gym: multi-node research needs pod_isolation 'none' (Landlock unshares the network)"
            )
    if config["research_dtype"] not in ("float32", "bfloat16"):
        raise SystemExit("koala_gym: research_dtype must be 'float32' or 'bfloat16'")
    if type(config["research_compile"]) is not bool:
        raise SystemExit("koala_gym: research_compile must be a boolean")
    total, limit = config["gpu_memory_total_bytes"], config["gpu_memory_limit_bytes"]
    if not (type(total) is int and type(limit) is int and 0 < limit <= total):
        raise SystemExit(
            "koala_gym: set gpu_memory_total_bytes >= gpu_memory_limit_bytes > 0"
        )
    smoke = config["official_smoke"]
    if smoke is not None and not (
        isinstance(smoke, dict) and int(smoke.get("steps", 0)) > 0
    ):
        raise SystemExit("koala_gym: official_smoke needs a positive 'steps'")
    stages = config["data_stage"]
    if not (
        isinstance(stages, list)
        and all(
            isinstance(p, list)
            and len(p) == 2
            and str(p[1]).startswith("s3://")
            and str(p[0]).startswith("/")
            for p in stages
        )
    ):
        raise SystemExit(
            "koala_gym: data_stage is a list of [pod-local prefix, s3:// prefix] pairs"
        )
    if object_store(config["remote_root"]) and int(config["final"]["nodes"]) != 1:
        raise SystemExit(
            "koala_gym: an s3:// remote_root has no shared directory for a multi-node final run; "
            "set final.nodes to 1"
        )
    if config["site"] is not None and not isinstance(config["site"], dict):
        raise SystemExit("koala_gym: site holds inline site values (a JSON object)")
    return config


def object_store(root):
    return str(root).startswith("s3://")


def local_root(config):
    """Where pods keep job files: remote_root itself, or its pod-local mirror under object storage."""
    return (
        config["pod_remote_dir"]
        if object_store(config["remote_root"])
        else config["remote_root"]
    )


class Store:
    """The s5cmd calls a pod makes against an s3:// remote_root (the pod's own credentials)."""

    def __init__(self, command="s5cmd"):
        self.command = command

    def run(self, *args, timeout=7200):
        done = subprocess.run(
            [self.command, *args], capture_output=True, text=True, timeout=timeout
        )
        if done.returncode:
            raise PodError(
                "resource_unavailable",
                f"s5cmd {args[0]} failed: {(done.stderr or done.stdout)[-500:]}",
            )
        return done.stdout

    def put(self, path, uri):
        self.run("cp", str(path), uri)

    def get(self, uri, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.run("cp", uri, str(path))

    def exists(self, uri):
        done = subprocess.run(
            [self.command, "ls", uri], capture_output=True, text=True, timeout=300
        )
        return done.returncode == 0 and any(
            line.split()[-1:] == [uri.rsplit("/", 1)[-1]]
            for line in done.stdout.splitlines()
        )

    def listing(self, prefix):
        """Objects under prefix: {relative key: size}."""
        done = subprocess.run(
            [self.command, "ls", prefix.rstrip("/") + "/*"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        found = {}
        for line in done.stdout.splitlines() if done.returncode == 0 else ():
            parts = line.split()
            if len(parts) >= 4 and parts[-2].isdigit():
                found[parts[-1]] = int(parts[-2])
        return found

    def mirror(self, prefix, directory):
        """Copy every object under prefix into directory, skipping files already there at the same size."""
        directory = Path(directory)
        lines = []
        for key, size in sorted(self.listing(prefix).items()):
            target = directory / key
            if not (target.is_file() and target.stat().st_size == size):
                target.parent.mkdir(parents=True, exist_ok=True)
                lines.append(f'cp "{prefix.rstrip("/")}/{key}" "{target}"')
        self.batch(lines)

    def upload(self, directory, prefix):
        directory = Path(directory)
        self.batch(
            [
                f'cp "{path}" "{prefix.rstrip("/")}/{path.relative_to(directory)}"'
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            ]
        )

    def batch(self, lines):
        if not lines:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as handle:
            handle.write("\n".join(lines) + "\n")
        try:
            self.run("run", handle.name)
        finally:
            os.unlink(handle.name)


def save_runtime(path, runtime):
    path = Path(path)
    value = json.loads(path.read_text())
    value["runtime"] = runtime
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------- koala


class Koala:
    """The Koala CLI calls this file needs, always with an explicit cluster."""

    def __init__(self, config):
        self.config = config
        self.env = dict(os.environ)
        # The Gym starts its workers with HOME set to its state directory; the CLI needs the operator's.
        self.env["HOME"] = config["home"]
        extra = [str(Path(config["koala"]).parent)] if os.sep in config["koala"] else []
        self.env["PATH"] = os.pathsep.join(
            extra + [os.environ.get("PATH", ""), "/usr/local/bin", "/usr/bin", "/bin"]
        )

    def run(self, *args, timeout=300):
        command = [self.config["koala"], "--cluster", self.config["cluster"], *args]
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=self.env,
                cwd=self.config["home"],
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(command, 124, "", "koala call timed out")

    @staticmethod
    def parse(result):
        if result.returncode:
            raise KoalaError((result.stderr or result.stdout).strip()[-1500:])
        text = result.stdout
        try:
            return json.loads(text[text.index("{") :])
        except ValueError:
            raise KoalaError(f"unreadable koala output: {text[-500:]}") from None

    def publish(self, directory):
        """Publish `directory` as an immutable bundle; only the previewed content is uploaded."""
        base = [
            "code",
            "publish",
            str(directory),
            "--project",
            self.config["code_project"],
            "--json",
        ]
        preview = self.parse(self.run(*base, "--dry-run"))
        staged = sum(1 for p in Path(directory).rglob("*") if p.is_file())
        if preview.get("files") != staged:
            raise KoalaError(
                f"bundle preview lists {preview.get('files')} files; {staged} were staged"
            )
        done = self.parse(
            self.run(
                *base, "--expect-sha256", preview["bundle_sha256"], "-y", timeout=900
            )
        )
        if (
            not done.get("bundle_uri")
            or done.get("bundle_sha256") != preview["bundle_sha256"]
        ):
            raise KoalaError(f"koala code publish returned {done}")
        return done["bundle_uri"], done["bundle_sha256"]

    def submit(self, command, bundle_uri, nodes, gpus):
        disk = (
            ["--large-ssd"]
            if self.config["large_ssd"]
            else ["--ssd", str(self.config["ssd_gib"])]
        )
        result = self.run(
            "submit",
            "-m",
            "normal",
            "-n",
            str(nodes),
            "-g",
            str(gpus),
            *disk,
            "-q",
            "-y",
            "-j",
            self.config["job_prefix"],
            "--code-bundle",
            f"{bundle_uri}:{self.config['code_mount']}",
            "-c",
            command,
            timeout=600,
        )
        names = [
            line.strip()
            for line in result.stdout.splitlines()
            if re.fullmatch(r"[a-z0-9][a-z0-9-]{8,}", line.strip())
        ]
        text = f"{result.stdout}\n{result.stderr}"
        if any(marker in text for marker in QUOTA_FULL):
            # A revoked job may already exist; stopping it again is harmless and leaves nothing behind.
            for name in set(
                re.findall(r"[a-z0-9][a-z0-9-]{8,}-normal-\d{8}-\d{6}", text)
            ):
                self.stop(name)
            raise QuotaFull(
                next(
                    line.strip()
                    for line in text.splitlines()
                    if any(marker in line for marker in QUOTA_FULL)
                )[-500:]
            )
        if result.returncode or not names:
            raise KoalaError(
                (result.stderr or result.stdout).strip()[-1500:]
                or "koala submit returned no job"
            )
        return names[-1]

    def status(self, job):
        result = self.run("get", job, "--json", timeout=120)
        if result.returncode:
            return None
        try:
            return self.parse(result).get("status") or ""
        except KoalaError:
            return None

    def log(self, job, worker=0):
        result = self.run(
            "log", job, "--worker", str(worker), "-n", "1000", timeout=120
        )
        return result.stdout.splitlines() if result.returncode == 0 else None

    def stop(self, job):
        return self.run("stop", "-y", job, timeout=300).returncode


def events(lines):
    found = []
    for line in lines or ():
        index = line.find(MARK)
        if index < 0:
            continue
        try:
            found.append(json.loads(line[index + len(MARK) :]))
        except ValueError:
            continue
    return found


def stage(config, name, payload, platform, snapshot=None):
    """Copy what a pod needs into a fresh directory outside any git work tree."""
    root = Path(config["staging_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inside = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if inside.stdout.strip() == "true":
        raise KoalaError(
            f"{root} is inside a git work tree; koala code publish would pack tracked files only"
        )
    target = root / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    import nexttransformer_gym

    for directory in PLATFORM_DIRS:
        shutil.copytree(Path(platform) / directory, target / directory, ignore=IGNORE)
    shutil.copytree(
        Path(nexttransformer_gym.__file__).parent,
        target / "gym" / "src" / "nexttransformer_gym",
        ignore=IGNORE,
    )
    for file in PLATFORM_FILES:
        if (Path(platform) / file).is_file():
            shutil.copy2(Path(platform) / file, target / file)
    if snapshot is not None:
        shutil.copytree(snapshot, target / "snapshot", ignore=IGNORE)
    shutil.copy2(THIS, target / "koala_gym.py")
    (target / "job.json").write_text(json.dumps(payload, indent=1))
    old = sorted(
        (p for p in root.iterdir() if p.is_dir() and p != target),
        key=lambda p: p.stat().st_mtime,
    )
    for path in old[: max(0, len(old) - int(config["keep_staging"]))]:
        shutil.rmtree(path, ignore_errors=True)
    return target


def pod_command(config, entry):
    return f"cd {shlex.quote(config['code_mount'])} && python3 koala_gym.py {entry} job.json"


# ------------------------------------------------------------------------------------------ controller


def virtual_ids(config):
    """Gym GPU leases coordinate processes per card identity; give each configuration its own cards."""
    return [
        VIRTUAL_UUID.format(int(config["virtual_gpu_offset"]) + i)
        for i in range(int(config["research_gpus"]))
    ]


def execution_settings(config, config_path):
    """The Gym execution configuration for a Koala configuration: this file's relay as the operator executor."""
    return {
        "schema_version": "gym.execution.v1",
        "gpu_memory_limit_bytes": config["gpu_memory_limit_bytes"],
        "research_microbatch_size": config["research_microbatch_size"],
        "research_dtype": config["research_dtype"],
        "research_compile": config["research_compile"],
        "max_concurrent_jobs": 1,
        # Static audits run on this machine.
        "isolation": config["local_isolation"],
        "executor": {
            "command": [
                sys.executable,
                str(THIS),
                "relay",
                str(Path(config_path).resolve()),
            ],
            # Only the relay receives the code-bundle credentials.
            "environment": list(config["relay_env"]),
            "devices": [
                {
                    "id": uuid,
                    "name": config["gpu_name"],
                    "memory_total_bytes": config["gpu_memory_total_bytes"],
                }
                for uuid in virtual_ids(config)
            ],
            "runtime": config["runtime"],
            "isolation": config["pod_isolation"],
            "network": config["pod_network"],
        },
    }


def execution_config(config, config_path):
    # One file per Koala configuration: sessions with other offsets or backends may start concurrently.
    config_path = Path(config_path).resolve()
    path = config_path.with_name(f"execution.{config_path.stem}.json")
    value = execution_settings(config, config_path)
    if not path.exists() or json.loads(path.read_text()) != value:
        path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def check_worker_python(config, python):
    """The local worker Python (static audits, prepare-final calibration) must carry the pod's torch build:
    formal admission compares the calibration's torch version with the pod's torch.__version__."""
    local = runtime_of(python)["dependencies"]
    pod = (config["runtime"] or {}).get("dependencies") or {}
    if local.get("torch") != pod.get("torch"):
        raise SystemExit(
            f"koala_gym: worker Python {python} has torch {local.get('torch')}, the pods have "
            f"{pod.get('torch')}; set worker_python to an environment with the pods' torch build "
            "(e.g. `UV_PROJECT_ENVIRONMENT=<dir> uv sync --frozen --extra gpu --python 3.12`)"
        )
    return python


def controller(config_path, mode, argv):
    """Write the execution configuration, then start the Gym's own entry point with it."""
    if os.geteuid() == 0:
        raise SystemExit("koala_gym: run the controller as an ordinary user")
    config = load_config(config_path)
    if not config["runtime"]:
        raise SystemExit(
            "koala_gym: no pod runtime recorded; run `koala_gym.py --koala ... probe` first"
        )
    if "--execution-config" not in argv:
        argv = ["--execution-config", str(execution_config(config, config_path)), *argv]
    if "--worker-python" in argv:
        check_worker_python(config, argv[argv.index("--worker-python") + 1])
    elif config["worker_python"]:
        argv = [
            "--worker-python",
            check_worker_python(config, config["worker_python"]),
            *argv,
        ]
    else:
        check_worker_python(config, sys.executable)
    import nexttransformer_gym

    if mode == "cli":
        command = [sys.executable, "-m", "nexttransformer_gym.cli", *argv]
    else:
        script = (
            Path(nexttransformer_gym.__file__).resolve().parents[2] / "run_trial.py"
        )
        command = [sys.executable, str(script), *argv]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, command)


# ----------------------------------------------------------------------------------------------- relay


# Updates billed ahead of the progress the relay observes (executor.Charges).
TRAIN_WINDOW_STEPS = 16


def utc(seconds=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def journal(state, **entry):
    with open(state / "koala-jobs.jsonl", "a") as handle:
        handle.write(json.dumps({"time": utc(), **entry}) + "\n")


def final_topology(config, profile):
    smoke = config["official_smoke"]
    shape = smoke or config["final"]
    nodes, gpus = int(shape.get("nodes", 1)), int(shape.get("gpus_per_node", 8))
    if not smoke and nodes * gpus != len(profile.get("cuda_visible_devices") or []):
        raise KoalaError(
            f"final topology {nodes}x{gpus} differs from the profile's GPU allocation"
        )
    return nodes, gpus


def build_job(config, request, remote):
    """The pod's job description: the Gym request with pod paths, and the snapshot to ship (final runs)."""
    quote, profile = request["quote"], request["profile"]
    operation = quote["operation"]
    work = f"{config['pod_work_dir'].rstrip('/')}/{quote['id']}"
    code = f"{work}/code"
    # With object storage the pod works in a local mirror of remote and uploads it when the job ends.
    uri = remote if object_store(remote) else None
    if uri:
        remote = f"{config['pod_remote_dir'].rstrip('/')}/{'/'.join(remote.rstrip('/').split('/')[-2:])}"
    job_request = json.loads(json.dumps(request))
    job_request.update(
        platform_root=code,
        output=f"{remote}/checkpoint.pt",
        final_output_root=f"{remote}/final-runs",
    )
    checkpoint = None
    if request.get("checkpoint"):
        pointer = json.loads(Path(request["checkpoint"]).read_text())
        if pointer.get("schema") != CHECKPOINT_SCHEMA:
            raise PodError(
                "incompatible_checkpoint",
                "the retained checkpoint is not a Koala checkpoint pointer",
            )
        checkpoint = {"path": pointer["remote_path"], "sha256": pointer["sha256"]}
        if object_store(pointer["remote_path"]):
            checkpoint.update(
                uri=pointer["remote_path"], path=f"{work}/input/checkpoint.pt"
            )
        job_request["checkpoint"] = checkpoint["path"]
    snapshot = None
    if operation == "official":
        plan = job_request["profile"]["official_plan"]
        snapshot = Path(plan["snapshot"])
        plan["snapshot"] = f"{code}/snapshot"
        nodes, gpus = final_topology(config, profile)
    else:
        nodes, gpus = research_topology(config, request.get("gpu"))
    payload = {
        "schema": "koala_gym.job.v1",
        "quote_id": quote["id"],
        "request": job_request,
        "remote": remote,
        "work": work,
        "runtime": config["runtime"],
        "checkpoint": checkpoint,
        "credit": str(int(quote["bound"])),
        "nodes": nodes,
        "gpus_per_node": gpus,
        "research_dtype": config["research_dtype"],
        "research_compile": config["research_compile"],
        "isolation": config["pod_isolation"],
        "network": config["pod_network"],
        "remote_root": config["pod_remote_dir"] if uri else config["remote_root"],
        "remote_uri": uri,
        "s5cmd": config["s5cmd"],
        "data_stage": config["data_stage"],
        "site": config["site"],
        "pod_python": config["pod_python"],
        "env_command": config["pod_env_command"],
        "site_config": config["site_config"],
        "smoke": config["official_smoke"] if operation == "official" else None,
        "rendezvous_seconds": config["rendezvous_seconds"],
        "spawn_wait_seconds": config["spawn_wait_seconds"],
        "master_port": config["master_port"],
        "mesh_timeout_seconds": config["research_mesh_timeout_seconds"],
    }
    return payload, snapshot


def research_topology(config, allocation):
    """(nodes, GPUs per node) of a non-official job: a group larger than one node spans whole nodes; a CPU job
    still gets one GPU pod."""
    from nexttransformer_gym import nodes

    return nodes.research_topology(
        len(nodes.allocated_devices(allocation)), int(config["research_gpus_per_node"])
    )


class KoalaLauncher:
    """The Koala transport of the Gym's multi-node executor (a nexttransformer_gym.nodes.Launcher): a job is one
    Koala job of `nodes` pods running this file's `pod` entry, its code a published bundle, and node 0's output the
    Koala log. The Gym's relay does the rest: the worker protocol, billing ahead of the observed work, cancel_job,
    the profile timeout, the journal (`koala-jobs.jsonl`) and the checkpoint pointer."""

    name = "Koala"
    mark = MARK
    journal_file = "koala-jobs.jsonl"
    checkpoint_schema = CHECKPOINT_SCHEMA

    def __init__(self, config):
        self.config, self.koala = config, Koala(config)
        self.margin_seconds = int(config["relay_margin_seconds"])
        self.poll_seconds = float(config["poll_seconds"])
        # While Koala refuses submissions because the Normal quota is in use.
        self.busy_seconds = float(config["quota_wait_seconds"])
        self.window = TRAIN_WINDOW_STEPS

    def plan(self, request, tag):
        remote = (
            f"{self.config['remote_root'].rstrip('/')}/{tag}/{request['quote']['id']}"
        )
        return build_job(self.config, request, remote)

    def prepare(self, payload, request, snapshot):
        staged = stage(
            self.config,
            payload["quote_id"],
            payload,
            request["platform_root"],
            snapshot,
        )
        bundle, digest = self.koala.publish(staged)
        return {
            "bundle": bundle,
            "bundle_sha256": digest,
            "remote": payload.get("remote_uri") or payload["remote"],
        }

    def start(self, record, nodes, gpus):
        from nexttransformer_gym.nodes import Busy

        try:
            return self.koala.submit(
                pod_command(self.config, "pod"), record["bundle"], nodes, gpus
            )
        except QuotaFull as error:
            raise Busy(str(error)) from None

    def lines(self, job):
        return self.koala.log(job)

    def ended(self, job):
        status = self.koala.status(job)
        return status if status in ENDED else None

    def stop(self, job):
        self.koala.stop(job)


def relay(config_path):
    """The executor command the Gym starts per job: the Gym's multi-node relay over Koala."""
    from nexttransformer_gym.nodes import relay as nodes_relay

    return nodes_relay(KoalaLauncher(load_config(config_path)))


# ---------------------------------------------------------------------------------------------- jobs

TERMINAL = ("ended", "stopped", "unknown")


def live_relays(state):
    """PIDs of this user's relays whose working directory is `state` (a relay runs in its Gym state)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = os.readlink(entry / "cwd")
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            cwd == str(state)
            and any(a.endswith(b"koala_gym.py") for a in argv)
            and b"relay" in argv
        ):
            found.append(int(entry.name))
    return found


def gym_jobs(state):
    """The Gym's job records by quote: (job id, owner session id, state); read-only."""
    from nexttransformer_gym.executor import jobs as records

    return {
        record["quote_id"]: (record["id"], record["session_id"], record["state"])
        for record in records(state)
        if record.get("quote_id")
    }


def jobs(config_path, argv):
    """List a Gym state's Koala jobs with their GPU-hours; optionally stop jobs no live relay drives."""
    parser = argparse.ArgumentParser(prog="koala_gym.py --koala CONFIG jobs")
    parser.add_argument(
        "--state", required=True, type=Path, help="the Gym state directory"
    )
    parser.add_argument(
        "--stop-orphans",
        action="store_true",
        help="stop open Koala jobs that no live relay drives, then print the reconcile commands",
    )
    args = parser.parse_args(argv)
    config, state = load_config(config_path), args.state.resolve()
    koala = Koala(config)
    path = state / "koala-jobs.jsonl"
    entries = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )
    by_job = {}
    for entry in entries:
        by_job.setdefault(entry["job"], []).append(entry)
    relays, records = live_relays(state), gym_jobs(state)
    open_jobs = [
        job
        for job, rows in by_job.items()
        if not any(r["event"] in TERMINAL for r in rows)
    ]
    # One worker slot per Gym state: a live relay drives the newest open job.
    driven = open_jobs[-1] if relays and open_jobs else None
    total, reconcile = 0.0, []
    for job, rows in by_job.items():
        first, last = rows[0], rows[-1]
        terminal = next((r for r in reversed(rows) if r["event"] in TERMINAL), None)
        hours = (terminal or {}).get("gpu_hours")
        total += hours or 0.0
        line = (
            f"{job}  {first.get('operation', '?'):<8} quote {first.get('quote_id')}  submitted {first['time']}  "
            f"last {last['event']} {last['time']}"
        )
        if terminal:
            line += f"  outcome {terminal.get('outcome') or terminal.get('reason')}"
            line += (
                f"  gpu_hours {hours}"
                if hours is not None
                else "  gpu_hours unrecorded"
            )
        else:
            status = koala.status(job)
            line += f"  OPEN (Koala: {status or 'unreadable'})"
            if job == driven:
                line += f"  driven by live relay {relays}"
            elif args.stop_orphans:
                if status in ENDED:
                    journal(
                        state,
                        quote_id=first.get("quote_id"),
                        job=job,
                        event="unknown",
                        reason=f"orphaned; the Koala job had already ended ({status})",
                    )
                    line += "  -> recorded as unknown"
                elif status is None:
                    line += "  -> not stopped: the Koala status is unreadable"
                    print(line)
                    continue
                else:
                    code = koala.stop(job)
                    if code:
                        line += f"  -> koala stop failed ({code}); not recorded"
                        print(line)
                        continue
                    # Pod time from the relay's "running" entry to the stop; unrecorded if the pod never reported.
                    pod_started = next(
                        (
                            r.get("pod_started_t")
                            for r in rows
                            if r["event"] == "running"
                        ),
                        None,
                    )
                    from nexttransformer_gym.nodes import usage

                    held = usage(
                        pod_started,
                        time.time(),
                        first.get("nodes", 1),
                        first.get("gpus_per_node", 1),
                    )
                    journal(
                        state,
                        quote_id=first.get("quote_id"),
                        job=job,
                        event="stopped",
                        reason="orphaned; stopped by the operator",
                        koala_status=status,
                        **held,
                    )
                    total += held.get("gpu_hours", 0.0)
                    line += "  -> stopped" + (
                        f"  gpu_hours {held['gpu_hours']}"
                        if held
                        else "  gpu_hours unrecorded"
                    )
                terminal = {"event": "unknown"}
        print(line)
        record = records.get(first.get("quote_id"))
        if (
            terminal
            and terminal["event"] in ("unknown", "stopped")
            and record
            and record[2] not in ("succeeded", "failed", "cancelled", "pre_failure")
        ):
            reconcile.append(record)
    print(
        f"{len(by_job)} Koala jobs, {len(open_jobs)} open, {round(total, 4)} GPU-hours recorded"
    )
    for job_id, owner, gym_state in reconcile:
        print(
            f"Gym job {job_id} is {gym_state}; after confirming its Koala job has stopped, reconcile it with the "
            f"trusted controller: nexttransformer_gym.cli ... reconcile --session-id {owner} --job-id {job_id} "
            "--worker-confirmed-stopped"
        )
    return 0


# --------------------------------------------------------------------------------------------- probe


def probe(config_path):
    """Run one 1-GPU job that builds the pod environment and reports its runtime; record it."""
    config = load_config(config_path)
    koala = Koala(config)
    import nexttransformer_gym

    platform = Path(nexttransformer_gym.__file__).resolve().parents[3]
    name = "probe-" + time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "schema": "koala_gym.probe.v1",
        "work": f"{config['pod_work_dir'].rstrip('/')}/{name}",
        "pod_python": config["pod_python"],
        "env_command": config["pod_env_command"],
        "isolation": config["pod_isolation"],
        "checks": config["probe_checks"],
        "remote": f"{local_root(config).rstrip('/')}/probes/{name}",
        "remote_root": local_root(config),
    }
    staged = stage(config, name, payload, platform)
    bundle, _ = koala.publish(staged)
    job = koala.submit(pod_command(config, "pod-probe"), bundle, 1, 1)
    print(f"probe job {job}", flush=True)
    while True:
        lines = koala.log(job)
        chunks = {}
        for event in events(lines):
            if event.get("kind") == "chunk" and event.get("name") == "confinement":
                chunks[event["i"]] = event["data"]
            if event.get("kind") == "probe":
                if event.get("confinement_chunks"):
                    if len(chunks) == event["confinement_chunks"]:
                        event["confinement"] = json.loads(
                            "".join(chunks[i] for i in range(len(chunks)))
                        )
                    else:
                        event["confinement"] = (
                            f"incomplete in the log; see {payload['remote']}/probe.json"
                        )
                saved = (
                    Path(config_path)
                    .resolve()
                    .with_name(f"{Path(config_path).stem}.{name}.json")
                )
                saved.write_text(json.dumps(event, indent=1, sort_keys=True) + "\n")
                print(json.dumps(event, indent=1))
                print(f"saved {saved}", flush=True)
                if event.get("reserved_ok") is False:
                    reached = {
                        name: c.get("reserved_reached")
                        for name, c in (event.get("confinement") or {}).items()
                        if isinstance(c, dict) and c.get("reserved_reached")
                    }
                    print(
                        f"RESERVED PATHS REACHED by the worker identity: {json.dumps(reached)}",
                        flush=True,
                    )
                if event.get("runtime"):
                    save_runtime(config_path, event["runtime"])
                    print(f"recorded the pod runtime in {config_path}")
                    isolated = (
                        event.get("isolation_ok") or config["pod_isolation"] == "none"
                    )
                    return (
                        0 if isolated and event.get("reserved_ok") is not False else 1
                    )
                return 1
        if koala.status(job) in ENDED and not any(
            e.get("kind") == "probe" for e in events(koala.log(job))
        ):
            print("\n".join((koala.log(job) or [])[-40:]))
            return 1
        time.sleep(float(config["poll_seconds"]))


# ------------------------------------------------------------------------------------------------ pod


class Emitter:
    """One numbered event per stdout line; node 0's lines are what the relay reads."""

    def __init__(self, seq=0):
        self.seq = seq

    def __call__(self, kind, **value):
        self.seq += 1
        event = dict(seq=self.seq, kind=kind, t=round(time.time(), 3), **value)
        print(
            MARK + json.dumps(event, allow_nan=False, separators=(",", ":")), flush=True
        )

    def blob(self, name, value):
        text = json.dumps(value, allow_nan=False, separators=(",", ":"))
        parts = [text[i : i + CHUNK] for i in range(0, len(text), CHUNK)] or [""]
        for i, part in enumerate(parts):
            self("chunk", name=name, i=i, data=part)
        return len(parts)


def host_name():
    return os.environ.get("KOALA_GYM_HOST") or socket.gethostname()


def host_address():
    return os.environ.get("KOALA_GYM_ADDRESS") or socket.gethostbyname(
        socket.gethostname()
    )


def worker_order(name):
    match = re.search(r"-worker-(\d+)$", name)
    return (int(match.group(1)), name) if match else (1 << 30, name)


def rendezvous(payload):
    """Node rank and node 0's address, agreed through the job's directory on shared storage."""
    if int(payload["nodes"]) == 1:
        return 0, host_address()
    directory = Path(payload["remote"]) / "rendezvous"
    directory.mkdir(parents=True, exist_ok=True)
    host = host_name()
    tmp = directory / f".{host}"
    tmp.write_text(host_address())
    tmp.replace(directory / host)
    deadline = time.time() + float(payload["rendezvous_seconds"])
    while True:
        hosts = sorted(
            (p.name for p in directory.iterdir() if not p.name.startswith(".")),
            key=worker_order,
        )
        if len(hosts) >= int(payload["nodes"]):
            break
        if time.time() > deadline:
            raise PodError(
                "resource_unavailable",
                f"only {len(hosts)} of {payload['nodes']} nodes met",
            )
        time.sleep(3)
    if len(hosts) != int(payload["nodes"]) or host not in hosts:
        raise PodError("resource_unavailable", f"unexpected rendezvous members {hosts}")
    return hosts.index(host), (directory / hosts[0]).read_text().strip()


def copy_code(source, target):
    """Copy the bundle onto local disk once; nodes sharing a disk (tests) race safely."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}-{os.getpid()}")
    shutil.copytree(source, tmp, ignore=IGNORE)
    try:
        tmp.rename(target)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not target.exists():
            raise


def build_env(payload, code, work):
    if payload.get("pod_python"):
        return payload["pod_python"]
    venv = work / "venv"
    # No package cache: on a pod's capped local disk it would double the environment's size.
    env = dict(
        os.environ,
        UV_PROJECT_ENVIRONMENT=str(venv),
        UV_NO_CACHE="1",
        UV_PYTHON_INSTALL_DIR=str(work / "uv-python"),
        UV_LINK_MODE="copy",
    )
    log = work / "env.log"
    with open(log, "w") as handle:
        code_ = subprocess.run(
            payload["env_command"],
            cwd=code,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode
    if code_:
        raise PodError(
            "resource_unavailable",
            "environment build failed: " + log.read_text()[-1500:],
        )
    return str(venv / "bin" / "python")


def pod(job_path):
    """Koala job entry (system python3): start once, meet the other nodes, build the environment."""
    payload = json.loads(Path(job_path).read_text())
    emit = Emitter()
    remote, work = Path(payload["remote"]), Path(payload["work"])
    host = host_name()
    emit("boot", host=host)
    remote.mkdir(parents=True, exist_ok=True)
    try:
        if payload.get("remote_uri"):
            # The pod-local disk goes with a replaced pod, so the start marker lives in the object store.
            store, marker = (
                Store(payload["s5cmd"]),
                f"{payload['remote_uri']}/started-{host}",
            )
            if store.exists(marker):
                raise FileExistsError(marker)
            (remote / f"started-{host}").write_text("")
            store.put(remote / f"started-{host}", marker)
        else:
            os.close(
                os.open(
                    remote / f"started-{host}", os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            )
    except FileExistsError:
        emit("restarted", host=host)
        return 1
    except PodError as error:
        emit(
            "done",
            outcome="pre_failure",
            error_code=error.code,
            detail=str(error)[-1500:],
        )
        return 1
    try:
        node, master = rendezvous(payload)
        work.mkdir(parents=True, exist_ok=True)
        code = work / "code"
        copy_code(Path(job_path).resolve().parent, code)
        python = build_env(payload, code, work)
    except PodError as error:
        emit(
            "done",
            outcome="pre_failure",
            error_code=error.code,
            detail=str(error)[-1500:],
        )
        return 1
    except Exception as error:
        emit(
            "done",
            outcome="pre_failure",
            error_code="resource_unavailable",
            detail=repr(error)[-1500:],
        )
        return 1
    emit("environment", node=node, python=python)
    sys.stdout.flush()
    argv = [
        python,
        str(code / "koala_gym.py"),
        "pod-run",
        str(code / "job.json"),
        str(node),
        master,
        str(emit.seq),
    ]
    os.execv(python, argv)


def runtime_of(python):
    """The Gym's runtime record of `python` (what the controller's revision binds)."""
    from nexttransformer_gym.remote import runtime

    return runtime(python)


def worker_env(payload, code, work, devices=None, official=False):
    """The Gym's worker environment in a pod (nexttransformer_gym.nodes.worker_env); a final run's trusted
    pipeline sees every GPU and runs in the official precision."""
    from nexttransformer_gym import nodes

    env = nodes.worker_env(
        dict(payload, request={}) if official else payload, code, work, devices
    )
    if official:
        env.pop("CUDA_VISIBLE_DEVICES")
        env.update(
            NANOCHAT_DTYPE="bfloat16",
            TORCHDYNAMO_DISABLE="0",
            FLA_DISABLE_BACKEND_DISPATCH="1",
            OMP_NUM_THREADS="8",
            PYTHONUNBUFFERED="1",
        )
        if payload.get("site_config"):
            env["NEXTTRANSFORMER_SITE_CONFIG"] = payload["site_config"]
    return env


def stage_inputs(payload, request):
    """Copy a job's data from object storage onto the pod (`data_stage`), readable by the worker identity.

    A research or check job stages exactly the files its profile names, and never the validation shard; a trusted
    final run mirrors every staged prefix (the whole dataset, the validation shard and CORE-22 included)."""
    stages = payload.get("data_stage") or []
    if not stages:
        return
    store = Store(payload["s5cmd"])
    if request["quote"]["operation"] == "official":
        for local, prefix in stages:
            store.mirror(prefix, local)
        return
    data = request["profile"].get("data") or {"kind": "synthetic_fixture"}
    if data.get("kind", "synthetic_fixture") == "synthetic_fixture":
        return
    from nexttransformer.runtime.files import VALIDATION_SHARD

    def source(path):
        for local, prefix in stages:
            if path == local or path.startswith(local.rstrip("/") + "/"):
                return prefix.rstrip("/") + path[len(local.rstrip("/")) :], local
        raise PodError(
            "resource_unavailable", f"{path} is outside every data_stage prefix"
        )

    files, directories = (
        [*data["train_paths"], data["dev_path"], data["token_bytes"]],
        [data["tokenizer"]],
    )
    if any(Path(p).name == VALIDATION_SHARD for p in files + directories):
        raise PodError(
            "denied_access", "The validation shard is reserved for the formal run."
        )
    lines, roots = [], set()
    for path in files:
        uri, root = source(str(path))
        roots.add(root)
        if not Path(path).is_file():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            lines.append(f'cp "{uri}" "{path}"')
    store.batch(lines)
    for path in directories:
        uri, root = source(str(path).rstrip("/"))
        roots.add(root)
        store.mirror(uri, path)

    # The worker identity reads the staged files: parents traversable, the stage tree readable.
    def allow(path, bits):
        mode = path.stat().st_mode
        if mode & bits != bits:
            path.chmod((mode & 0o7777) | bits)

    for root in map(Path, roots):
        for directory in root.parents:
            allow(directory, 0o001)
        for path in [root, *root.rglob("*")]:
            allow(path, 0o005 if path.is_dir() else 0o004)


def publisher(payload, remote):
    """Without shared storage (an s3:// remote_root) a pod uploads the job's files before node 0 reports, and
    later jobs find the checkpoint in the object store."""
    if not payload.get("remote_uri"):
        return None
    store = Store(payload["s5cmd"])

    def publish(checkpoint):
        store.upload(remote, payload["remote_uri"])
        if checkpoint:
            relative = Path(checkpoint["path"]).relative_to(remote)
            checkpoint = dict(checkpoint, path=f"{payload['remote_uri']}/{relative}")
        return checkpoint

    return publish


def pod_run(job_path, node, master, seq):
    """On each node, in the pod environment: the pod's inputs, then the Gym's node (nexttransformer_gym.nodes)."""
    payload = json.loads(Path(job_path).read_text())
    emit = Emitter(int(seq))
    node = int(node)
    work, remote = Path(payload["work"]), Path(payload["remote"])
    code = work / "code"
    sys.path[:0] = [str(code / "gym" / "src"), str(code)]
    from nexttransformer_gym import nodes

    try:
        nodes.check_runtime(payload)
        checkpoint = payload.get("checkpoint")
        if checkpoint and node == 0 and checkpoint.get("uri"):
            Store(payload["s5cmd"]).get(checkpoint["uri"], checkpoint["path"])
        nodes.check_checkpoint(payload, node)
        stage_inputs(payload, payload["request"])
        if payload.get("site"):
            site = work / "site.json"
            site.write_text(json.dumps(payload["site"], indent=1))
            payload["site_config"] = str(site)
    except Exception as error:
        return nodes.pre_failure(payload, node, emit, error)
    if payload["request"]["quote"]["operation"] == "official":
        (work / "tmp").mkdir(parents=True, exist_ok=True)
        return official_node(payload, node, master, emit, code, work, remote)
    return nodes.run(
        payload, node, master, emit, code, work, remote, publisher(payload, remote)
    )


# --------------------------------------------------------------------------------- multi-node final run


def topology(payload, node, master):
    return {
        "node": node,
        "nodes": int(payload["nodes"]),
        "gpus_per_node": int(payload["gpus_per_node"]),
        "master": master,
        "port": int(payload["master_port"]),
        "remote": payload["remote"],
        "smoke": payload.get("smoke"),
        "spawn_wait_seconds": float(payload["spawn_wait_seconds"]),
    }


def official_node(payload, node, master, emit, code, work, remote):
    from nexttransformer_gym import nodes

    env = worker_env(payload, code, work, official=True)
    env["KOALA_GYM_TOPOLOGY"] = json.dumps(topology(payload, node, master))
    handoff = remote / "official"
    handoff.mkdir(parents=True, exist_ok=True)
    if node != 0:
        with open(work / "satellite.log", "a") as log:
            code_ = subprocess.run(
                [sys.executable, str(code / "koala_gym.py"), "pod-satellite"],
                env=env,
                cwd=work / "tmp",
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        nodes.keep_log(work / "satellite.log", remote, host_name())
        emit("satellite", node=node, exit=code_)
        # Stay up until node 0 has logged its outcome: a job may end when one of its pods does.
        deadline = time.time() + 3600
        while code_ == 0 and not (handoff / "done").exists() and time.time() < deadline:
            time.sleep(5)
        return code_
    emit(
        "ready",
        nodes=payload["nodes"],
        gpus_per_node=payload["gpus_per_node"],
        smoke=payload.get("smoke"),
    )
    outcome = nodes.drive(
        [sys.executable, str(code / "koala_gym.py"), "pod-official"],
        env,
        payload["request"],
        int(payload["credit"]),
        emit,
        work / "tmp",
        work / "worker.stderr",
        progress_every=300.0,
    )
    if not (handoff / "spawn.pkl").exists():
        (handoff / "abort").write_text("node 0 ended before spawning ranks")
    if outcome["outcome"] == "succeeded" and payload.get("smoke"):
        # get_final_run reads dev_bpb from every non-official result; the bounded smoke never measures the dev
        # split, so it reports none rather than a value from a different split.
        outcome["result"] = dict(
            outcome["result"],
            official_compatible=False,
            accounting="koala_official_smoke",
            smoke=payload["smoke"],
            dev_bpb=None,
        )
    nodes.explain(outcome, work / "worker.stderr")
    nodes.keep_log(work / "worker.stderr", remote, host_name())
    code_ = nodes.report(emit, payload, outcome, remote, publisher(payload, remote))
    (handoff / "done").write_text(outcome["outcome"])
    return code_


def smoke_invoke(official, smoke):
    """Bound the trusted trainer for a test of the final path; audit and export are skipped."""
    real = official.invoke

    def invoke(module, arguments):
        if module == "nexttransformer.train":
            arguments = ["--research" if a == "--formal" else a for a in arguments]
            arguments += [
                "--smoke-steps",
                str(int(smoke["steps"])),
                "--smoke-val-tokens",
                str(int(smoke.get("val_tokens", 0))),
                "--smoke-accumulation",
                str(int(smoke.get("accumulation", 1))),
            ]
            return real(module, arguments)
        if module == "nexttransformer.audit_run":
            out = Path(arguments[arguments.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"status": "koala_smoke_not_audited"}))
            return None
        if module == "nexttransformer.export_run":
            name = arguments[arguments.index("--file") + 1]
            target = Path.cwd() / "results" / "runs" / f"{name}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "status": "koala_smoke_not_a_result",
                        "run": arguments[0],
                        "methodId": arguments[arguments.index("--method") + 1],
                    }
                )
            )
            return None
        return real(module, arguments)

    official.invoke = invoke


def install_official(topo, driver):
    import torch
    from nexttransformer_gym import official

    world = topo["nodes"] * topo["gpus_per_node"]
    if topo.get("smoke"):
        official.official_world = lambda profile: world
        smoke_invoke(official, topo["smoke"])
    if not driver:
        return
    import pickle
    import torch.multiprocessing as multiprocessing

    real_spawn = multiprocessing.spawn
    # official.run compares the visible GPUs with the whole world before it spawns ranks.
    torch.cuda.device_count = lambda: world

    def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
        if nprocs != world:
            raise RuntimeError(
                f"official pipeline asked for {nprocs} ranks; the Koala topology has {world}"
            )
        handoff = Path(topo["remote"]) / "official"
        tmp = handoff / ".spawn.pkl"
        tmp.write_bytes(pickle.dumps({"fn": fn, "args": args}))
        tmp.replace(handoff / "spawn.pkl")
        real_spawn(
            rank_entry, args=(topo, fn, args), nprocs=topo["gpus_per_node"], join=True
        )
        output = Path(args[2])
        remote_ranks = range(topo["gpus_per_node"], world)
        deadline = time.time() + 1800
        while True:
            failed = [
                k for k in remote_ranks if (output / f"failure-{k}.json").exists()
            ]
            missing = [
                k
                for k in remote_ranks
                if not (output / f"rank-{k}.json").exists()
                and not (output / f"failure-{k}.json").exists()
            ]
            if failed:
                raise multiprocessing.ProcessRaisedException(
                    f"remote ranks {failed} failed", failed[0], 0
                )
            if not missing:
                return None
            if time.time() > deadline:
                raise RuntimeError(f"remote ranks {missing} reported no outcome")
            time.sleep(5)

    multiprocessing.spawn = spawn


def rank_entry(local_rank, topo, fn, args):
    """One trusted rank on this node, with its global rank and a TCP meeting at node 0."""
    import torch.distributed as dist

    install_official(topo, driver=False)
    rank = topo["node"] * topo["gpus_per_node"] + local_rank
    update = os.environ.update

    def node_local_update(*values, **named):
        update(*values, **named)
        if "LOCAL_RANK" in os.environ:
            os.environ["LOCAL_RANK"] = str(local_rank)

    os.environ.update = node_local_update
    initialize = dist.init_process_group

    def tcp_group(*values, **named):
        named["init_method"] = f"tcp://{topo['master']}:{topo['port']}"
        return initialize(*values, **named)

    dist.init_process_group = tcp_group
    fn(rank, *args)


def pod_official():
    """The Gym worker on node 0 of a final run, with its rank spawn split across nodes."""
    install_official(json.loads(os.environ["KOALA_GYM_TOPOLOGY"]), driver=True)
    from nexttransformer_gym.worker import main

    main()


def pod_satellite():
    """Ranks of nodes 1..N-1: wait for node 0's spawn, then run this node's share."""
    import pickle
    import torch.multiprocessing as multiprocessing

    topo = json.loads(os.environ["KOALA_GYM_TOPOLOGY"])
    handoff = Path(topo["remote"]) / "official"
    deadline = time.time() + topo["spawn_wait_seconds"]
    while not (handoff / "spawn.pkl").exists():
        if (handoff / "abort").exists():
            print("node 0 ended before spawning ranks", flush=True)
            return 1
        if time.time() > deadline:
            print("node 0 did not spawn ranks in time", flush=True)
            return 1
        time.sleep(5)
    spec = pickle.loads((handoff / "spawn.pkl").read_bytes())
    multiprocessing.spawn(
        rank_entry,
        args=(topo, spec["fn"], spec["args"]),
        nprocs=topo["gpus_per_node"],
        join=True,
    )
    return 0


# Runs as the worker would, under each backend, and reports what it could reach. Only names of
# credential-like variables are reported, never values.
PROBE_SCRIPT = r"""
import errno, glob, json, os, socket, sys, time
spec = json.loads(sys.argv[1])

def attempt(fn):
    try:
        value = fn()
        return {"ok": True} if value is None else {"ok": True, "value": value}
    except OSError as error:
        return {"ok": False, "errno": errno.errorcode.get(error.errno, str(error.errno))}
    except Exception as error:
        return {"ok": False, "error": repr(error)[:300]}

def read(path):
    with open(path, "rb") as handle:
        return len(handle.read(1024))

def write(path):
    with open(path, "x") as handle:
        handle.write("x")
    os.unlink(path)

def connect(host, port):
    socket.create_connection((host, port), timeout=5).close()

def unix_socket():
    sock = socket.socket(socket.AF_UNIX)
    sock.bind("\0ntgym-probe-%d" % os.getpid())
    sock.close()

def metadata(path):
    # The cloud's instance metadata service. Only the HTTP status line is kept, never a body.
    sock = socket.create_connection(("100.100.100.200", 80), timeout=3)
    sock.sendall(("GET %s HTTP/1.0\r\nHost: 100.100.100.200\r\n\r\n" % path).encode())
    status = sock.recv(64).split(b"\r\n")[0].decode(errors="replace")
    sock.close()
    return status

out = {"uid": os.getuid(), "gid": os.getgid()}
out["read_allowed"] = {p: attempt(lambda p=p: read(p)) for p in spec["allowed"]}
out["read_forbidden"] = {p: attempt(lambda p=p: read(p)) for p in spec["forbidden"]}
out["list_forbidden_dirs"] = {d: attempt(lambda d=d: len(os.listdir(d)))
                              for d in sorted({os.path.dirname(p) for p in spec["forbidden"]})}
# A reserved directory counts as reached if it can be listed, a reserved file if it can be read.
out["read_reserved"] = {p: attempt(lambda p=p: len(os.listdir(p)) if os.path.isdir(p) else read(p))
                        for p in spec.get("reserved", [])}
out["proc1_environ"] = attempt(lambda: read("/proc/1/environ"))
out["credential_like_env"] = sorted(k for k in os.environ
                                    if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "OSS", "AWS")))
out["write_workspace"] = attempt(lambda: write(os.path.join(spec["workspace"], "inside-%d" % os.getpid())))
out["write_outside"] = {p: attempt(lambda p=p: write(p)) for p in spec["outside"]}
out["dns"] = {h: attempt(lambda h=h: socket.getaddrinfo(h, None)[0][4][0]) for h, _ in spec["network"]}
out["connect"] = {"%s:%d" % (h, p): attempt(lambda h=h, p=p: connect(h, p)) for h, p in spec["network"]}
out["unix_socket"] = attempt(unix_socket)
out["metadata"] = {p: attempt(lambda p=p: metadata(p))
                   for p in ("/latest/meta-data/", "/latest/meta-data/ram/security-credentials/")}
out["visible_pids"] = attempt(lambda: len([d for d in os.listdir("/proc") if d.isdigit()]))
out["signal_pid1"] = attempt(lambda: os.kill(1, 0))
out["chmod_code_file"] = attempt(lambda: os.chmod(spec["code_file"], 0o777))
out["nvidia_nodes"] = len(glob.glob("/dev/nvidia[0-9]*"))
if spec["cuda"]:
    def cuda():
        import torch
        x = torch.randn(256, 256, device="cuda")
        return {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                "sum": float((x @ x).sum().item() != 0)}
    out["cuda"] = attempt(cuda)
beat, stop = spec["daemon_beat"], spec["daemon_stop"]
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        # Detach from the launcher's pipes, as a daemon would; otherwise the launcher waits for it.
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        for _ in range(600):
            if os.path.exists(stop):
                break
            try:
                with open(beat, "w") as handle:
                    handle.write(str(time.time()))
            except OSError:
                pass
            time.sleep(1)
    os._exit(0)
os.wait()
time.sleep(2)
print("PROBE_RESULT " + json.dumps(out, sort_keys=True), flush=True)
"""


def confinement(payload, python, code, work, backend_name, device):
    """Run PROBE_SCRIPT as the worker identity under one backend; report what it reached."""
    from nexttransformer_gym import nodes, remote

    checks = payload["checks"]
    data = checks["data"]
    workspace = Path(payload["remote"]) / f"{device}-{backend_name}"
    request = {
        "mode": "execute",
        "profile": {"device": device, "data": data},
        "checkpoint": None,
        "output": str(workspace / "checkpoint.pt"),
    }
    identity = remote.worker_identity({})
    open_dir, plain = work / "world-writable", work / "world-readable.txt"
    open_dir.mkdir(exist_ok=True)
    os.chmod(open_dir, 0o1777)
    plain.write_text("outside the allowed paths\n")
    os.chmod(plain, 0o644)
    for path in (workspace, work / "home"):
        nodes.grant(path, identity) if identity else path.mkdir(
            parents=True, exist_ok=True
        )
    if identity:
        nodes.reach(
            workspace,
            payload.get("remote_root") or Path(payload["remote"]).parent.parent,
        )
        nodes.reach(work / "home", work)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(code / "gym" / "src"), str(code)]),
        "HOME": str(work / "home"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    if device == "cuda":
        card = nodes.local_gpus(1)[0]
        request["gpu"] = card
        env.update(CUDA_VISIBLE_DEVICES=card["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    beat, stop = workspace / "daemon.beat", workspace / "daemon.stop"
    spec = {
        "allowed": [*data["train_paths"], data["dev_path"]],
        "workspace": str(workspace),
        "forbidden": [*checks["forbidden"], str(plain)],
        "reserved": list(checks.get("reserved", [])),
        "outside": [
            str(open_dir / "x"),
            str(work / "x"),
            "/tmp/ntgym-probe-x",
            str(Path(payload["remote"]) / "x"),
        ],
        "network": checks.get("network", []),
        "code_file": str(code / "koala_gym.py"),
        "cuda": device == "cuda",
        "daemon_beat": str(beat),
        "daemon_stop": str(stop),
    }
    worker = [python, "-c", PROBE_SCRIPT, json.dumps(spec)]
    if backend_name == "none-netdeny":
        argv = remote.worker_command(python, code, request, "none", "deny", worker)
    else:
        argv = remote.worker_command(
            python, code, request, backend_name, "allow", worker
        )
    report_ = {}
    try:
        run = subprocess.run(
            argv,
            env=env,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=600,
            **identity,
        )
        lines = [
            line for line in run.stdout.splitlines() if line.startswith("PROBE_RESULT ")
        ]
        report_ = json.loads(lines[-1][len("PROBE_RESULT ") :]) if lines else {}
        report_["exit"] = run.returncode
        if not lines:
            report_["stderr"] = run.stderr[-1500:]
    except subprocess.TimeoutExpired:
        report_["error"] = "timeout"

    # Did the detached process outlive the worker, and the cleanup a pod runs next? Its heartbeat keeps changing
    # while it lives.
    def alive():
        first = beat.stat().st_mtime if beat.exists() else None
        time.sleep(4)
        return bool(first and beat.exists() and beat.stat().st_mtime > first)

    report_["reserved_reached"] = reserved_reached(checks, report_)
    report_["daemon_beat_written"] = beat.exists()
    report_["daemon_outlived_worker"] = alive()
    nodes.stop_identity(identity)
    report_["daemon_survived_cleanup"] = alive()
    stop.write_text("stop")
    return report_


def reserved_reached(checks, report_):
    """Reserved paths the worker reached; every one counts as reached when the probe left no answer."""
    answers = report_.get("read_reserved") or {}
    return sorted(
        p
        for p in checks.get("reserved", [])
        if (answers.get(p) or {"ok": True}).get("ok")
    )


def pod_probe_checks(job_path):
    """In the pod environment, as root: the backend's prerequisites, then the confinement checks."""
    payload = json.loads(Path(job_path).read_text())
    work = Path(payload["work"])
    code = work / "code"
    sys.path[:0] = [str(code / "gym" / "src"), str(code)]
    from nexttransformer_gym import remote

    out = {
        "kernel": os.uname().release,
        "uid": os.geteuid(),
        "landlock_abi": remote.landlock_abi(),
    }
    identity = remote.worker_identity({})
    for flags in (
        ["--user"],
        ["--user", "--net", "--ipc", "--uts"],
        ["--user", "--net", "--ipc", "--uts", "--pid", "--kill-child"],
    ):
        run = subprocess.run(
            ["unshare", *flags, "true"], capture_output=True, text=True, **identity
        )
        out["unshare " + " ".join(flags)] = (
            "ok" if run.returncode == 0 else run.stderr.strip()[-200:]
        )
    backend_name = payload["isolation"]
    try:
        remote.preflight(backend_name)
        out["isolation_ok"] = True
    except Exception as error:
        out["isolation_ok"] = False
        out["isolation_detail"] = repr(error)[-500:]
    if payload.get("checks"):
        devices = ["cpu"] + (["cuda"] if shutil.which("nvidia-smi") else [])
        backends = (
            [backend_name] if backend_name != "none" and out["isolation_ok"] else []
        ) + ["none", "none-netdeny"]
        out["confinement"] = {
            f"{device}/{name}": confinement(
                payload, sys.executable, code, work, name, device
            )
            for device in devices
            for name in backends
        }
        out["reserved_ok"] = not any(
            c.get("reserved_reached") for c in out["confinement"].values()
        )
    print("PROBE_CHECKS " + json.dumps(out, sort_keys=True), flush=True)
    return 0


def pod_probe(job_path):
    payload = json.loads(Path(job_path).read_text())
    emit = Emitter()
    work = Path(payload["work"])
    work.mkdir(parents=True, exist_ok=True)
    code = work / "code"
    copy_code(Path(job_path).resolve().parent, code)
    sys.path[:0] = [str(code / "gym" / "src"), str(code)]
    report_ = {"host": host_name()}
    try:
        python = build_env(payload, code, work)
        report_["runtime"] = runtime_of(python)
        report_["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.strip()
        run = subprocess.run(
            [
                python,
                str(code / "koala_gym.py"),
                "pod-probe-checks",
                str(code / "job.json"),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        lines = [
            line for line in run.stdout.splitlines() if line.startswith("PROBE_CHECKS ")
        ]
        if lines:
            checks = json.loads(lines[-1][len("PROBE_CHECKS ") :])
            if payload.get("remote"):
                Path(payload["remote"]).mkdir(parents=True, exist_ok=True)
                (Path(payload["remote"]) / "probe.json").write_text(
                    json.dumps(checks, indent=1, sort_keys=True)
                )
            report_.update(checks)
        else:
            report_["error"] = (run.stderr or run.stdout)[-1500:]
    except (PodError, OSError, subprocess.SubprocessError, ValueError) as error:
        report_["error"] = str(error)[-1500:]
    confinement_ = report_.pop("confinement", None)
    if confinement_ is not None:
        report_["confinement_chunks"] = emit.blob("confinement", confinement_)
    emit("probe", **report_)
    return 0


# ----------------------------------------------------------------------------------------------- main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in (
        "relay",
        "pod",
        "pod-run",
        "pod-official",
        "pod-satellite",
        "pod-probe",
        "pod-probe-checks",
    ):
        command, rest = argv[0], argv[1:]
        if command == "pod-probe-checks":
            return pod_probe_checks(rest[0])
        if command == "relay":
            return relay(rest[0])
        if command == "pod":
            return pod(rest[0])
        if command == "pod-run":
            return pod_run(*rest)
        if command == "pod-official":
            return pod_official()
        if command == "pod-satellite":
            return pod_satellite()
        return pod_probe(rest[0])
    parser = argparse.ArgumentParser(
        description="Run the Gym's numerical jobs on Koala."
    )
    parser.add_argument(
        "--koala", required=True, type=Path, help="koala_gym.v1 configuration JSON"
    )
    parser.add_argument("command", choices=("probe", "jobs", "trial", "cli"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "probe":
        return probe(args.koala)
    if args.command == "jobs":
        return jobs(args.koala, args.arguments)
    return controller(args.koala, args.command, args.arguments)


if __name__ == "__main__":
    sys.exit(main() or 0)
