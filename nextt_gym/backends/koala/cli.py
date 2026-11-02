"""The Koala platform operations the gym uses, exposed as Koala defines them.

These have no bare-metal counterpart, so they are not hidden behind the common Backend interface:
immutable code bundles (`koala code publish`), queued jobs with platform states, a wall-clock limit
per job, `koala stop`, and debug pods reachable only through `koala exec`. KoalaBackend composes them;
operators and tests may call them directly. Every call names its cluster explicitly.
"""
import json
from pathlib import Path
import shlex
import subprocess

# `koala get --json` states of a job that has not ended, and of one that has (Stopped after `koala stop`).
# The status is blank while Koala admits a queued job (H20, 2026-10-30): that is neither.
ACTIVE_STATES = ("Queuing", "Updating", "Running")
ENDED_STATES = ("Succeeded", "Failed", "Stopped", "Deleting")


class Koala:
    def __init__(self, cluster="h20"):
        self.cluster = cluster

    def run(self, *args, check=False, timeout=None):
        return subprocess.run(["koala", "--cluster", self.cluster, *args], capture_output=True, text=True,
                              check=check, timeout=timeout)

    def available(self):
        try:
            subprocess.run(["koala", "--help"], capture_output=True, check=True, timeout=30)
            return True
        except Exception:
            return False

    def publish(self, directory, project):
        """Publish `directory` as an immutable code bundle and return its URI.

        Previews first and publishes only the previewed SHA-256. The directory must lie outside any git
        work tree: there `koala code publish` packs tracked files only.
        """
        inside = subprocess.run(["git", "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True)
        if inside.stdout.strip() == "true":
            raise RuntimeError(f"{directory} is inside a git work tree; koala code publish would pack tracked "
                               "files only. Move NEXTT_GYM_HOME outside the repository")
        base = ["code", "publish", str(directory), "--project", project, "--json"]
        preview = json.loads(self.run(*base, "--dry-run", check=True).stdout)
        expected = len([p for p in Path(directory).iterdir()])
        if preview.get("files") != expected:
            raise RuntimeError(f"code bundle preview lists {preview.get('files')} files, the directory has {expected}")
        published = json.loads(self.run(*base, "--expect-sha256", preview["bundle_sha256"], "-y", check=True).stdout)
        if not published.get("bundle_uri"): raise RuntimeError(f"koala code publish returned {published}")
        return published["bundle_uri"]

    def submit(self, command, *, bundle_uri, code_dir, job_prefix, nodes=1, gpus=8, flags=()):
        """Submit one job running `command` with the bundle mounted at `code_dir`; returns the job name."""
        result = self.run("submit", *flags, "-n", str(nodes), "-g", str(gpus), "-j", job_prefix,
                          "--code-bundle", f"{bundle_uri}:{code_dir}", "-c", command)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError((result.stderr or result.stdout)[-2000:])
        return result.stdout.strip().splitlines()[-1]

    def state(self, job):
        """The job's platform state, or None when it cannot be read."""
        result = self.run("get", job, "--json")
        if result.returncode: return None
        try: return json.loads(result.stdout).get("status")
        except json.JSONDecodeError: return None

    def wait(self, job, timeout="2h"):
        """Block until the job ends or `timeout` passes; returns koala's exit code (0, 1, 2 or 130)."""
        return self.run("wait", job, "--timeout", timeout).returncode

    def log(self, job, tail=200):
        return self.run("log", job, "-n", str(tail)).stdout.splitlines()

    def stop(self, job):
        """Stop the job and release its resources; checkpoints on persistent storage stay."""
        return self.run("stop", "-y", job).returncode

    def exec_template(self, target=None, cpu=True):
        """A reader template (backends/reader.py) that runs `{script}` on a debug pod through `koala exec`.

        `target` is a pod name or unique substring; without it Koala needs a unique match, which `cpu`
        narrows to CPU debug pods.
        """
        options = " --cpu" if cpu else ""
        target = f" {shlex.quote(target)}" if target else ""
        return f"koala --cluster {shlex.quote(self.cluster)} exec{options} -q{target} -- bash -c {{script}}"
