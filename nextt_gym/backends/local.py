"""This machine: the remote command in a detached subprocess. Used for single-node runs and tests."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .base import RESULT_FILES, Backend, read_json, state_from, summarize


def _alive(pid):
    try: os.kill(pid, 0)
    except (ProcessLookupError, PermissionError): return False
    try:  # a finished child that was never reaped is a zombie, not alive
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError: return True


class LocalBackend(Backend):
    kind = "local"

    def submit(self, run_dir_local, bundle, sha256, run_spec_path, *, world_size, synthetic=False,
               model_overrides=None):
        if not synthetic: self.check_world_size(world_size)
        run_dir = Path(run_dir_local).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
        command = [self.config.get("python", sys.executable), "-m", "nextt_gym.remote", "run",
                   "--bundle", str(Path(bundle).resolve()), "--sha256", sha256,
                   "--run-spec", str(Path(run_spec_path).resolve()), "--output", str(run_dir),
                   "--nproc", str(world_size)]
        if self.config.get("site_config"): command += ["--site-config", self.config["site_config"]]
        if synthetic: command += ["--synthetic"]
        if model_overrides: command += ["--model-overrides", json.dumps(model_overrides)]
        command += ["--", *self.config.get("trainer_args", [])] if self.config.get("trainer_args") else []
        env = dict(os.environ, **{k: str(v) for k, v in self.config.get("env", {}).items()})
        # The remote entry must import from the unpacked bundle, not this checkout.
        bootstrap = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(bootstrap) + os.pathsep + env.get("PYTHONPATH", "")
        log = open(run_dir / "remote.log", "ab")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=run_dir,
                                   start_new_session=True)
        (run_dir / "local_pid").write_text(str(process.pid))
        return str(run_dir)

    def _files(self, handle):
        run_dir = Path(handle)
        pid = int((run_dir / "local_pid").read_text()) if (run_dir / "local_pid").exists() else None
        run_status = read_json((run_dir / "run" / "run_status.json").read_text()) if (run_dir / "run" / "run_status.json").exists() else None
        remote = read_json((run_dir / "remote_status.json").read_text()) if (run_dir / "remote_status.json").exists() else None
        budget = read_json((run_dir / "run" / "budget.json").read_text()) if (run_dir / "run" / "budget.json").exists() else None
        cancelled = (run_dir / "cancelled").exists()
        return pid, run_status, remote, budget, cancelled

    def status(self, handle):
        pid, run_status, remote, budget, cancelled = self._files(handle)
        alive = pid is not None and _alive(pid)
        state = "cancelled" if cancelled and not alive else state_from(run_status, remote, alive)
        if not alive and remote and remote.get("phase") == "preflight_failed": state = "failed"
        return dict(state=state, **summarize(budget, run_status))

    def logs(self, handle, tail=200):
        path = Path(handle) / "remote.log"
        return path.read_text(errors="replace").splitlines()[-tail:] if path.exists() else []

    def cancel(self, handle):
        pid, *_ = self._files(handle)
        (Path(handle) / "cancelled").write_text("1")
        if pid and _alive(pid):
            try: os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError: pass

    def fetch(self, handle, dest):
        src, dest = Path(handle) / "run", Path(dest); dest.mkdir(parents=True, exist_ok=True)
        for path in [*(src / n for n in RESULT_FILES), Path(handle) / "remote_status.json",
                     *src.glob("val_*.json"), *src.glob("core_final.json")]:
            if path.exists(): shutil.copy2(path, dest / path.name)
        return dest
