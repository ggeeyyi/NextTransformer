"""Plain SSH nodes (e.g. an A800 or H100 box): rsync the bundle, start the remote command under nohup.

Config keys (backends.local.toml):
  hosts = ["a800-01", ...]   SSH aliases; credentials stay in ~/.ssh/config
  workdir                    remote directory for runs
  env_setup                  shell prefix, e.g. "source /persistent/venv/bin/activate"
  python = "python"          remote interpreter (after env_setup)
  site_config                remote upstream site JSON
  check_occupancy = true     refuse to start on GPUs already in use (AGENTS.md)
  busy_memory_mb = 1024      a GPU using more memory than this counts as occupied
Multi-node runs use the first `world_size / gpus_per_node` hosts, rendezvous on hosts[0].
"""
import json
from pathlib import Path
import shlex
import subprocess

from .base import RESULT_FILES, Backend, read_json, state_from, summarize


class SSHBackend(Backend):
    kind = "ssh"

    def __init__(self, backend_id, config):
        super().__init__(backend_id, config)
        self.hosts = list(config["hosts"])
        self.workdir = config["workdir"].rstrip("/")
        self.python = config.get("python", "python")
        self.env_setup = config.get("env_setup", "true")

    def _ssh(self, host, command, check=True, timeout=120):
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True,
                              check=check, timeout=timeout)

    def occupied_gpus(self, host):
        out = self._ssh(host, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits").stdout
        limit = int(self.config.get("busy_memory_mb", 1024))
        return [int(i) for i, mem in (line.split(",") for line in out.strip().splitlines()) if int(mem) > limit]

    def availability(self):
        try:
            if self.config.get("check_occupancy", True) and self.occupied_gpus(self.hosts[0]): return "busy"
            return "available"
        except Exception:
            return "offline"

    def submit(self, run_dir_local, bundle, sha256, run_spec_path, *, world_size, synthetic=False,
               model_overrides=None):
        if not synthetic: self.check_world_size(world_size)
        nodes = max(1, world_size // self.gpus_per_node)
        hosts = self.hosts[:nodes]
        if len(hosts) < nodes: raise ValueError(f"{self.backend_id} has {len(self.hosts)} hosts, {nodes} needed")
        if self.config.get("check_occupancy", True):
            busy = {h: self.occupied_gpus(h) for h in hosts}
            if any(busy.values()): raise RuntimeError(f"GPUs already in use: {busy}")
        run_id = json.loads(Path(run_spec_path).read_text())["run_id"]   # the local directory is <run>/backend
        remote_dir = f"{self.workdir}/runs/{run_id}"
        nproc = min(world_size, self.gpus_per_node)
        for rank, host in enumerate(hosts):
            self._ssh(host, f"mkdir -p {shlex.quote(remote_dir)}")
            subprocess.run(["rsync", "-a", str(bundle), str(run_spec_path), f"{host}:{remote_dir}/"], check=True)
            args = ["run", "--bundle", f"{remote_dir}/{Path(bundle).name}", "--sha256", sha256,
                    "--run-spec", f"{remote_dir}/{Path(run_spec_path).name}", "--output", remote_dir,
                    "--nproc", str(nproc)]
            if nodes > 1:
                args += ["--nnodes", str(nodes), "--node-rank", str(rank), "--master-addr", hosts[0],
                         "--master-port", str(self.config.get("master_port", 29500))]
            if self.config.get("site_config"): args += ["--site-config", self.config["site_config"]]
            if synthetic: args += ["--synthetic"]
            if model_overrides: args += ["--model-overrides", json.dumps(model_overrides)]
            # The bootstrap import of nextt_gym.remote comes from the bundle itself.
            command = (f"cd {shlex.quote(remote_dir)} && mkdir -p boot && tar xzf {shlex.quote(Path(bundle).name)} "
                       f"-C boot nextt_gym && {self.env_setup} && PYTHONPATH=boot setsid nohup {self.python} -m "
                       f"nextt_gym.remote {' '.join(shlex.quote(a) for a in args)} < /dev/null > remote.log 2>&1 & "
                       f"echo $! > pid")
            self._ssh(host, command)
        return json.dumps(dict(hosts=hosts, dir=remote_dir))

    def _read(self, host, path):
        result = self._ssh(host, f"cat {shlex.quote(path)} 2>/dev/null", check=False)
        return read_json(result.stdout)

    def status(self, handle):
        h = json.loads(handle); host, d = h["hosts"][0], h["dir"]
        alive = self._ssh(host, f"kill -0 $(cat {d}/pid) 2>/dev/null && echo yes", check=False).stdout.strip() == "yes"
        run_status, remote, budget = (self._read(host, f"{d}/run/run_status.json"),
                                      self._read(host, f"{d}/remote_status.json"), self._read(host, f"{d}/run/budget.json"))
        cancelled = self._ssh(host, f"test -e {d}/cancelled && echo yes", check=False).stdout.strip() == "yes"
        state = "cancelled" if cancelled and not alive else state_from(run_status, remote, alive)
        return dict(state=state, **summarize(budget, run_status))

    def logs(self, handle, tail=200):
        h = json.loads(handle)
        return self._ssh(h["hosts"][0], f"tail -n {int(tail)} {h['dir']}/remote.log", check=False).stdout.splitlines()

    def cancel(self, handle):
        h = json.loads(handle)
        for host in h["hosts"]:
            # setsid made the remote command a process-group leader: its pid is the group id.
            self._ssh(host, f"touch {h['dir']}/cancelled; kill -TERM -- -$(cat {h['dir']}/pid)", check=False)

    def fetch(self, handle, dest):
        h = json.loads(handle); dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        patterns = [f"{h['dir']}/run/{n}" for n in RESULT_FILES] + [f"{h['dir']}/remote_status.json",
                                                                   f"{h['dir']}/run/val_*.json", f"{h['dir']}/run/core_final.json"]
        subprocess.run(["rsync", "-a", *(f"{h['hosts'][0]}:{p}" for p in patterns), f"{dest}/"], check=False)
        return dest
