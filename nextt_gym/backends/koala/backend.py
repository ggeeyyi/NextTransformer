"""KoalaBackend: the common Backend interface on Koala clusters (e.g. H20).

A Koala job has a wall-clock limit, so a run is a chain of segments (chain.py). Each segment unpacks
the gym bundle on the pod-local disk and runs run.sh. Runs write
to `workdir` on persistent storage (CPFS) that the gym host need not mount; result files come back
through the `reader` command (backends/reader.py), by default `koala exec` on a CPU debug pod. Keep
NEXTT_GYM_HOME outside any git work tree (see Koala.publish).

Config keys:
  cluster = "h20"                 koala --cluster value
  gpus_per_node = 8
  workdir                         persistent run directory visible to the pods
  reader                          command template running {script} next to `workdir` (reader.py);
                                  default: Koala.exec_template on the user's CPU debug pod
  reader_target                   optional pod name or unique substring for the default reader
  upload = "<command> {dir}"      optional: publish directory {dir} as a code bundle and print its URI;
                                  default: Koala.publish
  code_project = "nextt-gym"      `koala code publish --project`
  code_dir = "/data/work/run_codes"         where the job mounts the code bundle
  boot_dir = "/local-ssd/nextt-gym-boot"    pod-local copy of the repository that run.sh uses
  unpack_dir = "/local-ssd/nextt-gym-code"  pod-local `remote --code-dir` (the verified bundle the trainer runs)
  env_setup, python               optional: replace run.sh by `cd boot_dir && {env_setup} && {python} -m nextt_gym.remote`
  site_config                     upstream site JSON path inside the pod
  trainer_args = []               extra trainer options, e.g. ["--micro-batch", "16"]
  max_wall_hours_per_segment = 45.5
  max_segments = 8                restarted segments count too
  max_restarts = 0                restarts of a failed segment from the latest checkpoint (backends/base.py)
  submit_flags = ["-m", "normal", "--ssd", "64", "-q", "-y"]
  fetch_every_s = 60              minimum seconds between two reads of a running job's files
"""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from .. import reader
from ..base import RESULT_FILES, Backend, summarize
from .chain import FINAL, driver_running, read_chain, start_driver, write_json
from .cli import Koala

RUN_SH = "nextt_gym/backends/koala/run.sh"
# Remote files relative to the run directory; `remote` writes remote_status.json, the trainer the rest under run/.
FETCH_PATTERNS = [f"run/{n}" for n in RESULT_FILES if n != "remote_status.json"] + [
    "remote_status.json", "run/val_*.json", "run/core_final.json"]


class KoalaBackend(Backend):
    kind = "koala"
    supports_restarts = True

    def __init__(self, backend_id, config):
        super().__init__(backend_id, config)
        self.koala = Koala(config.get("cluster", "h20"))
        self.cluster = self.koala.cluster
        # A Koala job has a wall-clock limit, so every run has segment limits; describe() reports them.
        self.max_wall_hours_per_segment = float(config.get("max_wall_hours_per_segment", 45.5))
        self.max_segments = int(config.get("max_segments", 8))
        self.fetch_every_s = float(config.get("fetch_every_s", 60))
        self.reader = config.get("reader") or self.koala.exec_template(config.get("reader_target"))

    def describe(self):
        return dict(super().describe(), max_segments=self.max_segments)

    def availability(self):
        return "available" if self.koala.available() else "offline"

    def publish(self, staging):
        """The code bundle URI for the staging directory (bundle archive and run spec)."""
        template = self.config.get("upload")
        if not template: return self.koala.publish(staging, self.config.get("code_project", "nextt-gym"))
        out = subprocess.run(template.format(dir=shlex.quote(str(staging))), shell=True, capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()
        if not out: raise RuntimeError("upload command printed no bundle URI")
        return out[-1]

    def remote_dir(self, spec):
        """`workdir/runs/<run_id>-<spec hash>`: a fresh gym home that reuses run ids never resumes an old run."""
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.config['workdir'].rstrip('/')}/runs/{spec['run_id']}-{digest}"

    def segment_command(self, out, bundle_name, sha256, spec_name, world_size, synthetic, model_overrides):
        """The `koala submit -c` command of every segment of the run writing to `out`."""
        q = shlex.quote
        code = self.config.get("code_dir", "/data/work/run_codes").rstrip("/")
        boot = self.config.get("boot_dir", "/local-ssd/nextt-gym-boot")
        args = ["run", "--bundle", f"{code}/{bundle_name}", "--sha256", sha256, "--run-spec", f"{out}/{spec_name}",
                "--output", out, "--nproc", str(min(world_size, self.gpus_per_node)), "--resume", "auto",
                "--max-wall-hours", str(self.max_wall_hours_per_segment),
                "--code-dir", self.config.get("unpack_dir", "/local-ssd/nextt-gym-code")]
        if self.config.get("site_config"): args += ["--site-config", self.config["site_config"]]
        if synthetic: args += ["--synthetic"]
        if model_overrides: args += ["--model-overrides", json.dumps(model_overrides)]
        if self.config.get("trainer_args"): args += ["--", *self.config["trainer_args"]]
        remote = " ".join(q(a) for a in args)
        if self.config.get("env_setup"):
            entry = (f"cd {q(boot)} && {self.config['env_setup']} && "
                     f"{self.config.get('python', 'uv run --no-sync python')} -m nextt_gym.remote {remote}")
        else:
            entry = f"bash {q(boot)}/{RUN_SH} {remote}"
        # A stale run_status.json from an earlier segment must not describe this one.
        return (f"set -e; mkdir -p {q(out)} && cp {q(code)}/{q(spec_name)} {q(out)}/ && "
                f"rm -f {q(out)}/run/run_status.json && rm -rf {q(boot)} && mkdir -p {q(boot)} && "
                f"tar xzf {q(code)}/{q(bundle_name)} -C {q(boot)} && {entry}")

    def submit(self, run_dir_local, bundle, sha256, run_spec_path, *, world_size, synthetic=False,
               model_overrides=None):
        if not synthetic: self.check_world_size(world_size)
        if world_size > self.gpus_per_node:
            raise NotImplementedError("multi-node Koala segments need the run.sh rendezvous; use 8 ranks for now")
        run_dir = Path(run_dir_local).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
        spec = json.loads(Path(run_spec_path).read_text())   # the local directory is <run>/backend, not the run id
        out = self.remote_dir(spec)
        # The code bundle directory carries the bundle archive and the run spec.
        staging = run_dir / "codebundle"; staging.mkdir(exist_ok=True)
        for path in (bundle, run_spec_path): shutil.copy2(path, staging / Path(path).name)
        uri = self.publish(staging)
        command = self.segment_command(out, Path(bundle).name, sha256, Path(run_spec_path).name, world_size,
                                       synthetic, model_overrides)
        plan = dict(cluster=self.cluster, bundle_uri=uri, code_dir=self.config.get("code_dir", "/data/work/run_codes"),
                    command=command, remote_dir=out, reader=self.reader, nodes=1,
                    gpus=min(world_size, self.gpus_per_node),
                    flags=self.config.get("submit_flags", ["-m", "normal", "--ssd", "64", "-q", "-y"]),
                    max_segments=self.max_segments, max_restarts=self.max_restarts,
                    job_prefix="nextt-gym-" + re.sub(r"[^a-z0-9-]+", "-", spec["run_id"].lower()))
        write_json(run_dir / "koala_plan.json", plan)
        start_driver(run_dir)
        return str(run_dir)

    def status(self, handle):
        run_dir = Path(handle)
        chain = read_chain(run_dir)
        state = chain.get("state", "queued")
        if state not in FINAL and (run_dir / "koala_plan.json").exists() and not driver_running(run_dir):
            start_driver(run_dir)   # re-attaches to the unfinished job
        if state == "running" and (chain.get("segments") or [{}])[-1].get("job_state") in ("Queuing", "Updating"):
            state = "queued"        # submitted, but Koala has not started the pod yet
        # The keys of the other backends (base.summarize), from the trainer's files at the last segment's end.
        out = dict(state=state, **summarize(chain.get("budget"), chain.get("run_status")))
        return dict(out, phase=chain.get("phase"), step=chain.get("step"), consumed_flops=chain.get("consumed_flops"),
                    segments=chain.get("segments", []), restarts=chain.get("restarts", []))

    def logs(self, handle, tail=200):
        job = (read_chain(handle).get("segments") or [{}])[-1].get("job")
        return self.koala.log(job, tail) if job else []

    def cancel(self, handle):
        """Mark the run cancelled and stop its job; the driver records the final state."""
        run_dir = Path(handle); (run_dir / "cancelled").write_text("1")
        for segment in read_chain(run_dir).get("segments", []):
            if segment.get("job") and not segment.get("finished"): self.koala.stop(segment["job"])

    def fetch(self, handle, dest):
        """Read the run's result files through the reader; a running job at most every `fetch_every_s`."""
        run_dir, dest = Path(handle), Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        chain, stamp = read_chain(run_dir), run_dir / "fetched_at"
        if not chain.get("segments"): return dest
        if chain.get("state") not in FINAL and stamp.exists() and time.time() - float(stamp.read_text()) < self.fetch_every_s:
            return dest
        plan = json.loads((run_dir / "koala_plan.json").read_text())
        reader.fetch(plan["reader"], plan["remote_dir"], FETCH_PATTERNS, dest)
        stamp.write_text(str(time.time()))
        return dest
