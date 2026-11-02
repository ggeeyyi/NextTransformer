"""ModelArts (minimal): builds the per-node container command; job submission is not automated.

A ModelArts job runs one independent eight-GPU group per node; a gym
run is one such node. Submitting and polling ModelArts jobs needs
the site's ModelArts SDK/CLI, which is not part of this repository, so those
methods raise NotImplementedError rather than guess an API. Use `command()` to
fill in a job definition by hand, then register the result directory with the
service.
"""
import json
import shlex

from .base import Backend


class ModelArtsBackend(Backend):
    kind = "modelarts"

    def command(self, run_dir, bundle_name, sha256, spec_name, world_size, synthetic=False, model_overrides=None):
        if world_size != self.gpus_per_node:
            raise NotImplementedError("one ModelArts node = one eight-GPU run")
        code = self.config.get("code_dir")
        if not code: raise ValueError("a ModelArts backend needs code_dir, the job's code directory")
        args = ["run", "--bundle", f"{code}/{bundle_name}", "--sha256", sha256, "--run-spec", f"{code}/{spec_name}",
                "--output", run_dir, "--nproc", str(world_size)]
        if self.config.get("site_config"): args += ["--site-config", self.config["site_config"]]
        if synthetic: args += ["--synthetic"]
        if model_overrides: args += ["--model-overrides", json.dumps(model_overrides)]
        setup = self.config.get("env_setup", "true")
        return (f"cd {code} && mkdir -p boot && tar xzf {bundle_name} -C boot nextt_gym && {setup} && "
                f"PYTHONPATH=boot python -m nextt_gym.remote {' '.join(shlex.quote(a) for a in args)}")

    def availability(self):
        return "offline"   # not automated

    def submit(self, *args, **kwargs):
        raise NotImplementedError("ModelArts job submission is site-specific; use ModelArtsBackend.command()")

    def status(self, handle):
        raise NotImplementedError("ModelArts job polling is site-specific")

    def logs(self, handle, tail=200):
        raise NotImplementedError("ModelArts logs are site-specific")

    def cancel(self, handle):
        raise NotImplementedError("ModelArts job cancellation is site-specific")

    def fetch(self, handle, dest):
        raise NotImplementedError("copy results from the job's OBS output directory")
