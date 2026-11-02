"""Shared helpers for the execution-layer tests (synthetic CPU smoke runs)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "nextt_gym" / "templates"
DEBUG = json.loads((TEMPLATES / "config.json").read_text())["debug"]["model"]
COST = 1_000_003                      # per-update stage cost used by the smoke specs
FORWARD = 0.5                         # forward FLOPs per token used by the smoke specs


def env(tmp):
    return dict(os.environ, NANOCHAT_RUNTIME=str(Path(tmp) / "runtime"), CUDA_VISIBLE_DEVICES="",
                NANOCHAT_DTYPE="float32", TORCH_COMPILE_DISABLE="1", OMP_NUM_THREADS="2",
                PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def submission(tmp):
    d = Path(tmp) / "submission"; d.mkdir(parents=True, exist_ok=True)
    for name in ("model.py", "config.json"): shutil.copy(TEMPLATES / name, d / name)
    return d


def spec(tmp, sub_dir, *, steps=6, reserved=None, run_id="smoke"):
    from nextt_gym.contract import load_submission
    from nextt_gym.runspec import make_run_spec, write_run_spec
    sub = load_submission(sub_dir)
    report = dict(stages=[dict(name="joint", total=COST)], forward_flops_per_token=FORWARD)
    s = make_run_spec(sub, report, "S", run_id, 1, reserved or 10 ** 12, synthetic=True)
    s["steps"] = steps
    return write_run_spec(s, Path(tmp) / f"{run_id}.json")


def train(tmp, sub_dir, spec_path, output, *extra):
    command = [sys.executable, "-m", "nextt_gym.trainer", "--submission", str(sub_dir), "--run-spec", str(spec_path),
               "--output", str(output), "--synthetic", "--model-overrides", json.dumps(DEBUG),
               "--log-every", "1", "--rolling-every", "2", *extra]
    result = subprocess.run(command, cwd=tmp, env=env(tmp), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return json.loads((Path(output) / "run_status.json").read_text())
