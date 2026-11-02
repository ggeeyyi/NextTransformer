"""The one command every cluster runs:

    python -m nextt_gym.remote run --bundle B --sha256 H --run-spec S --site-config SITE --output OUT \
        [--nproc 8] [--nnodes 1 --node-rank 0 --master-addr A --master-port P] [--resume auto] [--synthetic]

Verifies and unpacks the bundle into OUT/code (reused across segments; `--code-dir`
moves it, for example to a node-local disk), runs a pre-flight check, then launches
torchrun of `nextt_gym.trainer` from the unpacked code. Only the standard library is
needed before the launch, so the caller may run it with any Python that can import
torch for the pre-flight.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


def atomic_json(path, payload):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(payload, indent=1) + "\n"); tmp.replace(path)


def pinned_torch(code):
    match = re.search(r'"torch==([^"]+)"', (Path(code) / "pyproject.toml").read_text())
    return match.group(1) if match else None


def preflight(code, nproc, synthetic):
    """GPU count/type and the torch pin; resources are verified again by the trainer."""
    import torch
    report = dict(host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), gpus=torch.cuda.device_count())
    if report["gpus"]: report["gpu_names"] = sorted({torch.cuda.get_device_name(i) for i in range(report["gpus"])})
    problems = []
    pin = pinned_torch(code)
    if pin and torch.__version__.split("+")[0] != pin: problems.append(f"torch {torch.__version__} != pinned {pin}")
    if not synthetic:
        if not report["cuda_available"]: problems.append("CUDA is not available")
        elif report["gpus"] < nproc: problems.append(f"{report['gpus']} GPUs visible, {nproc} required")
    report["problems"] = problems
    return report


def run(args):
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    status = out / "remote_status.json"
    code = Path(args.code_dir).resolve() if args.code_dir else out / "code"
    marker = code / ".bundle_sha256"
    if not (marker.exists() and marker.read_text().strip() == args.sha256):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from nextt_gym.bundle import unpack_bundle
        if code.exists():
            code.rename(code.parent / f"{code.name}.replaced-{int(time.time())}")   # same filesystem as code
        unpack_bundle(args.bundle, args.sha256, code)
    report = preflight(code, args.nproc, args.synthetic)
    if report["problems"]:
        atomic_json(status, dict(phase="preflight_failed", preflight=report, at=time.time()))
        return 2
    launch = [sys.executable, "-m", "torch.distributed.run"]
    if args.nnodes > 1:
        launch += [f"--nnodes={args.nnodes}", f"--node-rank={args.node_rank}", f"--master-addr={args.master_addr}",
                   f"--master-port={args.master_port}"]
    else:
        launch += ["--standalone"]
    launch += [f"--nproc-per-node={args.nproc}", "-m", "nextt_gym.trainer",
               "--submission", str(code / "submission"), "--run-spec", str(Path(args.run_spec).resolve()),
               "--output", str(out / "run"), "--resume", args.resume]
    if args.site_config: launch += ["--site-config", str(Path(args.site_config).resolve())]
    if args.max_wall_hours: launch += ["--max-wall-hours", str(args.max_wall_hours)]
    if args.synthetic: launch += ["--synthetic"]
    if args.model_overrides: launch += ["--model-overrides", args.model_overrides]
    launch += args.trainer_args
    env = dict(os.environ, PYTHONPATH=str(code) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1", FLA_DISABLE_BACKEND_DISPATCH="1")
    atomic_json(status, dict(phase="launched", preflight=report, command=launch, pid=os.getpid(), at=time.time()))
    code_rc = subprocess.call(launch, cwd=code, env=env)
    run_status = out / "run" / "run_status.json"
    final = json.loads(run_status.read_text()) if run_status.exists() else None
    atomic_json(status, dict(phase="exited", returncode=code_rc, preflight=report, run_status=final, at=time.time()))
    if final: print(f"NEXTT_RUN_STATUS {final['phase']} {final['step']} {final['consumed_flops']}", flush=True)
    return code_rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True); r.add_argument("--sha256", required=True)
    r.add_argument("--run-spec", required=True); r.add_argument("--site-config")
    r.add_argument("--output", required=True); r.add_argument("--nproc", type=int, default=8)
    r.add_argument("--nnodes", type=int, default=1); r.add_argument("--node-rank", type=int, default=0)
    r.add_argument("--master-addr", default="127.0.0.1"); r.add_argument("--master-port", type=int, default=29500)
    r.add_argument("--resume", default="auto"); r.add_argument("--max-wall-hours", type=float, default=0.)
    r.add_argument("--code-dir", help="where to unpack the bundle (default OUTPUT/code), e.g. a node-local disk")
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--model-overrides", help="smoke only, passed to the trainer")
    r.add_argument("trainer_args", nargs=argparse.REMAINDER, help="extra trainer options after --")
    args = parser.parse_args(argv)
    if args.trainer_args[:1] == ["--"]: args.trainer_args = args.trainer_args[1:]
    sys.exit(run(args))


if __name__ == "__main__":
    main()
