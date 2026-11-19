"""Running the Gym worker away from the controller, exactly as the controller would run it locally.

An operator executor (see `executor`) ships this package and the platform to another machine and runs one Gym worker
there per job. These functions carry the Gym's own rules, so that machine never re-implements them:

* `runtime(python)`: the numerical runtime record the controller's revision binds (the executor declares it in the
  execution configuration, and the remote side checks that its interpreter still matches);
* `local_devices(count)`: that machine's GPUs, as the Gym's device records, to stand in for the allocation's cards;
* `worker_identity(request)`: the unprivileged identity a root runner uses for every non-official worker;
* `worker_environment(...)`: the worker's explicit environment;
* `preflight(isolation)` and `worker_command(...)`: the confinement around the worker.

`python -m nexttransformer_gym.remote runtime [python]` prints the runtime record.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import GymError, require

# The numerical interpreter's identity: pinned dependency versions, Python, and torch's CUDA build.
RUNTIME_SCRIPT = """
import ast, importlib.metadata as m, json, sys
names = ('torch', 'numpy', 'pyarrow', 'tiktoken', 'rustbpe', 'filelock', 'fsspec', 'sympy')
version = m.distribution('torch').locate_file('torch/version.py').read_text()
cuda = None
for node in ast.walk(ast.parse(version)):
    if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == 'cuda':
        cuda = ast.literal_eval(node.value)
print(json.dumps({'dependencies': {n:m.version(n) for n in names}, 'python':list(sys.version_info[:3]), 'cuda_version':cuda}))
"""


def runtime(python=None, timeout=120):
    """The runtime record of `python` (this interpreter by default)."""
    return json.loads(
        subprocess.check_output(
            [str(python or sys.executable), "-c", RUNTIME_SCRIPT],
            text=True,
            timeout=timeout,
        )
    )


def valid_runtime(value):
    """Whether `value` has the shape of a runtime record."""
    return (
        isinstance(value, dict)
        and set(value) == {"dependencies", "python", "cuda_version"}
        and isinstance(value["dependencies"], dict)
        and all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in value["dependencies"].items()
        )
        and isinstance(value["python"], list)
        and all(type(x) is int for x in value["python"])
        and (value["cuda_version"] is None or isinstance(value["cuda_version"], str))
    )


def local_devices(count):
    """This machine's `count` GPUs in index order, in the Gym's device record format."""
    from . import gpu

    seen = gpu.discover()
    if seen.get("status") == "ok" and len(seen["gpus"]) == count:
        return sorted(seen["gpus"], key=lambda card: card["index"])
    # Busy or partly observable devices still have identities; nvidia-smi's XML report carries them.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=30)
    )
    cards = root.findall("gpu")
    require(
        len(cards) == count,
        "resource_unavailable",
        f"This machine has {len(cards)} GPUs, expected {count}.",
    )

    def record(index, card):
        mib = lambda path: int(card.findtext(path).split()[0]) * 1024**2  # noqa: E731
        uuid = card.findtext("uuid").strip()
        return {
            "id": uuid,
            "uuid": uuid,
            "index": index,
            "device_minor": int(card.findtext("minor_number")),
            "name": card.findtext("product_name").strip(),
            "memory_total_bytes": mib("fb_memory_usage/total"),
            "memory_free_bytes": mib("fb_memory_usage/free"),
            "utilization_percent": 0,
            "process_count": 0,
            "compute_process_count": 0,
            "graphics_process_count": 0,
        }

    return [record(index, card) for index, card in enumerate(cards)]


def landlock_abi():
    """This kernel's Landlock ABI version (0 without Landlock), or the error that prevented reading it."""
    from .landlock import abi_version

    try:
        return abi_version()
    except OSError as error:
        return f"error: {error}"


def worker_identity(request):
    """A root runner (a Docker controller, a cluster pod) runs every non-official worker as nobody; an
    unprivileged runner keeps its own identity. Official final runs execute the operator's admitted snapshot."""
    if os.geteuid() == 0 and request.get("quote", {}).get("operation") != "official":
        return {"user": 65534, "group": 65534, "extra_groups": []}
    return {}


def worker_environment(platform_root, home, profile=None, execution=None, devices=None):
    """The worker's explicit environment: no ambient plugin paths, credentials or proxies.

    A CUDA research profile sees exactly `devices` (the allocation's cards, in order) and runs in `execution`'s
    research precision and compiler setting (`research_dtype`, `research_compile`)."""
    home = Path(home)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).parent.parent), str(platform_root)]
        ),
        "NANOCHAT_DTYPE": "float32",
        "TORCHDYNAMO_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "USER": "gym-worker",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / "cache"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    if profile and str(profile.get("device", "cpu")).startswith("cuda"):
        require(
            bool(devices),
            "resource_unavailable",
            "Research GPU execution requires a scheduler allocation.",
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(card["id"] for card in devices)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["NANOCHAT_DTYPE"] = execution["research_dtype"]
        env["TORCHDYNAMO_DISABLE"] = "0" if execution["research_compile"] else "1"
        env["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    return env


def preflight(isolation):
    """The prerequisites of `isolation` on this machine; fails before any work when they are missing."""
    from .isolation import preflight as local_preflight

    if isolation == "bwrap":
        run = (
            subprocess.run(
                [
                    "bwrap",
                    "--unshare-all",
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--dev",
                    "/dev",
                    "--proc",
                    "/proc",
                    "true",
                ],
                capture_output=True,
                text=True,
            )
            if shutil.which("bwrap")
            else None
        )
        if run is None or run.returncode:
            raise GymError(
                "resource_unavailable",
                "The bwrap isolation backend is unavailable here: "
                + (run.stderr.strip()[-300:] if run else "bubblewrap is not installed"),
            )
    local_preflight(isolation)


def worker_command(python, platform, request, isolation, network="allow", command=None):
    """The worker (or `command`, a probe standing in for it) under `isolation`.

    `bwrap` and `landlock` sandbox it (without network); `none` leaves it unconfined, except that network `deny`
    refuses new IPv4/IPv6 sockets, and Unix sockets too for CPU work (CUDA needs them)."""
    from . import isolation as backends
    from . import netdeny

    worker = list(command or [python, "-m", "nexttransformer_gym.worker"])
    if isolation == "landlock":
        return backends.landlock_command(python, str(platform), request, worker)
    if isolation == "bwrap":
        return backends.bwrap_command(python, str(platform), request, worker)
    require(
        isolation == "none" and network in ("deny", "allow"),
        "invalid_request",
        "Unknown isolation backend or network policy.",
    )
    if network == "deny":
        cuda = str(request["profile"].get("device", "cpu")).startswith("cuda")
        return netdeny.command(python, worker, unix=not cuda)
    return worker


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["runtime"]:
        print(json.dumps(runtime(argv[1] if len(argv) > 1 else None)))
        return 0
    print(
        "usage: python -m nexttransformer_gym.remote runtime [python]", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
