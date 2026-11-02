"""Capture build identity without requiring or initializing a GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def inventory(root, source_revision, download_sources=None):
    files = {}
    for name in ("pyproject.toml", "uv.lock", "requirements", "nexttransformer", "nanochat",
                 "nextt_gym", "configs", "launch", "environments/eval"):
        path = root / name
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in paths:
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                files[str(item.relative_to(root))] = hashlib.sha256(item.read_bytes()).hexdigest()
    return dict(
        status="built_unvalidated", source_base_revision=source_revision,
        python=platform.python_version(), platform=platform.platform(),
        python_executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        download_sources=download_sources or {},
        packages=dict(sorted((d.metadata["Name"], d.version)
                             for d in importlib.metadata.distributions())),
        source_files=files,
        source_files_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        reference_kernel_snapshot_included=False,
        gpu_parity_validated=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    for name in ("cuda-image", "uv-image", "apt-mirror", "pypi-index", "python-mirror"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    sources = {k: v for k, v in vars(args).items() if k != "source_revision" and v is not None}
    print(json.dumps(inventory(Path(__file__).resolve().parents[2], args.source_revision, sources), indent=2))
