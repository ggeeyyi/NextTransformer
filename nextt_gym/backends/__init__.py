"""Execution backends, configured in the ignored `nextt_gym/backends.local.toml` (see backends.example.toml)."""
import os
from pathlib import Path

from .base import Backend, STATES
from .koala import KoalaBackend
from .local import LocalBackend
from .modelarts import ModelArtsBackend
from .ssh import SSHBackend

KINDS = {cls.kind: cls for cls in (LocalBackend, SSHBackend, KoalaBackend, ModelArtsBackend)}
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "backends.local.toml"


def _toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:   # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_backends(path=None):
    """backend_id -> Backend. Missing file: no backends."""
    path = Path(path or os.environ.get("NEXTT_GYM_BACKENDS") or DEFAULT_PATH)
    if not path.exists(): return {}
    out = {}
    for backend_id, config in _toml(path).items():
        kind = config.get("kind")
        if kind not in KINDS: raise ValueError(f"backend {backend_id}: unknown kind {kind!r}")
        out[backend_id] = KINDS[kind](backend_id, config)
    return out


__all__ = ["Backend", "STATES", "KINDS", "load_backends", "LocalBackend", "SSHBackend", "KoalaBackend",
           "ModelArtsBackend"]
