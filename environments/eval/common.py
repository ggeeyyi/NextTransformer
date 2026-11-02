"""Shared, dependency-free manifest helpers for scoring tooling."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_map(root, paths=None):
    root = Path(root)
    paths = sorted(root.rglob('*')) if paths is None else sorted(paths)
    return {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}


def verify_files(root, expected):
    require(bool(expected), 'Empty artifact manifest')
    root = Path(root).resolve()
    for name, value in expected.items():
        path = root / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                f'Invalid manifest path: {name}')
        require(path.is_file() and digest(path) == value, f'Artifact mismatch: {name}')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
