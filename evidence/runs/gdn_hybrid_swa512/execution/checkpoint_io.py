"""Parallel native SHA256 reads at checkpoint commit, preserving the native commit protocol."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time


def signature(path):
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def hash_payloads(directory, native_sha256):
    paths = sorted(directory.glob("*.pt"))
    expected = {"model.pt", *(f"rank_{rank:04d}.pt" for rank in range(8))}
    if {path.name for path in paths} != expected:
        raise RuntimeError("Expected the model and all eight native rank payloads")
    before = {path: signature(path) for path in paths}
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = dict(zip(paths, pool.map(native_sha256, paths)))
    if any(signature(path) != before[path] for path in paths):
        raise RuntimeError("Checkpoint payload changed during checksum calculation")
    return {path: (before[path], digests[path]) for path in paths}


def install():
    from nexttransformer.runtime import checkpoint

    native_sha256 = checkpoint.sha256
    save_code = checkpoint.save_checkpoint.__code__
    pending = {}

    def checksum(path):
        path = Path(path)
        if sys._getframe(1).f_code is not save_code:
            return native_sha256(path)
        if not pending:
            started = time.monotonic()
            pending.update(hash_payloads(path.parent, native_sha256))
            evidence = Path(__file__).resolve().parents[1] / "evidence"
            record = dict(at=time.time(), checkpoint=str(path.parent),
                          lease=os.environ["T2M_H1_GPU_LEASE"],
                          seconds=time.monotonic()-started, files=len(pending),
                          bytes=sum(value[0][1] for value in pending.values()))
            with (evidence / "checkpoint-hashing.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
        stamp, digest = pending.pop(path)
        if signature(path) != stamp:
            raise RuntimeError("Checkpoint payload changed before manifest commit")
        return digest

    checkpoint.sha256 = checksum
