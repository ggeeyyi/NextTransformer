"""Pinned Dolma 3 HQ data, pre-tokenized once into flat token files, and the fixed row order every run reads.

    python -m moe.data prepare --root <resources> [--track d1024] [--workers 32]
    python -m moe.data verify  --root <resources>

Stream definition (identical for every method):

- Training documents are read from shards train-00000, train-00001, ... in file order and row-group order.
  Every document is tokenized with the dolma2 tokenizer and prefixed with <|endoftext|>; documents are
  concatenated into one token stream (no truncation, no padding, no document masking).
- Row k of the stream is tokens[k*S : k*S + S + 1] (S = 4,096): S inputs and S shifted targets.
- Update u reads rows [u*R, (u+1)*R) with R rows per update. Rank r of a W-rank run reads the contiguous
  slice [u*R + r*R/W, u*R + (r+1)*R/W); micro-batches split that slice in order. The rows an update uses
  therefore do not depend on the world size or the micro-batch.
- Shard 255 is the development split for model selection. The six validation files are the reserved
  validation split: each source is its own stream, scored on its first 128 rows at every validation event
  and on all complete rows at the end; BPB is the macro average over the six sources.

`prepare` writes <root>/moe/{train.bin, dev.bin, val/<source>.bin, eval/{fda,swde}.parquet, manifest.json,
tokenizer/tokenizer.json} and the pinned DCLM bundle at <root>/eval_bundle.
Token files are little-endian uint32. The manifest pins every input file's SHA-256 and every output's
SHA-256; training refuses resources whose manifest does not verify.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import shutil

import numpy as np

from moe.spec import (DATA_REPO, DATA_REVISION, DEV_SHARD, PROTOCOL, TOKENIZER_JSON_SHA256, TOKENIZER_REPO,
                      TOKENIZER_REVISION, TRACKS, TRAIN_SHARD, VALIDATION_FILE, VALIDATION_SOURCES)
from moe.evaluation.recall import DATASETS as RECALL
from moe.tokenizer import EOT, Dolma2Tokenizer

DTYPE = np.dtype("<u4")
DEV_TOKENS = 64 * 2**20   # 67,108,864 development tokens are plenty for model selection


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def directory(root):
    return Path(root) / "moe"


def needed_tokens(track_names):
    return max(TRACKS[t].total_tokens for t in track_names) + 1


def _download(filename, root, repo=DATA_REPO, revision=DATA_REVISION, repo_type="dataset"):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, cache_dir=Path(root) / "hf"))


def _tokenize(texts):
    tok = Dolma2Tokenizer()
    rows = tok.encode(texts, prepend=EOT, num_threads=1)
    return np.fromiter((t for r in rows for t in r), dtype=DTYPE)


def _row_groups(path, batch=512):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        for i in range(0, len(texts), batch):
            yield texts[i:i + batch]


def _write_stream(paths, out, limit, workers):
    """Tokenize documents of `paths` in order into `out` until `limit` tokens; returns tokens written."""
    written = 0
    with open(out, "wb") as f, Pool(workers) as pool:
        def batches():
            for p in paths:
                yield from _row_groups(p)
        for array in pool.imap(_tokenize, batches(), chunksize=4):
            if limit is not None:
                array = array[:limit - written]
            f.write(array.tobytes())
            written += array.size
            if limit is not None and written >= limit:
                pool.terminate()
                break
    return written


def prepare(root, track_names=PROTOCOL.tracks, workers=None, limit_tokens=None):
    """limit_tokens (research only) stops the training and development streams early; the manifest says so."""
    out = directory(root)
    (out / "val").mkdir(parents=True, exist_ok=True)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tok_json = _download("tokenizer.json", root, TOKENIZER_REPO, TOKENIZER_REVISION, "model")
    if sha256(tok_json) != TOKENIZER_JSON_SHA256:
        raise ValueError("dolma2 tokenizer.json differs from the pinned revision")
    shutil.copyfile(tok_json, out / "tokenizer" / "tokenizer.json")
    inputs = {}
    # Training stream: download shards one at a time until the budget is covered.
    limit = needed_tokens(track_names) if limit_tokens is None else min(needed_tokens(track_names), limit_tokens)
    shards, have = [], 0
    stream = out / "train.bin"
    stream.unlink(missing_ok=True)
    for index in range(DEV_SHARD):
        name = TRAIN_SHARD.format(index=index)
        path = _download(name, root)
        inputs[name] = sha256(path)
        shards.append(path)
        part = out / f".part-{index:05d}.bin"
        have += _write_stream([path], part, limit - have, workers)
        with open(stream, "ab") as dst, open(part, "rb") as src:
            shutil.copyfileobj(src, dst)
        part.unlink()
        print(f"train: {name} -> {have:,} / {limit:,} tokens", flush=True)
        if have >= limit:
            break
    if have < limit:
        raise RuntimeError("training shards exhausted before the budget")
    dev_name = TRAIN_SHARD.format(index=DEV_SHARD)
    dev_path = _download(dev_name, root)
    inputs[dev_name] = sha256(dev_path)
    dev = _write_stream([dev_path], out / "dev.bin", DEV_TOKENS if limit_tokens is None else min(DEV_TOKENS, limit_tokens),
                        workers)
    val = {}
    for source in VALIDATION_SOURCES:
        name = VALIDATION_FILE.format(source=source)
        path = _download(name, root)
        inputs[name] = sha256(path)
        val[source] = _write_stream([path], out / "val" / f"{source}.bin", None, workers)
    # Short tasks: the dense protocol's pinned DCLM bundle (<root>/eval_bundle + core_manifest.json).
    from nexttransformer.runtime.resources import prepare_core
    from nexttransformer.evaluation.metrics import verify_core
    prepare_core(Path(root))
    verify_core(root)
    # FDA and SWDE (long-document recall), pinned by revision and SHA-256.
    (out / "eval").mkdir(exist_ok=True)
    for task, spec in RECALL.items():
        path = _download(spec["file"], root, spec["repo"], spec["revision"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"{task} data differ from the pinned file")
        shutil.copyfile(path, out / "eval" / f"{task}.parquet")
    outputs = {str(p.relative_to(out)): sha256(p) for p in sorted([*out.rglob("*.bin"), *out.glob("eval/*.parquet")])}
    outputs["tokenizer/tokenizer.json"] = TOKENIZER_JSON_SHA256
    manifest = dict(protocol=PROTOCOL.version, data_repo=DATA_REPO, data_revision=DATA_REVISION,
                    tokenizer_repo=TOKENIZER_REPO, tokenizer_revision=TOKENIZER_REVISION, tracks=list(track_names),
                    partial=limit_tokens is not None,
                    train_tokens=have, dev_tokens=dev, validation_tokens=val, inputs=inputs, outputs=outputs)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root, track_name=None, full=True):
    """The manifest digest every run records; checks token budgets and (optionally) every file hash."""
    out = directory(root)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["data_revision"] != DATA_REVISION or manifest["tokenizer_revision"] != TOKENIZER_REVISION:
        raise ValueError("prepared data come from a different data or tokenizer revision")
    if track_name and (manifest.get("partial") or manifest["train_tokens"] < TRACKS[track_name].total_tokens + 1):
        raise ValueError(f"prepared training stream is too short for {track_name}")
    if full:
        for name, digest in manifest["outputs"].items():
            if sha256(out / name) != digest:
                raise ValueError(f"prepared file differs from its manifest: {name}")
    return hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()


class Rows:
    """Random access to rows of a flat token file."""

    def __init__(self, path, sequence_len=PROTOCOL.sequence_len):
        self.tokens = np.memmap(path, dtype=DTYPE, mode="r")
        self.S = sequence_len
        self.count = (self.tokens.size - 1) // sequence_len

    def __len__(self):
        return self.count

    def get(self, start, stop):
        if not 0 <= start <= stop <= self.count:
            raise IndexError("row range outside the stream")
        S = self.S
        flat = np.asarray(self.tokens[start * S: stop * S + 1], dtype=np.int64)
        idx = np.arange(stop - start)[:, None] * S + np.arange(S + 1)[None, :]
        return flat[idx]


def update_rows(update, rank, world, rows_per_update):
    if rows_per_update % world:
        raise ValueError("the world size must divide the rows of an update")
    share = rows_per_update // world
    start = update * rows_per_update + rank * share
    return start, start + share


def micro_batches(rows, update, rank, world, rows_per_update, micro_batch, device):
    """Yield (inputs, targets) micro-batches of this rank's share of `update`."""
    import torch
    start, stop = update_rows(update, rank, world, rows_per_update)
    if (stop - start) % micro_batch:
        raise ValueError("the micro-batch must divide each rank's share of an update")
    for a in range(start, stop, micro_batch):
        block = torch.from_numpy(rows.get(a, a + micro_batch))
        if device.type == "cuda":
            block = block.pin_memory()
        block = block.to(device, non_blocking=True)
        yield block[:, :-1], block[:, 1:]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--track", action="append", choices=tuple(TRACKS))
    a.add_argument("--workers", type=int)
    a.add_argument("--limit-tokens", type=int, help="research only: stop the training stream after N tokens")
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--track", choices=tuple(TRACKS))
    args = p.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(args.root, tuple(args.track or PROTOCOL.tracks), args.workers, args.limit_tokens)
        print(json.dumps({k: manifest[k] for k in ("train_tokens", "dev_tokens", "validation_tokens")}, indent=2))
    else:
        print(verify(args.root, args.track))


if __name__ == "__main__":
    main()
