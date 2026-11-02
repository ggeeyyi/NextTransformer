"""Two-node (16-rank) data layout equals the frozen eight-rank layout row for row.

Real BestFitStreams over synthetic parquet shards; the BPB adapter and the
trainer run with 8 and 16 ranks and must read the same global rows in the same
order, with the same row ids and the same stream state after every update.
"""
from dataclasses import replace
import json
import os
import socket

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.multiprocessing as mp

from nexttransformer.runtime.data import BestFitStream
from nexttransformer.evaluation import formal as protocol
from nexttransformer.protocol.admission import data_layout, share_batches

CPU = torch.device("cpu")


class Tokenizer:
    def get_vocab_size(self): return 64
    def get_bos_token_id(self): return 63
    def encode(self, texts, prepend=None, num_threads=4):
        return [[prepend]+[ord(x) % 60 for x in s] for s in texts]


def shards(path):
    """Distinct documents; 20 row groups per shard, so even a wrong 16-stream split has data."""
    for n, name in enumerate(("shard_00000.parquet", "shard_00001.parquet", "shard_06542.parquet")):
        docs = [f"{n}:{i:04d}|" + "abcdefg"[i % 7]*(3+i % 11) for i in range(200)]
        pq.write_table(pa.table({"text": docs}), path/name, row_group_size=10)


def stream(path, rank, world, split="train"):
    return BestFitStream(path, Tokenizer(), split, rank, world, 16, buffer_size=12, tokenizer_batch_size=5)


def test_layout_shares_streams_only_for_multiples_of_eight():
    assert [data_layout(r, 8) for r in (0, 7)] == [(0, 8, 1), (7, 8, 1)]
    assert [data_layout(r, 16) for r in (0, 1, 2, 15)] == [(0, 8, 2), (0, 8, 2), (1, 8, 2), (7, 8, 2)]
    assert data_layout(0, 1) == (0, 1, 1) and data_layout(3, 4) == (3, 4, 1)
    with pytest.raises(ValueError, match="complete batches"):
        next(share_batches(None, 3, 8, 0, 2, CPU))


@pytest.mark.parametrize("world", [16, 32])
def test_registered_training_rows_and_state_equal_eight_rank(tmp_path, world):
    shards(tmp_path)
    eight = [stream(tmp_path, r, 8) for r in range(8)]
    expanded = [stream(tmp_path, *data_layout(r, world)[:2]) for r in range(world)]
    shares = world//8
    for update in range(4):
        expected = torch.cat([s.batch(4, CPU)[0] for s in eight])
        pieces = []
        for r, s in enumerate(expanded):
            batches = share_batches(s, 1, 4, r % shares, shares, CPU)
            pieces += [next(batches)[0] for _ in range(4//shares)]
            assert next(batches, None) is None
        assert torch.equal(torch.cat(pieces), expected), update
        for r, s in enumerate(expanded): assert s.state_dict() == eight[r//shares].state_dict()


def test_sixteen_rank_bpb_reads_the_eight_rank_rows(tmp_path, monkeypatch):
    shards(tmp_path)
    (tmp_path/"tokenizer").mkdir()
    torch.save(torch.ones(64, dtype=torch.int64), tmp_path/"tokenizer/token_bytes.pt")
    monkeypatch.setattr(protocol, "verify_resources", lambda resources, data_dir: None)

    class Model(torch.nn.Module):
        config = replace(protocol.Protocol(), sequence_len=16)  # only .sequence_len is read
        row_offset = 0
        def set_row_offset(self, offset): self.row_offset = offset
        def get_device(self): return CPU
        def forward(self, x, y=None, loss_reduction="mean"):
            seen.append((self.row_offset, x.clone())); return torch.zeros(x.shape)

    def read(world):
        seen.clear()
        for rank in range(world):
            protocol.evaluate_bpb(Model(), Tokenizer(), tmp_path, tmp_path, batch=2, rank=rank, world=world,
                                  device=CPU, tokens=2*16*16*2, split="val", smoke=True)
        offsets = [offset+i for offset, x in seen for i in range(len(x))]
        return offsets, torch.cat([x for _, x in seen])
    seen = []
    (o8, x8), (o16, x16) = read(8), read(16)
    assert o8 == o16 == list(range(len(x8))) and torch.equal(x8, x16)


def _rank(rank, world, port, root, name=None, extra=()):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    torch.set_num_threads(1)
    from nexttransformer import train
    from nexttransformer.models.config import debug_config
    stages = [dict(stage="joint", depth=None, cost=dict(total=5e15, phases=dict(forward=2e15)))]
    if os.environ.get("NEXTT_TEST_METHOD") == "ouro":  # gate-only stage from update 2
        model = replace(debug_config("ouro"), sequence_len=16, vocab_size=64,
                        ouro_gate_steps=train.Protocol().steps-2)
        stages.append(dict(stage="gate_only", depth=None, cost=dict(total=2e15, phases=dict(forward=1e15))))
    else:
        model = replace(debug_config("kda"), backend="fla_chunk64", sequence_len=16, vocab_size=64)
    proof = dict(formula_version="test", flops_per_token=1e9, total_training_flops=1, stages=stages)
    train.admit_config = lambda path: (model, proof)
    train.review_config = lambda path: (model, proof, [])
    train.verify_resources = lambda resources, data_dir: (dict(tokenizer_sha256="t"), "manifest")
    train.RustBPETokenizer.from_directory = staticmethod(lambda path: Tokenizer())
    train.BestFitStream = lambda data_dir, tokenizer, split, r, w, length: stream(data_dir, r, w, split)
    train.evaluate_bpb = lambda *a, tokens, split, smoke, **k: dict(bpb=1.0, input_tokens=tokens, split=split,
                                                                    seconds=0.0, smoke=smoke)
    train.evaluate_core = lambda *a, **k: dict(core_metric=0.0, results={})
    rows = []
    original = train.share_batches
    def recorded(*args):
        for x, y in original(*args): rows.append(x.clone()); yield x, y
    train.share_batches = recorded
    output = root/(name or f"world{world}")
    # 8 ranks x 2 microbatches or 16 ranks x 1: two rows of every stream per update.
    train.main(["--config", str(root/"config.json"), "--output", str(output), "--runtime-dir", str(root/"runtime"),
                "--resources", str(root), "--data-dir", str(root), "--micro-batch", "1", "--smoke-steps", "3",
                "--smoke-accumulation", str(16//world), "--rolling-every", "3", "--log-every", "1",
                "--wandb-mode", "disabled", *extra])
    torch.save(rows, output/f"rows_{rank:02d}.pt")


def _run(world, root, name=None, extra=()):
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    mp.start_processes(_rank, args=(world, port, root, name, tuple(extra)), nprocs=world, start_method="spawn")


@pytest.mark.ddp
def test_trainer_with_sixteen_ranks_reads_the_eight_rank_batches(tmp_path):
    shards(tmp_path)
    (tmp_path/"config.json").write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))
    for world in (8, 16): _run(world, tmp_path)
    def updates(world):  # per update: rows of all ranks in rank order
        per_rank = [torch.load(tmp_path/f"world{world}/rows_{r:02d}.pt") for r in range(world)]
        per_update = len(per_rank[0])//3
        return [torch.cat([torch.cat(rows[u*per_update:(u+1)*per_update]) for rows in per_rank]) for u in range(3)]
    for a, b in zip(updates(8), updates(16)): assert torch.equal(a, b)
    def events(world):
        lines = (tmp_path/f"world{world}/metrics.jsonl").read_text().splitlines()
        return [e for e in map(json.loads, lines) if e["event"] == "train"]
    a, b = events(8), events(16)
    assert [e["input_tokens"] for e in a] == [e["input_tokens"] for e in b] == [256, 512, 768]
    for x, y in zip(a, b): assert x["loss"] == pytest.approx(y["loss"], rel=1e-5)
    data = lambda world, r: torch.load(tmp_path/f"world{world}/checkpoint_00000003/rank_{r:04d}.pt",
                                       weights_only=False)["data"]
    for r in range(16): assert data(16, r) == data(8, r//2)


@pytest.mark.ddp
def test_resumed_segment_is_bitwise_with_eight_ddp_ranks(tmp_path):
    """A run segmented after its first update and resumed in a new process equals the continuous
    run bitwise under 8-rank DDP. Needs the trainer's DDP bucket warm-up: a new process's first
    backward would otherwise all-reduce with DDP's initial buckets, and a ring all-reduce over
    more than two ranks rounds by bucket layout (record 029)."""
    shards(tmp_path)
    (tmp_path/"config.json").write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))
    steps = ["--smoke-steps", "4"]  # argparse keeps the last occurrence
    _run(8, tmp_path, "continuous", steps)
    _run(8, tmp_path, "segmented", [*steps, "--max-wall-hours", "1e-12"])
    _run(8, tmp_path, "segmented", steps)
    def losses(name):
        lines = (tmp_path/name/"metrics.jsonl").read_text().splitlines()
        return {e["step"]: e["loss"] for e in map(json.loads, lines) if e["event"] == "train"}
    assert (tmp_path/"segmented"/"resume_00000001.json").exists()
    assert losses("continuous") == losses("segmented") and len(losses("continuous")) == 4
    weights = lambda name: torch.load(tmp_path/name/"checkpoint_00000004"/"model.pt", weights_only=False)["model"]
    a, b = weights("continuous"), weights("segmented")
    for key in a: assert torch.equal(a[key], b[key]), key


@pytest.mark.ddp
def test_ouro_gate_only_transition_is_bitwise_with_eight_ddp_ranks(tmp_path, monkeypatch):
    """The gate-only stage rebuilds the optimizer and the DDP reducer (then warms its buckets up).
    Crossing the boundary in a run, or resuming after it, equals the continuous 8-rank run bitwise."""
    monkeypatch.setenv("NEXTT_TEST_METHOD", "ouro")
    shards(tmp_path)
    (tmp_path/"config.json").write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))
    steps = ["--smoke-steps", "4"]
    _run(8, tmp_path, "continuous", steps)
    _run(8, tmp_path, "before", [*steps, "--max-wall-hours", "1e-12"])
    _run(8, tmp_path, "before", steps)
    _run(8, tmp_path, "after", ["--smoke-steps", "3"])
    _run(8, tmp_path, "after", steps)
    def events(name, kind):
        lines = (tmp_path/name/"metrics.jsonl").read_text().splitlines()
        return [e for e in map(json.loads, lines) if e["event"] == kind]
    assert [e["step"] for e in events("continuous", "stage_transition")] == [2]
    loss = lambda name: {e["step"]: e["loss"] for e in events(name, "train")}
    weights = lambda name: torch.load(tmp_path/name/"checkpoint_00000004"/"model.pt", weights_only=False)["model"]
    a = weights("continuous")
    for name in ("before", "after"):
        assert loss(name) == loss("continuous") and len(loss(name)) == 4
        b = weights(name)
        for key in a: assert torch.equal(a[key], b[key]), (name, key)
