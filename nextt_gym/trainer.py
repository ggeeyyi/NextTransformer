"""Gym training driver: one submitted architecture at one scale, under torchrun.

    python -m torch.distributed.run --standalone --nproc_per_node=8 -m nextt_gym.trainer \
        --submission DIR --run-spec spec.json --site-config SITE.json --output RUN_DIR [--resume auto]

Mirrors nexttransformer/train.py: the frozen eight-stream data layout, BestFitStream,
the pinned Muon/AdamW recipe (assigned by parameter role), the upstream LR and
momentum schedule scaled to the run's steps, DDP bucket warm-up and per-rank
kernel caches (records 029/031), atomic checkpoints and segmented resume. Each
update is charged the audited cost of its stage from the RunSpec; evaluations
are charged forward FLOPs per token times evaluated tokens. The run stops with
phase `budget_exceeded` before any update or evaluation that would pass
`stop_at_flops`.

`--synthetic` replaces data and tokenizer by deterministic random tokens. It is
a smoke mode for tests on CPU and can never produce a result.
"""
import os

from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime


def rank_kernel_caches(environ):
    """Per-local-rank Triton and Inductor cache directories (upstream record 031)."""
    if "LOCAL_RANK" in environ:
        for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if environ.get(key): environ[key] = f"{environ[key].rstrip('/')}/rank{environ['LOCAL_RANK']}"


if __name__ == "__main__":
    bootstrap_runtime()
    rank_kernel_caches(os.environ)

import argparse
from contextlib import nullcontext, redirect_stdout
import fcntl
import json
from pathlib import Path
import platform
import shutil
import socket
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.runtime.files import atomic_json, sha256
from nexttransformer.runtime.process import (append_event, initialize, reduce_sum, restore_rng, rewind_events,
                                         rng_state, source_manifest, synchronize)
from nanochat.loss_eval import evaluate_bpb as upstream_bpb
from nexttransformer.runtime.files import VALIDATION_SHARD
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.protocol.spec import Protocol

from .contract import files_sha256, load_submission
from .roles import build_optimizer
from .runspec import read_run_spec, validation_steps
from .task import DEV_SHARD, DEV_SHARD_SHA256, SCALES

share_batches = None   # nexttransformer.protocol.admission.share_batches, bound in main() for real data
# Event fields that stay in the run files but not in stdout, which agents can read through get_run_logs.
SEALED_FIELDS = ("val_bpb_official", "core")
PEAK_FLOPS = (("H200", 989e12), ("H100", 989e12), ("H20", 148e12), ("A800", 312e12), ("A100", 312e12))


class Schedule:
    """NativeTrainConfig schedule fields with the run's own step count."""
    def __init__(self, steps):
        base = NativeTrainConfig()
        self.steps, self.warmup_steps = steps, base.warmup_steps
        self.warmdown_ratio, self.final_lr_fraction = base.warmdown_ratio, base.final_lr_fraction


class SyntheticStream:
    """Deterministic random rows with the BestFitStream batch/state interface (smoke only)."""
    def __init__(self, vocab, sequence_len, seed, rank):
        self.vocab, self.sequence_len, self.seed, self.rank, self.rows = vocab, sequence_len, seed, rank, 0

    def batch(self, batch_size, device):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.rank * 10_007 + self.rows)
        rows = torch.randint(self.vocab, (batch_size, self.sequence_len + 1), generator=g)
        self.rows += batch_size
        return rows[:, :-1].contiguous().to(device), rows[:, 1:].contiguous().to(device)

    def batches(self, batch_size, device):
        while True: yield self.batch(batch_size, device)

    def state_dict(self): return dict(rows=self.rows)
    def load_state_dict(self, state): self.rows = state["rows"]


class EvalRows(torch.nn.Module):
    """Stable row offsets during evaluation and an input-token count (nexttransformer/evaluation/formal.py::EvalRows)."""
    def __init__(self, model, sub, offset, sequence_len):
        super().__init__()
        self.model, self.sub, self.offset, self.input_tokens = model, sub, offset, 0
        self.max_seq_len = sequence_len

    def get_device(self): return next(self.model.parameters()).device

    def forward(self, idx, *args, **kwargs):
        self.sub.set_row_offset(self.model, self.offset); self.offset += idx.shape[0]
        self.input_tokens += idx.numel()
        return self.model(idx, *args, **kwargs)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runtime_arguments(parser)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", default="auto", help="auto, never, or a checkpoint directory")
    parser.add_argument("--max-wall-hours", type=float, default=0.)
    parser.add_argument("--rolling-every", type=int, default=250)
    parser.add_argument("--rolling-keep", type=int, default=2)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="smoke only: random tokens, no data")
    parser.add_argument("--model-overrides", type=json.loads, default=None, help="smoke only: JSON model overrides")
    parser.add_argument("--smoke-rows", type=int, default=2, help="synthetic rows per rank and update")
    args = parser.parse_args(argv)
    if args.model_overrides is not None and not args.synthetic: parser.error("--model-overrides is smoke-only")
    return args


def checkpoints(output):
    complete, incomplete = {}, []
    for path in Path(output).glob("checkpoint_[0-9]*"):
        if path.is_dir():
            (complete.__setitem__(int(path.name.removeprefix("checkpoint_")), path)
             if (path / "complete.json").exists() else incomplete.append(path))
    return complete, incomplete


def choose_resume(output, resume, rank, world):
    complete, incomplete = checkpoints(output)
    if resume == "never" and (complete or (output / "provenance.json").exists()):
        raise ValueError("output already contains a run; use --resume auto or a new --output")
    if resume not in ("auto", "never"):
        chosen = Path(resume)
        if not (chosen / "complete.json").exists(): raise ValueError("--resume path is not a complete checkpoint")
    else:
        chosen = complete[max(complete)] if complete else None
    if rank == 0:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for path in incomplete:
            target = output / "incomplete_checkpoints" / f"{path.name}-{stamp}"
            target.parent.mkdir(exist_ok=True); path.rename(target)
    if world > 1: dist.barrier()
    return chosen


def save_checkpoint(output, model, optimizer, stream, state, rank, world):
    directory = Path(output) / f"checkpoint_{state['step']:08d}"
    if rank == 0: directory.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    local = dict(optimizer=optimizer.state_dict(), rng=rng_state(), data=stream.state_dict(),
                 routing_rng=model.routing_rng_state() if hasattr(model, "routing_rng_state") else None)
    tmp = directory / f"rank_{rank:04d}.pt.tmp"; torch.save(local, tmp); tmp.replace(directory / f"rank_{rank:04d}.pt")
    if rank == 0:
        tmp = directory / "model.pt.tmp"
        torch.save(dict(model=model.state_dict(), state=state, world_size=world), tmp)
        tmp.replace(directory / "model.pt")
    if world > 1: dist.barrier()
    if rank == 0:
        atomic_json(directory / "complete.json", dict(step=state["step"], world_size=world, files={
            p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(directory.glob("*.pt"))}))
        atomic_json(Path(output) / "latest.json", dict(step=state["step"], checkpoint=directory.name))
    if world > 1: dist.barrier()


def load_checkpoint(path, model, optimizer, stream, rank, world):
    manifest = json.loads((path / "complete.json").read_text())
    if manifest["world_size"] != world: raise ValueError("resume requires the original world size")
    for name in ("model.pt", f"rank_{rank:04d}.pt"):
        item = manifest["files"][name]
        if (path / name).stat().st_size != item["bytes"] or sha256(path / name) != item["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    saved = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    local = torch.load(path / f"rank_{rank:04d}.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(local["optimizer"]); stream.load_state_dict(local["data"]); restore_rng(local["rng"])
    if local["routing_rng"] is not None: model.restore_routing_rng(local["routing_rng"])
    return saved["state"]


def prune_rolling(output, keep, steps):
    complete, _ = checkpoints(output)
    rolling = sorted(s for s in complete if s != steps)
    for step in rolling[:max(len(rolling) - keep, 0)]: shutil.rmtree(complete[step])


def sealed(payload):
    """The stdout copy of an event, without official validation BPB or CORE."""
    if payload.get("split") == "official": return dict(payload, val_bpb="sealed")
    return {k: "sealed" if k in SEALED_FIELDS else v for k, v in payload.items()}


def check_dev_shard(data_dir, name):
    path = Path(data_dir) / name
    if not path.exists(): raise ValueError("dev shard is missing")
    if name != DEV_SHARD or sha256(path) != DEV_SHARD_SHA256:
        raise ValueError("dev shard differs from the pinned shard_06541 SHA256")


def dev_directory(output, data_dir, dev_shard, create):
    """BestFitStream reads all but the last file as `train`: [dev shard, pinned validation shard].

    Only one rank may `create` the links (the caller then waits at a barrier); concurrent ranks race
    between the existence check and the link.
    """
    directory = Path(output) / "dev_data"
    if not create: return directory
    directory.mkdir(exist_ok=True)
    for name in (dev_shard, VALIDATION_SHARD):
        link = directory / name
        if not link.exists(): link.symlink_to(Path(data_dir).resolve() / name)
    return directory


def main(argv=None):
    args = parse(argv)
    p = Protocol()
    spec = read_run_spec(args.run_spec)
    sub = load_submission(args.submission)
    if spec.get("submission_sha256") and spec["submission_sha256"] != files_sha256(args.submission):
        raise ValueError("submission files differ from the run spec")
    synthetic = args.synthetic
    if spec["synthetic"] != synthetic: raise ValueError("--synthetic must match the run spec")
    overrides = args.model_overrides or {}
    seq = int(overrides.get("sequence_len", sub.sequence_len))
    vocab = int(overrides.get("vocab_size", sub.vocab_size))
    steps = spec["steps"]
    scale = SCALES[spec["scale"]]
    # A prefix scale (S) runs the first `steps` updates of a `horizon`-update schedule.
    horizon = int(spec.get("schedule_steps") or steps)
    schedule = Schedule(horizon)
    stage_costs = {k: int(v) for k, v in spec["stage_costs"].items()}
    stop_at = int(spec["stop_at_flops"])
    forward_per_token = spec["forward_flops_per_token"]

    rank, world, device = initialize(spec["seed"])
    if not synthetic and world != spec["world_size"]: raise ValueError("WORLD_SIZE differs from the run spec")
    run_lock = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        run_lock = (args.output / "run.lock").open("a"); fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1: dist.barrier()
    resume = choose_resume(args.output, args.resume, rank, world)

    # ---- data -------------------------------------------------------------------
    if synthetic:
        micro, accumulation, shares, stream_rank, stream_world = args.smoke_rows, 1, 1, rank, world
        stream = SyntheticStream(vocab, seq, spec["seed"], rank)
        byte_table = torch.ones(vocab, dtype=torch.int64, device=device)
        tokenizer = data_dir = None; manifest_hash = "synthetic"
    else:
        # Imported here: the admission module pulls the upstream tokenizer (rustbpe), absent in CPU-only envs.
        global share_batches
        from nexttransformer.protocol.admission import data_layout, share_batches
        stream_rank, stream_world, shares = data_layout(rank, world)
        micro = spec.get("micro_batch") or (int(args.micro_batch) if args.micro_batch else None)
        if not micro: raise ValueError("micro batch required (run spec, --micro-batch or site micro_batch)")
        accumulation, rest = divmod(p.accumulation(micro, stream_world), shares)
        if rest: raise ValueError("micro batch does not divide this rank's share of the global batch")
        from nanochat.tokenizer import RustBPETokenizer
        record, manifest_hash = verify_resources(args.resources, args.data_dir)
        os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
        tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")
        if tokenizer.get_vocab_size() != vocab: raise ValueError("model/tokenizer vocabulary mismatch")
        data_dir = args.data_dir or args.resources / "base_data_climbmix"
        check_dev_shard(data_dir, spec["dev_shard"])
        stream = BestFitStream(data_dir, tokenizer, "train", stream_rank, stream_world, seq)
        if any(path.name == spec["dev_shard"] for path in stream.paths[:1]):
            raise ValueError("the dev shard cannot be the first training shard")
        byte_table = torch.load(Path(args.resources) / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)
    stream_rows = shares * accumulation * micro
    step_tokens = world * accumulation * micro * seq
    eval_batch = args.eval_batch or micro
    rows_per_update = step_tokens // seq

    # ---- model and optimizer ---------------------------------------------------
    start_step = 0
    stage = sub.stage_for_step(0, horizon)
    model = sub.instantiate(device, overrides=overrides or None, seed=spec["seed"])
    sub.set_stage(model, stage, audit=False, step=0, total_steps=horizon)
    optimizer = build_optimizer(model, sub.model_dim)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False) if world > 1 else model
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and sub.config.get(
        "autocast_bf16") else nullcontext()
    manifest = source_manifest()
    state = dict(step=0, input_tokens=0, consumed_train=0, consumed_eval=0, eval_tokens=0, training_seconds=0.,
                 evaluation_seconds=0., max_shard_index=-1, evaluations=[], stage=stage,
                 run_id=spec["run_id"], data_manifest_sha256=manifest_hash, micro_batch=micro)
    if resume is not None:
        previous = load_checkpoint(resume, model, optimizer, stream, rank, world)
        if previous["run_id"] != spec["run_id"] or previous["data_manifest_sha256"] != manifest_hash:
            raise ValueError("resume run/data identity differs from checkpoint")
        state = previous; start_step = state["step"]
        stage = sub.stage_for_step(min(start_step, steps - 1), horizon)
        if stage != state["stage"]: sub.set_stage(model, stage, audit=False, step=start_step, total_steps=horizon)
        state["stage"] = stage

    def consumed(): return state["consumed_train"] + state["consumed_eval"]

    def event(payload):
        if rank == 0: append_event(args.output, payload); print(json.dumps(sealed(payload)), flush=True)

    def write_budget():
        if rank == 0:
            atomic_json(args.output / "budget.json", dict(
                step=state["step"], steps=steps, input_tokens=state["input_tokens"],
                consumed_flops=str(consumed()), consumed_train_flops=str(state["consumed_train"]),
                consumed_eval_flops=str(state["consumed_eval"]), eval_tokens=state["eval_tokens"],
                reserved_flops=spec["reserved_flops"], stop_at_flops=spec["stop_at_flops"],
                gpu_hours=world * (state["training_seconds"] + state["evaluation_seconds"]) / 3600,
                max_shard_index=state["max_shard_index"]))

    if rank == 0:
        if resume is not None: rewind_events(args.output, state["step"])
        name = "provenance.json" if resume is None else f"resume_{state['step']:08d}.json"
        atomic_json(args.output / name, dict(
            run_spec=spec, submission=str(args.submission), submission_sha256=files_sha256(args.submission),
            source_commit=manifest["commit"] or None, native_source_sha256=manifest["sha256"],
            python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            host=socket.gethostname(), world_size=world, micro_batch=micro, accumulation=accumulation,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            parameters=sum(t.numel() for t in model.parameters()), synthetic=synthetic,
            model_overrides=overrides or None, command=sys.argv,
            kernel_caches={k: os.environ.get(k) for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}))

    def evaluate(split, tokens, step):
        """BPB on `split`; charges forward FLOPs for the evaluated tokens. None if over budget."""
        cost = int(forward_per_token * tokens)
        if consumed() + cost > stop_at: return None
        begin = time.monotonic()
        offset = rank * (tokens // seq // world)
        wrapper = EvalRows(model, sub, offset, seq)
        was_training = model.training; model.eval()
        try:
            if synthetic:
                evaluator = SyntheticStream(vocab, seq, spec["seed"] + (7 if split == "dev" else 11), rank)
                n = max(1, tokens // (eval_batch * world * seq))
                batches = evaluator.batches(eval_batch, device)
            else:
                directory = dev_directory(args.output, data_dir, spec["dev_shard"], rank == 0) if split == "dev" else data_dir
                if world > 1: dist.barrier()
                evaluator = BestFitStream(directory, tokenizer, "train" if split == "dev" else "val",
                                          stream_rank, stream_world, seq)
                denominator = eval_batch * world * seq
                if tokens % denominator: raise ValueError("evaluation tokens must contain complete batches")
                n = tokens // denominator
                batches = share_batches(evaluator, eval_batch, tokens // seq // stream_world, rank % shares, shares, device)
            with autocast:
                bpb = upstream_bpb(wrapper, batches, n, byte_table)
        finally:
            model.train(was_training)
        evaluated = wrapper.input_tokens * world
        seconds = time.monotonic() - begin
        state["consumed_eval"] += int(forward_per_token * evaluated)
        state["eval_tokens"] += evaluated; state["evaluation_seconds"] += seconds
        state["evaluations"].append([split, step])
        payload = dict(event="validation", split=split, step=step, val_bpb=bpb, input_tokens=evaluated, seconds=seconds)
        if rank == 0: atomic_json(args.output / f"val_{step:08d}_{split}.json", payload)
        event(payload); write_budget()
        return bpb

    splits = ["dev"] + (["official"] if spec["official_evaluated"] else [])
    scheduled = set(validation_steps(scale)) if not synthetic else {0}
    val_tokens = spec["val_tokens"] if not synthetic else eval_batch * world * seq
    final_tokens = spec["final_val_tokens"] if not synthetic else eval_batch * world * seq

    def scheduled_evaluations(step):
        for split in splits:
            if step in scheduled and [split, step] not in state["evaluations"]:
                if evaluate(split, val_tokens, step) is None: return False
        return True

    def checkpoint():
        save_checkpoint(args.output, model, optimizer, stream, state, rank, world)
        if rank == 0: prune_rolling(args.output, args.rolling_keep, steps)
        write_budget()

    def finish(phase, final=None):
        if rank == 0:
            atomic_json(args.output / "run_status.json", dict(
                phase=phase, step=state["step"], steps=steps, consumed_flops=str(consumed()),
                consumed_train_flops=str(state["consumed_train"]), consumed_eval_flops=str(state["consumed_eval"]),
                synthetic=synthetic, final=final, training_seconds=state["training_seconds"],
                evaluation_seconds=state["evaluation_seconds"], max_shard_index=state["max_shard_index"]))
            print(f"NEXTT_RUN_STATUS {phase} {state['step']} {consumed()}", flush=True)
        if world > 1: dist.destroy_process_group()

    if world > 1:
        # DDP rebuilds buckets after the first backward; one discarded backward fixes the layout
        # for every segment (upstream record 029). Data, optimizer, budget and RNG are unchanged.
        rngs = rng_state()
        routing = model.routing_rng_state() if hasattr(model, "routing_rng_state") else None
        dummy = torch.zeros(micro, seq, dtype=torch.long, device=device)
        with autocast: wrapped(dummy, dummy).backward()
        optimizer.zero_grad(set_to_none=True); restore_rng(rngs)
        if routing is not None: model.restore_routing_rng(routing)

    started = time.monotonic()
    model.train()
    if not scheduled_evaluations(state["step"]):
        write_budget(); return finish("budget_exceeded")
    phase = None
    for step in range(state["step"], steps):
        stage_now = sub.stage_for_step(step, horizon)
        cost = stage_costs[stage_now]
        if consumed() + cost > stop_at:
            phase = "budget_exceeded"; break
        if stage_now != state["stage"]:
            sub.set_stage(model, stage_now, audit=False, step=step, total_steps=horizon); state["stage"] = stage_now
        sub.set_step(model, step)
        synchronize(device); begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lr = set_schedule(optimizer, step, schedule)
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        batches = (stream.batches(micro, device) if synthetic else
                   share_batches(stream, micro, stream_rows, rank % shares, shares, device))
        for m in range(accumulation):
            x, y = next(batches)
            sub.set_row_offset(model, step * rows_per_update + rank * (rows_per_update // world) + m * micro)
            with (wrapped.no_sync() if world > 1 and m < accumulation - 1 else nullcontext()), autocast:
                loss = wrapped(x, y)
                (loss / accumulation).backward()
            totals += torch.stack((loss.detach().double() / (accumulation * world),
                                   torch.tensor(float(x.numel()), device=device, dtype=torch.float64)))
        if not synthetic and next(batches, None) is not None:
            raise RuntimeError("micro batches do not cover the stream's rows")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        loss_value, tokens = reduce_sum(totals).tolist()
        if int(tokens) != step_tokens: raise RuntimeError("training input-token budget drift")
        duration = time.perf_counter() - begin
        state.update(step=step + 1, input_tokens=state["input_tokens"] + step_tokens,
                     consumed_train=state["consumed_train"] + cost,
                     training_seconds=state["training_seconds"] + duration)
        if not synthetic:
            shard = stream.paths[min(stream.cursor["pq_idx"], len(stream.paths) - 1)].name
            index = int(shard.removeprefix("shard_").removesuffix(".parquet"))
            state["max_shard_index"] = max(state["max_shard_index"], index)
            if shard == spec["dev_shard"]: raise RuntimeError("training reached the dev shard")
        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == steps:
            event(dict(event="train", step=step + 1, stage=stage_now, loss=loss_value, lr_multiplier=lr,
                       grad_norm=float(grad_norm), input_tokens=state["input_tokens"], consumed_flops=str(consumed()),
                       step_seconds=duration, tokens_per_second=tokens / duration,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
            write_budget()
        if not scheduled_evaluations(step + 1):
            phase = "budget_exceeded"; checkpoint(); break
        segment_end = bool(args.max_wall_hours) and (time.monotonic() - started) / 3600 >= args.max_wall_hours \
            and step + 1 < steps
        if (step + 1) % args.rolling_every == 0 or step + 1 == steps or segment_end: checkpoint()
        if segment_end:
            phase = "segment_complete"; break

    if phase == "budget_exceeded":
        if state["step"] and not (args.output / f"checkpoint_{state['step']:08d}").exists(): checkpoint()
        event(dict(event="budget_exceeded", step=state["step"], consumed_flops=str(consumed()),
                   stop_at_flops=spec["stop_at_flops"]))
        write_budget(); return finish("budget_exceeded")
    if phase == "segment_complete":
        return finish("segment_complete")

    final = {}
    for split in splits:
        bpb = evaluate(split, final_tokens, steps) if [split, steps] not in state["evaluations"] else None
        if bpb is None and [split, steps] not in state["evaluations"]:
            write_budget(); return finish("budget_exceeded", final)
        if bpb is not None: final[f"val_bpb_{split}"] = bpb
    for split in splits:
        if f"val_bpb_{split}" not in final:
            path = args.output / f"val_{steps:08d}_{split}.json"
            if path.exists(): final[f"val_bpb_{split}"] = json.loads(path.read_text())["val_bpb"]
    if spec["core"] and not synthetic:
        from nexttransformer.evaluation.metrics import verify_core
        from nanochat.base_eval import evaluate_core
        verify_core(args.resources)
        wrapper = EvalRows(model, sub, 0, seq); model.eval(); begin = time.monotonic()
        # nanochat prints per-task CORE accuracies; they go to a file in the run directory, not to stdout.
        with open(args.output / "core_final.log" if rank == 0 else os.devnull, "w") as log, \
                redirect_stdout(log), torch.no_grad(), autocast:
            result = evaluate_core(wrapper, tokenizer, device)
        model.train()
        state["consumed_eval"] += int(forward_per_token * wrapper.input_tokens)
        state["eval_tokens"] += wrapper.input_tokens; state["evaluation_seconds"] += time.monotonic() - begin
        final["core"] = result.get("core_metric")
        if rank == 0: atomic_json(args.output / "core_final.json", dict(step=steps, **result))
    event(dict(event="final", step=steps, consumed_flops=str(consumed()), **final))
    write_budget()
    finish("smoke_complete" if synthetic else "training_complete", final)


if __name__ == "__main__":
    main()
