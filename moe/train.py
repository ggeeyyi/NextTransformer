"""Train a method on one MoE track.

    torchrun --nproc_per_node 8 -m moe.train --config moe/configs/d768/<method>.json \\
        --resources <resources> --output runs/moe-d768-<method> --micro-batch 8 --formal

Research mode (the default) accepts any configuration (--method/--track builds an uncalibrated one) and any
PyTorch build; its runs are never published. `--formal` requires an admitted configuration whose cost record
matches the current source files and PyTorch version, a verified data manifest and the full schedule.

Every run writes to --output:

    run.json         provenance (protocol, track, model, cost record and data fingerprints; mode; layout)
    metrics.jsonl    one JSON event per line: train (every 10 updates), val (every 500 updates and the end),
                     eval (MoE-CORE: 10 short tasks, RULER S-NIAH-1, FDA, SWDE at the track's four evaluation
                     steps), final (full validation BPB + MoE-CORE), throughput (steady-state tokens/s and MFU)
    ledger.json      updates, input tokens and charged training FLOPs
    checkpoint_<step>/  torch.distributed.checkpoint shards (model, optimizer) at evaluation steps and every 1,000 updates

On CUDA the model is wrapped with FSDP2 (bf16 parameters and compute, fp32 gradient reduction); routed experts
use torch._grouped_mm. Evaluation runs on an unsharded bf16 replica on every rank.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist

from moe import compute
from moe.data import Rows, directory, micro_batches, verify
from moe.model import build_model, eager_attention
from moe.offload import activation_storage
from moe.spec import PROTOCOL, REPO, learning_rate, load_config, schedule, track as get_track

CHECKPOINT_EVERY = 1000
LOG_EVERY = 10
PEAK_BF16_FLOPS = 989.4e12   # dense BF16 peak of the reference accelerator, for MFU only


def initialize():
    if "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(PROTOCOL.seed)
    return rank, world, device


def print0(*args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, flush=True)


def shard(model, device):
    """FSDP2 on CUDA: one unit per block plus the root; bf16 params/compute, fp32 reductions."""
    if device.type != "cuda":
        return model
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    return model


def optimizer(model, track):
    """Marin's MuonH / AdamH / Adam groups (moe/optim.py)."""
    from moe.optim import build
    return build(model, track)


def token_bytes(config):
    if config.scale == "debug":
        return torch.ones(config.vocab_size, dtype=torch.int32)
    from moe.tokenizer import Dolma2Tokenizer
    table = Dolma2Tokenizer().token_bytes()
    return torch.cat((table, torch.zeros(config.vocab_size - table.numel(), dtype=torch.int32)))


def git_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or None


class Run:
    """Events, ledger and provenance of one run directory (rank 0 writes)."""

    def __init__(self, output, rank):
        self.output, self.rank = Path(output), rank
        if rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **values):
        if self.rank == 0:
            with (self.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(event=kind, time=time.time(), **values)) + "\n")

    def write(self, name, value):
        if self.rank == 0:
            path = self.output / name
            path.with_suffix(".tmp").write_text(json.dumps(value, indent=2) + "\n")
            path.with_suffix(".tmp").replace(path)


def eval_replica(model, config, device):
    """An unsharded bf16 copy for evaluation, so ranks can score different examples independently."""
    if device.type != "cuda":
        return model
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
    replica = build_model(config, device, initialize=False)
    replica.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
    return replica.to(torch.bfloat16).eval()


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate_all(model, config, device, resources, *, max_per_task=-1, long_limit=None, final=False):
    """Validation BPB, then (formal scale) the MoE-CORE suite: 10 short tasks, RULER S-NIAH-1, FDA and SWDE."""
    from moe.evaluation import bpb, recall, ruler, score, short
    out = directory(resources)
    replica = eval_replica(model, config, device)
    replica.eval()
    result = {}
    with autocast(device), eager_attention():
        result["val"] = bpb.evaluate(replica, out / "val", token_bytes(config), device,
                                     rows_per_source=None if final else PROTOCOL.val_rows_per_source)
        if config.scale == "formal":
            result["short"] = short.evaluate(replica, resources, device, max_per_task=max_per_task)
            result["ruler"] = ruler.evaluate(replica, device, limit=long_limit)
            result["recall"] = recall.evaluate(replica, {t: out / "eval" / f"{t}.parquet" for t in recall.DATASETS},
                                               device, limit=long_limit)
            if max_per_task <= 0 and long_limit is None:
                result["score"] = score.moe_core(result["short"], result["ruler"], result["recall"])
    if replica is not model:
        del replica
        torch.cuda.empty_cache()
    model.train()
    return result


def save(model, opt, step, output):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    msd, osd = get_state_dict(model, opt)
    dcp.save(dict(model=msd, optim=osd, step=step), checkpoint_id=str(Path(output) / f"checkpoint_{step:05d}"))


def load(model, opt, path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    msd, osd = get_state_dict(model, opt)
    state = dict(model=msd, optim=osd, step=0)
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return int(state["step"])


def formal_checks(config, calibration, manifest_digest, args, world, track):
    if calibration.get("status") != "admitted":
        raise ValueError("formal runs need an admitted configuration (python -m moe.compute calibrate)")
    record = json.loads((REPO / calibration["cost_record"]).read_text())
    if record["source"]["sha256"] != compute.source_digest(config.method)["sha256"]:
        raise ValueError("the cost record no longer matches the method's source files; recalibrate")
    if record["torch"] != torch.__version__.split("+")[0]:
        raise ValueError(f"formal runs use the audited PyTorch build {record['torch']}")
    if abs(record["flops_per_token"] / compute.anchor(config.track) - 1) > PROTOCOL.preflight_tolerance:
        raise ValueError("configuration is outside the 1% admission window")
    if args.smoke_steps:
        raise ValueError("--smoke-steps is research only")
    if track.rows_per_update % world:
        raise ValueError("the world size must divide the rows of an update")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="admitted configuration (moe/configs/<track>/<method>.json)")
    p.add_argument("--method", help="research mode: build an uncalibrated configuration of this method")
    p.add_argument("--track", help="research mode: the track for --method")
    p.add_argument("--debug", action="store_true", help="tiny CPU shape (tests)")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--offload-activations", action="store_true", help="hold large saved activations in host memory")
    p.add_argument("--formal", action="store_true")
    p.add_argument("--resume", type=Path, help="a checkpoint_<step> directory of this run")
    p.add_argument("--smoke-steps", type=int, help="research: stop after this many updates")
    p.add_argument("--max-per-task", type=int, default=-1, help="research: examples per short task")
    p.add_argument("--long-limit", type=int, help="research: examples per long task (and RULER length)")
    p.add_argument("--no-eval", action="store_true", help="research: skip MoE-CORE events")
    args = p.parse_args(argv)

    from moe.plugins import config_for
    if args.config:
        config, calibration = load_config(args.config)
    elif args.method and args.track:
        config, calibration = config_for(args.method, args.track), {"status": "research"}
    else:
        p.error("pass --config, or --method and --track for research mode")
    if args.debug:
        config = config.debug()
    track = get_track(config.track)
    if args.formal and (args.debug or args.max_per_task > 0 or args.long_limit or args.no_eval):
        p.error("formal runs use the full model, schedule and evaluation")

    rank, world, device = initialize()
    run = Run(args.output, rank)
    manifest_digest = verify(args.resources, track.name if args.formal else None, full=args.formal)
    record = formal_checks(config, calibration, manifest_digest, args, world, track) if args.formal else None
    flops_per_token = record["flops_per_token"] if record else compute.analytic(config)["total"]

    model = build_model(config, device)
    model = shard(model, device)
    opt = optimizer(model, track)
    rows = Rows(directory(args.resources) / "train.bin", config.sequence_len)
    rows_per_update = track.rows_per_update if not args.debug else 4
    share = rows_per_update // world
    accumulation = share // args.micro_batch if share % args.micro_batch == 0 else None
    if not accumulation:
        raise ValueError(f"micro-batch {args.micro_batch} must divide the {share} rows each rank holds per update")
    if args.micro_batch % config.qb_group_rows:
        raise ValueError(f"micro-batch {args.micro_batch} must hold whole quantile-balancing groups "
                         f"({config.qb_group_rows} rows)")
    start = load(model, opt, args.resume) if args.resume else 0   # checkpoints store completed updates
    done = recorded(args.output) if args.resume else set()
    updates = min(track.updates, args.smoke_steps) if args.smoke_steps else track.updates
    run.write("run.json" if start == 0 else f"run.resume-{start:05d}.json", dict(
        protocol=PROTOCOL.version, protocol_hash=PROTOCOL.fingerprint, track=track.name, track_hash=track.fingerprint,
        method=config.method, model=config.serialized(), model_hash=config.fingerprint, calibration=calibration,
        formal=args.formal, data_manifest_sha256=manifest_digest, world_size=world, micro_batch=args.micro_batch,
        accumulation=accumulation, offload_activations=args.offload_activations,
        torch=torch.__version__, commit=git_commit(), resumed_from=start or None,
        flops_per_token=flops_per_token, parameters=build_model(config, "meta").parameter_counts()))
    if start == 0:
        run.event("start", step=0)
    print0(f"{config.method} @ {track.name}: {updates} updates x {track.tokens_per_update:,} tokens, world {world}, "
           f"micro-batch {args.micro_batch} x {accumulation}")

    anchor = _anchor_or_none(config) if not args.debug else None
    eval_steps = set() if args.no_eval or args.debug else set(track.core_steps)
    val_steps = set(track.validation_steps) | {updates}
    tokens_per_update = rows_per_update * config.sequence_len
    window, timed = [], 0
    for step in range(start, updates + 1):
        if (step in val_steps or step in eval_steps) and ("eval" if step in eval_steps else "val", step) not in done:
            is_eval = step in eval_steps
            result = evaluate_all(model, config, device, args.resources, final=False,
                                  max_per_task=args.max_per_task, long_limit=args.long_limit) \
                if is_eval else dict(val=_val_only(model, config, device, args.resources))
            run.event("eval" if is_eval else "val", step=step, **result)
            print0(f"step {step}: val macro BPB {result['val']['macro_bpb']:.4f}"
                   + (f", MoE-CORE {result['score']['moe_core']:.4f} (short {result['score']['short']:.4f}, "
                      f"long {result['score']['long']:.4f})" if "score" in result else ""))
        if step == updates:
            break
        started = time.perf_counter()
        lr = learning_rate(step, track)
        total, norm = update(model, opt, rows, step, schedule(step, track), rank, world, rows_per_update, args.micro_batch,
                             accumulation, device, offload_activations=args.offload_activations)
        elapsed = time.perf_counter() - started
        if step >= start + 5:   # skip warm-up updates (allocation, autotuning)
            window.append(elapsed)
            timed += 1
        if step % LOG_EVERY == 0 or step == updates - 1:
            norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
            tps = tokens_per_update / elapsed
            run.event("train", step=step + 1, loss=total.item(), lr=lr, grad_norm=float(norm), seconds=elapsed,
                      tokens_per_second=tps, imbalance=routing_imbalance(model))
            print0(f"step {step + 1}/{updates} loss {total.item():.4f} lr {lr:.2e} {tps:,.0f} tok/s")
        if (step + 1) % CHECKPOINT_EVERY == 0 or (step + 1) in eval_steps:
            save(model, opt, step + 1, args.output)
        consumed = (step + 1) * tokens_per_update
        run.write("ledger.json", dict(updates=step + 1, input_tokens=consumed, flops_per_token=flops_per_token,
                                      charged_flops=consumed * flops_per_token,
                                      anchor_flops=anchor * consumed if anchor else None))
    if window:
        steady = sorted(window)[len(window) // 10: len(window) - len(window) // 10] or window
        mean = sum(steady) / len(steady)
        run.event("throughput", step=step, updates_timed=timed, seconds_per_update=mean,
                  tokens_per_second=tokens_per_update / mean, tokens_per_second_per_device=tokens_per_update / mean / world,
                  mfu=flops_per_token * tokens_per_update / mean / (world * PEAK_BF16_FLOPS) if device.type == "cuda" else None)
    if args.smoke_steps and not args.debug:   # research: keep the end state so resuming can be exercised
        save(model, opt, updates, args.output)
    if updates == track.updates and not args.debug:
        final = evaluate_all(model, config, device, args.resources, final=True,
                             max_per_task=args.max_per_task, long_limit=args.long_limit)
        run.event("final", step=updates, **final)
        save(model, opt, updates, args.output)
    run.event("end", step=updates)
    if dist.is_initialized():
        dist.destroy_process_group()


def update(model, opt, rows, step, multiplier, rank, world, rows_per_update, micro_batch, accumulation, device,
           *, offload_activations=False):
    """One optimizer update of the shared recipe; returns (mean loss over the update, gradient norm).

    `multiplier` is the schedule value for this step (each optimizer group scales its own peak rate)."""
    opt.set_schedule(multiplier)
    total = torch.zeros((), device=device)
    for i, (x, y) in enumerate(micro_batches(rows, step, rank, world, rows_per_update, micro_batch, device)):
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(i == accumulation - 1)
        with autocast(device), activation_storage(offload_activations):
            loss = model(x, y)
            aux = getattr(model, "last_aux", None)
            objective = (loss + (aux if aux is not None else 0)) / accumulation
        objective.backward()
        total += loss.detach() / accumulation
    balance_routers(model, world)
    # Marin clips nothing; the norm is only logged (max_norm=inf leaves gradients unchanged)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    opt.step()
    opt.zero_grad(set_to_none=True)
    if world > 1:
        dist.all_reduce(total, op=dist.ReduceOp.AVG)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return total, norm


def _anchor_or_none(config):
    try:
        return compute.anchor(config.track)
    except FileNotFoundError:
        return None


def recorded(output):
    """(event, step) pairs already in a run's metrics, so a resumed run does not repeat them."""
    path = Path(output) / "metrics.jsonl"
    if not path.exists():
        return set()
    return {(e["event"], e.get("step")) for e in map(json.loads, path.read_text().splitlines())}


def _val_only(model, config, device, resources):
    from moe.evaluation import bpb
    replica = eval_replica(model, config, device)
    with autocast(device), eager_attention():
        result = bpb.evaluate(replica, directory(resources) / "val", token_bytes(config), device)
    if replica is not model:
        del replica
        torch.cuda.empty_cache()
    model.train()
    return result


@torch.no_grad()
def balance_routers(model, world):
    """Marin's QB update: the bias of every router becomes -beta + mean(beta), beta = mean over all groups."""
    for router in model.routers():
        stats = torch.cat((router.beta_sum, router.beta_count.view(1), router.load))
        if world > 1:
            dist.all_reduce(stats)
        E = router.beta_sum.numel()
        router.load.copy_(stats[E + 1:])
        router.balance(stats[:E], stats[E])
        router.beta_sum.zero_()
        router.beta_count.zero_()
        router.load.zero_()


@torch.no_grad()
def routing_imbalance(model):
    """max / mean expert load of the last update, averaged over layers (1.0 is perfectly balanced)."""
    values = [float(r.last_load.max() / r.last_load.mean().clamp_min(1e-9)) for r in model.routers()
              if getattr(r, "last_load", None) is not None]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
