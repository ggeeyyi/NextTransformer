"""Standard throughput harness of the MoE track.

    torchrun --nproc_per_node 8 -m moe.throughput --track d768 --methods baseline,<method> \\
        --resources <resources> --micro-batch 8 --output throughput-d768.json

Throughput is an outcome, not a control: a method is admitted on FLOPs per token, and its speed is reported
next to its quality. To make numbers comparable across machines, the harness measures the method and the
baseline back to back on the same node with the same code path the trainer uses (FSDP2, bf16, grouped GEMM,
real training rows of update 0, 1, 2, ...):

    warm-up   10 updates (allocation, autotuning; not timed)
    measured  30 updates, each synchronized; the 10% fastest and slowest are dropped

and reports, per method: seconds per update, tokens/s, tokens/s per device, MFU (protocol FLOPs per token x
tokens/s / (devices x 989.4 TFLOP/s)), peak memory, and the relative throughput against the baseline measured in
the same invocation. The relative throughput is the number published with a result.

Reference planning numbers per track (8 devices of the reference class) are in moe/references.json.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from moe import compute
from moe.data import Rows, directory
from moe.model import build_model
from moe.plugins import config_for
from moe.spec import PROTOCOL, config_path, load_config, schedule, track as get_track
from moe.train import PEAK_BF16_FLOPS, initialize, optimizer, print0, shard, update

WARMUP, MEASURED = 10, 30


def configuration(method, track_name):
    path = config_path(method, track_name)
    if path.exists():
        return load_config(path)[0], True
    return config_for(method, track_name), False


def measure(method, track_name, resources, micro_batch, rank, world, device, *, warmup=WARMUP, steps=MEASURED,
            debug=False, offload_activations=False):
    config, admitted = configuration(method, track_name)
    if debug:
        config = config.debug()
    track = get_track(track_name)
    rows_per_update = 4 if debug else track.rows_per_update
    share = rows_per_update // world
    if share % micro_batch:
        raise ValueError("the micro-batch must divide each rank's share of an update")
    accumulation = share // micro_batch
    model = shard(build_model(config, device), device)
    opt = optimizer(model, track)
    rows = Rows(directory(resources) / "train.bin", config.sequence_len)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(warmup + steps):
        started = time.perf_counter()
        update(model, opt, rows, step, schedule(step, track), rank, world, rows_per_update, micro_batch,
               accumulation, device, offload_activations=offload_activations)
        if step >= warmup:
            times.append(time.perf_counter() - started)
    trim = len(times) // 10
    steady = sorted(times)[trim:len(times) - trim] or times
    seconds = sum(steady) / len(steady)
    if world > 1:
        value = torch.tensor(seconds, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        seconds = value.item()
    tokens = rows_per_update * config.sequence_len
    flops = compute.analytic(config)["total"] if method == "baseline" or debug else _recorded_flops(config)
    result = dict(method=method, track=track_name, admitted=admitted, model_hash=config.fingerprint,
                  seconds_per_update=seconds, tokens_per_second=tokens / seconds,
                  tokens_per_second_per_device=tokens / seconds / world, flops_per_token=flops,
                  mfu=flops * tokens / seconds / (world * PEAK_BF16_FLOPS) if device.type == "cuda" else None,
                  peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
                  micro_batch=micro_batch, world_size=world, warmup=warmup, measured=steps,
                  offload_activations=offload_activations)
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _recorded_flops(config):
    path = config_path(config.method, config.track)
    if path.exists():
        return load_config(path)[1]["flops_per_token"]
    return compute.measure(config)["flops_per_token"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True)
    p.add_argument("--methods", default="baseline", help="comma-separated; the baseline is always measured first")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--offload-activations", action="store_true", help="hold large saved activations in host memory")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--steps", type=int, default=MEASURED)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    methods = ["baseline", *[m for m in args.methods.split(",") if m and m != "baseline"]]
    rank, world, device = initialize()
    results = [measure(m, args.track, args.resources, args.micro_batch, rank, world, device,
                       warmup=args.warmup, steps=args.steps, debug=args.debug,
                       offload_activations=args.offload_activations) for m in methods]
    base = results[0]["tokens_per_second"]
    for r in results:
        r["relative_throughput"] = r["tokens_per_second"] / base
        print0(f"{r['method']:>20} @ {r['track']}: {r['tokens_per_second']:,.0f} tok/s "
               f"({r['tokens_per_second_per_device']:,.0f}/device), x{r['relative_throughput']:.3f} of baseline"
               + (f", MFU {r['mfu']:.1%}" if r["mfu"] else ""))
    payload = dict(protocol=PROTOCOL.version, harness=dict(warmup=args.warmup, measured=args.steps, trim=0.1),
                   device_class=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   torch=torch.__version__, results=results)
    if args.output and rank == 0:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
