"""Evaluate a saved MoE checkpoint: full validation BPB and the MoE-CORE suite.

    torchrun --nproc_per_node 8 -m moe.evaluate --run-dir runs/moe-d768-<method> \\
        --checkpoint runs/moe-d768-<method>/checkpoint_16875 --resources <resources> --output eval.json

The configuration is read from the run's run.json; every rank loads the full checkpoint and scores its share of
the examples. `--max-per-task` / `--long-limit` give a quick, clearly labelled partial evaluation.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from moe.data import verify
from moe.model import MoEConfig, build_model
from moe.train import evaluate_all, initialize


def load_checkpoint(config, path, device):
    import torch.distributed.checkpoint as dcp
    model = build_model(config, device, initialize=False)
    state = dict(model=model.state_dict(), step=0)
    dcp.load(state, checkpoint_id=str(path))
    model.load_state_dict(state["model"])
    return model, int(state["step"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="the run directory (for run.json)")
    # torchrun owns --run-path, so the flag cannot be a prefix of it
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-per-task", type=int, default=-1, help="examples per short task (partial evaluation)")
    p.add_argument("--long-limit", type=int, help="examples per long task (partial evaluation)")
    args = p.parse_args(argv)
    run = json.loads((args.run_dir / "run.json").read_text())
    config = MoEConfig.from_serialized(run["model"])
    rank, world, device = initialize()
    if verify(args.resources, None, full=False) != run["data_manifest_sha256"]:
        raise ValueError("checkpoint was trained on different prepared resources")
    model, step = load_checkpoint(config, args.checkpoint, device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    result = evaluate_all(model, config, device, args.resources, final=True,
                          max_per_task=args.max_per_task, long_limit=args.long_limit)
    result.update(step=step, method=config.method, track=config.track, model_hash=config.fingerprint,
                  partial=args.max_per_task > 0 or args.long_limit is not None)
    if rank == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
