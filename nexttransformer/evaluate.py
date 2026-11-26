"""Evaluate a trained checkpoint with the protocol's BPB and CORE.

    python -m nexttransformer.evaluate --site-config site.json \\
        --checkpoint RUN/checkpoint_00046875 --output eval.json [--eval bpb,core]

Native-track checkpoints (SSSL Dense, MoR, LoopFormer, Circle) use the native evaluators.
Registered-method checkpoints use the formal evaluators: train/val BPB on the frozen
eight-stream layout (8, 16, or 32 ranks) and full CORE. On other world sizes pass
--informal; such BPB results are labelled smoke and are not formal results.
"""
from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime
if __name__ == "__main__":
    bootstrap_runtime()

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.context import add_core_max_tokens
from nexttransformer.models import load_model
from nexttransformer.runtime.data import verify_resources
from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.process import initialize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runtime_arguments(p)
    add_core_max_tokens(p)
    p.add_argument("--checkpoint", type=Path, required=True, help="a complete checkpoint_<step>/ directory")
    p.add_argument("--output", type=Path, required=True, help="result JSON")
    p.add_argument("--eval", default="bpb,core", help="bpb, core, or both")
    p.add_argument("--split-tokens", type=int, default=40*524288, help="native track: BPB tokens per split")
    p.add_argument("--max-per-task", type=int, default=-1, help="CORE examples per task (-1: all; others are smoke)")
    p.add_argument("--informal", action="store_true",
                   help="registered methods: allow a non-protocol world size (BPB labelled smoke)")
    p.add_argument("--expected-attention", choices=("fa3", "sdpa", "sdpa-noflash"),
                   help="Scoring contract: reject unexpected attention dispatch on every rank")
    p.add_argument("--sparse-bpb", choices=("reference", "kernel"), default="reference")
    p.add_argument("--sparse-core", choices=("reference", "kernel"), default="reference")
    args = p.parse_args(argv)
    modes = set(args.eval.split(","))
    if modes - {"bpb", "core"}: p.error("--eval must be bpb, core, or bpb,core")
    _, manifest_hash = verify_resources(args.resources, args.data_dir)
    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    rank, world, device = initialize(42)
    if args.expected_attention:
        from nanochat import flash_attention
        if args.expected_attention == "sdpa-noflash":
            torch.backends.cuda.enable_flash_sdp(False)
        actual = "fa3" if flash_attention.USE_FA3 else (
            "sdpa" if torch.backends.cuda.flash_sdp_enabled() else "sdpa-noflash")
        if actual != args.expected_attention:
            raise RuntimeError(f"Unexpected attention backend: {actual}")
    # The scorer forces weights-only loading; trainer checkpoints record torch.__version__,
    # a str subclass that runs no code when rebuilt.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    model, saved = load_model(args.checkpoint, device)
    if saved["state"]["data_manifest_sha256"] != manifest_hash:
        raise ValueError("checkpoint was trained on different prepared resources")
    tokenizer = RustBPETokenizer.from_directory(args.resources/"tokenizer")
    data_dir = args.data_dir or args.resources/"base_data_climbmix"
    batch = int(args.micro_batch or 4)
    raw = saved["model_config"]
    native = "architecture" in raw
    result = dict(step=saved["state"]["step"], checkpoint=str(args.checkpoint), native_manifest_sha256=manifest_hash,
                  track="native" if native else "registered", method=raw.get("method", raw.get("architecture")))
    if native:
        from nexttransformer.evaluation.native import evaluate_bpb, evaluate_core
        if "bpb" in modes:
            result["bpb"] = {split: evaluate_bpb(model, tokenizer, data_dir, args.resources, args.split_tokens,
                                                 batch, rank, world, device, split) for split in ("train", "val")}
        if "core" in modes:
            result["core"] = evaluate_core(model, tokenizer, args.resources, device, args.max_per_task,
                                           core_max_tokens=args.core_max_tokens)
    else:
        from nexttransformer.evaluation.formal import evaluate_bpb, evaluate_core
        from nexttransformer.protocol.spec import Protocol
        tokens = Protocol().final_bpb_tokens
        model.set_sparse_backend(args.sparse_bpb)
        if "bpb" in modes:
            result["bpb"] = {split: evaluate_bpb(model, tokenizer, data_dir, args.resources, batch=batch, rank=rank,
                                                 world=world, device=device, tokens=tokens, split=split,
                                                 smoke=args.informal) for split in ("train", "val")}
        if "core" in modes:
            model.set_sparse_backend(args.sparse_core)
            result["core"] = evaluate_core(model, tokenizer, args.resources, device,
                                           smoke_max_per_task=None if args.max_per_task < 0 else args.max_per_task,
                                           core_max_tokens=args.core_max_tokens)
    if rank == 0: atomic_json(args.output, result)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
