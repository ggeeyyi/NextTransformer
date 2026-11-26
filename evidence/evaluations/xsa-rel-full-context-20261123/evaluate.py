"""Re-evaluate final XSA/REL checkpoints with the pinned full CORE prompts.

Run with torchrun and the repository runtime arguments. Results are supplementary
checkpoint evaluations, never a replacement for the six-event training record.
"""
from nexttransformer.runtime.site import bootstrap_runtime
if __name__ == "__main__":
    bootstrap_runtime()

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist

from nanochat.base_eval import evaluate_core as upstream_core
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.context import core_context, validate_submission
from nexttransformer.evaluation.core import CoreInputs, inference_view
from nexttransformer.evaluation.formal import EvalRows
from nexttransformer.evaluation.metrics import verify_core
from nexttransformer.models import load_model
from nexttransformer.protocol.spec import Protocol
from nexttransformer.runtime.data import verify_resources
from nexttransformer.runtime.files import atomic_json, sha256
from nexttransformer.runtime.process import initialize
from nexttransformer.runtime.site import add_runtime_arguments
from rel_full_context import enable_full_context


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-max-tokens", type=int, choices=[2048])
    args = parser.parse_args()
    rank, world, device = initialize(42)
    _, manifest = verify_resources(args.resources, args.data_dir)
    pin = verify_core(args.resources)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    model, saved = load_model(args.checkpoint, device)
    assert saved["state"]["data_manifest_sha256"] == manifest
    assert saved["state"]["step"] == 46875 and model.config.sequence_len == 2048
    model.eval()
    is_rel = model.config.method == "inkling_rel"
    if is_rel and args.core_max_tokens is None:
        enable_full_context(model)
    tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")
    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    scorer = CoreInputs(EvalRows(inference_view(model, args.core_max_tokens)), args.core_max_tokens)
    started = time.monotonic()
    result = upstream_core(scorer, tokenizer, device, max_per_task=-1)
    longest = torch.tensor(scorer.longest, device=device)
    dist.all_reduce(longest, op=dist.ReduceOp.MAX)
    context = core_context(2048, args.core_max_tokens, int(longest.item()))
    validate_submission(context)
    if rank == 0:
        source = Path(__file__).resolve()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        atomic_json(args.output, dict(
            schema="checkpoint-core-evaluation-v1", method=model.config.method, step=46875,
            evaluationSourceCommit=revision,
            evaluationSourceSha256={p.name: sha256(p) for p in (source, source.with_name("rel_full_context.py"))},
            checkpointModelSha256=json.loads((args.checkpoint / "complete.json").read_text())["files"]["model.pt"]["sha256"],
            coreArchiveSha256=pin["archive_sha256"], dataManifestSha256=manifest,
            protocolHash=Protocol().fingerprint, full_core=True, expected_tasks=22,
            coreContext=context, relativeBiasOutsideExtent="zero" if is_rel else None,
            relativeExtent=model.config.option("rel_extent") if is_rel else None,
            worldSize=world, dtype=str(next(model.parameters()).dtype),
            seconds=time.monotonic()-started, **result))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
