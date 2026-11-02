"""Capture an operator-identified reference; never invent a reference from a build.

Run with the reference interpreter and its existing, verified kernel cache. Inputs
and the cache must be immutable during capture and subsequent evaluation.
"""
import argparse
import importlib.metadata
import os
import platform
import re
import sys
import subprocess
from pathlib import Path
from environments.eval.common import digest, file_map, read, require, write


def runtime():
    import torch
    return dict(python=platform.python_version(),
                packages=dict(sorted((d.metadata['Name'], d.version)
                                     for d in importlib.metadata.distributions())),
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'resources', 'data', 'kernel-cache', 'run-evidence', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attention', choices=('fa3', 'sdpa', 'sdpa-noflash'), required=True)
    p.add_argument('--sparse-bpb', choices=('reference', 'kernel'), default='reference')
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--source-commit', required=True, help='Reviewed training source commit')
    args = p.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite a reference manifest')
    require(re.fullmatch(r'[0-9a-f]{40}', args.source_commit) is not None, 'Use a full reviewed source commit hash')
    require(args.micro_batch > 0, 'micro-batch must be positive')
    from nexttransformer.runtime.data import verify_resources
    from nexttransformer.evaluation.metrics import verify_core
    from nexttransformer.protocol.spec import Protocol
    import yaml
    _, native_hash = verify_resources(args.resources, args.data)
    verify_core(args.resources)
    complete = read(args.checkpoint / 'complete.json')
    model_hash = digest(args.checkpoint / 'model.pt')
    require(model_hash == complete['files']['model.pt']['sha256'], 'Checkpoint checksum mismatch')
    tasks = [x['label'] for x in yaml.safe_load((args.resources / 'eval_bundle/core.yaml').read_text())['icl_tasks']]
    require(len(tasks) == len(set(tasks)) == 22, 'Expected 22 unique CORE tasks')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_CACHE=str(args.kernel_cache.resolve()))
    import torch
    from nanochat import flash_attention
    from nanochat.common import COMPUTE_DTYPE
    if args.attention == 'sdpa-noflash':
        torch.backends.cuda.enable_flash_sdp(False)
    actual = 'fa3' if flash_attention.USE_FA3 else 'sdpa' if torch.backends.cuda.flash_sdp_enabled() else 'sdpa-noflash'
    require(actual == args.attention, 'Reference attention dispatch differs from declared backend')
    require(COMPUTE_DTYPE == torch.bfloat16, 'Initial scorer supports the registered BF16 reference only')
    kernels = file_map(args.kernel_cache)
    require(args.attention != 'fa3' or bool(kernels), 'FA3 requires captured kernel artifacts')
    # Resource symlinks are hashed by content. Dataset files have a separate map.
    resources = file_map(args.resources, [args.resources / 'native_manifest.json',
        args.resources / 'core_manifest.json', *(args.resources / 'tokenizer').rglob('*'),
        *(args.resources / 'eval_bundle').rglob('*')])
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'],
                         text=True, capture_output=True, check=True).stdout.strip().splitlines()
    require(bool(gpu), 'Capture on the reference GPU host')
    write(args.output, dict(schema=1, status='captured_reference_not_release',
        source_commit=args.source_commit, python_executable_sha256=digest(sys.executable), run_evidence_sha256=digest(args.run_evidence),
        runtime=runtime(), gpu_classes=sorted(set(line.strip() for line in gpu)),
        compute_dtype='bfloat16', attention=args.attention, sparse_bpb=args.sparse_bpb, sparse_core='reference',
        micro_batch=args.micro_batch, world_size=8, protocol_hash=Protocol().fingerprint,
        checkpoint=dict(step=complete['step'], model_sha256=model_hash,
                        complete_sha256=digest(args.checkpoint / 'complete.json')),
        native_manifest_sha256=native_hash, core_tasks=tasks,
        resources=resources, data=file_map(args.data, args.data.glob('*.parquet')), kernels=kernels,
        environment=dict(FLA_DISABLE_BACKEND_DISPATCH=os.environ.get('FLA_DISABLE_BACKEND_DISPATCH'))))


if __name__ == '__main__':
    main()
