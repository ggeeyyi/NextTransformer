"""Verify immutable inputs before loading model code, then probe each allocated GPU."""
import argparse
import os
import subprocess
from pathlib import Path
from environments.eval.common import digest, file_map, read, require, verify_files, write
from environments.eval.capture import runtime


def inputs(contract, checkpoint, resources, data, kernels, root=Path('/opt/nextt')):
    require(contract['sparse_core'] == 'reference', 'CORE must preserve the reference sparse backend')
    require(contract['attention'] in ('fa3', 'sdpa', 'sdpa-noflash'), 'Unknown attention backend')
    require(contract['compute_dtype'] == 'bfloat16', 'Only registered BF16 scoring is supported')
    require(contract['schema'] == 1 and contract['world_size'] == 8, 'Only eight-rank scoring is supported')
    require(contract['checkpoint']['step'] == 46_875, 'Scoring requires the final checkpoint')
    require(digest(checkpoint / 'complete.json') == contract['checkpoint']['complete_sha256'], 'Completion manifest mismatch')
    require(digest(checkpoint / 'model.pt') == contract['checkpoint']['model_sha256'], 'Checkpoint model mismatch')
    verify_files(resources, contract['resources'])
    verify_files(data, contract['data'])
    require({p.name for p in data.glob('*.parquet')} == set(contract['data']), 'Unexpected data shard inventory')
    if contract['kernels']:
        verify_files(kernels, contract['kernels'])
    require(file_map(kernels) == contract['kernels'], 'Kernel cache contains missing or unexpected files')
    require(contract['attention'] != 'fa3' or bool(contract['kernels']), 'FA3 cache missing')
    from nexttransformer.runtime.data import verify_resources
    from nexttransformer.evaluation.metrics import verify_core
    _, actual = verify_resources(resources, data)
    require(actual == contract['native_manifest_sha256'], 'Native manifest mismatch')
    verify_core(resources)
    require(runtime() == contract['runtime'], 'Reference package/interpreter/CUDA inventory differs; resolve before scoring')
    require(os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') == contract['environment']['FLA_DISABLE_BACKEND_DISPATCH'] == '1',
            'FLA dispatch policy mismatch')
    inventory = read('/opt/nextt-manifest/build.json')
    verify_files(root, inventory['source_files'])
    require(inventory['source_base_revision'] == contract['source_commit'], 'Image source commit differs from reviewed contract')
    write('/logs/verifier/inputs.json', dict(verified=True, contract=contract,
          image_inventory_sha256=digest('/opt/nextt-manifest/build.json')))


def gpu(contract, output):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    require(world == contract['world_size'] == torch.cuda.device_count(), 'Allocated GPU count mismatch')
    torch.cuda.set_device(local)
    require(torch.cuda.is_bf16_supported(), 'BF16 unavailable')
    from nanochat import flash_attention as fa
    if contract['attention'] == 'sdpa-noflash':
        torch.backends.cuda.enable_flash_sdp(False)
    actual = 'fa3' if fa.USE_FA3 else 'sdpa' if torch.backends.cuda.flash_sdp_enabled() else 'sdpa-noflash'
    require(actual == contract['attention'], 'Unexpected attention fallback or backend')
    device = torch.device('cuda', local)
    props = torch.cuda.get_device_properties(local)
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0].strip()
    identity = f'{props.name}, {driver}, {props.major}.{props.minor}'
    require(identity in contract['gpu_classes'], 'GPU/driver class differs from reference')
    a = torch.ones((32, 32), device=device, dtype=torch.bfloat16)
    require(torch.equal(a @ a, torch.full_like(a, 32)), 'BF16 matrix probe failed')
    q = torch.randn((1, 32, 2, 64), device=device, dtype=torch.bfloat16)
    require(torch.isfinite(fa.flash_attn.flash_attn_func(q, q, q, causal=True)).all().item(), 'Attention probe failed')
    # Exercise the submitted method's registered execution path at bounded size.
    from dataclasses import replace
    from nexttransformer.models import build_model
    from nexttransformer.models.config import debug_config
    from environments.eval.checkpoint import read_model_payload
    saved = read_model_payload('/inputs/checkpoint')
    require('method' in saved['model_config'], 'Initial scoring path supports registered models only')
    config = debug_config(saved['model_config']['method'], width=256, sequence_len=256)
    config = replace(config, backend=saved['model_config']['backend'])
    model = build_model(config, device)
    model.eval()
    model.set_step(saved['state']['step'])
    for backend in set((contract['sparse_bpb'], contract['sparse_core'])):
        model.set_sparse_backend(backend)
        with torch.no_grad():
            result = model(torch.zeros((1, config.sequence_len), dtype=torch.long, device=device))
        logits = result['logits'] if isinstance(result, dict) else result
        require(torch.isfinite(logits).all().item(), 'Method kernel probe failed')
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    value = torch.tensor([rank + 1.], device=device)
    dist.all_reduce(value)
    require(value.item() == world * (world + 1) / 2, 'NCCL all-reduce mismatch')
    torch.cuda.synchronize()
    write(output / f'gpu-rank-{rank}.json', dict(passed=True, rank=rank, device=identity,
                                               attention=actual, method=config.method))
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('inputs', 'gpu'))
    p.add_argument('--contract', type=Path, default=Path('/inputs/contract.json'))
    args = p.parse_args()
    contract = read(args.contract)
    if args.mode == 'inputs':
        inputs(contract, Path('/inputs/checkpoint'), Path('/inputs/resources'), Path('/inputs/data'), Path('/inputs/kernels'))
    else:
        gpu(contract, Path('/logs/verifier'))


if __name__ == '__main__':
    main()
