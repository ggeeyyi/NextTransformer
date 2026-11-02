"""Shared container entry point for direct Docker and Harbor scoring."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from environments.eval.common import digest, read, require, write


def main():
    logs, work = Path('/logs/verifier'), Path('/work')
    logs.mkdir(parents=True, exist_ok=True)
    require(not (logs / 'status.json').exists(), 'Use fresh logs for every attempt')
    contract = read('/inputs/contract.json')
    status = dict(state='running', contract_sha256=digest('/inputs/contract.json'))
    write(logs / 'status.json', status)
    try:
        site = dict(runtime_dir=str(work), venv_dir='/opt/venv', resources='/inputs/resources',
                    data_dir='/inputs/data', upstream_base_dir='/inputs/resources', wandb_mode='disabled')
        work.mkdir(parents=True, exist_ok=True)
        write(work / 'site.json', site)
        from nexttransformer.runtime.site import bootstrap_runtime
        bootstrap_runtime(['--site-config', str(work / 'site.json')])
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
                   FLA_DISABLE_BACKEND_DISPATCH='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',
                   NANOCHAT_DTYPE=contract['compute_dtype'])
        # This also forces the legacy model loader to fail closed on unsupported objects.
        def run(command, name, timeout=None):
            with (logs / name).open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        run([sys.executable, '-m', 'environments.eval.preflight', 'inputs'], 'inputs.log')
        shutil.copytree('/inputs/kernels', work / 'huggingface/hub', dirs_exist_ok=True)
        distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
        run([*distributed, '-m', 'environments.eval.preflight', 'gpu'], 'gpu.log', timeout=600)
        run([*distributed, '-m', 'nexttransformer.evaluate', '--site-config', str(work / 'site.json'),
             '--checkpoint', '/inputs/checkpoint', '--output', str(logs / 'evaluation.json'),
             '--eval', 'bpb,core', '--max-per-task=-1', '--micro-batch', str(contract['micro_batch']),
             '--expected-attention', contract['attention'], '--sparse-bpb', contract['sparse_bpb'],
             '--sparse-core', contract['sparse_core']], 'evaluation.log')
        run([sys.executable, '-m', 'environments.eval.validate', '--contract', '/inputs/contract.json',
             '--result', str(logs / 'evaluation.json'), '--output', str(logs / 'validation.json')], 'validation.log')
        status['state'] = 'complete'
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        write(logs / 'status.json', status)


if __name__ == '__main__':
    main()
