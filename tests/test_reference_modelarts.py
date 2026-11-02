import json
from pathlib import Path

import pytest

from launch.modelarts import build_launch

METHODS = ('residual_standard', 'hc', 'mhc', 'attnres')


def configs(tmp_path, methods=METHODS):
    paths = []
    for method in methods:
        path = tmp_path / f'{method}.json'
        path.write_text(json.dumps({'model': {'method': method}}))
        paths.append(str(path))
    return paths


@pytest.mark.parametrize('index,method', enumerate(METHODS))
def test_four_nodes_run_four_independent_eight_gpu_trainings(tmp_path, index, method):
    env = {'VC_TASK_INDEX': str(index), 'VC_WORKER_NUM': '4', 'MA_NUM_HOSTS': '4', 'MA_NUM_GPUS': '8',
           'MA_VJ_NAME': 'job7', 'NANOCHAT_RUNTIME': str(tmp_path), 'PATH': '/bin',
           'WORLD_SIZE': '32', 'RANK': str(index * 8), 'MASTER_ADDR': 'another-node',
           'PET_NNODES': '4', 'TORCHELASTIC_RUN_ID': 'outer-job'}
    args = ['--configs', *configs(tmp_path), '--micro-batches', '8', '8', '4', '4', '--wandb-mode', 'offline']
    command, child, info, dry = build_launch(args, env)
    assert not dry and info['method'] == method and info['run_name'] == f'nextt-v2-job7_{method}_a800_seed42'
    assert command[1:8] == ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--node_rank=0',
                            '--nproc_per_node=8', '-m'] and command[8:10] == ['nexttransformer.train', '--formal']
    assert command[command.index('--config') + 1].endswith(f'{method}.json')
    assert command[command.index('--output') + 1] == str(tmp_path / 'runs' / info['run_name'])
    assert command[command.index('--micro-batch') + 1] == ('8', '8', '4', '4')[index]
    assert command[command.index('--wandb-group') + 1] == 'nextt-v2-job7'
    assert child['NANOCHAT_DTYPE'] == 'bfloat16' and child['NANOCHAT_RUNTIME'] == str(tmp_path)
    assert not any(k in child for k in ('WORLD_SIZE', 'RANK', 'MASTER_ADDR', 'PET_NNODES', 'TORCHELASTIC_RUN_ID'))


def test_single_configuration_is_a_single_node_job(tmp_path):
    env = {'NANOCHAT_RUNTIME': str(tmp_path), 'MA_NUM_HOSTS': '1', 'MA_NUM_GPUS': '8'}
    command, _, info, dry = build_launch(['--configs', *configs(tmp_path, ('dense_full',)), '--run-prefix', 'd',
                                          '--micro-batch', '8', '--dry-run'], env)
    assert dry and info['modelarts_node'] == 0 and info['nodes'] == 1 and info['method'] == 'dense_full'
    assert command.count('--micro-batch') == 1 and command.count('--nproc_per_node=8') == 1


@pytest.mark.parametrize('extra,args,match', [
    ({'VC_TASK_INDEX': '4'}, [], 'node index must'),
    ({'VC_WORKER_NUM': '8'}, [], 'one per node'),
    ({'MA_NUM_GPUS': '4'}, [], 'must be 8'),
    ({'DLS_TASK_INDEX': '2'}, [], 'conflicting'),
    ({'LOCAL_RANK': '0'}, [], 'once per node'),
    ({}, ['--output', 'shared'], 'assigned per node'),
    ({}, ['--micro-batches', '8', '8'], 'one positive value'),
    ({}, ['--micro-batches', '8', '8', '8', '8', '--micro-batch', '8'], 'either'),
])
def test_rejects_invalid_topology_and_shared_destinations(tmp_path, extra, args, match):
    env = {'VC_TASK_INDEX': '0', 'MA_VJ_NAME': 'job7', 'NANOCHAT_RUNTIME': str(tmp_path), **extra}
    with pytest.raises(ValueError, match=match):
        build_launch(['--configs', *configs(tmp_path), *args], env)


def test_rejects_duplicate_methods_and_missing_index(tmp_path):
    env = {'NANOCHAT_RUNTIME': str(tmp_path)}
    with pytest.raises(ValueError, match='distinct'):
        build_launch(['--configs', *configs(tmp_path, ('hc', 'hc')), '--node-index', '0', '--run-prefix', 'x'], env)
    with pytest.raises(ValueError, match='Missing'):
        build_launch(['--configs', *configs(tmp_path), '--run-prefix', 'x'], env)
