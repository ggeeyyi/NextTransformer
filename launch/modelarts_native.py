"""One ModelArts job: four nodes, one independent eight-GPU model per node.

Run this entry once per container, without an outer torchrun or mpirun.
Only the standard library is needed before the configured venv takes over.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nexttransformer.runtime.site import resolve

MODELS = ('dense', 'mor', 'loopformer', 'circle')
RANK_KEYS = ('VC_TASK_INDEX', 'DLS_TASK_INDEX')
DISTRIBUTED_KEYS = {
    'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
    'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
}


def build_launch(argv, environ):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--node-index', type=int, help='override container index, chiefly for dry-run')
    parser.add_argument('--run-prefix', help='shared run group; defaults to native-core393-core8k-${MA_VJ_NAME}')
    parser.add_argument('--hardware', choices=('a800', 'h200'), default='a800')
    args, training_args = parser.parse_known_args(argv)
    if any(key in environ for key in ('LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK')):
        raise ValueError('Run train_modelarts.py once per node, without an outer torchrun/mpirun')
    for key in ('VC_WORKER_NUM', 'MA_NUM_HOSTS'):
        if key in environ and int(environ[key]) != 4:
            raise ValueError(f'{key} must be 4: this entry assigns four models to four nodes')
    if 'MA_NUM_GPUS' in environ and int(environ['MA_NUM_GPUS']) != 8:
        raise ValueError('MA_NUM_GPUS must be 8 per node')
    indices = [int(environ[key]) for key in RANK_KEYS if key in environ]
    if args.node_index is not None:
        index = args.node_index
    elif indices and len(set(indices)) == 1:
        index = indices[0]
    else:
        raise ValueError('Missing or conflicting ModelArts node index; set VC_TASK_INDEX or --node-index')
    if index not in range(4):
        raise ValueError('ModelArts node index must be 0, 1, 2 or 3')
    # These options must differ by node or are fixed by this four-model entry.
    for option in training_args:
        if option.split('=', 1)[0] in ('--output', '--run-name', '--config', '--resume', '--gpus', '--world-size'):
            raise ValueError(f'{option.split("=", 1)[0]} belongs to the single-model entry; use --run-prefix here')
    prefix = args.run_prefix
    if prefix is None and environ.get('MA_VJ_NAME'):
        prefix = 'native-core393-core8k-' + environ['MA_VJ_NAME']
    if not prefix or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', prefix):
        raise ValueError('Supply --run-prefix (letters, digits, dot, underscore or hyphen), or MA_VJ_NAME')
    values = resolve(training_args, environ)
    model = MODELS[index]
    run_name = f'{prefix}_{model}_{args.hardware}_seed42'
    output = str(Path(values['NANOCHAT_RUNTIME'])/'runs'/run_name)
    command = ['bash', str(ROOT/'launch'/f'native_{args.hardware}.sh'), model, *training_args,
               '--gpus', '8', '--run-name', run_name, '--output', output]
    if not any(option.split('=', 1)[0] == '--wandb-group' for option in training_args):
        command += ['--wandb-group', prefix]
    # torchrun must construct a fresh local group, independent of platform ranks.
    env = {key: value for key, value in environ.items()
           if key not in DISTRIBUTED_KEYS and not key.startswith(('PET_', 'TORCHELASTIC_'))}
    info = {'modelarts_node': index, 'model': model, 'gpus': 8,
            'run_name': run_name, 'output': output}
    return command, env, info


def main():
    try:
        command, env, info = build_launch(sys.argv[1:], os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(info), file=sys.stderr, flush=True)
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
