"""One ModelArts job of admitted reference configurations: one independent eight-GPU run per node.

Pass one ``--configs`` entry per node, in node order. Node i trains configuration i with its own
standalone eight-rank ``nexttransformer.train`` group, so a four-node job with four configurations is
four separate eight-GPU experiments, not one 32-rank run. One configuration is an ordinary
single-node eight-GPU job. Run this entry once per container, without an outer torchrun or
mpirun; only the standard library is needed before the configured venv takes over. Training is
formal (``--formal``); a training option ``--research`` overrides it.
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

RANK_KEYS = ('VC_TASK_INDEX', 'DLS_TASK_INDEX')
HOST_KEYS = ('VC_WORKER_NUM', 'MA_NUM_HOSTS')
DISTRIBUTED_KEYS = {
    'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
    'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
}
PER_NODE = ('--output', '--config', '--resume', '--run-name', '--gpus', '--world-size', '--nproc-per-node')
GPUS = 8


def build_launch(argv, environ):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--configs', nargs='+', required=True, type=Path,
                        help='admitted configs/<family>/<method>.json files, one per node in node order')
    parser.add_argument('--micro-batches', nargs='+', type=int,
                        help='per-rank micro batch for each configuration (default: --micro-batch/MICRO_BATCH/site)')
    parser.add_argument('--node-index', type=int, help='override the container index, chiefly for dry runs')
    parser.add_argument('--run-prefix', help='shared run group; defaults to nextt-v2-${MA_VJ_NAME}')
    parser.add_argument('--hardware', choices=('a800', 'h200'), default='a800')
    parser.add_argument('--dry-run', action='store_true', help='print this node\'s command instead of training')
    args, training_args = parser.parse_known_args(argv)
    if any(key in environ for key in ('LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK')):
        raise ValueError('Run train_reference_modelarts.py once per node, without an outer torchrun/mpirun')
    nodes = len(args.configs)
    for key in HOST_KEYS:
        if key in environ and int(environ[key]) != nodes:
            raise ValueError(f'{key}={environ[key]} but {nodes} configuration(s) were given; use one per node')
    if 'MA_NUM_GPUS' in environ and int(environ['MA_NUM_GPUS']) != GPUS:
        raise ValueError(f'MA_NUM_GPUS must be {GPUS} per node')
    indices = [int(environ[key]) for key in RANK_KEYS if key in environ]
    if args.node_index is not None:
        index = args.node_index
    elif indices and len(set(indices)) == 1:
        index = indices[0]
    elif not indices and nodes == 1:
        index = 0
    else:
        raise ValueError('Missing or conflicting ModelArts node index; set VC_TASK_INDEX or --node-index')
    if index not in range(nodes):
        raise ValueError(f'ModelArts node index must be in 0..{nodes - 1}')
    for option in training_args:
        if option.split('=', 1)[0] in PER_NODE:
            raise ValueError(f'{option.split("=", 1)[0]} is assigned per node by this entry')
    if args.micro_batches is not None:
        if len(args.micro_batches) != nodes or min(args.micro_batches) < 1:
            raise ValueError('--micro-batches needs one positive value per configuration')
        if any(option.split('=', 1)[0] == '--micro-batch' for option in training_args):
            raise ValueError('use either --micro-batches or --micro-batch')
    configs = [path if path.is_absolute() else (ROOT / path) for path in args.configs]
    methods = [json.loads(path.read_text())['model']['method'] for path in configs]
    if len(set(methods)) != nodes:
        raise ValueError('every node needs a distinct method')
    prefix = args.run_prefix
    if prefix is None and environ.get('MA_VJ_NAME'):
        prefix = 'nextt-v2-' + environ['MA_VJ_NAME']
    if not prefix or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', prefix):
        raise ValueError('Supply --run-prefix (letters, digits, dot, underscore or hyphen), or MA_VJ_NAME')
    exports = resolve(training_args, environ)
    method = methods[index]
    run_name = f'{prefix}_{method}_{args.hardware}_seed42'
    output = str(Path(exports['NANOCHAT_RUNTIME']) / 'runs' / run_name)
    command = [str(Path(exports['NANOCHAT_VENV']) / 'bin' / 'python'), '-m', 'torch.distributed.run',
               '--standalone', '--nnodes=1', '--node_rank=0', f'--nproc_per_node={GPUS}',
               '-m', 'nexttransformer.train', '--formal', '--config', str(configs[index]), '--output', output,
               *training_args]
    if args.micro_batches is not None:
        command += ['--micro-batch', str(args.micro_batches[index])]
    if not any(option.split('=', 1)[0] == '--wandb-group' for option in training_args):
        command += ['--wandb-group', prefix]
    # torchrun must construct a fresh local group, independent of platform ranks.
    env = {key: value for key, value in environ.items()
           if key not in DISTRIBUTED_KEYS and not key.startswith(('PET_', 'TORCHELASTIC_'))}
    env.update(exports)
    env['NANOCHAT_DTYPE'] = 'bfloat16'
    info = {'modelarts_node': index, 'nodes': nodes, 'method': method, 'config': str(configs[index]),
            'gpus': GPUS, 'run_name': run_name, 'output': output}
    return command, env, info, args.dry_run


def main():
    try:
        command, env, info, dry_run = build_launch(sys.argv[1:], os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(info), file=sys.stderr, flush=True)
    if dry_run:
        print(' '.join(command))
        return
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
