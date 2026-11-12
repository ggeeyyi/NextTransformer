"""Checkpoints for nodes that do not share a filesystem.

Every node keeps a self-contained copy in checkpoint_<step>/node_<n>/: the replicated model, the optimizer
shards, RNG states and data cursors of its own ranks, and a manifest of every node's files. A node restarts
from its own disk; the restart point is the newest step whose manifest exists on every node.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import socket
import time

import torch
import torch.distributed as dist

from nexttransformer.runtime.files import atomic_json, sha256
from nexttransformer.runtime.process import rng_state, restore_rng


def topology():
    """(node, local_rank, local_world) for torchrun's contiguous per-node blocks of ranks."""
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
    if world % local_world or rank % local_world != local_rank:
        raise ValueError('ranks must be assigned to nodes in equal contiguous blocks')
    return rank//local_world, local_rank, local_world


def gather(value):
    if not dist.is_initialized():
        return [value]
    out = [None]*dist.get_world_size()
    dist.all_gather_object(out, value)
    return out


def agree(ok, device):
    """True only when every rank reports ok."""
    flag = torch.tensor(int(not ok), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return not flag.item()


def node_directory(output, step, node):
    return Path(output)/f'checkpoint_{step:08d}'/f'node_{node:03d}'


def complete_steps(output, node):
    return sorted(int(p.parent.parent.name.removeprefix('checkpoint_'))
                  for p in Path(output).glob(f'checkpoint_[0-9]*/node_{node:03d}/complete.json'))


def common_step(output):
    node, _, _ = topology()
    common = set.intersection(*(set(steps) for steps in gather(complete_steps(output, node))))
    return max(common) if common else None


def set_aside_after(output, step, node):
    """Move this node's checkpoints newer than `step`, complete or partial, out of the restart path."""
    archive = Path(output)/'abandoned'/str(time.time_ns())
    moved = []
    for path in sorted(Path(output).glob(f'checkpoint_[0-9]*/node_{node:03d}')):
        if int(path.parent.name.removeprefix('checkpoint_')) <= step:
            continue
        target = archive/path.parent.name/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append(path.parent.name)
        try:
            path.parent.rmdir()
        except OSError:
            pass
    return moved


def prune(output, node, keep):
    """Delete this node's oldest complete rolling checkpoints beyond `keep`; milestones are kept."""
    if keep < 2:
        raise ValueError('keep at least two rolling checkpoints so a common restart point survives')
    rolling = [node_directory(output, step, node) for step in complete_steps(output, node)]
    rolling = [d for d in rolling if not json.loads((d/'complete.json').read_text())['milestone']]
    for directory in rolling[:-keep]:
        shutil.rmtree(directory)
        try:
            directory.parent.rmdir()
        except OSError:
            pass


def save_checkpoint(output, model, optimizer, stream, mc, tc, state, milestone=False):
    node, local_rank, local_world = topology()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    directory = node_directory(output, state['step'], node)
    if local_rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    files = {}

    def write(name, payload):
        temporary = directory/(name+'.tmp')
        torch.save(payload, temporary)
        temporary.replace(directory/name)
        files[name] = {'bytes': (directory/name).stat().st_size, 'sha256': sha256(directory/name)}

    write(f'rank_{rank:04d}.pt', {'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'data': stream.state_dict(),
                                   'routing_rng': model.routing_rng_state()})
    if local_rank == 0:
        write('model.pt', {'model': model.state_dict(), 'model_config': asdict(mc), 'train_config': asdict(tc),
                           'state': state, 'world_size': world, 'local_world_size': local_world})
    nodes = {}
    for entry in gather({'node': node, 'host': socket.gethostname(), 'files': files}):
        nodes.setdefault(f"{entry['node']:03d}", {'host': entry['host'], 'files': {}})['files'].update(entry['files'])
    if local_rank == 0:
        atomic_json(directory/'complete.json', {'step': state['step'], 'world_size': world, 'local_world_size': local_world,
                                                'milestone': milestone, 'nodes': dict(sorted(nodes.items()))})
    if world > 1:
        dist.barrier()
    return directory


def load_checkpoint(output, step, model, optimizer, stream, device):
    node, local_rank, local_world = topology()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    directory = node_directory(output, step, node)
    try:
        manifest = json.loads((directory/'complete.json').read_text())
        files = manifest['nodes'][f'{node:03d}']['files']
        layout = (manifest['world_size'], manifest['local_world_size']) == (world, local_world)
        needed = [f'rank_{rank:04d}.pt']+(['model.pt'] if local_rank == 0 else [])
        ok = layout and all((directory/name).stat().st_size == files[name]['bytes'] and
                            sha256(directory/name) == files[name]['sha256'] for name in needed)
    except (OSError, KeyError, ValueError):
        ok = False
    if not agree(ok, device):
        raise ValueError(f'checkpoint {step} is missing, has another node/rank layout or fails its checksum on some rank')
    saved = torch.load(directory/'model.pt', map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    local = torch.load(directory/f'rank_{rank:04d}.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(local['optimizer'])
    stream.load_state_dict(local['data'])
    restore_rng(local['rng'])
    model.restore_routing_rng(local['routing_rng'])
    return saved
