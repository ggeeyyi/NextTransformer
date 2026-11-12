"""Scaling-stage configuration, the weight-decay transfer rule, and the trainer on nodes without shared storage."""
import json
import os
from pathlib import Path
import socket

import pytest
import torch
import torch.multiprocessing as mp

from nexttransformer.models.native.config import load_config as load_native
from nexttransformer.models.native.model import NativeExperiment, theoretical_training_flops
from nexttransformer.native.optimizer import build_optimizer as native_optimizer
from nexttransformer.scaling import train
from nexttransformer.scaling.checkpoint import prune, set_aside_after
from nexttransformer.scaling.config import ScalingTrainConfig, load_config
from nexttransformer.scaling.optimizer import build_optimizer, muon_weight_decay, scaling_params
from tests.test_data_layout import Tokenizer, shards

ROOT = Path(__file__).resolve().parents[1]


def test_dense_200b_plan(capsys):
    mc, tc = load_config(ROOT/'configs/scaling/dense_200b.json')
    assert (tc.steps, tc.accumulation(mc, 32, 2), tc.batch_scale) == (95367, 8, 2.0)
    assert abs(tc.total_tokens-200e9) < tc.global_batch_tokens
    assert tc.core_steps() == [20000, 40000, 60000, 80000, 95367] and len(tc.validation_steps()) == 97
    assert tc.is_milestone(95367)
    with torch.device('meta'): model = NativeExperiment(mc)
    d, layers, t = mc.width, mc.dense_layers, mc.sequence_len
    # Value-embedding gates sit on the 20 alternating layers that carry value embeddings.
    assert sum(p.numel() for p in model.transformer.h.parameters()) == 12*d*d*layers+20*12*mc.heads
    assert scaling_params(model) == 3_229_618_880
    assert muon_weight_decay(model, tc) == pytest.approx(0.28*2*110100912/3_229_618_880)
    assert theoretical_training_flops(mc) == pytest.approx(72*d*d*layers+12*t*d*layers+6*d*mc.vocab_size, rel=1e-5)
    train.main(['--config', str(ROOT/'configs/scaling/dense_200b.json'), '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert (plan['steps'], plan['accumulation'], plan['muon_lr']) == (95367, 8, .04)


def test_weight_decay_rule_reproduces_the_native_d24_groups():
    mc, native = load_native(ROOT/'configs/native/dense.json')
    tc = ScalingTrainConfig(total_tokens=native.total_tokens, global_batch_tokens=native.global_batch_tokens)
    with torch.device('meta'): model = NativeExperiment(mc)
    ours, theirs = build_optimizer(model, tc).param_groups, native_optimizer(model, native).param_groups
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert [id(p) for p in a['params']] == [id(p) for p in b['params']]
        for key in ('kind', 'lr', 'betas', 'weight_decay', 'initial_weight_decay'):
            assert a.get(key) == pytest.approx(b.get(key), rel=1e-12), key


@pytest.mark.parametrize('change', [dict(total_tokens=2_097_153), dict(protocol=' '), dict(milestone_every=1500),
                                    dict(muon_lr=0.), dict(warmdown_ratio=1.5)])
def test_invalid_training_configuration(change):
    with pytest.raises(ValueError):
        ScalingTrainConfig(**change)


def _checkpoint(root, step, node, milestone=False):
    directory = root/f'checkpoint_{step:08d}'/f'node_{node:03d}'
    directory.mkdir(parents=True)
    (directory/'complete.json').write_text(json.dumps({'step': step, 'milestone': milestone}))


def test_prune_keeps_the_newest_rolling_checkpoints_milestones_and_partial_saves(tmp_path):
    for step, milestone in ((1, False), (2, True), (3, False), (4, False), (5, False)):
        _checkpoint(tmp_path, step, 0, milestone)
    (tmp_path/'checkpoint_00000006'/'node_000').mkdir(parents=True)
    prune(tmp_path, 0, 2)
    assert sorted(p.parent.name for p in tmp_path.glob('checkpoint_*/node_000')) == \
        ['checkpoint_00000002', 'checkpoint_00000004', 'checkpoint_00000005', 'checkpoint_00000006']
    with pytest.raises(ValueError):
        prune(tmp_path, 0, 1)


def test_set_aside_moves_only_newer_checkpoints_of_this_node(tmp_path):
    for step in (1, 2, 3):
        _checkpoint(tmp_path, step, 0); _checkpoint(tmp_path, step, 1)
    assert set_aside_after(tmp_path, 1, 0) == ['checkpoint_00000002', 'checkpoint_00000003']
    assert [p.parent.name for p in tmp_path.glob('checkpoint_*/node_000')] == ['checkpoint_00000001']
    assert len(list(tmp_path.glob('checkpoint_*/node_001'))) == 3
    assert len(list(tmp_path.glob('abandoned/*/checkpoint_*/node_000'))) == 2


TINY = {'model': dict(architecture='dense', vocab_size=64, sequence_len=16, width=32, heads=2, mlp_width=128,
                      dense_layers=2, window_pattern='L'),
        'train': dict(protocol='test', total_tokens=8*128, global_batch_tokens=128, val_every=2, val_tokens=64,
                      final_bpb_tokens=64, checkpoint_every=2, milestone_every=4)}


def _rank(rank, port, root, run, argv):
    node = rank//2
    os.environ.update(RANK=str(rank), WORLD_SIZE='4', LOCAL_RANK=str(rank % 2), LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(1)
    from nexttransformer.scaling import train
    train.verify_resources = lambda resources, data_dir: (dict(data_revision='r', data_inventory_sha256='i',
                                                               validation_sha256='v', tokenizer_sha256='t'), 'manifest')
    train.RustBPETokenizer.from_directory = staticmethod(lambda path: Tokenizer())
    train.evaluate_bpb = lambda *a, **k: dict(val_bpb=1.0, forward_flops_estimate=0.0, seconds=0.0)
    train.evaluate_core = lambda *a, **k: dict(core_metric=0.0)
    # Each node writes only its own directory, as on machines without a shared filesystem.
    train.main(['--config', str(root/'config.json'), '--output', str(root/run/f'node{node}'), '--runtime-dir',
                str(root/'runtime'), '--resources', str(root), '--data-dir', str(root), '--micro-batch', '1',
                '--log-every', '1', '--wandb-mode', 'disabled', '--smoke-steps', '6', *argv])


def _run(root, run, *argv):
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    mp.start_processes(_rank, args=(port, root, run, argv), nprocs=4, start_method='spawn')


def _losses(root, run):
    lines = (root/run/'node0'/'metrics.jsonl').read_text().splitlines()
    return {e['step']: e['loss'] for e in map(json.loads, lines) if e['event'] == 'train'}


@pytest.mark.slow
def test_two_nodes_without_shared_storage_resume_bitwise(tmp_path, monkeypatch):
    """Two nodes of two ranks with separate run directories. A run stopped after update 3 and resumed, and a run
    whose second node lost its newest checkpoint, both equal the continuous run bitwise; so do runs whose gradients
    are not bucket views or whose block activations are recomputed."""
    # Spawned ranks import the attention module, which probes CUDA, before _rank runs.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    shards(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps(TINY))
    _run(tmp_path, 'continuous')
    _run(tmp_path, 'segmented', '--stop-after', '3')
    assert json.loads((tmp_path/'segmented/node1/status.node001.json').read_text())['phase'] == 'stopped'
    _run(tmp_path, 'segmented', '--resume', 'auto')
    _run(tmp_path, 'partial', '--stop-after', '3')
    (tmp_path/'partial/node1/checkpoint_00000003/node_001/complete.json').unlink()
    _run(tmp_path, 'partial', '--resume', 'auto')
    _run(tmp_path, 'buffers', '--no-bucket-view')
    _run(tmp_path, 'recompute', '--recompute-layers', '-1')
    expected = _losses(tmp_path, 'continuous')
    assert sorted(expected) == [1, 2, 3, 4, 5, 6]
    for run in ('segmented', 'partial', 'buffers', 'recompute'):
        assert _losses(tmp_path, run) == expected, run
    for node, other in ((0, 1), (1, 0)):
        directory = tmp_path/f'segmented/node{node}'
        assert not list(directory.glob(f'checkpoint_*/node_{other:03d}'))
        manifest = json.loads((directory/f'checkpoint_00000006/node_{node:03d}/complete.json').read_text())
        assert sorted(manifest['nodes']) == ['000', '001']
        assert sorted(manifest['nodes'][f'{node:03d}']['files']) == ['model.pt', f'rank_{2*node:04d}.pt', f'rank_{2*node+1:04d}.pt']
        assert (directory/'resume_00000003.json').exists()
        assert json.loads((directory/f'status.node{node:03d}.json').read_text())['phase'] == 'smoke_complete'
    assert (tmp_path/'partial/node0/resume_00000002.json').exists()
    assert list((tmp_path/'partial/node0/abandoned').glob('*/checkpoint_00000003/node_000'))
    weights = lambda run: torch.load(tmp_path/run/'node1/checkpoint_00000006/node_001/model.pt', weights_only=False)['model']
    a, b = weights('continuous'), weights('segmented')
    for key in a: assert torch.equal(a[key], b[key]), key
