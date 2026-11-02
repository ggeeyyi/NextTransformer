"""Scoring infrastructure rejects incomplete results and unsafe checkpoint payloads."""
import copy
import json
from pathlib import Path

import pytest

from environments.eval.common import digest, verify_files
from environments.eval.validate import validate
from environments.eval.compare import compare
from environments.eval.launch import command
from adapters.harbor_eval.generate import generate


def fixture_result():
    tasks = [f'task-{i}' for i in range(22)]
    contract = dict(checkpoint=dict(step=46875), native_manifest_sha256='resources',
                    protocol_hash='protocol', core_tasks=tasks)
    result = dict(step=46875, track='registered', method='dense_full', native_manifest_sha256='resources',
        bpb={s: dict(split=s, input_tokens=20971520, smoke=False, bpb=.7, protocol_hash='protocol')
             for s in ('train', 'val')},
        core=dict(full_core=True, expected_tasks=22, protocol_hash='protocol', core_metric=.5,
                  results=dict.fromkeys(tasks, .5), centered_results=dict.fromkeys(tasks, .5)))
    return result, contract


def test_complete_result_and_preregistered_comparison():
    result, contract = fixture_result()
    assert validate(result, contract)['valid']
    tolerances = dict.fromkeys(['bpb.train', 'bpb.val', 'core_metric',
                                *['core.' + x for x in contract['core_tasks']]], 0.)
    assert compare(result, result, contract, tolerances)['passed']
    changed = copy.deepcopy(result)
    changed['bpb']['val']['bpb'] += .01
    assert not compare(result, changed, contract, tolerances)['passed']
    with pytest.raises(ValueError):
        compare(result, result, contract, {})


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(step=1),
    lambda r: r.update(native_manifest_sha256='wrong'),
    lambda r: r['bpb']['train'].update(smoke=True),
    lambda r: r['bpb']['val'].update(input_tokens=1024),
    lambda r: r['bpb']['val'].update(bpb=float('nan')),
    lambda r: r['core'].update(full_core=False),
    lambda r: r['core']['results'].pop('task-0'),
    lambda r: r['core'].update(core_metric=.9),
    lambda r: r['core'].update(protocol_hash='wrong'),
])
def test_partial_or_mismatched_results_fail(mutation):
    result, contract = fixture_result()
    mutation(result)
    with pytest.raises((ValueError, KeyError)):
        validate(result, contract)


def test_artifact_hash_and_path_validation(tmp_path):
    (tmp_path / 'file').write_text('correct')
    expected = {'file': digest(tmp_path / 'file')}
    verify_files(tmp_path, expected)
    (tmp_path / 'file').write_text('changed')
    with pytest.raises(ValueError): verify_files(tmp_path, expected)
    with pytest.raises(ValueError): verify_files(tmp_path, {'../outside': '0' * 64})


def test_docker_and_harbor_share_entrypoint_and_explicit_devices(tmp_path):
    cfg = dict(image='local', checkpoint='/input/checkpoint', resources='/input/resources', data='/input/data',
               kernels='/input/kernels', contract='/input/contract.json', output='/new/output',
               gpu_devices=[str(i) for i in range(8)], timeout_seconds=100)
    cmd = command(cfg, 'sha256:' + 'a' * 64, 'test')
    assert cmd[-2:] == ['-m', 'environments.eval.run']
    assert cmd[cmd.index('--network')+1] == 'none'
    assert cmd[cmd.index('--gpus')+1] == '"device=0,1,2,3,4,5,6,7"'
    generate(cfg, tmp_path / 'task', 'sha256:' + 'a' * 64)
    service = json.loads((tmp_path / 'task/environment/docker-compose.yaml').read_text())['services']['main']
    assert service['deploy']['resources']['reservations']['devices'][0]['device_ids'] == cfg['gpu_devices']
    assert all(m['read_only'] for m in service['volumes'])
    script = (tmp_path / 'task/tests/test.sh').read_text()
    assert script.index('environments.eval.run') < script.index('printf "1')
    assert service['network_mode'] == 'none'


class Payload:
    def __reduce__(self):
        return (eval, ("__import__('os').environ.__setitem__('NEXTT_UNSAFE_PICKLE_EXECUTED', '1')",))


def test_checkpoint_loader_checks_hash_and_rejects_pickle_execution(tmp_path, monkeypatch):
    import torch
    from environments.eval.checkpoint import read_model_payload
    monkeypatch.delenv('NEXTT_UNSAFE_PICKLE_EXECUTED', raising=False)
    torch.save({'model_config': Payload()}, tmp_path / 'model.pt')
    manifest = dict(files={'model.pt': dict(bytes=(tmp_path / 'model.pt').stat().st_size,
                                          sha256=digest(tmp_path / 'model.pt'))})
    (tmp_path / 'complete.json').write_text(json.dumps(manifest))
    import pickle, os
    with pytest.raises(pickle.UnpicklingError): read_model_payload(tmp_path)
    assert 'NEXTT_UNSAFE_PICKLE_EXECUTED' not in os.environ
    # The scorer forces restricted loading even through the historical API.
    monkeypatch.setenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD', '1')
    from nexttransformer.models import load_model
    with pytest.raises(pickle.UnpicklingError): load_model(tmp_path)
    assert 'NEXTT_UNSAFE_PICKLE_EXECUTED' not in os.environ
    manifest['files']['model.pt']['sha256'] = '0' * 64
    (tmp_path / 'complete.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'): read_model_payload(tmp_path)


def test_checkpoint_loader_accepts_trainer_torch_version(tmp_path):
    import pickle
    import torch
    from environments.eval.checkpoint import read_model_payload
    torch.save({'model_config': {'method': 'dense_full'}, 'state': {'software': {'torch': torch.__version__}}},
               tmp_path / 'model.pt')
    (tmp_path / 'complete.json').write_text(json.dumps(dict(files={'model.pt': dict(
        bytes=(tmp_path / 'model.pt').stat().st_size, sha256=digest(tmp_path / 'model.pt'))})))
    with pytest.raises(pickle.UnpicklingError): torch.load(tmp_path / 'model.pt', weights_only=True)
    assert read_model_payload(tmp_path)['state']['software']['torch'] == torch.__version__


def test_arena_blocks_scoring_infrastructure():
    from nexttransformer.arena import classify
    for path in ('environments/eval/run.py', 'adapters/harbor_eval/generate.py'):
        assert classify([('M', path)])[0][0] == 'blocked'


@pytest.mark.parametrize('method', ['deltanet_hybrid', 'gdn_hybrid', 'gdn_hybrid_swa',
                                    'kda_hybrid', 'qsa_hybrid', 'moba_hybrid'])
@pytest.mark.parametrize('backends', [('kernel', 'reference'), ('reference', 'kernel')])
def test_hybrid_evaluator_selects_bpb_and_core_backends(method, backends, tmp_path, monkeypatch):
    from dataclasses import asdict
    import torch
    from nexttransformer import evaluate
    from nexttransformer.evaluation import formal
    from nexttransformer.models import build_model
    from nexttransformer.models.config import debug_config
    from nexttransformer.models.sparse import SparseMixer
    from nexttransformer.protocol.spec import Protocol
    model = build_model(debug_config(method))
    saved = dict(model_config=asdict(model.config), state=dict(step=512, data_manifest_sha256='resources'))
    monkeypatch.setattr(evaluate, 'verify_resources', lambda *args: ({}, 'resources'))
    monkeypatch.setattr(evaluate, 'initialize', lambda seed: (0, 1, torch.device('cpu')))
    monkeypatch.setattr(evaluate, 'load_model', lambda *args: (model, saved))
    monkeypatch.setattr(evaluate.RustBPETokenizer, 'from_directory', lambda path: object())
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    calls = []

    def inspect(kind, expected):
        mixers = [block.attn for block in model.transformer.h if isinstance(block.attn, SparseMixer)]
        assert bool(mixers) == (method in ('qsa_hybrid', 'moba_hybrid'))
        assert all(m.backend == expected for m in mixers)
        calls.append(kind)
        return {}

    def bpb(*args, **kwargs):
        assert kwargs['tokens'] == Protocol().final_bpb_tokens
        assert kwargs['smoke'] is True
        return inspect(kwargs['split'], backends[0])

    def core(*args, **kwargs):
        assert kwargs['smoke_max_per_task'] is None
        return inspect('core', backends[1])

    monkeypatch.setattr(formal, 'evaluate_bpb', bpb)
    monkeypatch.setattr(formal, 'evaluate_core', core)
    safe = torch.serialization.get_safe_globals()
    try:
        evaluate.main(['--checkpoint', str(tmp_path), '--resources', str(tmp_path),
                       '--output', str(tmp_path/'evaluation.json'), '--informal',
                       '--sparse-bpb', backends[0], '--sparse-core', backends[1]])
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(safe)
    assert calls == ['train', 'val', 'core']
    assert json.loads((tmp_path/'evaluation.json').read_text())['method'] == method


def test_launcher_rejects_missing_allocation_and_reused_output(tmp_path):
    from environments.eval.launch import configuration
    cfg = dict(image='candidate', gpu_devices=[str(i) for i in range(8)],
               timeout_seconds=120, output=str(tmp_path / 'new'))
    for name in ('checkpoint', 'resources', 'data', 'kernels'):
        path = tmp_path / name
        path.mkdir()
        cfg[name] = str(path)
    contract = tmp_path / 'reference.json'
    contract.write_text(json.dumps({'world_size': 8}))
    cfg['contract'] = str(contract)
    path = tmp_path / 'site.json'
    path.write_text(json.dumps(cfg))
    assert configuration(path)['gpu_devices'] == cfg['gpu_devices']
    cfg['gpu_devices'][-1] = '0'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='unique'): configuration(path)
    cfg['gpu_devices'][-1] = '7'
    Path(cfg['output']).mkdir()
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='already exists'): configuration(path)


def test_kernel_snapshot_materializes_blobs_without_credentials(tmp_path, monkeypatch):
    from environments.eval.snapshot_kernels import main
    root = tmp_path / 'hub'
    source = root / 'models--kernels-community--flash-attn3'
    (source / 'snapshots/revision').mkdir(parents=True)
    (source / 'refs').mkdir()
    (source / 'blobs').mkdir()
    (source / 'blobs/kernel').write_text('reference kernel')
    (source / 'refs/main').write_text('revision')
    (source / 'snapshots/revision/kernel.so').symlink_to('../../blobs/kernel')
    (root / 'token').write_text('must not copy')
    output = tmp_path / 'snapshot'
    monkeypatch.setattr('sys.argv', ['snapshot', '--hub-cache', str(root), '--repository',
                                   'kernels-community/flash-attn3', '--output', str(output)])
    main()
    target = output / source.name / 'snapshots/revision/kernel.so'
    assert target.read_text() == 'reference kernel' and not target.is_symlink()
    assert not (output / 'token').exists()
    with pytest.raises(ValueError): main()
