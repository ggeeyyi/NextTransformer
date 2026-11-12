"""Export of a finished scaling-stage run directory as a results/scaling record."""
import json
from pathlib import Path

import pytest

from nexttransformer.scaling import export
from nexttransformer.scaling.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/scaling/dense_200b.json'
TASKS = {f'task_{i}': i/100 for i in range(22)}


def write_run(root, skip_validation=None):
    mc, tc = load_config(CONFIG)
    root.mkdir()
    (root/'provenance.json').write_text(json.dumps({
        'config_sha256': config_hash(mc, tc), 'smoke': False, 'parameters': 4_990_000_000,
        'scaling_parameters': 3_229_618_880, 'world_size': 32, 'micro_batch': 2, 'gradient_accumulation': 8,
        'theoretical_GFLOPs_per_token': 21.9, 'source_sha256': 'a'*64, 'native_manifest_sha256': 'b'*64,
        'data': 'c'*64, 'torch': '2.9.1+cu128', 'cuda': '12.8', 'attention': 'fa3',
        'nodes': ['node-a'], 'device': 'accelerator', 'command': ['train']}))
    (root/'run_status.json').write_text(json.dumps({'phase': 'training_complete', 'step': tc.steps}))
    rows = [{'event': 'validation', 'step': s, 'val_bpb': 1-s/1e6} for s in tc.validation_steps() if s != skip_validation]
    rows += [{'event': 'core', 'step': s, 'core_metric': s/4e5, 'centered_results': TASKS, 'full_core': True}
             for s in tc.core_steps()]
    rows.append({'event': 'train', 'step': tc.steps, 'input_tokens': tc.total_tokens, 'mfu': .46,
                 'training_flops_estimate': 4.4e21, 'gpu_hours': 2400.})
    (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'bpb_final.json').write_text(json.dumps({'val': {'val_bpb': .8}, 'train': {'train_bpb': .79}}))
    (root/'core_final.json').write_text(json.dumps({'core_metric': tc.steps/4e5, 'centered_results': TASKS}))
    return tc


def test_export_writes_an_immutable_record_without_machine_details(tmp_path, monkeypatch):
    tc = write_run(tmp_path/'run')
    monkeypatch.setattr(export, 'RESULTS', tmp_path/'results')
    args = [str(tmp_path/'run'), '--config', str(CONFIG), '--id', 'dense_200b', '--name', 'Dense', '--family', 'dense',
            '--checkpoint', 'https://huggingface.co/org/model']
    assert export.main(args) == 0 and export.main(args) == 0
    run = json.loads((tmp_path/'results/dense_200b.json').read_text())
    assert tuple(run) == export.FIELDS and run['config'] == 'configs/scaling/dense_200b.json'
    assert run['parameters']['blocks'] == 3_229_618_880-2560*32768
    assert len(run['val']) == len(tc.validation_steps()) and run['core'][-1]['step'] == tc.steps
    assert not any(word in json.dumps(run) for word in ('node-a', 'accelerator', '"command"', '"nodes"'))
    assert export.check(tmp_path/'results') == []
    run['val'].reverse()
    (tmp_path/'results/dense_200b.json').write_text(json.dumps(run))
    assert export.check(tmp_path/'results')


@pytest.mark.parametrize('change', ['validation', 'status', 'core'])
def test_export_refuses_an_incomplete_run(tmp_path, change):
    tc = write_run(tmp_path/'run', skip_validation=1000 if change == 'validation' else None)
    if change == 'status':
        (tmp_path/'run/run_status.json').write_text(json.dumps({'phase': 'stopped', 'step': tc.steps-1}))
    if change == 'core':
        (tmp_path/'run/core_final.json').write_text(json.dumps({'core_metric': 0., 'centered_results': TASKS}))
    with pytest.raises(ValueError):
        export.record(tmp_path/'run', CONFIG, 'dense_200b', 'Dense', 'dense')
