from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from nexttransformer.runtime.tracking import RunTracker, metric_payload, metrics_schema
from nexttransformer.models.native.config import NativeModelConfig, NativeTrainConfig


COMMON = {'lm_loss','loss','input_tokens','training_flops_estimate','tokens_per_second',
          'grad_norm','lr_multiplier','gpu_hours','peak_memory_bytes'}
EXTRAS = {'dense':set(), 'loopformer':{'loop_condition_grad_norm','loop_modulation_saturated_fraction'},
    'mor':{'router_aux_loss','mean_active_depth','mean_recursions','budget_max_abs_error'},
    'circle':{'mean_ring_steps','cap_ratio','budget_target_steps','budget_error_ema','budget_weight_used','router_aux_loss','budget_shape_rmse'}}


def event():
    data={'event':'train','step':10,**{key:1. for key in COMMON|set.union(*EXTRAS.values())}}
    data.update(input_tokens=10*1048576,active_ratios=[1.]*27,exit_ratios=[0.]*26+[1.],
                budget_survival=[.75,.5,.25],budget_error=[0.,0.,0.],budget_dual=[0.,0.,0.],
                halter_delta_scale=.01,halter_depth_bias=[-2.]*18)
    return data


@pytest.mark.parametrize('arch,count',[('dense',9),('mor',13),('loopformer',11),('circle',16)])
def test_four_models_use_one_compact_native_schema(arch,count):
    c=NativeModelConfig(architecture=arch);data=event();original=deepcopy(data)
    payload=metric_payload(data,c)
    assert metrics_schema(c)=='native-v3'
    assert set(payload)=={'step'}|{'train/'+key for key in COMMON|EXTRAS[arch]}
    assert len(payload)-1==count
    assert data==original
    full=metric_payload(data,c,'full')
    assert full['train/active_ratios/27']==1.
    assert full['train/halter_depth_bias/18']==-2.
    assert full['train/budget_survival/3']==.25


@pytest.mark.parametrize('arch',['dense','mor','loopformer','circle'])
def test_same_bpb_core_and_final_bpb_names_across_models(arch):
    c=NativeModelConfig(architecture=arch)
    validation=metric_payload({'event':'validation','step':250,'val_bpb':1.2,
        'mean_ring_steps':18.,'cap_ratio':.8,'mean_recursions':2.5,'active_ratios':[1.]*20},c)
    expected={'step','validation/val_bpb'}
    if arch=='circle':expected|={'validation/mean_ring_steps','validation/cap_ratio'}
    if arch=='mor':expected.add('validation/mean_recursions')
    assert set(validation)==expected
    scores={f'task{i}':.3 for i in range(22)}
    core=metric_payload({'event':'core','step':2000,'core_metric':.2,'results':scores,
                         'full_core':False,'entries':22,'active_ratios':[1.]*20},c)
    assert len(core)==26 and core['core/core_metric']==.2
    final=metric_payload({'event':'final_bpb','step':46875,'train_bpb':1.,'val_bpb':1.1},c)
    assert final=={'step':46875,'final_bpb/train_bpb':1.,'final_bpb/val_bpb':1.1}


@pytest.mark.parametrize('arch',['dense','mor','loopformer','circle'])
def test_offline_sdk_records_native_schema_and_filtered_metrics(tmp_path,arch):
    pytest.importorskip('wandb')
    directory=tmp_path/arch;directory.mkdir()
    args=SimpleNamespace(output=directory,wandb_mode='offline',wandb_metrics='core',
        wandb_project='nextt-test',wandb_entity=None,wandb_group='native-d24-test',
        wandb_dir=directory/'wandb',run_name=arch,resume=None)
    c=NativeModelConfig(architecture=arch);data=event()
    with RunTracker(args,0,c,NativeTrainConfig(),{'step':0,'data_manifest_sha256':'test'},'test') as tracker:
        tracker.log(data)
        tracker.log({'event':'final_bpb','step':11,'train_bpb':1.,'val_bpb':1.1})
        # Real SDK summary serialization exercises filtering and metric names.
        summary=dict(tracker.run.summary)
        assert summary['train/lm_loss']==1. and summary['final_bpb/val_bpb']==1.1
        assert not any('active_ratios' in key or 'depth_bias' in key for key in summary)
        assert tracker.run.config['metrics_schema']=='native-v3'
    record=json.loads((directory/'tracking.json').read_text())
    assert record['metrics_schema']=='native-v3' and record['last_step']==11
