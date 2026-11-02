import json
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nexttransformer.runtime.site import resolve
from nexttransformer.models.native.config import NativeModelConfig as ModelConfig, NativeTrainConfig as TrainConfig
from nexttransformer.runtime.files import DATA_REPO, DATA_REVISION, atomic_json
from nexttransformer.runtime.process import source_manifest
from nexttransformer.runtime.tracking import RunTracker


def test_precedence_and_relative_paths(tmp_path):
    config=tmp_path/'machine.local.json'
    config.write_text(json.dumps({'runtime_dir':'file root','resources':'shared resources','gpus':4,'wandb_project':'file-project'}))
    env={'NANOCHAT_RUNTIME':str(tmp_path/'env'),'NEXTT_RESOURCES':str(tmp_path/'env-resources'),
         'GPUS':'2','WANDB_ENTITY':'my-team','PATH':'/bin'}
    values=resolve(['--site-config',str(config),'--runtime-dir',str(tmp_path/'cli root'),'--gpus','8'],env)
    assert values['NANOCHAT_RUNTIME']==str(tmp_path/'cli root')
    assert values['NEXTT_RESOURCES']==str(tmp_path/'shared resources')
    assert values['NANOCHAT_VENV']==str(tmp_path/'cli root'/'venv')
    assert values['GPUS']=='8' and values['WANDB_PROJECT']=='file-project' and values['WANDB_ENTITY']=='my-team'
    assert values['HF_HUB_CACHE']==str(tmp_path/'cli root'/'huggingface'/'hub')
    assert not (tmp_path/'cli root').exists()  # resolving alone does not write anything


@pytest.mark.parametrize('config', [{'wandb_api_key':'never-store-this'}, {'gpus':0}, {'wandb_mode':'invalid'}, {'wandb_metrics':'invalid'}, {'resources':False}])
def test_invalid_site_config(tmp_path,config):
    path=tmp_path/'site.json';path.write_text(json.dumps({'runtime_dir':str(tmp_path),**config}))
    with pytest.raises(ValueError): resolve(['--site-config',str(path)],{})


def test_runtime_is_explicit_and_caches_are_overridable(tmp_path):
    with pytest.raises(ValueError,match='runtime-dir'): resolve([],{})
    env={'NANOCHAT_RUNTIME':str(tmp_path),'TIKTOKEN_CACHE_DIR':str(tmp_path/'token-cache'),'HF_HOME':str(tmp_path/'hf')}
    values=resolve([],env)
    assert values['TIKTOKEN_CACHE_DIR']==env['TIKTOKEN_CACHE_DIR']
    assert values['HF_HUB_CACHE']==str(tmp_path/'hf'/'hub')
    assert values['WANDB_MODE']=='disabled'
    assert values['NEXTT_WANDB_METRICS']=='core'


def test_source_identity_ignores_machine_local_configuration():
    root=Path(__file__).resolve().parents[1]
    path=root/'configs'/'test-identity.local.json'
    assert not path.exists()
    original=source_manifest()['sha256']
    try:
        path.write_text('{"runtime_dir":"/some/other/machine"}')
        assert source_manifest()['sha256']==original
    finally:
        path.unlink(missing_ok=True)


def test_source_metadata_works_with_standard_hf_layout(tmp_path,monkeypatch):
    from nexttransformer.runtime.resources import source_manifest_path
    import huggingface_hub
    raw=tmp_path/'hf-download';raw.mkdir()
    resources=tmp_path/'resources'
    class Api:
        def dataset_info(self, repo, revision, files_metadata):
            assert repo==DATA_REPO and revision==DATA_REVISION and files_metadata
            return SimpleNamespace(sha=revision,siblings=[SimpleNamespace(rfilename='shard_00000.parquet',size=3,lfs=SimpleNamespace(sha256='a'*64))])
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    path=source_manifest_path(raw,resources)
    assert json.loads(path.read_text())['files'][0]['lfs']['sha256']=='a'*64
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:pytest.fail('cached metadata must not contact Hub'))
    assert source_manifest_path(raw,resources)==path
    with pytest.raises(FileNotFoundError): source_manifest_path(raw,resources,tmp_path/'missing.json')


def test_core_archive_is_checked_before_extraction(tmp_path):
    from nexttransformer.runtime.resources import prepare_core
    archive=tmp_path/'bad.zip';archive.write_bytes(b'wrong bundle')
    with pytest.raises(ValueError,match='SHA-256'): prepare_core(tmp_path/'resources',archive)
    assert not (tmp_path/'resources'/'eval_bundle').exists()


def tracker_args(tmp_path,mode='offline',resume=None):
    return SimpleNamespace(output=tmp_path,wandb_mode=mode,wandb_project='test-project',wandb_entity=None,
                           wandb_dir=None,wandb_group='group',run_name='test',resume=resume)


@pytest.fixture
def fake_wandb(monkeypatch):
    runs=[]
    class Run:
        def __init__(self,options):
            self.options=options; self.id=options['id'] or f'run-{len(runs)}'
            self.project=options['project']; self.entity=options['entity']; self.events=[];self.exit_code=None
        def define_metric(self,*a,**k): pass
        def log(self,event): self.events.append(event)
        def finish(self,exit_code): self.exit_code=exit_code
    def init(**options):
        # The SDK must not perturb training RNG even if its initialization uses it.
        random.random();np.random.rand();torch.rand(1)
        run=Run(options);runs.append(run);return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=lambda **kw:kw))
    return runs


def test_wandb_only_rank_zero_and_rng_is_preserved(tmp_path,fake_wandb):
    args=tracker_args(tmp_path)
    state={'step':0,'data_manifest_sha256':'abc'}
    for rank,mode in ((1,'offline'),(0,'disabled')):
        args.wandb_mode=mode
        with RunTracker(args,rank,ModelConfig(),TrainConfig(),state,'source') as run:
            run.log({'step':1,'event':'train','loss':2})
    assert not fake_wandb
    args.wandb_mode='offline';torch.manual_seed(42);expected=torch.rand(1);torch.manual_seed(42)
    with RunTracker(args,0,ModelConfig(),TrainConfig(),state,'source') as run:
        assert torch.equal(torch.rand(1),expected)
        run.log({'step':2,'event':'train','loss':2,'active_ratios':[1,.75]})
    assert fake_wandb[0].events==[{'step':2,'train/loss':2}]
    assert fake_wandb[0].exit_code==0


@pytest.mark.parametrize('mode,step,same', [('online',2,True),('online',1,False),('offline',2,False)])
def test_wandb_resume_policy(tmp_path,fake_wandb,mode,step,same):
    atomic_json(tmp_path/'tracking.json',{'id':'previous','project':'test-project','entity':None,'mode':mode,'last_step':2,'group':'group'})
    with RunTracker(tracker_args(tmp_path,mode,tmp_path/'checkpoint'),0,ModelConfig(),TrainConfig(),
                    {'step':step,'data_manifest_sha256':'abc'},'source'): pass
    opts=fake_wandb[0].options
    assert (opts['id']=='previous') == same
    assert opts['resume']==('must' if same else 'never')
    assert opts['config']['parent_run_id']==(None if same else 'previous')


def test_online_resume_rejects_destination_change(tmp_path,fake_wandb):
    atomic_json(tmp_path/'tracking.json',{'id':'previous','project':'other-project','entity':None,'mode':'online','last_step':2})
    with pytest.raises(ValueError,match='project/entity'):
        RunTracker(tracker_args(tmp_path,'online',tmp_path/'checkpoint'),0,ModelConfig(),TrainConfig(),{'step':2},'source')
    assert not fake_wandb
