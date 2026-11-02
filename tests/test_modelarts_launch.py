from pathlib import Path

import pytest

from launch.modelarts_native import MODELS, build_launch


@pytest.mark.parametrize('index,model', enumerate(MODELS))
def test_modelarts_assigns_distinct_local_eight_gpu_runs(tmp_path,index,model):
    env={'VC_TASK_INDEX':str(index),'VC_WORKER_NUM':'4','MA_NUM_HOSTS':'4','MA_NUM_GPUS':'8',
         'MA_VJ_NAME':'job123','NANOCHAT_RUNTIME':str(tmp_path),'PATH':'/bin',
         'WORLD_SIZE':'32','RANK':str(index*8),'MASTER_ADDR':'another-node',
         'PET_NNODES':'4','PET_NODE_RANK':str(index),'TORCHELASTIC_RUN_ID':'outer-job'}
    command,child,info=build_launch(['--dry-run','--wandb-mode','offline'],env)
    assert Path(command[1]).name=='native_a800.sh' and command[2]==model
    assert command[command.index('--gpus')+1]=='8'
    assert command[command.index('--output')+1]==str(tmp_path/'runs'/info['run_name'])
    assert info['run_name']==f'native-core393-core8k-job123_{model}_a800_seed42'
    assert '--dry-run' in command and child['PATH']=='/bin'
    assert not any(k in child for k in ('WORLD_SIZE','RANK','MASTER_ADDR','PET_NNODES','PET_NODE_RANK','TORCHELASTIC_RUN_ID'))


@pytest.mark.parametrize('extra,match', [
    ({'VC_TASK_INDEX':'4'},'node index must'),
    ({'VC_WORKER_NUM':'8'},'must be 4'),
    ({'MA_NUM_GPUS':'4'},'must be 8'),
    ({'DLS_TASK_INDEX':'2'},'conflicting'),
    ({'LOCAL_RANK':'0'},'once per node'),
])
def test_modelarts_rejects_invalid_topology(tmp_path,extra,match):
    env={'VC_TASK_INDEX':'0','MA_VJ_NAME':'job123','NANOCHAT_RUNTIME':str(tmp_path),**extra}
    with pytest.raises(ValueError,match=match):build_launch([],env)


def test_modelarts_explicit_index_and_group_and_legacy_index(tmp_path):
    env={'NANOCHAT_RUNTIME':str(tmp_path)}
    with pytest.raises(ValueError,match='Missing'):build_launch(['--run-prefix','test'],env)
    with pytest.raises(ValueError,match='run-prefix'):build_launch(['--node-index','0'],env)
    args=['--node-index','2','--run-prefix','test','--wandb-group','custom','--hardware','h200']
    command,_,info=build_launch(args,env)
    assert info['model']=='loopformer' and info['run_name']=='test_loopformer_h200_seed42'
    assert command.count('--wandb-group')==1 and command[command.index('--wandb-group')+1]=='custom'
    assert build_launch(['--run-prefix','test'],{**env,'DLS_TASK_INDEX':'3'})[2]['model']=='circle'
    with pytest.raises(ValueError,match='single-model'):
        build_launch(args+['--output','same-output-for-every-node'],env)
