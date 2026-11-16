"""Enable the validated site pacing, then execute the upstream trainer in its normal module mode."""
import os
import runpy
import sys
# Both target nodes have four GPUs per CPU NUMA domain. Pin separate CPU
# sets using the live node's topology and the process's allowed affinity.
import subprocess
import re
local=int(os.environ['LOCAL_RANK'])
topo=subprocess.check_output(['/usr/local/nvidia/bin/nvidia-smi','topo','-m'],text=True)
gpu_lines=[line.split() for line in topo.splitlines() if re.match(r'^GPU[0-7]\s',line)]
cpu_ranges={int(row[0][3:]):row[-3] for row in gpu_lines}
nearby=[]
for field in cpu_ranges[local].split(','):
    ends=list(map(int,field.split('-')))
    nearby.extend(range(ends[0],ends[-1]+1))
peers=sorted(i for i,cpus in cpu_ranges.items() if cpus==cpu_ranges[local])
allowed=sorted(set(nearby)&os.sched_getaffinity(0))
affinity=allowed[peers.index(local)::len(peers)]
if not affinity:raise RuntimeError('empty rank-local CPU set')
os.sched_setaffinity(0,affinity)
args=sys.argv[1:]
if args[:1]==['-u']:args=args[1:]
if args[:2]!=['-m','nexttransformer.train']:raise RuntimeError(f'unexpected rank command {args[:2]}')
assert 'T2M_H1_GPU_LEASE' in os.environ and 'LOCAL_RANK' in os.environ
sys.argv=['nexttransformer.train',*args[2:]]
from nexttransformer.runtime.site import bootstrap_runtime
bootstrap_runtime()
from timing import install
install()
from progress import install as record
record()
from checkpoint_io import install as install_checkpoint_io
install_checkpoint_io()
from core_graph import install as install_core_graph
install_core_graph()
runpy.run_module('nexttransformer.train',run_name='__main__')
