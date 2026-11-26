"""Bounded real-resource checks: data cursor, upstream BPB and CORE smoke.

Paths are explicit CLI inputs. A tiny width model retains the real 32K tokenizer
and 2K context; this checks interfaces, never model quality or full evaluation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nexttransformer.runtime.data import BestFitStream,verify_resources,tokenizer_fingerprint
from nanochat.tokenizer import RustBPETokenizer
from nanochat.loss_eval import evaluate_bpb as upstream_bpb
from nexttransformer.models.config import METHODS, debug_config
from nexttransformer.protocol.spec import Protocol
from nexttransformer.models.lm import build_model
from nexttransformer.evaluation.context import add_core_max_tokens
from nexttransformer.evaluation.formal import evaluate_bpb, evaluate_core, EvalRows
from nexttransformer.checks.validate import source_hash


def checksum(x,y):
    return hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_core_max_tokens(p)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--resources",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    p.add_argument("--core-methods",nargs="*",choices=METHODS,default=["dense","huginn","qsa","kda"])
    p.add_argument("--core-max-per-task",type=int,default=32,
                   help="smoke pool must exceed the largest CORE few-shot count (10)")
    args=p.parse_args();source=source_hash()
    if args.core_max_per_task<=10:raise ValueError("CORE smoke pool must contain more than 10 examples for few-shot sampling")
    os.environ["NANOCHAT_BASE_DIR"]=str(args.resources.resolve())
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    if args.device=="cuda":torch.cuda.set_device(local)
    device=torch.device("cuda",local) if args.device=="cuda" else torch.device("cpu")
    if world>1:dist.init_process_group("nccl" if args.device=="cuda" else "gloo")
    torch.set_num_threads(2)
    provenance,manifest=verify_resources(args.resources,args.data_dir)
    tok=RustBPETokenizer.from_directory(args.resources/"tokenizer")
    if tok.get_vocab_size()!=Protocol().vocab_size:raise ValueError("tokenizer vocabulary changed")
    a=BestFitStream(args.data_dir,tok,"train",rank,8)
    b=BestFitStream(args.data_dir,tok,"train",rank,8)
    ax,ay=a.batch(4,device)
    bx1,by1=b.batch(2,device);state=b.state_dict();bx2,by2=b.batch(2,device)
    if not torch.equal(ax,torch.cat((bx1,bx2))) or not torch.equal(ay,torch.cat((by1,by2))):
        raise RuntimeError("packing changed with microbatch size")
    b.load_state_dict(state);rx,ry=b.batch(2,device)
    if not torch.equal(rx,bx2) or not torch.equal(ry,by2):raise RuntimeError("data resume changed rows")
    c=replace(debug_config("dense",sequence_len=2048),vocab_size=32768)
    m=build_model(c,device).eval()
    wrapped=evaluate_bpb(m,tok,args.data_dir,args.resources,batch=2,rank=rank,world=world,
                         device=device,tokens=8192*world,smoke=True)
    stream=BestFitStream(args.data_dir,tok,"val",rank,world,2048)
    byte_table=torch.load(args.resources/"tokenizer/token_bytes.pt",map_location=device,weights_only=True)
    expected=upstream_bpb(EvalRows(m),stream.batches(2,device),2,byte_table)
    if wrapped["bpb"]!=expected:raise RuntimeError("BPB wrapper differs from the pinned evaluator")
    del m
    core=[]
    def save(status):
        if rank!=0:return
        out=dict(status=status,source_hash=source,protocol_hash=Protocol().fingerprint,
                 data_manifest_sha256=manifest,tokenizer_sha256=tokenizer_fingerprint(args.resources),
                 data_rows_sha256=checksum(ax,ay),microbatch_data_equal=True,resume_rows_equal=True,
                 bpb_upstream_exact=True,bpb_smoke=wrapped,core_smoke=core,world_size=world,
                 formal_evaluation=False,torch=torch.__version__)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,indent=2)+"\n")
    save("running")
    for method in args.core_methods:
        c=replace(debug_config(method,sequence_len=2048),vocab_size=32768)
        m=build_model(c,device).eval()
        result=evaluate_core(m,tok,args.resources,device,smoke_max_per_task=args.core_max_per_task,
                             core_max_tokens=args.core_max_tokens)
        core.append(dict(method=method,status="passed",full_core=False,result=result))
        save("running")
        if rank==0:print(json.dumps(dict(method=method,status="core_smoke_passed",max_per_task=args.core_max_per_task)),flush=True)
        del m
        if device.type=="cuda":torch.cuda.empty_cache()
    if source_hash()!=source:raise RuntimeError("source changed during integration check")
    save("passed")
    if rank==0:print(json.dumps(dict(status="passed",bpb_upstream_exact=True,core_methods=len(core))),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=="__main__":main()
