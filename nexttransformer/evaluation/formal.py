"""Formal BPB and CORE under the frozen protocol, for any registered model."""
import os
from pathlib import Path
import time

import torch

from nexttransformer.evaluation.core import score_core
from nanochat.loss_eval import evaluate_bpb as upstream_bpb
from nexttransformer.evaluation.metrics import verify_core
from nexttransformer.protocol.admission import data_layout, share_batches
from nexttransformer.protocol.spec import Protocol
from nexttransformer.runtime.data import BestFitStream, verify_resources


class EvalRows(torch.nn.Module):
    """Stable stream-row state noise independent of evaluation microbatch size."""
    def __init__(self,model,offset=0):
        super().__init__();self.model=model;self.offset=offset;self.input_tokens=0
        self.config=model.config;self.max_seq_len=model.config.sequence_len

    def get_device(self):return self.model.get_device()

    def forward(self,idx,*args,**kwargs):
        self.model.set_row_offset(self.offset);self.offset+=idx.shape[0]
        self.input_tokens+=idx.numel()
        return self.model(idx,*args,**kwargs)


@torch.no_grad()
def evaluate_bpb(model,tokenizer,data_dir,resources,*,batch,rank,world,device,
                 tokens=None,split="val",smoke=False):
    p=Protocol();tokens=p.val_tokens if tokens is None else tokens
    if not smoke and (model.config.scale!="formal" or world%p.data_parallel_size or tokens not in (p.val_tokens,p.final_bpb_tokens)):
        raise ValueError("formal BPB must use the frozen distributed/token protocol")
    denominator=batch*world*model.config.sequence_len
    if tokens%denominator:raise ValueError("evaluation budget must contain complete batches")
    verify_resources(resources,data_dir)
    stream_rank,stream_world,shares=data_layout(rank,world)
    stream=BestFitStream(data_dir,tokenizer,split,stream_rank,stream_world,model.config.sequence_len)
    batches=share_batches(stream,batch,tokens//model.config.sequence_len//stream_world,rank%shares,shares,device)
    byte_table=torch.load(Path(resources)/"tokenizer/token_bytes.pt",map_location=device,weights_only=True)
    wrapper=EvalRows(model,rank*(tokens//model.config.sequence_len//world))
    was_training=model.training;old_offset=model.row_offset;model.eval();started=time.monotonic()
    try:
        value=upstream_bpb(wrapper,batches,tokens//denominator,byte_table)
        if wrapper.input_tokens*world!=tokens:raise RuntimeError("BPB input budget drift")
        return dict(bpb=value,input_tokens=tokens,split=split,seconds=time.monotonic()-started,
                    protocol_hash=p.fingerprint,smoke=smoke)
    finally:model.train(was_training);model.set_row_offset(old_offset)


@torch.no_grad()
def evaluate_core(model,tokenizer,resources,device,*,smoke_max_per_task=None,core_max_tokens=None):
    if smoke_max_per_task is None and model.config.scale!="formal":
        raise ValueError("debug architecture requires an explicitly labelled CORE smoke limit")
    verify_core(resources)
    if Path(os.environ["NANOCHAT_BASE_DIR"]).resolve()!=Path(resources).resolve():
        raise ValueError("NANOCHAT_BASE_DIR must identify the verified CORE resources")
    was_training=model.training;offset=model.row_offset;model.eval()
    try:
        result=score_core(model, EvalRows, tokenizer, device,
                          max_per_task=-1 if smoke_max_per_task is None else smoke_max_per_task,
                          max_tokens=core_max_tokens)
        return dict(**result,full_core=smoke_max_per_task is None,expected_tasks=22,
                    protocol_hash=Protocol().fingerprint)
    finally:model.train(was_training);model.set_row_offset(offset)
