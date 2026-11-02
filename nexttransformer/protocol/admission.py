"""Fixed event, data, evaluation and cumulative-budget adapters for collaborators.

No new tokenizer, packing algorithm, BPB definition or CORE task list is added.
These adapters serve the reference trainers and bounded integration checks;
the historical four-model trainer retains its own entry point.
"""
from dataclasses import asdict,dataclass,field
import json
import math
import os
from pathlib import Path
import time

import torch

from nexttransformer.runtime.data import BestFitStream,verify_resources
from nexttransformer.protocol.spec import Protocol


def review_config(path):
    """(config, proof, findings) of a configuration and the cost proof its training charges.

    Raises when the proof cannot describe this configuration: training charges its per-stage
    costs, so model, protocol, stages and totals must agree. Findings are evidence that is stale
    or outside the shared budget: changed source, another PyTorch build, a different cost anchor,
    or a cost beyond the preflight tolerance. Formal admission refuses any finding; research
    training records them and cannot become a result."""
    from nexttransformer.protocol.spec import load_config
    from nexttransformer.protocol.compute import FORMULA_VERSION
    from nexttransformer.models.config import candidate_config
    path=Path(path);config=load_config(path);raw=json.loads(path.read_text())
    calibration=raw["calibration"];findings=[]
    proof=json.loads((path.parent/calibration["cost_record"]).read_text())
    p=Protocol()
    if proof["model_hash"]!=config.fingerprint or proof["protocol_hash"]!=p.fingerprint or proof["unknown"]:
        raise ValueError("cost proof does not cover this model/protocol")
    if calibration.get("status")!="theoretical_preflight_passed":findings.append("configuration has not passed cost preflight")
    if not source_matches(proof,config) or calibration["formula_version"]!=FORMULA_VERSION:
        findings.append("cost calibration is stale; the source this method runs changed after its audit")
    if calibration.get("torch")!=torch.__version__:
        findings.append("PyTorch version differs from cost evidence; re-audit in the target environment")
    if (proof.get("method_source") or {}).get("hash")!=calibration.get("method_source_hash") or proof.get("torch")!=torch.__version__:
        findings.append("cost proof source/software mismatch")
    baseline=json.loads((path.parent/calibration["cost_record"]).with_name("dense.json").read_text())
    anchor=candidate_config("dense")
    if (baseline["method"]!="dense" or baseline["model_hash"]!=anchor.fingerprint or not source_matches(baseline,anchor)
        or baseline["flops_per_token"]!=calibration["dense_flops_per_token"]):
        findings.append("Dense cost anchor mismatch")
    if sum(s["optimizer_steps"] for s in proof["stages"])!=p.steps:raise ValueError("cost proof misses training stages")
    weighted=sum(s["optimizer_steps"]*s["cost"]["total"] for s in proof["stages"])
    if weighted!=proof["total_training_flops"]:raise ValueError("stage-weighted cost mismatch")
    per_token=weighted/p.total_tokens
    if not math.isclose(per_token,calibration["flops_per_token"],rel_tol=1e-12):raise ValueError("configuration cost mismatch")
    error=per_token/calibration["dense_flops_per_token"]-1
    if abs(error)>p.preflight_tolerance:
        findings.append(f"theoretical training budget mismatch ({error:+.3%} versus the {p.preflight_tolerance:.0%} preflight)")
    return config,proof,findings


def source_matches(proof,config):
    """Whether the per-method source fingerprint recorded by `proof` matches the current source."""
    from nexttransformer.protocol.fingerprint import method_source
    source=proof.get("method_source")
    return bool(source) and source["hash"]==method_source(config.method,source["files"],config)


def admit_config(path):
    """Fail closed on stale/incomplete theory evidence; not a full-run verdict."""
    config,proof,findings=review_config(path)
    if findings:raise ValueError(findings[0])
    return config,proof


def events_at(step):
    p=Protocol()
    if not 0<=step<=p.steps:raise ValueError("step outside the common training budget")
    return dict(val=step in p.validation_steps,core=step in p.core_steps,final=step==p.steps)


def data_layout(rank,world):
    """(stream rank, stream world, shares) of the frozen eight-stream data layout.

    A multiple of eight ranks (two H20 nodes: 16) runs each of the eight
    BestFitStreams on `shares` consecutive ranks; see share_batches. Other world
    sizes keep one stream per rank (smoke runs only)."""
    p=Protocol()
    if world%p.data_parallel_size:return rank,world,1
    shares=world//p.data_parallel_size
    return rank//shares,p.data_parallel_size,shares


def share_batches(stream,batch,period,share,shares,device):
    """`batch`-row pieces of rows [share*n,(share+1)*n) of the stream's next `period` rows.

    Every copy of a stream packs all `period` rows, so once the generator is
    exhausted the stream state equals an unshared stream's, and the union over
    shares in rank order is the eight-rank global row order."""
    own,rest=divmod(period,shares)
    if rest or own%batch:raise ValueError("rank share must contain complete batches")
    cpu=torch.device("cpu")
    if share:stream.batch(share*own,cpu)
    for _ in range(own//batch):yield stream.batch(batch,device)
    if share<shares-1:stream.batch((shares-1-share)*own,cpu)


@dataclass
class BudgetState:
    protocol_hash:str=field(default_factory=lambda:Protocol().fingerprint)
    optimizer_step:int=0
    input_tokens:int=0
    theory_flops:int=0
    val_steps:list=field(default_factory=list)
    core_steps:list=field(default_factory=list)

    def validate(self):
        p=Protocol()
        if self.protocol_hash!=p.fingerprint:raise ValueError("protocol changed on resume")
        if self.input_tokens!=self.optimizer_step*p.global_batch_tokens:raise ValueError("token/step cursor mismatch")
        if not 0<=self.optimizer_step<=p.steps or self.theory_flops<0:raise ValueError("invalid cumulative state")
        for label,actual,allowed in (("val",self.val_steps,p.validation_steps),("core",self.core_steps,p.core_steps)):
            if actual!=sorted(set(actual)) or any(s>self.optimizer_step or s not in allowed for s in actual):
                raise ValueError(label+" event cursor mismatch")

    def advance(self,input_tokens,theory_flops):
        self.validate();p=Protocol()
        if input_tokens!=p.global_batch_tokens or self.optimizer_step>=p.steps or theory_flops<=0:
            raise ValueError("an update must consume exactly the common global batch and a positive audited cost")
        self.optimizer_step+=1;self.input_tokens+=input_tokens;self.theory_flops+=int(theory_flops)

    def record_eval(self,kind):
        self.validate()
        if kind not in ("val","core") or not events_at(self.optimizer_step)[kind]:
            raise ValueError("evaluation is not scheduled at this optimizer step")
        history=self.val_steps if kind=="val" else self.core_steps
        if self.optimizer_step in history:raise ValueError("evaluation already recorded")
        history.append(self.optimizer_step)

    def state_dict(self):self.validate();return asdict(self)

    @classmethod
    def from_state_dict(cls,state):
        result=cls(**state);result.validate();return result

    def final_status(self,dense_flops_per_token):
        self.validate();p=Protocol()
        if self.input_tokens!=p.total_tokens:return "incomplete_training"
        if self.val_steps!=list(p.validation_steps) or self.core_steps!=list(p.core_steps):return "incomplete_eval"
        error=abs(self.theory_flops/(p.total_tokens*dense_flops_per_token)-1)
        return "matched" if error<=p.cumulative_tolerance else "budget_mismatch"
