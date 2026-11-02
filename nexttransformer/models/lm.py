"""Native nanochat shell with isolated, inspectable architecture adaptations."""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE
from nexttransformer.models.backbone import GPT, GPTConfig, Linear, norm
from nexttransformer.models.config import HYBRID, LINEAR, LOOPED, MLA, RESIDUAL, SPARSE, ModelConfig
from nexttransformer.models.plugins import check_shell, load as load_plugin, reset as reset_plugin
from nexttransformer.models.registry import PLUGIN_METHODS
from nexttransformer.protocol.spec import Protocol
from nexttransformer.models.linear import LinearMixer, init_gdn_gate
from nexttransformer.models.others import MLAAttention
from nexttransformer.models.residual import HyperConnection, ManifoldHyperConnection, depth_attention
from nexttransformer.models.sparse import BACKENDS as SPARSE_BACKENDS, SparseMixer
from nexttransformer.models.looped import (DeepLoopBlock, SandwichBlock, act_weights, exit_distribution,
                            huginn_depth, ouro_gate_loss, sinusoidal_coordinates)

# Native learned scalars that a method's own residual path replaces; kept frozen and unused.
# DeepLoop's post-norm scaled recurrence replaces residual/x0 mixing; the Residual family
# also replaces backout.
FROZEN_NATIVE = {"deeploop": ("resid_lambdas", "x0_lambdas"),
                 **{method: ("resid_lambdas", "x0_lambdas", "backout_lambda") for method in RESIDUAL}}


class FlexibleMLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c_fc=Linear(c.width,c.mlp_width,bias=False)
        self.c_proj=Linear(c.mlp_width,c.width,bias=False)

    def forward(self,x): return self.c_proj(F.relu(self.c_fc(x)).square())


class ReferenceLM(GPT):
    def __init__(self, config: ModelConfig):
        native=GPTConfig(sequence_len=config.sequence_len,vocab_size=config.vocab_size,
                         n_layer=config.n_layer,n_embd=config.width,n_head=config.heads,
                         n_kv_head=config.heads,window_pattern=config.window_pattern)
        super().__init__(native)
        self.config=config
        if not config.value_embeddings:self.value_embeds=nn.ModuleDict()
        self.optimizer_step=0
        self.last_trace=[]
        self.last_stats={}
        self.last_aux_loss=None
        self.last_lm_loss=None
        self.row_offset=0
        self.plugin=load_plugin(config.method) if config.method in PLUGIN_METHODS else None
        for i,block in enumerate(self.transformer.h):
            if config.mlp_width != 4*config.width: block.mlp=FlexibleMLP(config)
            if config.method in LINEAR: block.attn=LinearMixer(config,i)
            elif config.method in SPARSE: block.attn=SparseMixer(config,i)
            # Hybrid attention layers keep the native B01 attention built by GPT.
            elif config.method in HYBRID and config.layer_mixers[i] in LINEAR:
                block.attn=LinearMixer(config,i,kind=config.layer_mixers[i])
            elif config.method in HYBRID and config.layer_mixers[i] in SPARSE:
                block.attn=SparseMixer(config,i,kind=config.layer_mixers[i])
            elif config.method in MLA or (config.method in HYBRID and config.layer_mixers[i] in MLA):
                block.attn=MLAAttention(config,i)
            if self.plugin and self.plugin.mixer: block.attn=self.plugin.mixer(config,i)
            if self.plugin and self.plugin.ffn: block.mlp=self.plugin.ffn(config,i)
            if not config.value_embeddings:block.attn.ve_gate=None
            if config.method=="deeploop":
                self.transformer.h[i]=DeepLoopBlock(block,config.maximum_depth)
            elif config.method=="huginn":
                self.transformer.h[i]=SandwichBlock(block)
            if self.plugin and self.plugin.block: self.transformer.h[i]=self.plugin.block(block,config,i)
        if self.plugin: check_shell(self)
        if config.method in ("ut_act","ouro"):
            self.halter=nn.Linear(config.width,1)
        if config.method=="huginn":
            self.state_adapter=Linear(2*config.width,config.width,bias=False)
        if config.method=="hc":
            self.residual_maps=nn.ModuleList(
                HyperConnection(config.width,config.residual_streams,k,config.residual_gate_init)
                for k in range(2*config.n_layer))
        elif config.method=="mhc":
            self.residual_maps=nn.ModuleList(
                ManifoldHyperConnection(config.width,config.residual_streams,k,config.residual_gate_init,
                                        config.sinkhorn_iterations)
                for k in range(2*config.n_layer))
        elif config.method=="attnres":
            # One pseudo-query per attention and FFN sublayer, plus one for the final readout.
            self.depth_queries=nn.Parameter(torch.empty(2*config.n_layer+1,config.width))
        for name in self._frozen_native():
            getattr(self,name).requires_grad_(False)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        c=self.config
        if hasattr(self,"halter"):
            nn.init.normal_(self.halter.weight,std=c.width**-.5)
            nn.init.constant_(self.halter.bias,-1.)
        if c.method=="deeploop":
            # Official DeepLoop (lszshu/DeepLoop model/init_utils.py + gpt_base.py): every hidden matrix gets the
            # spectral std 0.5/sqrt(n_in)*min(1,sqrt(n_out/n_in)); residual output projections are further divided
            # by the physical depth K; then LoopDeepNorm multiplies V/O/FFN matrices by beta (Q/K keep the base).
            K=len(self.transformer.h)
            for block in self.transformer.h:
                for proj,residual,gain in ((block.attn.c_q,False,1.),(block.attn.c_k,False,1.),
                                           (block.attn.c_v,False,block.beta),(block.attn.c_proj,True,block.beta),
                                           (block.mlp.c_fc,False,block.beta),(block.mlp.c_proj,True,block.beta)):
                    n_out,n_in=proj.weight.shape
                    std=(.5/math.sqrt(n_in))*min(1.,math.sqrt(n_out/n_in))
                    if residual:std/=K
                    nn.init.normal_(proj.weight,mean=0.,std=std*gain)
            # Native Value Embeddings are an additional V path absent from the
            # original block. Scale their initial values too so they do not
            # bypass the beta-scaled V projection; this is a native adaptation.
            for embedding in self.value_embeds.values():
                embedding.weight.mul_((8*c.maximum_depth)**-.5)
        for module in getattr(self,"residual_maps",()):
            module.reset_parameters()
        if hasattr(self,"depth_queries"):
            nn.init.zeros_(self.depth_queries)  # AttnRes requires uniform initial depth weights.
        for block in self.transformer.h:
            if isinstance(block.attn,MLAAttention):block.attn.init_weights()
        if self.plugin and self.plugin.init: self.plugin.init(self)

    def _frozen_native(self):
        """Native parameters that the method's own path replaces; they stay frozen and unused."""
        return tuple(self.plugin.frozen or ()) if self.plugin else FROZEN_NATIVE.get(self.config.method,())

    def set_step(self,step):
        if not 0 <= step <= Protocol().steps: raise ValueError("optimizer step outside protocol")
        self.optimizer_step=int(step)
        for block in self.transformer.h:
            if isinstance(block.attn,SparseMixer):block.attn.in_burnin=int(step)<self.config.burnin_steps
        if self.plugin:
            for module in self.modules():
                if module is not self and callable(getattr(module,"set_step",None)):module.set_step(int(step))

    def set_sparse_backend(self,backend):
        """Execution path of the sparse mixers ("reference" | "kernel"); both compute the same function."""
        if backend not in SPARSE_BACKENDS: raise ValueError(f"unknown sparse backend {backend!r}")
        for block in self.transformer.h:
            if isinstance(block.attn,SparseMixer):block.attn.backend=backend

    def set_row_offset(self,offset):
        """Stable global row id, supplied by the data/evaluation adapter.

        Huginn state noise must not restart at each accumulation microbatch.
        Offset counts rows from the beginning of this train or eval stream.
        """
        if offset<0:raise ValueError("row offset must be nonnegative")
        self.row_offset=int(offset)

    def routing_rng_state(self):
        return dict(policy="step-keyed-depth/row-keyed-state-v1",seed=self.config.routing_seed,
                    optimizer_step=self.optimizer_step,row_offset=self.row_offset)

    def restore_routing_rng(self,state):
        if state["policy"]!="step-keyed-depth/row-keyed-state-v1" or state["seed"]!=self.config.routing_seed:
            raise ValueError("reference routing RNG policy changed")
        self.set_step(state["optimizer_step"]);self.set_row_offset(state["row_offset"])

    def set_training_stage(self,step):
        """Call before optimizer grouping; preserve existing state on a stage change.

        No implicit requires_grad toggles in forward, which would break DDP.
        """
        self.set_step(step)
        gate_only=self.config.method=="ouro" and step>=Protocol().steps-self.config.ouro_gate_steps
        frozen=self._frozen_native()
        for name,p in self.named_parameters():
            if name in frozen:
                continue
            p.requires_grad_(not gate_only or name.startswith("halter."))
        return "gate_only" if gate_only else "joint"

    def _record(self,kind,**fields):
        self.last_trace.append(dict(kind=kind,backward=torch.is_grad_enabled() and self.training,**fields))

    def _embed_reference(self,idx):
        x=norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if x.shape[1]>1:
            gate=self.smear_lambda.to(x.dtype)*self.smear_gate(x[:,1:,:24]).sigmoid()
            x=torch.cat((x[:,:1],x[:,1:]+gate*x[:,:-1]),1)
        self._record("embedding",tokens=idx.numel())
        return x

    def _window(self,virtual,total):
        c=self.config
        if c.method in HYBRID:
            # Physical == virtual (no loops). Full layers use B01's (T, 0); a sliding
            # window of w keys including the current token is the left window w-1.
            if c.layer_mixers[virtual]=="swa":return (c.sliding_window-1,0)
            return (c.sequence_len,0)
        short=-(-c.sequence_len//4//128)*128
        long=virtual==total-1 or c.window_pattern[virtual%len(c.window_pattern)]=="L"
        return (c.sequence_len if long else short,0)

    def _block(self,x,x0,idx,physical,virtual,total,alive=None):
        c=self.config
        mixed=x if c.method=="deeploop" else self.resid_lambdas[physical]*x+self.x0_lambdas[physical]*x0
        ve=self.value_embeds[str(physical)](idx).to(x.dtype) if str(physical) in self.value_embeds else None
        window=self._window(virtual,total)
        out=self.transformer.h[physical](mixed,ve,(self.cos[:,:idx.shape[1]],self.sin[:,:idx.shape[1]]),window,None)
        if alive is not None: out=torch.where(alive[...,None],out,x)
        kind=c.layer_mixers[physical] if c.method in HYBRID else c.method
        self._record("block",physical=physical,virtual=virtual,window=window[0],
                     tokens=idx.numel(),sequence_len=idx.shape[1],ve=ve is not None,
                     mixer=kind if kind in LINEAR|SPARSE|MLA|PLUGIN_METHODS else "softmax")
        return out

    def _readout(self,x,backout):
        if backout is not None: x=x-self.backout_lambda.to(x.dtype)*backout
        logits=self.lm_head(norm(x))[...,:self.config.vocab_size].float()
        self._record("head",tokens=x.shape[0]*x.shape[1])
        return 15*torch.tanh(logits/15)

    @staticmethod
    def _token_loss(logits,targets):
        return F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-1,reduction="none").reshape_as(targets)

    def _ordinary_hidden(self,idx):
        c=self.config
        x=self._embed_reference(idx); x0=x; cached=None
        if c.method in LOOPED:
            ids=[*range(c.pre_layers),*list(range(c.pre_layers,c.pre_layers+c.core_layers))*c.loops,
                 *range(c.pre_layers+c.core_layers,c.n_layer)]
        else: ids=list(range(c.n_layer))
        for virtual,physical in enumerate(ids):
            x=self._block(x,x0,idx,physical,virtual,len(ids))
            if virtual==len(ids)//2:cached=x
        return x,cached

    def _attention_branch(self,x,idx,physical):
        """Pre-norm attention output of a physical layer; the caller owns the residual update."""
        c=self.config;t=idx.shape[1]
        ve=self.value_embeds[str(physical)](idx).to(x.dtype) if str(physical) in self.value_embeds else None
        window=self._window(physical,c.n_layer)
        out=self.transformer.h[physical].attn(norm(x),ve,(self.cos[:,:t],self.sin[:,:t]),window,None)
        self._record("block",physical=physical,virtual=physical,window=window[0],tokens=idx.numel(),
                     sequence_len=t,ve=ve is not None,mixer="softmax",residual=c.method)
        return out

    def _residual_hidden(self,idx):
        """Residual-family trunk without native residual/x0 scalars or backout; returns the readout input."""
        c=self.config
        x=self._embed_reference(idx)
        mlp=lambda h,physical:self.transformer.h[physical].mlp(norm(h))
        if c.method=="residual_standard":
            for i in range(c.n_layer):
                x=x+self._attention_branch(x,idx,i)
                x=x+mlp(x,i)
            return x
        if c.method in ("hc","mhc"):
            # The embedding is replicated into n streams; the final streams are summed.
            streams=x[:,:,None].expand(-1,-1,c.residual_streams,-1)
            for k,hyper in enumerate(self.residual_maps):
                h,mixed,weights=hyper.read(streams)
                out=self._attention_branch(h,idx,k//2) if k%2==0 else mlp(h,k//2)
                streams=hyper.write(mixed,out,weights)
                self._record("residual_map",sublayer=k,streams=c.residual_streams,tokens=idx.numel())
            return streams.sum(2)
        # Block AttnRes: sources are the embedding b0, completed group sums and the current partial sum.
        starts=set(c.attnres_group_starts)
        blocks,keys,partial=[],[],x
        for k in range(2*c.n_layer):
            key=norm(partial)
            h=depth_attention(self.depth_queries[k],blocks+[partial],keys+[key])
            self._record("depth_read",sublayer=k,sources=len(blocks)+1,tokens=idx.numel())
            if k%2==0 and k//2 in starts:
                blocks.append(partial);keys.append(key);partial=None
            out=self._attention_branch(h,idx,k//2) if k%2==0 else mlp(h,k//2)
            partial=out if partial is None else partial+out
        h=depth_attention(self.depth_queries[-1],blocks+[partial],keys+[norm(partial)])
        self._record("depth_read",sublayer=2*c.n_layer,sources=len(blocks)+1,tokens=idx.numel())
        return h

    def _loop_hidden(self,idx,targets=None):
        c=self.config
        is_gate_stage=c.method=="ouro" and self.optimizer_step>=Protocol().steps-c.ouro_gate_steps
        guard=torch.no_grad if is_gate_stage else nullcontext
        with guard():
            x=self._embed_reference(idx); x0=x
            for i in range(c.pre_layers):x=self._block(x,x0,idx,i,i,c.maximum_depth)
        prelude=x
        n=c.max_loops
        if c.method=="huginn":
            n=getattr(self,"accounting_depth",None) or (huginn_depth(c,self.optimizer_step) if self.training else c.eval_loops)
            noise=[]
            for row in range(idx.shape[0]):
                generator=torch.Generator(device=idx.device).manual_seed(c.routing_seed+self.row_offset+row)
                noise.append(torch.randn((1,*x.shape[1:]),device=x.device,dtype=x.dtype,generator=generator))
            x=torch.cat(noise,0)*c.width**-.5
        total=c.pre_layers+n*c.core_layers+c.post_layers
        alive=torch.ones(idx.shape,dtype=torch.bool,device=idx.device)
        cumulative=x.new_zeros(idx.shape,dtype=torch.float32)
        remainder=torch.zeros_like(cumulative); updates=torch.zeros_like(cumulative)
        weighted=torch.zeros_like(x)
        gates=[]; logits_list=[]; losses=[]; masses=[]
        for r in range(n):
            no_grad=is_gate_stage or (c.method=="huginn" and r<n-c.huginn_backprop_loops)
            with torch.no_grad() if no_grad else nullcontext():
                if c.method=="ut_act":
                    coordinates=sinusoidal_coordinates(idx.shape[1],c.width,r+1,x.device,x.dtype)
                    x=torch.where(alive[...,None],x+coordinates,x)
                if c.method=="huginn":
                    if r==n-c.huginn_backprop_loops:x=x.detach()
                    x=norm(self.state_adapter(torch.cat((x,prelude),-1)))
                    self._record("state_adapter",tokens=idx.numel())
                if c.method=="ut_act":
                    p=self.halter(norm(x.float())).squeeze(-1).sigmoid()
                    weight,cumulative,next_alive,rem=act_weights(p,cumulative,alive,r==n-1,c.act_epsilon)
                    remainder=remainder+rem; updates=updates+alive
                    self._record("act_gate",tokens=idx.numel())
                for j in range(c.core_layers):
                    x=self._block(x,x0,idx,c.pre_layers+j,c.pre_layers+r*c.core_layers+j,total,
                                  alive=alive if c.method=="ut_act" else None)
                if c.method=="ut_act":
                    weighted=weighted+weight[...,None].to(x.dtype)*x
                    masses.append(weight); alive=next_alive
            if c.method=="ouro":
                with guard():
                    y=x
                    for j in range(c.post_layers):
                        y=self._block(y,x0,idx,c.pre_layers+c.core_layers+j,
                                      c.pre_layers+(r+1)*c.core_layers+j,total)
                    logits=self._readout(y,prelude)
                    logits_list.append(logits)
                    if targets is not None:losses.append(self._token_loss(logits,targets))
                if r<n-1:
                    # Ouro: lambda_t = sigmoid(Linear(h^(t))), h^(t) = final-layer hidden state at loop t, i.e. the
                    # input of the LM head (here: after the coda of this loop), not the pre-coda core output.
                    gates.append(self.halter(norm(y.detach().float() if is_gate_stage else y.float())).squeeze(-1))
                    self._record("ouro_gate",tokens=idx.numel())
        if c.method=="ouro":
            distribution=exit_distribution(gates)
            self.last_stats.update(exit_distribution=distribution.detach(),stage="gate_only" if is_gate_stage else "joint")
            if not self.training:
                selected=(distribution.cumsum(-1)<c.ouro_exit_quantile).sum(-1).clamp_max(n-1)
                stack=torch.stack(logits_list,-2)
                logits=stack.gather(-2,selected[...,None,None].expand(*selected.shape,1,c.vocab_size)).squeeze(-2)
                return logits,x.new_zeros(()),None
            if targets is None:return logits_list[-1],x.new_zeros(()),None
            mask=(targets!=-1).to(distribution.dtype)
            denominator=mask.sum().clamp_min(1)
            if is_gate_stage:
                auxiliary=(ouro_gate_loss(gates,losses)*mask[...,None]).sum()/(denominator*n)
                token_loss=losses[-1].detach()
                objective=auxiliary
            else:
                token_loss=(distribution*torch.stack(losses,-1)).sum(-1)
                entropy=-(distribution*distribution.clamp_min(1e-12).log()).sum(-1)
                auxiliary=-c.ouro_entropy_weight*(entropy*mask).sum()/denominator
                objective=(token_loss*mask).sum()/denominator+auxiliary
            return logits_list[-1],auxiliary,(token_loss,objective)
        if c.method=="ut_act":
            x=weighted
            auxiliary=c.ponder_weight*(updates+remainder).mean()
            self.last_stats.update(act_weights=torch.stack(masses,-1).detach(),
                                   mean_loops=float(updates.mean().detach()))
        else:
            auxiliary=x.new_zeros(())
            self.last_stats.update(sampled_loops=n,backprop_loops=min(n,c.huginn_backprop_loops))
        for j in range(c.post_layers):
            x=self._block(x,x0,idx,c.pre_layers+c.core_layers+j,c.pre_layers+n*c.core_layers+j,total)
        return self._readout(x,prelude),auxiliary,None

    def forward(self,idx,targets=None,kv_cache=None,loss_reduction="mean"):
        c=self.config
        if idx.ndim!=2 or not 0<idx.shape[1]<=c.sequence_len:
            raise ValueError("reference requires packed rows with 0 < T <= context")
        if kv_cache is not None:
            raise NotImplementedError("reference focuses on full-sequence train/eval")
        self.last_trace=[];self.last_stats={};special=None
        if c.method=="dense":
            result=super().forward(idx,targets,kv_cache,loss_reduction)
            self.last_stats={"native_dense_delegate":True}
            return result
        if c.method in ("ut_act","huginn","ouro"):
            logits,auxiliary,special=self._loop_hidden(idx,targets)
        elif c.method in RESIDUAL:
            logits=self._readout(self._residual_hidden(idx),None)
            auxiliary=logits.new_zeros(())
        else:
            x,cached=self._ordinary_hidden(idx)
            logits=self._readout(x,cached)
            if self.plugin:
                auxiliary=sum((m.last_aux for m in self.modules() if m is not self and torch.is_tensor(getattr(m,"last_aux",None))),
                              logits.new_zeros(()))
            else:
                auxiliary=sum((block.attn.last_aux for block in self.transformer.h
                               if isinstance(block.attn,(LinearMixer,SparseMixer))),logits.new_zeros(()))
        self.last_aux_loss=auxiliary
        if targets is None:return logits
        token_loss=self._token_loss(logits,targets)
        mask=(targets!=-1).to(token_loss.dtype)
        self.last_lm_loss=(token_loss*mask).sum()/mask.sum().clamp_min(1)
        # BPB always measures ordinary token CE, never auxiliary training loss.
        if loss_reduction=="none":return token_loss.flatten()
        if loss_reduction=="sum":return token_loss.sum()
        if loss_reduction!="mean":raise ValueError("invalid loss reduction")
        if self.training and special is not None:return special[1]
        return self.last_lm_loss+(auxiliary if self.training else 0)


def build_model(config,device="cpu",seed=42):
    """Meta construction avoids allocating a full model before device selection."""
    torch.manual_seed(seed)
    plugin=load_plugin(config.method) if config.method in PLUGIN_METHODS else None
    with torch.device("meta"):model=(plugin.model if plugin and plugin.model else ReferenceLM)(config)
    if str(device)=="meta":return model
    model.to_empty(device=device)
    # to_empty invalidates constructor-initialized auxiliary parameters too.
    for module in model.modules():
        if isinstance(module,nn.Linear):module.reset_parameters()
        elif isinstance(module,nn.LayerNorm):module.reset_parameters()
        elif hasattr(module,"weight") and module.__class__.__name__=="CausalDepthwise":
            nn.init.normal_(module.weight,std=module.size**-.5)
    for module in model.modules():
        if isinstance(module,LinearMixer):
            if hasattr(module,"output_gain"):nn.init.ones_(module.output_gain)
            if module.kind=="gdn":init_gdn_gate(module.log_decay,module.dt_bias)
            elif hasattr(module,"log_decay"):nn.init.zeros_(module.log_decay);nn.init.constant_(module.dt_bias,-3.)
        if isinstance(module,SparseMixer) and hasattr(module,"compress_position"):
            nn.init.zeros_(module.compress_position)
    if plugin:reset_plugin(model)
    model.init_weights()
    return model
