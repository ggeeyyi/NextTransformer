"""Model configuration, method families, and labelled candidate/debug configurations."""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path

from nexttransformer.models.registry import METHODS, PLUGINS, members

LOOPED = members("looped")
SPARSE = members("sparse")
LINEAR = members("linear")
RESIDUAL = members("residual")
# v2 hybrids (docs/plans/V2.md §3): a registered mixer kind per physical layer.
HYBRID = members("hybrid")
MLA = frozenset(("mla_nope",))
MLA_CONFIG = MLA | {"kda_hybrid"}
NO_VE = frozenset(("kda_hybrid", "kda_no_ve", "mla_nope", "mha_no_ve"))
# Plugin methods are new stacks and follow the same registered v2 calibration grid.
V2_CALIBRATION = HYBRID | NO_VE | frozenset(PLUGINS)
# v2 rows with full causal MHA in every layer: the Dense baseline and the Residual family.
FULL_ATTENTION = frozenset(("dense_full", *RESIDUAL, *MLA, "mha_no_ve"))
# Fields read only by some methods. Every other method keeps the default and omits them
# from its serialized configuration and fingerprint, so earlier configuration hashes hold.
SCOPED_FIELDS = {"residual_streams": ("hc", "mhc"), "residual_gate_init": ("hc", "mhc"),
                 "sinkhorn_iterations": ("mhc",), "attnres_groups": ("attnres",),
                 "sliding_window": ("gdn_hybrid_swa",),
                 **{name: tuple(sorted(MLA_CONFIG)) for name in
                    ("q_rank", "kv_rank", "qk_nope_dim", "qk_rope_dim", "v_head_dim")}}
# Registered chunked training backends (name -> chunk length) and their methods.
# The chunk algorithm itself is audited; see COSTS.md "Chunked linear backends".
CHUNK_BACKENDS = {"fla_chunk64": 64}
CHUNKED_METHODS = frozenset(("deltanet", "kda", "kda_no_ve", "retnet", "elu_linear", "elu_norm", "elu_qknorm", "gdn", *(HYBRID-{"moba_hybrid"})))
# Layer-count multiple fixed by each hybrid's or plugin's repeating unit; calibration steps by it.
LAYER_MULTIPLE = {"deltanet_hybrid": 2, "gdn_hybrid": 4, "gdn_hybrid_swa": 2, "kda_hybrid": 4,
                  "qsa_hybrid": 4, "moba_hybrid": 1,
                  **{name: spec.get("layer_multiple", 1) for name, spec in PLUGINS.items()}}


def frozen_options(options):
    """Plugin options as sorted (name, value) pairs, lists as tuples, so a configuration stays hashable."""
    freeze = lambda value: tuple(freeze(v) for v in value) if isinstance(value, (list, tuple)) else value
    pairs = tuple(sorted((str(k), freeze(v)) for k, v in (options.items() if isinstance(options, dict) else options)))
    if len({k for k, _ in pairs}) != len(pairs): raise ValueError("an option is given twice")
    return pairs


def hybrid_layout(method, layers):
    """Mixer kind of each physical layer, 0-based.

    deltanet_hybrid (H01): full attention at one-based 2 and N/2+1, DeltaNet elsewhere.
    gdn_hybrid (H02, Qwen3-Next 3:1): GDN, GDN, GDN, attention, repeated.
    kda_hybrid (H04): KDA, KDA, KDA, MLA-NoPE, repeated; VE disabled throughout.
    gdn_hybrid_swa (H03, Gated DeltaNet H1, official lit_gpt gated_delta_per_layer=2):
    GDN at even 0-based positions, sliding-window attention at odd ones.
    "attention" is B01's full causal MHA; "swa" is the same attention with a window.
    """
    if layers % LAYER_MULTIPLE[method]:
        raise ValueError(f"{method} needs a layer count that is a multiple of {LAYER_MULTIPLE[method]}")
    if method == "deltanet_hybrid":
        if layers < 4:
            raise ValueError("deltanet_hybrid needs at least four layers for two distinct full-attention positions")
        return tuple("attention" if i in (1, layers//2) else "deltanet" for i in range(layers))
    if method == "gdn_hybrid":
        return tuple("attention" if i % 4 == 3 else "gdn" for i in range(layers))
    if method == "kda_hybrid":
        return tuple("mla_nope" if i % 4 == 3 else "kda" for i in range(layers))
    if method == "qsa_hybrid":
        return tuple("qsa" if i % 4 == 3 else "gdn" for i in range(layers))
    if method == "moba_hybrid":
        if layers < 4:raise ValueError("moba_hybrid needs at least one MoBA layer and three full-attention layers")
        return ("moba",)*(layers-3)+("attention",)*3
    return tuple("swa" if i % 2 else "gdn" for i in range(layers))

@dataclass(frozen=True)
class ModelConfig:
    method: str = "dense"
    width: int = 1536
    heads: int = 12
    vocab_size: int = 32768
    sequence_len: int = 2048
    mlp_width: int = 6144
    layers: int = 24
    pre_layers: int = 3
    core_layers: int = 9
    post_layers: int = 3
    loops: int = 2
    window_pattern: str = "SSSL"
    max_loops: int = 3
    act_epsilon: float = .01
    ponder_weight: float = .01
    huginn_mean: float = 3.
    huginn_sigma: float = .5
    huginn_backprop_loops: int = 2
    routing_seed: int = 1729
    eval_loops: int = 3
    ouro_entropy_weight: float = .1
    ouro_gate_steps: int = 1024
    ouro_exit_quantile: float = .5
    sparse_topk: int = 256
    block_size: int = 128
    compression_size: int = 4
    local_window: int = 512
    index_heads: int = 4
    index_dim: int = 128
    index_rope_dim: int = 64
    index_loss_weight: float = 1.
    burnin_steps: int = 512
    slots: int = 64
    gate_rank: int = 16
    conv_size: int = 4
    reference_chunk: int = 32
    backend: str = "reference"
    scale: str = "formal"
    residual_streams: int = 4
    residual_gate_init: float = .01
    sinkhorn_iterations: int = 20
    attnres_groups: int = 8
    # Keys each sliding-window query sees, the current token included (left window w-1).
    sliding_window: int = 2048
    q_rank: int = 768
    kv_rank: int = 256
    qk_nope_dim: int = 64
    qk_rope_dim: int = 64
    v_head_dim: int = 128
    # A plugin method's own parameters as (name, value) pairs; empty for built-in methods.
    options: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,"options",frozen_options(self.options))
        if self.method not in METHODS: raise ValueError("unknown method")
        if self.method in PLUGINS:
            declared=PLUGINS[self.method].get("options",{})
            if {name for name,_ in self.options}!=set(declared):
                raise ValueError(f"{self.method} options must be exactly: {', '.join(sorted(declared)) or 'none'}")
            if self.layers%LAYER_MULTIPLE[self.method]:
                raise ValueError(f"{self.method} needs a layer count that is a multiple of {LAYER_MULTIPLE[self.method]}")
        elif self.options:
            raise ValueError("options are parameters of plugin methods")
        defaults={f.name:f.default for f in fields(self)}
        for name,users in SCOPED_FIELDS.items():
            if self.method not in users and getattr(self,name)!=defaults[name]:
                raise ValueError(f"{name} is not a parameter of {self.method}")
        if min(self.residual_streams,self.sinkhorn_iterations,self.attnres_groups)<=0 or not self.residual_gate_init>0:
            raise ValueError("residual stream, iteration and group counts and gate scale must be positive")
        if self.method in FULL_ATTENTION and self.window_pattern!="L":
            raise ValueError(f"{self.method} uses full causal attention in every layer (window_pattern L)")
        if self.method in HYBRID:
            if self.window_pattern!="L":
                raise ValueError(f"{self.method} sets attention windows per layer (window_pattern L)")
            hybrid_layout(self.method,self.layers)
        if self.sliding_window<=0:
            raise ValueError("sliding window must contain the current token")
        if self.method in MLA_CONFIG:
            if min(self.q_rank,self.kv_rank,self.qk_nope_dim,self.v_head_dim)<=0 or self.qk_rope_dim!=0:
                raise ValueError("MLA-NoPE requires positive latent/head dimensions and zero RoPE dimensions")
            if self.scale=="formal" and (self.q_rank,self.kv_rank,self.qk_nope_dim,self.v_head_dim)!=(768,256,128,128):
                raise ValueError("formal MLA-NoPE retains the registered O03-N ranks and head dimensions")
        if self.method=="attnres" and self.attnres_groups>self.layers:
            raise ValueError("AttnRes needs at least one transformer block per group")
        if self.width < 24 or self.width % self.heads or self.head_dim % 2:
            raise ValueError("native width>=24 and even head dimension required")
        if min(self.layers,self.core_layers,self.loops,self.max_loops,self.sequence_len,
               self.mlp_width,self.vocab_size,self.sparse_topk,self.block_size,
               self.compression_size,self.index_dim,self.slots,self.reference_chunk) <= 0:
            raise ValueError("dimensions and counts must be positive")
        if min(self.pre_layers,self.post_layers,self.burnin_steps) < 0:
            raise ValueError("negative stage depth")
        if not 0 < self.act_epsilon < 1 or not 0 < self.ouro_exit_quantile <= 1:
            raise ValueError("invalid halting threshold")
        if self.huginn_mean <= 1 or not 0 < self.huginn_backprop_loops <= self.max_loops:
            raise ValueError("invalid Huginn distribution/backprop depth")
        if self.index_rope_dim % 2 or not 0 <= self.index_rope_dim <= self.index_dim:
            raise ValueError("invalid partial RoPE size")
        if self.backend not in ("reference", *CHUNK_BACKENDS):
            raise ValueError("this reference does not silently substitute an unavailable backend")
        if self.backend in CHUNK_BACKENDS and self.method not in CHUNKED_METHODS:
            raise ValueError("chunked backend is registered only for DeltaNet, KDA, RetNet, ELU+1, GDN and hybrids")
        if self.scale not in ("formal", "debug"):
            raise ValueError("scale must be formal or debug")
        if self.scale == "formal":
            if (self.width,self.heads,self.vocab_size,self.sequence_len)!=(1536,12,32768,2048):
                raise ValueError("formal configurations must retain the common d24 dimensions")
            if not 3*self.width <= self.mlp_width <= 5*self.width or self.mlp_width % 128:
                raise ValueError("formal FFN must be in the registered 3d..5d, multiple-128 grid")
        if self.method in PLUGINS:
            from nexttransformer.models.plugins import load
            check=load(self.method).check
            if check:check(self)

    def option(self,name):
        """Value of a plugin option."""
        return dict(self.options)[name]
    @property
    def head_dim(self): return self.width // self.heads
    @property
    def n_embd(self): return self.width
    @property
    def n_head(self): return self.heads
    @property
    def n_kv_head(self): return self.heads
    @property
    def value_embeddings(self):
        """The H04/control shell removes both VE tables and every per-layer VE gate."""
        return self.method not in NO_VE
    @property
    def has_sparse(self):
        return any(k in SPARSE for k in (self.layer_mixers if self.method in HYBRID else (self.method,)))
    @property
    def has_sparse_burnin(self):
        return any(k in ("dsa","qsa","seer") for k in
                   (self.layer_mixers if self.method in HYBRID else (self.method,)))
    @property
    def n_layer(self):
        return self.pre_layers+self.core_layers+self.post_layers if self.method in LOOPED else self.layers
    @property
    def maximum_depth(self):
        n=self.max_loops if self.method in ("ut_act","huginn","ouro") else self.loops
        return self.pre_layers+n*self.core_layers+self.post_layers if self.method in LOOPED else self.layers
    @property
    def attnres_group_sizes(self):
        """Transformer blocks per AttnRes group: balanced, larger groups first, remainder last."""
        whole,extra=divmod(self.layers,self.attnres_groups)
        return (whole+1,)*extra+(whole,)*(self.attnres_groups-extra)
    @property
    def attnres_group_starts(self):
        """Physical layer index of the first block of each AttnRes group."""
        starts=[0]
        for size in self.attnres_group_sizes[:-1]:starts.append(starts[-1]+size)
        return tuple(starts)
    @property
    def layer_mixers(self):
        """Registered mixer kind of each physical layer of a hybrid."""
        if self.method not in HYBRID:raise ValueError(f"{self.method} is not a per-layer hybrid")
        return hybrid_layout(self.method,self.layers)
    def serialized(self):
        """Scientific fields of this method; scoped fields of other methods are omitted.

        A hybrid also records its derived per-layer layout, so the layout is part of
        the fingerprint and readable without code; loading verifies it is unchanged.
        A plugin method records its options; built-in methods have none and omit the field.
        """
        data=asdict(self)
        for name,users in SCOPED_FIELDS.items():
            if self.method not in users:data.pop(name)
        if self.method in HYBRID:data["layer_mixers"]=list(self.layer_mixers)
        thaw=lambda value:[thaw(v) for v in value] if isinstance(value,tuple) else value
        if self.options:data["options"]={name:thaw(value) for name,value in self.options}
        else:data.pop("options")
        return data
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.serialized(),sort_keys=True).encode()).hexdigest()


def candidate_config(method):
    if method in PLUGINS:
        # A new stack starts from the v2 shell (24 layers, FFN 6144, full attention) and its option defaults.
        spec=PLUGINS[method]
        shell={"layers":24,"mlp_width":6144,"window_pattern":"L",**spec.get("shell",{})}
        return ModelConfig(method=method,options=spec.get("options",{}),**shell)
    changes={"ut_act":dict(core_layers=6),"huginn":dict(core_layers=8,max_loops=4),
             "ouro":dict(core_layers=4),"deeploop":dict(pre_layers=0,post_layers=0,core_layers=12),
             "nsa":dict(block_size=32,compression_size=32),"seer":dict(block_size=64),
             # A06 stabilized variants (2026-10-20) and A07 GDN (2026-10-22):
             # registered on their training algorithm.
             "elu_norm":dict(backend="fla_chunk64"),"elu_qknorm":dict(backend="fla_chunk64"),
             "gdn":dict(backend="fla_chunk64"),
             "deltanet_hybrid":dict(backend="fla_chunk64",window_pattern="L"),
             # v2 hybrids (2026-10-24): A07's GDN layers with B01's attention windows set per layer.
             "gdn_hybrid":dict(backend="fla_chunk64",window_pattern="L"),
             "gdn_hybrid_swa":dict(backend="fla_chunk64",window_pattern="L")}
    changes.update(kda_hybrid=dict(backend="fla_chunk64",window_pattern="L",qk_nope_dim=128,qk_rope_dim=0),
                   qsa_hybrid=dict(backend="fla_chunk64",window_pattern="L"),
                   moba_hybrid=dict(window_pattern="L"),
                   kda_no_ve=dict(backend="fla_chunk64"),
                   mla_nope=dict(qk_nope_dim=128,qk_rope_dim=0))
    if method in FULL_ATTENTION:changes[method]=dict(changes.get(method,{}),window_pattern="L")
    return ModelConfig(method=method,**changes.get(method,{}))


def debug_config(method, *, width=48, sequence_len=12):
    c=candidate_config(method)
    plugin=PLUGINS.get(method)
    # A chunk-backend debug model keeps every dimension off the chunk length (64).
    return replace(c,scale="debug",width=width,heads=2,vocab_size=96 if c.backend in CHUNK_BACKENDS else 64,
                   sequence_len=sequence_len,mlp_width=4*width,
                   layers=3 if method=="attnres" else 4 if method in ("deltanet_hybrid","moba_hybrid")
                   else LAYER_MULTIPLE[method]*(2 if LAYER_MULTIPLE[method]==1 else 1) if plugin
                   else LAYER_MULTIPLE.get(method,2),
                   **(dict(options=dict(c.options,**plugin.get("debug_options",{}))) if plugin else {}),
                   pre_layers=0 if method=="deeploop" else 1,core_layers=1,
                   post_layers=0 if method=="deeploop" else 1,index_dim=8,index_heads=2,
                   index_rope_dim=4,slots=4,block_size=4,compression_size=2,
                   sparse_topk=4,local_window=4,reference_chunk=4,gate_rank=4,
                   # Three blocks in two AttnRes groups exercise an uneven boundary.
                   **(dict(attnres_groups=2) if method=="attnres" else {}),
                   **(dict(q_rank=20,kv_rank=8,qk_nope_dim=16,qk_rope_dim=0,v_head_dim=16)
                      if method in MLA_CONFIG else {}),
                   # A window shorter than the debug context, so the window is exercised.
                   **(dict(sliding_window=4) if method=="gdn_hybrid_swa" else {}))
