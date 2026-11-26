"""Reference MoE language model of the two tracks: Marin's Grug MoE ("May Recipe") in PyTorch.

Ported from marin-community/marin experiments/grug/moe/model.py at commit d259a97068, which launched
`moe_may_compute_opt_d{768,1024}_ep1`. Everything a token passes through:

    h = embed_gated(embed_norm(E[tok]))                                   RMSNorm, then GatedNorm
    block i:  a = attn_gated(attn_norm(x));   x = x + Attention_i(a)
              m = mlp_gated(mlp_norm(x));     x = x + MoE_i(m) + Shared_i(m)
    logits = final_gated(final_norm(x)) @ head                            untied embedding and head

    RMSNorm     fp32 statistics, eps 1e-5, learned gain
    GatedNorm   y = x * sigmoid(silu(x W_down) W_up), rank 128, no biases
    Attention   GQA, head_dim 128, no biases. PKO on long layers (the rope-free half of each key is replaced
                by the previous token's, zero at document starts), non-parametric QK RMS norm (eps 1e-6),
                half-RoPE (first 64 dims, theta 10,000, split-half pairs), q *= 1.3, softmax scale 1/sqrt(128),
                causal + same-document mask, a 2,048-key window except on long layers (i % 4 == 3 and the last),
                XSA (remove each head output's component along the token's own value), a head-wise gate
                2*sigmoid(a @ G) (G zero-initialized), then W_o
    MoE         256 SwiGLU experts, top 4, dropless; plus one always-on SwiGLU shared expert of width d
    Router      bf16 logits; experts chosen on logits + a quantile-balancing (QB) bias; combine weights are
                sigmoid(logits) renormalized to sum to 2.5. No auxiliary loss, no z-loss
    Init        truncated normal (+-3 std), std = 0.5 / sqrt(d), for every matrix (embedding and head included);
                head-wise gate zero; norm gains one

The only change from Marin is the vocabulary (dolma2, 100,278 tokens padded to 100,352, instead of Llama 3's
128,256) and document boundaries: a document starts at its <|endoftext|> separator. Parameter names are stable:
checkpoints store tensors by name.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from moe.spec import PROTOCOL, long_layers, track as get_track

GROUPED_MM = hasattr(torch, "_grouped_mm")
DOCUMENT_TOKEN = 100257   # <|endoftext|>


@dataclass(frozen=True)
class MoEConfig:
    track: str
    method: str = "baseline"
    width: int = 768
    layers: int = 8
    heads: int = 6
    kv_heads: int = 1
    head_dim: int = 128
    routed_experts: int = 256
    top_k: int = 4
    expert_width: int = 384
    shared_expert_width: int = 768
    vocab_size: int = PROTOCOL.padded_vocab_size
    sequence_len: int = PROTOCOL.sequence_len
    sliding_window: int = PROTOCOL.sliding_window
    long_layer_every: int = PROTOCOL.long_layer_every
    rope_theta: float = 10_000.0
    rope_dims: int = 64
    qk_mult: float = 1.3
    gated_norm_rank: int = 128
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-6
    routing_renorm_sum: float = 2.5
    qb_group_rows: int = 4           # rows per quantile-balancing group (one Marin device's share of a batch)
    init_scale: float = 0.5          # std = init_scale / sqrt(width)
    document_token: int = DOCUMENT_TOKEN
    options: tuple = field(default=())   # sorted (name, value) pairs owned by a plugin
    scale: str = "formal"                # "formal" or "debug" (tiny CPU shape used by tests)

    def __post_init__(self):
        if self.width % 64 or self.heads % self.kv_heads or min(self.layers, self.heads, self.kv_heads) < 1:
            raise ValueError("width must be a multiple of 64 and heads a multiple of kv_heads")
        if not 1 <= self.top_k < self.routed_experts or self.expert_width <= 0 or self.shared_expert_width < 0:
            raise ValueError("invalid expert configuration (quantile balancing needs top_k < routed_experts)")
        if self.rope_dims % 2 or not 0 < self.rope_dims <= self.head_dim:
            raise ValueError("rope_dims must be even and at most head_dim")
        if self.scale not in ("formal", "debug"):
            raise ValueError("scale is formal or debug")
        object.__setattr__(self, "options", tuple(sorted((str(k), v) for k, v in dict(self.options).items())))

    @classmethod
    def baseline(cls, track_name, method="baseline", **overrides):
        t = get_track(track_name)
        shape = dict(width=t.width, layers=t.layers, heads=t.heads, kv_heads=t.kv_heads, head_dim=t.head_dim,
                     routed_experts=t.routed_experts, top_k=t.top_k, expert_width=t.expert_width,
                     shared_expert_width=t.shared_expert_width, qb_group_rows=t.qb_group_rows)
        return cls(track=track_name, method=method, **{**shape, **overrides})

    def debug(self):
        """A tiny CPU shape with the same structure: 4 layers so that one is long and one is short."""
        return replace(self, width=64, layers=4, heads=4, kv_heads=2, head_dim=16, rope_dims=8, routed_experts=8,
                       top_k=2, expert_width=32, shared_expert_width=64 if self.shared_expert_width else 0,
                       gated_norm_rank=8, vocab_size=512, sequence_len=64, sliding_window=16, qb_group_rows=1,
                       document_token=0, scale="debug")

    def option(self, name, default=None):
        return dict(self.options).get(name, default)

    @property
    def long_layers(self):
        return long_layers(self.layers, self.long_layer_every)

    def window(self, layer):
        """Left context in keys before the current one; -1 means full causal attention."""
        return -1 if layer in self.long_layers else self.sliding_window - 1

    @property
    def init_std(self):
        return self.init_scale / math.sqrt(self.width)

    def serialized(self):
        data = asdict(self)
        data["options"] = {k: v for k, v in self.options}
        return data

    @classmethod
    def from_serialized(cls, data):
        data = dict(data)
        data["options"] = tuple(sorted(dict(data.get("options", {})).items()))
        return cls(**data)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.serialized(), sort_keys=True).encode()).hexdigest()


# --- norms ----------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return y.to(x.dtype)


class GatedNorm(nn.Module):
    """y = x * sigmoid(silu(x W_down) W_up): a low-rank multiplicative gate after a norm."""

    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.up(F.silu(self.down(x)))).to(x.dtype)


# --- attention ------------------------------------------------------------------------------------------------

def segments(idx, document_token):
    """Document id of every position: a document starts at its separator token."""
    return torch.cumsum((idx == document_token).int(), dim=1)


def rotary(config, length, device):
    half = config.rope_dims // 2
    inv_freq = config.rope_theta ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    angle = torch.outer(torch.arange(length, dtype=torch.float32, device=device), inv_freq)
    return angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]


def half_rope(x, cos_sin, dims):
    """Rotate the first `dims` channels with split-half pairs; the rest pass through (positions are absolute)."""
    cos, sin = (t[:, :x.size(1)] for t in cos_sin)
    rot, rest = x[..., :dims].float(), x[..., dims:]
    a, b = rot.chunk(2, dim=-1)
    rot = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)
    return torch.cat((rot, rest), dim=-1)


class Masks:
    """Causal + same-document masks of one forward pass, with and without the sliding window."""

    def __init__(self, seg, config):
        self.seg, self.config = seg, config
        self.starts = torch.ones_like(seg, dtype=torch.bool)
        self.starts[:, 1:] = seg[:, 1:] != seg[:, :-1]
        self._cache = {}

    def get(self, window):
        if window not in self._cache:
            self._cache[window] = build_mask(self.seg, window)
        return self._cache[window]


FLEX = True       # compiled FlexAttention on CUDA; evaluation uses the SDPA path (see eager_attention)


@contextmanager
def eager_attention():
    """Compute attention with SDPA and an explicit mask, as on CPU.

    Training always runs the 4,096-token context, so one compiled FlexAttention kernel serves every update.
    Evaluation prompts vary in length, which would recompile it per length, so evaluation uses this path. The
    two compute the same function (tests/test_moe_track.py and the GPU equivalence check).
    """
    global FLEX
    FLEX, previous = False, FLEX
    try:
        yield
    finally:
        FLEX = previous


def build_mask(seg, window):
    if seg.is_cuda and FLEX:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q, k):
            ok = (k <= q) & (seg[b, q] == seg[b, k])
            return ok & (q - k <= window) if window >= 0 else ok
        B, T = seg.shape
        return create_block_mask(mask_mod, B, None, T, T, device=seg.device, _compile=True)
    T = seg.size(1)
    pos = torch.arange(T, device=seg.device)
    allowed = (pos[None, :] <= pos[:, None])[None] & (seg[:, :, None] == seg[:, None, :])
    if window >= 0:
        allowed &= (pos[:, None] - pos[None, :] <= window)[None]
    return allowed[:, None]   # (B, 1, T, T)


_FLEX = None


def attention(q, k, v, mask):
    """Masked softmax attention, (B, T, H, D) layout, GQA by head groups, scale 1/sqrt(D).

    Every softmax attention of the track goes through this function: the cost ledger charges its causal or
    windowed edges by rule (moe.compute.attention_edges) instead of measuring the kernel. CUDA uses compiled
    FlexAttention with a block mask; elsewhere SDPA with an explicit boolean mask computes the same function.
    """
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if q.is_cuda and FLEX:
        global _FLEX
        if _FLEX is None:
            from torch.nn.attention.flex_attention import flex_attention
            _FLEX = torch.compile(flex_attention, dynamic=False)
        y = _FLEX(q, k, v, block_mask=mask, enable_gqa=k.size(1) != q.size(1))
    else:
        groups = q.size(1) // k.size(1)
        k, v = (t.repeat_interleave(groups, dim=1) for t in (k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return y.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config, self.layer = config, layer
        d, hd = config.width, config.head_dim
        self.long = layer in config.long_layers
        self.q = nn.Linear(d, config.heads * hd, bias=False)
        self.k = nn.Linear(d, config.kv_heads * hd, bias=False)
        self.v = nn.Linear(d, config.kv_heads * hd, bias=False)
        self.o = nn.Linear(config.heads * hd, d, bias=False)
        self.head_gate = nn.Linear(d, config.heads, bias=False)

    def forward(self, x, cos_sin, masks):
        B, T, _ = x.shape
        c = self.config
        q = self.q(x).view(B, T, c.heads, c.head_dim)
        k = self.k(x).view(B, T, c.kv_heads, c.head_dim)
        v = self.v(x).view(B, T, c.kv_heads, c.head_dim)
        if self.long:   # PKO: the rope-free half of each key comes from the previous token of the same document
            r = c.rope_dims
            shifted = torch.cat((torch.zeros_like(k[:, :1, :, r:]), k[:, :-1, :, r:]), dim=1)
            shifted = shifted.masked_fill(masks.starts[:, :, None, None], 0)
            k = torch.cat((k[..., :r], shifted), dim=-1)
        q = half_rope(F.rms_norm(q, (c.head_dim,), eps=c.qk_norm_eps), cos_sin, c.rope_dims) * c.qk_mult
        k = half_rope(F.rms_norm(k, (c.head_dim,), eps=c.qk_norm_eps), cos_sin, c.rope_dims)
        y = attention(q, k, v, masks.get(c.window(self.layer)))
        # XSA: remove the component along the token's own value vector, per head
        own = v.repeat_interleave(c.heads // c.kv_heads, dim=2)
        y = y - ((y * own).sum(-1, keepdim=True) / ((own * own).sum(-1, keepdim=True) + 1e-6)) * own
        y = y * (2 * torch.sigmoid(self.head_gate(x))).unsqueeze(-1).to(y.dtype)
        return self.o(y.reshape(B, T, -1))


# --- feed-forward ---------------------------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    """Marin's QB router: top-k on logits plus a quantile-balancing bias, sigmoid gates renormalized to 2.5.

    In training every forward records, per group of `qb_group_rows` rows, beta_e = the q-th largest value of
    (logit_e - alpha) over the group's tokens, where alpha is the token's (k+1)-th largest biased logit and
    q = max(1, group_tokens * k // E). After the update the bias becomes -mean(beta) + mean_e(mean(beta)),
    averaged over every group of the update (moe.train.balance_routers).
    """

    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.routed_experts, config.width))
        self.register_buffer("bias", torch.zeros(config.routed_experts), persistent=True)
        self.register_buffer("beta_sum", torch.zeros(config.routed_experts), persistent=False)
        self.register_buffer("beta_count", torch.zeros(()), persistent=False)
        self.register_buffer("load", torch.zeros(config.routed_experts), persistent=False)
        self.tokens_per_row = None

    def forward(self, x):
        c = self.config
        logits = F.linear(x, self.weight.to(x.dtype)).float()
        biased = logits + self.bias.to(x.dtype).float()
        values, chosen = torch.topk(biased, c.top_k + 1, dim=-1)
        index = chosen[:, :c.top_k]
        gates = torch.sigmoid(logits.gather(-1, index))
        gates = (gates * (c.routing_renorm_sum / (gates.sum(-1, keepdim=True) + 1e-9))).to(x.dtype)
        if self.training:
            with torch.no_grad():
                group = c.qb_group_rows * (self.tokens_per_row or x.size(0))
                if x.size(0) % group:
                    raise ValueError("a micro-batch must hold whole quantile-balancing groups")
                s = (logits - values[:, c.top_k:]).view(-1, group, c.routed_experts)
                q = max(1, group * c.top_k // c.routed_experts)
                beta = torch.topk(s, q, dim=1).values[:, -1]
                self.beta_sum += beta.sum(0)
                self.beta_count += beta.size(0)
                self.load += torch.bincount(index.flatten(), minlength=c.routed_experts).float()
        return index, gates, None

    @torch.no_grad()
    def balance(self, beta_sum, beta_count):
        """Set the bias from the mean group quantiles of the last update (Marin's QB update)."""
        if beta_count > 0:
            beta = beta_sum / beta_count
            self.bias.copy_(-beta + beta.mean())
        self.last_load = self.load.clone()


class Experts(nn.Module):
    """Routed SwiGLU experts with dropless dispatch: grouped GEMM on CUDA, a per-expert loop elsewhere.

    Weights are stored output-major, (E, out, in), like nn.Linear; the grouped GEMM takes their transposed views
    and per-expert row groups padded to multiples of 16 (the layout torchtitan uses with torch._grouped_mm).
    """
    ALIGN = 16

    def __init__(self, config, layer):
        super().__init__()
        E, d, f = config.routed_experts, config.width, config.expert_width
        self.w_gate = nn.Parameter(torch.empty(E, f, d))
        self.w_up = nn.Parameter(torch.empty(E, f, d))
        self.w_down = nn.Parameter(torch.empty(E, d, f))

    def forward(self, x, index, gates):
        N, d = x.shape
        k = index.size(1)
        flat = index.flatten()
        order = torch.argsort(flat, stable=True)
        token = order // k
        counts = torch.bincount(flat, minlength=self.w_gate.size(0))
        xs = x[token]
        weights = tuple(w.to(x.dtype) for w in (self.w_gate, self.w_up, self.w_down))
        if GROUPED_MM and x.is_cuda and x.dtype == torch.bfloat16:
            ys = self._grouped(xs, counts, *weights)
        else:
            # unbind once: indexing a weight per expert would build a full-size gradient per expert
            parts = []
            for rows, wg, wu, wd in zip(torch.split(xs, counts.tolist()), *(w.unbind(0) for w in weights)):
                if rows.size(0):
                    parts.append(F.linear(F.silu(F.linear(rows, wg)) * F.linear(rows, wu), wd))
            ys = torch.cat(parts) if parts else xs.new_zeros(0, d)
        ys = ys * gates.flatten()[order].to(ys.dtype)[:, None]
        return x.new_zeros(N, d).index_add_(0, token, ys)

    def _grouped(self, xs, counts, w_gate, w_up, w_down):
        a = self.ALIGN
        padded = (counts + a - 1) // a * a
        ends = torch.cumsum(padded, 0)
        starts = ends - padded
        within = torch.arange(xs.size(0), device=xs.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts)
        slot = torch.repeat_interleave(starts, counts) + within
        buffer = xs.new_zeros(int(ends[-1]) if ends.numel() else 0, xs.size(1)).index_copy(0, slot, xs)
        offsets = ends.to(torch.int32)
        mm = torch._grouped_mm
        h = F.silu(mm(buffer, w_gate.transpose(-2, -1), offs=offsets)) * mm(buffer, w_up.transpose(-2, -1),
                                                                            offs=offsets)
        return mm(h, w_down.transpose(-2, -1), offs=offsets)[slot]


class MoE(nn.Module):
    def __init__(self, config, layer, router=None, experts=None):
        super().__init__()
        self.router = router or Router(config, layer)
        self.experts = experts or Experts(config, layer)
        self.shared = SwiGLU(config.width, config.shared_expert_width) if config.shared_expert_width else None
        self.last_aux = None

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        if isinstance(self.router, Router):
            self.router.tokens_per_row = T
        index, gates, aux = self.router(flat)
        self.last_aux = aux
        y = self.experts(flat, index, gates)
        if self.shared is not None:
            y = y + self.shared(flat)
        return y.view(B, T, d)


class Block(nn.Module):
    def __init__(self, config, layer, hooks):
        super().__init__()
        rank = config.gated_norm_rank
        self.attn_norm = RMSNorm(config.width, config.norm_eps)
        self.attn_gated = GatedNorm(config.width, rank)
        self.attn = hooks.mixer(config, layer) if hooks.mixer else Attention(config, layer)
        self.mlp_norm = RMSNorm(config.width, config.norm_eps)
        self.mlp_gated = GatedNorm(config.width, rank)
        if hooks.ffn:
            self.ffn = hooks.ffn(config, layer)
        else:
            router = hooks.router(config, layer) if hooks.router else None
            self.ffn = MoE(config, layer, router=router)

    def forward(self, x, cos_sin, masks):
        x = x + self.attn(self.attn_gated(self.attn_norm(x)), cos_sin, masks)
        return x + self.ffn(self.mlp_gated(self.mlp_norm(x)))


class MoELM(nn.Module):
    def __init__(self, config, hooks=None):
        super().__init__()
        from moe.plugins import hooks_for
        hooks = hooks or hooks_for(config.method)
        self.config = config
        rank = config.gated_norm_rank
        self.embed = nn.Embedding(config.vocab_size, config.width)
        self.embed_norm = RMSNorm(config.width, config.norm_eps)
        self.embed_gated = GatedNorm(config.width, rank)
        self.blocks = nn.ModuleList(Block(config, i, hooks) for i in range(config.layers))
        if hooks.block:
            self.blocks = nn.ModuleList(hooks.block(b, config, i) for i, b in enumerate(self.blocks))
        self.final_norm = RMSNorm(config.width, config.norm_eps)
        self.final_gated = GatedNorm(config.width, rank)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.max_seq_len = config.sequence_len
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._init_hook = hooks.init

    @torch.no_grad()
    def reset_buffers(self):
        """Rotary tables (twice the training context, for length-extrapolation probes) and routing state."""
        device = self.embed.weight.device
        self.cos, self.sin = rotary(self.config, 2 * self.config.sequence_len, device)
        for router in self.routers():
            for name in ("bias", "beta_sum", "load"):
                getattr(router, name).zero_()
            router.beta_count.zero_()
        return self

    @torch.no_grad()
    def initialize(self):
        """Marin's initialization, then the plugin's own `init(model)`."""
        self.reset_buffers()
        initialize_parameters(self, self.config)
        if self._init_hook:
            self._init_hook(self)
        return self

    def get_device(self):
        return self.embed.weight.device

    def routers(self):
        return [m for m in self.modules() if isinstance(m, Router)]

    def forward(self, idx, targets=None, reduction="mean"):
        T = idx.size(1)
        if T > self.cos.size(1):
            raise ValueError(f"input of {T} tokens exceeds the rotary table ({self.cos.size(1)})")
        masks = Masks(segments(idx, self.config.document_token), self.config)
        x = self.embed_gated(self.embed_norm(self.embed(idx)))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin, masks)
        logits = self.head(self.final_gated(self.final_norm(x))).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
                               reduction=reduction)
        aux = [m.last_aux for m in self.modules() if getattr(m, "last_aux", None) is not None]
        self.last_aux = sum(aux) if aux else None
        return loss

    def parameter_counts(self):
        """Total parameters and the parameters one token activates (routed experts counted top_k times)."""
        c = self.config
        total = sum(p.numel() for p in self.parameters())
        routed = sum(p.numel() for m in self.modules() if isinstance(m, Experts) for p in m.parameters())
        embedding = self.embed.weight.numel()
        active = total - routed + routed * c.top_k // c.routed_experts
        return dict(total=total, active=active, embedding=embedding, head=self.head.weight.numel(),
                    active_non_embedding=active - embedding - self.head.weight.numel(),
                    total_non_embedding=total - embedding - self.head.weight.numel())


@torch.no_grad()
def initialize_parameters(module, config):
    """Truncated normal (+-3 std, std = 0.5/sqrt(d)) for matrices, zero head-wise gates, unit norm gains."""
    std = config.init_std
    for name, p in module.named_parameters():
        if p.ndim == 1:
            nn.init.ones_(p)
        elif name.endswith("head_gate.weight"):
            nn.init.zeros_(p)
        else:
            nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)


def build_model(config, device="cpu", initialize=True):
    """Build on the meta device, then materialize on `device` and run the reference initialization."""
    from moe.plugins import model_class
    cls = model_class(config.method) or MoELM
    with torch.device("meta"):
        model = cls(config)
    if device == "meta":
        return model
    model.to_empty(device=device)
    return model.initialize() if initialize else model.reset_buffers()
