"""N-gram memory + Full AttnRes + Canon-A/B/C/D.

Trunk: Full Attention Residuals (every sublayer reads a softmax mixture over depth sources: the embedding and every
earlier block's output, through learned pseudo-queries) with Canon layers (zero-initialized causal depthwise
convolutions) on every sublayer input (A/C), on q, k, v (B) and on the MLP hidden pre-activation (D). With every option
at its neutral value this is the published `attnres_full_canon_abcd`.

Additions, each behind an option and each starting as the identity:
- `ngram_rows` > 0: hashed n-gram memory. For every position, the last n tokens (n in `ngram_orders`) are hashed by
  `ngram_heads` independent multiplicative hashes into rows of zero-initialized embedding tables; the signed lookups
  of all orders and heads are concatenated, scaled by `ngram_gain`, projected to the model width and added to the
  token embedding, i.e. to depth source b0, which every sublayer and the readout attend to. Lookups cost no matrix
  arithmetic; the tables train under AdamW (their parameter names place them in the auxiliary group).
- `depth_heads` > 1: the depth read splits the width into channel groups, each with its own softmax over sources.
- `depth_sources`: "block" keeps one depth source per block (Full AttnRes); "sublayer" keeps every attention and
  every FFN output as its own source.
- `ngram_inject`: "embed" adds the memory to the token embedding (b0); "source" keeps it as a depth source of its own;
  "none" keeps it out of the residual path (value path only).
- `ngram_group`: "aux" keeps the tables as FP32 parameters of the auxiliary AdamW group; "embed" keeps them as
  embedding tables next to the value embeddings, in the shell's embedding AdamW group and compute dtype.
- `ngram_project`: false adds the concatenated lookups directly (their total size must equal the width).
- `ngram_sign`: false drops the hashed sign of each lookup.
- `ngram_value`: "proj" adds a second projection of the same lookups to the value embedding of every layer that has
  one, behind that layer's value-embedding gate; "direct" adds the lookups themselves (they must have the model width).
- `canon_ve`: Canon-B on the values acts after the gated value embedding is added, so it also mixes the value
  embeddings of the last few tokens.
- `canon_readout`: a Canon layer on the final depth read, before the readout norm.
- `canon_write`: a Canon layer on every completed block sum before it becomes a depth source (block sources only);
  `canon_write_size` is its kernel size (the other Canon layers use `canon_size`).
- `canon_embed`: the same write-side Canon layer on the embedding source b0 (after the n-gram memory is added).
- `depth_bias` > 0: every depth read adds a learned per-source scalar to its logits (a static preference over depth,
  next to the content-dependent pseudo-query), scaled by this value.
- `depth_qkv`: every attention sublayer makes three depth reads, one each for its query, key and value inputs, with
  separate pseudo-queries and Canon-A layers; all three start as the same uniform read.
- `softcap`: the logit soft-cap of the readout.
- `mlp_act`: "relu2" (shell) or "leaky2" (squared leaky ReLU with slope 0.5).
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import coefficient_precision, depth_attention

ACTIVATIONS = ("relu2", "leaky2")
DEPTH_SOURCES = ("block", "sublayer")
NGRAM_INJECT = ("embed", "source", "none")
NGRAM_VALUE = ("none", "proj", "direct")
NGRAM_GROUP = ("aux", "embed")
# Odd 62-bit multipliers for the multiplicative n-gram hashes: one row per hash head, one column per n-gram slot.
_MULTIPLIERS = (
    (0x1F3D5B79A2C4E681, 0x2545F4914F6CDD1D, 0x27D4EB2F165667C5, 0x165667B19E3779F9, 0x3C6EF372FE94F82B),
    (0x369DEA0F31A53F85, 0x1B873593CC9E2D51, 0x2127599BF4325C37, 0x0CF1BBCDCB7A5643, 0x2B992DDFA23249D7),
    (0x14057B7EF767814F, 0x3A8F05C5B1D3A8E9, 0x0B4B82F6A1E5C92D, 0x31848BABD5A2F0B7, 0x1D8E4E27C47D124F),
    (0x2D358DCCAA6C78A5, 0x0A0761D6478BD643, 0x38D01377BE6ACD0F, 0x23B4A1C5E9F7D36B, 0x19E3779B97F4A7C1),
)
_MIX = 0x2545F4914F6CDD1D
_MASK = 0x7FFFFFFFFFFFFFFF


class Canon(nn.Module):
    """Causal depthwise convolution over the last `size` positions, added to its input; starts as the identity."""

    def __init__(self, width, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        self.size = size

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)), self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


class CanonAttention(CausalSelfAttention):
    """The shell's attention with a Canon layer on each of q, k and v (Canon-B)."""

    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.canon_qkv = nn.ModuleList(Canon(self.n_embd, config.option("canon_size")) for _ in range(3))
        self.canon_ve = config.option("canon_ve")

    def forward(self, x, ve, cos_sin, window_size, kv_cache, xk=None, xv=None):
        """`x` feeds the queries; `xk` and `xv` feed the keys and values (both default to `x`)."""
        if kv_cache is not None:
            raise NotImplementedError("training and full-sequence evaluation only")
        B, T, C = x.size()
        if xk is None: xk = x
        if xv is None: xv = x
        q, k, v = self.canon_qkv[0](self.c_q(x)), self.canon_qkv[1](self.c_k(xk)), self.c_v(xv)
        if not self.canon_ve: v = self.canon_qkv[2](v)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(xv[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        if self.canon_ve: v = self.canon_qkv[2](v.reshape(B, T, C)).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class CanonMLP(nn.Module):
    """The shell's squared-ReLU MLP with a Canon layer on its hidden activations, before the nonlinearity (Canon-D)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.width, config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)
        self.canon = Canon(config.mlp_width, config.option("canon_size"))
        self.leaky = config.option("mlp_act") == "leaky2"

    def forward(self, x):
        x = self.canon(self.c_fc(x))
        x = torch.where(x > 0, x, 0.5 * x) if self.leaky else F.relu(x)
        return self.c_proj(x.square())


class PlainLinear(Linear):
    """A bias-free linear layer that the plugin's `init` draws, after the shared shell's initialization.

    build_model resets every nn.Linear before init_weights; skipping that draw here keeps the random sequence, and so
    the initial weights of the shared trunk, independent of whether the n-gram memory is present.
    """

    def reset_parameters(self):
        return None


class NGramMemory(nn.Module):
    """Hashed n-gram embeddings: signed lookups of every order and hash head, concatenated, then projected."""

    def __init__(self, config):
        super().__init__()
        self.rows, self.dim = config.option("ngram_rows"), config.option("ngram_dim")
        self.orders = tuple(config.option("ngram_orders"))
        self.heads = config.option("ngram_heads")
        self.signed = config.option("ngram_sign")
        self.pad = config.vocab_size                      # stands for the tokens before the start of the row
        self.count = len(self.orders) * self.heads
        # "aux": [rows, 1, dim], not a matrix and named `.index_*`, so the tables train under auxiliary AdamW, never
        # Muon. "embed": the model keeps the tables in `value_embeds` (the shell's embedding group) and passes them in.
        self.index_tables = nn.ParameterList(
            nn.Parameter(torch.empty(self.rows, 1, self.dim)) for _ in range(self.count)
        ) if config.option("ngram_group") == "aux" else None
        self.gain = float(config.option("ngram_gain"))
        total = self.count * self.dim
        self.proj = PlainLinear(total, config.width, bias=False) if config.option("ngram_project") else None
        self.value_mode = config.option("ngram_value")
        self.proj_value = PlainLinear(total, config.width, bias=False) if self.value_mode == "proj" else None

    def forward(self, idx, tables=None):
        B, T = idx.shape
        if tables is None: tables = [table.squeeze(1) for table in self.index_tables]
        reach = max(self.orders) - 1
        ext = torch.cat((idx.new_full((B, reach), self.pad), idx), 1)
        slots = [ext[:, reach - j: reach - j + T] for j in range(reach + 1)]     # slots[j]: the token j positions back
        parts = []
        for o, order in enumerate(self.orders):
            for h in range(self.heads):
                key = slots[0] * _MULTIPLIERS[h][0]
                for j in range(1, order):
                    key = key ^ (slots[j] * _MULTIPLIERS[h][j])
                key = (key ^ (key >> 31)) * _MIX
                key = (key ^ (key >> 29)) & _MASK
                e = F.embedding(key % self.rows, tables[o * self.heads + h])
                if self.signed:
                    sign = 1 - 2 * ((key >> 40) & 1)
                    e = e * sign.unsqueeze(-1).to(e.dtype)
                parts.append(e)
        raw = (torch.cat(parts, -1) if len(parts) > 1 else parts[0]).to(COMPUTE_DTYPE)
        # `ngram_gain` scales the residual path only; the value path sits behind the value-embedding gate.
        value = self.proj_value(raw) if self.value_mode == "proj" else raw if self.value_mode == "direct" else None
        e = raw * self.gain if self.gain != 1.0 else raw
        return (self.proj(e) if self.proj is not None else e), value


def _drawn_by_init_weights():
    """Stands in for nn.Embedding.reset_parameters: init_weights draws the embeddings."""


def grouped_depth_attention(query, sources, keys, groups, bias=None):
    """Depth read with one softmax over sources per channel group; `groups` = 1 without bias is `depth_attention`.

    `bias` is one scalar per source, added to that source's logits at every position and group.
    """
    b, t, d = keys[0].shape
    q = query.to(keys[0].dtype).view(groups, d // groups)
    logits = coefficient_precision(torch.stack([(key.reshape(b, t, groups, d // groups) * q).sum(-1) for key in keys]))
    if bias is not None: logits = logits + bias.to(logits.dtype)[:, None, None, None]
    weights = logits.softmax(0).to(sources[0].dtype)
    out = weights[0, ..., None] * sources[0].reshape(b, t, groups, d // groups)
    for weight, value in zip(weights[1:], sources[1:]):
        out = out + weight[..., None] * value.reshape(b, t, groups, d // groups)
    return out.reshape(b, t, d)


class NGramCanonAttnRes(ReferenceLM):
    """Full AttnRes trunk with Canon-A/C on every sublayer input and an optional hashed n-gram memory in b0."""

    def __init__(self, config):
        super().__init__(config)
        # One pseudo-query per attention and FFN sublayer, plus one for the final readout.
        self.depth_queries = nn.Parameter(torch.empty(2 * config.n_layer + 1, config.width))
        self.canon = nn.ModuleList(Canon(config.width, config.option("canon_size")) for _ in range(2 * config.n_layer))
        self.depth_heads = config.option("depth_heads")
        self.sublayer_sources = config.option("depth_sources") == "sublayer"
        self.softcap = float(config.option("softcap"))
        self.ngram = NGramMemory(config) if config.option("ngram_rows") > 0 else None
        self.ngram_keys = ()
        if self.ngram is not None and config.option("ngram_group") == "embed":
            # Non-numeric keys: no layer reads them as its value embedding; init_weights casts them like the others.
            self.ngram_keys = tuple(f"ngram{i}" for i in range(self.ngram.count))
            for key in self.ngram_keys:
                self.value_embeds[key] = nn.Embedding(self.ngram.rows, self.ngram.dim)
        self.ngram_inject = config.option("ngram_inject")
        self.ngram_source = self.ngram_inject == "source"
        self.canon_out = Canon(config.width, config.option("canon_size")) if config.option("canon_readout") else None
        self.canon_write = nn.ModuleList(
            Canon(config.width, config.option("canon_write_size")) for _ in range(config.n_layer)
        ) if config.option("canon_write") else None
        self.canon_embed = Canon(config.width, config.option("canon_write_size")) if config.option("canon_embed") else None
        self.depth_qkv = config.option("depth_qkv")
        if self.depth_qkv:
            # Pseudo-queries of the key and value reads of every attention sublayer, and their Canon-A layers.
            self.depth_queries_kv = nn.Parameter(torch.empty(config.n_layer, 2, config.width))
            self.canon_kv = nn.ModuleList(
                Canon(config.width, config.option("canon_size")) for _ in range(2 * config.n_layer))
        self.depth_bias_gain = float(config.option("depth_bias"))
        if self.depth_bias_gain > 0:
            # One scalar per (read, source); its name keeps it in the auxiliary AdamW group with the pseudo-queries.
            self.depth_queries_bias = nn.Parameter(torch.empty(2 * config.n_layer + 1, 2 * config.n_layer + 2))
        for module in self.modules():
            if isinstance(module, nn.Embedding): module.reset_parameters = _drawn_by_init_weights

    def _read(self, k, sources, keys, query=None):
        """Depth read number `k`; `query` replaces its pseudo-query (the key and value reads of an attention sublayer)."""
        if query is None: query = self.depth_queries[k]
        if self.depth_bias_gain > 0:
            bias = self.depth_queries_bias[k, :len(sources)] * self.depth_bias_gain
            return grouped_depth_attention(query, sources, keys, self.depth_heads, bias)
        if self.depth_heads == 1: return depth_attention(query, sources, keys)
        return grouped_depth_attention(query, sources, keys, self.depth_heads)

    def _read_kv(self, k, sources, keys):
        """The key and value inputs of attention sublayer `k` (even), each from its own depth read and Canon-A."""
        if not self.depth_qkv: return None, None
        j = k // 2
        hk = self.canon_kv[2 * j](self._read(k, sources, keys, self.depth_queries_kv[j, 0]))
        hv = self.canon_kv[2 * j + 1](self._read(k, sources, keys, self.depth_queries_kv[j, 1]))
        return hk, hv

    def _attend(self, h, idx, physical, value, hk=None, hv=None):
        """ReferenceLM._attention_branch, with the n-gram value projection added to the layer's value embedding."""
        c = self.config; t = idx.shape[1]
        ve = self.value_embeds[str(physical)](idx).to(h.dtype) if str(physical) in self.value_embeds else None
        if ve is not None and value is not None: ve = ve + value
        window = self._window(physical, c.n_layer)
        attn, rope = self.transformer.h[physical].attn, (self.cos[:, :t], self.sin[:, :t])
        if hk is None: out = attn(norm(h), ve, rope, window, None)
        else: out = attn(norm(h), ve, rope, window, None, xk=norm(hk), xv=norm(hv))
        self._record("block", physical=physical, virtual=physical, window=window[0], tokens=idx.numel(),
                     sequence_len=t, ve=ve is not None, mixer="softmax", residual=c.method)
        return out

    def _ordinary_hidden(self, idx):
        c = self.config
        x = self._embed_reference(idx)
        memory = value = None
        if self.ngram is not None:
            tables = [self.value_embeds[key].weight for key in self.ngram_keys] if self.ngram_keys else None
            memory, value = self.ngram(idx, tables)
            self._record("ngram_memory", tokens=idx.numel())
            if self.ngram_inject == "embed": x = x + memory
        if self.canon_embed is not None: x = self.canon_embed(x)
        # Depth sources that stay fixed: the n-gram memory (if it is its own source), then the embedding.
        fixed = [memory] if self.ngram is not None and self.ngram_source else []
        fixed_keys = [norm(source) for source in fixed]
        mlp = lambda h, physical: self.transformer.h[physical].mlp(norm(h))
        if self.sublayer_sources:
            # Every sublayer output is its own depth source (no block sums).
            sources, keys = fixed + [x], fixed_keys + [norm(x)]
            for k in range(2 * c.n_layer):
                h = self._read(k, sources, keys)
                self._record("depth_read", sublayer=k, sources=len(sources), tokens=idx.numel())
                h = self.canon[k](h)
                if k % 2 == 0:
                    hk, hv = self._read_kv(k, sources, keys)
                    out = self._attend(h, idx, k // 2, value, hk, hv)
                else: out = mlp(h, k // 2)
                sources.append(out); keys.append(norm(out))
            h = self._read(2 * c.n_layer, sources, keys)
            self._record("depth_read", sublayer=2 * c.n_layer, sources=len(sources), tokens=idx.numel())
        else:
            # Full AttnRes: sources are the embedding, every completed block sum and the current partial sum.
            blocks, keys, partial = fixed, fixed_keys, x
            for k in range(2 * c.n_layer):
                key = norm(partial)
                h = self._read(k, blocks + [partial], keys + [key])
                self._record("depth_read", sublayer=k, sources=len(blocks) + 1, tokens=idx.numel())
                hk, hv = self._read_kv(k, blocks + [partial], keys + [key]) if k % 2 == 0 else (None, None)
                if k % 2 == 0:
                    blocks.append(partial); keys.append(key); partial = None
                h = self.canon[k](h)
                out = self._attend(h, idx, k // 2, value, hk, hv) if k % 2 == 0 else mlp(h, k // 2)
                partial = out if partial is None else partial + out
                if k % 2 == 1 and self.canon_write is not None: partial = self.canon_write[k // 2](partial)
            h = self._read(2 * c.n_layer, blocks + [partial], keys + [norm(partial)])
            self._record("depth_read", sublayer=2 * c.n_layer, sources=len(blocks) + 1, tokens=idx.numel())
        if self.canon_out is not None: h = self.canon_out(h)
        return h, None      # no backout, as the residual family

    def _readout(self, x, backout):
        if backout is not None: x = x - self.backout_lambda.to(x.dtype) * backout
        logits = self.lm_head(norm(x))[..., :self.config.vocab_size].float()
        self._record("head", tokens=x.shape[0] * x.shape[1])
        return self.softcap * torch.tanh(logits / self.softcap)


def mixer(config, layer):
    return CanonAttention(config, layer)


def ffn(config, layer):
    return CanonMLP(config)


model = NGramCanonAttnRes
# The residual-family trunk does not use the shell's residual and x0 scalars or the backout.
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


@torch.no_grad()
def init(model):
    """Canon layers and the n-gram tables start at zero, so the model starts as plain Full AttnRes.

    ReferenceLM.init_weights already zeroes `depth_queries` (uniform reads). The n-gram projection is drawn here, after
    every shared-shell draw, with the shell's uniform rule for input projections.
    """
    for module in model.modules():
        if isinstance(module, Canon): nn.init.zeros_(module.weight)
    if model.depth_bias_gain > 0: nn.init.zeros_(model.depth_queries_bias)
    if model.depth_qkv: nn.init.zeros_(model.depth_queries_kv)
    if model.ngram is not None:
        tables = ([model.value_embeds[key].weight for key in model.ngram_keys] if model.ngram_keys
                  else list(model.ngram.index_tables))
        for table in tables: nn.init.zeros_(table)
        for proj in (model.ngram.proj, model.ngram.proj_value):
            if proj is not None:
                bound = 3 ** 0.5 * proj.weight.shape[1] ** -0.5
                nn.init.uniform_(proj.weight, -bound, bound)


def _integer(value, low, high):
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def check(config):
    for name in ("canon_size", "canon_write_size"):
        if not _integer(config.option(name), 1, 64):
            raise ValueError(f"{name} must be an integer from 1 to 64")
    heads = config.option("depth_heads")
    if not _integer(heads, 1, config.width) or config.width % heads:
        raise ValueError("depth_heads must divide the width")
    softcap = config.option("softcap")
    if isinstance(softcap, bool) or not isinstance(softcap, (int, float)) or not 1 <= softcap <= 100:
        raise ValueError("softcap must be a number from 1 to 100")
    if config.option("mlp_act") not in ACTIVATIONS:
        raise ValueError(f"mlp_act must be one of {', '.join(ACTIVATIONS)}")
    if config.option("depth_sources") not in DEPTH_SOURCES:
        raise ValueError(f"depth_sources must be one of {', '.join(DEPTH_SOURCES)}")
    if config.option("ngram_inject") not in NGRAM_INJECT:
        raise ValueError(f"ngram_inject must be one of {', '.join(NGRAM_INJECT)}")
    if config.option("ngram_group") not in NGRAM_GROUP:
        raise ValueError(f"ngram_group must be one of {', '.join(NGRAM_GROUP)}")
    if config.option("ngram_value") not in NGRAM_VALUE:
        raise ValueError(f"ngram_value must be one of {', '.join(NGRAM_VALUE)}")
    bias = config.option("depth_bias")
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not 0 <= bias <= 1000:
        raise ValueError("depth_bias must be a number from 0 (off) to 1000")
    if config.option("canon_write") is True and config.option("depth_sources") != "block":
        raise ValueError("canon_write applies to block depth sources")
    for name in ("canon_embed", "canon_readout", "canon_ve", "canon_write", "depth_qkv", "ngram_project", "ngram_sign"):
        if not isinstance(config.option(name), bool): raise ValueError(f"{name} must be true or false")
    gain = config.option("ngram_gain")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 < gain <= 1000:
        raise ValueError("ngram_gain must be a positive number")
    if not _integer(config.option("ngram_rows"), 0, 1 << 26):
        raise ValueError("ngram_rows must be an integer from 0 (no n-gram memory) to 2**26")
    if not _integer(config.option("ngram_dim"), 1, config.width):
        raise ValueError("ngram_dim must be an integer from 1 to the width")
    if not _integer(config.option("ngram_heads"), 1, len(_MULTIPLIERS)):
        raise ValueError(f"ngram_heads must be an integer from 1 to {len(_MULTIPLIERS)}")
    orders = config.option("ngram_orders")
    if not (isinstance(orders, (list, tuple)) and orders and all(_integer(n, 1, len(_MULTIPLIERS[0])) for n in orders)
            and len(set(orders)) == len(orders)):
        raise ValueError(f"ngram_orders must be distinct integers from 1 to {len(_MULTIPLIERS[0])}")
    total = len(orders) * config.option("ngram_heads") * config.option("ngram_dim")
    if config.option("ngram_rows") > 0:
        if not config.option("ngram_project") and config.option("ngram_inject") != "none" and total != config.width:
            raise ValueError("without ngram_project the concatenated lookups must have the model width")
        if config.option("ngram_value") == "direct" and total != config.width:
            raise ValueError("ngram_value direct needs concatenated lookups of the model width")
        if config.option("ngram_inject") == "none" and config.option("ngram_value") == "none":
            raise ValueError("the n-gram memory needs a residual or a value path")
        if config.option("ngram_inject") == "none" and config.option("ngram_project"):
            raise ValueError("ngram_inject none leaves the residual projection unused; set ngram_project false")
