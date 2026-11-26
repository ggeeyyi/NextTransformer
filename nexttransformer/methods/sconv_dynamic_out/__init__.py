"""Dynamic K/V with static attention-output and MLP-output short convolutions."""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, GPTConfig, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import FlexibleMLP, ReferenceLM


class ShortConvolution(nn.Module):
    """Residual causal depthwise filter, with independent packed rows."""

    def __init__(self, width, size):
        super().__init__()
        self.size = size
        # Non-matrix parameters use the shared optimizer's AdamW rule.
        self.weight = nn.Parameter(torch.empty(width, 1, size))

    def forward(self, x):
        channels = F.pad(x.transpose(1, 2), (self.size - 1, 0))
        mixed = F.conv1d(channels, self.weight.to(x.dtype), groups=x.shape[-1])
        return x + mixed.transpose(1, 2)


class ZeroLinear(Linear):
    """Bias-free generator whose construction and reset consume no RNG draws."""

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class DynamicShortConvolution(ShortConvolution):
    """S-out static arithmetic plus current-position, head-shared dynamic taps.

    Static weight[..., size - 1 - lag] and generator[..., lag] both address
    z[t - lag]. The independent dynamic sum uses FP32 for BF16/FP16 inputs;
    the existing static convolution and residual keep their input precision.
    """

    def __init__(self, width, size, heads):
        super().__init__(width, size)
        self.heads = heads
        self.weight_proj = ZeroLinear(width, heads * size, bias=False)

    def forward(self, z, x):
        batch, time, width = z.shape
        static = super().forward(z)
        accumulation_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        taps = self.weight_proj(x).to(accumulation_dtype).view(batch, time, self.heads, self.size)
        context = F.pad(z.to(accumulation_dtype), (0, 0, self.size - 1, 0))
        dynamic = None
        for lag in range(self.size):
            start = self.size - 1 - lag
            shifted = context[:, start:start + time].reshape(batch, time, self.heads, width // self.heads)
            term = taps[..., lag, None] * shifted
            dynamic = term if dynamic is None else dynamic + term
        return static + dynamic.reshape(batch, time, width).to(z.dtype)


class SConvAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        native = GPTConfig(sequence_len=config.sequence_len, vocab_size=config.vocab_size,
                           n_layer=config.n_layer, n_embd=config.width, n_head=config.heads,
                           n_kv_head=config.heads, window_pattern=config.window_pattern)
        super().__init__(native, layer)
        size = config.option("kernel_size")
        self.k_sconv = DynamicShortConvolution(self.n_kv_head * self.head_dim, size, self.n_kv_head)
        self.v_sconv = DynamicShortConvolution(self.n_kv_head * self.head_dim, size, self.n_kv_head)
        self.out_sconv = ShortConvolution(config.width, size)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("sconv_dynamic_out supports training and full-sequence evaluation only")
        batch, time, _ = x.shape
        q = self.c_q(x).view(batch, time, self.n_head, self.head_dim)
        k = self.k_sconv(self.c_k(x), x).view(batch, time, self.n_kv_head, self.head_dim)
        v = self.v_sconv(self.c_v(x), x).view(batch, time, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(batch, time, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.out_sconv(self.c_proj(y.contiguous().view(batch, time, -1)))


class SConvMLP(FlexibleMLP):
    def __init__(self, config):
        super().__init__(config)
        self.out_sconv = ShortConvolution(config.width, config.option("kernel_size"))

    def forward(self, x):
        return self.out_sconv(super().forward(x))


def _initialized_by_shell():
    """ReferenceLM.init_weights already initializes embeddings."""


class SConvLM(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        # Match the built-in Dense initialization without extra embedding RNG draws.
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.reset_parameters = _initialized_by_shell

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is not None:
            raise NotImplementedError("sconv_dynamic_out requires full-sequence evaluation")
        return super().forward(idx, targets, kv_cache, loss_reduction)


model = SConvLM


def mixer(config, layer):
    return SConvAttention(config, layer)


def ffn(config, layer):
    return SConvMLP(config)


@torch.no_grad()
def init(model):
    for module in model.modules():
        if isinstance(module, ShortConvolution):
            nn.init.zeros_(module.weight)
        if isinstance(module, ZeroLinear):
            module.reset_parameters()


def check(config):
    size = config.option("kernel_size")
    if type(size) is not int or size < 1:
        raise ValueError("kernel_size must be a positive integer")
    if config.window_pattern != "L":
        raise ValueError("sconv_dynamic_out uses full causal attention in every layer")
