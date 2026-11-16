"""Exclusive Self Attention (Zhai, arXiv:2603.09078v1), Eq. 2 and Algorithm 1.

Ordinary causal attention still includes the self key in softmax. Before c_proj,
each head removes the output component along its current token's value:
u = normalize(v); z = y - sum(y * u) * u. Gradients flow through both y and v.

The shared shell retains RoPE, QK norm, attention scaling and value embeddings.
The exclusion direction uses the same effective v (including the gated value
embedding) as attention. Half-precision inputs use FP32 projection arithmetic;
FP32 and FP64 preserve their precision. No extra parameters or attention kernel.
"""
import torch
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, apply_rotary_emb, norm


def exclusive_projection(y, v):
    """Algorithm 1 on per-head (..., head_dim) tensors, with normalize's default epsilon."""
    dtype = y.dtype
    if dtype in (torch.float16, torch.bfloat16):
        y, v = y.float(), v.float()
    vn = F.normalize(v, dim=-1, eps=1e-12)
    return (y - (y * vn).sum(dim=-1, keepdim=True) * vn).to(dtype)


class ExclusiveSelfAttention(CausalSelfAttention):
    """The shared attention module with XSA's per-head projection before c_proj."""

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        batch, length, _ = x.shape
        q = self.c_q(x).view(batch, length, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, length, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch, length, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(batch, length, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v, cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(length)

        # The protocol uses MHA. Match each query head to its KV head for direct GQA use as well.
        if self.n_head != self.n_kv_head:
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        z = exclusive_projection(y, v)
        return self.c_proj(z.contiguous().view(batch, length, self.n_embd))


def mixer(config, layer):
    return ExclusiveSelfAttention(config, layer)


def check(config):
    if config.window_pattern != "L":
        raise ValueError("XSA uses full causal attention in every layer (window_pattern L)")
