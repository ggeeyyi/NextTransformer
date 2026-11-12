"""Gated Attention (Qiu et al., arXiv 2505.06708): a query-dependent sigmoid gate on each head's SDPA output.

The Dense baseline's causal attention with the paper's gate after scaled dot-product attention and
before the output projection, multiplicative, head-specific and sigmoid,

    Y' = Y ⊙ σ(X W_θ)

where Y is the per-head attention output and X the block's pre-normalized input, the same input as q, k and v.
`granularity` "elementwise" gives one score per head and channel (the paper's best row; Qwen3-Next and Qwen3.5
use this layout), "headwise" one score per head. As in the official code (github.com/qiuzh20/gated_attention,
modeling_qwen3.py), the gate has no bias and no extra scale.

Adaptation to the shell: the official code fuses W_θ into q_proj and trains every weight with AdamW. Here W_θ is
the separate bias-free linear layer `output_gate`, which computes the same function; its name places it in the
auxiliary AdamW group (training/optimizer.py), like the output gates of the linear-attention methods, while
c_q/c_k/c_v/c_proj stay with Muon. Fusing it into c_q would hand the gate to Muon instead. It is initialized
with c_q's uniform rule, which is what the fused layout gives the gate rows. Value embeddings, RoPE, QK norm, the
attention scale and the zero-initialized c_proj are the Dense baseline's.
"""
import torch
from torch import nn

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm

GRANULARITY = ("elementwise", "headwise")


class GatedAttention(CausalSelfAttention):
    """CausalSelfAttention whose per-head output is multiplied by σ(output_gate(x)) before c_proj.

    output_gate: width -> heads * head_dim (elementwise) or width -> heads (headwise)
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.gate_channels = self.head_dim if config.option("granularity") == "elementwise" else 1
        self.output_gate = Linear(self.n_embd, self.n_head * self.gate_channels, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Identical to CausalSelfAttention.forward up to the gate before c_proj.
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v, cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        # Gate each head's SDPA output, (B, T, H, D) * (B, T, H, D or 1)
        y = y * torch.sigmoid(self.output_gate(x)).view(B, T, self.n_head, self.gate_channels)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    # c_q's rule in the shell: uniform with the std of 1/sqrt(width).
    s = 3**0.5 * model.config.width**-0.5
    for block in model.transformer.h:
        nn.init.uniform_(block.attn.output_gate.weight, -s, s)


def check(config):
    if config.option("granularity") not in GRANULARITY:
        raise ValueError(f"granularity must be one of {', '.join(GRANULARITY)}")
