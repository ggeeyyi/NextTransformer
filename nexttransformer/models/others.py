"""Reused from xxy/others-attention commit 8cba0a3 (MLA training implementation).

v2 Others: multi-head latent attention (MLA, MLA-NoPE).

MQA and GQA need no module here: the native attention already takes fewer KV
heads (`n_kv_head`), applies its RoPE and QK normalization to the shared keys,
and the fused attention kernel broadcasts each KV head over its query group.

MLA follows DeepSeek-V2 (arXiv 2405.04434v5 §2.1.2-2.1.3, official
`modeling_deepseek.py`): queries and keys/values are compressed into low-rank
latents, normalized, and expanded per head; RoPE lives on separate query
dimensions and on one key shared by all heads. Training reconstructs K and V
explicitly and calls the same fused attention as the native shell, so this is
the training path, not a slow oracle. Adaptations to the native shell:

- latent norms are the shell's parameter-free RMSNorm (no learned gains);
- RoPE uses the shell's rotate-half convention and base 100000 on the RoPE
  sub-dimension (the official code de-interleaves pairs; equivalent up to a
  fixed permutation of a learned projection);
- no per-head QK normalization: MLA normalizes its latents instead, and a norm
  of the reconstructed keys would forbid absorbing the key up-projection into
  the query. The softmax scale is (nope + rope) ** -0.5, as in DeepSeek-V2.
"""
import torch
from torch import nn

from nexttransformer.models.backbone import Linear, apply_rotary_emb, norm
from nanochat.flash_attention import flash_attn


class MLAAttention(nn.Module):
    """Multi-head latent attention with explicitly reconstructed keys and values.

    c_q:    query down-projection W^DQ,             width -> q_rank
    q_up:   query up-projection [W^UQ; W^QR],       q_rank -> heads * (nope + rope)
    c_k:    joint KV down-projection [W^DKV; W^KR], width -> kv_rank + rope (one RoPE key for all heads)
    c_v:    KV up-projection [W^UK; W^UV],          kv_rank -> heads * (nope + v)
    c_proj: output projection W^O,                  heads * v -> width

    The native initialization covers c_q and c_k (fan-in = width) and zeroes
    c_proj; `init_weights` then gives q_up and c_v the same uniform rule with
    their own fan-in, so every reconstructed query/key/value starts at unit scale.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.heads, self.nope, self.rope = config.heads, config.qk_nope_dim, config.qk_rope_dim
        self.v_dim, self.kv_rank = config.v_head_dim, config.kv_rank
        self.c_q = Linear(config.width, config.q_rank, bias=False)
        self.q_up = Linear(config.q_rank, self.heads * (self.nope + self.rope), bias=False)
        self.c_k = Linear(config.width, self.kv_rank + self.rope, bias=False)
        self.c_v = Linear(self.kv_rank, self.heads * (self.nope + self.v_dim), bias=False)
        self.c_proj = Linear(self.heads * self.v_dim, config.width, bias=False)
        self.ve_gate = None  # the Others family has no Value Embeddings

    @torch.no_grad()
    def init_weights(self):
        for proj in (self.q_up, self.c_v):
            bound = 3**.5 * proj.weight.shape[1]**-.5
            nn.init.uniform_(proj.weight, -bound, bound)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if ve is not None:
            raise ValueError("MLA has no Value Embedding interface")
        if kv_cache is not None:
            raise NotImplementedError("the latent KV cache is not part of the training reference")
        b, t, _ = x.shape
        h = self.heads
        q = self.q_up(norm(self.c_q(x))).view(b, t, h, self.nope + self.rope)
        compressed = self.c_k(x)
        latent = compressed[..., :self.kv_rank]
        k_nope, v = self.c_v(norm(latent)).view(b, t, h, self.nope + self.v_dim).split((self.nope, self.v_dim), -1)
        if self.rope:
            cos, sin = cos_sin
            q = torch.cat((q[..., :self.nope], apply_rotary_emb(q[..., self.nope:], cos, sin)), -1)
            k_rope = apply_rotary_emb(compressed[..., self.kv_rank:].view(b, t, 1, self.rope), cos, sin)
            k = torch.cat((k_nope, k_rope.expand(b, t, h, self.rope)), -1)
        else:
            k = k_nope
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(b, t, h * self.v_dim))
