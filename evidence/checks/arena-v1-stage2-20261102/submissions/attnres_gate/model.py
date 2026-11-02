"""AttnRes with head-wise gated attention output.

Parent: attnres (att_000025). Every attention output is gated per head.
"""
import torch
from torch import nn

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, GPTConfig, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import ReferenceLM
from nextt_gym.upstream_model import model_config, set_stage, stage_for_step  # noqa: F401


class GatedAttention(CausalSelfAttention):
    """B01 attention with a head-wise sigmoid gate on its output, before the output projection.

    Gated Attention (Qiu et al., 2025, "Gated Attention for Large Language Models"), head-wise variant: the gate is
    computed from the attention input. Its projection starts at zero, so every gate starts at 0.5; the output
    projection itself starts at zero (B01), so the initial model is unchanged.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.c_gate = Linear(self.n_embd, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("training and full-sequence evaluation only")
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y * torch.sigmoid(self.c_gate(x)).unsqueeze(-1).to(y.dtype)
        return self.c_proj(y.contiguous().view(B, T, -1))


class GatedLM(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        native = GPTConfig(sequence_len=config.sequence_len, vocab_size=config.vocab_size, n_layer=config.n_layer,
                           n_embd=config.width, n_head=config.heads, n_kv_head=config.heads,
                           window_pattern=config.window_pattern)
        for i, block in enumerate(self.transformer.h):
            block.attn = GatedAttention(native, i)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for block in self.transformer.h:
            nn.init.zeros_(block.attn.c_gate.weight)


def build(config):
    """As nexttransformer.models.lm.build_model, for this subclass; the caller has seeded the generator."""
    c = model_config(config)
    device = torch.get_default_device()
    with torch.device("meta"):
        model = GatedLM(c)
    if str(device) == "meta":
        return model
    model.to_empty(device=device)
    for module in model.modules():
        if isinstance(module, nn.Linear): module.reset_parameters()
    model.init_weights()
    return model
