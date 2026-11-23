import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm
from .model import Canon


class CanonAttention(CausalSelfAttention):
    """The shell's attention with a Canon layer on each of q, k and v (Canon-B)."""

    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.canon_qkv = nn.ModuleList(Canon(self.n_embd) for _ in range(3))

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("training and full-sequence evaluation only")
        B, T, C = x.size()
        q, k, v = (layer(t) for layer, t in zip(self.canon_qkv, (self.c_q(x), self.c_k(x), self.c_v(x))))
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class CanonMLP(nn.Module):
    """Squared-ReLU MLP with a Canon layer on its hidden activations (Canon-D)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.width, config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)
        self.canon = Canon(config.mlp_width)

    def forward(self, x):
        return self.c_proj(F.relu(self.canon(self.c_fc(x))).square())


class MoEFFN(nn.Module):
    """Shared Canon-D expert plus top-k routed squared-ReLU experts with causal per-row capacity.

    A token's routed choice is kept iff fewer than `capacity` earlier tokens of the same row chose that expert,
    so the output at position t depends on positions <= t of its own row only. Capacity is fixed by the
    configured context, not by the input length. Dropped choices contribute nothing (the shared expert and
    the residual path still carry the token).
    """

    def __init__(self, config, layer):
        super().__init__()
        d = config.width
        self.E = config.option("moe_experts")
        self.k = config.option("moe_topk")
        shared = config.option("shared_width")
        he = config.option("expert_width")
        self.aux_weight = float(config.option("moe_aux"))
        self.capacity = int(math.ceil(float(config.option("moe_capacity")) * self.k * config.sequence_len / self.E))
        # The shell initializes c_fc / c_proj; here they are the shared expert.
        self.c_fc = Linear(d, shared, bias=False)
        self.c_proj = Linear(shared, d, bias=False)
        self.canon = Canon(shared)
        self.router = Linear(d, self.E, bias=False)
        self.w_in = nn.ModuleList(Linear(d, he, bias=False) for _ in range(self.E))
        self.w_out = nn.ModuleList(Linear(he, d, bias=False) for _ in range(self.E))
        self.last_aux = None

    def forward(self, x):
        B, T, D = x.shape
        E, K = self.E, self.k
        y = self.c_proj(F.relu(self.canon(self.c_fc(x))).square())
        logits = self.router(x).float()                                   # (B,T,E)
        probs = logits.softmax(-1)
        experts = torch.arange(E, device=x.device)
        masked, picks, values = probs, [], []
        for _ in range(K):                                                # top-k by repeated argmax
            choice = masked.argmax(-1)                                    # (B,T)
            hit = choice.unsqueeze(-1) == experts                         # (B,T,E)
            picks.append(choice)
            values.append(probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1))
            masked = torch.where(hit, torch.full_like(masked, -1.0), masked)
        top_i = torch.stack(picks, -1)                                    # (B,T,K)
        top_p = torch.stack(values, -1)
        gate = top_p / top_p.sum(-1, keepdim=True) if K > 1 else top_p
        onehot = (top_i.unsqueeze(-1) == experts).long()                  # (B,T,K,E)
        assign = onehot.sum(2)                                            # (B,T,E) in {0,1}
        before = assign.cumsum(1) - assign                                # earlier same-row tokens per expert
        rank = (before.unsqueeze(2) * onehot).sum(-1)                     # (B,T,K)
        C = min(self.capacity, T)
        keep = rank < C
        rows = torch.arange(B, device=x.device).view(B, 1, 1)
        dump = E * B * C
        slot = torch.where(keep, (top_i * B + rows) * C + rank, torch.full_like(rank, dump))  # (B,T,K)
        tokens = torch.arange(B * T, device=x.device).view(B, T, 1).expand(B, T, K)
        source = torch.full((dump + 1,), B * T, dtype=torch.long, device=x.device)
        source = source.scatter(0, slot.reshape(-1), tokens.reshape(-1))[:dump]
        padded = torch.cat((x.reshape(B * T, D), x.new_zeros(1, D)), 0)
        xin = padded[source].view(E, B * C, D)
        w_in = torch.stack([w.weight.to(x.dtype) for w in self.w_in])     # (E, he, D)
        w_out = torch.stack([w.weight.to(x.dtype) for w in self.w_out])   # (E, D, he)
        hidden = F.relu(torch.bmm(xin, w_in.transpose(1, 2))).square()   # (E, B*C, he)
        outs = torch.bmm(hidden, w_out.transpose(1, 2))                   # (E, B*C, D)
        yout = torch.cat((outs.reshape(E * B * C, D), x.new_zeros(1, D)), 0)  # (E*B*C+1, D)
        picked = yout[slot.reshape(-1)].view(B, T, K, D)
        weight = (gate * keep).to(x.dtype)
        y = y + (picked * weight.unsqueeze(-1)).sum(2)
        if self.training and self.aux_weight > 0:
            frac = assign.float().mean((0, 1)) / K                        # fraction of choices per expert
            self.last_aux = self.aux_weight * E * (frac.detach() * probs.mean((0, 1))).sum()
        else:
            self.last_aux = None
        return y


def make_ffn(config, layer):
    if config.option("moe_experts") > 0:
        return MoEFFN(config, layer)
    return CanonMLP(config)


@torch.no_grad()
def init_ffn(model):
    s = 3 ** 0.5 * model.config.width ** -0.5
    for block in model.transformer.h:
        mlp = block.mlp
        if isinstance(mlp, MoEFFN):
            for w in mlp.w_in:
                nn.init.uniform_(w.weight, -s * 0.4, s * 0.4)
            for w in mlp.w_out:
                nn.init.zeros_(w.weight)
            nn.init.normal_(mlp.router.weight, std=float(model.config.option("router_init")))
