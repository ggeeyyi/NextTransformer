"""AttnRes with a mixture-of-experts MLP (16 experts, top-4).

Parent: attnres (att_000025). Every MLP becomes 16 squared-ReLU experts of width 1216, top-4 token choice with capacity 1.25 per sequence and a Switch load-balancing loss.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import Linear
from nexttransformer.models.lm import ReferenceLM
from nextt_gym.upstream_model import model_config, set_stage, stage_for_step  # noqa: F401


class MoEMLP(nn.Module):
    """Token-choice mixture of squared-ReLU experts with a fixed capacity per sequence.

    Each token picks its top-k experts by router probability; the k weights are renormalized to sum to one. Expert e
    takes at most C = ceil(capacity * T * k / E) assignments per sequence, filled in position order (and choice order
    within a token), so whether an assignment is dropped depends only on earlier positions: the layer stays causal.
    Shapes never depend on the data: every expert computes all C slots, and empty slots are charged. The experts are
    stored stacked, c_fc as one (E * w, d) matrix and c_proj as one (d, E * w) matrix, so the native initialization
    (c_fc uniform, c_proj zero) and the Muon matrix role apply unchanged. The Switch Transformer load-balancing loss
    (Fedus et al., 2022), weight `aux_weight`, is kept in `last_aux` during training.
    """

    def __init__(self, width, experts, top_k, expert_width, capacity, aux_weight):
        super().__init__()
        self.experts, self.top_k, self.expert_width = experts, top_k, expert_width
        self.capacity, self.aux_weight = capacity, aux_weight
        self.router = Linear(width, experts, bias=False)
        self.c_fc = Linear(width, experts * expert_width, bias=False)
        self.c_proj = Linear(experts * expert_width, width, bias=False)
        self.last_aux = None

    def forward(self, x):
        B, T, D = x.shape
        E, k, w = self.experts, self.top_k, self.expert_width
        C = math.ceil(self.capacity * T * k / E)
        probs = F.linear(x, self.router.weight.to(x.dtype)).float().softmax(-1)          # (B, T, E)
        expert = probs.topk(k, dim=-1).indices                                          # (B, T, k)
        weight = probs.gather(-1, expert)                                               # backward: scatter_add
        weight = weight / weight.sum(-1, keepdim=True)
        choice = F.one_hot(expert, E).view(B, T * k, E)                                 # position-major order
        slot = ((choice.cumsum(1) - 1) * choice).sum(-1)                                # rank within its expert
        dest = torch.where(slot < C, expert.view(B, T * k) * C + slot, E * C)            # E * C: dropped
        if self.training:
            share = choice.float().mean((0, 1))                                          # f_e: share of assignments
            self.last_aux = self.aux_weight * E * (share * probs.mean((0, 1))).sum()       # alpha when balanced
        # Slot -> source token (T is an appended zero row), then the experts on their C slots each.
        source = torch.full((B, E * C + 1), T, dtype=torch.long, device=x.device)
        source.scatter_(1, dest, torch.arange(T * k, device=x.device).div(k, rounding_mode="floor").expand(B, -1))
        padded = torch.cat((x, x.new_zeros(B, 1, D)), 1)
        h = padded.gather(1, source[:, :E * C, None].expand(-1, -1, D)).view(B, E, C, D)
        h = torch.einsum("becd,ewd->becw", h, self.c_fc.weight.to(x.dtype).view(E, w, D))
        h = F.relu(h).square()
        y = torch.einsum("becw,edw->becd", h, self.c_proj.weight.to(x.dtype).view(D, E, w).transpose(0, 1))
        y = torch.cat((y.reshape(B, E * C, D), y.new_zeros(B, 1, D)), 1)
        y = y.gather(1, dest[..., None].expand(-1, -1, D)) * weight.view(B, T * k, 1).to(x.dtype)
        return y.view(B, T, k, D).sum(2)


class MoEAttnResLM(ReferenceLM):
    """Block AttnRes (ReferenceLM, method attnres) with a mixture-of-experts MLP in every layer."""

    EXPERTS, TOP_K, CAPACITY, AUX_WEIGHT = 16, 4, 1.25, 0.01
    EXPERT_WIDTH = 1216

    def __init__(self, config):
        super().__init__(config)
        expert_width = self.EXPERT_WIDTH * config.width // 1536          # formal width 1536; smaller at debug scale
        for block in self.transformer.h:
            block.mlp = MoEMLP(config.width, self.EXPERTS, self.TOP_K, expert_width, self.CAPACITY, self.AUX_WEIGHT)

    def param_roles(self):
        """Routers are AdamW auxiliary parameters; the stacked experts keep the default matrix (Muon) role."""
        return {name: "aux" for name, _ in self.named_parameters() if ".mlp.router." in name}

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        for block in self.transformer.h: block.mlp.last_aux = None
        out = super().forward(idx, targets, kv_cache, loss_reduction)
        aux = [block.mlp.last_aux for block in self.transformer.h if block.mlp.last_aux is not None]
        if aux:
            self.last_aux_loss = torch.stack(aux).sum()
            if targets is not None and loss_reduction == "mean": out = out + self.last_aux_loss
        return out


def build(config):
    """As nexttransformer.models.lm.build_model, for this subclass; the caller has seeded the generator."""
    c = model_config(config)
    device = torch.get_default_device()
    with torch.device("meta"):
        model = MoEAttnResLM(c)
    if str(device) == "meta":
        return model
    model.to_empty(device=device)
    for module in model.modules():
        if isinstance(module, nn.Linear): module.reset_parameters()
    model.init_weights()
    return model
