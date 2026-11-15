"""Small CPU reference blocks for contributors to the 2K scaling study.

Run `python -m nexttransformer.scaling.reference_blocks` in the project environment.
The examples exercise training math on synthetic tokens. Distributed dispatch,
throughput, cost proofs and formal optimizer recipes belong to the T3 integration.
"""
from types import SimpleNamespace

import torch
from torch import nn

from nexttransformer.models import build_model
from nexttransformer.models.backbone import Linear, norm
from nexttransformer.models.config import debug_config
from nexttransformer.models.linear import LinearMixer
from nexttransformer.models.lm import FlexibleMLP
from nexttransformer.models.sparse import SparseMixer


class RoutedFFN(nn.Module):
    """A shared ReLU² expert and a normalized top-k mixture of routed ReLU² experts."""

    def __init__(self, config, num_experts, top_k, expert_width):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        self.top_k = top_k
        self.router = Linear(config.width, num_experts, bias=False)
        if expert_width <= 0:
            raise ValueError("expert_width must be positive")
        expert_config = SimpleNamespace(width=config.width, mlp_width=expert_width)
        self.shared = FlexibleMLP(expert_config)
        self.experts = nn.ModuleList(FlexibleMLP(expert_config) for _ in range(num_experts))

    def forward(self, hidden):
        rows = hidden.flatten(0, -2)
        logits, indices = self.router(rows.float()).topk(self.top_k)
        weights = logits.softmax(dim=-1)
        output = self.shared(rows)
        for index, expert in enumerate(self.experts):
            token, slot = torch.where(indices == index)
            if token.numel():
                routed = expert(rows[token]) * weights[token, slot, None].to(rows.dtype)
                output = output.index_add(0, token, routed)
        return output.reshape_as(hidden)


class KdaDsaGroup(nn.Module):
    """A four-layer, dense-FFN reference group reusing the project's KDA and DSA."""

    def __init__(self, config):
        super().__init__()
        self.mixers = nn.ModuleList(LinearMixer(config, i, kind="kda") for i in range(3))
        self.mixers.append(SparseMixer(config, 3, kind="dsa"))
        for mixer in self.mixers:
            mixer.ve_gate = None
        self.ffns = nn.ModuleList(FlexibleMLP(config) for _ in self.mixers)
        self.burnin_steps = config.burnin_steps

    def forward(self, hidden, cos_sin, step):
        sparse = self.mixers[-1]
        sparse.in_burnin = step < self.burnin_steps
        for mixer, ffn in zip(self.mixers, self.ffns):
            hidden = hidden + mixer(norm(hidden), None, cos_sin, (-1, 0), None)
            hidden = hidden + ffn(norm(hidden))
        return hidden, sparse.last_aux


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    for method in ("mha_no_ve", "fixed_recursive"):
        config = debug_config(method)
        model = build_model(config)
        tokens = torch.randint(config.vocab_size, (2, config.sequence_len + 1))
        loss = model(tokens[:, :-1], targets=tokens[:, 1:])
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {method}")
        print(f"{method}: benchmark backbone completed forward and backward")

    config = debug_config("dsa")
    hidden = torch.randn(2, config.sequence_len, config.width)
    # Reduced expert counts keep the CPU example small. T3 uses 256 / 8 / 1.
    moe = RoutedFFN(config, num_experts=8, top_k=2, expert_width=config.width // 4)
    moe(hidden).square().mean().backward()
    print("MoE: shared and selected experts completed forward and backward")
    hybrid = KdaDsaGroup(config)
    frequency = 10000 ** (-torch.arange(0, config.head_dim, 2) / config.head_dim)
    angle = torch.arange(config.sequence_len)[:, None] * frequency
    rotary = (angle.cos()[None, :, None], angle.sin()[None, :, None])
    for step in (0, config.burnin_steps):
        hybrid.zero_grad(set_to_none=True)
        output, index_loss = hybrid(hidden, rotary, step)
        loss = output.square().mean() + index_loss
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite reference loss")
        print(f"KDA–DSA: step {step}, forward and backward complete")


if __name__ == "__main__":
    main()
