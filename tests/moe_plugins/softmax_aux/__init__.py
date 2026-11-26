"""Test plugin: a router hook. Softmax gates, top-k, and the Switch load-balancing loss as `aux`."""
import torch
import torch.nn.functional as F

from moe.model import Router


class SoftmaxAuxRouter(Router):
    def forward(self, x):
        c = self.config
        probs = F.softmax(F.linear(x.float(), self.weight.float()), dim=-1)
        gates, index = torch.topk(probs, c.top_k, dim=-1)
        gates = gates / gates.sum(-1, keepdim=True)
        load = torch.bincount(index.flatten(), minlength=c.routed_experts).float()
        if self.training:
            self.load += load
        aux = c.option("aux_coeff") * c.routed_experts * (probs.mean(0) * load / load.sum()).sum()
        return index, gates, aux


def router(config, layer):
    return SoftmaxAuxRouter(config, layer)


def check(config):
    if not config.option("aux_coeff") >= 0:
        raise ValueError("aux_coeff must be nonnegative")
