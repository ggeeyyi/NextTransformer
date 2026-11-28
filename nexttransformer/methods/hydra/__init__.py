"""Hydra: Full AttnRes + Canon trunk with an optional causal-capacity mixture-of-experts FFN."""
from .model import HydraLM, Canon
from .memory import NgramMemory
from .layers import CanonAttention, make_ffn, init_ffn
import torch
from torch import nn

model = HydraLM
# The AttnRes trunk does not use the shell's residual and x0 scalars or the backout.
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


def mixer(config, layer):
    return CanonAttention(config, layer)


def ffn(config, layer):
    return make_ffn(config, layer)


@torch.no_grad()
def init(model):
    for module in model.modules():
        if isinstance(module, Canon):
            nn.init.zeros_(module.weight)
    init_ffn(model)
    if getattr(model, "memory", None) is not None:
        for key in model.memory.keys():
            model.value_embeds[key].weight.normal_(0.0, float(model.config.option("ngram_init")))
        nn.init.zeros_(model.memory.ngram_proj.weight)


def check(config):
    if config.option("value_embeds") not in ("shell", "none"):
        raise ValueError("value_embeds must be shell or none")
    orders = config.option("ngram_orders")
    if not (isinstance(orders, (list, tuple)) and all(isinstance(n, int) and 2 <= n <= 4 for n in orders)
            and len(orders) <= 4 and len(set(orders)) == len(orders)):
        raise ValueError("ngram_orders must be a list of distinct integers in 2..4")
    if orders:
        if not (isinstance(config.option("ngram_rows"), int) and config.option("ngram_rows") > 1):
            raise ValueError("ngram_rows must be an integer > 1")
        if not (isinstance(config.option("ngram_dim"), int) and config.option("ngram_dim") > 0):
            raise ValueError("ngram_dim must be a positive integer")
        heads = config.option("ngram_heads")
        if not (isinstance(heads, int) and 1 <= heads <= 8 and config.option("ngram_dim") % heads == 0
                and len(orders) * heads <= 8):
            raise ValueError("ngram_heads must divide ngram_dim, with at most 8 tables in total")
        if config.option("ngram_inject") not in ("source", "embed"):
            raise ValueError("ngram_inject must be source or embed")
    E = config.option("moe_experts")
    k = config.option("moe_topk")
    if not (isinstance(E, int) and E >= 0):
        raise ValueError("moe_experts must be a non-negative integer")
    if E > 0:
        if not (isinstance(k, int) and 1 <= k <= E):
            raise ValueError("moe_topk must be in 1..moe_experts")
        for name in ("shared_width", "expert_width"):
            w = config.option(name)
            if not (isinstance(w, int) and w > 0 and w % 16 == 0):
                raise ValueError(f"{name} must be a positive multiple of 16")
        if not float(config.option("moe_capacity")) > 0:
            raise ValueError("moe_capacity must be positive")
