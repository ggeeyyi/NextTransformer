"""AttnRes with a value embedding in every layer.

Parent: attnres (att_000025). Every layer gets a value embedding and its gate.
"""
import torch
from torch import nn

from nexttransformer.models.backbone import Linear
from nexttransformer.models.lm import ReferenceLM
from nextt_gym.upstream_model import model_config, set_stage, stage_for_step  # noqa: F401


class AllValueEmbeddingLM(ReferenceLM):
    """A value embedding and its per-head gate in every layer, instead of every other layer."""

    def __init__(self, config):
        super().__init__(config)
        rows, width = next(iter(self.value_embeds.values())).weight.shape
        for i, block in enumerate(self.transformer.h):
            if str(i) not in self.value_embeds:
                self.value_embeds[str(i)] = nn.Embedding(rows, width)
                block.attn.ve_gate = Linear(block.attn.ve_gate_channels, block.attn.n_kv_head, bias=False)


def build(config):
    """As nexttransformer.models.lm.build_model, for this subclass; the caller has seeded the generator."""
    c = model_config(config)
    device = torch.get_default_device()
    with torch.device("meta"):
        model = AllValueEmbeddingLM(c)
    if str(device) == "meta":
        return model
    model.to_empty(device=device)
    for module in model.modules():
        if isinstance(module, nn.Linear): module.reset_parameters()
    model.init_weights()
    return model
