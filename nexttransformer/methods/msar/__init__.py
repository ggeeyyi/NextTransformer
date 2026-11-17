"""Multi-Stream Attention Residuals (MSAR), arXiv:2610.13630."""

import torch
from torch import nn

from nexttransformer.models.backbone import norm
from nexttransformer.models.lm import ReferenceLM
from .ops import DepthRead, TemporalAugmentation, source_key, completed_read, merge_partial


frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


def check(config):
    size = config.option("block_size_layers")
    if type(size) is not int or size < 1:
        raise ValueError("block_size_layers must be a positive integer")
    if config.window_pattern != "L":
        raise ValueError("MSAR retains full causal attention in every layer")


class MSARModel(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        width = config.width
        self.block_size_layers = config.option("block_size_layers")
        self.block_sizes = tuple(min(self.block_size_layers, config.n_layer - start)
                                 for start in range(0, config.n_layer, self.block_size_layers))
        # The existing auxiliary-AdamW namespace owns queries and HPre.
        # Read 0 is absent: the first attention sees the shell embedding directly.
        self.residual_maps = nn.ModuleList(DepthRead(width)
                                          for _ in range(2 * config.n_layer - 1))
        self.msar_initial = TemporalAugmentation(width)
        self.msar_writes = nn.ModuleList(TemporalAugmentation(width)
                                        for _ in range(2 * config.n_layer - 1))

    def _ordinary_hidden(self, idx):
        embedding = self._embed_reference(idx)
        initial = self.msar_initial(embedding)
        completed = [initial]
        keys = [source_key(initial)]
        event = 0
        for layers in self.block_sizes:
            stop = event + 2 * layers
            first_read = max(event, 1)
            queries = torch.stack([self.residual_maps[k - 1].query for k in range(first_read, stop)])
            retrieved, lse = completed_read(queries, completed, keys)
            # One unbind backward collects consumers without one full zero/scatter per read.
            retrieved, lse = retrieved.unbind(0), lse.unbind(0)
            raw, partial = None, None
            for step in range(event, stop):
                if step == 0:
                    hidden = embedding
                else:
                    read = self.residual_maps[step - 1]
                    streams = retrieved[step - first_read]
                    if partial is not None:
                        partial_key = source_key(partial)
                        streams = merge_partial(streams, lse[step - first_read],
                                                partial, partial_key, read.query)
                    hidden = read(streams)
                    self._record("depth_read", sublayer=step, sources=len(completed) + int(partial is not None),
                                 streams=4, tokens=idx.numel())
                output = (self._attention_branch(hidden, idx, step // 2) if step % 2 == 0
                          else self.transformer.h[step // 2].mlp(norm(hidden)))
                if step == 2 * self.config.n_layer - 1:
                    # last_bda is the pre-norm MLP input plus its delta, not a new depth read.
                    return hidden + output, None
                raw = output if raw is None else raw + output
                # Augment the new raw sum with THIS writer's taps, never sum augmented deltas.
                partial = self.msar_writes[step](raw)
            completed.append(partial)
            keys.append(source_key(partial))
            event = stop
        raise RuntimeError("MSAR did not execute its terminal MLP")


model = MSARModel


@torch.no_grad()
def init(model):
    """Called after shared initialization and meta-to-empty materialization."""
    model.msar_initial.reset_parameters()
    for augmentation in model.msar_writes:
        augmentation.reset_parameters()
