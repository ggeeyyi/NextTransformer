"""Hyper-Connections on the MoE track (Zhu et al., arXiv 2409.19606, Eqs. 10-14 and Algorithm 2).

The residual state holds `streams` copies of width d. The embedding is replicated into every stream and the
streams are summed before the final norm and head. Every sublayer -- the attention and the MoE FFN of each
block, 2L in total -- reads `A_m^T H`, mixes the residual as `A_r^T H` and writes `B^T F` back into every
stream. A and B are the static coefficients plus a dynamic, width-conditioned correction; the dynamic
mapping's normalization is the parameter-free RMSNorm, as in the paper.

Both of a block's connections live inside the block, because moe/train.py shards one FSDP unit per block: a
unit's parameters are gathered by its own pre-forward hook, so a trunk that reached into `block.attn` from
the outside would read them while they are still sharded.

Two deviations from a literal transcription, both forced by this track's frozen optimizer (moe/optim.py):

1. The static coefficients, the two gate scales and the static write weights are stored as 1-D parameters, so
   `group_of` routes them to Adam. MuonH would otherwise orthogonalize an n x (n+1) table of mixing
   coefficients, which is not a map between feature spaces, and renormalize it to its initial Frobenius norm.
2. The dynamic projections start at N(0, init_std) rather than exactly zero. MuonH's hyperball update is
   `W' = W - lr U |W|/|U|` followed by `W <- W' |W|/|W'|`, so a matrix whose norm starts at zero stays zero
   for the whole run. With `gate_init` at 0.01 the first updates are still within rounding of the
   pre-norm-equivalent start the paper prescribes.
"""
import torch
from torch import nn
import torch.nn.functional as F

from moe.model import Masks, MoELM, segments


def coefficient_precision(x):
    """Residual coefficients in FP32 for BF16/FP32 activations; FP64 inputs stay FP64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


class HyperConnection(nn.Module):
    """Dynamic hyper-connection around one sublayer."""

    def __init__(self, width, streams, index, gate_init, init_std):
        super().__init__()
        self.streams, self.index, self.gate_init, self.init_std = streams, index, gate_init, init_std
        self.static_alpha = nn.Parameter(torch.empty(streams * (streams + 1)))
        self.static_beta = nn.Parameter(torch.empty(streams))
        self.dynamic_alpha = nn.Parameter(torch.empty(width, streams + 1))
        self.dynamic_beta = nn.Parameter(torch.empty(width, 1))
        self.scales = nn.Parameter(torch.empty(2))

    @torch.no_grad()
    def reset_parameters(self):
        """Pre-Norm-equivalent start: read stream `index mod n`, identity residual, unit writes."""
        n = self.streams
        static = torch.zeros(n, n + 1, device=self.static_alpha.device)
        static[self.index % n, 0] = 1.0
        static[:, 1:] = torch.eye(n, device=static.device)
        self.static_alpha.copy_(static.flatten())
        self.static_beta.fill_(1.0)
        nn.init.normal_(self.dynamic_alpha, std=self.init_std)
        nn.init.normal_(self.dynamic_beta, std=self.init_std)
        self.scales.fill_(self.gate_init)

    def read(self, streams):
        n = self.streams
        normed = rms(streams)
        alpha = (self.scales[0] * torch.tanh(coefficient_precision(normed @ self.dynamic_alpha.to(normed.dtype)))
                 + self.static_alpha.view(n, n + 1))
        beta = (self.scales[1] * torch.tanh(coefficient_precision(
            normed @ self.dynamic_beta.to(normed.dtype)).squeeze(-1)) + self.static_beta)
        mixed = torch.einsum("btjk,btjd->btkd", alpha.to(streams.dtype), streams)
        return mixed[:, :, 0], mixed[:, :, 1:], beta

    @staticmethod
    def write(residual, output, weights):
        return residual + weights.to(output.dtype)[..., None] * output[:, :, None]


class HyperConnectedBlock(nn.Module):
    """One block's attention and FFN, each wrapped in its own hyper-connection."""

    def __init__(self, block, config, layer):
        super().__init__()
        self.block = block
        self.attn_connection = self.connection(config, 2 * layer)
        self.mlp_connection = self.connection(config, 2 * layer + 1)

    @staticmethod
    def connection(config, index):
        return HyperConnection(config.width, config.option("streams"), index,
                               config.option("gate_init"), config.option("init_std"))

    def forward(self, streams, cos_sin, masks):
        block = self.block
        x, residual, weights = self.attn_connection.read(streams)
        output = block.attn(block.attn_gated(block.attn_norm(x)), cos_sin, masks)
        streams = self.attn_connection.write(residual, output, weights)
        x, residual, weights = self.mlp_connection.read(streams)
        return self.mlp_connection.write(residual, block.ffn(block.mlp_gated(block.mlp_norm(x))), weights)


class HyperConnectionLM(MoELM):
    """The reference trunk with an n-stream residual: replicate after the embedding, sum before the head."""

    def forward(self, idx, targets=None, reduction="mean"):
        T = idx.size(1)
        if T > self.cos.size(1):
            raise ValueError(f"input of {T} tokens exceeds the rotary table ({self.cos.size(1)})")
        masks = Masks(segments(idx, self.config.document_token), self.config)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = self.embed_gated(self.embed_norm(self.embed(idx)))
        streams = x[:, :, None].expand(-1, -1, self.config.option("streams"), -1)
        for block in self.blocks:
            streams = block(streams, cos_sin, masks)
        logits = self.head(self.final_gated(self.final_norm(streams.sum(2)))).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
                               reduction=reduction)
        aux = [m.last_aux for m in self.modules() if getattr(m, "last_aux", None) is not None]
        self.last_aux = sum(aux) if aux else None
        return loss


block = HyperConnectedBlock
model = HyperConnectionLM


def init(lm):
    for wrapped in lm.blocks:
        wrapped.attn_connection.reset_parameters()
        wrapped.mlp_connection.reset_parameters()


def check(config):
    streams = config.option("streams")
    if not isinstance(streams, int) or isinstance(streams, bool) or streams < 1:
        raise ValueError("streams must be a positive integer")
    for name in ("gate_init", "init_std"):
        value = config.option(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
