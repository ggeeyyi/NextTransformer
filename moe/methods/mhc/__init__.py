"""Manifold-constrained Hyper-Connections on the MoE track (Xie et al., arXiv 2512.24880).

The residual state holds `streams` copies of width d: the embedding is replicated into every stream and the
streams are summed before the final norm and head. For each of the 2L sublayers (attention and MoE FFN of
every block) the flattened n x d state is RMS-normalized and one projection yields the pre, post and residual
logits. `H_pre = sigmoid` selects the sublayer input, `H_post = 2 sigmoid` scales the write, and `H_res` is
an approximation to doubly stochastic residual mixing obtained with the configured number of Sinkhorn-Knopp iterations.

Both of a block's connections live inside the block, because moe/train.py shards one FSDP unit per block: a
unit's parameters are gathered by its own pre-forward hook, so a trunk that reached into `block.attn` from
the outside would read them while they are still sharded.

Two deviations from a literal transcription, both forced by this track's frozen optimizer (moe/optim.py):

1. The static biases and the three gate scales are 1-D, so `group_of` routes them to Adam rather than MuonH,
   whose hyperball update would orthogonalize and renormalize tables of mixing coefficients.
2. The projection starts at N(0, init_std) rather than exactly zero. MuonH's hyperball keeps every matrix at
   its initial Frobenius norm, so a zero-initialized projection would never move for the whole run. With
   `gate_init` at 0.01 the maps still start within rounding of the HC Eq. 14 values the biases encode.
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


def sinkhorn_knopp(logits, iterations):
    """Approximate doubly stochastic projection of exp(logits) over the last two dimensions (Eq. 9).

    Each iteration rescales columns, then rows, so rows sum to one exactly and columns approximately. The
    iterates stay logarithmic: subtracting a log-sum-exp is the division by the sum, so a row whose entries
    all underflow under FP32 exp never becomes 0/0.
    """
    def normalize(log, dim):
        centered = log - log.amax(dim, keepdim=True).detach()
        return centered - centered.exp().sum(dim, keepdim=True).log()

    log = logits
    for _ in range(iterations):
        log = normalize(normalize(log, -2), -1)
    return log.exp()


class ManifoldHyperConnection(nn.Module):
    """Manifold-constrained hyper-connection around one sublayer."""

    def __init__(self, width, streams, index, gate_init, init_std, iterations):
        super().__init__()
        self.streams, self.index, self.gate_init = streams, index, gate_init
        self.init_std, self.iterations = init_std, iterations
        self.projection = nn.Parameter(torch.empty(streams * width, streams * streams + 2 * streams))
        self.bias = nn.Parameter(torch.empty(streams * streams + 2 * streams))
        self.gates = nn.Parameter(torch.empty(3))

    @torch.no_grad()
    def reset_parameters(self):
        """Gate scales as the paper fixes them; static biases start at HC's Eq. 14 values."""
        n = self.streams
        nn.init.normal_(self.projection, std=self.init_std)
        self.bias.zero_()
        self.bias[self.index % n] = 1.0
        self.bias[n:2 * n] = 1.0
        self.bias[2 * n:] = torch.eye(n, device=self.bias.device).flatten()
        self.gates.fill_(self.gate_init)

    def maps(self, streams):
        """(H_pre [B,T,n], H_post [B,T,n], H_res [B,T,n,n]) in coefficient precision."""
        b, t, n, d = streams.shape
        flat = rms(streams.reshape(b, t, n * d))
        logits = coefficient_precision(flat @ self.projection.to(flat.dtype))
        pre, post, res = logits.split((n, n, n * n), -1)
        bias_pre, bias_post, bias_res = self.bias.split((n, n, n * n))
        pre = torch.sigmoid(self.gates[0] * pre + bias_pre)
        post = 2 * torch.sigmoid(self.gates[1] * post + bias_post)
        res = sinkhorn_knopp((self.gates[2] * res + bias_res).unflatten(-1, (n, n)), self.iterations)
        return pre, post, res

    def read(self, streams):
        pre, post, res = self.maps(streams)
        layer_input = torch.einsum("btj,btjd->btd", pre.to(streams.dtype), streams)
        mixed = torch.einsum("btij,btjd->btid", res.to(streams.dtype), streams)
        return layer_input, mixed, post

    @staticmethod
    def write(residual, output, weights):
        return residual + weights.to(output.dtype)[..., None] * output[:, :, None]


class ManifoldHyperConnectedBlock(nn.Module):
    """One block's attention and FFN, each wrapped in its own manifold-constrained connection."""

    def __init__(self, block, config, layer):
        super().__init__()
        self.block = block
        self.attn_connection = self.connection(config, 2 * layer)
        self.mlp_connection = self.connection(config, 2 * layer + 1)

    @staticmethod
    def connection(config, index):
        return ManifoldHyperConnection(config.width, config.option("streams"), index,
                                       config.option("gate_init"), config.option("init_std"),
                                       config.option("sinkhorn_iterations"))

    def forward(self, streams, cos_sin, masks):
        block = self.block
        x, residual, weights = self.attn_connection.read(streams)
        output = block.attn(block.attn_gated(block.attn_norm(x)), cos_sin, masks)
        streams = self.attn_connection.write(residual, output, weights)
        x, residual, weights = self.mlp_connection.read(streams)
        return self.mlp_connection.write(residual, block.ffn(block.mlp_gated(block.mlp_norm(x))), weights)


class ManifoldHyperConnectionLM(MoELM):
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


block = ManifoldHyperConnectedBlock
model = ManifoldHyperConnectionLM


def init(lm):
    for wrapped in lm.blocks:
        wrapped.attn_connection.reset_parameters()
        wrapped.mlp_connection.reset_parameters()


def check(config):
    for name in ("streams", "sinkhorn_iterations"):
        value = config.option(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("gate_init", "init_std"):
        value = config.option(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
