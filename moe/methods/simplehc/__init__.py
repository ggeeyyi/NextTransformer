"""SimpleHC on the MoE track (Zhang et al., arXiv 2611.05984).

Hyper-Connections with identity transitions and shared gate inputs. The residual state holds `streams`
copies of width d; the embedding is replicated into every stream and the streams are summed before the final
norm and head. Sublayer j reads u_j = X_j^T h_pre_j and writes X_{j+1} = X_j + h_post_j o_j^T, with no
residual mixing (Sec. 3.1) and mHC's full-state gates h_pre = sigmoid, h_post = 2 sigmoid of
alpha (W^T RMSNorm(vec X)) + b.

Shared memory access (Sec. 3.2) runs over a group of consecutive sublayers: one projection of the
group-entry state X_L produces every gate in the group, earlier outputs reach later inputs through the
scalar corrections of Eq. 4, d_{j,k} = h_pre_j . h_post_k, and the group writes back once (Eq. 5), so no
intermediate n*d-wide state is formed. Here the group is one block, the attention sublayer and the MoE FFN
sublayer: moe/train.py shards one FSDP unit per block, and a group that spanned two units would have to
carry its partial state across a resharding boundary. Smaller groups refresh the gates more often, so this
is the more expressive end of the paper's trade-off; what it gives up is the fused kernels' amortization,
which this PyTorch transcription does not reproduce in any case.

Initialization follows the paper (Appendix A.1): projections N(0, init_std), gate scales alpha_init, zero
biases, so every stream starts with read gate 0.5 and write gate 1. One deviation from a literal
transcription: the gate scales and biases are 1-D parameters so that moe/optim.py routes them to Adam. MuonH
would otherwise apply Newton-Schulz and its hyperball renormalization to tables of coefficients, and would
freeze the zero-initialized biases permanently, since a matrix whose Frobenius norm starts at zero keeps it.
"""
import torch
from torch import nn
import torch.nn.functional as F

from moe.model import Masks, MoELM, segments

SUBLAYERS = 2       # the group: one block's attention sublayer and its MoE FFN sublayer


def coefficient_precision(x):
    """Residual coefficients in FP32 for BF16/FP32 activations; FP64 inputs stay FP64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


class GroupGates(nn.Module):
    """Read and write gates of the group's sublayers from the group-entry state (Eqs. 3 and 6)."""

    def __init__(self, width, streams, size, init_std, alpha_init):
        super().__init__()
        self.streams, self.size, self.init_std, self.alpha_init = streams, size, init_std, alpha_init
        self.projection = nn.Parameter(torch.empty(streams * width, 2 * size * streams))
        self.alpha = nn.Parameter(torch.empty(2 * size))
        self.bias = nn.Parameter(torch.empty(2 * size * streams))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.projection, std=self.init_std)
        self.alpha.fill_(self.alpha_init)
        self.bias.zero_()

    def forward(self, streams):
        """(h_pre, h_post), each [B, T, size, n] in coefficient precision."""
        b, t, n, d = streams.shape
        flat = rms(streams.reshape(b, t, n * d))
        logits = coefficient_precision(flat @ self.projection.to(flat.dtype)).unflatten(-1, (2, self.size, n))
        logits = self.alpha.view(2, self.size, 1) * logits + self.bias.view(2, self.size, n)
        return torch.sigmoid(logits[..., 0, :, :]), 2 * torch.sigmoid(logits[..., 1, :, :])


class SimpleHCBlock(nn.Module):
    """One block as one SimpleHC group: gates once, two sublayers, one write back."""

    def __init__(self, block, config, layer):
        super().__init__()
        self.block = block
        self.gates = GroupGates(config.width, config.option("streams"), SUBLAYERS,
                                config.option("init_std"), config.option("alpha_init"))

    def forward(self, streams, cos_sin, masks):
        block = self.block
        pre, post = self.gates(streams)
        base = torch.einsum("btkn,btnd->btkd", pre.to(streams.dtype), streams)
        attention = block.attn(block.attn_gated(block.attn_norm(base[:, :, 0])), cos_sin, masks)
        # Eq. 4: the FFN reads the attention output through a scalar, not through the n*d-wide state.
        correction = (pre[:, :, 1] * post[:, :, 0]).sum(-1).to(attention.dtype)
        mixed = base[:, :, 1] + correction[..., None] * attention
        outputs = torch.stack((attention, block.ffn(block.mlp_gated(block.mlp_norm(mixed)))), 2)
        return streams + torch.einsum("btkn,btkd->btnd", post.to(streams.dtype), outputs)


class SimpleHCLM(MoELM):
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


block = SimpleHCBlock
model = SimpleHCLM


def init(lm):
    for wrapped in lm.blocks:
        wrapped.gates.reset_parameters()


def check(config):
    streams = config.option("streams")
    if not isinstance(streams, int) or isinstance(streams, bool) or streams < 1:
        raise ValueError("streams must be a positive integer")
    for name in ("init_std", "alpha_init"):
        value = config.option(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
