"""SimpleHC: Hyper-Connections with identity transitions and shared gate inputs (Zhang et al., arXiv 2611.05984).

The residual state holds `streams` copies of width C; as for the built-in `hc` and `mhc`, the embedding is replicated
into every stream and the final streams are summed. Each sublayer j reads u_j = X_j^T h_pre_j and writes
X_{j+1} = X_j + h_post_j o_j^T: no residual mixing (Sec. 3.1), with mHC's full-state gates h_pre = sigmoid and
h_post = 2 sigmoid of alpha * (W^T RMSNorm(vec X)) + b.

Shared memory access (Sec. 3.2): each run of `group` consecutive sublayers (K) generates its layer-specific gates from
the group-entry state X_L. Inside a group, earlier outputs reach later inputs through the scalar corrections of Eq. 4,
d_{j,k} = h_pre_j . h_post_k, so no intermediate nC-wide state is formed, and the group writes back once (Eq. 5).
This is the paper's algebra in PyTorch; its fused DeepGEMM/Triton kernels are not reproduced. When the sublayer count
is not a multiple of K, the last group is shorter.

Initialization follows the paper (Appendix A.1, the mHC scheme of the xHC repository): projections N(0, init_std),
gate scales alpha_init, zero biases, so every stream starts with read gate 0.5 and write gate 1.
"""
import torch
from torch import nn

from nexttransformer.models.backbone import norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import coefficient_precision


def group_sizes(sublayers, group):
    """Consecutive groups of `group` sublayers; the last one holds the remainder."""
    whole, rest = divmod(sublayers, group)
    return (group,) * whole + ((rest,) if rest else ())


class GroupGates(nn.Module):
    """Read and write gates of `size` consecutive sublayers from one group-entry state (Eqs. 3 and 6)."""

    def __init__(self, width, streams, size, init_std, alpha_init):
        super().__init__()
        self.streams, self.size, self.init_std, self.alpha_init = streams, size, init_std, alpha_init
        self.projection = nn.Parameter(torch.empty(streams * width, 2 * size * streams))
        self.alpha = nn.Parameter(torch.empty(2, size, 1))
        self.bias = nn.Parameter(torch.empty(2, size, streams))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.projection, std=self.init_std)
        self.alpha.fill_(self.alpha_init)
        self.bias.zero_()

    def forward(self, streams):
        """(h_pre, h_post), each [B, T, size, n] in coefficient precision."""
        b, t, n, d = streams.shape
        flat = norm(streams.reshape(b, t, n * d))
        logits = coefficient_precision(flat @ self.projection.to(flat.dtype)).unflatten(-1, (2, self.size, n))
        logits = self.alpha * logits + self.bias
        return torch.sigmoid(logits[..., 0, :, :]), 2 * torch.sigmoid(logits[..., 1, :, :])


class SimpleHC(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        streams, group = config.option("streams"), config.option("group")
        self.residual_maps = nn.ModuleList(
            GroupGates(config.width, streams, size, config.option("init_std"), config.option("alpha_init"))
            for size in group_sizes(2 * config.n_layer, group))

    def _sublayer(self, h, idx, k):
        return self._attention_branch(h, idx, k // 2) if k % 2 == 0 else self.transformer.h[k // 2].mlp(norm(h))

    def _ordinary_hidden(self, idx):
        c = self.config
        x = self._embed_reference(idx)
        streams = x[:, :, None].expand(-1, -1, c.option("streams"), -1)
        k = 0
        for gates in self.residual_maps:
            pre, post = gates(streams)
            base = torch.einsum("btkn,btnd->btkd", pre.to(streams.dtype), streams)
            outputs = []
            for a in range(gates.size):
                h = base[:, :, a]
                if a:
                    corrections = (pre[:, :, a, None] * post[:, :, :a]).sum(-1).to(h.dtype)
                    for b, out in enumerate(outputs):
                        h = h + corrections[..., b, None] * out
                outputs.append(self._sublayer(h, idx, k))
                self._record("residual_map", sublayer=k, streams=c.option("streams"), tokens=idx.numel())
                k += 1
            streams = streams + torch.einsum("btkn,btkd->btnd", post.to(streams.dtype), torch.stack(outputs, 2))
        return streams.sum(2), None     # no backout, as the residual family


model = SimpleHC
# The residual family's trunk does not use the shell's residual and x0 scalars or the backout.
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


def check(config):
    for name in ("streams", "group"):
        value = config.option(name)
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("init_std", "alpha_init"):
        value = config.option(name)
        if not (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
            raise ValueError(f"{name} must be positive")
