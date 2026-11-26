"""Restore Inkling's zero relative bias outside the trained distance bank.

This evaluation adapter leaves the published plugin, learned parameters and
2,048-token training path intact. It implements the original Inkling definition:
the relative bias is zero at distances >= extent; content attention stays causal.
"""
import torch
from torch.nn import functional as F

from nexttransformer.methods.inkling_rel import RelativeSelfAttention
from nexttransformer.methods.inkling_rel.reference import relative_attention_reference
from nexttransformer.models.backbone import norm


class FullContextRelativeAttention(RelativeSelfAttention):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if x.size(1) <= self.rel_extent:
            return super().forward(x, ve, cos_sin, window_size, kv_cache)
        if kv_cache is not None:
            raise ValueError("REL full-context evaluation requires full sequences")
        if self.training or torch.is_grad_enabled():
            raise ValueError("The full-context adapter is evaluation-only")
        b, t, _ = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.view_as(v)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        r = self.c_r(x).view(b, t, self.n_head, self.d_rel)
        # Padding is an inference tensor, never a learned parameter or checkpoint edit.
        table = F.pad(self.rel_logits_proj.proj.to(r.dtype), (0, t - self.rel_extent))
        if q.is_cuda:
            from nexttransformer.methods.inkling_rel.kernel import relative_attention
            y = relative_attention(q, k, v, r, table)
        else:
            y = relative_attention_reference(q, k, v, r, table)
        return self.c_proj(y.contiguous().view(b, t, self.n_embd))


def enable_full_context(model):
    """Apply to a freshly loaded evaluation model, preserving its state dict exactly."""
    if model.training:
        raise ValueError("Call model.eval() before enabling full-context evaluation")
    modules = [m for m in model.modules() if isinstance(m, RelativeSelfAttention)]
    if not modules:
        raise ValueError("Expected an Inkling REL checkpoint")
    for module in modules:
        module.__class__ = FullContextRelativeAttention
    return model
