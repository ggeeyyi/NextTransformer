"""Differentiable causal-edge reference for REL, including every trained input.

Edges are packed in row order, so both attention matrix products compute only
causal pairs. This slow, memory-heavy reference is used for small CPU checks and
shape-only cost auditing. GPU execution uses the tiled implementation.
"""
import torch


def relative_attention_reference(q, k, v, r, table):
    """BT(H)D tensors; per-layer table [d_rel, extent], shared across heads."""
    b, t, h, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("REL currently requires full-sequence MHA with equal Q/K/V shapes")
    if t > table.shape[1]:
        raise ValueError("relative table must cover the full sequence")
    logits = r @ table
    rows, cols = torch.tril_indices(t, t, device=q.device)
    query_index = rows[None, :, None, None].expand(b, -1, h, d)
    key_index = cols[None, :, None, None].expand(b, -1, h, d)
    scores = (q.gather(1, query_index) * k.gather(1, key_index)).sum(-1) * d**-0.5
    heads = torch.arange(h, device=q.device)
    bias_index = ((rows[:, None] * h + heads[None, :]) * table.shape[1]
                  + (rows-cols)[:, None])
    bias = logits.reshape(b, -1).gather(1, bias_index.reshape(1, -1).expand(b, -1))
    scores = scores + bias.reshape(b, -1, h)
    # A row's causal softmax is independent; padding exists only in this
    # reference's softmax storage, never in the pairwise matrix products.
    dense = torch.full((b, h, t, t), -torch.inf, dtype=scores.dtype, device=scores.device)
    dense[:, :, rows, cols] = scores.transpose(1, 2)
    probs = dense.softmax(-1)[:, :, rows, cols].transpose(1, 2)
    products = probs.unsqueeze(-1) * v.gather(1, key_index)
    output = torch.zeros_like(q)
    indices = rows[None, :, None, None].expand_as(products)
    return output.scatter_add(1, indices, products)


def dense_oracle(q, k, v, r, table):
    """Independent dense oracle for validation; uses FP64/FP32 as supplied."""
    b, t, h, d = q.shape
    query, key, value = (a.transpose(1, 2) for a in (q, k, v))
    rel_logits = (r @ table).transpose(1, 2)
    distance = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
    bias = rel_logits.gather(-1, distance.clamp_min(0)[None, None].expand(b, h, t, t))
    scores = query @ key.transpose(-1, -2) * d**-0.5 + bias
    scores = scores.masked_fill(distance[None, None] < 0, -torch.inf)
    return (scores.softmax(-1) @ value).transpose(1, 2)
