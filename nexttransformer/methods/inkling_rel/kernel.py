"""Tiled causal REL attention with gradients for Q, K, V and both REL factors.

A BF16 distance-logit GEMM is followed by online-softmax attention. No NxN
attention probabilities are saved. Backward recomputes distance logits, writes
one gradient per query/distance (no atomics), then differentiates the two REL
factors with GEMMs. Q and KV gradients use separate ownership tiles.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, Bias, O, Lse, T: tl.constexpr, H: tl.constexpr,
             D: tl.constexpr, E: tl.constexpr, SCALE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q_ptr = (batch * T + rows[:, None]) * H * D + head * D + dims[None, :]
    q = tl.load(Q + q_ptr, rows[:, None] < T, 0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    output = tl.zeros((BM, D), tl.float32)
    for start in range(0, tl.minimum((block + 1) * BM, T), BN):
        cols = start + tl.arange(0, BN)
        kv_ptr = (batch * T + cols[:, None]) * H * D + head * D + dims[None, :]
        k = tl.load(K + kv_ptr, cols[:, None] < T, 0)
        v = tl.load(V + kv_ptr, cols[:, None] < T, 0)
        distance = rows[:, None] - cols[None, :]
        valid = (rows[:, None] < T) & (cols[None, :] < T) & (distance >= 0)
        bias_ptr = ((batch * T + rows[:, None]) * H + head) * E + distance
        bias = tl.load(Bias + bias_ptr, valid, 0).to(tl.float32)
        scores = tl.dot(q, tl.trans(k), input_precision='ieee') * SCALE + bias
        scores = tl.where(valid, scores, -float('inf')) * 1.4426950408889634
        updated = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - updated)
        probability = tl.exp2(scores - updated[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        output = output * alpha[:, None] + tl.dot(probability.to(v.dtype), v, input_precision='ieee')
        maximum = updated
    output = output / denominator[:, None]
    tl.store(O + q_ptr, output.to(O.dtype.element_ty), rows[:, None] < T)
    tl.store(Lse + bh * T + rows, maximum + tl.log2(denominator), rows < T)


@triton.jit
def _delta(O, DO, Delta, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
           BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    dims = tl.arange(0, D)
    offsets = ((bh // H * T + rows[:, None]) * H + bh % H) * D + dims[None, :]
    output = tl.load(O + offsets, rows[:, None] < T, 0).to(tl.float32)
    grad = tl.load(DO + offsets, rows[:, None] < T, 0).to(tl.float32)
    tl.store(Delta + bh * T + rows, tl.sum(output * grad, 1), rows < T)


@triton.jit
def _backward_q(Q, K, V, Bias, DO, Lse, Delta, DQ, DBias,
                T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    offsets = ((batch * T + rows[:, None]) * H + head) * D + dims[None, :]
    query = tl.load(Q + offsets, rows[:, None] < T, 0)
    dout = tl.load(DO + offsets, rows[:, None] < T, 0)
    lse = tl.load(Lse + bh*T + rows, rows < T, 0)
    delta = tl.load(Delta + bh*T + rows, rows < T, 0)
    dq = tl.zeros((BM, D), tl.float32)
    for start in range(0, tl.minimum((block + 1) * BM, T), BN):
        cols = start + tl.arange(0, BN)
        kv_offsets = ((batch*T + cols[:, None])*H + head)*D + dims[None, :]
        key = tl.load(K + kv_offsets, cols[:, None] < T, 0)
        value = tl.load(V + kv_offsets, cols[:, None] < T, 0)
        distance = rows[:, None] - cols[None, :]
        valid = (rows[:, None] < T) & (cols[None, :] < T) & (distance >= 0)
        bias_offsets = ((batch*T + rows[:, None])*H + head)*E + distance
        bias = tl.load(Bias + bias_offsets, valid, 0).to(tl.float32)
        scores = tl.dot(query, tl.trans(key), input_precision='ieee') * SCALE + bias
        probability = tl.exp2(scores * 1.4426950408889634 - lse[:, None])
        probability = tl.where(valid, probability, 0.)
        dp = tl.dot(dout, tl.trans(value), input_precision='ieee')
        ds = tl.where(rows[:, None] > 0, probability * (dp - delta[:, None]), 0.)
        tl.store(DBias + bias_offsets, ds.to(DBias.dtype.element_ty), valid)
        dq += tl.dot(ds.to(key.dtype), key, input_precision='ieee') * SCALE
    tl.store(DQ + offsets, dq.to(DQ.dtype.element_ty), rows[:, None] < T)


@triton.jit
def _backward_kv(Q, K, V, Bias, DO, Lse, Delta, DK, DV,
                 T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                 SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    cols = block * BN + tl.arange(0, BN)
    dims = tl.arange(0, D)
    kv_offsets = ((batch*T + cols[:, None])*H + head)*D + dims[None, :]
    key = tl.load(K + kv_offsets, cols[:, None] < T, 0)
    value = tl.load(V + kv_offsets, cols[:, None] < T, 0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    for start in range(block * BN // BM * BM, T, BM):
        rows = start + tl.arange(0, BM)
        offsets = ((batch*T + rows[:, None])*H + head)*D + dims[None, :]
        query = tl.load(Q + offsets, rows[:, None] < T, 0)
        dout = tl.load(DO + offsets, rows[:, None] < T, 0)
        lse = tl.load(Lse + bh*T + rows, rows < T, 0)
        delta = tl.load(Delta + bh*T + rows, rows < T, 0)
        distance = rows[:, None] - cols[None, :]
        valid = (rows[:, None] < T) & (cols[None, :] < T) & (distance >= 0)
        bias_offsets = ((batch*T + rows[:, None])*H + head)*E + distance
        bias = tl.load(Bias + bias_offsets, valid, 0).to(tl.float32)
        scores = tl.dot(query, tl.trans(key), input_precision='ieee') * SCALE + bias
        probability = tl.exp2(scores * 1.4426950408889634 - lse[:, None])
        probability = tl.where(valid, probability, 0.)
        dp = tl.dot(dout, tl.trans(value), input_precision='ieee')
        ds = tl.where(rows[:, None] > 0, probability * (dp - delta[:, None]), 0.)
        dv += tl.dot(tl.trans(probability.to(dout.dtype)), dout, input_precision='ieee')
        dk += tl.dot(tl.trans(ds.to(query.dtype)), query, input_precision='ieee') * SCALE
    tl.store(DK + kv_offsets, dk.to(DK.dtype.element_ty), cols[:, None] < T)
    tl.store(DV + kv_offsets, dv.to(DV.dtype.element_ty), cols[:, None] < T)


class _RelativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, r, table):
        b, t, h, d = q.shape
        logits = r @ table
        output = torch.empty_like(q)
        lse = torch.empty((b, h, t), device=q.device, dtype=torch.float32)
        _forward[(triton.cdiv(t, 32), b*h)](q, k, v, logits, output, lse,
            t, h, d, table.shape[1], d**-0.5, 32, 64, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, r, table, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad):
        q, k, v, r, table, output, lse = ctx.saved_tensors
        grad = grad.contiguous()
        b, t, h, d = q.shape
        logits = r @ table
        dlogits = torch.zeros_like(logits)
        delta = torch.empty_like(lse)
        dq, dk, dv = (torch.empty_like(a) for a in (q, k, v))
        _delta[(triton.cdiv(t, 32), b*h)](output, grad, delta, t, h, d, 32, num_warps=4)
        args = (q, k, v, logits, grad, lse, delta)
        shape = (t, h, d, table.shape[1], d**-0.5, 32, 64)
        _backward_q[(triton.cdiv(t, 32), b*h)](*args, dq, dlogits, *shape, num_warps=4, num_stages=2)
        _backward_kv[(triton.cdiv(t, 64), b*h)](*args, dk, dv, *shape, num_warps=4, num_stages=2)
        dr = dlogits @ table.T
        dtable = r.reshape(-1, r.shape[-1]).T @ dlogits.reshape(-1, table.shape[1])
        return dq, dk, dv, dr, dtable


def relative_attention(q, k, v, r, table):
    """Full-sequence MHA; causal REL covers distances 0..T-1, including self."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("REL kernel requires equal BTHD query/key/value shapes")
    if r.shape[:3] != q.shape[:3] or table.ndim != 2 or r.shape[-1] != table.shape[0]:
        raise ValueError("relative factor shapes do not match attention")
    if table.shape[1] < q.shape[1]:
        raise ValueError("REL table must cover the complete context")
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        raise ValueError("unsupported head dimension")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("REL kernel supports BF16, FP16 and FP32")
    if any(a.device != q.device or a.dtype != q.dtype for a in (k, v, r, table)):
        raise ValueError("REL kernel inputs must have one device and dtype")
    return _RelativeAttention.apply(*(a.contiguous() for a in (q, k, v, r, table)))
