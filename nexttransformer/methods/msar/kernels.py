"""First-order CUDA adapters for the auditable MSAR reference operators."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from ._temporal_kernels import (
    v3_temporal_conv_k4812_fwd_kernel,
    v3_temporal_conv_k4812_dx_kernel,
    v3_temporal_conv_wgrad_partial_kernel,
    v3_temporal_conv_wgrad_reduce_kernel,
)


class _Temporal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2):
        s, b, c = x.shape
        out = x.new_empty((s, b, 3, c))
        grid = (triton.cdiv(s, 8), b, triton.cdiv(c, 64))
        v3_temporal_conv_k4812_fwd_kernel[grid](
            x, w0, x, w1, x, w2, x, out, S=s, B=b, C=c,
            HAS_BIAS=False, BLOCK_T=8, BLOCK_C=64, num_warps=2)
        ctx.save_for_backward(x, w0, w1, w2)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, *weights = ctx.saved_tensors
        s, b, c = x.shape
        dx = torch.empty_like(x)
        strides = dict(zip(("stride_go_s", "stride_go_b", "stride_go_k", "stride_go_c"), grad.stride()))
        v3_temporal_conv_k4812_dx_kernel[(triton.cdiv(s, 8), b, triton.cdiv(c, 64))](
            grad, *weights, dx, S=s, B=b, C=c, **strides,
            BLOCK_T=8, BLOCK_C=64, num_warps=2)
        tiles = triton.cdiv(s, 1024)
        gradients = []
        for branch, weight in enumerate(weights):
            k = weight.shape[-1]
            partial = torch.empty((tiles*b, c, k), device=x.device, dtype=torch.float32)
            dw = torch.empty_like(weight)
            v3_temporal_conv_wgrad_partial_kernel[(triton.cdiv(c, 32), tiles, b)](
                x, grad, partial, dw, S=s, B=b, C=c, **strides, BRANCH=branch,
                K=k, HAS_BIAS=False, BLOCK_T=1024, BLOCK_C=32, num_warps=8)
            v3_temporal_conv_wgrad_reduce_kernel[(triton.cdiv(c, 32),)](
                partial, dw, dw, dw, C=c, K=k, NUM_PARTIALS=tiles*b,
                HAS_BIAS=False, BLOCK_C=32, num_warps=1)
            gradients.append(dw)
        return dx, *gradients


def temporal(raw, weights):
    # The existing xHC training kernels use SBC; master parameters remain FP32.
    x = raw.transpose(0, 1).contiguous()
    branches = _Temporal.apply(x, *(w.to(raw.dtype) for w in weights))
    return torch.cat((x.unsqueeze(2), branches), 2).transpose(0, 1).contiguous()


@triton.jit
def _values_fwd(Values, W, Out, M: tl.constexpr, D: tl.constexpr,
                C: tl.constexpr, Q: tl.constexpr, BLOCK: tl.constexpr):
    # Adapt xHC's pointer-tuple read to BTNC and the reference BF16 boundaries.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.program_id(1)
    row = i // D
    acc = tl.full((BLOCK,), 0, tl.float32)
    for c in tl.static_range(C):
        v = tl.load(Values[c] + i, i < M*D, other=0).to(tl.float32)
        w = tl.load(W + (q*C+c)*M+row, row < M, other=0).to(tl.float32)
        product = (w*v).to(tl.bfloat16).to(tl.float32)
        acc = (acc+product).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + q*M*D+i, acc, i < M*D)


@triton.jit
def _values_dx(Grad, W, Outputs, M: tl.constexpr, D: tl.constexpr,
               C: tl.constexpr, Q: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row = i // D
    for c in tl.static_range(C):
        acc = tl.full((BLOCK,), 0, tl.float32)
        for q in tl.static_range(Q):
            g = tl.load(Grad+q*M*D+i, i < M*D, other=0).to(tl.float32)
            w = tl.load(W+(q*C+c)*M+row, row < M, other=0).to(tl.float32)
            acc += (g*w).to(tl.bfloat16).to(tl.float32)
        tl.store(Outputs[c]+i, acc, i < M*D)


@triton.jit
def _values_dw(Values, Grad, Out, M: tl.constexpr, D: tl.constexpr,
               C: tl.constexpr, Q: tl.constexpr, BD: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0)*ROWS + tl.arange(0, ROWS)
    q = tl.program_id(1)
    d = tl.arange(0, BD)
    mask = (row[:, None] < M) & (d[None, :] < D)
    g = tl.load(Grad+(q*M+row[:, None])*D+d[None, :], mask, other=0).to(tl.float32)
    for c in tl.static_range(C):
        v = tl.load(Values[c]+row[:, None]*D+d[None, :], mask, other=0).to(tl.float32)
        product = (g*v).to(tl.bfloat16).to(tl.float32)
        tl.store(Out+(q*C+c)*M+row, tl.sum(product, 1), row < M)


class _CompletedValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, *sources):
        weights = weights.contiguous()
        q, c = weights.shape[:2]
        d = sources[0].shape[-1]
        m = sources[0].numel() // d
        out = sources[0].new_empty((q, *sources[0].shape))
        _values_fwd[(triton.cdiv(m*d, 1024), q)](
            sources, weights, out, m, d, c, q, 1024, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(weights, *sources)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        weights, *sources = ctx.saved_tensors
        sources = tuple(sources)
        grad = grad.contiguous()
        q, c = weights.shape[:2]
        d = sources[0].shape[-1]
        m = sources[0].numel() // d
        dw = torch.empty_like(weights)
        dx = tuple(torch.empty_like(source) for source in sources)
        _values_dx[(triton.cdiv(m*d, 1024),)](
            grad, weights, dx, m, d, c, q, 1024, num_warps=4, enable_fp_fusion=False)
        _values_dw[(triton.cdiv(m, 4), q)](
            sources, grad, dw, m, d, c, q, triton.next_power_of_2(d), 4,
            num_warps=4, enable_fp_fusion=False)
        return dw, *dx


def completed_values(weights, sources):
    return _CompletedValues.apply(weights, *sources)
