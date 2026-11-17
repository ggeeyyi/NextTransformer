"""Training temporal kernels extracted unchanged from xHC fused_xhc_v3.py.

Only K4/8/12 forward, input gradient and split weight gradient are included.
The source describes these operators as adapted from Sunan's implementation.
Layout and dtype adapters are in kernels.py. The source license follows.
"""

# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Source SHA256: dcb2f90d9e90b64e760659206113098be8b81dcf4fe020f7f4fd2269b9bbf79d
import triton
import triton.language as tl

@triton.jit
def v3_temporal_conv_k4812_fwd_kernel(
    x_ptr,      # [S, B, C]
    w0_ptr,     # [C, 1, 4]
    b0_ptr,     # [C] or dummy
    w1_ptr,     # [C, 1, 8]
    b1_ptr,
    w2_ptr,     # [C, 1, 12]
    b2_ptr,
    out_ptr,    # [S, B, 3, C]
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = offs_t < S
    c_mask = offs_c < C

    acc0 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for r in tl.static_range(4):
        t_in = offs_t + r - 3
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w0_ptr + offs_c * 4 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc0 += x * w[None, :]

    for r in tl.static_range(8):
        t_in = offs_t + r - 7
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w1_ptr + offs_c * 8 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc1 += x * w[None, :]

    for r in tl.static_range(12):
        t_in = offs_t + r - 11
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w2_ptr + offs_c * 12 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc2 += x * w[None, :]

    if HAS_BIAS:
        acc0 += tl.load(b0_ptr + offs_c, mask=c_mask, other=0.0)[None, :]
        acc1 += tl.load(b1_ptr + offs_c, mask=c_mask, other=0.0)[None, :]
        acc2 += tl.load(b2_ptr + offs_c, mask=c_mask, other=0.0)[None, :]

    out_base = ((offs_t[:, None] * B + pid_b) * 3) * C + offs_c[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_base + 0 * C, acc0.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + 1 * C, acc1.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + 2 * C, acc2.to(tl.bfloat16), mask=mask)


@triton.jit
def v3_temporal_conv_k4812_dx_kernel(
    go_ptr,     # [S, B, 3, C]
    w0_ptr,     # [C, 1, 4]
    w1_ptr,     # [C, 1, 8]
    w2_ptr,     # [C, 1, 12]
    gx_ptr,     # [S, B, C]
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = offs_t < S
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for r in tl.static_range(4):
        t_out = offs_t - r + 3
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 0 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w0_ptr + offs_c * 4 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    for r in tl.static_range(8):
        t_out = offs_t - r + 7
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 1 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w1_ptr + offs_c * 8 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    for r in tl.static_range(12):
        t_out = offs_t - r + 11
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 2 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w2_ptr + offs_c * 12 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    mask = t_mask[:, None] & c_mask[None, :]
    tl.store(gx_ptr + (offs_t[:, None] * B + pid_b) * C + offs_c[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def v3_temporal_conv_wgrad_partial_kernel(
    x_ptr,      # [S, B, C]
    go_ptr,     # [S, B, 3, C]
    partial_w_ptr,  # [NUM_TILES * B, C, K] fp32
    partial_b_ptr,  # [NUM_TILES * B, C] fp32 or dummy
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    BRANCH: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_mask = offs_c < C
    t_mask = offs_t < S

    bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    g = tl.load(
        go_ptr
        + offs_t[:, None] * stride_go_s
        + pid_b * stride_go_b
        + BRANCH * stride_go_k
        + offs_c[None, :] * stride_go_c,
        mask=t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    bias_acc += tl.sum(g, axis=0)

    t_in0 = offs_t - (K - 1)
    x0 = tl.load(
        x_ptr + (t_in0[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in0[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc0 += tl.sum(x0 * g, axis=0)

    t_in1 = offs_t + 1 - (K - 1)
    x1 = tl.load(
        x_ptr + (t_in1[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in1[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc1 += tl.sum(x1 * g, axis=0)

    t_in2 = offs_t + 2 - (K - 1)
    x2 = tl.load(
        x_ptr + (t_in2[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in2[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc2 += tl.sum(x2 * g, axis=0)

    t_in3 = offs_t + 3 - (K - 1)
    x3 = tl.load(
        x_ptr + (t_in3[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in3[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc3 += tl.sum(x3 * g, axis=0)

    if K >= 8:
        t_in4 = offs_t + 4 - (K - 1)
        x4 = tl.load(
            x_ptr + (t_in4[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in4[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc4 += tl.sum(x4 * g, axis=0)

        t_in5 = offs_t + 5 - (K - 1)
        x5 = tl.load(
            x_ptr + (t_in5[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in5[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc5 += tl.sum(x5 * g, axis=0)

        t_in6 = offs_t + 6 - (K - 1)
        x6 = tl.load(
            x_ptr + (t_in6[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in6[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc6 += tl.sum(x6 * g, axis=0)

        t_in7 = offs_t + 7 - (K - 1)
        x7 = tl.load(
            x_ptr + (t_in7[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in7[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc7 += tl.sum(x7 * g, axis=0)

    if K >= 12:
        t_in8 = offs_t + 8 - (K - 1)
        x8 = tl.load(
            x_ptr + (t_in8[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in8[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc8 += tl.sum(x8 * g, axis=0)

        t_in9 = offs_t + 9 - (K - 1)
        x9 = tl.load(
            x_ptr + (t_in9[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in9[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc9 += tl.sum(x9 * g, axis=0)

        t_in10 = offs_t + 10 - (K - 1)
        x10 = tl.load(
            x_ptr + (t_in10[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in10[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc10 += tl.sum(x10 * g, axis=0)

        t_in11 = offs_t + 11 - (K - 1)
        x11 = tl.load(
            x_ptr + (t_in11[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in11[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc11 += tl.sum(x11 * g, axis=0)

    partial_idx = pid_t * B + pid_b
    base = (partial_idx * C + offs_c) * K
    tl.store(partial_w_ptr + base + 0, acc0, mask=c_mask)
    tl.store(partial_w_ptr + base + 1, acc1, mask=c_mask)
    tl.store(partial_w_ptr + base + 2, acc2, mask=c_mask)
    tl.store(partial_w_ptr + base + 3, acc3, mask=c_mask)
    if K >= 8:
        tl.store(partial_w_ptr + base + 4, acc4, mask=c_mask)
        tl.store(partial_w_ptr + base + 5, acc5, mask=c_mask)
        tl.store(partial_w_ptr + base + 6, acc6, mask=c_mask)
        tl.store(partial_w_ptr + base + 7, acc7, mask=c_mask)
    if K >= 12:
        tl.store(partial_w_ptr + base + 8, acc8, mask=c_mask)
        tl.store(partial_w_ptr + base + 9, acc9, mask=c_mask)
        tl.store(partial_w_ptr + base + 10, acc10, mask=c_mask)
        tl.store(partial_w_ptr + base + 11, acc11, mask=c_mask)
    if HAS_BIAS:
        tl.store(partial_b_ptr + partial_idx * C + offs_c, bias_acc, mask=c_mask)


@triton.jit
def v3_temporal_conv_wgrad_reduce_kernel(
    partial_w_ptr,  # [NUM_PARTIALS, C, K] fp32
    partial_b_ptr,  # [NUM_PARTIALS, C] fp32 or dummy
    gw_ptr,         # [C, 1, K]
    gb_ptr,         # [C] or dummy
    C: tl.constexpr,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    bacc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for p in tl.static_range(NUM_PARTIALS):
        base = (p * C + offs_c) * K
        acc0 += tl.load(partial_w_ptr + base + 0, mask=c_mask, other=0.0)
        acc1 += tl.load(partial_w_ptr + base + 1, mask=c_mask, other=0.0)
        acc2 += tl.load(partial_w_ptr + base + 2, mask=c_mask, other=0.0)
        acc3 += tl.load(partial_w_ptr + base + 3, mask=c_mask, other=0.0)
        if K >= 8:
            acc4 += tl.load(partial_w_ptr + base + 4, mask=c_mask, other=0.0)
            acc5 += tl.load(partial_w_ptr + base + 5, mask=c_mask, other=0.0)
            acc6 += tl.load(partial_w_ptr + base + 6, mask=c_mask, other=0.0)
            acc7 += tl.load(partial_w_ptr + base + 7, mask=c_mask, other=0.0)
        if K >= 12:
            acc8 += tl.load(partial_w_ptr + base + 8, mask=c_mask, other=0.0)
            acc9 += tl.load(partial_w_ptr + base + 9, mask=c_mask, other=0.0)
            acc10 += tl.load(partial_w_ptr + base + 10, mask=c_mask, other=0.0)
            acc11 += tl.load(partial_w_ptr + base + 11, mask=c_mask, other=0.0)
        if HAS_BIAS:
            bacc += tl.load(partial_b_ptr + p * C + offs_c, mask=c_mask, other=0.0)

    tl.store(gw_ptr + offs_c * K + 0, acc0.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 1, acc1.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 2, acc2.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 3, acc3.to(tl.bfloat16), mask=c_mask)
    if K >= 8:
        tl.store(gw_ptr + offs_c * K + 4, acc4.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 5, acc5.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 6, acc6.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 7, acc7.to(tl.bfloat16), mask=c_mask)
    if K >= 12:
        tl.store(gw_ptr + offs_c * K + 8, acc8.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 9, acc9.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 10, acc10.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 11, acc11.to(tl.bfloat16), mask=c_mask)
    if HAS_BIAS:
        tl.store(gb_ptr + offs_c, bacc.to(tl.bfloat16), mask=c_mask)
