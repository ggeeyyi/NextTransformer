"""Readable recurrent mathematics plus the registered chunked training form.

Equations are implemented independently; source/implementation comparisons are
listed in SOURCES.md. States are local to a packed row and never cross rows.
Inputs use [batch, time, head, feature]; FP32 accumulation for BF16/FP16 only.
The `fla_chunk64` backend trains DeltaNet, KDA, RetNet and GDN with FLA Triton kernels on CUDA
(GDN through the KDA kernels with its head gate broadcast over key features).
The ELU methods use the same IEEE FP32 chunk implementation on CPU and CUDA,
including the shape-only cost audit. `chunk_reference` is an independent,
unbatched form of the registered chunk algorithms.
"""
from contextlib import contextmanager
import functools
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from nexttransformer.models.backbone import Linear, has_ve, norm
from nexttransformer.models.config import CHUNK_BACKENDS


def accumulation_dtype(x):
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


# A06'' elu_qknorm: per-token q/k RMS of each head before phi, fixed at the
# scale the stable A06' pilot reached (EXPERIMENTS record 021). Not learnable.
ELU_QK_TEMPERATURE = 8.

# A07 GDN: the per-head gate initialization of FLA's GatedDeltaNet layer (fla 0.5.2,
# the authors' reference): A ~ U(0, 16), dt log-uniform in [1e-3, 1e-1], floor 1e-4.
GDN_A_RANGE, GDN_DT_RANGE, GDN_DT_FLOOR = (0., 16.), (1e-3, 1e-1), 1e-4


@torch.no_grad()
def init_gdn_gate(log_decay, dt_bias):
    """log_decay = log A and dt_bias = softplus^-1(dt), drawn in FLA's order."""
    log_decay.copy_(torch.empty_like(log_decay).uniform_(*GDN_A_RANGE).log())
    low, high = (math.log(x) for x in GDN_DT_RANGE)
    dt = (torch.rand_like(dt_bias)*(high-low)+low).exp().clamp_min(GDN_DT_FLOOR)
    dt_bias.copy_(dt+torch.log(-torch.expm1(-dt)))


def elu_feature_map(x):
    """ELU(x)+1 without subtracting then adding 1 in the negative tail.

    Keep the features in accumulation precision through attention and output
    normalization. Clamp only the unused exponential branch: torch.where
    evaluates both branches, so exp(large_positive) would poison its backward.
    FP64 is retained for the independent numerical checks.
    """
    x = x.to(accumulation_dtype(x))
    return torch.where(x > 0, x+1, x.clamp_max(0).exp())


class _KDAScan(torch.autograd.Function):
    """First-order BPTT storing one state per step, without checkpointing.

    The ordinary autograd expression saves both decayed and updated states.
    This backward uses the algebraic factorization g * (S @ dpred) and needs
    only the preceding state, the updated state and the small value residual.
    It does not replay forward state updates. See the FP64 gradient oracle.
    """
    @staticmethod
    def forward(ctx, q, k, v, decay, beta, initial):
        state=initial
        states=[state]; residuals=[]; outputs=[]
        for i in range(q.shape[1]):
            state=state*decay[:,i].unsqueeze(-1)
            residual=v[:,i]-(k[:,i].unsqueeze(-1)*state).sum(-2)
            state=state+k[:,i].unsqueeze(-1)*(beta[:,i].unsqueeze(-1)*residual).unsqueeze(-2)
            outputs.append((q[:,i].unsqueeze(-1)*state).sum(-2))
            residuals.append(residual);states.append(state)
        ctx.save_for_backward(q,k,v,decay,beta,torch.stack(states,1),torch.stack(residuals,1))
        return torch.stack(outputs,1),state

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dlast):
        q,k,v,g,beta,states,residuals=ctx.saved_tensors
        carry=torch.zeros_like(states[:,0]) if dlast is None else dlast
        dq=[];dk=[];dv=[];dg=[];db=[]
        for i in range(q.shape[1]-1,-1,-1):
            qi,ki,gi,bi=q[:,i],k[:,i],g[:,i],beta[:,i]
            incoming=torch.zeros_like(v[:,i]) if dy is None else dy[:,i]
            previous,current=states[:,i],states[:,i+1]
            dq.append((current*incoming.unsqueeze(-2)).sum(-1))
            total=carry+qi.unsqueeze(-1)*incoming.unsqueeze(-2)
            du=(ki.unsqueeze(-1)*total).sum(-2)
            correction=bi.unsqueeze(-1)*residuals[:,i]
            dpred=-bi.unsqueeze(-1)*du
            dk.append((total*correction.unsqueeze(-2)).sum(-1)
                      +gi*(previous*dpred.unsqueeze(-2)).sum(-1))
            dv.append(bi.unsqueeze(-1)*du)
            db.append((du*residuals[:,i]).sum(-1))
            before_decay=total+ki.unsqueeze(-1)*dpred.unsqueeze(-2)
            dg.append((before_decay*previous).sum(-1))
            carry=before_decay*gi.unsqueeze(-1)
        pack=lambda values:torch.stack(values[::-1],1)
        return pack(dq),pack(dk),pack(dv),pack(dg),pack(db),carry


def recurrent_attention(q, k, v, *, kind, decay=None, beta=None,
                        initial_state=None, eps=1e-6):
    """Reference scan returning both outputs and final state (for oracle tests).

    q/k are already feature-mapped/normalized. Decay is multiplicative, not log.
    No detach, in-place recurrent updates or hidden checkpoint recomputation.
    KDA uses an algebraically equivalent first-order custom backward to avoid
    retaining two large state histories; other methods use ordinary autograd.
    """
    dtype = q.dtype
    q, k, v = (x.to(accumulation_dtype(x)) for x in (q, k, v))
    b, t, h, d = q.shape
    value_dim = v.shape[-1]
    if t == 0:
        raise ValueError("empty sequences have no defined output")
    if kind == "gdn":
        # GDN is the KDA recurrence with one decay per head: broadcast it over key features.
        kind, decay = "kda", decay.expand(b, t, h, d)
    if kind == "kda" and torch.is_grad_enabled() and any(
            x.requires_grad for x in (q,k,v,decay,beta)):
        initial=q.new_zeros(b,h,d,value_dim) if initial_state is None else initial_state
        output,final=_KDAScan.apply(q,k,v,decay.to(q),beta.to(q),initial)
        return output.to(dtype),final
    # A single UnbindBackward stacks the per-token gradients. Repeated
    # q[:, i] slice backward would allocate and add T full-length gradients,
    # introducing artificial O(T²) floating additions into a linear scan.
    queries,keys,values=q.unbind(1),k.unbind(1),v.unbind(1)
    decays=decay.unbind(1) if decay is not None else None
    betas=beta.unbind(1) if beta is not None else None
    state = q.new_zeros(b, h, d, value_dim)
    normalizer = q.new_zeros(b, h, d)
    if initial_state is not None:
        if kind == "elu_linear": state, normalizer = initial_state
        else: state = initial_state
    out = []
    for i in range(t):
        qi, ki, vi = queries[i], keys[i], values[i]
        if decay is not None:
            state = state * decays[i].to(q).unsqueeze(-1)
        if kind in ("deltanet", "kda"):
            prediction = (ki.unsqueeze(-1) * state).sum(-2)
            correction = (vi - prediction) * betas[i].to(q).unsqueeze(-1)
            state = state + ki.unsqueeze(-1) * correction.unsqueeze(-2)
        else:
            state = state + ki.unsqueeze(-1) * vi.unsqueeze(-2)
        value = (qi.unsqueeze(-1) * state).sum(-2)
        if kind == "elu_linear":
            normalizer = normalizer + ki
            value = value / ((qi * normalizer).sum(-1, keepdim=True) + eps)
        out.append(value)
    final = (state, normalizer) if kind == "elu_linear" else state
    return torch.stack(out, 1).to(dtype), final


def _to_chunks(x, chunk):
    """[B,T,H,...] -> [B,H,N,C,...]; zero-pad the final partial chunk (causal, harmless)."""
    pad = -x.shape[1] % chunk
    if pad: x = F.pad(x, (0, 0)*(x.dim()-2)+(0, pad))
    return x.unflatten(1, (x.shape[1]//chunk, chunk)).movedim(3, 1)


def _simple_gla_chunks(q, k, v, g, chunk):
    """Scalar-decay chunk form (RetNet; ELU+1 numerator when g is None).

    Intra-chunk causal scores with a pairwise decay mask, inter-chunk state read
    and one state update per chunk, as in FLA simple_gla. Chunks are unbound once
    so the backward has no per-chunk slice accumulation.
    """
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    scores = q @ k.transpose(-1, -2)
    if g is None:
        scores = scores.masked_fill(~causal, 0)
        carries = [None]*q.shape[2]
    else:
        b = g.cumsum(-1)
        scores = scores*(b[..., :, None]-b[..., None, :]).masked_fill(~causal, -math.inf).exp()
        q, k = q*b[..., None].exp(), k*(b[..., -1:]-b)[..., None].exp()
        carries = b[..., -1].exp().unbind(2)
    out = scores @ v
    # Rows start from a zero state, and the state after the final chunk is never
    # read: neither the first read-out nor the last update is computed.
    state, inter, last = None, [], q.shape[2]-1
    for n, (qn, kn, vn, carry) in enumerate(zip(q.unbind(2), k.unbind(2), v.unbind(2), carries)):
        inter.append(qn.new_zeros(*qn.shape[:-1], v.shape[-1]) if state is None else qn @ state)
        if n == last: break
        update = kn.transpose(-1, -2) @ vn
        if state is None: state = update
        else: state = (state if carry is None else state*carry[..., None, None])+update
    return out+torch.stack(inter, 2)


def _kda_chunks(q, k, v, g, beta, chunk):
    """Per-feature gated delta rule in chunk (WY/UT) form, as in FLA chunk_kda.

    With Gamma the within-chunk cumulative decay: pseudo-values solve
    (I + strict_tril(beta (k*Gamma)(k/Gamma)^T)) [w|u0] = beta [k*Gamma | v],
    then u = u0 - w S, o = (q*Gamma) S + tril((q*Gamma)(k/Gamma)^T) u and
    S <- Gamma_C * S + (k*Gamma_C/Gamma)^T u.
    """
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    b = g.cumsum(-2)
    grow, shrink = b.exp(), (-b).exp()
    qs, ks, kinv = q*grow, k*grow, k*shrink
    akk = ks @ kinv.transpose(-1, -2)
    aqk = (qs @ kinv.transpose(-1, -2)).masked_fill(~causal, 0)
    weight = beta[..., None]
    # unitriangular: the solver reads (and differentiates) only the strictly
    # lower part of beta*akk; the identity is implicit, so no mask is needed.
    w, u0 = torch.linalg.solve_triangular(weight*akk, torch.cat((weight*ks, weight*v), -1),
                                          upper=False, unitriangular=True).split((k.shape[-1], v.shape[-1]), -1)
    final = b[..., -1:, :]
    kw = k*(final-b).exp()
    # Zero initial state and unread final state, as in the scalar-decay form.
    state, out, last = None, [], q.shape[2]-1
    for n, (qn, an, wn, un, kn, carry) in enumerate(zip(qs.unbind(2), aqk.unbind(2), w.unbind(2), u0.unbind(2),
                                                        kw.unbind(2), final.squeeze(-2).exp().unbind(2))):
        if state is None:
            u = un
            out.append(an @ u)
        else:
            u = un-wn @ state
            out.append(qn @ state+an @ u)
        if n == last: break
        update = kn.transpose(-1, -2) @ u
        state = update if state is None else state*carry[..., None]+update
    return torch.stack(out, 2)


def _gdn_chunks(q, k, v, g, beta, chunk):
    """Scalar-gated delta rule in chunk (WY/UT) form, as in FLA chunk_gated_delta_rule.

    The KDA algebra with one log decay per head and token. With b the within-chunk
    cumulative log decay and D_ij = exp(b_i-b_j) for i >= j (zero above):
    (I + strict_tril(beta (k k^T)*D)) [w|u0] = beta [k*exp(b) | v], u = u0 - w S,
    o = (q*exp(b)) S + ((q k^T)*D) u and S <- exp(b_C) S + (k*exp(b_C-b))^T u.
    Every exponent is <= 0, so unlike `_kda_chunks` no gate strength overflows.
    """
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    weight = beta[..., None]
    if g is None:
        # DeltaNet shares the WY solve and state update, without decay operations.
        akk = k @ k.transpose(-1, -2)
        aqk = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0)
        ks, qs, kw = k, q, k
        carries = [None]*q.shape[2]
    else:
        b = g.cumsum(-1)
        pair = (b[..., :, None]-b[..., None, :]).masked_fill(~causal, -math.inf).exp()
        akk = (k @ k.transpose(-1, -2))*pair
        aqk = (q @ k.transpose(-1, -2))*pair
        grow = b.exp()[..., None]
        final = b[..., -1:]
        ks, qs, kw = k*grow, q*grow, k*(final-b).exp()[..., None]
        carries = final.squeeze(-1).exp().unbind(2)
    w, u0 = torch.linalg.solve_triangular(weight*akk, torch.cat((weight*ks, weight*v), -1),
                                          upper=False, unitriangular=True).split((k.shape[-1], v.shape[-1]), -1)
    # Zero initial state and unread final state, as in the other chunk forms.
    state, out, last = None, [], q.shape[2]-1
    for n, (qn, an, wn, un, kn, carry) in enumerate(zip(qs.unbind(2), aqk.unbind(2), w.unbind(2), u0.unbind(2),
                                                        kw.unbind(2), carries)):
        if state is None:
            u = un
            out.append(an @ u)
        else:
            u = un-wn @ state
            out.append(qn @ state+an @ u)
        if n == last: break
        update = kn.transpose(-1, -2) @ u
        state = update if state is None else (state if carry is None else state*carry[..., None, None])+update
    return torch.stack(out, 2)


def chunk_reference(q, k, v, *, kind, g=None, beta=None, chunk):
    """Plain-torch registered chunk algorithm; g is a log decay (KDA per feature,
    GDN and RetNet per head and token, None for the ELU+1 numerator)."""
    t = q.shape[1]
    q, k, v = (_to_chunks(x, chunk) for x in (q, k, v))
    if kind == "kda":
        out = _kda_chunks(q, k, v, _to_chunks(g, chunk), _to_chunks(beta, chunk), chunk)
    elif kind in ("gdn", "deltanet"):
        out = _gdn_chunks(q, k, v, None if g is None else _to_chunks(g, chunk), _to_chunks(beta, chunk), chunk)
    else:
        out = _simple_gla_chunks(q, k, v, None if g is None else _to_chunks(g, chunk), chunk)
    return out.movedim(1, 3).flatten(1, 2)[:, :t]


def _fla_chunks(q, k, v, *, kind, g, beta, chunk, qk_l2norm_scale=None):
    """FLA Triton kernels with every scale and dtype explicit at the boundary."""
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    from fla.ops import backends
    if not backends._DISPATCH_DISABLED:
        raise RuntimeError("set FLA_DISABLE_BACKEND_DISPATCH=1 before importing FLA (registered Triton path)")
    if not q.dtype == k.dtype == v.dtype:
        raise TypeError("FLA operands must share one dtype")
    q, k, v = (x.contiguous() for x in (q, k, v))
    if kind == "deltanet":
        from fla.ops.delta_rule import chunk_delta_rule
        # The mixer already normalizes/scales q/k. State is local to each packed row.
        y, _ = chunk_delta_rule(q, k, v, beta.contiguous(), scale=1.0, chunk_size=chunk,
                                initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False)
    elif kind == "kda":
        from fla.ops.kda import chunk_kda
        # disable_recompute: keep w/u/h for backward instead of recomputing them;
        # outputs and gradients are bitwise identical (EXPERIMENTS record 016).
        y, _ = chunk_kda(q, k, v, g.float().contiguous(), beta.contiguous(), scale=1.0, chunk_size=chunk,
                         disable_recompute=True)
    elif kind == "gdn":
        from fla.ops.kda import chunk_kda
        # fla 0.5.2 refuses chunk_gated_delta_rule's backward on Hopper with Triton 3.4-3.7.0
        # (gated chunk_bwd_dqkwg, issue #640). The KDA kernels evaluate the same algebra with the
        # head gate broadcast over key features, exactly (see _gdn_chunks); the redundant
        # per-feature gate work is execution cost, not theory cost (COSTS.md). The q/k L2 norm
        # and the q scale run inside the kernel (FP32 norm, fused backward; record 025).
        gate = g.float().unsqueeze(-1).expand(*k.shape).contiguous()
        y, _ = chunk_kda(q, k, v, gate, beta.contiguous(), scale=qk_l2norm_scale, chunk_size=chunk,
                         disable_recompute=True, use_qk_l2norm_in_kernel=True)
    else:
        from fla.ops.simple_gla import chunk_simple_gla
        y, _ = chunk_simple_gla(q, k, v, g_gamma=None if g is None else g.float().contiguous(),
                                scale=1.0, chunk_size=chunk)
    return y


@contextmanager
def _ieee_fp32_matmul():
    """Disable TF32 (enabled globally by the runtime) for the enclosed matmuls."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _elu_numerator(q, k, v, chunk):
    """`chunk_reference` for ELU+1 with the chunk loop batched: the same
    per-chunk updates, the same nC-2 state additions (as one cumsum) and the
    same read-outs, as one matmul each instead of one per chunk."""
    t = q.shape[1]
    q, k, v = (_to_chunks(x, chunk) for x in (q, k, v))
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    out = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0) @ v
    if q.shape[2] > 1:
        states = (k[:, :, :-1].transpose(-1, -2) @ v[:, :, :-1]).cumsum(2)
        out = torch.cat((out[:, :, :1], out[:, :, 1:]+q[:, :, 1:] @ states), 2)
    return out.movedim(1, 3).flatten(1, 2)[:, :t]


def _elu_ratio(q, k, v, chunk):
    num = _elu_numerator(q, k, v, chunk)
    return num/((q*k.cumsum(1)).sum(-1, keepdim=True)+1e-6)


class _ELUChunkFP32(torch.autograd.Function):
    """Both ELU chunk forms with forward/backward in IEEE FP32 (TF32 off).

    The q/k gradients of phi(q)S/phi(q)z are a small difference of two large
    terms once the phi(k) sums grow; a BF16 numerator against an FP32
    denominator loses them entirely (EXPERIMENTS record 017). The same
    registered chunk algebra is built inside forward and differentiated in
    backward under IEEE precision; nothing is recomputed. elu_norm also needs
    this precision before its output RMSNorm. Return the input feature dtype,
    which LinearMixer keeps at FP32 until after division or normalization.
    """
    @staticmethod
    def forward(ctx, q, k, v, chunk, kind):
        acc = torch.float64 if q.dtype == torch.float64 else torch.float32
        leaves = [x.detach().to(acc).requires_grad_(x.requires_grad) for x in (q, k, v)]
        with _ieee_fp32_matmul(), torch.enable_grad():
            y = (_elu_ratio if kind == "elu_linear" else _elu_numerator)(*leaves, chunk)
        ctx.graph, ctx.dtypes = (leaves, y), (q.dtype, k.dtype, v.dtype)
        return y.detach().to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        leaves, y = ctx.graph
        need = [x for x in leaves if x.requires_grad]
        with _ieee_fp32_matmul():
            found = iter(torch.autograd.grad(y, need, grad.to(y.dtype)))
        del ctx.graph
        return (*(next(found).to(d) if x.requires_grad else None for x, d in zip(leaves, ctx.dtypes)), None, None)


def chunk_attention(q, k, v, *, kind, g=None, beta=None, chunk, implementation=None, qk_l2norm_scale=None):
    """Chunked mixing: FLA for KDA/RetNet/GDN; IEEE FP32 for the ELU methods.

    RetNet passes g as the per-head log decay [H], GDN as [B, T, H]. The ELU+1 normalizer keeps the
    reference definition exactly: phi(q).cumsum(phi(k)) + 1e-6 in FP32. On CUDA
    the ELU methods use the same batched torch algebra as the CPU cost audit.
    GDN passes raw q/k with qk_l2norm_scale = d^-1/2: q <- scale * q/|q|, k <- k/|k| here
    (torch, the audited form) or inside the FLA kernel.
    """
    implementation = implementation or ("fla" if q.is_cuda else "torch")
    if implementation not in ("torch", "fla"):
        raise ValueError("unknown chunk implementation")
    if (kind == "gdn") != (qk_l2norm_scale is not None):
        raise ValueError("GDN, and only GDN, normalizes q/k inside chunk_attention")
    if kind in ("elu_linear", "elu_norm", "elu_qknorm"):
        return _ELUChunkFP32.apply(q, k, v, chunk, kind)
    if implementation == "fla":
        y = _fla_chunks(q, k, v, kind=kind, g=g, beta=beta, chunk=chunk, qk_l2norm_scale=qk_l2norm_scale)
    elif implementation == "torch":
        if kind == "retnet": g = g.expand(*q.shape[:2], -1)
        if qk_l2norm_scale is not None:
            q, k = F.normalize(q, dim=-1)*qk_l2norm_scale, F.normalize(k, dim=-1)
        y = chunk_reference(q, k, v, kind=kind, g=g, beta=beta, chunk=chunk)
    return y


@functools.cache
def _compiled_gated_output():
    return torch.compile(lambda y, gate, gain: (y*gain)*F.silu(gate), dynamic=False, fullgraph=True,
                         options={"emulate_precision_casts": True, "triton.autotune_pointwise": False,
                                  "triton.coalesce_tiling_analysis": False, "dynamic_scale_rblock": False})


def fused_gated_rms_norm(y, gate, gain, preserve_rounding=False):
    """RMSNorm(y) * gain * SiLU(gate), preserving KDA's native norm/rounding when requested.

    GDN uses FLA's FP32 internals and one output rounding. Its epsilon is the framework
    norm()'s: the FP32 compute-type epsilon for BF16/FP32 inputs, not finfo(bfloat16).
    """
    if preserve_rounding:
        # KDA retains native RMSNorm and its BF16 boundary before the fused gain/gate.
        return _compiled_gated_output()(norm(y), gate, gain)
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    from fla.modules.fused_norm_gate import rms_norm_gated
    eps = torch.finfo(torch.float64 if y.dtype == torch.float64 else torch.float32).eps
    return rms_norm_gated(y, gate, gain, None, activation="silu", eps=eps)


class CausalDepthwise(nn.Module):
    def __init__(self, width, size, fused=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        nn.init.normal_(self.weight, std=size**-.5)
        self.size = size
        self.fused = fused

    def forward(self, x):
        if self.fused and x.is_cuda:
            # Same causal depthwise conv + SiLU in one FLA Triton kernel on the
            # [B, T, D] layout (no transpose/pad copies, FP32 accumulation).
            # The CPU path below is the audited and tested form.
            os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
            from fla.modules.convolution import causal_conv1d
            return causal_conv1d(x.contiguous(), weight=self.weight.squeeze(1).to(x.dtype), activation="silu")[0]
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size-1, 0)),
                     self.weight.to(x.dtype), groups=x.shape[-1])
        return F.silu(y.transpose(1, 2))


class LinearMixer(nn.Module):
    """Method-specific mixer inside the common native outer block.

    VE is applied to projected V, an explicit native adaptation. Q/K use each
    method's own feature/position rule rather than native softmax QK norm.
    """
    def __init__(self, config, layer_idx, kind=None):
        super().__init__()
        self.config = config
        # A hybrid passes its per-layer kind; a pure row uses its method.
        self.kind = kind or ("kda" if config.method == "kda_no_ve" else config.method)
        d, h = config.width, config.heads
        self.c_q = Linear(d, d, bias=False)
        self.c_k = Linear(d, d, bias=False)
        self.c_v = Linear(d, d, bias=False)
        self.c_proj = Linear(d, d, bias=False)
        self.ve_gate = Linear(12, h, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.last_aux = None
        self.last_stats = {}
        if self.kind in ("deltanet", "kda", "gdn"):
            fused = config.backend in CHUNK_BACKENDS
            self.q_conv = CausalDepthwise(d, config.conv_size, fused)
            self.k_conv = CausalDepthwise(d, config.conv_size, fused)
            self.v_conv = CausalDepthwise(d, config.conv_size, fused)
            self.beta_proj = Linear(d, h, bias=False)
        if self.kind in ("gla", "kda"):
            self.gate_down = Linear(d, config.gate_rank, bias=False)
            self.gate_up = Linear(config.gate_rank, d, bias=False)
        if self.kind == "kda":
            self.log_decay = nn.Parameter(torch.zeros(h, config.head_dim))
            self.dt_bias = nn.Parameter(torch.full((h, config.head_dim), -3.))
        if self.kind == "gdn":
            # One scalar forget gate per head (GDN), not KDA's per-feature low-rank gate.
            self.decay_proj = Linear(d, h, bias=False)
            self.log_decay = nn.Parameter(torch.empty(h))
            self.dt_bias = nn.Parameter(torch.empty(h))
            init_gdn_gate(self.log_decay, self.dt_bias)
        if self.kind in ("gla", "kda", "gdn", "retnet"):
            self.output_gate = Linear(d, d, bias=False)
        if self.kind != "elu_linear":
            self.output_gain = nn.Parameter(torch.ones(config.head_dim))

    def _retention_position(self, x):
        d = x.shape[-1]
        inv = 10000.**(-torch.arange(0, d, 2, device=x.device, dtype=torch.float32)/d)
        angles = torch.arange(x.shape[1], device=x.device)[:, None]*inv[None]
        cos, sin = angles.cos().to(x)[None, :, None], angles.sin().to(x)[None, :, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even*cos-odd*sin, even*sin+odd*cos), -1).flatten(-2)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("reference linear state API is separate from native KV cache")
        c = self.config
        chunked = c.backend in CHUNK_BACKENDS
        b, t, d = x.shape
        q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        if self.kind in ("deltanet", "kda", "gdn"):
            q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)
        q, k, v = (z.reshape(b, t, c.heads, c.head_dim) for z in (q, k, v))
        if ve is not None:
            gate = 3*torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate.unsqueeze(-1)*ve.reshape_as(v)
        beta, decay = None, None
        if self.kind in ("elu_linear", "elu_norm", "elu_qknorm"):
            # elu_norm: the same features and state without the phi(q).z
            # denominator; the shared output RMSNorm and gain below normalize it.
            # elu_qknorm also applies the shell's parameter-free RMSNorm (in FP32)
            # to q/k first, so the loss no longer depends on the projection scale.
            if self.kind == "elu_qknorm":
                q, k = (norm(z.to(accumulation_dtype(z)))*ELU_QK_TEMPERATURE for z in (q, k))
            q, k = elu_feature_map(q), elu_feature_map(k)
        elif self.kind in ("deltanet", "kda", "gdn"):
            if not (self.kind == "gdn" and chunked):  # GDN's chunk path normalizes in chunk_attention
                q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
                q = q*c.head_dim**-.5
            beta = torch.sigmoid(self.beta_proj(x))
        elif self.kind == "retnet":
            q, k = self._retention_position(q), self._retention_position(k)
            q = q*c.head_dim**-.5
            gamma = 1-2.**(-5-torch.arange(c.heads, device=x.device, dtype=torch.float32))
            decay = gamma.log() if chunked else gamma[None, None, :, None].expand(b, t, -1, 1)
        elif self.kind == "gla":
            q = q*c.head_dim**-.5
        if self.kind in ("gla", "kda"):
            gate = self.gate_up(self.gate_down(x)).reshape(b, t, c.heads, c.head_dim)
            if self.kind == "gla": decay = (F.logsigmoid(gate.float())/16).exp()
            else:
                decay = -self.log_decay.exp()*F.softplus(gate.float()+self.dt_bias)
                if not chunked: decay = decay.exp()
        elif self.kind == "gdn":
            # log alpha = -A_h softplus(W_a x + dt_bias_h), one per head and token (Mamba2 form).
            decay = -self.log_decay.exp()*F.softplus(self.decay_proj(x).float()+self.dt_bias)
            if not chunked: decay = decay.exp().unsqueeze(-1)
        if chunked:
            # Chunk kernels take log decay; RetNet's is one constant per head, GDN's one per head and token.
            y = chunk_attention(q, k, v, kind=self.kind, g=decay, beta=beta, chunk=CHUNK_BACKENDS[c.backend],
                                qk_l2norm_scale=c.head_dim**-.5 if self.kind == "gdn" else None)
        else:
            y, _ = recurrent_attention(q, k, v, kind=self.kind, decay=decay, beta=beta)
        # KDA preserves its native norm boundary; GDN retains the registered FLA fusion.
        fused_gate = self.kind in ("gdn", "kda") and chunked and y.is_cuda
        if self.kind in ("elu_norm", "elu_qknorm"):
            # Use the registered ELU epsilon and gain quantization without a
            # BF16 attention boundary. This explicit BF16 epsilon differs from
            # native norm(), which uses FP32 compute-type epsilon for BF16.
            y = F.rms_norm(y, (c.head_dim,), eps=torch.finfo(x.dtype).eps)
            y = (y*self.output_gain.to(x.dtype)).to(x.dtype)
        elif self.kind == "elu_linear":
            y = y.to(x.dtype)
        elif fused_gate:
            gain = self.output_gain.to(y.dtype) if self.kind == "kda" else self.output_gain
            y = fused_gated_rms_norm(y, self.output_gate(x).reshape(b, t, c.heads, c.head_dim), gain,
                                    preserve_rounding=self.kind == "kda")
        else:
            y = norm(y)*self.output_gain.to(y.dtype)
        y = y.reshape(b, t, d)
        if hasattr(self, "output_gate") and not fused_gate:
            y = y*F.silu(self.output_gate(x))
        self.last_aux = x.new_zeros(())
        if chunked:
            # Registered FLA policy: FP32 carry/accumulation, BF16 operands and
            # chunk-boundary states, TF32 KDA triangular inverse.
            precision = ("ieee-fp32-features-attention-normalization" if self.kind in ("elu_linear", "elu_norm", "elu_qknorm")
                         else "fp32-carry/bf16-chunk-states" if x.is_cuda else str(x.dtype))
            self.last_stats = {"state_precision": precision, "backend": c.backend, "sequence_length": t}
        else:
            self.last_stats = {"state_precision": str(accumulation_dtype(x)), "sequence_length": t}
        return self.c_proj(y)
