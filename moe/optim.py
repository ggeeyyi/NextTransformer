"""Marin's MuonH optimizer (experiments/grug/moe/optimizer.py, grugmuon.py, adamh.py at d259a97068) in PyTorch.

Parameter groups (Marin's create_mask, applied by name):

    adam    the embedding, every router weight, the head-wise attention gates and all 1-D gains
    adamh   the output head
    muonh   every other 2-D or 3-D matrix: attention, shared and routed experts (per expert), GatedNorms

    adam    m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;  p -= lr_adam * m_hat / (sqrt(v_hat) + eps)
    adamh   u = m_hat / (sqrt(v_hat) + eps);  W' = W - lr * u * |W| / |u|;  W <- W' * |W| / |W'|
    muonh   b = 0.95 b + g;  u = 0.95 b + g (Nesterov);  U = NewtonSchulz5(u) per matrix (per expert for 3-D);
            W' = W - lr * U * |W| / |U|;  W <- W' * |W| / |W'|       (|.| = Frobenius norm of one matrix)

The hyperball updates keep the Frobenius norm of every MuonH/AdamH matrix at its initial value. lr is the
MuonH rate for muonh and adamh, the Adam rate for adam, both on the same schedule. There is no weight decay
and no gradient clipping. Parameters may be FSDP2 DTensors sharded on dim 0: 3-D expert tensors shard whole
experts, so their per-expert work is local; 2-D matrices are gathered for Newton-Schulz and their norms are
reduced across the shard group.
"""
import torch
import torch.distributed as dist

QUINTIC = ((4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
           (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012))


def group_of(name, p):
    if name == "embed.weight" or ".router." in f".{name}" or name.endswith("head_gate.weight") or p.ndim < 2:
        return "adam"
    if name == "head.weight":
        return "adamh"
    if p.ndim in (2, 3):
        return "muonh"
    return "adam"


def newton_schulz(u, steps=5, eps=1e-8):
    """Quintic Newton-Schulz orthogonalization of each matrix of a (..., m, n) tensor, in fp32."""
    x = u.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    for a, b, c in QUINTIC[:steps]:
        A = x @ x.transpose(-2, -1)
        B = b * A + c * A @ A
        x = a * x + B @ x
    return x.transpose(-2, -1) if transposed else x


def _is_dtensor(t):
    return t.__class__.__name__ == "DTensor"


def _local(t):
    return t.to_local() if _is_dtensor(t) else t


def _group(p):
    if not _is_dtensor(p):
        return None
    return p.device_mesh.get_group()


def _sumsq(local, p, dims=None):
    """Global squared Frobenius norm (dims=None) or per-leading-index norms (3-D, dims=(1, 2), local)."""
    if dims is not None:
        return local.float().square().sum(dim=dims, keepdim=True)
    value = local.float().square().sum()
    group = _group(p)
    if group is not None:
        dist.all_reduce(value, group=group)
    return value


class MuonH(torch.optim.Optimizer):
    def __init__(self, named_parameters, *, adam_lr, muonh_lr, beta1, beta2, eps, momentum=0.95, ns_steps=5):
        groups = {"adam": [], "adamh": [], "muonh": []}
        self.names = {}
        for name, p in named_parameters:
            if p.requires_grad:
                groups[group_of(name, p)].append(p)
                self.names[p] = name
        param_groups = [dict(params=ps, kind=k, lr=adam_lr if k == "adam" else muonh_lr,
                             peak_lr=adam_lr if k == "adam" else muonh_lr)
                        for k, ps in groups.items() if ps]
        super().__init__(param_groups, dict(beta1=beta1, beta2=beta2, eps=eps, momentum=momentum, ns_steps=ns_steps))

    def set_schedule(self, multiplier):
        for group in self.param_groups:
            group["lr"] = group["peak_lr"] * multiplier

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                kind = group["kind"]
                if kind == "muonh":
                    self._muonh(p, group)
                else:
                    self._adam(p, group, hyperball=kind == "adamh")
        return loss

    def _adam(self, p, group, hyperball):
        state = self.state[p]
        w, g = _local(p), _local(p.grad).float()
        if not state:
            # State keeps the parameter's own type: a DTensor stays a DTensor, so distributed
            # checkpointing records its sharding and a resume on any world size reads it back.
            state["step"] = 0
            state["m"] = torch.zeros_like(p, dtype=torch.float32)
            state["v"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] += 1
        b1, b2, eps, t = group["beta1"], group["beta2"], group["eps"], state["step"]
        m, v = _local(state["m"]), _local(state["v"])
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        u = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        if not hyperball:
            w.add_(u.to(w.dtype), alpha=-group["lr"])
            return
        self._hyperball(p, w, u, group["lr"], dims=None)

    def _muonh(self, p, group):
        state = self.state[p]
        w, g = _local(p), _local(p.grad).float()
        if "buf" not in state:
            state["buf"] = torch.zeros_like(p, dtype=torch.float32)
        buf = _local(state["buf"])
        buf.mul_(group["momentum"]).add_(g)
        u = buf * group["momentum"] + g
        if p.ndim == 3:
            update = newton_schulz(u, group["ns_steps"])          # whole experts are local
            self._hyperball(p, w, update, group["lr"], dims=(1, 2))
            return
        if _is_dtensor(p):
            from torch.distributed.tensor import DTensor, distribute_tensor
            full = DTensor.from_local(u, p.device_mesh, p.placements, shape=p.shape, stride=p.stride()).full_tensor()
            update = distribute_tensor(newton_schulz(full, group["ns_steps"]), p.device_mesh, p.placements,
                                       src_data_rank=None).to_local()
        else:
            update = newton_schulz(u, group["ns_steps"])
        self._hyperball(p, w, update, group["lr"], dims=None)

    @staticmethod
    def _hyperball(p, w, update, lr, dims):
        """W' = W - lr * U * |W| / |U|;  W <- W' * |W| / |W'|  (norm per matrix; per expert for 3-D)."""
        pn = _sumsq(w, p, dims).sqrt()
        un = _sumsq(update, p, dims).sqrt()
        moved = w.float() - lr * update * (pn / un.clamp_min(1e-10))
        mn = _sumsq(moved, p, dims).sqrt()
        w.copy_((moved * (pn / mn.clamp_min(1e-10))).to(w.dtype))


def build(model, track):
    from moe.spec import PROTOCOL
    return MuonH(model.named_parameters(), adam_lr=track.adam_lr, muonh_lr=track.muonh_lr, beta1=PROTOCOL.beta1,
                 beta2=track.beta2, eps=track.eps, momentum=PROTOCOL.muon_momentum,
                 ns_steps=PROTOCOL.newton_schulz_steps)
