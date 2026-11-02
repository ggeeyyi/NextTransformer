"""Training-FLOPs audit for any submitted architecture, on the upstream counter.

Same accounting as nexttransformer/protocol/compute.py and calibrate.py (formula
reference-aten-math-v2): a FakeTensor execution at formal dimensions and the
logical 512 x 2048 global batch, counting forward, autograd backward and the
unfused optimizer body per training stage. The upstream Ledger, attention edge
rules and optimizer kernels are reused unchanged; only model construction,
stage scheduling and optimizer grouping (by parameter role) are generic here.
tests/test_audit_contract.py requires bit-identical totals with the upstream
cost proofs for registered methods.
"""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nanochat.optim import adamw_step_fused, muon_step_fused
from nexttransformer.protocol.compute import FORMULA_VERSION, Ledger, attention_rules
from nexttransformer.protocol.spec import Protocol

from .roles import parameter_groups
from .task import ANCHOR_FLOPS_PER_TOKEN, FLOPS_TOLERANCE, SCALES

FIELDS = ("calls", "theory", "padded_envelope", "matrix")


def count_optimizer(groups, ledger):
    """One logical optimizer update of the given groups (nexttransformer/protocol/compute.py::count_optimizer)."""
    adam = getattr(adamw_step_fused, "_torchdynamo_orig_callable", adamw_step_fused)
    muon = getattr(muon_step_fused, "_torchdynamo_orig_callable", muon_step_fused)
    scalar = lambda value: torch.tensor(float(value))
    ledger.phase = "optimizer"
    with torch.no_grad(), ledger:
        for group in groups:
            if group["kind"] == "adamw":
                for p in group["params"]:
                    adam(p, torch.empty_like(p), torch.empty_like(p), torch.empty_like(p),
                         *[scalar(x) for x in (1, group["lr"], *group["betas"], group["eps"], group["weight_decay"])])
            else:
                params = group["params"]; m, n = params[0].shape; shape = (len(params), m, n)
                stack = torch.empty(shape); red = -1 if m >= n else -2
                second = torch.empty((len(params), m, 1) if red == -1 else (len(params), 1, n))
                muon(torch.empty_like(stack), stack, torch.empty_like(stack), second,
                     *[scalar(x) for x in (group["momentum"], group["lr"], group["weight_decay"], group["beta2"])],
                     group["ns_steps"], red)


def audit_graph(sub, stage, *, sequence_len=None, rows=None, optimizer=True):
    """Per-update cost of one training stage; no real allocation, no data."""
    p = Protocol()
    t = sequence_len or sub.sequence_len
    rows = rows or p.global_batch_tokens // p.sequence_len
    pair = sub.audit.pair_block
    if pair is not None and sub.audit_dimensions_hit_chunk():
        raise ValueError("a model dimension equals the chunk length; C x C tiles would be ambiguous")
    ledger = Ledger(pair)
    with FakeTensorMode(allow_non_fake_inputs=True):
        model = sub.instantiate("cpu")
        label = sub.set_stage(model, stage, audit=True, step=0, total_steps=p.steps)
        x = torch.zeros((rows, t), dtype=torch.long)
        with attention_rules(model, ledger, label), ledger:
            loss = model(x, x)
            ledger.phase = "backward"; loss.backward()
        if optimizer: count_optimizer(parameter_groups(model, sub.model_dim), ledger)
        parameters = sum(q.numel() for q in model.parameters())
    report = ledger.report()
    report.update(stage=stage, sequence_len=t, batch=rows, parameters=parameters)
    return report


def _affine(sub, stage, lengths):
    """Operator-wise affine extrapolation to the full context, verified at a third length."""
    step = lengths[1] - lengths[0]
    if lengths[2] - lengths[1] != step: raise ValueError("probe lengths must be evenly spaced")
    probes = [audit_graph(sub, stage, sequence_len=t) for t in lengths]
    keys = set().union(*(r["operators"] for r in probes))
    ops = {}
    for key in keys:
        item = {}
        for field in FIELDS:
            a, b, test = [r["operators"].get(key, {}).get(field, 0) for r in probes]
            if test != 2 * b - a: raise RuntimeError(f"non-affine scan count: {key}/{field}")
            numerator = (b - a) * (sub.sequence_len - lengths[0])
            if numerator % step: raise RuntimeError("nonintegral extrapolated arithmetic")
            item[field] = a + numerator // step
        ops[key] = item
    return probes[0], ops


def stage_cost(sub, stage):
    """Full-context per-update cost of `stage` (nexttransformer/protocol/calibrate.py::Auditor.full)."""
    settings = sub.audit
    if settings.mode == "affine":
        first, ops = _affine(sub, stage, settings.probe_lengths)
        out = dict(first)
        out["operators"] = dict(sorted(ops.items())); out["sequence_len"] = sub.sequence_len
        out["phases"] = {ph: sum(v["theory"] for k, v in ops.items() if k.startswith(ph + "/"))
                         for ph in ("forward", "backward", "optimizer")}
        out["total"] = sum(out["phases"].values())
        out["matrix_total"] = sum(v["matrix"] for v in ops.values())
        out["padded_envelope"] = sum(v["padded_envelope"] for v in ops.values())
        out["extrapolation"] = dict(kind="operator-wise affine scan", verified_lengths=list(settings.probe_lengths))
        return out
    direct = audit_graph(sub, stage)
    if settings.cross_check_lengths:
        _, ops = _affine(sub, stage, settings.cross_check_lengths)
        if ops != {k: {f: v[f] for f in FIELDS} for k, v in direct["operators"].items()}:
            raise RuntimeError("direct chunk audit differs from the affine cross-check")
        direct["extrapolation"] = dict(kind="direct; operator-wise affine cross-check",
                                       verified_lengths=list(settings.cross_check_lengths))
    return direct


def top_ops(record, k=10):
    total = record["total"] or 1
    merged = {}
    for key, v in record["operators"].items():
        op = key.split("/", 1)[1].removeprefix("aten.").removesuffix(".default")
        merged[op] = merged.get(op, 0) + v["theory"]
    return [dict(op=op, share=n / total) for op, n in sorted(merged.items(), key=lambda kv: -kv[1])[:k]]


def audit_submission(sub, scales=None):
    """Stage costs, then the FLOPs/token of every scale's stage schedule versus the anchor."""
    scales = list(scales or SCALES)
    # A prefix scale is matched when its parent run is: the budget is the whole run's average.
    needed = scales + [SCALES[n].prefix_of for n in scales if SCALES[n].prefix_of and SCALES[n].prefix_of not in scales]
    schedules = {name: sub.stage_schedule(SCALES[name].steps, SCALES[name].schedule_steps) for name in needed}
    stages = sorted(set().union(*schedules.values()))
    costs = {stage: stage_cost(sub, stage) for stage in stages}
    unknown = sorted({op for r in costs.values() for op in r["unknown"]})
    p = Protocol()
    per_scale = {}
    for name, schedule in schedules.items():
        scale = SCALES[name]
        total = sum(n * costs[s]["total"] for s, n in schedule.items())
        per_token = total / scale.input_tokens
        error = per_token / ANCHOR_FLOPS_PER_TOKEN - 1
        per_scale[name] = dict(stage_steps=schedule, total_training_flops=total,
                               total_training_flops_exact=str(total), flops_per_token=per_token,
                               relative_error_vs_anchor=error, within_tolerance=abs(error) <= FLOPS_TOLERANCE)
    for name in list(per_scale):
        parent = SCALES[name].prefix_of
        if parent:
            per_scale[name].update(prefix_of=parent, within_tolerance=per_scale[parent]["within_tolerance"],
                                   matched_by=f"{parent} average (prefix of the {parent} schedule)")
    per_scale = {name: per_scale[name] for name in scales}
    first = costs[stages[0]]
    joint = costs.get("joint", first)
    return dict(
        formula_version=FORMULA_VERSION, unknown_ops=unknown,
        parameters=first["parameters"],
        stages=[dict(name=s, forward=costs[s]["phases"]["forward"], backward=costs[s]["phases"]["backward"],
                     optimizer=costs[s]["phases"]["optimizer"], total=costs[s]["total"]) for s in stages],
        forward_flops_per_token=joint["phases"]["forward"] / p.global_batch_tokens,
        scales=per_scale,
        within_tolerance=not unknown and all(v["within_tolerance"] for v in per_scale.values()),
        top_ops=top_ops(joint), stage_records=costs)
