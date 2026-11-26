"""Cost ledger of the MoE track: parameters, training FLOPs per token, calibration and admission.

    python -m moe.compute report                               # both tracks: shape, parameters, FLOPs, budgets
    python -m moe.compute calibrate --method <name> --track d768 [--track d1024]

Training FLOPs per input token = measured matmul FLOPs + attention edges charged by rule.

- Matmul FLOPs are measured, not declared: every block (and the head) runs one forward and backward pass at
  the full configured size under torch.utils.flop_counter.FlopCounterMode on CPU, one block at a time, with
  `tokens` input tokens. Dropless routing makes matmul FLOPs independent of which experts are chosen, so a
  short random input measures the per-token cost exactly. Embedding lookups and elementwise work are free,
  as in the dense protocol's main-term convention.
- Softmax attention (moe.model.attention) is replaced by a zero-FLOP stand-in during measurement and charged
  by rule: 3 x 4 x heads x head_dim x (causal or windowed key edges per query at the 4,096-token context).

The baseline's measured value is the track's compute anchor. A method is admitted when its FLOPs per token are
within 1% of the anchor; a finished run must stay within 2% cumulatively. Calibration moves one declared shape
knob (expert_width by default) to get there. The cost record fingerprints the shared core and the method's own
files, so formal training refuses a record that no longer matches the code.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import datetime
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from moe import model as model_lib
from moe.spec import PROTOCOL, REPO, TRACKS, config_path, track as get_track, write_config

FORMULA_VERSION = "moe-matmul-measured-attention-rule-v1"
EVIDENCE = Path(__file__).resolve().parent / "evidence" / "costs"


def attention_edges(length, left):
    """Query-key pairs of causal attention over `length` tokens; `left` previous keys (-1: unlimited)."""
    width = length if left < 0 else min(length, left + 1)
    return width * (width + 1) // 2 + (length - width) * width


def attention_flops(config):
    """Training FLOPs per token of the softmax-attention kernels (QK^T and AV, forward + backward)."""
    S = config.sequence_len
    edges = sum(attention_edges(S, config.window(i)) for i in range(config.layers))
    return 3 * 4 * config.heads * config.head_dim * edges / S


def analytic(config):
    """Closed-form training FLOPs per token of the *baseline* structure (a cross-check of the measurement)."""
    d, L, hd, r = config.width, config.layers, config.head_dim, config.gated_norm_rank
    attn_proj = 2 * d * hd * (2 * config.heads + 2 * config.kv_heads)
    head_gate = 2 * d * config.heads
    gated_norms = 2 * (2 * d * r + 2 * r * d)          # attention and MLP GatedNorms of a block
    router = 2 * d * config.routed_experts
    routed = config.top_k * 6 * d * config.expert_width
    shared = 6 * d * config.shared_expert_width
    forward = dict(attention_projections=L * (attn_proj + head_gate), gated_norms=L * gated_norms + 2 * 4 * d * r,
                   router=L * router, routed_experts=L * routed, shared_expert=L * shared,
                   head=2 * d * config.vocab_size)
    train = {k: 3 * v for k, v in forward.items()}
    train["attention_kernel"] = attention_flops(config)
    train["total"] = sum(train.values())
    return train


@contextmanager
def charged_attention():
    """Replace softmax attention by a zero-matmul stand-in that keeps every input in the autograd graph."""
    original = model_lib.attention

    def stand_in(q, k, v, mask):
        groups = q.size(2) // k.size(2)
        k, v = (t.repeat_interleave(groups, dim=2) for t in (k, v))
        return v + 0 * (q + k)

    model_lib.attention = stand_in
    try:
        yield
    finally:
        model_lib.attention = original


def _count(fn):
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def measure(config, tokens=256, seed=0):
    """Measured training matmul FLOPs per token, block by block, plus the attention rule."""
    from moe.plugins import hooks_for
    torch.manual_seed(seed)
    hooks = hooks_for(config.method)
    rows = max(1, tokens // config.sequence_len)
    length = tokens // rows
    rows = max(rows, config.qb_group_rows)        # the router balances whole groups of rows
    tokens = rows * length
    per_block = []
    with charged_attention():
        if hooks.model is not None:
            lm = model_lib.build_model(config, "cpu").train()
            idx = torch.randint(0, config.vocab_size, (rows, length))
            matmul = _count(lambda: lm(idx, idx).backward())
            per_block = None
        else:
            cos_sin = model_lib.rotary(config, length, "cpu")
            masks = model_lib.Masks(torch.zeros(rows, length, dtype=torch.long), config)
            for layer in range(config.layers):
                block = model_lib.Block(config, layer, hooks)
                if hooks.block:
                    block = hooks.block(block, config, layer)
                model_lib.initialize_parameters(block, config)
                block.train()
                x = torch.randn(rows, length, config.width, requires_grad=True)
                per_block.append(_count(lambda: block(x, cos_sin, masks).float().square().mean().backward()) / tokens)
                del block
            ends = torch.nn.ModuleDict(dict(embed_gated=model_lib.GatedNorm(config.width, config.gated_norm_rank),
                                            final_gated=model_lib.GatedNorm(config.width, config.gated_norm_rank),
                                            head=torch.nn.Linear(config.width, config.vocab_size, bias=False)))
            model_lib.initialize_parameters(ends, config)
            h = torch.randn(tokens, config.width, requires_grad=True)
            outer = _count(lambda: ends.head(ends.final_gated(ends.embed_gated(h))).square().mean().backward()) / tokens
            matmul = (sum(per_block) + outer) * tokens
    matmul_per_token = matmul / tokens
    attention = attention_flops(config)
    return dict(matmul_per_token=matmul_per_token, attention_per_token=attention,
                flops_per_token=matmul_per_token + attention, per_block=per_block, tokens_measured=tokens)


def source_digest(method):
    from moe.plugins import source_files
    files = {str(p.resolve().relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source_files(method)}
    return dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def parameters(config):
    lm = model_lib.build_model(config, "meta")
    return lm.parameter_counts()


def cost_record(config, tokens=256):
    m = measure(config, tokens)
    tr = get_track(config.track)
    return dict(formula=FORMULA_VERSION, method=config.method, track=config.track, model_hash=config.fingerprint,
                torch=torch.__version__.split("+")[0], source=source_digest(config.method),
                flops_per_token=m["flops_per_token"], matmul_per_token=m["matmul_per_token"],
                attention_per_token=m["attention_per_token"], per_block=m["per_block"],
                training_flops=m["flops_per_token"] * tr.total_tokens, parameters=parameters(config))


def anchor(track_name):
    """The track's compute anchor: the admitted baseline's recorded FLOPs per token."""
    path = EVIDENCE / track_name / "baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"no baseline cost record for {track_name}; run python -m moe.compute calibrate "
                                f"--method baseline --track {track_name}")
    return json.loads(path.read_text())["flops_per_token"]


STEPS = {"expert_width": 32, "shared_expert_width": 64, "layers": 1}
FLOORS = {"expert_width": 32, "shared_expert_width": 0, "layers": 2}


def calibrate(method, track_name, tokens=256, write=True):
    """Fit the method's declared knob so its FLOPs per token land within 1% of the track anchor.

    Matmul FLOPs per token are affine in each knob, so two measurements give the slope; the nearest grid
    values around the solution are then measured and the closest one is kept.
    """
    from moe.plugins import config_for, declaration
    config = config_for(method, track_name)
    knob = declaration(method).get("calibrate", "expert_width")
    if method == "baseline":
        record = cost_record(config, tokens)
        target = record["flops_per_token"]
    else:
        target = anchor(track_name)
        record = cost_record(config, tokens)
        if knob != "none" and abs(record["flops_per_token"] / target - 1) > PROTOCOL.preflight_tolerance:
            step, base = STEPS[knob], getattr(config, knob)
            probe = replace(config, **{knob: base + 8 * step})
            slope = (cost_record(probe, tokens)["flops_per_token"] - record["flops_per_token"]) / (8 * step)
            if slope <= 0:
                raise ValueError(f"{knob} does not change this method's cost; declare another knob")
            solution = base + (target - record["flops_per_token"]) / slope
            grid = sorted({max(FLOORS[knob], step * math.floor(solution / step)),
                           max(FLOORS[knob], step * math.ceil(solution / step))})
            tried = [(c, cost_record(c, tokens)) for c in (replace(config, **{knob: v}) for v in grid)]
            config, record = min(tried, key=lambda t: abs(t[1]["flops_per_token"] / target - 1))
    gap = record["flops_per_token"] / target - 1
    admitted = abs(gap) <= PROTOCOL.preflight_tolerance
    record.update(anchor_flops_per_token=target, relative_gap=gap, admitted=admitted, knob=knob,
                  created=datetime.date.today().isoformat())
    if write:
        evidence = EVIDENCE / track_name / f"{method}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(record, indent=2) + "\n")
        if admitted:
            write_config(config_path(method, track_name), config, dict(
                status="admitted", cost_record=str(evidence.relative_to(REPO)), flops_per_token=record["flops_per_token"],
                relative_gap=gap, source_sha256=record["source"]["sha256"], torch=record["torch"]))
    return config, record


def report():
    rows = {}
    for name, tr in TRACKS.items():
        config = model_lib.MoEConfig.baseline(name)
        a = analytic(config)
        counts = parameters(config)
        # Marin's budget formula: 2 x (attention, router, active and shared experts) + non-causal attention,
        # without the head, the GatedNorms, the head-wise gates and the gains.
        d, r = config.width, config.gated_norm_rank
        gates = (2 * config.layers + 2) * 2 * d * r + config.layers * d * config.heads + (2 * config.layers + 2) * d
        marin_forward = (2 * (counts["active_non_embedding"] - gates)
                         + 4 * config.sequence_len * config.heads * config.head_dim * config.layers)
        rows[name] = dict(
            marin_run=tr.marin_run, shape=dict(width=tr.width, layers=tr.layers, heads=tr.heads, kv_heads=tr.kv_heads,
                                               head_dim=tr.head_dim, routed_experts=tr.routed_experts, top_k=tr.top_k,
                                               expert_width=tr.expert_width, shared_expert_width=tr.shared_expert_width,
                                               long_layers=list(config.long_layers)),
            parameters=counts, sequence_len=PROTOCOL.sequence_len, tokens_per_update=tr.tokens_per_update,
            updates=tr.updates, total_tokens=tr.total_tokens, tokens_per_active_parameter=tr.total_tokens / counts["active"],
            flops_per_token=a["total"], flops_breakdown=a, training_flops=a["total"] * tr.total_tokens,
            marin_convention_budget=3 * marin_forward * tr.total_tokens, marin_reported_budget=tr.marin_budget_flops)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="print shapes, parameters and FLOPs of both tracks as JSON")
    c = sub.add_parser("calibrate", help="measure a method, fit its knob to the anchor, write config + evidence")
    c.add_argument("--method", required=True)
    c.add_argument("--track", action="append", choices=tuple(TRACKS), required=True)
    c.add_argument("--tokens", type=int, default=256)
    c.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.command == "report":
        print(json.dumps(report(), indent=2))
        return
    for track_name in args.track:
        config, record = calibrate(args.method, track_name, args.tokens, write=not args.dry_run)
        print(f"{args.method} @ {track_name}: {record['flops_per_token']:.6e} FLOPs/token, "
              f"gap {record['relative_gap']:+.4%}, {'admitted' if record['admitted'] else 'NOT admitted'}"
              + (f" ({record['knob']}={getattr(config, record['knob'])})" if record["knob"] != "none" else ""))
        if not record["admitted"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
