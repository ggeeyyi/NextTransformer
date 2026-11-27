"""Gym hooks for the upstream registered methods (configs/<family>/*.json).

Imported attempts use a two-line `model.py` that re-exports these hooks, so the
upstream ReferenceLM runs unchanged through the gym audit and trainer. The
stage rules mirror nexttransformer/protocol/calibrate.py::Auditor.stages and audit_graph; at
non-L scales the fixed-length stages (sparse burn-in, Ouro gate-only) keep their
share of the run.
"""
import json

import torch

from nexttransformer.models.looped import huginn_depth
from nexttransformer.models.lm import build_model
from nexttransformer.models.config import CHUNK_BACKENDS, LINEAR, ModelConfig, debug_config
from nexttransformer.protocol.spec import Protocol

_cache = {}


def _fields(config):
    """ModelConfig fields; `layer_mixers` is a derived hybrid layout that serialization adds."""
    return {k: v for k, v in config.items() if k != "layer_mixers"}


def model_config(config):
    fields = _fields(config)
    key = json.dumps(fields, sort_keys=True)     # a plugin's `options` is a dict, so the fields themselves are no key
    if key not in _cache: _cache[key] = ModelConfig(**fields)
    c = _cache[key]
    if "layer_mixers" in config and config["layer_mixers"] != c.serialized().get("layer_mixers"):
        raise ValueError("layer_mixers differs from the registered hybrid layout")
    return c


def build(config):
    return build_model(model_config(config), torch.get_default_device(), seed=Protocol().seed)


def _scaled(steps, total_steps):
    """Upstream fixed-length stage, scaled to a run of `total_steps` updates."""
    full = Protocol().steps
    return steps if total_steps == full else round(steps * total_steps / full)


def stage_for_step(config, step, total_steps):
    c = model_config(config)
    if c.method == "huginn": return f"depth{huginn_depth(c, step)}"
    if c.has_sparse_burnin:
        return "burnin" if step < _scaled(c.burnin_steps, total_steps) else "joint"
    if c.method == "ouro":
        return "gate_only" if step >= total_steps - _scaled(c.ouro_gate_steps, total_steps) else "joint"
    return "joint"


def set_stage(model, stage, *, audit, step, total_steps):
    c = model.config
    if audit:
        # Representative upstream steps, exactly as nexttransformer/protocol/compute.py::audit_graph.
        p = Protocol()
        if stage.startswith("depth"):
            model.set_training_stage(c.burnin_steps + 1)
            model.accounting_depth = int(stage[5:])
            return "joint"
        representative = 0 if stage == "burnin" else p.steps - c.ouro_gate_steps if stage == "gate_only" \
            else c.burnin_steps + 1
        model.set_training_stage(representative)
        return stage
    if c.method == "ouro":
        raise NotImplementedError("Ouro's gate-only optimizer transition is not supported by the gym trainer")
    model.set_training_stage(step)
    return stage


def submission_config(method_config, *, method=None):
    """config.json content for an imported upstream method."""
    c = model_config(method_config)
    debug = debug_config(c.method)
    debug_model = {k: v for k, v in _fields(debug.serialized()).items() if method_config.get(k) != v}
    audit = dict(mode="direct")
    # As nexttransformer/protocol/calibrate.py::Auditor.full: only pure linear methods are cross-checked
    # (chunk) or extrapolated (scan); hybrids keep softmax layers and are audited directly.
    if c.backend in CHUNK_BACKENDS:
        chunk = CHUNK_BACKENDS[c.backend]
        audit = dict(mode="direct", pair_block=chunk,
                     chunk_fields=["width", "heads", "mlp_width", "vocab_size", "gate_rank", "conv_size"],
                     chunk_values=[c.head_dim])
        if c.method in LINEAR: audit["cross_check_lengths"] = [3 * chunk, 4 * chunk, 5 * chunk]
    elif c.method in LINEAR:
        audit = dict(mode="affine", probe_lengths=[16, 32, 48])
    return dict(upstream_method=method or c.method, model=_fields(c.serialized()), model_dim=c.width,
                model_dim_field="width", vocab_size=c.vocab_size, sequence_len=c.sequence_len,
                aux_loss=c.has_sparse_burnin or c.method in ("ut_act", "ouro", "nsa"),
                tunable={}, audit=audit,
                debug=dict(model=debug_model, vocab_size=debug.vocab_size, sequence_len=debug.sequence_len))


MODEL_PY = '''"""Upstream registered method, unchanged: hooks live in nextt_gym/upstream_model.py.

Read the implementation through the read-only resource nextt://upstream/methods/{method}.
"""
from nextt_gym.upstream_model import build, set_stage, stage_for_step  # noqa: F401
'''
