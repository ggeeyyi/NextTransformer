"""Fixed optimizer recipe by parameter role.

The upstream groups (nexttransformer/training/optimizer.py::parameter_groups) are tied to the
native GPT shell. Any architecture here declares a role per parameter instead,
and every role receives exactly the upstream hyperparameters, so the optimizer
recipe stays a controlled variable. tests/test_roles.py checks that the default
naming rules reproduce the upstream groups for every registered method.
"""
from collections import defaultdict
import math

from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.training.sharded import ExperimentOptimizer
from nexttransformer.native.optimizer import D12_SCALING_PARAMS, D24_SCALING_PARAMS
from nexttransformer.protocol.spec import Protocol

ROLES = ("unembedding", "embedding", "value_embedding", "resid_scalar", "x0_scalar",
         "gate_scalar", "aux", "vector", "matrix")

# Upstream naming rules, used when a model does not declare roles itself.
_AUXILIARY = ("halter.", ".index_", ".beta_proj.", ".gate_down.", ".gate_up.", ".decay_proj.",
              ".slot_proj.", ".branch_gate.", ".output_gate.",
              ".log_decay", ".dt_bias", ".compress_position", "residual_maps.", "depth_queries")


def default_role(name, param):
    if name.startswith("lm_head."): return "unembedding"
    if name.startswith("transformer.wte."): return "embedding"
    if name.startswith("value_embeds."): return "value_embedding"
    if name == "resid_lambdas": return "resid_scalar"
    if name == "x0_lambdas": return "x0_scalar"
    if name in ("smear_gate.weight", "smear_lambda", "backout_lambda"): return "gate_scalar"
    if any(key in name for key in _AUXILIARY): return "aux"
    return "matrix" if param.ndim == 2 else "vector"


def parameter_roles(model):
    """name -> role for every parameter; a model may override via `param_roles()`."""
    declared = model.param_roles() if hasattr(model, "param_roles") else {}
    roles = {}
    for name, p in model.named_parameters():
        role = declared.get(name) or default_role(name, p)
        if role not in ROLES: raise ValueError(f"unknown role {role!r} for parameter {name}")
        if role == "matrix" and p.ndim != 2: raise ValueError(f"matrix role requires a 2D parameter: {name}")
        roles[name] = role
    unknown = set(declared) - set(roles)
    if unknown: raise ValueError(f"param_roles names unknown parameters: {sorted(unknown)}")
    return roles


def parameter_groups(model, model_dim, train=None):
    """Upstream Muon/AdamW groups, assigned by role. `model_dim` sets the upstream LR width scaling."""
    c = train or NativeTrainConfig()
    Protocol().validate_train_config(c)
    dscale = (model_dim / 768) ** -.5
    bscale = math.sqrt(c.global_batch_tokens / 524288)
    roles = parameter_roles(model)
    by_role = defaultdict(list)
    for name, p in model.named_parameters():
        if p.requires_grad: by_role[roles[name]].append((name, p))
    adam = {  # role: (lr, betas, weight_decay); order matches the upstream group order
        "unembedding": (c.unembedding_lr * bscale * dscale, (.8, .96), .01),
        "embedding": (c.embedding_lr * bscale * dscale, (.8, .995), .001),
        "value_embedding": (c.embedding_lr * bscale * dscale * .5, (.8, .995), .01),
        "resid_scalar": (c.scalar_lr * bscale * .01, (.8, .95), .05),
        "x0_scalar": (c.scalar_lr * bscale, (.96, .95), 0.),
        "gate_scalar": (.2, (.8, .95), 0.),
        "aux": (c.adamw_lr * bscale, (.9, .95), 0.),
        "vector": (c.adamw_lr * bscale, (.9, .95), 0.),
    }
    groups = []
    for role, (lr, betas, decay) in adam.items():
        items = by_role.get(role)
        if items:
            groups.append(dict(kind="adamw", role=role, params=[p for _, p in items], lr=lr, initial_lr=lr,
                               betas=betas, eps=1e-10, weight_decay=decay,
                               parameter_names=[n for n, _ in items]))
    wd = c.weight_decay * bscale * D12_SCALING_PARAMS / D24_SCALING_PARAMS
    shapes = defaultdict(list)
    for name, p in by_role.get("matrix", []): shapes[tuple(p.shape)].append((name, p))
    for shape in sorted(shapes):
        items = shapes[shape]
        groups.append(dict(kind="muon", role="matrix", params=[p for _, p in items], lr=c.muon_lr * bscale,
                           initial_lr=c.muon_lr * bscale, momentum=.95, ns_steps=5, beta2=.9,
                           weight_decay=wd, initial_weight_decay=wd, parameter_names=[n for n, _ in items]))
    owned = [id(p) for g in groups for p in g["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("trainable parameters must have exactly one optimizer owner")
    return groups


def build_optimizer(model, model_dim, train=None):
    groups = parameter_groups(model, model_dim, train)
    for g in groups: g.pop("role")
    return ExperimentOptimizer(groups)
