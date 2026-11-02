"""Methods contributed as plugins: one package per method, added without editing shared files.

A plugin lives in `nexttransformer/methods/<name>/`; NEXTT_METHOD_PATHS may list more plugin roots
(separated by the path separator). Its `method.json` declares the method and is read without
importing code:

    {"name": "<name>", "family": "others", "title": "...", "paper": {"title": "...", "url": "..."},
     "options": {"<option>": <default>}, "debug_options": {"<option>": <debug value>},
     "shell": {"layers": 24, "mlp_width": 6144, "window_pattern": "L"}, "layer_multiple": 1}

Its package defines one or more hooks, imported when the method is first built:

    mixer(config, layer) -> module          the attention of a layer, called as (x, ve, cos_sin, window, kv_cache)
    ffn(config, layer) -> module            the feed-forward network of a layer, called as (x)
    block(block, config, layer) -> module   wraps or replaces a block, called as (x, ve, cos_sin, window, kv_cache)
    model                                   a subclass of models.lm.ReferenceLM, for a different forward flow
    frozen = ("backout_lambda", ...)        shared-shell parameters the method's own path leaves unused
    init(model)                             initializes parameters that the shared shell does not
    stages(config) -> [(name, steps), ...]  contiguous training stages in order; default one "joint" stage
    check(config)                           raises ValueError for invalid option values

Every block keeps `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` linear
layers, which the shared shell initializes (projections into the residual stream start at zero). Every
trainable parameter must receive a gradient at every update; list unused shell parameters in `frozen`. A
module may expose a tensor `last_aux` (an auxiliary training loss) and `set_step(step)`. The model rules in
config.py cover built-in methods only, so a plugin never changes a built-in method.
"""
from functools import lru_cache
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "methods"
FAMILIES = ("looped", "sparse", "linear", "residual", "hybrid", "others")
HOOKS = ("mixer", "ffn", "block", "model", "frozen", "init", "stages", "check")
SHELL = ("layers", "mlp_width", "window_pattern")
NAME = re.compile(r"[a-z][a-z0-9_]{1,40}")
SCALARS = (str, int, float, bool)


def roots():
    """Plugin roots: this package's methods/ directory, then those listed in NEXTT_METHOD_PATHS."""
    extra = [Path(path).resolve() for path in os.environ.get("NEXTT_METHOD_PATHS", "").split(os.pathsep) if path]
    return [ROOT, *extra]


def _value(value):
    return isinstance(value, SCALARS) or isinstance(value, list) and all(_value(item) for item in value)


def discover(reserved):
    """name -> declaration of every plugin, read from its method.json without importing its code."""
    found = {}
    for root in roots():
        for path in sorted(root.glob("*/method.json")):
            spec = json.loads(path.read_text())
            name, options = spec.get("name"), spec.get("options", {})
            multiple = spec.get("layer_multiple", 1)
            problems = [message for ok, message in (
                (isinstance(name, str) and NAME.fullmatch(name) is not None, "name: lowercase letters, digits and _"),
                (name == path.parent.name, "name must equal its directory"),
                (name not in reserved and name not in found, "name is already registered"),
                (spec.get("family") in FAMILIES, f"family must be one of {', '.join(FAMILIES)}"),
                (isinstance(options, dict) and all(NAME.fullmatch(k) and _value(v) for k, v in options.items()),
                 "options map lowercase names to JSON numbers, strings, booleans or lists"),
                (set(spec.get("debug_options", {})) <= set(options), "debug_options may only set declared options"),
                (set(spec.get("shell", {})) <= set(SHELL), f"shell may set only {', '.join(SHELL)}"),
                (isinstance(multiple, int) and multiple > 0, "layer_multiple must be a positive integer"),
                ((path.parent / "__init__.py").is_file(), "the method directory needs an __init__.py"),
            ) if not ok]
            if problems: raise ValueError(f"plugin {path.parent}: " + "; ".join(problems))
            found[name] = dict(spec, path=str(path.parent))
    return found


@lru_cache(maxsize=None)
def load(name):
    """The hooks of a plugin method; its package is imported once."""
    from nexttransformer.models.registry import PLUGINS
    spec = PLUGINS[name]
    path = Path(spec["path"])
    if path.parent == ROOT:
        module = importlib.import_module(f"nexttransformer.methods.{name}")
    else:
        module_name = f"nexttransformer_plugin_{name}"
        found = importlib.util.spec_from_file_location(module_name, path / "__init__.py",
                                                       submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(found)
        sys.modules[module_name] = module
        found.loader.exec_module(module)
    hooks = SimpleNamespace(spec=spec, **{hook: getattr(module, hook, None) for hook in HOOKS})
    if not any((hooks.mixer, hooks.ffn, hooks.block, hooks.model)):
        raise ValueError(f"plugin {name} defines none of mixer, ffn, block or model")
    if hooks.model is not None:
        from nexttransformer.models.lm import ReferenceLM
        if not (isinstance(hooks.model, type) and issubclass(hooks.model, ReferenceLM)):
            raise ValueError(f"plugin {name}: model must be a subclass of ReferenceLM")
    return hooks


def stages(config):
    """Contiguous (stage, optimizer steps) of a plugin method, in training order."""
    from nexttransformer.protocol.spec import Protocol
    steps = Protocol().steps
    hook = load(config.method).stages
    out = [tuple(stage) for stage in hook(config)] if hook else [("joint", steps)]
    names = [stage[0] for stage in out]
    if (len(set(names)) != len(names) or any(len(stage) != 2 or not isinstance(stage[0], str) or
                                            not isinstance(stage[1], int) or stage[1] <= 0 for stage in out)
            or sum(stage[1] for stage in out) != steps):
        raise ValueError(f"{config.method}: stages must be distinct names with positive steps summing to {steps}")
    return out


def stage_at(config, step):
    """(stage, first update of that stage) that trains update `step`, counted from 0."""
    start = 0
    for name, count in stages(config):
        if step < start + count: return name, start
        start += count
    raise ValueError("update outside the training schedule")


def stage_start(config, name):
    """First update of a named plugin stage."""
    start = 0
    for stage, count in stages(config):
        if stage == name: return start
        start += count
    raise ValueError(f"{config.method} has no stage {name!r}")


def check_shell(model):
    """Every block keeps the linear layers that the shared shell initializes."""
    from torch import nn
    for index, block in enumerate(model.transformer.h):
        for part, names in (("attn", ("c_q", "c_k", "c_v", "c_proj")), ("mlp", ("c_fc", "c_proj"))):
            module = getattr(block, part, None)
            if module is None or not all(isinstance(getattr(module, n, None), nn.Linear) for n in names):
                raise ValueError(f"{model.config.method}: block {index} needs {part}." + "/".join(names) +
                                 " linear layers, which the shared shell initializes")


def reset(model):
    """Reset plugin modules that define reset_parameters; `to_empty` leaves their tensors unset."""
    from torch import nn
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.LayerNorm)) and callable(getattr(module, "reset_parameters", None)):
            module.reset_parameters()
