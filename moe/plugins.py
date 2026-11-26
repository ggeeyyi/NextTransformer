"""MoE methods as plugins: one directory per method under moe/methods/<name>/, added without editing shared files.

Layout and rules mirror nexttransformer/models/plugins.py. The declaration `method.json` is read without
importing code:

    {"name": "<name>", "title": "...", "paper": {"title": "...", "url": "..."},
     "options": {"<option>": <default>}, "debug_options": {"<option>": <value>},
     "calibrate": "expert_width"}

`calibrate` names the one shape knob calibration may move to bring the method back to the track's FLOPs per
token: expert_width (default), shared_expert_width, layers or none.

Hooks, imported when the method is first built:

    router(config, layer) -> module     called as (x[N, d]) -> (index[N, k], gates[N, k], aux or None)
    ffn(config, layer) -> module        replaces the whole MoE FFN, called as (x[B, T, d]); may set `last_aux`
    mixer(config, layer) -> module      replaces attention, called as (x, cos_sin, masks); softmax attention must
                                        call moe.model.attention(q, k, v, masks.get(window)) so the ledger
                                        can charge its edges
    block(block, config, layer)         wraps or replaces a block, called as (x, cos_sin, masks)
    model                               a subclass of moe.model.MoELM, for a different forward flow
    init(model)                         initializes the method's own parameters after the reference init

Optimizer groups follow Marin's rules by parameter name (moe/optim.py): matrices get MuonH unless they are the
embedding, a router (`.router.` in the name), a head-wise gate or the head.
    check(config)                       raises ValueError for invalid options

A module may expose `last_aux` (a scalar auxiliary loss added in training). Routers that subclass
moe.model.Router keep the balancing bias; a router that does not is responsible for its own balancing.
"""
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent / "methods"
HOOKS = ("router", "ffn", "mixer", "block", "model", "init", "check")
KNOBS = ("expert_width", "shared_expert_width", "layers", "none")
NAME = re.compile(r"[a-z][a-z0-9_]{1,40}")
RESERVED = {"baseline"}


def roots():
    extra = [Path(p).resolve() for p in os.environ.get("NEXTT_MOE_METHOD_PATHS", "").split(os.pathsep) if p]
    return [ROOT, *extra]


@lru_cache(maxsize=None)
def registry():
    """name -> declaration of every MoE plugin."""
    found = {}
    for root in roots():
        for path in sorted(root.glob("*/method.json")):
            spec = json.loads(path.read_text())
            name, options = spec.get("name"), spec.get("options", {})
            problems = [message for ok, message in (
                (isinstance(name, str) and NAME.fullmatch(name) is not None, "name: lowercase letters, digits and _"),
                (name == path.parent.name, "name must equal its directory"),
                (name not in RESERVED and name not in found, "name is already registered"),
                (isinstance(options, dict) and all(NAME.fullmatch(k) for k in options), "options map lowercase names"),
                (set(spec.get("debug_options", {})) <= set(options), "debug_options may only set declared options"),
                (spec.get("calibrate", "expert_width") in KNOBS, f"calibrate is one of {', '.join(KNOBS)}"),
                ((path.parent / "__init__.py").is_file(), "the method directory needs an __init__.py"),
            ) if not ok]
            if problems:
                raise ValueError(f"MoE plugin {path.parent}: " + "; ".join(problems))
            found[name] = dict(spec, path=str(path.parent))
    return found


def methods():
    return ("baseline", *registry())


def declaration(name):
    if name == "baseline":
        return dict(name="baseline", title="Marin Grug MoE reference", options={}, calibrate="none")
    if name not in registry():
        raise ValueError(f"unknown MoE method {name!r}; registered: {', '.join(methods())}")
    return registry()[name]


@lru_cache(maxsize=None)
def module(name):
    if name == "baseline":
        return None
    spec = declaration(name)
    package = f"moe_plugin_{name}"
    if package in sys.modules:
        return sys.modules[package]
    init = Path(spec["path"]) / "__init__.py"
    loader = importlib.util.spec_from_file_location(package, init, submodule_search_locations=[str(init.parent)])
    mod = importlib.util.module_from_spec(loader)
    sys.modules[package] = mod
    loader.loader.exec_module(mod)
    return mod


def hooks_for(name):
    mod = module(name)
    return SimpleNamespace(**{h: getattr(mod, h, None) if mod else None for h in HOOKS})


def model_class(name):
    return hooks_for(name).model


def source_files(name):
    """Files whose code a method's cost record covers: its package and the shared MoE core."""
    core = [Path(__file__).resolve().parent / f for f in ("model.py", "optim.py", "plugins.py", "compute.py", "spec.py")]
    own = sorted(Path(declaration(name)["path"]).rglob("*.py")) if name != "baseline" else []
    return [*core, *own]


def config_for(name, track_name, debug=False):
    """The method's starting configuration on a track (before calibration)."""
    from moe.model import MoEConfig
    spec = declaration(name)
    config = MoEConfig.baseline(track_name, method=name, options=tuple(spec.get("options", {}).items()))
    if debug:
        config = config.debug()
        from dataclasses import replace
        config = replace(config, options=tuple({**spec.get("options", {}), **spec.get("debug_options", {})}.items()))
    check = hooks_for(name).check
    if check:
        check(config)
    return config
