"""Gym attempts ported to method plugins compute the attempts' function (G1, arena v1 Stage 3).

Each port must have the attempt's parameters (names and full-size shapes), and with the attempt's weights loaded it
must give the same loss and the same gradient for every parameter, bit for bit, in the gym and in the upstream
optimizer groups. The attempts are the Stage 2 submissions kept in the evidence.
"""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nextt_gym.contract import load_submission
from nextt_gym.roles import parameter_groups
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups as upstream_groups

ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / "evidence/checks/arena-v1-stage2-20261102/submissions"
# plugin -> (Stage 2 attempt, AttnRes groups of the attempt's debug model of three layers)
PORTS = {"attnres_canon_abcd": ("attnres_canon_abcd", 2), "attnres_full_canon_abcd": ("attnres_full_canon_abcd", 3)}


def attempt(name):
    return load_submission(SUBMISSIONS / name)


def plugin_debug(method, groups):
    """The attempt's debug shape (three layers) under the plugin; the full variant has one group per layer."""
    c = replace(debug_config(method), layers=3)
    return replace(c, options={"groups": groups}) if c.options else c


@pytest.mark.parametrize("method", PORTS)
def test_port_has_the_attempts_parameters_at_full_size(method):
    sub = attempt(PORTS[method][0])
    with torch.device("meta"):
        gym = sub.instantiate("meta")
    port = build_model(candidate_config(method), "meta")
    shapes = lambda m: {n: (tuple(p.shape), p.requires_grad) for n, p in m.named_parameters()}
    assert shapes(port) == shapes(gym)
    assert (port.config.layers, port.config.mlp_width, port.config.width) == (23, 6144, 1536)


@pytest.mark.parametrize("method", PORTS)
def test_port_starts_from_the_attempts_initial_weights(method):
    """The same seed gives the same initial weights: a formal run starts where the attempt's Stage 2 run started."""
    name, groups = PORTS[method]
    sub = attempt(name)
    gym = sub.instantiate("cpu", overrides=sub.debug_overrides, seed=42)
    port = build_model(plugin_debug(method, groups), "cpu", seed=42)
    a, b = gym.state_dict(), port.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a), [n for n in a if not torch.equal(a[n], b[n])]


@pytest.mark.parametrize("method", PORTS)
def test_port_computes_the_attempt_bit_for_bit(method):
    name, groups = PORTS[method]
    sub = attempt(name)
    gym = sub.instantiate("cpu", overrides=sub.debug_overrides, seed=42)
    # Zero-initialized projections and Canon weights would hide most paths: perturb every weight first.
    torch.manual_seed(1)
    with torch.no_grad():
        for p in gym.parameters(): p.add_(0.05 * torch.randn_like(p))
    port = build_model(plugin_debug(method, groups), "cpu", seed=7)       # other weights until the load below
    port.load_state_dict(gym.state_dict(), strict=True)
    torch.manual_seed(0)
    x = torch.randint(0, port.config.vocab_size, (3, port.config.sequence_len))
    y = torch.randint(0, port.config.vocab_size, (3, port.config.sequence_len))
    for model in (gym, port):
        model.train(); model.zero_grad(set_to_none=True)
    losses = [model(x, y) for model in (gym, port)]
    for loss in losses: loss.backward()
    assert torch.equal(losses[0], losses[1])
    grads = [{n: p.grad for n, p in model.named_parameters() if p.requires_grad} for model in (gym, port)]
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0]), \
        [n for n in grads[0] if not torch.equal(grads[0][n], grads[1][n])]
    assert any(g.abs().sum() > 0 for n, g in grads[1].items() if "canon" in n)


@pytest.mark.parametrize("method", PORTS)
def test_port_optimizer_groups_equal_the_gym_roles(method):
    """The formal trainer's groups give every parameter the hyperparameters the gym gave it in Stage 2."""
    name, groups = PORTS[method]
    gym = attempt(name).instantiate("cpu", overrides=attempt(name).debug_overrides, seed=42)
    port = build_model(plugin_debug(method, groups), "cpu")
    def by_name(groups, names):
        out = {}
        for g in groups:
            for p in g["params"]:
                out[names[id(p)]] = (g["kind"], g["lr"], tuple(g.get("betas", ())), g["weight_decay"], g.get("momentum"))
        return out
    gym_groups = by_name(parameter_groups(gym, gym.config.width), {id(p): n for n, p in gym.named_parameters()})
    port_groups = by_name(upstream_groups(port), {id(p): n for n, p in port.named_parameters()})
    assert port_groups == gym_groups
