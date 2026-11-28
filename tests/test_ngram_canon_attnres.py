"""The `ngram_canon_attnres` plugin: a hashed bigram memory and Canon at more sites on the AttnRes-Canon trunk.

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here: with
every option at its neutral value the plugin is exactly `attnres_full_canon_abcd`; the registered additions start as
the identity; the memory reads only the current and earlier tokens; and the registered size is the frozen entry's.
"""
from dataclasses import replace

import torch

from nexttransformer.methods.ngram_canon_attnres import Canon, NGramMemory
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups

NEUTRAL = dict(canon_size=4, canon_readout=False, canon_ve=False, canon_write=False, canon_write_size=4,
               canon_embed=False, ngram_rows=0)


def neutral(config):
    return replace(config, options=dict(dict(config.options), **NEUTRAL))


def tokens(config):
    return torch.randint(0, config.vocab_size, (2, config.sequence_len))


def test_neutral_options_are_attnres_canon():
    torch.manual_seed(0)
    reference = build_model(replace(debug_config("attnres_full_canon_abcd"), layers=3), "cpu", seed=3)
    for name, parameter in reference.named_parameters():
        if "canon" in name: torch.nn.init.normal_(parameter, std=0.1)  # exercise every Canon path
    plugin = build_model(neutral(replace(debug_config("ngram_canon_attnres"), layers=3)), "cpu", seed=11)
    plugin.load_state_dict(reference.state_dict())  # the same parameter names and shapes
    x = tokens(plugin.config)
    for model in (reference, plugin): model.eval()
    assert torch.equal(reference(x), plugin(x))


def test_registered_additions_start_as_the_identity():
    torch.manual_seed(0)
    config = debug_config("ngram_canon_attnres")
    plugin = build_model(config, "cpu", seed=5)
    assert plugin.ngram is not None and len(plugin.ngram_keys) == 2
    assert all(torch.count_nonzero(m.weight) == 0 for m in plugin.modules() if isinstance(m, Canon))
    assert all(torch.count_nonzero(plugin.value_embeds[key].weight) == 0 for key in plugin.ngram_keys)
    trunk = build_model(neutral(config), "cpu", seed=7)
    shapes = {name: value.shape for name, value in trunk.state_dict().items()}
    shared = {name: value for name, value in plugin.state_dict().items() if shapes.get(name) == value.shape}
    missing, unexpected = trunk.load_state_dict(shared, strict=False)
    assert not unexpected and all("canon" in name for name in missing)  # only the kernels differ in size; all zero
    x = tokens(config)
    for model in (plugin, trunk): model.eval()
    assert torch.equal(plugin(x), trunk(x))


def test_memory_reads_only_the_current_and_previous_token():
    torch.manual_seed(0)
    config = debug_config("ngram_canon_attnres")
    memory = NGramMemory(config)
    tables = [torch.randn(memory.rows, memory.dim) for _ in range(memory.count)]
    x = tokens(config)
    y = x.clone(); y[:, 3] = (y[:, 3] + 1) % config.vocab_size
    a, b = memory(x, tables)[0], memory(y, tables)[0]
    assert a.shape == (2, config.sequence_len, config.width)
    changed = (a != b).any(-1)
    assert changed[:, 3:5].all() and not changed[:, :3].any() and not changed[:, 5:].any()
    # The first position pairs its token with the padding id, never with a token of another row.
    z = x.clone(); z[:, 1:] = (z[:, 1:] + 1) % config.vocab_size
    assert torch.equal(a[:, 0], memory(z, tables)[0][:, 0])


def test_registered_configuration_is_the_frozen_entry():
    config = candidate_config("ngram_canon_attnres")
    model = build_model(config, "meta")
    assert (config.layers, config.mlp_width, config.window_pattern) == (23, 6144, "L")
    assert sum(p.numel() for p in model.parameters()) == 1_850_401_032
    tables = [model.value_embeds[key].weight for key in model.ngram_keys]
    assert [tuple(table.shape) for table in tables] == [(320_000, 768)] * 2
    # The tables train with the value embeddings under AdamW, never under Muon.
    kinds = {group["kind"] for group in parameter_groups(model) if any(p is tables[0] for p in group["params"])}
    assert kinds == {"adamw"}
