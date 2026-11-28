"""The `hydra` plugin: a deeper, narrower AttnRes-Canon trunk that reads a hashed n-gram memory as a depth source.

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here:
without the memory the plugin is exactly `attnres_full_canon_abcd`; the memory reads only the current and earlier
tokens and is one more depth source; and the registered size is the frozen entry's.
"""
from dataclasses import replace

import torch

from nexttransformer.methods.hydra import CanonAttention
from nexttransformer.methods.hydra.layers import CanonMLP
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def tokens(config):
    return torch.randint(0, config.vocab_size, (2, config.sequence_len))


def prime(n):
    return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))


def test_without_memory_is_attnres_canon():
    torch.manual_seed(0)
    reference = build_model(replace(debug_config("attnres_full_canon_abcd"), layers=3), "cpu", seed=3)
    for name, parameter in reference.named_parameters():
        if "canon" in name: torch.nn.init.normal_(parameter, std=0.1)  # exercise every Canon path
    config = replace(debug_config("hydra"), layers=3)
    plugin = build_model(replace(config, options=dict(dict(config.options), ngram_orders=[])), "cpu", seed=11)
    assert plugin.memory is None
    plugin.load_state_dict(reference.state_dict())  # the same parameter names and shapes
    x = tokens(config)
    for model in (reference, plugin): model.eval()
    assert torch.equal(reference(x), plugin(x))


def test_memory_reads_only_the_current_and_earlier_tokens():
    torch.manual_seed(0)
    config = debug_config("hydra")
    model = build_model(config, "cpu", seed=1)
    memory = model.memory
    assert torch.count_nonzero(memory.ngram_proj.weight) == 0  # the memory source starts at zero
    torch.nn.init.normal_(memory.ngram_proj.weight)
    read = lambda idx: memory(idx, model.value_embeds, torch.float32)
    x = tokens(config)
    y = x.clone(); y[:, 3] = (y[:, 3] + 1) % config.vocab_size
    a, b = read(x), read(y)
    assert a.shape == (2, config.sequence_len, config.width)
    changed = (a != b).any(-1)
    # Bigrams and trigrams: a token reaches its own position and the next two.
    assert changed[:, 3:6].all() and not changed[:, :3].any() and not changed[:, 6:].any()


def test_memory_is_one_more_depth_source():
    config = debug_config("hydra")
    with_memory = build_model(config, "cpu", seed=1)
    without = build_model(replace(config, options=dict(dict(config.options), ngram_orders=[])), "cpu", seed=1)
    x = tokens(config)
    reads = []
    for model in (with_memory, without):
        model.eval(); model(x)
        reads.append([row["sources"] for row in model.last_trace if row["kind"] == "depth_read"])
    # One read per sublayer and one for the readout; the memory adds exactly one source to each.
    assert len(reads[1]) == 2 * config.layers + 1 and reads[0] == [sources + 1 for sources in reads[1]]
    assert set(with_memory.value_embeds) - set(without.value_embeds) == set(with_memory.memory.keys())


def test_registered_configuration_is_the_frozen_entry():
    config = candidate_config("hydra")
    model = build_model(config, "meta")
    assert (config.layers, config.mlp_width, config.window_pattern) == (27, 4608, "L")
    assert sum(p.numel() for p in model.parameters()) == 1_780_828_720
    memory = model.memory
    assert memory.keys() == ["ngram2_0", "ngram2_1", "ngram3_0", "ngram3_1"]
    sizes = [model.value_embeds[key].weight.shape for key in memory.keys()]
    assert all(dim == 256 and prime(rows) for rows, dim in sizes)
    assert len({rows for rows, _ in sizes}) == 4 and max(rows for rows, _ in sizes) <= 327_680
    # The mixture-of-experts option is off: every block keeps the Canon attention and the Canon MLP.
    assert all(isinstance(block.attn, CanonAttention) and type(block.mlp) is CanonMLP for block in model.transformer.h)
    # Tables train as embeddings under AdamW; the projection is a Muon matrix.
    groups = parameter_groups(model)
    kind = lambda p: {group["kind"] for group in groups if any(q is p for q in group["params"])}
    assert kind(model.value_embeds["ngram2_0"].weight) == {"adamw"}
    assert kind(memory.ngram_proj.weight) == {"muon"}
