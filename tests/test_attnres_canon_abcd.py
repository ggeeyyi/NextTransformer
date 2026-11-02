"""The Canon AttnRes plugins (attnres_canon_abcd, attnres_full_canon_abcd).

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here:
with Canon weights at zero (their initialization) a plugin is exactly `attnres` with the same weights and groups, its
group layout follows `attnres`, and the options are checked.
"""
from dataclasses import replace

import pytest
import torch

from nexttransformer.methods.attnres_canon_abcd import Canon, group_starts
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model


@pytest.mark.parametrize("method, groups", [("attnres_canon_abcd", 2), ("attnres_full_canon_abcd", 3)])
def test_zero_canon_is_attnres(method, groups):
    """Three layers: two groups (uneven, as attnres's debug model) or one per layer for the full variant."""
    reference = build_model(replace(debug_config("attnres"), layers=3, attnres_groups=groups), "cpu", seed=3)
    plugin_config = replace(debug_config(method), layers=3)
    if plugin_config.options: plugin_config = replace(plugin_config, options={"groups": groups})
    plugin = build_model(plugin_config, "cpu", seed=11)
    missing, unexpected = plugin.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and missing and all("canon" in name for name in missing)
    assert all(torch.count_nonzero(m.weight) == 0 for m in plugin.modules() if isinstance(m, Canon))
    assert plugin.group_starts == reference.config.attnres_group_starts
    x = torch.randint(0, plugin.config.vocab_size, (2, plugin.config.sequence_len))
    for model in (reference, plugin): model.eval()
    assert torch.equal(reference(x), plugin(x))


def test_group_layout_matches_attnres():
    for layers in (3, 7, 23):
        for groups in range(1, layers + 1):
            c = replace(candidate_config("attnres"), layers=layers, attnres_groups=groups)
            assert group_starts(layers, groups) == c.attnres_group_starts
    assert len(build_model(candidate_config("attnres_full_canon_abcd"), "meta").group_starts) == 23
    assert len(build_model(candidate_config("attnres_canon_abcd"), "meta").group_starts) == 8


@pytest.mark.parametrize("groups", [0, 24, 2.5, True])
def test_groups_option_is_checked(groups):
    with pytest.raises(ValueError, match="groups"):
        replace(candidate_config("attnres_canon_abcd"), options={"groups": groups})
