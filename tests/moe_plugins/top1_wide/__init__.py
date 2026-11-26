"""Test plugin: changes top-k through `check`-validated options and relies on calibration of expert_width."""
from dataclasses import replace

from moe.model import MoE


def ffn(config, layer):
    # One routed expert per token: calibration widens the experts to restore the track's FLOPs per token.
    return MoE(replace(config, top_k=1), layer)
