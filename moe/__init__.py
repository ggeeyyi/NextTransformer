"""Compute-controlled MoE comparisons on Marin's May backbones.

The original tracks and shared controls are frozen in moe.spec. Additional budgets
are registered by name and carry their own track fingerprints.
"""
from moe.budgets import tpp240_tracks
from moe.spec import TRACKS

TRACKS.update(tpp240_tracks(TRACKS))
