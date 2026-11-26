"""Additional token budgets for the frozen MoE backbones in moe.spec."""
from dataclasses import replace

from moe.spec import PROTOCOL


# Per-token active baseline parameters, excluding the input embedding and output head.
# The same denominator fixes the token budget of every method at that width.
ACTIVE_TRUNK_PARAMS = {"d768": 58_639_872, "d1024": 141_934_592}


def tpp240_tracks(reference_tracks):
    tracks = {}
    for name, parameters in ACTIVE_TRUNK_PARAMS.items():
        parent = reference_tracks[name]
        batch = parent.tokens_per_update
        updates = (240 * parameters + batch // 2) // batch
        name_240 = f"{name}-240tpp"
        tracks[name_240] = replace(
            parent, name=name_240, marin_run="derived_from_" + parent.marin_run,
            updates=updates, core_steps=tuple(round(step * updates / parent.updates) for step in parent.core_steps),
            marin_budget_flops=None, marin_tokens_per_second=None,
        )
    return tracks


# Initial d768 runs serialized their track list and its release name into the shared
# protocol hash. Their model and track fingerprints match the public d768-240tpp recipe.
# Keep the original identity when auditing or exporting those runs.
D768_LAUNCH_PROTOCOL = replace(PROTOCOL, version="nextt-moe-240tpp-v1", tracks=(*PROTOCOL.tracks, "d768-240tpp"))


def run_protocol(track_name, protocol_hash):
    if track_name == "d768-240tpp" and protocol_hash == D768_LAUNCH_PROTOCOL.fingerprint:
        return D768_LAUNCH_PROTOCOL
    return PROTOCOL
