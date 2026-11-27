"""Load a submitted architecture and expose it through one uniform interface.

A submission is a directory with `model.py` and `config.json` (see spec.md).
`model.py` must define `build(config) -> torch.nn.Module` and may define the
optional module-level hooks below. The loader never executes the submission at
import time beyond defining functions and classes (V0 checks this statically).
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

from nexttransformer.protocol.spec import Protocol

REQUIRED_FILES = ("model.py", "config.json")
OPTIONAL_FILES = ("kernels.py", "NOTES.md")
DEFAULT_STAGE = "joint"


def files_sha256(directory):
    """Content hash over the submission files, in a fixed order."""
    digest = hashlib.sha256()
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        path = Path(directory) / name
        if path.exists():
            digest.update(name.encode()); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class AuditSettings:
    mode: str = "direct"                  # "direct" | "affine" (operator-wise extrapolation for scans)
    probe_lengths: tuple = ()             # affine mode: three lengths in arithmetic progression
    pair_block: int | None = None         # chunk length C for causal-pair charging of C x C tiles
    cross_check_lengths: tuple = ()       # direct mode with a chunk backend: affine cross-check lengths
    chunk_fields: tuple | None = None     # model fields that must differ from C (None: every int field)
    chunk_values: tuple = ()              # extra derived dimensions that must differ from C


class Submission:
    """A loaded architecture: config, factory and the optional hooks."""

    def __init__(self, directory, module, config, upstream=False):
        self.directory = Path(directory) if directory else None
        self.module, self.config, self.upstream = module, config, upstream
        p = Protocol()
        self.model_config = config["model"]
        self.model_dim = int(config["model_dim"])
        self.vocab_size = int(config.get("vocab_size", p.vocab_size))
        self.sequence_len = int(config.get("sequence_len", p.sequence_len))
        self.tunable = config.get("tunable", {})
        # Small-model settings for the V2 correctness checks: model overrides and the
        # vocabulary/context the downscaled model accepts.
        debug = config.get("debug", {})
        self.debug_overrides = debug.get("model", {})
        self.debug_vocab_size = int(debug.get("vocab_size", self.vocab_size))
        self.debug_sequence_len = int(debug.get("sequence_len", 32))
        self.aux_loss = bool(config.get("aux_loss", False))
        audit = config.get("audit", {})
        self.audit = AuditSettings(audit.get("mode", "direct"), tuple(audit.get("probe_lengths", ())),
                                   audit.get("pair_block"), tuple(audit.get("cross_check_lengths", ())),
                                   tuple(audit["chunk_fields"]) if "chunk_fields" in audit else None,
                                   tuple(audit.get("chunk_values", ())))

    # ---- construction -------------------------------------------------------
    def instantiate(self, device="cpu", *, overrides=None, seed=None):
        cfg = dict(self.model_config, **(overrides or {}))
        torch.manual_seed(Protocol().seed if seed is None else seed)
        with torch.device(device):
            model = self.module.build(cfg)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("build(config) must return a torch.nn.Module")
        return model

    def with_model_config(self, **changes):
        """A copy with changed model fields (used by calibration)."""
        config = json.loads(json.dumps(self.config))
        config["model"].update(changes)
        if "model_dim_field" in config and config["model_dim_field"] in changes:
            config["model_dim"] = changes[config["model_dim_field"]]
        return Submission(self.directory, self.module, config, self.upstream)

    def audit_dimensions_hit_chunk(self):
        """A dimension equal to the chunk length would make C x C tile classification ambiguous."""
        c = self.audit.pair_block
        fields = self.audit.chunk_fields
        if fields is None:
            fields = [k for k, v in self.model_config.items() if isinstance(v, int) and not isinstance(v, bool)
                      and k != "sequence_len"]
        values = [self.model_config[k] for k in fields if k in self.model_config] + list(self.audit.chunk_values)
        return c in values or c == self.model_dim

    # ---- optional hooks -----------------------------------------------------
    def stage_for_step(self, step, total_steps):
        hook = getattr(self.module, "stage_for_step", None)
        return hook(self.model_config, step, total_steps) if hook else DEFAULT_STAGE

    def stage_schedule(self, total_steps, horizon=None):
        """{stage: optimizer steps} for the first `total_steps` updates of a `horizon`-update schedule."""
        counts = {}
        for step in range(total_steps):
            stage = self.stage_for_step(step, horizon or total_steps)
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    def set_stage(self, model, stage, *, audit, step, total_steps):
        """Prepare `model` for `stage` (before optimizer grouping and at each stage change).

        `audit=True` is the shape-only cost run, `step` a representative step of the
        stage; in training `step` is the current optimizer step. Returns the
        attention-rule stage label for the audit (defaults to the stage name).
        """
        hook = getattr(self.module, "set_stage", None)
        label = hook(model, stage, audit=audit, step=step, total_steps=total_steps) if hook else None
        return label or stage

    @staticmethod
    def set_step(model, step):
        if hasattr(model, "set_step"): model.set_step(step)

    @staticmethod
    def set_row_offset(model, row):
        if hasattr(model, "set_row_offset"): model.set_row_offset(row)


def load_module(path, name=None):
    path = Path(path)
    name = name or "nextt_gym_submission_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_submission(directory):
    directory = Path(directory)
    missing = [n for n in REQUIRED_FILES if not (directory / n).exists()]
    if missing: raise FileNotFoundError(f"submission is missing {missing}")
    config = json.loads((directory / "config.json").read_text())
    for key in ("model", "model_dim"):
        if key not in config: raise ValueError(f"config.json requires {key!r}")
    module = load_module(directory / "model.py")
    if not callable(getattr(module, "build", None)): raise ValueError("model.py must define build(config)")
    return Submission(directory, module, config, upstream=bool(config.get("upstream_method")))


def submission_from_hooks(config, **hooks):
    """An in-memory submission (tests, upstream adapters)."""
    module = types.SimpleNamespace(**hooks)
    return Submission(None, module, config, upstream=bool(config.get("upstream_method")))
