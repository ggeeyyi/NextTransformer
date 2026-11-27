"""The fixed research task: what is controlled, how runs are scaled, what the agent may see.

Everything here is shared scientific configuration. Machine paths and cluster
details belong in the ignored `backends.local.toml` and the upstream site JSON.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json

from nexttransformer.protocol.spec import Protocol

TASK_ID = "nextt-iso-flop-arch-v1"
SPEC_VERSION = "nextt-gym-model-v1"

# Upstream SSSL cost anchor (configs/others/dense.json); a matched model spends
# this many theoretical training FLOPs per input token, within the protocol tolerance.
ANCHOR_FLOPS_PER_TOKEN = 4672398325.44785
FLOPS_TOLERANCE = Protocol().preflight_tolerance          # 1%
CUMULATIVE_TOLERANCE = Protocol().cumulative_tolerance    # 2%

# The arena's dev split (arena rules, nexttransformer.arena.DEV_SHARD): outside the prefix of shards
# that any run reaches (training reads shards in order; 49.152B tokens is far below shard 6541).
DEV_SHARD = "shard_06541.parquet"
# The SHA-256 of its content, so that a run cannot select on a different file under the same name. The pinned
# data inventory records only names and sizes.
DEV_SHARD_SHA256 = "51da1a9bc052dfd5a77988d6ca4d1cbe62b89eedb247b56850fc1b9b45e56c73"


@dataclass(frozen=True)
class Scale:
    name: str
    input_tokens: int
    val_every: int            # optimizer steps between scheduled validations
    val_tokens: int           # tokens per scheduled validation
    final_val_tokens: int     # tokens per final validation (each evaluated split)
    official_evaluated: bool  # official val BPB (upstream shard) is evaluated; sealed from the agent until revealed
    core: bool                # run CORE-22 at the end; sealed like the official split
    prefix_of: str | None = None   # a prefix of another scale's run: its LR and stage schedule, stopped early
    origin: str = "gym"            # "upstream protocol" or "gym"

    @property
    def steps(self):
        p = Protocol()
        if self.input_tokens % p.global_batch_tokens:
            raise ValueError("scale must contain complete optimizer updates")
        return self.input_tokens // p.global_batch_tokens

    @property
    def schedule_steps(self):
        """Steps of the LR and stage schedule: the parent run's for a prefix scale, else this scale's."""
        return SCALES[self.prefix_of].steps if self.prefix_of else self.steps

    @property
    def theory_flops(self):
        return ANCHOR_FLOPS_PER_TOKEN * self.input_tokens


_P = Protocol()
SCALES = {
    # S is a research run with the shape of a protocol pilot (the first 512 updates of the formal
    # schedule and data prefix), but it is not a pilot and has no pilot quota. C has S's tokens but its
    # own 512-update schedule: the formal schedule code over 512 steps (warmup 40, the last 65% decaying
    # to 5%, staged methods scaled by upstream_model._scaled), so it ends after a decay. The Muon momentum
    # ramp keeps its fixed 400 updates there. M is the 2,000-update prefix of the formal schedule (about 4% of
    # its tokens), evaluated every 256 updates so that update 512 is also S's measurement. L follows the formal
    # schedule. Gym results never become upstream formal results by themselves (arena rules).
    "S": Scale("S", 512 * _P.global_batch_tokens, 128, 8 * _P.global_batch_tokens,
               8 * _P.global_batch_tokens, official_evaluated=False, core=False, prefix_of="L", origin="gym"),
    "C": Scale("C", 512 * _P.global_batch_tokens, 128, 8 * _P.global_batch_tokens,
               8 * _P.global_batch_tokens, official_evaluated=False, core=False, origin="gym"),
    "M": Scale("M", 2000 * _P.global_batch_tokens, 256, 8 * _P.global_batch_tokens,
               8 * _P.global_batch_tokens, official_evaluated=False, core=False, prefix_of="L", origin="gym"),
    "L": Scale("L", _P.total_tokens, _P.val_every, _P.val_tokens,
               _P.final_bpb_tokens, official_evaluated=True, core=True, origin="upstream protocol"),
}


@dataclass(frozen=True)
class Policy:
    require_s_before_l: bool = True         # an L run needs a succeeded S run of the same attempt
    seed_offsets_at_l: bool = False         # L runs use the protocol seed (no seed shopping)
    max_concurrent_runs: int = 12          # 96 H20 for arena v1 Stage 2 (operator, 2026-11-01)
    max_run_share_of_cap: float = 0.25      # one run may reserve at most this share of the cap
    estimate_cost_per_hour: int = 30
    submission_max_bytes: int = 256 * 1024
    duplicate_similarity: float = 0.97
    reserve_margin: float = 0.02            # extra reservation above the planned cost
    default_cap_flops: float = 10 * ANCHOR_FLOPS_PER_TOKEN * _P.total_tokens   # ten L runs


@dataclass(frozen=True)
class Task:
    task_id: str = TASK_ID
    objective: str = "minimize validation BPB at matched training FLOPs per input token"
    spec_version: str = SPEC_VERSION
    anchor_flops_per_token: float = ANCHOR_FLOPS_PER_TOKEN
    flops_tolerance: float = FLOPS_TOLERANCE
    dev_shard: str = DEV_SHARD
    policy: Policy = field(default_factory=Policy)

    def describe(self):
        p = Protocol()
        return dict(
            task_id=self.task_id, objective=self.objective, spec_version=self.spec_version,
            controlled=dict(tokenizer_vocab=p.vocab_size, sequence_len=p.sequence_len,
                            global_batch_tokens=p.global_batch_tokens,
                            data=dict(dataset="ClimbMix", revision=p.data_revision),
                            optimizer="muon_adamw, fixed hyperparameters per parameter role",
                            schedule=dict(warmup_steps=p.warmup_steps, warmdown_ratio=p.warmdown_ratio,
                                          final_lr_fraction=p.final_lr_fraction),
                            seed=p.seed, dtype=p.dtype),
            flops_anchor=dict(flops_per_token=self.anchor_flops_per_token, tolerance=self.flops_tolerance,
                              source="upstream SSSL dense, reference-aten-math-v2"),
            scales={k: dict(input_tokens=s.input_tokens, steps=s.steps, schedule_steps=s.schedule_steps,
                            prefix_of=s.prefix_of, origin=s.origin, theory_flops=s.theory_flops,
                            val_every=s.val_every, official_evaluated=s.official_evaluated, core=s.core)
                    for k, s in SCALES.items()},
            visible_splits={k: ["dev"] for k in SCALES},
            sealed={k: ["official"] + (["core"] if s.core else []) for k, s in SCALES.items() if s.official_evaluated},
            sealing=("official validation BPB and CORE are withheld from the agent until the operator reveals the "
                     "attempt, for gym runs and imported upstream runs alike; upstream records are served without "
                     "the lines that report results"),
            upstream_protocol_hash=p.fingerprint)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.describe() | {"policy": asdict(self.policy)},
                                         sort_keys=True).encode()).hexdigest()


TASK = Task()
