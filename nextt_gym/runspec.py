"""RunSpec: the server-generated description of one training run (the agent never writes it).

FLOPs values are exact decimal strings: one L run is ~2.3e20, beyond 2^53.
"""
import json
from pathlib import Path

from nexttransformer.protocol.spec import Protocol

from .contract import files_sha256
from .task import DEV_SHARD, SCALES, TASK

WORLD_SIZES = (8, 16, 32)
# CORE-22 input tokens are data dependent; this fixed estimate only sizes the reservation.
CORE_TOKENS_ESTIMATE = 200_000_000


def validation_steps(scale):
    """Scheduled validation steps (step 0, every val_every, excluding the final step)."""
    return [s for s in range(0, scale.steps, scale.val_every)]


def planned_cost(sub, audit_report, scale_name):
    """Planned theory FLOPs of a run: training by stage schedule plus all scheduled evaluations."""
    scale = SCALES[scale_name]
    costs = {s["name"]: int(s["total"]) for s in audit_report["stages"]}
    schedule = sub.stage_schedule(scale.steps, scale.schedule_steps)
    missing = set(schedule) - set(costs)
    if missing: raise ValueError(f"audit report lacks stages {sorted(missing)}")
    train = sum(n * costs[s] for s, n in schedule.items())
    splits = 2 if scale.official_evaluated else 1
    eval_tokens = splits * (len(validation_steps(scale)) * scale.val_tokens + scale.final_val_tokens)
    if scale.core: eval_tokens += CORE_TOKENS_ESTIMATE
    forward = audit_report["forward_flops_per_token"]
    return dict(train=train, eval=int(forward * eval_tokens), eval_tokens=eval_tokens)


def make_run_spec(sub, audit_report, scale_name, run_id, world_size, reserved_flops, seed_offset=0,
                  *, synthetic=False, micro_batch=None):
    scale = SCALES[scale_name]
    spec = dict(
        run_id=run_id, scale=scale_name, steps=scale.steps, schedule_steps=scale.schedule_steps,
        input_tokens=scale.input_tokens,
        world_size=int(world_size), seed=Protocol().seed + int(seed_offset), seed_offset=int(seed_offset),
        micro_batch=micro_batch,
        stage_costs={s["name"]: str(int(s["total"])) for s in audit_report["stages"]},
        forward_flops_per_token=float(audit_report["forward_flops_per_token"]),
        val_every=scale.val_every, val_tokens=scale.val_tokens, final_val_tokens=scale.final_val_tokens,
        official_evaluated=scale.official_evaluated, core=scale.core, dev_shard=DEV_SHARD,
        reserved_flops=str(int(reserved_flops)), stop_at_flops=str(int(reserved_flops)),
        synthetic=bool(synthetic), task_fingerprint=TASK.fingerprint,
        submission_sha256=files_sha256(sub.directory) if sub.directory else None)
    validate_run_spec(spec, sub)
    return spec


def validate_run_spec(spec, sub=None):
    required = ("run_id", "scale", "steps", "world_size", "seed", "stage_costs", "forward_flops_per_token",
                "val_every", "val_tokens", "final_val_tokens", "official_evaluated", "core", "dev_shard",
                "reserved_flops", "stop_at_flops", "synthetic", "task_fingerprint")
    missing = [k for k in required if k not in spec]
    if missing: raise ValueError(f"run spec is missing {missing}")
    if spec["scale"] not in SCALES: raise ValueError("unknown scale")
    scale = SCALES[spec["scale"]]
    if not spec["synthetic"]:
        if spec["world_size"] not in WORLD_SIZES: raise ValueError(f"world_size must be one of {WORLD_SIZES}")
        if spec["steps"] != scale.steps: raise ValueError("steps must equal the scale definition")
        if spec["task_fingerprint"] != TASK.fingerprint: raise ValueError("run spec belongs to another task version")
    elif spec["steps"] < 1:
        raise ValueError("steps must be positive")
    for key in ("reserved_flops", "stop_at_flops", *(f"stage_costs.{s}" for s in spec["stage_costs"])):
        value = spec["stage_costs"][key.split(".", 1)[1]] if key.startswith("stage_costs.") else spec[key]
        if not (isinstance(value, str) and value.isdigit()): raise ValueError(f"{key} must be an exact decimal string")
    if int(spec["stop_at_flops"]) > int(spec["reserved_flops"]): raise ValueError("stop_at_flops exceeds the reservation")
    if sub is not None:
        stages = set(sub.stage_schedule(spec["steps"], spec.get("schedule_steps")))
        if not stages <= set(spec["stage_costs"]):
            raise ValueError(f"run spec lacks costs for stages {sorted(stages - set(spec['stage_costs']))}")
    return spec


def write_run_spec(spec, path):
    Path(path).write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    return Path(path)


def read_run_spec(path):
    return validate_run_spec(json.loads(Path(path).read_text()))
