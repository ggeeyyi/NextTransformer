"""Server-side verification of a submission: V0 static, V1 structure, V2 correctness, V3 FLOPs audit.

Run in a separate process (the service does this) because V1-V3 execute the
submitted code:  python -m nextt_gym.verify <submission_dir> --out result.json
V4 (on-GPU throughput and measured-FLOPs cross-check) runs at the start of a run.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import time
import traceback

from .contract import OPTIONAL_FILES, REQUIRED_FILES
from .errors import error
from .task import FLOPS_TOLERANCE, SCALES, TASK

ALLOWED_IMPORTS = ("torch", "math", "functools", "itertools", "collections", "dataclasses", "typing",
                   "numbers", "enum", "einops", "nexttransformer.models", "nanochat.flash_attention", "nanochat.common",
                   "nextt_gym.upstream_model")
KERNEL_IMPORTS = ("triton",)
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input",
                "breakpoint", "exit", "quit", "delattr"}
BANNED_ATTRS = {"__torch_function__", "__torch_dispatch__", "__globals__", "__builtins__", "__subclasses__",
                "__code__", "__dict__", "__class__", "__bases__", "__mro__", "__getattribute__",
                "manual_seed", "seed", "set_rng_state", "ops", "library", "cpp_extension", "_C",
                "compile", "load_inline", "save", "load"}
TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
             ast.AnnAssign)


# ---- V0 -------------------------------------------------------------------------
def _allowed(module, extra=()):
    return any(module == a or module.startswith(a + ".") for a in ALLOWED_IMPORTS + tuple(extra))


def static_check(path, *, kernels=False):
    """List of violations in one source file (empty when clean)."""
    out = []
    source = Path(path).read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path.name}:{exc.lineno}: syntax error: {exc.msg}"]
    modules = set()
    extra = KERNEL_IMPORTS if kernels else ()
    for node in tree.body:
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        if not (isinstance(node, TOP_LEVEL) or docstring):
            out.append(f"{path.name}:{node.lineno}: only imports, definitions and assignments are allowed at module level")
    for node in ast.walk(tree):
        where = f"{path.name}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _allowed(alias.name, extra): out.append(f"{where}: import {alias.name} is not allowed")
                modules.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level: out.append(f"{where}: relative imports are not allowed"); continue
            if not _allowed(node.module or "", extra): out.append(f"{where}: from {node.module} import is not allowed")
            for alias in node.names:
                if alias.name == "*": out.append(f"{where}: star imports are not allowed")
                modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            out.append(f"{where}: call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            out.append(f"{where}: attribute .{node.attr} is not allowed")
        elif isinstance(node, ast.Name) and (node.id in {"__builtins__", "__import__"} or node.id in BANNED_ATTRS
                                              and node.id.startswith("__")):
            out.append(f"{where}: name {node.id} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in BANNED_ATTRS \
                and node.name.startswith("__"):
            out.append(f"{where}: defining {node.name} is not allowed")
    # No assignment to attributes of imported modules (monkey-patching).
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AugAssign, ast.AnnAssign)) else node.targets if isinstance(node, ast.Delete) else []
        for target in targets:
            root = target
            while isinstance(root, (ast.Attribute, ast.Subscript)): root = root.value
            if isinstance(root, ast.Name) and root.id in modules and root is not target:
                out.append(f"{path.name}:{node.lineno}: modifying imported module {root.id!r} is not allowed")
    return out


def v0(directory):
    directory = Path(directory)
    problems = [f"missing {name}" for name in REQUIRED_FILES if not (directory / name).exists()]
    allowed = set(REQUIRED_FILES + OPTIONAL_FILES)
    problems += [f"unexpected file {p.name}" for p in directory.iterdir() if p.name not in allowed]
    size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    if size > TASK.policy.submission_max_bytes:
        return [error("PAYLOAD_TOO_LARGE", f"submission is {size} bytes", details=dict(limit=TASK.policy.submission_max_bytes))]
    if not problems:
        problems += static_check(directory / "model.py")
        if (directory / "kernels.py").exists(): problems += static_check(directory / "kernels.py", kernels=True)
        try:
            config = json.loads((directory / "config.json").read_text())
            for key in ("model", "model_dim"):
                if key not in config: problems.append(f"config.json: missing {key!r}")
            for key, value in (("vocab_size", 32768), ("sequence_len", 2048)):
                if config.get(key, value) != value: problems.append(f"config.json: {key} is fixed at {value}")
            for name, rule in config.get("tunable", {}).items():
                if name not in config.get("model", {}) or not isinstance(config["model"][name], int):
                    problems.append(f"config.json: tunable field {name!r} must be an integer model field")
                if not {"min", "max"} <= set(rule): problems.append(f"config.json: tunable {name!r} needs min/max")
        except json.JSONDecodeError as exc:
            problems.append(f"config.json: invalid JSON ({exc})")
    return [error("SPEC_VIOLATION", "submission violates the model contract", details=dict(violations=problems),
                  hint="see nextt://spec")] if problems else []


# ---- V1 -------------------------------------------------------------------------
def v1(sub):
    from .roles import parameter_groups, parameter_roles
    problems = []
    forward = sub.module.build  # signature checked on the instance
    model = sub.instantiate("meta")
    params = inspect.signature(model.forward).parameters
    for name in ("idx", "targets", "loss_reduction"):
        if name not in params: problems.append(f"forward() must accept {name!r}")
    try:
        parameter_roles(model)
        horizon = SCALES["S"].schedule_steps
        stage0 = sub.stage_for_step(0, horizon)
        sub.set_stage(model, stage0, audit=False, step=0, total_steps=horizon)
        parameter_groups(model, sub.model_dim)
    except Exception as exc:
        problems.append(f"parameter roles / optimizer grouping: {exc}")
    if problems:
        return [error("SPEC_VIOLATION", "structure check failed", details=dict(violations=problems))], {}
    return [], dict(parameters=sum(p.numel() for p in model.parameters()))


# ---- V2 -------------------------------------------------------------------------
def v2(sub, *, batch=2, seed=0):
    import torch
    errors = []
    t = min(sub.debug_sequence_len, 32)
    gen = torch.Generator().manual_seed(seed)
    model = sub.instantiate("cpu", overrides=sub.debug_overrides)
    total = SCALES["S"].schedule_steps
    sub.set_stage(model, sub.stage_for_step(0, total), audit=False, step=0, total_steps=total)
    sub.set_step(model, 0); sub.set_row_offset(model, 0)
    x = torch.randint(sub.debug_vocab_size, (batch, t), generator=gen)
    y = torch.randint(sub.debug_vocab_size, (batch, t), generator=gen)
    model.eval()
    with torch.no_grad():
        logits = model(x).float()
        if tuple(logits.shape[:2]) != (batch, t):
            return [error("SPEC_VIOLATION", f"logits shape {tuple(logits.shape)} must start with {(batch, t)}")]
        again = model(x).float()
        if not torch.equal(logits, again):
            errors.append(error("NUMERIC_CHECK_FAILED", "eval forward is not deterministic"))
        for j in (1, t // 2, t - 1):
            changed = x.clone(); changed[:, j:] = torch.randint(sub.debug_vocab_size, changed[:, j:].shape, generator=gen)
            out = model(changed).float()
            if not torch.allclose(out[:, :j], logits[:, :j], rtol=1e-4, atol=1e-5):
                errors.append(error("CAUSALITY_VIOLATION", "changing tokens at or after position j changed earlier logits",
                                    details=dict(leak_position=j)))
                break
    model.train()
    loss = model(x, y)
    if loss.ndim != 0 or not torch.isfinite(loss):
        return errors + [error("NUMERIC_CHECK_FAILED", "training loss must be a finite scalar")]
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, g in grads if g is not None and not torch.isfinite(g).all()]
    if bad: errors.append(error("NUMERIC_CHECK_FAILED", "non-finite gradients", details=dict(parameters=bad[:20])))
    if not any(g is not None and g.abs().sum() > 0 for _, g in grads):
        errors.append(error("NUMERIC_CHECK_FAILED", "no parameter received a gradient"))
    model.eval()
    with torch.no_grad():
        token_loss = model(x, y, loss_reduction="none").float()
        if token_loss.numel() != batch * t or not torch.isfinite(token_loss).all():
            errors.append(error("SPEC_VIOLATION", "loss_reduction='none' must return one finite loss per token"))
        elif not sub.aux_loss:
            mean = model(x, y).float()
            if not torch.allclose(token_loss.mean(), mean, rtol=1e-4, atol=1e-5):
                errors.append(error("SPEC_VIOLATION", "mean of per-token loss differs from the training loss; "
                                    "declare aux_loss=true if the objective adds an auxiliary term"))
    return errors


# ---- V3 -------------------------------------------------------------------------
def v3(sub):
    from .audit import audit_submission
    report = audit_submission(sub)
    report.pop("stage_records", None)
    errors = []
    if report["unknown_ops"]:
        errors.append(error("AUDIT_UNKNOWN_OP", "the cost audit met operators it cannot count",
                            details=dict(ops=report["unknown_ops"]),
                            hint="use nanochat.flash_attention.flash_attn.flash_attn_func or explicit matmul/softmax; "
                                 "see nextt://upstream/costs for counted operators"))
    off = {k: v["relative_error_vs_anchor"] for k, v in report["scales"].items() if not v["within_tolerance"]}
    if off and not report["unknown_ops"]:
        worst = max(off.values(), key=abs)
        errors.append(error("FLOPS_OUT_OF_TOLERANCE", f"training FLOPs/token is {worst:+.2%} from the anchor "
                            f"(tolerance {FLOPS_TOLERANCE:.0%})", details=dict(relative_error=off),
                            hint="adjust widths/depths, or call calibrate with a tunable field"))
    return errors, report


def verify(directory, levels=("V0", "V1", "V2", "V3")):
    """Run the requested levels in order; stop at the first failing level."""
    from .contract import load_submission
    result = dict(status="passed", levels=[], audit=None, parameters=None)
    sub = None
    for level in ("V0", "V1", "V2", "V3"):
        if level not in levels:
            continue
        started = time.monotonic(); errors = []
        try:
            if level == "V0": errors = v0(directory)
            else:
                sub = sub or load_submission(directory)
                if level == "V1":
                    errors, info = v1(sub); result["parameters"] = info.get("parameters")
                elif level == "V2": errors = v2(sub)
                else: errors, result["audit"] = v3(sub)
        except Exception as exc:
            errors = [error("SPEC_VIOLATION", f"{level} raised {type(exc).__name__}: {exc}",
                            details=dict(traceback=traceback.format_exc(limit=6)[-2000:]))]
        result["levels"].append(dict(level=level, status="failed" if errors else "passed",
                                     seconds=round(time.monotonic() - started, 2), errors=errors))
        if errors:
            result["status"] = "failed"; break
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission"); parser.add_argument("--out")
    parser.add_argument("--levels", default="V0,V1,V2,V3")
    args = parser.parse_args(argv)
    result = verify(args.submission, tuple(args.levels.split(",")))
    text = json.dumps(result, indent=1, default=str)
    if args.out: Path(args.out).write_text(text)
    else: print(text)


if __name__ == "__main__":
    main()
