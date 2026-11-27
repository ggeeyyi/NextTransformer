"""Blocking evaluate(): one architecture in, one score out. For frameworks that call an evaluation function.

    from nextt_gym.adapter import evaluate
    result = evaluate("candidate_model.py", config="config.json", scale="S", backend="a100-smoke")
    result["combined_score"]   # -validation BPB (higher is better); worst_score on any failure

    python -m nextt_gym.adapter candidate_model.py --config config.json --scale S --backend a100-smoke

It submits, waits for verification, optionally calibrates declared tunable fields, quotes, launches, and
waits for the run. Calls are idempotent on content: the same model.py + config.json re-attaches to the
existing attempt and run instead of launching again, so framework retries and timeouts never double-spend.
Every failure, including an exhausted budget, returns ok=False with worst_score; nothing raises.
"""
import argparse
import json
from pathlib import Path
import time

from .errors import GymError
from .service import REPO_ROOT, TERMINAL, GymService
from .contract import files_sha256

WORST_SCORE = -1e9
TEMPLATE_CONFIG = REPO_ROOT / "nextt_gym/templates/config.json"


def _text(value):
    if isinstance(value, dict): return json.dumps(value, indent=2) + "\n"
    path = Path(value)
    return path.read_text() if "\n" not in str(value) and path.exists() else str(value)


def _content_key(files):
    import hashlib, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items(): (Path(tmp) / name).write_text(text)
        return files_sha256(tmp)


class Evaluator:
    """Holds one GymService; evaluate() may be called repeatedly (and from several threads)."""

    def __init__(self, service=None, *, home=None, backends_path=None):
        self.service = service or GymService(home=home, backends_path=backends_path)

    def _existing_attempt(self, files):
        sha = _content_key(files)
        row = self.service.store.one("SELECT id FROM attempts WHERE files_sha256=? ORDER BY seq DESC", (sha,))
        return row["id"] if row else None

    def _wait_verification(self, attempt_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            attempt = self.service.get_attempt(attempt_id)
            if attempt["status"] in ("verified", "rejected"): return attempt
            time.sleep(2)
        raise GymError("INTERNAL", f"verification of {attempt_id} did not finish in {timeout} s")

    def _wait_run(self, run_id, timeout, poll):
        deadline = time.monotonic() + timeout
        while True:
            run = self.service.get_run(run_id)
            if run["status"] in TERMINAL or time.monotonic() >= deadline: return run
            time.sleep(poll)

    def evaluate(self, model, config=None, *, scale="S", backend=None, world_size=None, calibrate=(),
                 title=None, hypothesis="", parent_ids=(), tags=(), verify_timeout=1800, run_timeout=None,
                 poll_seconds=30, worst_score=WORST_SCORE):
        files = {"model.py": _text(model), "config.json": _text(config if config is not None else TEMPLATE_CONFIG)}
        out = dict(ok=False, combined_score=worst_score, scale=scale, attempt_id=None, run_id=None,
                   metrics={}, error=None, feedback="")
        try:
            backend = backend or next(iter(self.service.backends), None)
            if backend is None: raise GymError("BACKEND_UNAVAILABLE", "no backend configured")
            attempt_id = self._existing_attempt(files)
            if attempt_id is None:
                key = "adapter-" + _content_key(files)
                try:
                    attempt_id = self.service.submit_attempt(
                        files, title=title or f"adapter {key[8:20]}", hypothesis=hypothesis or "submitted by adapter",
                        parent_ids=list(parent_ids), tags=list(tags) + ["adapter"], idempotency_key=key)["attempt_id"]
                except GymError as exc:
                    if exc.code != "DUPLICATE_ATTEMPT": raise
                    attempt_id = exc.details["duplicate_of"]
            out["attempt_id"] = attempt_id
            attempt = self._wait_verification(attempt_id, verify_timeout)
            if attempt["status"] == "rejected":
                verification = self.service.get_verification(attempt["verification_id"])
                errors = [e for level in verification["levels"] for e in level.get("errors", [])]
                only_flops = errors and all(e["code"] == "FLOPS_OUT_OF_TOLERANCE" for e in errors)
                if not (only_flops and calibrate):
                    out.update(error=errors[0] if errors else dict(code="SPEC_VIOLATION", message="rejected"),
                               feedback="\n".join(f"{e['code']}: {e['message']}" for e in errors)); return out
                child = next((a for a in self.service.list_attempts(descendant_of=attempt_id, limit=5)["items"]
                              if a["status"] == "verified"), None)
                attempt_id = child["attempt_id"] if child else self.service.calibrate(attempt_id, list(calibrate))["attempt_id"]
                out["attempt_id"] = attempt_id
            runs = [r for r in self.service.archive.list_runs(attempt_id=attempt_id, scale=scale)
                    if r["status"] not in ("failed", "cancelled", "budget_exceeded")]
            if runs:
                run_id = runs[-1]["run_id"]
            else:
                quote = self.service.quote_run(attempt_id, scale, backend, world_size)
                if not quote["feasible"]:
                    blocker = quote["blockers"][0]
                    out.update(error=blocker, feedback=f"{blocker['code']}: {blocker['message']}"); return out
                run_id = self.service.launch_run(quote["quote_id"], f"adapter-{attempt_id}-{scale}")["run_id"]
            out["run_id"] = run_id
            run = self._wait_run(run_id, run_timeout or 10 ** 9, poll_seconds)
            final = run.get("final") or {}
            out["metrics"] = dict(final, consumed_flops=run["consumed"]["flops"], status=run["status"],
                                  **{k: v for k, v in (run.get("latest") or {}).items()
                                     if k in ("step_seconds", "tokens_per_second", "peak_memory_gb")})
            bpb = final.get("val_bpb_dev")
            if run["status"] != "succeeded" or bpb is None:
                out.update(error=run.get("failure") or dict(code="INTERNAL", message=f"run {run['status']}"),
                           feedback=f"run {run_id} ended {run['status']}"); return out
            out.update(ok=True, combined_score=-float(bpb),
                       feedback=f"val_bpb_dev={bpb:.4f} at scale {scale}; consumed {run['consumed']['flops']:.3g} FLOPs")
        except GymError as exc:
            out.update(error=exc.to_dict(), feedback=f"{exc.code}: {exc.message}")
        return out


_default = None


def evaluate(model, config=None, **kwargs):
    """Module-level convenience: one shared Evaluator (NEXTT_GYM_HOME / NEXTT_GYM_BACKENDS from the environment)."""
    global _default
    if _default is None: _default = Evaluator()
    return _default.evaluate(model, config, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model"); parser.add_argument("--config")
    parser.add_argument("--scale", default="S"); parser.add_argument("--backend")
    parser.add_argument("--world-size", type=int); parser.add_argument("--calibrate", default="",
                                                                      help="comma-separated tunable fields")
    parser.add_argument("--run-timeout", type=float); parser.add_argument("--home"); parser.add_argument("--backends")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    result = Evaluator(home=args.home, backends_path=args.backends).evaluate(
        args.model, args.config, scale=args.scale, backend=args.backend, world_size=args.world_size,
        calibrate=[k for k in args.calibrate.split(",") if k], run_timeout=args.run_timeout)
    text = json.dumps(result, indent=1, default=str)
    if args.json_out: Path(args.json_out).write_text(text)
    print(text)
    print(f"combined_score: {result['combined_score']}")


if __name__ == "__main__":
    main()
