"""adapter.evaluate(): score in one call, idempotent on content, failures as worst_score."""
from pathlib import Path

from nextt_gym.adapter import WORST_SCORE, Evaluator

from .test_service_e2e import gym  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "nextt_gym/templates"


def test_evaluate_scores_and_reattaches(gym):
    ev = Evaluator(gym)
    first = ev.evaluate(TEMPLATE / "model.py", TEMPLATE / "config.json", scale="S", calibrate=["mlp_width"],
                        poll_seconds=2, run_timeout=600)
    assert first["ok"], first
    assert first["combined_score"] == -first["metrics"]["val_bpb_dev"] and first["run_id"]
    runs_before = len(gym.archive.list_runs(origin="agent"))
    again = ev.evaluate(TEMPLATE / "model.py", TEMPLATE / "config.json", scale="S", calibrate=["mlp_width"],
                        poll_seconds=2, run_timeout=600)
    assert again["run_id"] == first["run_id"] and again["combined_score"] == first["combined_score"]
    assert len(gym.archive.list_runs(origin="agent")) == runs_before          # no second launch


def test_evaluate_failures_are_scores_not_exceptions(gym):
    ev = Evaluator(gym)
    broken = ev.evaluate("import os\ndef build(c):\n    return None\n", TEMPLATE / "config.json")
    assert not broken["ok"] and broken["combined_score"] == WORST_SCORE
    assert broken["error"]["code"] == "SPEC_VIOLATION" and "import os" in str(broken["error"])
    over = ev.evaluate(TEMPLATE / "model.py", TEMPLATE / "config.json")       # +4% FLOPs, no calibrate knobs
    assert not over["ok"] and over["error"]["code"] == "FLOPS_OUT_OF_TOLERANCE"
