"""CORE input lengths for evaluation and result submissions."""
import argparse
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def benchmark_lengths():
    return json.loads((Path(__file__).with_name("core-lengths.json")).read_text())


def positive_tokens(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("the CORE token limit must be positive")
    return count


def add_core_max_tokens(parser):
    parser.add_argument("--core-max-tokens", type=positive_tokens, metavar="N",
                        help="left-truncate each CORE input to N tokens (default: use the full prompt)")


def check_full_context(config, max_tokens):
    """Check the fixed bundle before starting a full run or a full evaluation."""
    lengths = benchmark_lengths()
    minimum = lengths["longestAnswerTokens"] + 1
    if max_tokens is not None and max_tokens < minimum:
        raise ValueError(f"--core-max-tokens must be at least {minimum} for full CORE "
                         "to retain every scored answer and its preceding token")
    extent = dict(getattr(config, "options", ())).get("rel_extent")
    required = lengths["maxPromptTokens"]
    if max_tokens is not None:
        required = min(required, max_tokens)
    if extent is not None and required > extent:
        raise ValueError(f"CORE needs {required} tokens but the learned REL table covers {extent}. "
                         f"Set --core-max-tokens {extent} to use left truncation.")


def core_context(training_tokens, max_tokens, measured):
    return {"trainingTokens": training_tokens, "truncation": "none" if max_tokens is None else "left",
            "maxTokens": max_tokens, "maxEvaluatedTokens": measured}


def validate_submission(context):
    """All full leaderboard runs share the same tokenizer, 22 tasks and training length."""
    lengths = benchmark_lengths()
    if not isinstance(context, dict) or not {"trainingTokens", "truncation", "maxTokens",
                                              "maxEvaluatedTokens"} <= context.keys():
        raise ValueError("CORE context must declare trainingTokens, truncation, maxTokens and maxEvaluatedTokens")
    cap = context["maxTokens"]
    if cap is not None and (type(cap) is not int or cap <= lengths["longestAnswerTokens"]):
        raise ValueError("CORE maxTokens must retain every answer and at least one preceding token")
    expected = min(cap, lengths["maxPromptTokens"]) if cap is not None else lengths["maxPromptTokens"]
    if (type(context["trainingTokens"]) is not int or type(context["maxEvaluatedTokens"]) is not int
            or context != core_context(lengths["trainingTokens"], cap, expected)):
        raise ValueError("CORE context disagrees with the truncation setting or the pinned prompt lengths")


def resume_core_limit(state, requested, training_tokens):
    """Old checkpoints used the training context. Keep the policy across segments."""
    previous = state.get("core_max_tokens", training_tokens)
    if requested is not None and requested != previous:
        raise ValueError(f"resume CORE length policy differs from the checkpoint (max tokens: {previous})")
    return previous
