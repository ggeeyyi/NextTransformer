"""The arithmetic charges of a research job, from its quote's trusted cost report.

The worker charges these amounts operation by operation, and the quote's bound is their sum. An executor that bills
ahead of work (`executor.Charges`) uses the same functions, so the cost model lives in one place.
"""


def initialization_cost(report):
    return int(report["initialization"])


def update_cost(report):
    """One optimizer update of the global batch, split as the research worker splits it."""
    return (
        int(report.get("research_forward", report["forward"]))
        + int(report.get("research_backward", report["backward"]))
        + int(report.get("research_reduction", 0))
        + int(report["optimizer"])
    )


def evaluation_cost(report, dev_batches):
    return int(report.get("research_evaluation", report["evaluation"])) * int(
        dev_batches
    )


def training_bound(report, updates, dev_batches):
    """A train quote's bound: initialization, every update, and the dev evaluation."""
    return (
        initialization_cost(report)
        + int(updates) * update_cost(report)
        + evaluation_cost(report, dev_batches)
    )
