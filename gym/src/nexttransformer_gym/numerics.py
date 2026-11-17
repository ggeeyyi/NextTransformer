"""Native model construction and NextTransformer cost inspection in a worker."""

from .native_plugin import activate, model_config, check_interface

# Training FLOPs per input token of the pinned dense anchor (evidence/costs/*/dense.json).
ANCHOR_FLOPS_PER_TOKEN = 4672398325.44785


def build(candidate, profile):
    activate(candidate)
    from nexttransformer.models import build_model

    model = build_model(
        model_config(candidate, profile), profile.get("device", "cpu"), seed=42
    )
    check_interface(model)
    return model


def _inspect(candidate, profile, admission=True):
    activate(candidate)
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from nexttransformer.protocol.compute import (
        Ledger,
        attention_rules,
        count_optimizer,
    )
    from nexttransformer.protocol.spec import Protocol

    def train_step(model, rows, ledger):
        x = torch.zeros((rows, profile["sequence_len"]), dtype=torch.long)
        with attention_rules(model, ledger, "joint"), ledger:
            loss = model(x, x)
            ledger.phase = "backward"
            loss.backward()
        count_optimizer(model, ledger)
        return x

    # The forward, backward and optimizer bounds price one step of this profile. Admission
    # instead compares training FLOPs per token with the anchor at the protocol's update size
    # (one optimizer step per 1,048,576 tokens, 512 rows of 2048), as the protocol calibrates.
    rows = max(1, Protocol().global_batch_tokens // profile["sequence_len"])
    initialization = Ledger()
    ledger = Ledger()
    update = ledger if not admission or rows == profile["batch_size"] else Ledger()
    evaluation = Ledger()
    prediction = Ledger()
    with FakeTensorMode(allow_non_fake_inputs=True):
        with initialization:
            model = build(candidate, {**profile, "device": "cpu"})
        x = train_step(model, profile["batch_size"], ledger)
        parameters = sum(p.numel() for p in model.parameters())
        if update is not ledger:
            train_step(build(candidate, {**profile, "device": "cpu"}), rows, update)
        model.eval()
        with torch.no_grad(), attention_rules(model, prediction, "joint"), prediction:
            model(x)
        with torch.no_grad(), attention_rules(model, evaluation, "joint"), evaluation:
            logits = model(x)
            losses = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                x.reshape(-1),
                reduction="none",
            )
            valid = torch.ones_like(x.reshape(-1), dtype=torch.bool)
            (losses * valid).sum()
    report = ledger.report()
    admitted_update = update.report()
    init = initialization.report()
    unknown = {
        **report["unknown"],
        **admitted_update["unknown"],
        **init["unknown"],
        **evaluation.report()["unknown"],
        **prediction.report()["unknown"],
    }
    traced_rows = rows if update is not ledger else profile["batch_size"]
    per_token = admitted_update["total"] / (traced_rows * profile["sequence_len"])
    return {
        "coverage": "complete" if not unknown else "incomplete",
        "unknown": unknown,
        "initialization": str(init["total"]),
        "evaluation": str(evaluation.report()["total"]),
        "prediction": str(prediction.report()["total"]),
        "forward": str(report["phases"]["forward"]),
        "backward": str(report["phases"]["backward"]),
        "optimizer": str(report["phases"]["optimizer"]),
        "parameters": parameters,
        "activation_depth": model.config.maximum_depth,
        "training_flops_per_token": per_token,
        "anchor_ratio": per_token / ANCHOR_FLOPS_PER_TOKEN,
        "admission_tokens_per_update": traced_rows * profile["sequence_len"],
        "accounting_version": report["formula_version"],
        "scope": "native plugin trace for the pinned configuration; not a proof of every Python branch",
    }


def inspect(candidate, profile):
    """Trace actual research microbatch shapes, retaining full-batch diagnostics."""
    from collections import Counter
    from .research import microbatches, replica_slices

    report = _inspect(candidate, profile)
    batch = profile["batch_size"]
    micro = profile.get("microbatch_size", batch)
    devices = profile.get("research_device_count", 1)
    shapes = Counter()
    rounds = 0
    for start, end in microbatches(batch, micro, devices):
        rounds += 1
        for first, last in replica_slices(start, end, devices):
            shapes[last - first] += 1
    evaluation_shapes = Counter(shapes)
    traces = {batch: report}
    for size in shapes.keys() | evaluation_shapes.keys():
        if size not in traces:
            traces[size] = _inspect(
                candidate, {**profile, "batch_size": size}, admission=False
            )
        if traces[size]["coverage"] != "complete":
            report["coverage"] = "incomplete"
            report["unknown"].update(traces[size]["unknown"])
    report["research_forward"] = str(
        sum(int(traces[size]["forward"]) * count for size, count in shapes.items())
    )
    report["research_backward"] = str(
        sum(int(traces[size]["backward"]) * count for size, count in shapes.items())
    )
    # Conservative arithmetic bound for loss scaling, gradient reduction and
    # accumulation. Tensor copies/communication do not count as arithmetic.
    report["research_reduction"] = str(
        2 * sum(shapes.values())
        + report["parameters"] * (sum(shapes.values()) + rounds)
        if devices > 1 or micro < batch
        else 0
    )
    report["research_evaluation"] = str(
        sum(
            int(traces[size]["evaluation"]) * count
            for size, count in evaluation_shapes.items()
        )
    )
    return report
