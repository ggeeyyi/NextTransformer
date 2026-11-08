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


def inspect(candidate, profile):
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
    update = ledger if rows == profile["batch_size"] else Ledger()
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
    admission = update.report()
    init = initialization.report()
    unknown = {
        **report["unknown"],
        **admission["unknown"],
        **init["unknown"],
        **evaluation.report()["unknown"],
        **prediction.report()["unknown"],
    }
    per_token = admission["total"] / (rows * profile["sequence_len"])
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
        "training_flops_per_token": per_token,
        "anchor_ratio": per_token / ANCHOR_FLOPS_PER_TOKEN,
        "admission_tokens_per_update": rows * profile["sequence_len"],
        "accounting_version": report["formula_version"],
        "scope": "native plugin trace for the pinned configuration; not a proof of every Python branch",
    }
