"""MSAR: independent single-phase oracle, boundaries, causality and optimizer ownership."""

from dataclasses import replace
import io

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.msar import frozen
from nexttransformer.methods.msar.ops import (DepthRead, TemporalAugmentation, completed_read,
                                             merge_partial, source_key)
from nexttransformer.models.backbone import norm
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.compute import Ledger, attention_rules, audit_graph, count_optimizer
from nexttransformer.protocol.spec import load_config, write_config
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def config(layers=7, sequence_len=7):
    c = debug_config("msar", sequence_len=sequence_len)
    return replace(c, layers=layers)


def activate(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
        for name, parameter in model.named_parameters():
            if name.startswith(("residual_maps.", "msar_initial.", "msar_writes.")):
                parameter.add_(torch.randn_like(parameter) * .1)


def augment_oracle(raw, augmentation):
    """Explicit causal shifts, independent of Conv1d and its padding convention."""
    branches = [raw]
    for weight in augmentation.taps:
        branch = torch.zeros_like(raw)
        for tap in range(weight.shape[-1]):
            lag = weight.shape[-1] - 1 - tap
            if lag < raw.shape[1]:
                shifted = F.pad(raw[:, :raw.shape[1] - lag], (0, 0, lag, 0))
                branch = branch + shifted * weight[:, 0, tap].to(raw.dtype)
        branches.append(branch)
    return torch.stack(branches, dim=2)


def read_oracle(model, sources, step):
    reader = model.residual_maps[step - 1]
    streams = []
    for stream in range(4):
        values = torch.stack([source[:, :, stream] for source in sources])
        keys = values / (values.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        scores = (keys * reader.query[stream]).sum(-1)
        streams.append((scores.softmax(0).unsqueeze(-1) * values).sum(0))
    streams = torch.stack(streams, dim=2)
    flat = streams.flatten(-2)
    rms = flat.square().mean(-1, keepdim=True).sqrt()
    rms = torch.where(rms > 0, rms, torch.ones_like(rms))
    gates = torch.sigmoid((flat @ reader.proj_pre) / rms * reader.alpha_pre + reader.bias_pre)
    return (streams * gates.unsqueeze(-1)).sum(2)


def hidden_oracle(model, tokens):
    x = model._embed_reference(tokens)
    completed = [augment_oracle(x, model.msar_initial)]
    raw, partial = None, None
    for step in range(2 * model.config.n_layer):
        sources = completed + ([] if partial is None else [partial])
        hidden = x if step == 0 else read_oracle(model, sources, step)
        output = (model._attention_branch(hidden, tokens, step // 2) if step % 2 == 0
                  else model.transformer.h[step // 2].mlp(norm(hidden)))
        if step == 2 * model.config.n_layer - 1:
            return hidden + output
        raw = output if raw is None else raw + output
        partial = augment_oracle(raw, model.msar_writes[step])
        if (step + 1) % (2 * model.block_size_layers) == 0:
            completed.append(partial)
            raw, partial = None, None


@pytest.mark.parametrize("layers", [1, 3, 4, 6, 7, 23])
def test_aug_block_sum_matches_independent_forward_and_all_gradients(layers):
    c = config(layers, sequence_len=4)
    model = build_model(c)
    activate(model)
    tokens = torch.randint(c.vocab_size, (1, c.sequence_len))
    actual, backout = model._ordinary_hidden(tokens)
    expected = hidden_oracle(model, tokens)
    assert backout is None
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    params = [p for p in model.parameters() if p.requires_grad and p is not model.lm_head.weight]
    # The hidden-state oracle excludes the LM head but includes every backbone/MSAR parameter.
    actual_grad = torch.autograd.grad(actual.square().mean(), params, retain_graph=True)
    oracle_grad = torch.autograd.grad(expected.square().mean(), params)
    for a, b in zip(actual_grad, oracle_grad):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert model.block_sizes == tuple(min(3, layers - start) for start in range(0, layers, 3))


def test_write_augments_raw_sum_with_latest_writer_not_stream_sum():
    model = build_model(config(3))
    activate(model)
    seen = []
    outputs = []
    handles = [module.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
               for module in model.msar_writes]
    handles += [module.register_forward_hook(lambda module, args, result: outputs.append(result.detach().clone()))
                for block in model.transformer.h for module in (block.attn, block.mlp)]
    try:
        model(torch.randint(64, (1, 7)))
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 5 and len(outputs) == 6
    for step, raw in enumerate(seen):
        torch.testing.assert_close(raw, sum(outputs[:step + 1]))
    correct = model.msar_writes[1](outputs[0] + outputs[1])
    incorrect = model.msar_writes[0](outputs[0]) + model.msar_writes[1](outputs[1])
    assert not torch.allclose(correct, incorrect)


def test_skip_first_keeps_prenorm_and_last_bda_keeps_raw_mlp_input():
    model = build_model(config(4))
    activate(model)
    tokens = torch.randint(64, (1, 7))
    embedding = model._embed_reference(tokens)
    first, last = [], []
    handles = [model.transformer.h[0].attn.register_forward_pre_hook(lambda m, args: first.append(args[0])),
               model.residual_maps[-1].register_forward_hook(lambda m, args, out: last.append(out)),
               model.transformer.h[-1].mlp.register_forward_pre_hook(lambda m, args: last.append(args[0])),
               model.transformer.h[-1].mlp.register_forward_hook(lambda m, args, out: last.append(out))]
    try:
        hidden, _ = model._ordinary_hidden(tokens)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(first[0], norm(embedding), rtol=0, atol=0)
    torch.testing.assert_close(last[1], norm(last[0]), rtol=0, atol=0)
    torch.testing.assert_close(hidden, last[0] + last[2], rtol=0, atol=0)
    assert len(model.residual_maps) == 7 and len(model.msar_writes) == 7
    assert not any(row.get("sublayer") == 0 for row in model.last_trace if row["kind"] == "depth_read")


def test_initialization_gradients_optimizer_groups_and_unused_parameters():
    model = build_model(config())
    assert model.residual_maps[0].query.shape == (4, 48)
    model.set_training_stage(0)
    for module in [model.msar_initial, *model.msar_writes]:
        assert all(torch.count_nonzero(weight) == 0 for weight in module.taps)
    for read in model.residual_maps:
        assert not read.query.any() and not read.bias_pre.any()
        assert read.alpha_pre.item() == pytest.approx(.01)
    embedding = torch.randn(1, 7, 48)
    initial = model.msar_initial(embedding)
    torch.testing.assert_close(initial[:, :, 0], embedding)
    assert not initial[:, :, 1:].any()
    owner = {name: group for group in parameter_groups(model) for name in group["parameter_names"]}
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert not parameter.requires_grad and name not in owner
        elif name.startswith(("residual_maps.", "msar_initial.", "msar_writes.")):
            assert owner[name]["kind"] == "adamw" and owner[name]["weight_decay"] == 0
    tokens = torch.randint(64, (2, 7))
    model(tokens, tokens).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_causality_batch_isolation_and_interleaved_forward_lifetimes():
    model = build_model(config())
    activate(model)
    model.eval()
    x = torch.randint(64, (2, 7))
    changed = x.clone()
    changed[0, 4:] = (changed[0, 4:] + 9) % 64
    changed[1] = (changed[1] + 17) % 64
    first = model(x)
    second = model(changed)
    torch.testing.assert_close(first[0, :4], second[0, :4], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(first, model(x), rtol=0, atol=0)
    (first.square().sum() + second.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    assert not any("cache" in name or "partial" in name for name in model.state_dict())


def test_checkpoint_and_config_roundtrip(tmp_path):
    model = build_model(config())
    activate(model)
    x = torch.randint(64, (2, 7))
    expected = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(model.config, seed=123)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    path = tmp_path / "msar.json"
    write_config(path, model.config)
    assert load_config(path) == model.config
    assert candidate_config("msar").width == 1536
    assert replace(candidate_config("msar"), layers=23).n_layer == 23
    for size in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="block_size_layers"):
            replace(model.config, options=dict(model.config.options, block_size_layers=size))
    with pytest.raises(ValueError, match="full causal"):
        replace(model.config, window_pattern="SSSL")


@pytest.mark.parametrize("length", [1, 3, 13])
def test_causal_convolution_matches_shift_oracle_including_gradients(length):
    module = TemporalAugmentation(4).double()
    with torch.no_grad():
        for weight in module.taps:
            weight.normal_()
    x = torch.randn(2, length, 4, dtype=torch.float64, requires_grad=True)
    actual, expected = module(x), augment_oracle(x, module)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (x, *module.parameters())
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scale", [1., 1000.])
def test_two_phase_fp64_value_key_query_gradients_match_single_softmax(scale):
    sources = [torch.randn(1, 3, 4, 8, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    keys = [torch.randn(1, 3, 4, 2, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    queries = (torch.randn(3, 4, 2, dtype=torch.float64) * scale).requires_grad_()
    value, lse = completed_read(queries, sources[:-1], keys[:-1])
    actual = torch.stack([merge_partial(value[i], lse[i], sources[-1], keys[-1], q)
                          for i, q in enumerate(queries)])
    scores = torch.einsum("qnk,cbtnk->qcbtn", queries, torch.stack(keys))
    expected = (scores.softmax(1).unsqueeze(-1) * torch.stack(sources).unsqueeze(0)).sum(1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    inputs = (queries, *sources, *keys)
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_hpre_and_key_zero_sources_have_finite_gradients(dtype):
    reader = DepthRead(8).to(dtype)
    x = torch.zeros(1, 2, 4, 8, dtype=dtype, requires_grad=True)
    (reader(x).sum() + source_key(x).sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in [reader.proj_pre, reader.alpha_pre, reader.bias_pre])


def test_bf16_autocast_keeps_depth_coefficients_fp32_and_values_bf16():
    sources = [torch.randn(1, 3, 4, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    query = torch.randn(2, 4, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        keys = [source_key(source) for source in sources]
        value, lse = completed_read(query, sources[:2], keys[:2])
        output = merge_partial(value[0], lse[0], sources[2], keys[2], query[0])
    assert all(key.dtype == torch.float32 for key in keys)
    assert lse.dtype == torch.float32 and output.dtype == value.dtype == torch.bfloat16
    scores = torch.einsum("qnk,cbtnk->qcbtn", query, torch.stack(keys))
    oracle = (scores.softmax(1).unsqueeze(-1) * torch.stack(sources).float().unsqueeze(0)).sum(1)
    torch.testing.assert_close(output.float(), oracle[0], rtol=.03, atol=.02)
    output.float().square().sum().backward()
    assert torch.isfinite(query.grad).all()
    assert all(torch.isfinite(source.grad).all() for source in sources)


def test_cost_ledger_covers_forward_backward_and_real_optimizer_groups():
    # Real tiny tensors also exercise the public operator ledger without depending
    # on a particular PyTorch FakeTensor/meta materialization implementation.
    model = build_model(config(4, 4))
    model.set_training_stage(0)
    ledger = Ledger()
    tokens = torch.zeros((2, 4), dtype=torch.long)
    with attention_rules(model, ledger, "joint"), ledger:
        loss = model(tokens, tokens)
        ledger.phase = "backward"
        loss.backward()
    count_optimizer(model, ledger)
    proof = ledger.report()
    assert not proof["unknown"], proof["unknown"]
    assert all(proof["phases"][phase] > 0 for phase in ("forward", "backward", "optimizer"))
    assert any("convolution_backward" in op for op in proof["operators"])


@pytest.mark.skipif(torch.__version__ != "2.9.1+cu128", reason="formal FakeTensor proof requires audited PyTorch")
def test_formal_fake_tensor_audit():
    proof = audit_graph(config(4, 4), batch=2)
    assert not proof["unknown"], proof["unknown"]
    assert all(proof["phases"][phase] > 0 for phase in ("forward", "backward", "optimizer"))
    assert any("convolution_backward" in op for op in proof["operators"])
    assert any("optimizer/" in op for op in proof["operators"])
