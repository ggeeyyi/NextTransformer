"""Global-batch gradient and row-identity contracts for local research."""

import copy

import pytest
import torch
from torch import nn

from nexttransformer_gym.research import training_backward


class RowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.offset = 0

    def set_row_offset(self, offset):
        self.offset = offset

    def forward(self, x, y):
        rows = torch.arange(self.offset, self.offset + len(x), device=x.device)
        prediction = self.linear(x) + rows[:, None] * 0.01
        return (prediction - y).square().mean()


@pytest.mark.parametrize("microbatch", [1, 2, 3, 7, 10])
def test_accumulation_matches_global_batch(microbatch):
    torch.manual_seed(42)
    model = RowModel()
    reference = copy.deepcopy(model)
    x, y = torch.randn(7, 3), torch.randn(7, 2)
    reference.set_row_offset(21)
    expected = reference(x, y)
    expected.backward()
    actual = training_backward(model, x, y, 21, microbatch, 1)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6)
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad)
    assert model.offset == 21


# Explicit opt-in prevents a normal test invocation from occupying host GPUs.
@pytest.mark.skipif(
    __import__("os").environ.get("GYM_TEST_MULTI_GPU") != "1",
    reason="requires an explicitly reserved local GPU group",
)
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_two_gpu_gradient_matches_single_gpu(microbatch):
    assert torch.cuda.device_count() == 2
    torch.manual_seed(42)
    model = RowModel().cuda()
    reference = copy.deepcopy(model)
    x, y = torch.randn(7, 3, device="cuda"), torch.randn(7, 2, device="cuda")
    reference.set_row_offset(21)
    expected = reference(x, y)
    expected.backward()
    actual = training_backward(model, x, y, 21, microbatch, 2)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-5)
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad)
    assert torch.cuda.max_memory_allocated(1) > 0


def test_accumulated_checkpoint_continuation_and_token_count(tmp_path):
    from test_gym import session, run_job

    context = session(tmp_path)
    profile = context[0].backend.profiles["research-fixture"]
    profile.update(batch_size=7, microbatch_size=2)
    first = run_job(context, target_update=1)
    resumed = run_job(context, target_update=2, checkpoint_id=first["checkpoint_id"])
    full = run_job(context, target_update=2)
    backend = context[0].backend
    a = torch.load(
        backend.checkpoint_path(resumed["checkpoint_id"]), weights_only=False
    )
    b = torch.load(backend.checkpoint_path(full["checkpoint_id"]), weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["data"] == b["data"]
    assert resumed["training_tokens"] == full["training_tokens"] == 112
    assert context[2]("get_budget")["reserved"] == "0"


class EvalRowModel(nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, 6)
        self.head = nn.Linear(6, vocab)
        self.offset = 0

    def set_row_offset(self, offset):
        self.offset = offset

    def forward(self, x):
        rows = torch.arange(self.offset, self.offset + len(x), device=x.device)
        hidden = self.embed(x) + rows[:, None, None] * 0.01
        return self.head(hidden)


@pytest.mark.parametrize("replicas,microbatch", [(1, 2), (2, 1), (2, 2), (3, 2)])
def test_parallel_evaluation_matches_full_batch_with_row_offsets(replicas, microbatch):
    from nexttransformer_gym.research import ResearchParallel

    torch.manual_seed(42)
    vocab = 16
    model = EvalRowModel(vocab)
    parallel = ResearchParallel(model, 1)
    for _ in range(1, replicas):
        parallel.models.append(copy.deepcopy(model))

    x = torch.randint(0, vocab, (7, 5))
    labels = torch.randint(0, vocab, (7, 5))
    sizes = torch.tensor([0 if i == 0 else (i % 3) + 1 for i in range(vocab)])

    reference = copy.deepcopy(model)
    reference.eval()
    with torch.no_grad():
        reference.set_row_offset(14)
        logits = reference(x)
        targets = labels.reshape(-1)
        losses = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, vocab), targets, reduction="none"
        )
        valid = sizes[targets] > 0
        expected_nats = float((losses * valid).sum())
        expected_bytes = int(sizes[targets][valid].sum())
        expected_tokens = int(targets.numel())

    nats, byte_count, tokens = parallel.evaluate_batch(x, labels, sizes, 14, microbatch)
    assert nats == pytest.approx(expected_nats, rel=1e-6)
    assert byte_count == expected_bytes
    assert tokens == expected_tokens
    assert all(m.offset == 14 and not m.training for m in parallel.models)
