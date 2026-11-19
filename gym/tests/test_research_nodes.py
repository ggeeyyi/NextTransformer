"""Research data parallelism across nodes: CPU processes in a gloo group stand in for the nodes."""

import socket

import pytest
import torch
import torch.distributed as dist
from nexttransformer_gym import research
from nexttransformer_gym.research import ResearchParallel
from torch import multiprocessing, nn

VOCAB, ROWS, LENGTH = 11, 7, 4


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 5)
        self.head = nn.Linear(5, VOCAB)
        self.offset, self.rng = 0, 0

    def set_row_offset(self, offset):
        self.offset = offset

    def set_step(self, step):
        self.rng = step

    def routing_rng_state(self):
        return {"offset": self.offset, "rng": self.rng}

    def restore_routing_rng(self, state):
        self.rng = state["rng"] + 1000

    def forward(self, x, y=None):
        rows = torch.arange(self.offset, self.offset + len(x))[:, None, None] * 0.01
        logits = self.head(self.embed(x)) + rows
        if y is None:
            return logits
        return nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))


def inputs(rows):
    generator = torch.Generator().manual_seed(7)
    x = torch.randint(0, VOCAB, (rows, LENGTH), generator=generator)
    y = torch.randint(0, VOCAB, (rows, LENGTH), generator=generator)
    sizes = torch.tensor([0, 1, 2, 3, 1, 1, 2, 4, 1, 0, 3])
    return x, y, sizes


def model():
    torch.manual_seed(42)
    return TokenModel()


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def node_main(rank, world, port, rows, micro, out):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    research.BUCKET_BYTES = 64  # several buckets even for this small model
    net = model()
    x, y, sizes = inputs(rows)
    parallel = ResearchParallel(net, 1)
    loss = parallel.backward(x, y, 21, micro, step=3)
    grads = [p.grad.clone() for p in net.parameters()]
    scores = parallel.evaluate_batch(x, y, sizes, 21, micro)
    states = parallel.routing_states()
    parallel.restore_routing_states(states)
    restored = net.rng
    reference = net.head.weight.detach().clone()
    if rank == 1:  # a node whose update differs must take node 0's state
        with torch.no_grad():
            net.head.weight.add_(1.0)
    dist.all_reduce(torch.zeros(1))
    parallel.synchronize()
    torch.save(
        {
            "loss": loss,
            "grads": grads,
            "scores": scores,
            "states": states,
            "restored": restored,
            "resyncs": parallel.resyncs,
            "before": reference,
            "after": net.head.weight.detach().clone(),
            "total": parallel.total,
        },
        f"{out}/rank-{rank}.pt",
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("rows, micro", [(7, 1), (7, 2), (7, 3), (7, 10), (1, 1)])
def test_two_nodes_match_one_machine(tmp_path, rows, micro):
    multiprocessing.spawn(
        node_main, args=(2, free_port(), rows, micro, str(tmp_path)), nprocs=2
    )
    nodes = [torch.load(tmp_path / f"rank-{k}.pt") for k in range(2)]
    net = model()
    x, y, sizes = inputs(rows)
    single = ResearchParallel(net, 1)
    expected_loss = single.backward(x, y, 21, micro, step=3)
    expected_scores = single.evaluate_batch(x, y, sizes, 21, micro)
    for node in nodes:
        assert node["total"] == 2
        assert node["loss"] == pytest.approx(expected_loss, rel=1e-6)
        for actual, expected in zip(node["grads"], net.parameters()):
            torch.testing.assert_close(actual, expected.grad, rtol=1e-5, atol=1e-6)
        assert node["scores"][0] == pytest.approx(expected_scores[0], rel=1e-6)
        assert node["scores"][1:] == expected_scores[1:]
        # Global replica order; each node restores its own replica's state.
        assert [s["rng"] for s in node["states"]] == [3, 3]
        assert node["restored"] == 1003
    # Reduced values are identical on every node, not merely close.
    assert nodes[0]["loss"] == nodes[1]["loss"]
    for left, right in zip(nodes[0]["grads"], nodes[1]["grads"]):
        assert torch.equal(left, right)
    assert nodes[0]["scores"] == nodes[1]["scores"]
    for node in nodes:
        assert node["resyncs"] == 1
        assert torch.equal(node["after"], nodes[0]["before"])


def test_a_checkpoint_from_another_replica_count_is_refused():
    parallel = ResearchParallel(model(), 1)
    with pytest.raises(ValueError, match="replica count"):
        parallel.restore_routing_states([{"rng": 0}, {"rng": 0}])
