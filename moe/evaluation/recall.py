"""FDA and SWDE: in-context information extraction from long documents (Based, arXiv 2402.18668; JRT, 2407.05483).

Exactly lm-evaluation-harness's `fda` and `swde` tasks: the prompt is the dataset's `text` (a document followed
by "{key}:"), the model generates greedily until a newline or 48 tokens, and a prediction counts when it contains
the gold `value` (case-insensitive). FDA documents run to about 2K tokens (dolma2: median 1,955, max 2,185),
SWDE to about 1K. Both are standard recall probes in architecture comparisons (Based, JRT, Gated DeltaNet,
MesaNet). The datasets are pinned by revision and SHA-256 and fetched by moe.data prepare.
"""
import re

import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from moe.evaluation.generate import greedy
from moe.tokenizer import EOT, Dolma2Tokenizer

DATASETS = {
    "fda": dict(repo="hazyresearch/based-fda", revision="42569d301e12fbcf8d5a69e04e892aa013e20314",
                file="data/validation-00000-of-00001.parquet",
                sha256="8be771261b43c826e39525697ca6b2d84102e5034066781221f171504c1a2542", rows=1102),
    "swde": dict(repo="hazyresearch/based-swde", revision="27f0223028a5d1ce174011051d3d22a5fafda830",
                 file="data/validation-00000-of-00001.parquet",
                 sha256="5a23577a07a8318667d0f1d353a46dda7716c7d60af6a1c8e676b2c56397980a", rows=1111),
}
MAX_NEW_TOKENS = 48


def contains(prediction, label):
    return bool(re.search(re.escape(label), prediction, re.IGNORECASE))


def rows(path):
    table = pq.read_table(path, columns=["text", "value"])
    return [(t.strip(), v.strip()) for t, v in zip(table.column("text").to_pylist(), table.column("value").to_pylist())]


@torch.no_grad()
def evaluate(model, paths, device, *, limit=None, batch_size=8):
    """paths: {task: parquet path}. Returns accuracy per task."""
    tokenizer = Dolma2Tokenizer()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    result = {}
    for task, path in paths.items():
        data = rows(path)[:limit]
        mine = list(range(rank, len(data), world))
        prompts = [[EOT] + tokenizer.encode(data[i][0]) for i in mine]

        def stop(j, generated):   # generate_until "\n"
            return "\n" in tokenizer.decode(generated)
        outputs = greedy(model, prompts, device, max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size, done=stop)
        correct = torch.zeros(len(data), device=device)
        for i, ids in zip(mine, outputs):
            prediction = tokenizer.decode([t for t in ids if t != EOT]).split("\n")[0]
            correct[i] = float(contains(prediction, data[i][1]))
        if world > 1:
            dist.all_reduce(correct)
        result[task] = correct.mean().item()
    return result
