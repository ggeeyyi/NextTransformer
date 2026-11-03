"""Run nexttransformer.train with its BPB evaluations on the arena's dev split instead of the validation shard.

Operator harness for the bounded Stage 3 checks (debug pod, G2b): the validation shard is reserved for formal runs,
and a smoke run of the formal trainer would still evaluate it at its first and last step. The run is the formal
trainer's own: the same start (runtime bootstrap and per-rank kernel caches, as `python -m nexttransformer.train`),
the same data, resource and inventory checks on the full dataset, the same training. Only the stream an evaluation
reads changes, as the gym evaluates its runs: DEV_DIR holds links [shard_06541 (dev), the pinned validation shard]
(nextt_gym.trainer.dev_directory), and BestFitStream's "train" split reads every file but the last, so it reads the
dev shard alone. The trainer still labels those events "val"; in these runs they are dev-split values. Refuses
anything but a smoke run without CORE.

    torchrun ... formal_dev.py DEV_DIR <nexttransformer.train arguments>
"""
import os
import sys

from nexttransformer.runtime.site import bootstrap_runtime

DEV_DIR, ARGS = sys.argv[1], sys.argv[2:]
# nexttransformer/train.py under `if __name__ == "__main__"`, before torch is imported: the site's paths and caches,
# then one Triton and Inductor cache directory per local rank (train.rank_kernel_caches).
bootstrap_runtime(ARGS)
if "LOCAL_RANK" in os.environ:
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if os.environ.get(key): os.environ[key] = f"{os.environ[key].rstrip('/')}/rank{os.environ['LOCAL_RANK']}"

from nexttransformer import train                                    # noqa: E402
from nexttransformer.evaluation import formal as evaluation          # noqa: E402

options = train.parse(ARGS)
if options.formal or not options.smoke_steps > 0 or options.smoke_core_per_task != 0:
    raise SystemExit("formal_dev.py runs smoke runs only (--smoke-steps > 0), without CORE and without --formal")

stream = evaluation.BestFitStream


def dev_stream(data_dir, tokenizer, split, *args, **kwargs):
    """The evaluation's stream: the dev shard, after evaluate_bpb has checked the full dataset in data_dir."""
    if split != "val": raise RuntimeError(f"formal_dev.py: unexpected evaluation split {split!r}")
    return stream(DEV_DIR, tokenizer, "train", *args, **kwargs)


def no_core(*args, **kwargs):
    raise RuntimeError("formal_dev.py: CORE is reserved for formal runs")


evaluation.BestFitStream = dev_stream
train.evaluate_core = no_core
train.main(ARGS)
