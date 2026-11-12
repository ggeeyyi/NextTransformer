# Scaling-stage runs

`nexttransformer.scaling.train` pretrains the dense backbone with a scaling-stage configuration such as
`configs/scaling/dense_200b.json` (4,096-token context, about 200B tokens). It is separate from the frozen 49B comparison
protocol and its results are not leaderboard entries.

It is built for nodes that do not share a filesystem. Each node keeps its own copy of the data, resources and
checkpoints; the trainer checks before training that every node has the same source files, data inventory, tokenizer
and configuration.

## Per-node preparation

1. Check out the same commit on every node.
2. Download the pinned ClimbMix revision and run `python -m nexttransformer.prepare` on every node, with a tokenizer
   trained once and copied to the others ([data guide](../../docs/src/content/docs/getting-started/data.md)).
3. Write a site file (`site.local.json`, ignored by git) with `runtime_dir`, `resources`, `data_dir` and caches on local
   storage. Set `wandb_mode` to `online` to follow the run on W&B.
4. Put site-specific exports in a file outside the repository and point `SCALING_ENV` at it: network settings for
   NCCL, and `WANDB_API_KEY` on node 0 when W&B is online. Keep it readable only by you.

Every launcher reads `NNODES`, `NODE_RANK` (0 to `NNODES-1`, fixed per node across restarts), `MASTER_ADDR` (node 0)
and `MASTER_PORT`. Optional: `GPUS` (8), `MICRO_BATCH` (2), `OUTPUT_DIR`.

## Checks before a run

```bash
# Plan: parameters, steps, accumulation, learning rates, FLOPs.
python -m nexttransformer.scaling.train --config configs/scaling/dense_200b.json --site-config site.local.json --dry-run
# On every node at once: throughput and memory for micro-batch, compile and global-batch settings.
launch/scaling/benchmark.sh configs/scaling/dense_200b.json --site-config site.local.json
# On every node at once: a bounded run that checkpoints, stops, and resumes.
OUTPUT_DIR=$RUNS/smoke launch/scaling/node.sh configs/scaling/dense_200b.json --site-config site.local.json \
  --smoke-steps 6 --stop-after 3
OUTPUT_DIR=$RUNS/smoke launch/scaling/node.sh configs/scaling/dense_200b.json --site-config site.local.json \
  --smoke-steps 6 --resume auto
```

A resumed run replays the same data and the same updates as a continuous one; on GPUs, attention backward kernels
that accumulate atomically can make the two differ in the last bits.

Memory: without recomputation, each 4,096-token sequence of `dense_200b` holds about 37 GB of activations on top of
about 39 GB of weights, gradients and BF16 weight copies (measured on 80 GB GPUs). `--recompute-layers N` recomputes
the activations of the first N blocks during backward and frees most of their memory, at the cost of roughly one
extra forward pass through those blocks.

## The run

On every node, start the watchdog and the checkpoint mirror:

```bash
OUTPUT_DIR=$RUNS/dense_200b launch/scaling/watchdog.sh configs/scaling/dense_200b.json --site-config site.local.json \
  --wandb-metrics full
OUTPUT_DIR=$RUNS/dense_200b S3_PREFIX=s3://bucket/prefix/dense_200b launch/scaling/s3_sync.sh
```

- Every launch resumes from the newest checkpoint complete on every node (`--resume auto`). A rolling checkpoint is
  written every 1,000 updates; each node keeps the newest three and every milestone (every 10,000 updates and the last).
- `touch $OUTPUT_DIR/STOP` on node 0 checkpoints and stops every node at the end of the current update.
- A non-finite loss or gradient stops the run before the optimizer step and without a checkpoint. The watchdog does
  not restart it, because a restart would replay the same update; the newest checkpoint is still intact.
- Each node writes `status.node<k>.json`; node 0 writes `metrics.jsonl`, the validation and CORE records,
  `run_status.json` and the W&B run.
- Losing a node stops the run: optimizer shards and data cursors belong to a fixed node and rank layout, so the run
  resumes only with the same number of nodes and GPUs per node.

## Publishing a finished run

```bash
python -m nexttransformer.scaling.export $RUNS/dense_200b --config configs/scaling/dense_200b.json --id dense_200b \
  --name 'Dense · 200B tokens · 4K' --family dense --wandb <W&B run URL> --checkpoint <Hugging Face model URL>
python -m nexttransformer.scaling.export --check
```

Run the export on node 0, which holds the event history. It writes `results/scaling/runs/dense_200b.json` and refuses
an incomplete run. The final milestone's `checkpoint_<steps>/node_000/model.pt` holds the weights and both
configurations; publish that file, not the per-rank optimizer shards.
