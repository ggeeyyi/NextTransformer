---
title: Contribute to scaling
description: What the community can build for the scaling stage, and how to train a ladder cell, add a family or register a forecast.
---

Every contribution is a file in the repository, added by a pull request that a maintainer reviews. The observatory's
[Scaling page](https://interactive-active-ai.github.io/NextTransformer/#scaling) shows the ladder and which cells are
open.

| Contribution | Compute | What you submit |
|---|---|---|
| [Train a ladder cell](#train-a-ladder-cell) | About 10²⁰ training FLOPs at T3, 2.3 × 10²⁰ at T4 | The cell's configuration, its record and a checkpoint |
| [Add an architecture family](#add-an-architecture-family) | Configurations and cost checks; the family's cells are then trained like any other | A method, a size rule and the family's cells |
| [Register a forecast](#register-a-forecast) | None | A forecast file, before the C200 runs start |
| [Evaluate released checkpoints](#evaluate-released-checkpoints) | Inference | Evaluation results for published checkpoints |
| [Improve the fits](#improve-the-fits) | A laptop | Code or an analysis |
| [Reproduce a cell](#reproduce-a-cell) | One cell, or none to re-run checks | A second-seed record, or a check report |

## Train a ladder cell

The example below trains `dense-T3-L24`: 24 layers, width 1,536, 16,153,313,280 tokens in 15,405 updates of 1,048,576
tokens. Every cell's size and token count are in `results/scaling/ladder.json`.

**1. Claim it.** Open a [scaling cell claim](https://github.com/Interactive-Active-AI/NextTransformer/issues/new?template=scaling_cell.yml)
with the cell id and your compute plan. A maintainer marks the cell claimed; a claim without progress for two weeks is
released. Cells marked as waiting for the recipe gate cannot be claimed yet.

**2. Prepare.** Check out `main`, prepare the pinned ClimbMix revision and the tokenizer as in the
[data guide](../../getting-started/data/), and write a site file as described in `launch/scaling/README.md`. On several
nodes without a shared filesystem, every node needs its own copy.

**3. Configure.** Copy the reference configuration and change only the size, the token count, the batch and the
protocol name:

```bash
cp configs/scaling/dense_200b.json configs/scaling/dense-T3-L24.json
# model: dense_layers 24, width 1536, heads 12, mlp_width 6144
# train: protocol "nextt-scale-4k-v1", total_tokens 16153313280, global_batch_tokens 1048576
python -m nexttransformer.scaling.train --config configs/scaling/dense-T3-L24.json --dry-run
```

The dry run prints the plan: check that the steps match the cell and that the training FLOPs are within 1% of the tier
budget.

**4. Train.** Start the watchdog on every node; it resumes from the newest checkpoint after any interruption.

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 OUTPUT_DIR=$RUNS/dense-T3-L24 \
  launch/scaling/watchdog.sh configs/scaling/dense-T3-L24.json --site-config site.local.json --wandb-metrics full
```

Any GPUs that can train the recipe are fine; the budget is counted in FLOPs. `MICRO_BATCH` sets sequences per GPU and
changes only speed and memory.

**5. Export.** On node 0, which holds the event history:

```bash
python -m nexttransformer.scaling.export $RUNS/dense-T3-L24 --config configs/scaling/dense-T3-L24.json \
  --id dense-T3-L24 --name 'Dense · 24 layers · T3' --family dense \
  --wandb <W&B run URL> --checkpoint <Hugging Face model URL>
python -m nexttransformer.scaling.export --check
```

The export refuses an incomplete run. Publish the final checkpoint's `checkpoint_<steps>/node_000/model.pt` before
you export: the record is immutable, so its links must be final.

**6. Open a pull request** with `configs/scaling/dense-T3-L24.json`, `results/scaling/runs/dense-T3-L24.json`, and the
cell in `results/scaling/ladder.json` with `"record": "dense-T3-L24"` and your claim. Run
`python -m nexttransformer.checks.public` first: records, configurations and documentation must not name GPU models,
hosts, paths or personal e-mail addresses. After the merge the cell shows as recorded and the run appears in the 3D plot.

## Add an architecture family

1. Register the method under the shared protocol first, as in the [plugin guide](../../guides/plugins/).
2. Propose a size rule in an issue: one size knob, the number of layers, and how every other dimension follows from it.
3. Extend `nexttransformer/scaling/` to build the family; the scaling trainer builds the Dense backbone today.
4. Pass the [stability gate](../protocol/#families) at T3 and at the C200 scale.
5. Add the family and its ten T3 and T4 cells to `results/scaling/ladder.json`, sized with the
   [budget accounting](../protocol/#budget). Its cells then open for claims like any other.

## Register a forecast

Predict a family's compute-optimal size at C200 and its final validation BPB and CORE, with 90% intervals, from the
published records. Add `evidence/forecasts/<family>-C200-<author>.json` in the
[documented format](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/forecasts). It must be
merged before the family's first C200 run starts; the ladder shows that cell as waiting for the forecast until then.
Forecasts are never edited; each one is scored against the measured run.

## Evaluate released checkpoints

Every record links its checkpoint. Long-context retrieval, RAG-style question answering and further downstream tasks
on these checkpoints are welcome. Add `results/scaling/evals/<record>-<benchmark>.json` with the benchmark and its
version, the settings, the scores and a link to the code you ran.

## Improve the fits

The fitting method is in the [scaling protocol](../protocol/#fitting). Better estimators, uncertainty estimates or
checks for the compute-optimal frontier can be proposed as issues, or as code under `nexttransformer/scaling/` with
tests.

## Reproduce a cell

Retrain a recorded cell with another seed: claim it as a second-seed run, set `train.seed` in the configuration, and
export it with the id `<cell>-seed<k>`. Re-running `python -m nexttransformer.scaling.export --check` and comparing a
record with its W&B run needs no GPUs.
