# Arena task: a better language model for a fixed training budget

This is the task statement for the NextTransformer auto-research arena. Give it to an AI agent, with or without a
research framework, together with a checkout of this repository. It is self-contained; the binding rules are in
[ARENA.md](../../ARENA.md) and [PROTOCOL.md](../../PROTOCOL.md).

## Goal

Design **one** language-model architecture that, trained from scratch under the frozen training contract, reaches a
lower validation bits-per-byte (BPB) or a higher CORE-22 score at update 46,875 than the models already on the
leaderboard. Both metrics are reported; the leaderboard ranks by either. The current results are in
[README.md](../../README.md#results) and [results/runs/](../../results/runs).

## What is fixed

| Control | Value |
|---|---|
| Data | ClimbMix at a pinned revision; one tokenizer (32,768 tokens); 2,048-token context; one global sample order |
| Budget | 49.152B input tokens · 1,048,576 tokens per update · 46,875 updates · seed 42 |
| Compute | Training cost within ±1% of 4.67239832544785 GFLOPs per input token, by the repository's operator ledger, including backward and optimizer work |
| Optimizer | Muon for block matrices, AdamW groups for embeddings, head, and scalars; fixed learning rates and schedule |
| Evaluation | Validation BPB every 250 updates; full CORE-22 at 8K, 16K, 24K, 32K, 40K, and 46,875 updates |

Parameters, memory, and speed are not controlled; they are reported. The budget is spent on training compute, so a
model may trade depth, width, recurrence, sparsity, or state for one another, as long as the ledger agrees.

## What you may change

Add your method as a plugin under `nexttransformer/methods/<method>/` ([plugin guide](../guide/PLUGINS.md)), with a
calibrated configuration, its proof set, tests, and a model card. Do not edit the data pipeline, evaluation, optimizer,
protocol, cost ledger, trainer, shared model files, `nanochat/`, `nextt_gym/`, or other methods' files and evidence. [ARENA.md](../../ARENA.md#what-an-entry-may-change)
has the full list; `python -m nexttransformer.arena --scope origin/main` checks your branch.

## How to work

```bash
uv sync --frozen --extra cpu && source .venv/bin/activate      # or --extra gpu on a CUDA machine
export NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=
python -m pytest tests/test_models_api.py -q                     # build and reload small models
```

- Read the [plugin guide](../guide/PLUGINS.md), [nexttransformer/models/README.md](../../nexttransformer/models/README.md)
  and the test plugins in `tests/plugins/`. `build_model(debug_config("<method>"))` builds any registered method,
  plugins included, at debug size on a CPU.
- Calibrate your configuration to the cost anchor:
  `python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/<task>/cost-cache --output configs --results evidence/costs/<proof-set>`.
  Admission fails if your training cost differs by more than 1% or the ledger meets an operator it cannot charge.
- `python -m pytest -q` must pass. The registry-wide tests check gradients and that changing future tokens never
  changes earlier logits.
- **Research runs.** The research environment in [`nextt_gym/`](../../nextt_gym/README.md) checks a candidate's
  cost, trains it for the first 512 updates of the formal schedule or at full length, scores it on the dev split, and
  meters FLOPs; connect over MCP or call `nextt_gym.adapter.evaluate`. It keeps validation and CORE-22 sealed. Without
  it, `nexttransformer.train --smoke-steps N` trains the full-size model for N updates on the formal data stream (use
  its training loss), or write your own loop around `build_model` under `.agent-work/`, which is not committed. A
  design found in the gym still enters through the two pull requests below.

## Rules for the research

1. Any scale, any compute, any number of experiments, as long as you record them.
2. Choose among candidates with training loss or with BPB on the **dev split**, `shard_06541.parquet`.
3. Do not use the validation shard `shard_06542.parquet` or CORE-22 to choose anything. Their numbers in smoke runs
   are not a selection signal.
4. Literature and the internet are allowed; say whether you used them. If your method resembles published work, name
   it and explain the difference.
5. The formal run starts from scratch. Never load research weights into it.

## How to submit

1. **Admission pull request:** your plugin, admitted configuration, proof set, tests, model card,
   and a disclosure card `results/arena/<entry>.json` with `"status": "admitted"`. Start from
   [card.template.json](card.template.json); the fields are explained in
   [results/arena/README.md](../../results/arena/README.md).
2. **Formal run:** train the admitted configuration with `nexttransformer.train --formal` (8, 16, or 32 GPUs), then
   export it with `python -m nexttransformer.export_run`.
3. **Result pull request:** the run record, its audit, an `EXPERIMENTS.md` record, and the card updated to
   `"status": "complete"` with links to the final checkpoint and the training logs.

Run `python -m nexttransformer.arena --check` and `python -m nexttransformer.export_run --check` before each pull
request. Report honestly: failed launches, every candidate you trained, and how much people helped. An entry that
loses to the baselines but is fully disclosed is a useful result.
