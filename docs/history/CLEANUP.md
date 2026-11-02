# Historical cleanup: 2026-10-13

> Historical record. Paths, module names, and commands are those in use at the time; the 2026-10-24
> reorganization maps them as described in [the repository guide](../guide/REPOSITORY.md).

This record documents the removal of 57 obsolete GPT-2/native experiment files from the pre-cleanup base `0ea4cb1`. It does not describe the 2026-10-23 model relocation; see [repository layout](../guide/REPOSITORY.md) for that change. Frozen jobs, datasets, checkpoints, cloud tracking history, and unrelated projects were unchanged.

The following paths are a historical deletion inventory, not current links.

## Former GPT-2 models, configurations, and entry points

- `configs/circle.json`
- `configs/circle_fixed.json`
- `configs/circle_g2.json`
- `configs/circle_g4.json`
- `configs/circle_g8.json`
- `configs/circle_pg.json`
- `configs/circle_random.json`
- `configs/dense.json`
- `configs/hardware.json`
- `configs/loopformer.json`
- `configs/mor.json`
- `nanochat/experiment/budget.py`
- `nanochat/experiment/circle.py`
- `nanochat/experiment/config.py`
- `nanochat/experiment/model.py`
- `runs/benchmark_batches.py`
- `runs/train_a800.sh`
- `runs/train_common.sh`
- `runs/train_h200.sh`
- `scripts/gpt2_benchmark.py`
- `scripts/gpt2_budget_pilot.py`
- `scripts/gpt2_compare.py`
- `scripts/gpt2_eval.py`
- `scripts/gpt2_prepare.py`
- `scripts/gpt2_profile.py`
- `scripts/gpt2_train.py`
- `site.example.json`

## Tests specific to removed implementations

- `tests/circle_budget_distributed.py`
- `tests/circle_capacity.py`
- `tests/circle_distributed.py`
- `tests/experiment_distributed.py`
- `tests/test_circle.py`
- `tests/test_circle_budget.py`
- `tests/test_comparison.py`
- `tests/test_experiment.py`
- `tests/test_tracking_metrics.py`

## Old protocol documents and one-off reports

- `docs/BATCH_TUNING.md`
- `docs/CIRCLE.md`
- `docs/CIRCLE_BUDGET.md`
- `docs/CIRCLE_NATIVE_ANALYSIS.md`
- `docs/CIRCLE_VALIDATION.md`
- `docs/CLEANUP_PROPOSAL.md`
- `docs/LEGACY_GPT2_PLAN.md`
- `docs/LEGACY_GPT2_README.md`
- `docs/NATIVE_BATCH_TUNING.md`
- `docs/VALIDATION.md`
- `docs/batch_tuning_evidence.json`
- `docs/circle_budget_evidence.json`
- `docs/circle_budget_full.png`
- `docs/circle_budget_gains.png`
- `docs/circle_validation_evidence.json`
- `docs/native_batch_evidence.json`
- `docs/native_circle_enhanced_evidence.json`
- `docs/native_functional_evidence.json`
- `docs/portability_evidence.json`
- `docs/validation_evidence.json`
- `docs/wandb_metrics_evidence.json`

## Internal changes and preserved behavior

- Native configuration and time conditioning became independent of the old GPT-2 inheritance tree.
- Removed full-candidate Circle-ST, old linear/phase halters, and unused PI/REINFORCE branches; retained Sparse Circle and its core regressions.
- Moved useful CORE preparation and data-manifest checks to `nanochat/native_experiment/resources.py`. Retained shared path settings, W&B, distributed runtime, statistics, and Muon/AdamW adapters; removed old token pre-encoding and GPT-2 tokenizer/checkpoint wrappers.
- Preserved upstream equivalence, native causality/routing/budget/packing/BPB/exact-resume, settings/tracking, preflight, and train→resume→CORE checks. Removed tests served removed code only.
- Preserved upstream GPT, tokenizer, CORE/BPB, chat/SFT/RL, upstream tests, licenses, and the original README.
- Kept frozen source and training artifacts in their runtime archives. Temporary acceptance checkpoints and duplicate logs were removed after their compact evidence was recorded.

Historical native entry points became `scripts.native_*`, `runs/train_native_*`, and `configs/native/`. The expanded project now also provides the reference workflow. Hardware tables and one-off evidence were consolidated into [native acceptance](NATIVE_VALIDATION.md), with scope and limitations retained.

## Validation and recovery

The cleanup passed CPU regressions, eight-A800 deepest-path/full-update checks, fresh-process recovery for all four native models, BPB, small-sample CORE-22, launcher dry-runs, pinned-source checks, lock checks, and documentation link checks. Exact results are in the [acceptance record](NATIVE_VALIDATION.md).

Inspect removed files with `git show 0ea4cb1:<path>`. Reproduce old checkpoints in a checkout of their original commit; never relax current source or scientific-config checks to force old checkpoints into new code.
