# NextTransformer contributor instructions

## Scientific contract

Follow [PROTOCOL.md](PROTOCOL.md), [PLAN.md](PLAN.md), and the [v2 work plan](docs/plans/V2.md). The controlled variables are admission requirements: same data/content/order, tokenizer/byte table, packing, 2K context, 49.152B input tokens, 1,048,576 tokens/update, 46,875 updates, single seed 42, Muon+AdamW recipe, 189 validation events, and six complete CORE-22 events.

SSSL is the historical `dense` API and the frozen 4.67239832544785 GFLOPs/input-token cost anchor; it belongs to the Others family. Full-attention Dense uses `dense_full`, the only baseline, with independent implementation, calibration, and training. Never assign SSSL scores to it or enable Dense deltas before complete accepted evidence.

Count actual forward, backward, auxiliary/teacher/gate, explicit recomputation, state operations, and optimizer costs. Fully computed masked candidates are charged. Preflight error is ≤1%, final cumulative error ≤2%, and rolling deviations over 3% are recorded. Parameters, inference costs, memory, speed, and GPU-hours are outcomes. Do not change thresholds or the data budget to admit a method.

Freeze method semantics, layout, required shell changes, paired controls, and legal calibration grid before formal runs. Model/source/software/kernel changes require new proofs, never edited historical hashes. Resume restores optimizer/schedule, all RNG, complete data cursor/buffers, stage, and cumulative cost. Pilots remain separately accounted and cannot supply weights to a from-scratch run.

## Arena entrants

If you were given [docs/arena/TASK.md](docs/arena/TASK.md), you are entering the auto-research arena; [ARENA.md](ARENA.md) binds you in addition to this file. Add a method only, as a plugin under `nexttransformer/methods/<method>/` ([plugin guide](docs/guide/PLUGINS.md)); never edit the training contract, evaluation, data pipeline, optimizer, cost ledger, trainer, `backbone.py`, `nanochat/`, the research environment `nextt_gym/`, or other methods' files and evidence (`python -m nexttransformer.arena --scope origin/main` checks this). Select candidates with training loss or the dev split `shard_06541.parquet`, never with the validation shard or CORE-22. Record every candidate, formal launch, and human intervention truthfully in the disclosure card.

## Code layout

- Built-in architectures live in `nexttransformer/models/` (registered in `registry.py`, implemented in family modules); new methods are plugins under `nexttransformer/methods/` that change no shared file ([plugin guide](docs/guide/PLUGINS.md)). Protocol, admission, and the cost ledger live in `nexttransformer/protocol/`. Cost proofs fingerprint only the code a method runs: audit a new or changed method with `protocol.calibrate --methods <method>`, and keep `python -m nexttransformer.protocol.fingerprint` passing, which fails when a published method's code changed. The one trainer is `nexttransformer.train` (8/16/32 ranks, Ouro gate-only stage, `--sparse-backend`); evaluation is `nexttransformer.evaluate`.
- `nanochat/` is the upstream runtime. Six upstream files are byte-pinned by the prepared-resource manifests; never edit `nexttransformer/models/backbone.py` or `nanochat/{dataloader,loss_eval,core_eval,base_eval,tok_train}.py`.
- Checkpoints store tensors by name and a plain-dict configuration. Keep parameter names and configuration fields stable, and check `load_model` on an existing checkpoint after model changes.
- Published results are files in `results/` written by `nexttransformer.export_run`. The website builds from them and must not query a tracking service.
- Keep documentation, comments, and docstrings in English. Disposable checks and logs belong in `.agent-work/<task>/`.
- Keep runtime paths, environments, data, caches, W&B accounts, and credentials in site settings, never in scientific configurations.
- Distinguish planned, implemented, bounded-validated, running, and complete/matched states in PLAN.md and the experiment record.

## Execution scope

A repository change authorizes code/documentation work and bounded validation; it does not launch 49.152B runs, replace frozen environments, change repository visibility, or rewrite historical results. Bounded GPU tests may use an isolated checkout after checking occupancy; do not disturb other jobs, data, checkpoints, or caches. Model or common-runtime changes require relevant regressions and a source-aware cost re-audit. Do not bypass admission to make a test pass.

## Git integration

Only the maintainer (@Selen-Suyue) updates `main`. Agents acting for the maintainer commit to `main` after verifying the remote state; agents acting for collaborators work on `<handle>/<topic>` branches and open pull requests. Never force-push or rewrite shared history. Before merging collaborator work, inspect dependencies and conflicts; unresolved semantic conflicts require review. Preserve collaborator branches unless asked to remove them.
