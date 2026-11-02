# Contributing

NextTransformer compares architectures under one fixed protocol. Every change arrives as a pull request that the
maintainer reviews and merges. Read [PROTOCOL.md](PROTOCOL.md) and [PLAN.md](PLAN.md) first.

This file owns the contribution workflow. Scientific rules are defined in [PROTOCOL.md](PROTOCOL.md);
[ARENA.md](ARENA.md) adds Arena entry and disclosure rules. The [run schema](results/README.md) and
[Arena card schema](results/arena/README.md) describe stored fields. [AGENTS.md](AGENTS.md) and the
[Arena task](docs/arena/TASK.md) retain self-contained constraints for agents.

## Branch policy

- **Only the maintainer (@Selen-Suyue) updates `main`**, including agents acting for that account. Collaborators, and
  agents working for collaborators, never push to, merge into, force-push, or delete `main`.
- Work on a branch named `<handle>/<topic>` (for example `geyi/gdn-hybrids`) and open a pull request into `main`.
- Bring your branch up to date by merging or rebasing it on `main`. Do not rewrite `main`'s history.
- GitHub does not enforce this rule for this repository; it is binding all the same.

## What a pull request contains

| Contribution | Files |
|---|---|
| A new method | A plugin `nexttransformer/methods/<method>/` ([plugin guide](docs/guide/PLUGINS.md)), a calibrated `configs/<family>/<method>.json`, its proof under `evidence/costs/`, tests, a model card `results/methods/<id>.json`, and an entry in [docs/methods/SOURCES.md](docs/methods/SOURCES.md) |
| A finished run | `results/runs/<model>.json` written by `export_run`, the run's cumulative audit under `evidence/checks/`, a record in [docs/records/EXPERIMENTS.md](docs/records/EXPERIMENTS.md), and the status line in [PLAN.md](PLAN.md) |
| A fix or refactor | Code and tests. A change to the code a method runs needs a new cost proof for that method; `python -m nexttransformer.protocol.fingerprint` lists them, and a change to a published method needs maintainer review |
| An arena entry | A new method, and later its finished run, as above, each with the entry's disclosure card `results/arena/<entry>.json`. The admission pull request comes before the formal run; see [ARENA.md](ARENA.md) and the [task statement](docs/arena/TASK.md) |

## Adding a method

1. Create the plugin `nexttransformer/methods/<method>/`: a `method.json` declaration and the hooks that build the
   method ([plugin guide](docs/guide/PLUGINS.md)). The registry discovers it; shared files stay unchanged. Reuse the
   shared backbone, data, evaluation, and optimizer.
2. Keep parameter names stable once a run exists: checkpoints store tensors by name and rebuild the model from its saved
   configuration.
3. Calibrate the configuration to the SSSL cost anchor (±1%):
   ```bash
   python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/<task>/cost-cache \
       --output configs --results evidence/costs/<new-proof-set>
   ```
4. Add numerical, gradient, causality, protocol, and resume tests, and a model card in `results/methods/`.
5. Run `python -m nexttransformer.protocol.fingerprint` and `python -m nexttransformer.arena --scope origin/main`. A
   method submission leaves every published method matching its proof and adds only its own files; proofs fingerprint
   only the code each method runs, so a plugin never affects another method.

## Submitting a run

```bash
python -m nexttransformer.train --formal --config configs/<family>/<method>.json --output RUN ...   # admitted
python -m nexttransformer.export_run RUN --method <method> --name "<display name>" --hardware "8 × H20" \
    --audit evidence/checks/<run>-formal.json --commit <training-code SHA>
python -m nexttransformer.export_run --check
```

`export_run` refuses incomplete validation or CORE schedules, CORE tasks that do not reproduce the aggregate, an
unmatched cumulative audit, a segment trained without `--formal`, a different tokenizer, or a different protocol. The observatory rebuilds from `main` after
the merge; no tracking service or database is involved.

No container is required: train and evaluate in the environment that
[`launch/setup_environment.sh`](launch/setup_environment.sh) installs from `uv.lock`, and submit results and code by
pull request. This applies to arena entries too; the fixed GPU image in `environments/eval/` serves only the arena gym.

### Frozen historical exports

DeltaNet and GLA retain their original source fingerprints and all-backends-replay accounting. Use the
frozen exporter for those existing runs, with their accepted cumulative audit:

```bash
python -m nexttransformer.export_frozen RUN_DIR \
  --audit evidence/checks/xiangyue-a01-a03-20261027/<run>-formal.json
```

This compatibility path does not admit a new-source run or replace its required formal export. Preserve the
original identity and limitations in [record 039](docs/records/EXPERIMENTS.md#record-039).

## Protocol rules

Do not change the data budget, global sample order, optimizer schedule, seed, or evaluation points to help one method.
Do not reuse SSSL scores as full-attention Dense. Hardware and local paths belong in site settings, not scientific
configurations. Record failed attempts; publish only complete runs with their cumulative compute evidence.

## Development checks

```bash
uv sync --frozen --extra cpu && source .venv/bin/activate
CUDA_VISIBLE_DEVICES= NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m nexttransformer.export_run --check
python -m nexttransformer.arena --check                # and, for an arena entry, --scope origin/main
cd website && npm ci && node --experimental-strip-types --test tests/*.test.mjs && npm run build
```

CUDA-only tests are skipped on CPU. A short GPU check is not evidence of full-training convergence.

## File policy

Track source, configurations, required tests, documentation, and curated evidence. Put disposable scripts and logs in
`.agent-work/<task>/` (ignored). Never commit credentials, datasets, checkpoints, caches, or private absolute paths.
Keep documentation and code comments in English. Preserve attribution and licenses, and never rewrite frozen evidence.
