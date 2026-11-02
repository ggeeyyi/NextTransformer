## What this changes

<!-- A new method, a finished run, a fix, or documentation. -->

## Checklist

- [ ] Branch `<handle>/<topic>`; this PR targets `main` and the maintainer merges it.
- [ ] Protocol unchanged: data, token budget, global batch, schedule, seed, and evaluation points.
- [ ] New or changed fingerprinted source: a new proof set under `evidence/costs/`, and every configuration admitted.
- [ ] New method: registry entry, calibrated configuration, tests, model card in `results/methods/`, SOURCES.md entry.
- [ ] Finished run: record written by `nexttransformer.export_run`, audit under `evidence/checks/`, EXPERIMENTS.md record.
- [ ] Arena entry: disclosure card in `results/arena/`; `python -m nexttransformer.arena --check` and `--scope origin/main` pass.
- [ ] `python -m pytest -q` and `python -m nexttransformer.export_run --check` pass; website tests pass if it changed.
- [ ] Parameter names and configuration fields unchanged for methods with trained checkpoints.
