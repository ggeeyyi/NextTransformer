## What this changes

<!-- A result, a new method, documentation, or a fix. -->

Closes #<!-- The issue this pull request completes: a T3 claim, a method proposal or a bug report. GitHub closes it on merge only when the line reads "Closes #N". If the issue covers work this pull request does not deliver, such as other cells in a multi-cell claim, write "Part of #N" instead. -->

<!-- For a leaderboard result, fill in the CORE settings from the exported run record. -->
CORE truncation (none / left):
CORE token limit (none / number):

## Checklist

<!-- For a new method, complete both the New method and Result items and link the proposal issue. For Arena entries, link the preregistration in that issue and include the completed disclosure card. -->
- [ ] The training contract is unchanged: data, tokenizer, token budget, batch, schedule, seed and evaluation points.
- [ ] **Result:** audit from `python -m nexttransformer.audit_run` in `evidence/audits/`, record written by `python -m nexttransformer.export_run`, a run-notes page, and tables refreshed with `python docs/scripts/generate.py`.
- [ ] **New method:** plugin under `nexttransformer/methods/<method>/`, calibrated configuration and cost proof, tests, model card, and a source-notes entry.
- [ ] **Shared code:** agreed in an issue first; `python -m nexttransformer.checks.golden --base origin/main` reports no difference, or this pull request explains why one is intended.
- [ ] `python -m pytest -q -n auto` passes locally.

## AI assistance

- [ ] An AI agent prepared or substantially wrote this pull request. Agent and model: <!-- e.g. "Claude Code, model X" -->
