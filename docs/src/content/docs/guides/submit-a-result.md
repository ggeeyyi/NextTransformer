---
title: Submit a result
description: Audit a finished formal run, export it, and open a pull request.
---

This guide covers Dense. For MoE, follow the [MoE admission and export guide](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/README.md#submitting-a-method) for the [MoE leaderboard](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp).

A result is a finished **formal** run of an admitted configuration: 46,875 updates, all 189 validation events, all six
CORE-22 events, every segment trained with `--formal`. For a new method, submit these artifacts in the same pull request as the implementation.

## 1. Audit the run directory

```bash
python -m nexttransformer.audit_run runs/<method> --method <method>
```

The audit re-reads everything the trainer wrote and checks it against the protocol: the ledger (updates, tokens,
charged compute), the training events, the 189 validation events and their files, the six full CORE events with their
centering and aggregates, the independent final BPB, the final checkpoint, and the provenance of every segment
(identical fingerprints, formal mode, a registered batch layout). It writes `evidence/audits/<method>.json` and exits
non-zero if anything is missing; the file then lists the problems. The audit records the data layout but never host
names, paths or devices.

## 2. Export the record

```bash
python -m nexttransformer.export_run runs/<method> --method <method> --name "<display name>" \
    --audit evidence/audits/<method>.json
python -m nexttransformer.export_run --check
```

`export_run` writes `results/runs/<method>.json`. It refuses incomplete validation or CORE schedules, CORE tasks that do
not reproduce the aggregate, an unmatched cumulative audit, any research-mode segment, a different tokenizer or a
different protocol. Add `--note "..."` when the run needs a provenance note (for example, trained from a frozen source
revision).

Declare the CORE truncation setting and exact token limit in the result announcement and pull request. Use
`none` with no limit for full prompts, or `left` with the number supplied to `--core-max-tokens N`.
The exporter reads the measurements from all six CORE events. It writes a `coreContext` object containing
`trainingTokens`, `truncation`, `maxTokens` (null for full prompts), and `maxEvaluatedTokens`. For logs from the
original evaluator, pass `--legacy-core-2048` only after verifying the original 2,048-token left crop.
The leaderboard groups methods by the longest input actually scored.

## 3. Write run notes

Add `docs/src/content/docs/results/runs/<method>.md`: what was run, any resumes, failed attempts, and the limitations a
reader should know. Keep it factual and short. If the method has no model card yet, add `results/methods/<method>.json`;
for a registered method whose card has `"status": "pending"`, remove that field.

## 4. Refresh the generated tables

```bash
python docs/scripts/generate.py
```

This updates the leaderboard in the documentation and the README snapshots, and the model zoo when a card changed. CI
fails while they are out of date.

## 5. Open the pull request

Commit the files and open a pull request. CI runs only the data checks and the site build for a results-only
pull request, typically in one to two minutes. A maintainer reviews the audit and notes; after the merge the website
updates automatically.

Never edit a published record or audit. A second run of the same method uses `--file <name>` and its own audit.
