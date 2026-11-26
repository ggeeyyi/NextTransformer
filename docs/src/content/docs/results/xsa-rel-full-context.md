---
title: XSA and REL with full CORE prompts
description: Supplementary full-prompt CORE-22 measurements of the published final checkpoints.
---

The published XSA and Inkling REL runs already completed the fixed 49.152B-token
training protocol, 189 validation events and six full CORE-22 events. Their
historical evaluator left-cropped inputs to 2,048 tokens. This supplementary
measurement evaluates each same final checkpoint once with the current default
full prompts, retaining BOS and all few-shot examples. It does not retrain,
select a checkpoint or replace the original training records and audits.

| Method | Historical CORE ×100 | Full-prompt CORE ×100 | Change | Historical SQuAD % | Full-prompt SQuAD % |
|---|---:|---:|---:|---:|---:|
| XSA | 31.7346 | 31.8072 | +0.0726 | 18.0416 | 18.7417 |
| Inkling REL | 33.1733 | 33.3340 | +0.1607 | 40.1325 | 45.0899 |

Training context remains 2,048 tokens. Both supplementary evaluations declare
`truncation: none`, `maxTokens: null`, and the observed maximum of 5,835 tokens.
All 22 tasks use the pinned tokenizer, prompts, answer scoring and centering.
The trained attention policies are also retained: XSA uses the shell's
2,048-token backward attention window; REL attends to all preceding keys, with
zero positional bias at distances outside its bank. A full input prompt does
not change either model's attention policy.
BPB is unchanged and was not recomputed because the context-policy update only
affects CORE. Only SQuAD, BoolQ and Language ID contain over-2,048-token inputs;
per-task comparisons are available in the validation artifact.

Independent 16-rank controls used this same evaluation entry point with the
historical 2,048-token crop. Both controls reproduced all 22 original task
scores and their aggregate CORE exactly. All 19 tasks without overlength
prompts also reproduce exactly in the full-prompt measurements.

Inkling's original REL definition uses zero relative bias at distances outside
the trained bank, retaining ordinary causal content attention. For the existing
checkpoint, pairs at distance 0–2047 keep their learned bias and pairs at distance
2048 or greater receive zero bias. This is the standard Inkling boundary rule,
not a newly tuned extrapolation setting. Absolute positions beyond 2048 still
use REL for nearby pairs. An evaluation-only adapter restores this behavior
around the published plugin's shorter-input guard; no learned parameters or
checkpoint files change. The source and tests accompany the measurements.

These are post-training endpoint evaluations and do not constitute six new
in-training evaluation events. Compare methods under the same context policy;
the difference from the historical result measures evaluation-context sensitivity.
No architecture or seed was selected using the new held-out scores.

An initial synthetic GPU check was interrupted after one worker blocked on a
cold shared bytecode-cache read. The retry used the existing cache with writes
disabled and passed at lengths 2048, 2049 and 5835 on all 32 ranks. No held-out
evaluation ran in that failed check. Full evaluation attempts and any further
failures are retained in local operational logs.

[Reproduction code, primary-source definition and measured records](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/evaluations/xsa-rel-full-context-20261123)
