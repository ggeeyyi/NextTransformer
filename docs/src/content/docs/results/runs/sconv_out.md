---
title: Static ShortConv · MLP output
description: Formal static ShortConv run under the fixed protocol.
---

Static causal depthwise convolutions on K/V and attention/MLP branch outputs, kernel four, identity initialized, in the full-attention Dense shell without AttnRes. The MLP convolution follows the down projection.

The run completed 46,875 updates / 49.152B input tokens in 2 segment(s), all in formal mode, with the registered 32-rank layout. It includes all 189 validation and six CORE-22 events and passed the cumulative audit. A separate three-update training-only preflight checked checkpoint recovery; formal training started from scratch. A separate static pre-activation experiment was also authorized and run; it is outside this submission. Both completed MLP-output candidates are reported, including the lower-scoring dynamic result.

At update 43,700, the user requested a checkpoint-cadence change from every 100 updates to the protocol's 5,000-update cadence and final checkpoint. Training resumed from a complete checkpoint with unchanged source, model, data order, optimizer, precision, and evaluation schedule. This was an operational adjustment; original attempt logs and resume provenance were retained.

This is a single-seed fixed-budget placement adaptation. Packed rows reset convolution state; BOS inside a row does not. Shared RoPE, QK normalization, value embeddings, optimizer and evaluation are retained. Incremental KV-cache decoding is unsupported. Formal-safe exact execution speedups were used.

## Results and evaluation context

At update 46,875, periodic validation BPB is **0.665012**, independent final validation BPB is **0.662302**, and CORE ×100 is **34.5295**. The leaderboard uses periodic validation BPB. This is one run at seed 42.

Training context was 2,048 tokens. All six full CORE-22 events used **left truncation at 2,048 tokens**, verified in the frozen evaluator and declared through the current exporter's `legacy_core_2048` option. These are the original scores, not the newer full-prompt policy.

Frozen training source: `f5843ac74bfac4552debd2a4f2fc2a47164eb941`. Plugin arithmetic, configuration, cost proof and cumulative audit are preserved from that run. Public display names are expanded for readability; method IDs remain stable for checkpoint and proof compatibility. No arena preregistration is claimed.

An initial GPU preflight failed because the pinned attention kernel was absent from the offline cache. The cache was populated and preflight passed before formal training. This was an environment retry, not candidate selection.
