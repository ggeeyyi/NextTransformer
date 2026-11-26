---
title: Dynamic-KV ShortConv · MLP output
description: Formal dynamic-KV ShortConv run under the fixed protocol.
---

Input-conditioned, head-shared dynamic K/V convolutions with static attention/MLP output convolutions, kernel four, identity initialized, in the full-attention Dense shell without AttnRes. The MLP convolution follows the down projection.

The run completed 46,875 updates / 49.152B input tokens in 4 segment(s), all in formal mode, with the registered 16-rank layout. It includes all 189 validation and six CORE-22 events and passed the cumulative audit. A separate three-update training-only preflight checked checkpoint recovery; formal training started from scratch. A separate static pre-activation experiment was also authorized and run; it is outside this submission. Both completed MLP-output candidates are reported, including the lower-scoring dynamic result.

This is a single-seed fixed-budget placement adaptation. Packed rows reset convolution state; BOS inside a row does not. Shared RoPE, QK normalization, value embeddings, optimizer and evaluation are retained. Incremental KV-cache decoding is unsupported. Formal-safe exact execution speedups were used. Generators and static taps start at zero; the reference scaling and document masking are omitted for the Static ShortConv comparison.

Recorded failed node attempts: 10. Formal resume segments: 3. The user authorized the candidate and complete two-node formal run in the handoff. Calibration used only the registered structural cost grid; no validation or CORE scores were used for candidate selection. Detailed verification retries and node-attempt logs are preserved in the local disclosure bundle.

At complete checkpoint 12,400, the user requested changing rolling checkpoint cadence from 100 to 5000 updates to reduce storage overhead. The run resumed with the same frozen source and scientific configuration. Validation and CORE events were retained. The corresponding node termination was intentional, and prior logs and launch settings are preserved in the local disclosure bundle.

## Results and evaluation context

At update 46,875, periodic validation BPB is **0.667611**, independent final validation BPB is **0.664908**, and CORE ×100 is **33.3858**. The leaderboard uses periodic validation BPB. This is one run at seed 42.

Training context was 2,048 tokens. All six full CORE-22 events used **left truncation at 2,048 tokens**, verified in the frozen evaluator and declared through the current exporter's `legacy_core_2048` option. These are the original scores, not the newer full-prompt policy.

Frozen training source: `e48693b57ab009061524df07c05fd3d589affc73`. Plugin arithmetic, configuration, cost proof and cumulative audit are preserved from that run. Public display names are expanded for readability; method IDs remain stable for checkpoint and proof compatibility. No arena preregistration is claimed.

The three resumes started at updates 1,400, 8,300 and 12,400. The first two recovered from 600-second NCCL timeouts; the last implemented the user-requested checkpoint cadence change. The ten failed node attempts include verification, preflight and operational attempts; they are not ten independent formal runs. Earlier prechecks retried after a rendezvous port collision and cold shared-storage cache contention. Training showed transient loss/gradient spikes; no optimizer or model change was made in response, and the completed result is published unchanged.
