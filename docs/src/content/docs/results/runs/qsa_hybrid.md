---
title: Hybrid GDN + QSA
description: H05 formal result, configuration, resume history and execution evidence.
---

H05 completed 46,875 updates and 49.152B input tokens at seed 42. Final periodic validation BPB was **0.66650463**, independent final validation BPB was **0.66375966**, and **CORE ×100 was 33.002036**. The cumulative charge was **4.691636932 GFLOPs per input token**, **+0.411750%** against the SSSL cost anchor. The cumulative audit passed with 189 validation events and six complete CORE-22 events.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/qsa_hybrid.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/qsa_hybrid.json) · [Reproducible inputs and execution evidence](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/runs/qsa_hybrid)

## Configuration and identity

The registered `qsa_hybrid` model has 24 independent layers, repeating three GDN blocks and one QSA block six times. It uses width 1536, 12 heads, FFN width 5760 and 1,405,174,202 parameters. QSA has a 256-token selection budget, compression factor 4 and indexer auxiliary loss on QSA layers. The first 512 updates use dense burn-in; the remaining 46,363 use joint sparse training. Both stages are included in the cumulative compute charge.

Training used the frozen revision [`4926c5de832968957cd84411b1fa24ca0ada81c4`](https://github.com/Interactive-Active-AI/NextTransformer/tree/4926c5de832968957cd84411b1fa24ca0ada81c4), FLA chunk-64 GDN, the QSA kernel backend and deterministic SDPA with FlashAttention disabled. The original configuration and verification-only cost proof are archived. The model and method-source hashes match those records; no new implementation or architecture is introduced by this result.

## Training and resumes

Training started from random initialization. The retained segments began at updates 0, 1,000 and 7,500, each with 8 ranks, micro-batch 8 and accumulation 8. All segments record formal admission and identical model, data, tokenizer, source, optimizer recipe and cost fingerprints.

After an earlier pause, the complete update-1,000 state was restored on the continuation host. An initial startup attempt stopped before training progressed. A later attempt reached the first scheduled CORE evaluation, where the original variable-length reference evaluation path was interrupted by the resource guard. The run resumed from the complete update-7,500 checkpoint after bounded evaluation checks. A brief continuation was restarted from the same checkpoint to apply the final shape-group synchronization change.

The canonical log retains one sequence of updates and scheduled evaluations. The public replay check compares the overlapping logged loss, language-model loss, auxiliary loss, gradient norm, token and compute fields from both retained pre-restart histories against this sequence. These comparisons passed exactly; they cover the listed logged steps, not every update or complete tensor state. Failed attempts and replay add execution overhead beyond the retained sequence.

## CORE execution

The frozen trainer switches QSA to the reference backend for variable-length CORE inputs. An external wrapper used cached original input tensors and original-shape CUDA graphs in the original rank processes. It checks task metadata, dataset and evaluation-source hashes, rank assignment and continuation spans. For each shape group, its first replay is compared with native forward losses and predictions using exact equality, with the original scoring rules retained.

During capture, two diagnostic edge-count conversions in `SparseMixer.forward` were deferred from Python integers to tensor sums. The original forward was restored immediately after capture; the training forward was unchanged. Shape groups used bounded synchronization to control rank skew.

A complete checkpoint-7,500 CORE execution check passed its numerical assertions, but the subsequent resource handoff exceeded its limit. A targeted check of the last 64 shape groups per rank validated the synchronization correction before continuation. These checks are separate from the six scheduled CORE evaluations. Padded batching and independent-process evaluation trials changed some answers and were rejected; their results are excluded. A correction record also documents that a rejected debug reporter had incorrectly emitted an equality flag.

The frozen source fingerprint does not cover external wrappers. Their exact sources and separate SHA-256 hashes are included with the evidence, along with pacing, rank CPU affinity, progress reporting and parallel checkpoint hashing.

## Checkpoint and review limits

The final model and all eight rank payloads were rehashed during submission preparation and matched the complete-checkpoint manifest and generation-pinned archive SHA-256 receipts. Their sizes and hashes are public; the weights remain in the contributor archive, with public download URLs pending.

This is a single-seed result at 2K context. Gated attention, n-grams, MoE, multi-token prediction and long-context continued pretraining of the source model are not reproduced. Reported timings include site pacing; failed attempts add separate execution overhead. Deployment speed and long-context quality require separate measurements. The leaderboard uses final periodic validation BPB; independent final BPB uses 20,971,520 tokens per split and is reported separately.
