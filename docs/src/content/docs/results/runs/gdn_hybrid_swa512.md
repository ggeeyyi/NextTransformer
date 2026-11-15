---
title: Hybrid GDN + SWA
description: H03-SWA512 configuration, full formal result, resume history and execution evidence.
---

H03-SWA512 completed 46,875 updates and 49.152B input tokens at seed 42. Final periodic validation BPB was **0.66651494**, independent final validation BPB was **0.66379960**, and **CORE ×100 was 33.089409**. The cumulative charge was **4.685968542 GFLOPs per input token**, **+0.290434%** against the SSSL cost anchor. The read-only cumulative audit passed with 189 validation events, six complete CORE-22 events and zero recorded budget anomalies.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gdn_hybrid_swa512.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gdn_hybrid_swa512.json) · [Reproducible audit inputs and execution evidence](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/runs/gdn_hybrid_swa512)

## Configuration and identity

The model has 24 independent layers, alternating 12 GDN and 12 SWA blocks, width 1536, 12 heads, FFN width 6016 and 1,403,661,866 parameters. Its 512-key window includes the current token and up to 511 preceding tokens within the 2048-token context. Training used the FLA chunk-64 GDN backend and deterministic SDPA with FlashAttention disabled for SWA.

The training implementation is `gdn_hybrid_swa` at frozen revision [`4926c5de832968957cd84411b1fa24ca0ada81c4`](https://github.com/Interactive-Active-AI/NextTransformer/tree/4926c5de832968957cd84411b1fa24ca0ada81c4). `gdn_hybrid_swa512` is the publication identity for this configuration. The archived configuration and cost proof retain the original training method name and model hash. The selected 24-layer, FFN-6016 candidate received a verification-only cost proof before the formal run. Its proof records no structural search.

## Training and resumes

Training started from random initialization. The three retained segments began at updates 0, 1,000 and 7,500, each with 8 ranks, micro-batch 8 and accumulation 8. Every segment records formal admission and identical model, data, tokenizer, source and cost fingerprints.

After an earlier pause, the complete update-1,000 state was restored on the continuation host. A startup attempt ended before training progressed, and a later attempt reached update 8,000 before CORE evaluation was interrupted by the resource guard. The run resumed from the complete update-7,500 checkpoint after bounded evaluation checks. Replayed updates replaced the discarded suffix in the canonical log. The recorded loss, gradient norm, input-token and compute fields at logged updates 7,510 through 8,000 matched the earlier attempt exactly. The published metrics contain one retained sequence of updates and scheduled evaluations. Failed attempts and replay add execution overhead beyond that sequence.

## CORE execution

The continuation used an external evaluation wrapper with cached inputs from the original CORE example builder and CUDA graphs at each original input shape. It retained the task metadata, input tokens, continuation spans, rank assignment and scoring rules. For each shape group, the first graph replay was compared with native forward losses and token predictions using exact equality. The wrapper sources, input-cache manifest and complete checkpoint-7,500 CORE execution check are included in the evidence directory. That bounded check is separate from the six scheduled evaluations.

Padded batching and evaluation in independent processes were tried and rejected after answer differences. Their outputs are excluded from the published run. The source fingerprint covers the frozen repository code. The external execution wrappers have separate SHA256 hashes in `submission.json`, including pacing, CPU affinity, progress reporting and parallel checkpoint checksumming.

## Checkpoint and review limits

The final checkpoint contains the model and all eight rank payloads. Submission preparation rehashed each payload and compared it with the complete-checkpoint manifest and the generation-pinned archive readback receipts. The public checkpoint manifest contains their sizes and hashes. The payloads are retained in the contributor archive, with public download URLs pending.

The published 2048-window configuration uses 22 layers and FFN width 6528. This 512-window result also changes depth and FFN width under the common compute budget, so the comparison includes all three changes. It is a single-seed 2K-context result. Timing includes site pacing, and long-context quality and deployment efficiency require separate measurements. The main leaderboard uses the final periodic validation BPB. The independently sampled final BPB uses 20,971,520 tokens per split and remains a separate field.
