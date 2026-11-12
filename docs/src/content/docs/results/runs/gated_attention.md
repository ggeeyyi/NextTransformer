---
title: Gated Attention · elementwise
description: The paper's best gate at matched compute. 21 layers, one segment, and where its early lead went.
---

At 46,875 updates (49.152B tokens), Gated Attention · elementwise (`gated_attention`) reached **0.669459 periodic validation BPB**, **0.666731 independent final validation BPB** (0.668827 on the training split) and **33.5841 CORE ×100** at seed 42. The elementwise gate costs 7.0% of the shared budget at Dense's 23 layers, so calibration reduced the depth to 21 layers and widened the FFN to 6272, leaving **1,306,658,420 parameters** and one value-embedding table fewer than Dense.

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| Gated Attention · headwise (`gated_attention_headwise`) | 23 | 6144 | 1,356,234,504 | 0.667417 | 0.664678 | 34.2736 | +0.00022 / +1.18 |
| Gated Attention · elementwise (`gated_attention`) | 21 | 6272 | 1,306,658,420 | 0.669459 | 0.666731 | 33.5841 | +0.00227 / +0.49 |

The gate helped early and the deficit in depth showed late. Periodic validation BPB was 0.0178 below Dense at update 250 and 0.0007 to 0.0038 below it from update 5,000 to 24,000. The two curves crossed near update 32,000, during the warmdown; from update 33,000 the gated model stayed above Dense and finished 0.0023 above it.

Most of the CORE difference comes from one task. Dense's centered CommonsenseQA at the final update is 5.1, among the lowest of the published runs; this run scores 30.5. Without CommonsenseQA, the mean of the other 21 tasks is 33.73 against Dense's 34.43 (−0.70). The largest per-task deficits are AGIEval LSAT-AR (−6.52 centered points), COPA (−6.00) and BIG-bench CS Algorithms (−4.17); the largest gain besides CommonsenseQA is BoolQ (+10.86).

The formal run completed in one segment on 16 ranks (two nodes of eight) with microbatch eight and four accumulation steps, without a resume. The shell is commit `41b6016`, recorded in the run record, with this method's plugin added; method source hash `3ec6f564f745` in the cost proof identifies the plugin. Attention used SDPA. The gate W_θ is a separate layer trained by AdamW in the auxiliary group; the official code fuses it into `q_proj`.

The cumulative training charge is **4.641914 GFLOPs per token** (−0.652% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The audit covers all 189 scheduled validations and six full CORE-22 evaluations. Gradient norms stayed finite; after update 500 the median was 0.30 and the maximum 0.65.

Before the formal run, a two-update smoke on one GPU and a 16-rank smoke that trained ten updates, saved, and resumed to twenty passed. A 500-update research screening of three gate variants preceded the runs. It used a different parametrization (a separate, zero-initialized gate trained by Muon, outside the repository) and compared validation BPB. The two registered granularities follow the paper and the optimizer choice follows the paper's AdamW training; the screening did not select between them.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gated_attention.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gated_attention.json)
