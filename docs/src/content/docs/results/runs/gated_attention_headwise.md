---
title: Gated Attention · headwise
description: One gate per head on Dense's own shell, a paired comparison at matched compute.
---

At 46,875 updates (49.152B tokens), Gated Attention · headwise (`gated_attention_headwise`) reached **0.667417 periodic validation BPB**, **0.664678 independent final validation BPB** (0.666714 on the training split) and **34.2736 CORE ×100** at seed 42. The gate adds 0.42M parameters and 0.06% compute, so calibration kept Dense's shell: 23 layers, FFN 6144, **1,356,234,504 parameters**. The pair with Dense isolates the headwise gate.

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| Gated Attention · headwise (`gated_attention_headwise`) | 23 | 6144 | 1,356,234,504 | 0.667417 | 0.664678 | 34.2736 | +0.00022 / +1.18 |
| Gated Attention · elementwise (`gated_attention`) | 21 | 6272 | 1,306,658,420 | 0.669459 | 0.666731 | 33.5841 | +0.00227 / +0.49 |

Periodic validation BPB was 0.0046 below Dense at update 250 and up to 0.0036 below it in the first 10,000 updates. From update 16,000 the two curves stayed within 0.0009 of each other, and within 0.0003 from update 28,000; the final BPB is 0.00022 above Dense.

The CORE difference comes from CommonsenseQA. Dense's centered CommonsenseQA at the final update is 5.1, among the lowest of the published runs; this run scores 34.2. Without CommonsenseQA, the mean of the other 21 tasks is 34.28 against Dense's 34.43 (−0.15). The largest per-task deficit is AGIEval LSAT-AR (−10.33 centered points); the largest gains besides CommonsenseQA are COPA (+4.00) and SQuAD (+3.52).

The formal run completed in one segment on 16 ranks (two nodes of eight) with microbatch eight and four accumulation steps, without a resume. The shell is commit `41b6016`, recorded in the run record, with this method's plugin added; method source hash `c37a3eb357a1` in the cost proof identifies the plugin. Attention used SDPA. The gate scores come from a separate layer trained by AdamW in the auxiliary group; the official code appends them to `q_proj`.

The cumulative training charge is **4.677384 GFLOPs per token** (+0.107% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The audit covers all 189 scheduled validations and six full CORE-22 evaluations. Gradient norms stayed finite; after update 500 the median was 0.28 and the maximum 0.78.

The pre-run smokes and the 500-update research screening are described with the [elementwise run](../gated_attention/).

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gated_attention_headwise.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gated_attention_headwise.json)
