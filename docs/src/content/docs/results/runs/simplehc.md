---
title: SimpleHC
description: Identity residual transitions with four-sublayer shared gate inputs on the full-attention shell, in one segment.
---

At 46,875 updates (49.152B tokens), SimpleHC (`simplehc`) reached **0.662833 periodic validation BPB**, **0.660070 independent final validation BPB** (0.662097 on the training split) and **35.1129 CORE ×100** at seed 42. Calibration kept the 23 full-attention layers and 6144-wide FFN of Standard Residual, Dense Baseline and HC, with **1,358,072,020 parameters** (+0.441% versus the SSSL anchor).

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Standard Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard Residual (`residual_standard`) | 23 | 6144 | 1,355,810,568 | 0.673959 | 0.671214 | 31.9513 | — |
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | −0.00679 / +1.14 |
| Hyper-Connections (`hc`) | 23 | 6144 | 1,356,235,700 | 0.664400 | 0.661683 | 35.0943 | −0.00956 / +3.14 |
| Block AttnRes (`attnres`) | 23 | 6144 | 1,355,882,760 | 0.662963 | 0.660217 | 33.9971 | −0.01100 / +2.05 |
| SimpleHC (`simplehc`) | 23 | 6144 | 1,358,072,020 | 0.662833 | 0.660070 | 35.1129 | −0.01113 / +3.16 |
| mHC (`mhc`) | 23 | 6016 | 1,353,550,818 | 0.661665 | 0.658908 | 35.3278 | −0.01229 / +3.38 |

SimpleHC drops learned stream-to-stream mixing (`H_res = I`) while keeping mHC's full-state sigmoid read and `2σ` write gates, amortized across consecutive groups of four sublayers via intra-group scalar corrections. Periodic validation BPB stayed between mHC and HC throughout training, remaining +0.0010 to +0.0012 above mHC from update 8,000 onward and finishing +0.00117 above mHC, −0.00013 below Block AttnRes, −0.00157 below HC and −0.00434 below the Dense Baseline (+2.02 CORE points versus Dense).

CORE trailed mHC at intermediate checkpoints (28.23 versus 29.28 at update 32,000; 32.76 versus 33.31 at update 40,000) and closed most of the gap during late warmdown, finishing 0.21 points below mHC and 0.02 above HC. Across the 22 tasks at update 46,875, SimpleHC and mHC split 11–11; the largest gains over Standard Residual are CommonsenseQA (+27.44 centered points), Winograd (+16.85) and SQuAD (+11.15), and the largest deficits are BIG-bench Dyck Languages (−5.60) and AGIEval LSAT-AR (−5.43).

Removing learned stream mixing also eliminated HC's gradient spikes. Gradient norms stayed finite on every update, peaking at 3.10 at update 10; after update 500 the median was 0.32 and the maximum 0.68 (against median 0.90 and a spike of 1,824 in HC).

The formal run completed in one segment on 16 ranks (two nodes of eight) with microbatch eight, four accumulation steps, `--speedups exact` and `--formal`, without a resume. It trained at commit `0619bc7` (upstream `df179cb` with this plugin added, before [#42](https://github.com/Interactive-Active-AI/NextTransformer/pull/42) updated Hopper FlashAttention 3 loading in `nanochat/flash_attention.py`); the plugin source, `model_hash`, parameter count, operator counts and per-token FLOPs are unchanged on `main`. Attention used SDPA.

The cumulative training charge is **4.692986 GFLOPs per token** (+0.441% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The audit covers all 189 scheduled validations and six full CORE-22 evaluations. Two 30-update research smokes on eight and 16 ranks (including checkpoint save at update 20 and resume to update 30) preceded the formal launch; no candidate screening or failed formal attempt occurred.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/simplehc.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/simplehc.json)
