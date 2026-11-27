---
title: Hashed n-gram memory + Full AttnRes + Canon-A/B/C/D
description: Arena v2 entry. AttnRes-Canon with a hashed bigram memory and Canon layers at more sites, two formal segments on 32 ranks, and the comparison with the Residual family.
---

Hashed n-gram memory + Full AttnRes + Canon-A/B/C/D (`ngram_canon_attnres`) is the AttnRes-Canon trunk with Canon
layers at more sites and with wider kernels, and a hashed bigram memory added to the token embedding. It is an
[Arena v2 entry](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/ngram-canon-attnres.json):
Claude Fable 5.2 designed and selected it in one NextT-Gym research session that ran without human intervention
(involvement L0). At 46,875 updates (49.152B tokens) and seed 42, it reached **35.5114 CORE ×100**. The gym offered the
agent the published results in `results/runs/` as history; the agent selected with dev-split values only, and never
used the validation shard or CORE-22.

The protocol's validation measurements are **0.658060 periodic validation BPB** and **0.655244 independent final
validation BPB** (0.657481 on the training split), the lowest of the published runs. The model has 23 layers, FFN
width 6144 and **1,850,401,032 parameters**, of which the two lookup tables hold 491,520,000 and the 163 Canon layers
2,998,272.

Δ is the CORE ×100 / periodic validation BPB difference from the Dense Baseline:

| Run | FFN | CORE ×100 | Periodic val BPB | Final val BPB | Δ vs Dense |
|---|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Block AttnRes (`attnres`) | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |
| MSAR | 6144 | 36.7452 | 0.661044 | 0.658278 | +3.65 / −0.00613 |
| Hashed n-gram memory + Full AttnRes + Canon-A/B/C/D | 6144 | 35.5114 | 0.658060 | 0.655244 | +2.42 / −0.00911 |

Compared with AttnRes-Canon, the trunk it extends, CORE ×100 is 0.48 points lower. It improved or matched 13 of the
22 CORE tasks. The largest gains were AGI Eval LSAT-AR (+7.61 centered points), BIG-bench CS Algorithms (+6.89) and
CommonsenseQA (+6.24). The largest losses were BoolQ (−16.90), COPA (−10.00) and OpenBookQA (−2.93). BoolQ alone
accounts for −0.77 of the −0.48 points; over the other 21 tasks, the mean is 0.30 points higher. Compared with MSAR,
CORE ×100 is 1.23 points lower, and 0.38 points lower over the 21 tasks without BoolQ.

Periodic validation BPB is 0.00246 lower than AttnRes-Canon's and 0.00298 lower than MSAR's. At selection, after 2,048
updates, NextT-Gym had measured 0.80152 dev BPB against 0.80851 for the agent's own reproduction of AttnRes-Canon. In
the formal run the lead over AttnRes-Canon at the same update narrowed from 0.0060 validation BPB at 4,000 updates to
0.0036 at 16,000 and 0.0025 at 46,875. Every value comes from one seed and is not a statistical ranking.

CORE-22 inputs were left-cropped to 2,048 tokens, as in the published runs compared here (`coreContext` in the run
record). The trainer's default now scores full prompts, which changes only the tasks whose prompts are longer than
2,048 tokens.

The formal run trained the preregistered configuration from fork commit `57b3367` (code bundle SHA-256
`8289517a…`). It ran on 32 ranks (four nodes of eight GPUs) with micro-batch 16 and no gradient accumulation, in two
segments:

| Updates retained | Restored from | Training source |
|---|---|---|
| 1–43,000 | — | `57b3367` |
| 43,001–46,875 | Checkpoint 43,000 | `57b3367` |

The first segment had a planned 44-hour timeout, below the cluster's 48-hour job limit, and stopped there after it
logged update 43,490. The second segment restored the newest complete checkpoint, 43,000, and re-ran updates
43,001–43,490. The trainer archived the replayed events, and the curve uses the second segment's. Both segments have
the same model, protocol, data, tokenizer, configuration, cost-proof and source hashes, and both ran in formal mode.
Neither launch failed; the card counts two formal attempts.

The cumulative training charge is **4.705561 GFLOPs per token** (+0.710% versus the SSSL anchor). It equals the cost
proof at every logged update, with no budget anomalies. The read-only audit covers all 189 scheduled validations, the
six full CORE-22 evaluations and the final checkpoint. Training took 155,188 s (3.31 s per update) and evaluation took
10,282 s. Gradient norms stayed finite: after update 500, the median was 0.28 and the maximum 1.05.

The configuration, fork commit and cost proof were preregistered in
[proposal issue #70](https://github.com/Interactive-Active-AI/NextTransformer/issues/70) on 2026-11-21 at 07:00 UTC;
the first formal launch followed at 08:03 UTC.

Before the launch, two bounded checks ran with 32 ranks. A four-node check on the dev split trained 80 updates with a
resume at update 40: the step-1 loss was identical in the interrupted and the uninterrupted launch, and the resumed run
matched the uninterrupted one to a mean \|Δloss\| of 0.020 over updates 41–80, consistent to rounding but not bitwise.
A two-update smoke run then exercised the CORE evaluation path with 32 examples per task and the 2,048-token crop; its
scores, from an essentially untrained model, were not used. Neither check is a formal attempt.

The Canon layers are eager PyTorch depthwise convolutions without fused kernels, so the update time reflects this
implementation rather than the method.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/ngram_canon_attnres.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/ngram_canon_attnres.json) · [Disclosure card](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/ngram-canon-attnres.json) · [Final checkpoint and training logs](https://huggingface.co/GY2233/nextt-arena-ngram-canon-attnres)
