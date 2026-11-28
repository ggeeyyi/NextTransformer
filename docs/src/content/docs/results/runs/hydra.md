---
title: Hydra
description: Arena v2 entry. A deeper, narrower AttnRes-Canon trunk with a hashed n-gram memory read as a depth source, two formal segments on 32 ranks, and the comparison with the Residual family.
---

Hydra (`hydra`) is a deeper, narrower AttnRes-Canon trunk (27 layers, FFN 4608) that reads a hashed bigram and
trigram memory as one more depth source. It is an
[Arena v2 entry](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/hydra.json): Claude
Opus 5.5 designed and selected it in one NextT-Gym research session that ran without human intervention (involvement
L0). At 46,875 updates (49.152B tokens) and seed 42, it reached **37.0390 CORE ×100**, the highest of the published
runs. The gym offered the agent the published results in `results/runs/` as history; the agent selected with
dev-split values only, and never used the validation shard or CORE-22.

The protocol's validation measurements are **0.657746 periodic validation BPB** and **0.654993 independent final
validation BPB** (0.657100 on the training split), also the lowest of the published runs. The model has 27 layers, FFN
width 4608 and **1,780,828,720 parameters**, of which the four lookup tables hold 335,525,888, the projection 1,572,864
and the 162 Canon layers 1,327,104.

Δ is the CORE ×100 / periodic validation BPB difference from the Dense Baseline:

| Run | FFN | CORE ×100 | Periodic val BPB | Final val BPB | Δ vs Dense |
|---|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Block AttnRes (`attnres`) | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |
| MSAR | 6144 | 36.7452 | 0.661044 | 0.658278 | +3.65 / −0.00613 |
| Hydra | 4608 | 37.0390 | 0.657746 | 0.654993 | +3.94 / −0.00943 |

Compared with AttnRes-Canon, the trunk it extends, CORE ×100 is 1.05 points higher. It improved or matched 17 of the
22 CORE tasks. The largest gains were SQuAD (+7.57 centered points), BIG-bench Dyck Languages (+5.80) and ARC
Challenge (+3.87). The largest losses were COPA (−8.00), BIG-bench Repeat Copy Logic (−6.25) and BIG-bench Operators
(−0.95). Without BoolQ (+2.49) the mean over the other 21 tasks is 0.98 points higher, so the gain is not carried by
one task. Compared with MSAR, the previous highest CORE, CORE ×100 is 0.29 points higher, with 15 of 22 tasks improved
or matched and 0.29 points over the 21 tasks without BoolQ; the largest gains were CommonsenseQA (+5.94), ARC
Challenge (+3.87) and SQuAD (+3.69), the largest losses COPA (−8.00), Winograd (−5.86) and BIG-bench CS Algorithms
(−4.62).

Periodic validation BPB is 0.00277 lower than AttnRes-Canon's and 0.00330 lower than MSAR's. Against the published
AttnRes-Canon at the same update, the lead stayed between 0.0017 and 0.0036 validation BPB from update 2,000 to
46,875; it did not narrow. At selection, after 2,000 updates, NextT-Gym had measured 0.80610 dev BPB against 0.80791
for the agent's own reproduction of AttnRes-Canon, and 0.77636 at 6,000 updates. Every value comes from one seed and is
not a statistical ranking; in particular, the 0.29-point lead over MSAR compares one seed of each run, and the seed
variance of CORE at this scale has not been measured.

CORE-22 inputs were left-cropped to 2,048 tokens, as in the published runs compared here (`coreContext` in the run
record). The trainer's default now scores full prompts, which changes only the tasks whose prompts are longer than
2,048 tokens.

The formal run trained the preregistered configuration from fork commit `e5d2851` (code bundle SHA-256
`b6f95982…`). It ran on 32 ranks (four nodes of eight GPUs) with micro-batch 16 and no gradient accumulation, in two
segments:

| Updates retained | Restored from | Training source |
|---|---|---|
| 1–41,500 | — | `e5d2851` |
| 41,501–46,875 | Checkpoint 41,500 | `e5d2851` |

The first segment had a planned 44-hour timeout, below the cluster's 48-hour job limit, and stopped there after it
logged update 41,750. The second segment restored the newest complete checkpoint, 41,500, and re-ran updates
41,501–41,750. The trainer archived the replayed events, and the curve uses the second segment's. Both segments have
the same model, protocol, data, tokenizer, configuration, cost-proof and source hashes, and both ran in formal mode.
Neither launch failed; the card counts two formal attempts.

The cumulative training charge is **4.702309 GFLOPs per token** (+0.640% versus the SSSL anchor). It equals the cost
proof at every logged update, with no budget anomalies. The read-only audit covers all 189 scheduled validations, the
six full CORE-22 evaluations and the final checkpoint. Training took 157,263 s (3.36 s per update) and evaluation took
10,805 s. Gradient norms stayed finite: after update 500, the median was 0.96 and the maximum 5.26.

The configuration, fork commit and cost proof were preregistered in
[proposal issue #71](https://github.com/Interactive-Active-AI/NextTransformer/issues/71) on 2026-11-21 at 07:00 UTC;
the first formal launch followed at 08:04 UTC.

Before the launch, two bounded checks ran with 32 ranks. A four-node check on the dev split trained 80 updates with a
resume at update 40: the step-1 loss was identical in the interrupted and the uninterrupted launch, and the resumed run
matched the uninterrupted one to a mean \|Δloss\| of 0.059 over updates 41–80, consistent to rounding but not bitwise.
A two-update smoke run then exercised the CORE evaluation path with 32 examples per task and the 2,048-token crop; its
scores, from an essentially untrained model, were not used. Neither check is a formal attempt.

The Canon layers are eager PyTorch depthwise convolutions without fused kernels, so the update time reflects this
implementation rather than the method.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/hydra.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/hydra.json) · [Disclosure card](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/hydra.json) · [Final checkpoint and training logs](https://huggingface.co/GY2233/nextt-arena-hydra)
