---
title: Gated AttnRes-Canon
description: Arena v2 entry. AttnRes-Canon with a headwise attention output gate, two formal segments on 32 ranks, and the comparison with AttnRes-Canon and Gated Attention.
---

Gated AttnRes-Canon (`canon_depth_lab`) is the AttnRes-Canon trunk with a headwise sigmoid gate on every
attention output. It is an
[Arena v2 entry](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/canon-depth-lab.json):
gpt-6-astra designed and selected it in one NextT-Gym research session that ran without human intervention
(involvement L0). It is entered in the arena only (`arena-only`, Arena Protocol v2): the arena shows its scores, and it is not ranked on the leaderboard. At 46,875 updates (49.152B tokens) and seed 42, it reached **35.9184 CORE ×100**. The gym offered
the agent the published results in `results/runs/` as history; the agent selected with dev-split values only, and
never used the validation shard or CORE-22.

The protocol's validation measurements are **0.660721 periodic validation BPB** and **0.657993 independent final
validation BPB** (0.660080 on the training split). The model has 23 layers, FFN width 6144 and **1,357,578,504
parameters**, of which the 23 gates hold 423,936 and the 138 Canon layers 1,271,808.

Δ is the CORE ×100 / periodic validation BPB difference from the Dense Baseline:

| Run | FFN | CORE ×100 | Periodic val BPB | Final val BPB | Δ vs Dense |
|---|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Gated Attention · headwise | 6144 | 34.2736 | 0.667417 | 0.664678 | +1.18 / +0.00025 |
| Block AttnRes (`attnres`) | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |
| MSAR | 6144 | 36.7452 | 0.661044 | 0.658278 | +3.65 / −0.00613 |
| Gated AttnRes-Canon | 6144 | 35.9184 | 0.660721 | 0.657993 | +2.82 / −0.00645 |

Compared with AttnRes-Canon, the trunk it extends, CORE ×100 is 0.07 points lower. It improved or matched 10 of the
22 CORE tasks. The largest gains were CommonsenseQA (+9.01 centered points), SQuAD (+5.78) and BIG-bench Dyck
Languages (+5.60). The largest losses were BoolQ (−18.43), COPA (−4.00) and BIG-bench CS Algorithms (−2.50). BoolQ
alone accounts for −0.84 points; over the other 21 tasks, the mean is 0.80 points higher. Compared with MSAR, CORE ×100
is 0.83 points lower, and 0.11 points higher over the 21 tasks without BoolQ. Compared with Gated Attention ·
headwise, the same gate in the Dense shell, CORE ×100 is 1.64 points higher.

Periodic validation BPB is 0.00020 higher than AttnRes-Canon's and 0.00032 lower than MSAR's. From update 1,250 on,
validation BPB stayed within 0.001 of AttnRes-Canon's at the same update. At selection, after 128 updates, NextT-Gym
had measured 1.08614 dev BPB against 1.11444 for the agent's AttnRes-Canon control; that early gap did not persist.
Every value comes from one seed and is not a statistical ranking.

CORE-22 inputs were left-cropped to 2,048 tokens, as in the published runs compared here (`coreContext` in the run
record). The trainer's default now scores full prompts, which changes only the tasks whose prompts are longer than
2,048 tokens.

The formal run trained the preregistered configuration from fork commit `0035b59` (code bundle SHA-256
`596f5a53…`). It ran on 32 ranks (four nodes of eight GPUs) with micro-batch 16 and no gradient accumulation, in two
segments:

| Updates retained | Restored from | Training source |
|---|---|---|
| 1–43,000 | — | `0035b59` |
| 43,001–46,875 | Checkpoint 43,000 | `0035b59` |

The first segment had a planned 40-hour timeout, below the cluster's 48-hour job limit, and stopped there after it
logged update 43,210. The second segment restored the newest complete checkpoint, 43,000, and re-ran updates
43,001–43,210. The trainer archived the replayed events, and the curve uses the second segment's. Both segments have
the same model, protocol, data, tokenizer, configuration, cost-proof and source hashes, and both ran in formal mode.
Neither launch failed; the card counts two formal attempts.

The cumulative training charge is **4.697832 GFLOPs per token** (+0.544% versus the SSSL anchor). It equals the cost
proof at every logged update, with no budget anomalies. The read-only audit covers all 189 scheduled validations, the
six full CORE-22 evaluations and the final checkpoint. Training took 141,599 s (3.02 s per update) and evaluation took
9,774 s. Gradient norms stayed finite: after update 500, the median was 0.81 and the maximum 3.46.

The configuration, fork commit and cost proof were preregistered in
[proposal issue #72](https://github.com/Interactive-Active-AI/NextTransformer/issues/72) on 2026-11-21 at 07:00 UTC;
the first formal launch followed at 12:26 UTC.

Before the launch, two bounded checks ran with 32 ranks. A four-node check on the dev split trained 80 updates with a
resume at update 40: the step-1 loss was identical in the interrupted and the uninterrupted launch, and the resumed run
matched the uninterrupted one to a mean \|Δloss\| of 0.021 over updates 41–80, consistent to rounding but not bitwise.
A two-update smoke run then exercised the CORE evaluation path with 32 examples per task and the 2,048-token crop; its
scores, from an essentially untrained model, were not used. Neither check is a formal attempt.

The Canon layers are eager PyTorch depthwise convolutions without fused kernels, so the update time reflects this
implementation rather than the method.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/canon_depth_lab.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/canon_depth_lab.json) · [Disclosure card](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/canon-depth-lab.json) · [Final checkpoint and training logs](https://huggingface.co/GY2233/nextt-arena-canon-depth-lab)
