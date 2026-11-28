---
title: N-gram AttnRes-Canon
description: Arena v2 参赛方法：AttnRes-Canon 加哈希 bigram 记忆和更多位置的 Canon 层，在 32 个 rank 上分两段正式训练，以及与残差族的比较。
---

N-gram AttnRes-Canon（`ngram_canon_attnres`）以 AttnRes-Canon 为主干，在更多位置加入卷积核更宽的 Canon 层，并把哈希 bigram 记忆加到 token 嵌入上。它是一个 [Arena v2 参赛方法](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/ngram-canon-attnres.json)：由 Claude Fable 5.2 在一次无人干预的 NextT-Gym 研究会话中设计并选出，人工参与等级为 L0。它只进入竞技场（`arena-only`，Arena Protocol v2）：竞技场展示它的分数，它不在排行榜上排名。它以种子 42 训练到第 46,875 步（49.152B tokens），CORE ×100 为 **35.5114**。Gym 把 `results/runs/` 中已发布的结果作为历史提供给 agent；agent 选型只用 dev 集上的数值，从未使用验证分片或 CORE-22。

协议规定的验证集测量结果如下，在已发布的运行中仅次于 Hydra（0.654993）：

- 终点周期验证 BPB 为 **0.658060**；
- 独立终验 BPB 为 **0.655244**，训练集上为 0.657481。

模型有 23 层，FFN 宽度为 6144，共 **1,850,401,032 个参数**，其中两张查找表占 491,520,000 个，163 个 Canon 层占 2,998,272 个。

下表中，Δ 为相对 Dense 的 CORE ×100 分差 / 周期验证 BPB 差：

| 运行 | FFN | CORE ×100 | 周期验证 BPB | 独立终验 BPB | 相对 Dense 的 Δ |
|---|---:|---:|---:|---:|---:|
| Dense（`dense_full`） | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Block AttnRes（`attnres`） | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |
| MSAR | 6144 | 36.7452 | 0.661044 | 0.658278 | +3.65 / −0.00613 |
| N-gram AttnRes-Canon | 6144 | 35.5114 | 0.658060 | 0.655244 | +2.42 / −0.00911 |

与它所扩展的主干 AttnRes-Canon 相比，CORE 低 0.48 分；22 个 CORE 任务中有 13 个持平或更好：

- 提升最多：AGI Eval LSAT-AR（中心化后 +7.61 分）、BIG-bench CS Algorithms（+6.89）、CommonsenseQA（+6.24）；
- 落后最多：BoolQ（−16.90）、COPA（−10.00）、OpenBookQA（−2.93）。

BoolQ 一项就贡献了 −0.77 分，超过总差距 −0.48 分；在其余 21 个任务上，平均高 0.30 分。与 MSAR 相比，CORE 低 1.23 分；不计 BoolQ，在其余 21 个任务上平均低 0.38 分。

周期验证 BPB 比 AttnRes-Canon 低 0.00246，比 MSAR 低 0.00298。选型时训练到第 2,048 步，NextT-Gym 在 dev 集上测得 0.80152，agent 自己复现的 AttnRes-Canon 为 0.80851。正式训练中，同一步相对 AttnRes-Canon 的验证 BPB 领先逐渐缩小：第 4,000 步为 0.0060，第 16,000 步为 0.0036，第 46,875 步为 0.0025。以上均为单种子观测，不构成统计意义上的排名。

CORE-22 的输入按与对比的已发布运行相同的方式，从左侧截断到 2,048 个 token（见运行记录中的 `coreContext`）。训练器现在默认使用完整 prompt 评测，只有 prompt 超过 2,048 个 token 的任务会因此改变。

正式训练使用预注册的配置，训练源码为 fork 上的 commit `57b3367`，代码包 SHA-256 为 `8289517a…`。训练在 32 个 rank 上进行（4 个节点，每个节点 8 张 GPU），每卡 microbatch 16，不做梯度累积，分两段完成：

| 纳入曲线的更新 | 恢复自 | 训练源码 |
|---|---|---|
| 1–43,000 | — | `57b3367` |
| 43,001–46,875 | 第 43,000 步检查点 | `57b3367` |

第一段设有 44 小时的计划超时（低于集群 48 小时的作业时限），按计划在记录到第 43,490 步后停止。第二段从最新的完整检查点（第 43,000 步）恢复，并重跑了第 43,001–43,490 步。训练器把被重放的事件另行归档，曲线采用第二段的记录。两段的模型、协议、数据、tokenizer、配置、成本凭证和源码哈希完全相同，都以正式模式训练。两次启动都没有失败，披露卡记为两次正式启动。

累计训练计算量为 **4.705561 GFLOPs/token**，比 SSSL 锚点高 0.710%。每个记录步都与成本凭证一致，没有预算异常。只读审计覆盖全部 189 次验证、六次完整 CORE-22 评测和最终检查点。

训练耗时 155,188 秒，即每步 3.31 秒；评测耗时 10,282 秒。梯度范数始终为有限值，第 500 步之后的中位数为 0.28，最大值为 1.05。

配置、fork commit 和成本凭证于 2026-11-21 07:00 UTC 预注册在 [proposal issue #70](https://github.com/Interactive-Active-AI/NextTransformer/issues/70) 中，第一次正式启动在 08:03 UTC。

启动前，用 32 个 rank 做了两次有界检查：

- 在 dev 集上的四节点检查训练 80 步，在第 40 步中断后恢复。中断与不中断两次启动的第 1 步 loss 完全相同；恢复后的运行与不中断的运行相比，第 41–80 步平均 \|Δloss\| 为 0.020，在舍入误差范围内一致，但不是逐位一致。
- 随后用两步冒烟运行检查 CORE 评测流程，每个任务 32 个样例，截断到 2,048 个 token。这些分数来自几乎未训练的模型，没有被使用。

两次检查都不算正式启动。

Canon 层用的是没有融合 kernel 的 eager PyTorch depthwise 卷积。因此每步耗时反映的是这一实现，而不是方法本身。

[运行记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/ngram_canon_attnres.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/ngram_canon_attnres.json) · [披露卡](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/ngram-canon-attnres.json) · [最终检查点和训练日志](https://huggingface.co/GY2233/nextt-arena-ngram-canon-attnres)
