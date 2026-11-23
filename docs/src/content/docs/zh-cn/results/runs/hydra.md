---
title: Hydra
description: Arena v1 参赛方法：更深更窄的 AttnRes-Canon 主干，把哈希 n-gram 记忆作为深度来源读取，在 32 个 rank 上分两段正式训练，以及与残差族的比较。
---

Hydra（`hydra`）是更深更窄的 AttnRes-Canon 主干（27 层，FFN 4608），把哈希 bigram 和 trigram 记忆作为额外的一个深度来源读取。它是一个 [arena v1 参赛方法](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/hydra.json)：由 Claude Opus 5.5 在一次无人干预的 NextT-Gym 研究会话中设计并选出，人工参与等级为 L0。它以种子 42 训练到第 46,875 步（49.152B tokens），CORE ×100 为 **37.0390**，在已发布的运行中最高。Gym 把 `results/runs/` 中已发布的结果作为历史提供给 agent；agent 选型只用 dev 集上的数值，从未使用验证分片或 CORE-22。

协议规定的验证集测量结果如下，也是已发布运行中最低的：

- 终点周期验证 BPB 为 **0.657746**；
- 独立终验 BPB 为 **0.654993**，训练集上为 0.657100。

模型有 27 层，FFN 宽度为 4608，共 **1,780,828,720 个参数**，其中四张查找表占 335,525,888 个，投影矩阵占 1,572,864 个，162 个 Canon 层占 1,327,104 个。

下表中，Δ 为相对 Dense 的 CORE ×100 分差 / 周期验证 BPB 差：

| 运行 | FFN | CORE ×100 | 周期验证 BPB | 独立终验 BPB | 相对 Dense 的 Δ |
|---|---:|---:|---:|---:|---:|
| Dense（`dense_full`） | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Block AttnRes（`attnres`） | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |
| MSAR | 6144 | 36.7452 | 0.661044 | 0.658278 | +3.65 / −0.00613 |
| Hydra | 4608 | 37.0390 | 0.657746 | 0.654993 | +3.94 / −0.00943 |

与它所扩展的主干 AttnRes-Canon 相比，CORE 高 1.05 分；22 个 CORE 任务中有 17 个持平或更好：

- 提升最多：SQuAD（中心化后 +7.57 分）、BIG-bench Dyck Languages（+5.80）、ARC Challenge（+3.87）；
- 落后最多：COPA（−8.00）、BIG-bench Repeat Copy Logic（−6.25）、BIG-bench Operators（−0.95）。

不计 BoolQ（+2.49），其余 21 个任务平均高 0.98 分，所以提升并不依赖单个任务。

与此前 CORE 最高的 MSAR 相比，CORE 高 0.29 分；22 个任务中有 15 个持平或更好，不计 BoolQ 也高 0.29 分：

- 提升最多：CommonsenseQA（+5.94）、ARC Challenge（+3.87）、SQuAD（+3.69）；
- 落后最多：COPA（−8.00）、Winograd（−5.86）、BIG-bench CS Algorithms（−4.62）。

周期验证 BPB 比 AttnRes-Canon 低 0.00277，比 MSAR 低 0.00330。与已发布的 AttnRes-Canon 在同一步相比，从第 2,000 步到第 46,875 步，验证 BPB 的领先始终在 0.0017 到 0.0036 之间，没有缩小。选型时训练到第 2,000 步，NextT-Gym 在 dev 集上测得 0.80610，agent 自己复现的 AttnRes-Canon 为 0.80791；到第 6,000 步为 0.77636。以上均为单种子观测，不构成统计意义上的排名；尤其是相对 MSAR 的 0.29 分，比较的是两次各一个种子的运行，而这一规模下 CORE 的种子方差尚未测量。

CORE-22 的输入按与对比的已发布运行相同的方式，从左侧截断到 2,048 个 token（见运行记录中的 `coreContext`）。训练器现在默认使用完整 prompt 评测，只有 prompt 超过 2,048 个 token 的任务会因此改变。

正式训练使用预注册的配置，训练源码为 fork 上的 commit `e5d2851`，代码包 SHA-256 为 `b6f95982…`。训练在 32 个 rank 上进行（4 个节点，每个节点 8 张 GPU），每卡 microbatch 16，不做梯度累积，分两段完成：

| 纳入曲线的更新 | 恢复自 | 训练源码 |
|---|---|---|
| 1–41,500 | — | `e5d2851` |
| 41,501–46,875 | 第 41,500 步检查点 | `e5d2851` |

第一段设有 44 小时的计划超时（低于集群 48 小时的作业时限），按计划在记录到第 41,750 步后停止。第二段从最新的完整检查点（第 41,500 步）恢复，并重跑了第 41,501–41,750 步。训练器把被重放的事件另行归档，曲线采用第二段的记录。两段的模型、协议、数据、tokenizer、配置、成本凭证和源码哈希完全相同，都以正式模式训练。两次启动都没有失败，披露卡记为两次正式启动。

累计训练计算量为 **4.702309 GFLOPs/token**，比 SSSL 锚点高 0.640%。每个记录步都与成本凭证一致，没有预算异常。只读审计覆盖全部 189 次验证、六次完整 CORE-22 评测和最终检查点。

训练耗时 157,263 秒，即每步 3.36 秒；评测耗时 10,805 秒。梯度范数始终为有限值，第 500 步之后的中位数为 0.96，最大值为 5.26。

配置、fork commit 和成本凭证于 2026-11-21 07:00 UTC 预注册在 [proposal issue #71](https://github.com/Interactive-Active-AI/NextTransformer/issues/71) 中，第一次正式启动在 08:04 UTC。

启动前，用 32 个 rank 做了两次有界检查：

- 在 dev 集上的四节点检查训练 80 步，在第 40 步中断后恢复。中断与不中断两次启动的第 1 步 loss 完全相同；恢复后的运行与不中断的运行相比，第 41–80 步平均 \|Δloss\| 为 0.059，在舍入误差范围内一致，但不是逐位一致。
- 随后用两步冒烟运行检查 CORE 评测流程，每个任务 32 个样例，截断到 2,048 个 token。这些分数来自几乎未训练的模型，没有被使用。

两次检查都不算正式启动。

Canon 层用的是没有融合 kernel 的 eager PyTorch depthwise 卷积。因此每步耗时反映的是这一实现，而不是方法本身。

[运行记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/hydra.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/hydra.json) · [披露卡](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/hydra.json) · [最终检查点和训练日志](https://huggingface.co/GY2233/nextt-arena-hydra)
