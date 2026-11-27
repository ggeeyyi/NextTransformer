---
title: MoE 赛道
description: 面向 Mixture-of-Experts 语言模型的算力受控对比平台，含两个复现 Marin 的尺寸。
---

MoE 赛道把 NextTransformer 的契约用到稀疏模型上：数据相同、顺序相同、每 token 训练 FLOPs 相同、评测相同；参数量和速度是
结果，不是约束。

两个固定赛道（d768、d1024）各自完整复现 Marin 一个 compute-optimal Grug MoE 单元，包括模型、MuonH 优化器、学习率调度、
QB 路由、上下文、batch 和更新步数。

| 赛道 | 结构 | 上下文 | 训练 token | 每 token FLOPs | 总参数 / 激活参数 |
|---|---|---|---|---|---|
| **d768** | 8 层 × 768，GQA 6/1，256 个宽 384 的专家选 4，另有宽 768 的共享专家 | 4,096 | 4.42B | 9.369 × 10⁸ | 2.00B / 213M |
| **d1024** | 11 层 × 1024，GQA 8/2，256 个宽 512 的专家选 4，另有宽 1024 的共享专家 | 4,096 | 8.43B | 1.695 × 10⁹ | 4.71B / 347M |

[NextTransformer 240 TPP](../../../moe/protocol/#240-tpp-comparison) · d768 baseline / HC / mHC / SimpleHC / iHC (14.07B tokens) · d1024 baseline / HC / mHC / SimpleHC / iHC (34.06B tokens).

- **总分 MoE-CORE**：对 14 个已发表任务的中心化准确率取平均，每项等权。
  - 短任务（10 项）：ARC-E/C、HellaSwag、PIQA、WinoGrande、CSQA、OBQA、LAMBADA、SQuAD、TriviaQA。
  - 长任务（4 项）：RULER S-NIAH-1 的 2,048 和 4,096 两档、FDA、SWDE。
- **吞吐**：统一用 `moe.throughput` 测量，每次在同一节点上先后测 baseline 和待测方法，发布两者的相对吞吐。

在观测站查看 [MoE 榜单](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp)和[训练曲线、最终逐任务得分](https://interactive-active-ai.github.io/NextTransformer/#results?track=d768-240tpp)。

完整规范见 [协议](../protocol/)（以英文为准）。
