---
title: 固定算力下的规模扩展
description: 从 T3 开始比较四个架构家族，先登记预测，再用独立训练检验。
---

给定相同的训练算力，每个家族应该选多大的模型、读多少 tokens？2K 研究从 **T3** 起步，四个家族各扫四个执行深度。每次训练预算为 **10²⁰ FLOPs**。

| 家族 | 研究对象 |
|---|---|
| Vanilla Dense | 各层独立的全注意力与稠密 FFN |
| Vanilla MoE | 全注意力与按 token 激活的专家容量 |
| Dense KDA–DSA | 每三层 KDA 接一层 DSA，所有 FFN 稠密 |
| Fixed Recursive | 每个 token 都两次通过同一组核心层 |

[观测站](https://interactive-active-ai.github.io/NextTransformer/#scaling)可查看各尺寸配置并切换硬件预算。[T3 配置清单](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json)提供机器可读设计，注明 token 数的估算口径与待核算项。

## 阶段顺序

T3 扫描第一档的尺寸曲线，T4 完善拟合，T5 检验用小预算数据事前登记的预测。先给 T5 预测打分，再锁定四家的最终预测，在 C200 做独立盲测。后续配置和预算随各阶段逐步公开。

历史 [4K Dense 记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/scaling/runs/dense_200b.json)提供算力参照。新 2K 拟合只接收本协议下的质量观测。详见[协议](../protocol/)与[参与方式](../contribute/)。
