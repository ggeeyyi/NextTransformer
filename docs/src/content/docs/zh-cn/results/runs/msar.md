---
title: MSAR
description: Multi-Stream Attention Residuals 方法及其 NextTransformer 正式评测结果。
---

[Multi-Stream Attention Residuals（MSAR）](https://arxiv.org/abs/2610.13630)
由 Xiaohan Qin 等人提出，是结合多流深度注意力、时间上下文增强与自适应流混合的
残差架构。本页报告 MSAR 在 NextTransformer 统一协议下的评测结果。

## 基准结果

MSAR 在 seed 42 下完成 46,875 次更新、49.152B 输入 tokens，最终周期验证
BPB 为 **0.661044**，独立最终验证 BPB 为 **0.658278**，**CORE x100 为
36.7452**。模型为 23 层、hidden 1,536、FFN 6,144，共 1,358,888,937 个参数。

本次实验使用四条时间增强流、三层深度 block、softmax depth read、每流一个标量的
HPre、two-phase read、首层 attention 直接读取 embedding，以及最后 MLP 输入
加输出的读出。保留参考全注意力骨干、ReLU-squared FFN、value embeddings、
smear 和训练配方。

累计审计覆盖全部 189 次周期验证和 6 次完整 CORE-22 评估，结果为 `matched`，
无审计问题及预算异常。每输入 token 的训练成本为 **4.716253378 GFLOPs**，
比算力锚点高 **0.938598%**，累计 **2.3181328602510056e20 FLOPs**。

正式训练使用 32 ranks、microbatch 8、梯度累积 2，在 step 500 和 1,000 恢复。
训练源码为 `538653b` 加 MSAR 实现的冻结快照，attention 使用 SDPA。
发布版更新了上游代码并精简配置格式，源码指纹、数值对比和 checkpoint 加载检查见
[源码说明](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/evidence/runs/msar/README.md)。

[运行记录](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/results/runs/msar.json) · [累计审计](https://github.com/jianke0604/NextTransformer/blob/feat/msar-release/evidence/audits/msar.json)
