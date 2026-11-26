---
title: XSA 和 REL 的完整 CORE prompt 评测
description: 已发布最终 checkpoint 在完整 prompt 设置下的补充 CORE-22 结果。
---

本次使用原始最终 checkpoint，训练长度仍为 2048，不重新训练。CORE 保留完整 prompt 和 BOS，最长实际输入为 5835 token。REL 按 Inkling 原定义，在相对距离超过已训练距离表时把位置 bias 置零，保留普通因果注意力。原始训练结果、审计和 FLOPs 证明保持不变。

详见[英文结果及复现说明](../../../results/xsa-rel-full-context/)。
