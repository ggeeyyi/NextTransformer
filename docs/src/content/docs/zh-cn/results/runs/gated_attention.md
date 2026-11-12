---
title: Gated Attention · elementwise
description: 等算力下的论文最佳门控：21 层、单段训练，以及早期优势如何消失。
---

Gated Attention · elementwise（`gated_attention`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.669459**，独立终验 BPB 为 **0.666731**（训练集为 0.668827），CORE ×100 为 **33.5841**。在 Dense 的 23 层下，逐通道门要占共享预算的 7.0%，因此校准把深度降到 21 层、FFN 加宽到 6272，总参数量为 **1,306,658,420**，比 Dense 少一张值嵌入表。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Dense Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| Gated Attention · headwise（`gated_attention_headwise`） | 23 | 6144 | 1,356,234,504 | 0.667417 | 0.664678 | 34.2736 | +0.00022 / +1.18 |
| Gated Attention · elementwise（`gated_attention`） | 21 | 6272 | 1,306,658,420 | 0.669459 | 0.666731 | 33.5841 | +0.00227 / +0.49 |

门控在前期占优，深度减少的代价在后期显现。第 250 步周期验证 BPB 比 Dense 低 0.0178，第 5,000 至 24,000 步低 0.0007 至 0.0038。两条曲线在学习率衰减阶段的第 32,000 步附近交叉；第 33,000 步之后门控模型一直高于 Dense，终点高 0.0023。

CORE 的差距主要来自一个任务。Dense 在终点的 CommonsenseQA 中心化分数为 5.1，在已发布运行中属于最低的一档；本运行为 30.5。去掉 CommonsenseQA 后，其余 21 个任务的均值为 33.73，Dense 为 34.43（−0.70）。逐任务落后最多的是 AGIEval LSAT-AR（−6.52 个中心化分点）、COPA（−6.00）和 BIG-bench CS Algorithms（−4.17）；除 CommonsenseQA 外提升最多的是 BoolQ（+10.86）。

正式运行在 16 个 rank（两台八卡节点）上单段完成，microbatch 为 8、梯度累积 4 步，没有续训。骨干为运行记录中的提交 `41b6016`，在其上加入本方法的插件；cost proof 中的方法源码哈希 `3ec6f564f745` 标识插件源码。注意力使用 SDPA。门控 W_θ 是独立的层，由 AdamW 辅助组训练；官方代码把它并入 `q_proj`。

累计训练计费为每 token **4.641914 GFLOPs**（相对 SSSL 锚点 −0.652%），每个记录的更新步都与 cost proof 一致，预算异常为零。审计覆盖全部 189 次计划验证和 6 次完整的 CORE-22 评测。梯度范数全程有限；第 500 步之后中位数为 0.30，最大值为 0.65。

正式运行前通过了单卡 2 步 smoke，以及 16 rank 下训练 10 步、保存后续训到 20 步的 smoke。正式运行之前还做过一轮 500 步的研究性筛选，比较了三种门控变体。它使用不同的参数化（仓库外的独立门控层，零初始化，由 Muon 训练），并比较了验证 BPB。两种注册的粒度来自论文，优化器选择依据论文的 AdamW 训练；筛选没有用于在两者之间做选择。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gated_attention.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gated_attention.json)
