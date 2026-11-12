---
title: Gated Attention · headwise
description: 在 Dense 原有骨干上为每个头加一个门：等算力下的配对比较。
---

Gated Attention · headwise（`gated_attention_headwise`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.667417**，独立终验 BPB 为 **0.664678**（训练集为 0.666714），CORE ×100 为 **34.2736**。门控只增加 0.42M 参数和 0.06% 算力，校准保留了 Dense 的骨干：23 层、FFN 6144，总参数量为 **1,356,234,504**。它与 Dense 的配对单独衡量逐头门控的作用。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Dense Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| Gated Attention · headwise（`gated_attention_headwise`） | 23 | 6144 | 1,356,234,504 | 0.667417 | 0.664678 | 34.2736 | +0.00022 / +1.18 |
| Gated Attention · elementwise（`gated_attention`） | 21 | 6272 | 1,306,658,420 | 0.669459 | 0.666731 | 33.5841 | +0.00227 / +0.49 |

第 250 步周期验证 BPB 比 Dense 低 0.0046，前 10,000 步最多低 0.0036。第 16,000 步起两条曲线相差不超过 0.0009，第 28,000 步起不超过 0.0003；终点 BPB 比 Dense 高 0.00022。

CORE 的差距来自 CommonsenseQA。Dense 在终点的 CommonsenseQA 中心化分数为 5.1，在已发布运行中属于最低的一档；本运行为 34.2。去掉 CommonsenseQA 后，其余 21 个任务的均值为 34.28，Dense 为 34.43（−0.15）。逐任务落后最多的是 AGIEval LSAT-AR（−10.33 个中心化分点）；除 CommonsenseQA 外提升最多的是 COPA（+4.00）和 SQuAD（+3.52）。

正式运行在 16 个 rank（两台八卡节点）上单段完成，microbatch 为 8、梯度累积 4 步，没有续训。骨干为运行记录中的提交 `41b6016`，在其上加入本方法的插件；cost proof 中的方法源码哈希 `c37a3eb357a1` 标识插件源码。注意力使用 SDPA。门控分数来自独立的层，由 AdamW 辅助组训练；官方代码把它们附加在 `q_proj` 的输出中。

累计训练计费为每 token **4.677384 GFLOPs**（相对 SSSL 锚点 +0.107%），每个记录的更新步都与 cost proof 一致，预算异常为零。审计覆盖全部 189 次计划验证和 6 次完整的 CORE-22 评测。梯度范数全程有限；第 500 步之后中位数为 0.28，最大值为 0.78。

正式运行前的 smoke 和 500 步研究性筛选见 [elementwise 运行](../gated_attention/)。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gated_attention_headwise.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gated_attention_headwise.json)
