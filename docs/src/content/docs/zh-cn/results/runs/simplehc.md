---
title: SimpleHC
description: 完整注意力骨干上的恒等残差传递与四子层共享门控输入：单段完成训练及残差族对比。
---

SimpleHC（`simplehc`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.662833**，独立终验 BPB 为 **0.660070**（训练集为 0.662097），CORE ×100 为 **35.1129**。校准后保持与 Standard Residual、Dense Baseline 和 HC 相同的 23 层完整注意力与 6144 宽 FFN，总参数量为 **1,358,072,020**（相对 SSSL 锚点 +0.441%）。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Standard Residual Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard Residual（`residual_standard`） | 23 | 6144 | 1,355,810,568 | 0.673959 | 0.671214 | 31.9513 | — |
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | −0.00679 / +1.14 |
| Hyper-Connections（`hc`） | 23 | 6144 | 1,356,235,700 | 0.664400 | 0.661683 | 35.0943 | −0.00956 / +3.14 |
| Block AttnRes（`attnres`） | 23 | 6144 | 1,355,882,760 | 0.662963 | 0.660217 | 33.9971 | −0.01100 / +2.05 |
| SimpleHC（`simplehc`） | 23 | 6144 | 1,358,072,020 | 0.662833 | 0.660070 | 35.1129 | −0.01113 / +3.16 |
| mHC（`mhc`） | 23 | 6016 | 1,353,550,818 | 0.661665 | 0.658908 | 35.3278 | −0.01229 / +3.38 |

SimpleHC 去掉可学习的流间混合矩阵（`H_res = I`），保留 mHC 基于完整多流状态的 sigmoid 读门与 `2σ` 写门，并通过组内标量修正让每四个连续子层共用一次门控输入。周期验证 BPB 全程位于 mHC 与 HC 之间：自第 8,000 步起稳定比 mHC 高 0.0010 至 0.0012，终点比 mHC 高 +0.00117，比 Block AttnRes 低 −0.00013，比 HC 低 −0.00157，比 Dense Baseline 低 −0.00434（CORE 高 +2.02 分）。

CORE 在中间评测点落后于 mHC（第 32,000 步为 28.23 对 29.28；第 40,000 步为 32.76 对 33.31），在学习率深衰减末段追回大部分差距，终点比 mHC 低 0.21 分、比 HC 高 0.02 分。第 46,875 步的 22 个子任务上，SimpleHC 与 mHC 各占 11 项优势；相对 Standard Residual 提升最大的是 CommonsenseQA（+27.44 个中心化分点）、Winograd（+16.85）与 SQuAD（+11.15），落后最多的是 BIG-bench Dyck Languages（−5.60）与 AGIEval LSAT-AR（−5.43）。

去掉可学习的流间混合后，HC 中的梯度尖峰随之消失。梯度范数全程有限，最大值出现在第 10 步（3.10）；第 500 步之后中位数为 0.32，最大值为 0.68（HC 中位数为 0.90，最高尖峰达 1,824）。

正式运行在 16 个 rank（两台八卡节点）上单段完成，microbatch 为 8、梯度累积 4 步，开启 `--speedups exact` 与 `--formal`，没有续训。训练基于提交 `0619bc7`（在上游 `df179cb` 上加入本方法插件，早于 [#42](https://github.com/Interactive-Active-AI/NextTransformer/pull/42) 对 `nanochat/flash_attention.py` 中 Hopper FlashAttention 3 加载逻辑的更新）；`main` 上的插件源码、`model_hash`、参数量、算子计数及每 token FLOPs 均未变化。注意力使用 SDPA。

累计训练计费为每 token **4.692986 GFLOPs**（相对 SSSL 锚点 +0.441%），每个记录的更新步都与算力凭证一致，预算异常为零。审计覆盖全部 189 次计划验证和 6 次完整的 CORE-22 评测。正式运行前完成了 8 rank 与 16 rank 下的两次 30 步研究模式 smoke（含第 20 步存盘与续训至第 30 步），无候选筛选或失败的正式运行尝试。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/simplehc.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/simplehc.json)
