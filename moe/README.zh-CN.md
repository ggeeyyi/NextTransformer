# NextTransformer-MoE

[240 TPP 模型榜单](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp) · [训练曲线与任务分数](https://interactive-active-ai.github.io/NextTransformer/#results?track=d768-240tpp)

面向 **Mixture-of-Experts** 语言模型的算力受控对比平台。沿用 NextTransformer 的契约：数据相同、顺序相同、每 token 训练
FLOPs 相同、评测相同；参数量和速度是结果，不是约束。平台设两个固定赛道，形状与预算复现 Marin 的 compute-optimal
Grug MoE：

| 赛道 | 结构 | 上下文 | 训练 token | 每 token FLOPs | 训练总 FLOPs | 总参数 / 激活参数 |
|---|---|---|---|---|---|---|
| **d768** | 8 层 × 768，GQA 6/1，256 个宽 384 的专家选 4，另有宽 768 的共享专家 | 4,096 | 4.42B（16,875 × 262,144） | 9.369 × 10⁸ | 4.14 × 10¹⁸ | 2.00B / 213M |
| **d1024** | 11 层 × 1024，GQA 8/2，256 个宽 512 的专家选 4，另有宽 1024 的共享专家 | 4,096 | 8.43B（16,080 × 524,288） | 1.695 × 10⁹ | 1.43 × 10¹⁹ | 4.71B / 347M |

[NextTransformer 240 TPP](PROTOCOL.md#240-tpp-comparison) · d768 baseline / HC / mHC / SimpleHC / iHC (14.07B tokens) · d1024 baseline (34.06B tokens).

- **模型与训练**：按 Marin 跑出这两个结果时的 commit 逐行移植。
  - 组件：GatedNorm、XSA、head 门控、half-RoPE + PKO、文档掩码、QB 路由。
  - 优化器：MuonH / AdamH / Adam，学习率公式和调度同 Marin。
  - 唯一的差异是分词器和数据。
- **数据**：[`xjxyys/dolma3_hq_300B_shuffled_clean`](https://huggingface.co/datasets/xjxyys/dolma3_hq_300B_shuffled_clean)，固定 revision。
  - 分词器为 dolma2（tiktoken 实现，与 HF 逐 id 一致）。
  - shard 按固定顺序读；00255 作开发集；验证集保留给正式评测。
- **总分 MoE-CORE**：对 14 个已发表任务的中心化准确率取平均，每项等权。
  - 短任务（10 项）：ARC-E/C、HellaSwag、PIQA、WinoGrande、CSQA、OBQA、LAMBADA、SQuAD、TriviaQA。
  - 长任务（4 项）：RULER S-NIAH-1 的 2K 和 4K 两档、FDA、SWDE。
- **吞吐**：统一由 `moe.throughput` 测量，每次在同一节点上先后测 baseline 和方法，发布两者的相对吞吐。
  - 8 卡实测：d768 为 686k tokens/s（MFU 8.1%，跑完 1.8 小时），d1024 为 536k tokens/s（MFU 11.5%，4.4 小时），micro-batch 都是 8。
- **算力控制**：matmul FLOPs 逐 block 实测，注意力按边数规则计费；准入要求与 anchor 相差不超过 1%，整个训练累计不超过 2%。

规范文本见 [PROTOCOL.md](PROTOCOL.md)，提交流程见 [README.md](README.md#submitting-a-method)。提交流程与原
NextTransformer 一致：插件 → 校准 → `--formal` 训练 → 吞吐测量 → 审计与导出 → PR。
