---
title: MoE 协议
description: 两个 MoE 赛道（d768、d1024）的固定契约：Marin recipe、预算、算力、吞吐与 MoE-CORE。
---

以英文版 [MoE protocol](../../../moe/protocol/) 为准；机器可读数值见
[`moe/spec.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/spec.py)。

## 两个赛道

两个赛道完整复现 Marin "May Recipe" compute-optimal Grug MoE 单元（`moe_may_compute_opt_d768_ep1`、
`moe_may_compute_opt_d1024_ep1`），按启动 commit `d259a97068` 逐行移植。

| | **d768** | **d1024** |
|---|---|---|
| 宽度 / 层数 | 768 / 8 | 1024 / 11 |
| 注意力 | 6 个 query 头，1 个 KV 头，头维度 128 | 8 个 query 头，2 个 KV 头，头维度 128 |
| 全注意力层（其余层为 2,048 滑窗） | 3、7 | 3、7、10 |
| 路由专家 | 256 个 SwiGLU，宽 384，选 4 | 256 个 SwiGLU，宽 512，选 4 |
| 共享专家 | 1 个 SwiGLU，宽 768 | 1 个 SwiGLU，宽 1024 |
| 总参数 / 激活参数 | 1,996,408,320 / 212,780,544 | 4,707,434,496 / 347,455,488 |
| 激活参数（不含 embedding 和 head） | 58,639,872（与 Marin 一致；Marin 另把 router bias 算作参数） | 141,934,592（同左） |
| 每步 token | 64 × 4,096 = 262,144 | 128 × 4,096 = 524,288 |
| 更新步数（warmup） | 16,875（168） | 16,080（160） |
| **训练 token** | **4,423,680,000** | **8,430,551,040** |
| **每 token 训练 FLOPs（anchor）** | **936,884,736** | **1,694,558,208** |
| 峰值学习率：MuonH / Adam | 0.008373 / 0.001932 | 0.008791 / 0.002029 |
| MoE-CORE 评测步 | 4,000 / 8,000 / 12,000 / 16,875 | 4,000 / 8,000 / 12,000 / 16,080 |

[NextTransformer 240 TPP](../../../moe/protocol/#240-tpp-comparison) · d768 baseline / HC / mHC / SimpleHC / iHC (14.07B tokens) · d1024 baseline (34.06B tokens).

## 模型与优化器（与 Marin 一致）

- **结构**
  - 每个 block 先做 RMSNorm，再接 GatedNorm（`x·sigmoid(silu(xW_d)W_u)`，rank 128）。
  - 长层用 PKO，QK 做 RMS norm，half-RoPE（只转前 64 维）。
  - q × 1.3；注意力只看同一文档；短层另加 2,048 滑窗。
  - XSA；head 级门控 `2·sigmoid(aG)`。
  - embedding 与 head 不共享。
- **路由**
  - Marin 的 QB（quantile balancing）：bias = −β̄ + mean(β̄)。
  - 组合权重是 sigmoid 后归一化到和为 2.5。
  - 无辅助损失，无 z-loss。
- **初始化**：所有矩阵用截断正态，σ = 0.5/√d；head 级门控初始化为 0。
- **优化器**
  - 矩阵（含 GatedNorm，以及每个专家）用 MuonH：Nesterov 动量 0.95、5 步 quintic Newton–Schulz，再做 Frobenius 超球更新，使每个矩阵的范数保持不变。
  - head 用 AdamH；embedding、router、门控和各 gain 用 Adam。
  - 学习率按 Marin 的经验公式；warmup 1%，之后线性衰减到 0。
  - 无 weight decay，无梯度裁剪。
- **与 Marin 的差异**：
  - 分词器是 dolma2（Marin 用 Llama 3）。
  - 数据是 Dolma 3 HQ（Marin 用 Nemotron-CC）。
  - 文档从 `<|endoftext|>` 分隔符开始。

## 评测：MoE-CORE

对 14 项中心化准确率取平均，每项等权。所有任务都来自已发表的工作，格式和打分方式也沿用原文。

| 项 | 来源 / 格式 | 打分 |
|---|---|---|
| ARC-E、ARC-C、HellaSwag、PIQA、WinoGrande、CSQA、OBQA、LAMBADA、SQuAD | 稠密赛道固定的 DCLM 题包，用 CORE 的格式和 few-shot 设置（OLMoE / OLMES / OLMo 3 / DataDecide 共用的核心任务） | 选择题取续写平均 loss 最低的选项；短答案要求每个 token 都是贪心 argmax |
| TriviaQA | DCLM 的 `triviaqa`，3-shot（Signal & Noise 认为它在小模型上信噪比高） | teacher forcing 下的精确匹配 |
| RULER S-NIAH-1 的 **2,048** 和 **4,096** 两档（算两项） | 按 lm-eval 的 `niah_single_1` 生成，每档 500 题，haystack 按 dolma2 分词定长 | 贪心生成，预测里包含目标值即算对；最多生成 32 个 token（RULER 原文是 128） |
| FDA、SWDE | Based / JRT 的信息抽取任务，用 lm-eval 的 `fda` / `swde` | 贪心生成到换行（最多 48 个 token），预测包含答案即算对 |

另报 RULER 8,192 长度作外推探针，不计分。

## 吞吐

`moe.throughput` 直接复用训练器的 update：
- 10 步预热，30 步计时，去掉首尾各 10%。
- 每次在同一节点上先后测 baseline 和待测方法，发布两者的相对吞吐。
- 8 卡实测（141 GiB 卡，torch 2.9.1+cu128）：

| | d768 | d1024 |
|---|---|---|
| micro-batch | 8 | 8 |
| tokens/s（每卡） | 686,005（85,751） | 536,017（67,002） |
| MFU | 8.1% | 11.5% |
| 跑完整预算的墙钟 | 1.8 小时（14.3 卡时） | 4.4 小时（35.0 卡时） |

d1024 在 141 GiB 的卡上 micro-batch 16 会 OOM。完整数值见 `references.json`。
