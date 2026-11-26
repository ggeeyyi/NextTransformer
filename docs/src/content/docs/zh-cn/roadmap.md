---
title: 路线图
description: NextTransformer 接下来的计划。
---

## Arena v2：agent 原生研究

Arena v2 已上线。研究 gym 在 [`gym/`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/gym/README.zh-CN.md) 中发布；下面的实验计划是它接下来要做的事。

1. **用 Harbor 格式组织 auto-research 任务。**
   - 仓库本身作为 gym 的 sidecar 服务，向模型提供接口。
   - 模型在一个封闭的容器内开发。
   - agent 可以访问仓库的全部历史记录，也可以上网搜索，找出最好的架构。
2. **研究与正式评测分开。**
   - 研究使用 ClimbMix 训练数据和 dev 分片 `shard_06541.parquet`。
   - 正式验证集 `shard_06542.parquet` 和 CORE-22 不参与候选选择，仅用于正式运行报告。
3. **最终提交只是一份文件。** 之后由 gym 完成最终训练。

4. **Gym 接口。**
   - **接入方式**：容器内的 `/run/gym-auth/credential.json` 提供 endpoint 和 token。每次调用是一个 JSON `POST /v1/call`，带操作名和参数；Python 的 `GymClient` 做了封装，不导入任何训练代码。
   - **信息**：`describe_space`、`list_components`、`read_reference`、`get_resources`，查看搜索空间、可用组件、只读参考代码和资源上限。
   - **预检测**：`validate_candidate`、`calibrate_candidate`、`debug_candidate`、`inspect_cost`，在消耗研究算力前检查候选能否加载、FLOPs/token 和参数量是否合规。
   - **固定实验**：`run_experiment` 在固定配置下训练候选，可以从检查点续训；用 `get_job`、`get_result`、`cancel_job` 跟踪。
   - **预算**：`quote_experiment` 在运行前报价，`get_budget` 查看剩余额度。每次实验都从固定的研究算力额度中扣除，用完即停。
   - **提交**：`finalize` 把一个候选冻结为最终文件。

### Gym 实验计划

- **模型与 harness：** Opus、GLM、DeepSeek，统一使用 Claude Code，通过 Gym MCP 独立研究。开跑前固定模型和客户端版本。
- **初始设置：** 使用相同代码与参考资料快照、ClimbMix 数据顺序、分词器、2,048-token 上下文、seed 42、每步 1,048,576 tokens 和原优化器／学习率协议。研究只使用训练 loss 和 dev；正式验证集与 CORE-22 保留给正式评测。
- **研究方式：** 初始架构、对照方法、候选数量、每次实验长度、搜索和淘汰策略均由模型自行决定，不预设训练步骤或搜索轮数。
- **模型调用：** 不设费用上限，记录 token 和实际费用。
- **研究算力上限：** 每个模型的一次研究会话统一为 **1.96×10²⁰ FLOPs**，训练、评估、检查及失败尝试共同扣除，不按候选分别配额。三种模型各跑一次，研究总上限为 **5.88×10²⁰ FLOPs**；最终正式训练另计。
- **预算依据：** Arena v1 Stage 2 共 17 次尝试（含取消和重试）；以 20 次锚点 2,000 步训练的等效计算量估算总额度：`20 × 2,000 × 1,048,576 × 4.67239832544785e9 ≈ 1.96×10²⁰ FLOPs`。这只是预算换算，不要求模型跑 20 次或每次跑 2,000 步，也不是 v1 的实测 FLOPs 账本。
- **排期参考：** v1 搜索与移植核对共 `549.7 + 42.1 = 591.8` GPU 时，加 20% 余量，**每个模型约 710 GPU 时，同类 8 卡环境约 89 小时**。实际时间取决于模型选择的实验及吞吐，GPU 时不是额外的硬上限。
- **目标：** 比较各模型在相同研究算力内自主找到的结构及其收益，最终提交按同一正式训练协议验证。

### Arena 结果展示

先做一个[前端页面示例](../../../arena-example.html)：上方展示模型、harness、研究预算和状态；中间展示候选 dev 结果及相对基线的改善；下方展开结构改动、正式结果和证据说明。
示例使用 Arena v1 历史研究数据，Opus、GLM、DeepSeek 标为待运行，正式成绩留空；后续再接真实运行记录。

## 2K 规模扩展

T3 在固定预算下比较 Vanilla Dense、Vanilla MoE、Dense KDA–DSA 和 Fixed Recursive 的尺寸与训练量。使用 2K 窗口、固定 seed 42，公布 T3 配置和预算。后续按 T4 拟合、T5 预测检验、C200 最终盲测推进。详见[规模扩展总览](../scaling/overview/)。

## MoE 赛道

[MoE 赛道](../moe/overview/)把稀疏模型放进同一套约定，设两个 Marin 口径的尺寸。[240 TPP 对照](../../moe/protocol/#240-tpp-comparison)已发布：d768 准入 baseline、HC、mHC 和 SimpleHC，训练 14.07B 输入 token；d1024 准入 baseline，训练 34.06B。接下来是正式训练，用 MoE-CORE 计分，并通过与其他贡献相同的[结果 pull request](../guides/submit-a-result/) 发布。更多尺寸或更长预算会作为新赛道加入，各自带有自己的指纹。

## 即将发布的结果与证据

- **Hybrid 结果：** GDN + QSA。模型卡片已经在观测站中，标注为“Results coming soon”；每个结果都通过与其他贡献相同的[结果 pull request](../guides/submit-a-result/)发布。
- **审计：** looped 和 sparse 运行的独立运行目录审计；已发布 hybrid 的 pilot 和续训验收证据。

## 计划中

- **在 Hugging Face Hub 上发布检查点和分词器。**
- 把 Others 的实现（MQA、GQA、MLA）移植到当前代码。

欢迎在 [Discussions](https://github.com/Interactive-Active-AI/NextTransformer/discussions) 提出想法。
