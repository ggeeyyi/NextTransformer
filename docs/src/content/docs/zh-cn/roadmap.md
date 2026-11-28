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
   - agent 还可以读所有已发布的结果和运行（包括 Arena v1 的研究运行）。历史在会话开始时冻结，之后发布的结果不算新反馈；网络访问会被记录。
2. **研究与正式评测分开。**
   - 研究使用 ClimbMix 训练数据和 dev 分片 `shard_06541.parquet`。
   - 正式验证集 `shard_06542.parquet` 和 CORE-22 不参与候选选择，仅用于正式运行报告。
   - 研究过程中 agent 的优化目标是 dev BPB；从不为它的候选评测 CORE-22，最终训练的 CORE 在会话关闭前一直封存。
3. **最终提交只是一份文件。** 之后由 gym 完成最终训练。

4. **Gym 接口。**
   - **接入方式**：容器内的 `/run/gym-auth/credential.json` 提供 endpoint 和 token。每次调用是一个 JSON `POST /v1/call`，带操作名和参数；Python 的 `GymClient` 做了封装，不导入任何训练代码。
   - **信息**：`describe_space`、`list_components`、`read_reference`、`get_resources`，查看搜索空间、可用组件、只读参考代码和资源上限。
   - **预检测**：`validate_candidate`、`calibrate_candidate`、`debug_candidate`、`inspect_cost`，在消耗研究算力前检查候选能否加载、FLOPs/token 和参数量是否合规。
   - **固定实验**：`run_experiment` 在固定配置下训练候选，可以从检查点续训；用 `get_job`、`get_result`、`cancel_job` 跟踪。
   - **预算**：`quote_experiment` 在运行前报价，`get_budget` 查看剩余额度。每次实验都从固定的研究算力额度中扣除，用完即停。
   - **提交**：`finalize` 把一个候选冻结为最终文件。
   - **效率**：`inspect_cost` 和 `get_result` 同时报告训练效率和推理效率。
   - **决策**：`record_decision` 保存 agent 的研究计划、预登记规则、假设和每次选择，由服务端打时间戳。
5. **以 Arena v1 为基础的研究工作流。**
   - Arena v1 完成了一次有人监督的研究过程：工具验收、代理规模研究、三轮搜索，以及胜出者的正式训练。结论是：512 步可以筛掉候选，但不能排名；2,000 步的训练前缀可以排名。
   - 工作流分为：熟悉情况、预登记、锚点和噪声、生成候选、逐轮淘汰（512 → 2,000 → 更长的步数）、确认、冻结提交。
   - 开始训练前，agent 通过 gym 冻结自己的规则：筛选和排名阈值、噪声估计方案、效率门槛、预算分配和停止规则；任何修订都要在受影响的运行开始之前记录。
   - 每个候选相对一个指定的父方案只做一项主要改动，并写下预测。
6. **训练效率和推理效率。**
   - 训练：每步耗时、每卡 tokens/s、MFU 和峰值显存，并写明所用的硬件和设置。
   - 推理：参数量、prefill 和 decode FLOPs、每 token 的 KV cache 或状态字节数，以及每解码一个 token 要读的字节数；每个数值都标明是实测还是估计（[#19](https://github.com/Interactive-Active-AI/NextTransformer/issues/19)）。
   - 训练前先检查效率门槛；差距在噪声以内的候选，效率更高的胜出。
7. **完整的研究轨迹。**
   - 每次会话保存一个只能追加的轨迹包：模型调用、工作区快照、网络访问、gym 调用、决策和人工操作。
   - 披露卡（包括 LLM token 用量和费用）由它自动生成。

后续扩展：

- **Anti-hack 检测：** 检查 auto-research 系统最终轨迹里有没有 hacking 行为。
- **视觉看板：** 展示系统如何从初始架构一步步进化到最终结果。
- **跨会话的自我改进：** 只有新版本在相同预算下比旧版本找到更好的架构时，才改变 agent 的搜索方式。

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

[Arena 页面](../../../arena-example.html)根据 `results/arena/sessions/` 中的记录展示研究会话：上方是模型、harness、已用算力、token 和最终提交；中间是每个会话的作业时间线、按训练步数的 dev 结果，以及同一步数下各候选的排名；下方是每个候选的改动和假设，以及提交方案的正式训练结果。

[竞技场与 gym 计划](../arena/plan/)跟踪这些条目背后的工作、进度和待决定的事项。

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
