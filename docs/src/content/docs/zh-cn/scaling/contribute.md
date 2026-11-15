---
title: 参与 T3 研究
description: 选择计划点、核对实现与成本，并提交可复核的记录。
---

从[观测站](../../../../#scaling)选择家族与深度，下载 [T3 配置清单](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json)。例如 `dense-T3-L20` 为 20 层、宽 1,280、10 个头。按主项估算训练 29,433 步，约 30.86B tokens；正式值以实际成本证明为准。

## 四个家族的参考代码

在[网站的 Contribute 入口](../../../../#contribute)选择任一 T3 配置，即可查看参考代码、待接入的工作和协作 issue。建议先用 `dense-T3-L20` 验证各家族共用的训练与评测流程。

| 家族 | 可复用的代码 | T3 还需要的贡献 |
| --- | --- | --- |
| Vanilla Dense | [全注意力骨干](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/backbone.py)和[关闭 VE 的基准配置](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/others/mha_no_ve.json) | 按所选宽度和深度接入全因果注意力，验证 FLOPs、断点恢复和评测计划。 |
| Vanilla MoE | [`RoutedFFN`](../../../../#contribute?cell=moe-T3-L20) 实现归一化 top-k 路由和一个共享的 ReLU² 专家。分布式实现可对照 [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py)。 | 首层保留 Dense FFN，后续层接入 256 个路由专家，每 token 选 8 个，另加 1 个共享专家；用预实验验证负载均衡策略和开销。 |
| Dense KDA–DSA | [`KdaDsaGroup`](../../../../#contribute?cell=kda-dsa-T3-L20) 将三个 [KDA 层](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/linear.py)与一个 [DSA 层](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/sparse.py)组合，每层各接一个 Dense FFN。 | 按目标深度重复该组并接入索引器损失。分别核算前 512 次更新的预热成本和后续稀疏阶段成本。 |
| Fixed Recursive | [共享核心层的执行代码](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/lm.py)和[现有基准配置](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/looped/fixed_recursive.json) | 核心层用同一份权重执行两次；将旧配置改为全注意力并关闭 VE，计入两次执行的前向与反向成本。 |

### 在 CPU 上运行示例

在仓库根目录[安装项目依赖](../../getting-started/installation/)后运行：

```sh
NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 python -m nexttransformer.scaling.reference_blocks
```

示例用合成输入对已有的 Dense、Fixed Recursive 基准模型以及新增的两个模块执行前向和反向。MoE 的小例子从 8 个专家中选 2 个，KDA–DSA 会分别运行预热和稀疏阶段。正式 T3 实验还需要完成训练器接入、成本核算和吞吐实测。

## 认领与提交

1. 在 [T3 协作 issue](https://github.com/Interactive-Active-AI/NextTransformer/issues/new?template=scaling_cell.yml) 标注配置编号、想参与的工作及可用资源。安排算力前，与维护者确认实现和证明。
2. 按 [2K 协议](../protocol/)核对各家族的内核、数据前缀、显存、恢复行为与阶段成本，区分规划吞吐和实测吞吐。
3. 正式运行锁定经核对的实现、配置、证明与数据指纹，固定 seed 42，完整保留计划中的评测。记录独立最终 BPB、完整 CORE-22、实际 tokens 与累计 FLOPs，并附检查点和来源链接。

也可参与拟合分析、审计和已发布模型的额外评测。后续各档随配置公布而开放。
