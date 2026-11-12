---
title: 参与规模扩展
description: 社区可以为规模扩展阶段做什么，以及如何训练阶梯中的一格、新增家族或登记预测。
---

每一项贡献都是仓库中的一个文件，通过维护者审阅的 pull request 加入。观测站的 [Scaling 页面](https://interactive-active-ai.github.io/NextTransformer/#scaling)展示了阶梯以及哪些格子开放认领。

| 贡献 | 算力 | 提交内容 |
|---|---|---|
| [训练阶梯中的一格](#训练阶梯中的一格) | T3 一格约 10²⁰ 训练 FLOPs，T4 约 2.3 × 10²⁰ | 这一格的配置、记录和检查点 |
| [新增架构家族](#新增架构家族) | 配置和成本核对；家族的格子之后像其他格子一样训练 | 方法、尺寸规则和该家族的格子 |
| [登记预测](#登记预测) | 不需要 | C200 运行开始前提交一份预测文件 |
| [评测已发布的检查点](#评测已发布的检查点) | 推理 | 已发布检查点的评测结果 |
| [改进拟合](#改进拟合) | 一台笔记本 | 代码或分析 |
| [复现一格](#复现一格) | 一格，或只重跑检查则不需要 | 第二个种子的记录，或检查报告 |

## 训练阶梯中的一格

下面以 `dense-T3-L24` 为例：24 层、宽 1,536，共 16,153,313,280 个 token，15,405 次更新，每次 1,048,576 个 token。每一格的尺寸和 token 数都在 `results/scaling/ladder.json` 中。

**1. 认领。** 提交一个 [scaling cell claim](https://github.com/Interactive-Active-AI/NextTransformer/issues/new?template=scaling_cell.yml)，写明格子编号和算力计划。维护者会把这一格标为已认领；两周没有进展的认领会被释放。标为等待配方门槛的格子暂时不能认领。

**2. 准备。** 检出 `main`，按[数据指南](../../getting-started/data/)准备锁定版本的 ClimbMix 和分词器，并按 `launch/scaling/README.md` 写好站点文件。多个节点没有共享文件系统时，每个节点都要有自己的一份。

**3. 配置。** 复制参照配置，只改尺寸、token 数、批大小和协议名：

```bash
cp configs/scaling/dense_200b.json configs/scaling/dense-T3-L24.json
# model: dense_layers 24, width 1536, heads 12, mlp_width 6144
# train: protocol "nextt-scale-4k-v1", total_tokens 16153313280, global_batch_tokens 1048576
python -m nexttransformer.scaling.train --config configs/scaling/dense-T3-L24.json --dry-run
```

dry run 会打印训练计划：核对步数与这一格一致，训练 FLOPs 与档位预算相差不超过 1%。

**4. 训练。** 在每个节点上启动看门狗；任何中断之后，它都会从最新的检查点续训。

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 OUTPUT_DIR=$RUNS/dense-T3-L24 \
  launch/scaling/watchdog.sh configs/scaling/dense-T3-L24.json --site-config site.local.json --wandb-metrics full
```

任何能训练这套配方的 GPU 都可以，预算按 FLOPs 计算。`MICRO_BATCH` 设定每张卡的序列数，只影响速度和显存。

**5. 导出。** 在保存事件记录的 node 0 上执行：

```bash
python -m nexttransformer.scaling.export $RUNS/dense-T3-L24 --config configs/scaling/dense-T3-L24.json \
  --id dense-T3-L24 --name 'Dense · 24 layers · T3' --family dense \
  --wandb <W&B run URL> --checkpoint <Hugging Face model URL>
python -m nexttransformer.scaling.export --check
```

导出会拒绝未完成的运行。请先发布最终检查点 `checkpoint_<steps>/node_000/model.pt` 再导出：记录不可修改，其中的链接必须是最终的。

**6. 提交 pull request**，包含 `configs/scaling/dense-T3-L24.json`、`results/scaling/runs/dense-T3-L24.json`，以及 `results/scaling/ladder.json` 中这一格的更新（`"record": "dense-T3-L24"` 和你的认领信息）。提交前运行 `python -m nexttransformer.checks.public`：记录、配置和文档中不能出现 GPU 型号、主机名、路径或私人邮箱。合并后，这一格显示为已记录，运行出现在三维图中。

## 新增架构家族

1. 先按[插件指南](../../guides/plugins/)在共享协议下登记方法。
2. 在 issue 中提出尺寸规则：一个尺寸旋钮（层数），以及其余维度如何由它推出。
3. 扩展 `nexttransformer/scaling/` 以构建该家族；目前规模扩展训练器只构建 Dense 骨干。
4. 在 T3 和 C200 规模上通过[稳定性门槛](../protocol/#架构家族)。
5. 按[预算口径](../protocol/#预算)把该家族及其 T3、T4 的十个格子加入 `results/scaling/ladder.json`。之后这些格子像其他格子一样开放认领。

## 登记预测

根据已发布的记录，预测某个家族在 C200 的计算最优尺寸及其最终验证 BPB 和 CORE，并给出 90% 区间。按[规定格式](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/forecasts)添加 `evidence/forecasts/<family>-C200-<author>.json`。它必须在该家族第一次 C200 运行开始前合入；在此之前，阶梯上这一格显示为等待预测。预测不可修改，每一份都会与实测结果对照打分。

## 评测已发布的检查点

每份记录都链接了它的检查点。欢迎在这些检查点上做长上下文检索、RAG 式问答和更多下游任务的评测。请添加 `results/scaling/evals/<record>-<benchmark>.json`，写明基准及其版本、设置、分数，以及所用代码的链接。

## 改进拟合

拟合方法见[规模扩展协议](../protocol/#拟合)。更好的估计方法、不确定性估计或对计算最优前沿的检查，可以以 issue 的形式提出，也可以作为带测试的代码提交到 `nexttransformer/scaling/`。

## 复现一格

用另一个种子重训一个已记录的格子：以第二种子运行的方式认领，在配置中设置 `train.seed`，并以 `<cell>-seed<k>` 为编号导出。重跑 `python -m nexttransformer.scaling.export --check`、对照记录和它的 W&B 运行，都不需要 GPU。
