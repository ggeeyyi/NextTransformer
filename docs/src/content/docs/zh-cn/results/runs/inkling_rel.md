---
title: Inkling REL
description: 完整 49.152B token 运行、原协议分数及训练后的 BOS 诊断。
---

在 46,875 次更新后，`inkling_rel` 的**周期验证 BPB 为 0.668387**，**最终验证 BPB 为 0.665655**，**最终训练 BPB 为 0.667653**，**CORE ×100 为 33.1733**。未修改评测器下的最终 SQuAD 为 **40.1325%**。

校准配置为 **22 层**、宽度 **1536**、**12 个头**、FFN 宽度 **6272**，共 **1,293,026,934 个参数**。训练成本为每输入 token **4.629592730 GFLOPs**，相对 SSSL 锚点 **−0.91614%**。累计成本与凭证一致，无预算异常。

## 协议与执行

训练使用固定数据、tokenizer、八条数据流顺序、seed 42、2048 token 上下文、Muon + AdamW 调度、49.152B 输入 token 和每次更新 1,048,576 token。189 次验证与六次完整 CORE-22 评测均已完成。64 ranks、microbatch 8、累积 1 的布局先通过训练数据上的完整更新及保存/恢复预检，正式训练单段完成，无训练重试。校准与预检未使用留出集分数选择配置。

**执行布局：**本次运行使用 64 ranks，已由 [#57](https://github.com/Interactive-Active-AI/NextTransformer/pull/57) 登记进运行审计，原版审计命令即可接受，随附审计没有任何问题。`data_layout` 对任意 8 的倍数都驱动同样的八条 `BestFitStream`，且 64 x 8 x 1 x 2048 = 1,048,576 与协议的全局批一致，因此每次更新所见数据与 8、16、32 ranks 的运行完全相同。机器相关启动脚本不属于本次方法实现。

训练时冻结的源码提交为 `5310e8971a9519762f2eca77564561a2fee11494`。插件源码及科学计算指纹与提交的实现一致；导出的记录省略了私有源码目录。原始训练 provenance 与分数文件均未改写。

独立八卡 scorer 加载最终 checkpoint，精确复现全部 22 项 CORE 分数及最终验证 BPB，训练 BPB 差异小于 1.2e-7。独立评测首次因共享存储 stale-file-handle 错误失败，自动第二次尝试成功。本地 finalizer 曾将可重试错误误判为终止，随后已修正，独立分数比较通过。这是评测与编排恢复，没有重启训练。

## BOS 对结果解读的影响

训练 best-fit 数据行以 BOS 开头，但 CORE 左截断使 10,570 个 SQuAD 样本中的 6,895 个丢失 BOS。训练后配对诊断精确复现原始正确数。仅把截断样本保留下来的首 token 替换为 BOS，保持长度及其他 token 不变，全量诊断 SQuAD 升至 **55.2223%**，增加 **15.0899 个百分点**。修改输入后的分数仅用于诊断，不能替换原协议的 SQuAD 或 CORE 分数。

同样恢复 BOS 的 Dense 诊断为 53.6329%，该次诊断原始分数为 50.0378%，比已发表的 50.0000% 多四个正确样本。因此原始 CORE 排名不能单独视为架构质量结论。BOS 也无法解释所有任务：REL 在有限的截断 BoolQ 对照中恢复 BOS 后下降，Language ID 证据不确定，LSAT-AR 没有截断。这里没有声称已定位注意力 sink 回路或 kernel 缺陷。

[完整计数与对照](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/diagnostics/xsa-rel-bos-20261116) · [运行记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/inkling_rel.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/inkling_rel.json)
