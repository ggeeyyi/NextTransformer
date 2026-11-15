---
title: Hybrid GDN + SWA
description: H03-SWA512 的配置、完整正式结果、恢复记录与执行证据。
---

H03-SWA512 在 seed 42 下完成 46,875 次更新和 49.152B 输入 tokens。最后一次周期验证 BPB 为 **0.66651494**，独立最终验证 BPB 为 **0.66379960**，**CORE ×100 为 33.089409**。累计训练成本为 **4.685968542 GFLOPs / 输入 token**，相对 SSSL 成本锚点为 **+0.290434%**。只读累计审计通过，包含 189 次验证、6 次完整 CORE-22 评测，记录的预算异常数为零。

[结果记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gdn_hybrid_swa512.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gdn_hybrid_swa512.json) · [可复现审计输入与执行证据](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/runs/gdn_hybrid_swa512)

## 配置与身份

模型包含 24 个独立层，12 个 GDN 与 12 个 SWA 交替排列，宽度 1536、12 个头、FFN 宽度 6016，共 1,403,661,866 个参数。训练上下文为 2048 tokens，512-key 窗口包含当前 token 和最多 511 个前序 tokens。GDN 使用 FLA chunk-64 后端，SWA 使用关闭 FlashAttention 的确定性 SDPA。

实际训练实现为冻结提交 [`4926c5de832968957cd84411b1fa24ca0ada81c4`](https://github.com/Interactive-Active-AI/NextTransformer/tree/4926c5de832968957cd84411b1fa24ca0ada81c4) 中的 `gdn_hybrid_swa`。`gdn_hybrid_swa512` 是本配置的发布标识。归档配置和成本证明保留原始训练方法名与模型哈希。24 层、FFN 6016 的固定候选在正式训练前通过成本核验，证明中没有结构搜索记录。

## 训练与恢复

训练从随机初始化开始，保留的三个分段分别始于第 0、1,000 和 7,500 步，均使用 8 ranks、micro-batch 8、梯度累积 8。各段都记录了正式准入，模型、数据、tokenizer、源码和成本指纹一致。

早期暂停后，在续跑节点恢复了第 1,000 步完整状态。一次启动尝试在训练推进前退出，后续尝试达到第 8,000 步，但 CORE 评测被资源保护器中断。完成有界评测检查后，从第 7,500 步完整检查点恢复。重放的更新替换了规范日志中被舍弃的末段。第 7,510 至 8,000 步各日志点的损失、梯度范数、输入 token 数和计算量与先前尝试完全一致。公开指标保留单条更新及计划评测序列，失败尝试和重放另有执行开销。

## CORE 执行

续跑使用外部评测包装，缓存原始 CORE 样例构造器生成的输入，并按原始输入形状使用 CUDA Graph。任务元数据、输入 tokens、续写评分范围、rank 分配和评分规则均保留。每个形状组的第一次 graph replay 都与原生 forward 的损失和 token 预测进行精确相等检查。证据目录包含包装源码、输入缓存清单以及第 7,500 步检查点的完整 CORE 执行检查。该有界检查与 6 次计划评测分别记录。

补齐长度的批处理和独立进程评测曾因答案差异被否决，相应输出未进入公开结果。源码指纹覆盖冻结仓库代码，外部包装另在 `submission.json` 中记录 SHA256，包括节流、CPU 亲和性、进度记录和并行检查点校验。

## 检查点与适用范围

最终检查点包含模型和全部 8 个 rank 文件。提交准备时重新计算了各文件哈希，并与完整检查点清单及固定 generation 的归档回读收据核对。公开检查点清单包含文件大小与哈希。文件本体保存在贡献者归档中，公开下载链接待补充。

已发表的 2048 窗口配置为 22 层、FFN 6528。本次 512 窗口结果还在共同计算预算下改变了层数和 FFN 宽度，对照包含这三项变化。结果来自单种子、2K 上下文，计时包含执行节流，长上下文质量和部署效率需要单独测量。主榜使用最后一次周期验证 BPB，独立最终 BPB 每个 split 使用 20,971,520 tokens，单独保存。
