---
title: 版本与标识符
description: 记录中保存的协议和竞技场标识符，以及它们如何变更。
---

:::note
协议以[英文版](../../../protocol/versions/)为准；中文版仅便于阅读。
:::

| 名称 | 数据中保存的标识符 | 含义 |
|---|---|---|
| Reference Protocol v1 · 50B | `nextt-reference-50b-v1` | 本节各页描述的训练契约 |
| Cost ledger v2 | `reference-aten-math-v2` | [算力核算](../compute/)中的算子级核算 |
| Arena Protocol v1 | `arena-v1` | 自动研究参赛作品的规则，直到 `arena-v2` |
| Arena Protocol v2 | `arena-v2` | Arena Protocol v1 加上 `arena-only` 状态：参赛作品在竞技场展示，但不在排行榜上排名 |

协议标识符是协议 SHA-256 指纹的一部分，每个配置和运行记录都携带该指纹；因此已有协议的标识符永远不会改变。网站和文档显示的是展示名称。

任何受控值的改变都会定义一个新协议，带有新的标识符和指纹。运行、证明和竞技场卡片保留它们创建时的版本，永远不会被改写。文档可以重新组织，而不改变协议。
