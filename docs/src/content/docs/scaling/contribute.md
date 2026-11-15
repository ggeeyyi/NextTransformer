---
title: Contribute to the T3 sweep
description: Review a planned cell, verify its implementation and cost, and record the result.
---

Pick a family and depth in the [T3 explorer](../../../#scaling) and download its [design manifest](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/scaling/t3.json). For example, `dense-T3-L20` has 20 layers, width 1,280 and 10 heads. Its leading-term estimate is 29,433 updates, or 30.86B tokens. These are planning figures until checked against the actual run.

## Model reference code

The [Contribute tab](../../../#contribute) opens the reference links, implementation work and coordination issue for any of the 16 T3 designs. Start with `dense-T3-L20` for the shared training and evaluation checks.

| Family | Start in the repository | T3 work to contribute |
| --- | --- | --- |
| Vanilla Dense | [Full-attention backbone](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/backbone.py) and [no-VE benchmark configuration](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/others/mha_no_ve.json) | Use the selected width and depth with full causal attention. Verify cost, checkpoint resume and the evaluation schedule. |
| Vanilla MoE | [`RoutedFFN`](../../../#contribute?cell=moe-T3-L20) computes a normalized top-k mixture with one shared ReLU² expert. [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) provides a distributed implementation for comparison. | Keep a dense first layer. Connect expert dispatch for 256 routed experts, eight selected per token and one shared expert. Pilot the balancing rule and measure cost. |
| Dense KDA–DSA | [`KdaDsaGroup`](../../../#contribute?cell=kda-dsa-T3-L20) composes three [KDA mixers](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/linear.py) and one [DSA mixer](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/sparse.py), with a dense FFN after each mixer. | Repeat the group to the selected depth. Include the indexer loss and verify training FLOPs for the first 512 updates and the following sparse phase separately. |
| Fixed Recursive | [Shared-core execution](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/models/lm.py) and the [benchmark configuration](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/configs/looped/fixed_recursive.json) | Reuse the same core weights on both passes. Adapt the benchmark's attention windows and value embeddings to full attention with VE off. Charge both executions, including backward. |

### Try the references on CPU

From the checkout with the [project dependencies](../../getting-started/installation/) installed:

```sh
NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 python -m nexttransformer.scaling.reference_blocks
```

The command runs a forward and backward pass on synthetic inputs for the existing Dense and Fixed Recursive benchmark models and for the two new blocks. MoE uses eight small experts and selects two. KDA–DSA exercises both sides of the burn-in boundary. A formal T3 run still needs a matching trainer, verified cost and measured throughput.

## Coordinate a contribution

1. Open a [T3 coordination issue](https://github.com/Interactive-Active-AI/NextTransformer/issues/new?template=scaling_cell.yml) with the cell ID, intended contribution and availability. Verify the implementation and cost proof with the maintainers before booking compute.
2. For implementation work, match the [2K protocol](../protocol/), check the kernels, data stream, memory, resume behavior and FLOPs for each training phase. Publish estimated and measured throughput separately.
3. For a formal run, pin the approved implementation, configuration, cost proof and data hashes. Preserve seed 42 and report every scheduled evaluation. Keep the final validation BPB, full CORE-22, actual tokens and cumulative training FLOPs together with checkpoint and provenance links.

Analysis of the fits, audits and evaluations of released checkpoints are also welcome. Later phases open when their own specifications are published.
