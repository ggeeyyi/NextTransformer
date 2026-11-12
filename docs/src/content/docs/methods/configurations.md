---
title: Configurations
description: Admitted shapes, parameters and audited compute for the documented models and their reference configurations.
---

Each method in the table has an admitted configuration, `configs/<family>/<method>.json`, and a cost proof in the current
proof set. The table is generated from those files: physical blocks, FFN width, independent parameters, audited training
GFLOPs per input token and the deviation from the SSSL anchor (4.67239832544785). Parameters are outcomes, not controls.

| Family | Method | Blocks | FFN | Parameters | GFLOPs / token | vs SSSL | Stages |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | `dense_full` | 23 | 6,144 | 1,355,810,568 | 4.674662 | +0.048% | joint |
| looped | `deeploop` | 12 | 6,144 | 742,392,722 | 4.660529 | -0.254% | joint |
| looped | `fixed_recursive` | 15 | 6,144 | 927,990,968 | 4.663054 | -0.200% | joint |
| looped | `huginn` | 14 | 6,528 | 870,581,286 | 4.649327 | -0.494% | joint |
| looped | `ouro` | 9 | 6,144 | 607,127,805 | 4.658210 | -0.304% | joint, gate_only |
| looped | `ut_act` | 12 | 6,144 | 742,394,259 | 4.660085 | -0.264% | joint |
| sparse | `dsa` | 22 | 6,400 | 1,316,236,406 | 4.650934 | -0.459% | burnin, joint |
| sparse | `moba` | 25 | 6,144 | 1,462,765,468 | 4.673521 | +0.024% | joint |
| sparse | `nsa` | 23 | 6,144 | 1,381,293,832 | 4.638686 | -0.722% | joint |
| sparse | `qsa` | 23 | 6,144 | 1,378,420,488 | 4.628037 | -0.949% | burnin, joint |
| sparse | `seer` | 23 | 6,272 | 1,387,464,456 | 4.659564 | -0.275% | burnin, joint |
| linear | `deltanet` | 25 | 6,144 | 1,463,690,268 | 4.674979 | +0.055% | joint |
| linear | `elu_norm` | 25 | 6,144 | 1,462,768,668 | 4.651242 | -0.453% | joint |
| linear | `elu_qknorm` | 25 | 6,144 | 1,462,768,668 | 4.652243 | -0.431% | joint |
| linear | `gdn` | 23 | 6,144 | 1,411,349,680 | 4.687197 | +0.317% | joint |
| linear | `gla` | 23 | 6,144 | 1,411,207,816 | 4.642048 | -0.650% | joint |
| linear | `kda` | 23 | 6,144 | 1,412,126,344 | 4.692443 | +0.429% | joint |
| linear | `retnet` | 23 | 6,144 | 1,410,077,320 | 4.630060 | -0.906% | joint |
| residual | `attnres` | 23 | 6,144 | 1,355,882,760 | 4.679857 | +0.160% | joint |
| residual | `attnres_canon_abcd` | 23 | 6,144 | 1,357,154,568 | 4.688124 | +0.337% | joint |
| residual | `attnres_full_canon_abcd` | 23 | 6,144 | 1,357,154,568 | 4.695110 | +0.486% | joint |
| residual | `hc` | 23 | 6,144 | 1,356,235,700 | 4.698483 | +0.558% | joint |
| residual | `mhc` | 23 | 6,016 | 1,353,550,818 | 4.674363 | +0.042% | joint |
| residual | `residual_standard` | 23 | 6,144 | 1,355,810,568 | 4.674298 | +0.041% | joint |
| hybrid | `deltanet_hybrid` | 24 | 6,400 | 1,403,810,314 | 4.675326 | +0.063% | joint |
| hybrid | `gdn_hybrid` | 24 | 5,760 | 1,399,275,962 | 4.703800 | +0.672% | joint |
| hybrid | `gdn_hybrid_swa` | 22 | 6,528 | 1,329,681,662 | 4.647087 | -0.542% | joint |
| hybrid | `kda_hybrid` | 24 | 6,016 | 793,542,986 | 4.692309 | +0.426% | joint |
| hybrid | `moba_hybrid` | 25 | 6,016 | 1,452,935,068 | 4.659989 | -0.266% | joint |
| hybrid | `qsa_hybrid` | 24 | 5,760 | 1,405,174,202 | 4.691637 | +0.412% | burnin, joint |
| others | `dense` (SSSL) | 24 | 6,144 | 1,384,122,122 | 4.672398 | +0.000% | joint |
| others | `gated_attention` | 21 | 6,272 | 1,306,658,420 | 4.641914 | -0.652% | joint |
| others | `gated_attention_headwise` | 23 | 6,144 | 1,356,234,504 | 4.677384 | +0.107% | joint |

Blocks count physical layers; looped methods reuse them across loops, so their unfolded depth is larger (see the
[source notes](../sources/)). Multi-stage methods weight each stage by its updates: DSA, QSA, Seer and the QSA hybrid run
512 dense burn-in updates before sparse training; Ouro trains its exit gate alone for the final 1,024 updates; Huginn
samples a depth per update from a fixed seed, and its proof enumerates all 46,875 samples.

## Parameter groups

Value-embedding tables account for a large share of the parameter counts above. The two dense references split as
follows, using the groups of `GPT.num_scaling_params` in `nexttransformer/models/backbone.py`. The groups are derived
from the configurations, and their totals match the parameter counts in the cost proofs:

| Group | `dense_full` (Baseline) | `dense` (SSSL anchor) |
|---|---:|---:|
| Transformer blocks (`transformer.h`, including value-embedding gates) | 651,167,424 | 679,478,976 |
| Token embedding (`wte`) | 50,331,648 | 50,331,648 |
| Output head (`lm_head`) | 50,331,648 | 50,331,648 |
| Value embeddings (`value_embeds`, 12 tables) | 603,979,776 | 603,979,776 |
| Scalars | 72 | 74 |
| **Total** | **1,355,810,568** | **1,384,122,122** |

Each value-embedding table is a 32,768 × 1,536 lookup on alternating layers, the last layer included.
Together they hold 44.5% of the Baseline's parameters, yet the cost ledger charges no FLOPs for the lookup
and 1,536 FLOPs per token per table for its backward, against about 4.67 GFLOPs per token in total. Parameter
counts therefore do not track per-token compute: `kda_hybrid` has no value embeddings and about 0.6B fewer parameters
than the other hybrids at the same compute. Ratios of tokens to parameters change with the groups they count.

The leaderboard's Dense Baseline and the compute anchor are different models. The Baseline is `dense_full`, with full
causal attention in all 23 layers. The anchor that every method is calibrated against is `dense`, with
24 layers in the SSSL pattern: three short sliding-window layers (at most 513 visible tokens) to one
full causal layer, and the last layer always full, so 6 layers see the whole context (see the
[model shell](../../protocol/model-shell/)). The "about 1.4B parameters" quoted elsewhere in the documentation is the
total over all groups.

## Reading a configuration

A configuration holds the model fields (`model`), the protocol it is admitted under (`protocol`, with its fingerprint)
and the calibration record (`calibration`: the audited FLOPs per token and their deviation, the method's source
fingerprint, the PyTorch build of the audit and the path of the cost proof). Configurations are written by
`python -m nexttransformer.protocol.calibrate` and checked by `admit_config`; never edit them by hand.

The historical native track (SSSL, MoR, LoopFormer, Circle Loop) keeps its own configurations under `configs/native/`,
with the historical matrix-FLOPs accounting.
