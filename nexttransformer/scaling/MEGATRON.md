# Megatron-Core adapter for the 2K scaling MoE family

Status: CPU mathematics and optimizer continuation verified. CUDA dispatch, full-size memory,
throughput, distributed checkpoint recovery, and long-training stability have not been verified.
This is an integration/pilot adapter, not a formal T3 training launcher. It does not change the
49B leaderboard protocol or the other three scaling families.

## Architecture

`moe_recipe.py` reads the four MoE cells in `configs/scaling/t3.json` and refuses an incompatible
architecture. Each cell has native full causal MHA (including the native QK normalization and
RoPE), no value embeddings, a first dense 4d FFN, and later layers with 256 routed experts,
normalized top-8 weights and one ungated shared expert. Each expert is a two-matrix ReLU-squared
FFN of width d/4. The native embedding, residual, normalization and readout code is reused.

`megatron_moe.py` replaces only FFNs after the first layer. The GPU path constructs the real
MCore `MoELayer`, `TopKRouter`, `SequentialMLP`, `SharedExpertMLP` and all-to-all dispatcher.
The CPU path uses the same router and expert modules with a local expert-major gather/scatter;
its router subclass overrides only the CUDA-only gating allocation for explicit CPU testing.
The custom TP=1 linear spec keeps NextT's FP32 master weights and casts them to activation dtype
for matmuls. Expert initialization is keyed by global expert ID, not local shard index.

Initial limitations: TP=1, expert-TP=1, PP=1 and CP=1; sequential local expert execution;
no Transformer Engine or grouped-GEMM speed claim; no activation checkpointing. These choices
make the first adapter inspectable and preserve the registered two-matrix FFN. Kernel and
optimizer-state sharding optimizations require subsequent parity and GPU validation.

## Pilot routing and optimizer

The initial *pilot* settings are FP32 routing, top-k followed by softmax, micro-batch auxiliary
load-balancing loss with coefficient 0.01, and router Z-loss with coefficient 0.001. Forced/random
load balancing, expert bias, token dropping, shared-expert gating and routing fusion are disabled.
The coefficients and router initialization std (0.02) remain pilot choices, not frozen outcomes.

`megatron_optimizer.py` preserves the native Muon/AdamW update kernels, schedules and width/batch
learning-rate scaling. Embedding/head/native scalar groups stay on AdamW; router matrices have
a separate AdamW group without weight decay; attention, dense FFN and each 2D expert matrix use
Muon. Expert matrices remain separate 2D parameters, rather than flattening a whole 3D expert bank
into one Muon update. Weight-decay normalization uses the logical global parameter count.

The adapter optimizer owns gradient averaging: dense/shared/router gradients average over dense
sample owners; routed-expert gradients reduce only over replicas of the *same* expert, with an
additional division by EP size because dispatch backward sums EP sample-owner contributions.
Do not also wrap the model in DDP. Optimizer states are initially unsharded.

Before backward with gradient accumulation, call `set_aux_loss_scale(1 / accumulation, device=...)`
and multiply the main loss by `1 / accumulation` as well. MCore attaches auxiliary gradients through its
autoscaler; the native returned CE and BPB continue to exclude router regularizers. The CPU
continuation experiment saves/restores model, optimizer and CPU RNG and verifies the next update.
A future distributed trainer must additionally save every rank's RNG, data position and routing
state; the singleton CPU experiment does not certify that path.

## Reproduce CPU checks

Tested with PyTorch 2.9.1 and MCore 0.21.0 at commit
`b6d7b650df555987a618ec440654306b9c8539a8`. Use a separate environment or an existing compatible
stack; this adapter does not add Megatron to the default project dependencies. Point
`MEGATRON_ROOT` to that checkout, with the project and MCore dependencies available in the Python
interpreter. All commands below hide CUDA devices explicitly.

```sh
# Inspect a T3 recipe without importing Megatron.
CUDA_VISIBLE_DEVICES='' python -m nexttransformer.scaling.megatron_moe --cell moe-T3-L20

# Math, all-parameter gradients, optimizer groups, aux-gradient accumulation and resume.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=".:$MEGATRON_ROOT" \
  python -m unittest discover -s tests -p test_megatron_moe.py -v

# Quantitative report: 256/top-8 parity plus a reduced complete model continuation.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=".:$MEGATRON_ROOT" \
  python -m nexttransformer.scaling.megatron_cpu \
  --output artifacts/moe-megatron-2k/cpu-experiments.json

# Four CPU processes: EP=2 with two replicas for each expert shard.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=".:$MEGATRON_ROOT" \
  torchrun --standalone --nproc-per-node=4 -m nexttransformer.scaling.megatron_cpu --gradient-groups
```

The 256/top-8 experiment uses width 32, not the full T3 width. It randomizes expert output weights
before the parity comparison, so agreement cannot come from zero-initialized outputs. The CPU
BF16 experiment also checks activation-dtype matmuls with FP32 masters and finite gradients;
MCore weights expert activations before FC2 whereas the native reference weights the FC2
output, so BF16 rounding differs even though the FP32 algebra agrees. It also
covers experts receiving no tokens. The reduced full-model experiment uses 3 layers, width 32,
8 routed experts/top-2 and synthetic tokens; it is not a convergence or quality result.
The four-process test checks gradient ownership/scaling with prescribed gradients, not CUDA
all-to-all forward/backward communication.

## GPU validation handoff

When GPUs are available, initialize MCore's process groups for the requested EP topology and pass
its collection to `build_model(recipe, pg_collection=..., cpu_math=False)`. Start with a reduced
single-GPU model, then the same-node EP topology. GPU checks must establish:

1. Forward/backward and update agreement with the CPU/reference math, including BF16 casts,
   auxiliary loss scaling, empty experts and nonzero residual output weights.
2. Rank-consistent initialization, dense/expert gradient reductions, and saved/resumed updates
   at the actual EP topology. Verify token order and RNG/data-cursor continuation.
3. Full T3 sizes and global batch: peak memory, throughput, per-layer expert counts and router
   statistics. Compare CE separately from balancing/Z losses. Check both small and large sizes.
4. Operator-level cost accounting including router/auxiliary paths, then the same pinned token
   stream, BPB/CORE-22 evaluation and checkpoint/export integration as the other scaling families.
5. A bounded stability pilot with the new architecture and optimizer before freezing the recipe.

Until these checks pass, manifests explicitly report `gpu_validated=false`,
`formal_training_ready=false` and `cost_proof_complete=false`. The legacy 4K Dense launcher does
not accept this MoE adapter; do not point it at a T3 manifest and treat it as a formal run.
