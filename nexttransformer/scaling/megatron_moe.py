"""Megatron FFNs inside the unchanged NextT full-MHA/no-VE backbone.

CPU math uses real MCore routers/MLPs with a local gather/scatter dispatcher.
CUDA uses the MCore MoELayer all-to-all dispatcher. CPU results do not validate
CUDA communication, performance, or long-training stability. TP/PP/CP are 1 in
this first adapter; expert parallelism and expert-data replicas are explicit.
"""
from contextlib import contextmanager
from dataclasses import fields, replace
from functools import partial
import json
import math
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from nexttransformer.scaling.moe_recipe import MoERecipe, load_t3_recipe


def relu_squared(x):
    return F.relu(x).square()


def _input_init(weight):
    bound = 0.4 * math.sqrt(3 / weight.shape[-1])
    return nn.init.uniform_(weight, -bound, bound)


def transformer_config(recipe, *, cpu=False, expert_parallel_size=1):
    from megatron.core.transformer.transformer_config import TransformerConfig
    if recipe.routed_experts % expert_parallel_size:
        raise ValueError("experts must divide evenly across the expert group")
    return TransformerConfig(
        num_layers=recipe.layers, hidden_size=recipe.width, num_attention_heads=recipe.heads,
        num_query_groups=recipe.heads, kv_channels=recipe.width // recipe.heads,
        ffn_hidden_size=4 * recipe.width, num_moe_experts=recipe.routed_experts,
        moe_router_topk=recipe.top_k, moe_ffn_hidden_size=recipe.expert_width,
        moe_shared_expert_intermediate_size=recipe.expert_width, moe_shared_expert_gate=False,
        moe_shared_expert_overlap=False, moe_router_dtype="fp32",
        moe_router_score_function="softmax", moe_router_pre_softmax=False,
        moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=recipe.aux_loss_coeff,
        moe_z_loss_coeff=recipe.z_loss_coeff, moe_router_force_load_balancing=False,
        moe_expert_capacity_factor=None, moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False, moe_permute_fusion=False, moe_router_fusion=False,
        expert_model_parallel_size=expert_parallel_size, expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        params_dtype=torch.float32, use_cpu_initialization=cpu,
        activation_func=relu_squared, gated_linear_unit=False, add_bias_linear=False,
        bias_activation_fusion=False, gradient_accumulation_fusion=False,
        init_method=_input_init, output_layer_init_method=nn.init.zeros_,
        hidden_dropout=0., attention_dropout=0., sequence_parallel=False,
    )


class MasterLinear(nn.Module):
    """TP=1 MCore linear spec with NextT's FP32 master / activation-dtype matmul."""
    def __init__(self, input_size, output_size, *, config, init_method, bias=False,
                 is_expert=False, tp_group=None, **kwargs):
        super().__init__()
        if bias or config.tensor_model_parallel_size != 1 or config.expert_tensor_parallel_size != 1:
            raise ValueError("this adapter requires bias-free linears and TP=expert-TP=1")
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, device=device, dtype=torch.float32))
        self.bias = None
        if config.perform_initialization:
            init_method(self.weight)
        self.weight.allreduce = not (is_expert and config.expert_model_parallel_size > 1)
        self.weight.tensor_model_parallel = False
        self.weight.sequence_parallel = False

    def forward(self, x, **kwargs):
        return F.linear(x, self.weight.to(x.dtype)), None

    def backward_dw(self):
        return None


def _submodules():
    from megatron.core.transformer.mlp import MLPSubmodules
    return MLPSubmodules(MasterLinear, MasterLinear)


def _router_class(recipe, *, cpu):
    from megatron.core.transformer.moe.router import TopKRouter

    class NativeRouter(TopKRouter):
        def reset_parameters(self):
            super().reset_parameters()
            if self.config.perform_initialization:
                nn.init.normal_(self.weight, std=recipe.router_init_std)

        def gating(self, x):
            if x.device.type == "cpu":
                if not cpu:
                    raise RuntimeError("CPU router is restricted to the explicit CPU math path")
                return F.linear(x.float(), self.weight.float())
            return super().gating(x)

    return NativeRouter


def _initialize_ffn(module, recipe, layer_number, ep_rank):
    # Expert IDs, not local rank IDs, define initialization. Shared/router weights
    # are identical on all expert ranks. fork_rng preserves the caller's data RNG.
    devices = [] if next(module.parameters()).device.type == "cpu" else [torch.cuda.current_device()]
    def initialize(expert, seed):
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            _input_init(expert.linear_fc1.weight)
            nn.init.zeros_(expert.linear_fc2.weight)
    local = module.experts.local_experts
    for i, expert in enumerate(local):
        initialize(expert, recipe.seed + 100003 * layer_number + ep_rank * len(local) + i)
    initialize(module.shared_experts, recipe.seed + 100003 * layer_number + recipe.routed_experts)
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(recipe.seed + 100003 * layer_number + recipe.routed_experts + 1)
        nn.init.normal_(module.router.weight, std=recipe.router_init_std)


class CPUMathMoE(nn.Module):
    """Real MCore router/expert math; local dispatch only, no CUDA emulation."""
    def __init__(self, recipe, layer_number, pg_collection):
        super().__init__()
        from megatron.core.transformer.moe.experts import SequentialMLP
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
        self.config = transformer_config(recipe, cpu=True)
        self.router = _router_class(recipe, cpu=True)(config=self.config, pg_collection=pg_collection)
        self.router.set_layer_number(layer_number)
        self.experts = SequentialMLP(recipe.routed_experts, self.config, _submodules(), pg_collection)
        self.shared_experts = SharedExpertMLP(self.config, _submodules(), False, pg_collection)
        self.last_load = None
        _initialize_ffn(self, recipe, layer_number, 0)

    def forward(self, hidden):
        probs, mapping = self.router(hidden)
        rows = hidden.flatten(0, 1)
        # Expert-major permutation, matching the MCore experts interface. Includes
        # empty experts and keeps all expert weights in the backward graph.
        ids = mapping.T.nonzero(as_tuple=False)[:, 1]
        counts = mapping.sum(0)
        selected_probs = probs.T[mapping.T]
        output, bias = self.experts(rows[ids], counts, selected_probs)
        if bias is not None:
            raise RuntimeError("unexpected expert bias")
        result = torch.zeros_like(rows).index_add(0, ids, output).reshape_as(hidden)
        self.last_load = counts.detach()
        return result + self.shared_experts(hidden), None


class MegatronFFN(nn.Module):
    def __init__(self, recipe, layer_number, *, pg_collection, cpu_math=False):
        super().__init__()
        if cpu_math:
            if dist.get_world_size() != 1:
                raise ValueError("CPU math adapter requires a singleton Gloo group")
            self.engine = CPUMathMoE(recipe, layer_number, pg_collection)
        else:
            from megatron.core.transformer.moe.experts import SequentialMLP
            from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
            from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
            ep = pg_collection.ep.size()
            config = transformer_config(recipe, expert_parallel_size=ep)
            self.engine = MoELayer(
                config=config, layer_number=layer_number, pg_collection=pg_collection,
                submodules=MoESubmodules(
                    experts=partial(SequentialMLP, submodules=_submodules()),
                    shared_experts=partial(SharedExpertMLP, submodules=_submodules()),
                    router=_router_class(recipe, cpu=False)),
            )
            self.engine.to(device=torch.cuda.current_device())
            _initialize_ffn(self.engine, recipe, layer_number, pg_collection.ep.rank())

    def forward(self, hidden):
        # NextT uses B,S,D; MCore uses S,B,D. Contiguous conversion preserves
        # per-sequence aux-loss semantics and restores the native block interface.
        output, bias = self.engine(hidden.transpose(0, 1).contiguous())
        if bias is not None:
            raise RuntimeError("unexpected MoE bias")
        return output.transpose(0, 1).contiguous()


def build_model(recipe, *, pg_collection, cpu_math=False):
    from nexttransformer.models.config import debug_config
    from nexttransformer.models.lm import build_model as native_model
    device = "cpu" if cpu_math else f"cuda:{torch.cuda.current_device()}"
    config = replace(debug_config("mha_no_ve", width=recipe.width, sequence_len=recipe.sequence_len),
                     layers=recipe.layers, heads=recipe.heads, mlp_width=4 * recipe.width,
                     vocab_size=recipe.vocab_size, window_pattern="L")
    model = native_model(config, device=device, seed=recipe.seed)
    for index, block in enumerate(model.transformer.h):
        if index > 0:
            block.mlp = MegatronFFN(recipe, index + 1, pg_collection=pg_collection, cpu_math=cpu_math)
    model.moe_recipe = recipe
    model.moe_cpu_math = cpu_math
    return model


@contextmanager
def cpu_process_groups():
    """One Gloo rank for CPU math tests; never initialize a CUDA device."""
    from megatron.core.process_groups_config import ProcessGroupCollection
    owns = not dist.is_initialized()
    if not owns and (dist.get_world_size() != 1 or dist.get_backend() != "gloo"):
        raise RuntimeError("CPU checks require an isolated singleton Gloo process")
    with tempfile.TemporaryDirectory() as tmp:
        if owns:
            dist.init_process_group("gloo", init_method="file://" + tmp + "/init", rank=0, world_size=1)
        pg = ProcessGroupCollection()
        # The CPU test has no distinct parallel axes. MCore's collection fields
        # are init=False and the tested router/MLPs access several of them.
        for field in fields(pg):
            setattr(pg, field.name, dist.group.WORLD)
        try:
            yield pg
        finally:
            if owns:
                dist.destroy_process_group()


def set_aux_loss_scale(scale, *, device="cpu"):
    """Call before backward: auxiliary gradients must use the same 1/accum scale."""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(float(scale), dtype=torch.float32, device=device))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="moe-T3-L20")
    args = parser.parse_args()
    print(json.dumps(load_t3_recipe(args.cell).manifest(), indent=2))


if __name__ == "__main__":
    main()
