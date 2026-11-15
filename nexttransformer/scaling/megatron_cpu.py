"""Reproducible CPU math/gradient experiments for the scaling MoE adapter.

python -m nexttransformer.scaling.megatron_cpu --output report.json
CPU torchrun --nproc-per-node=4 -m nexttransformer.scaling.megatron_cpu --gradient-groups

No CUDA calls or training-data reads are needed. GPU validation is a separate stage.
"""
from dataclasses import replace
import argparse
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nexttransformer.scaling.moe_recipe import load_t3_recipe, TESTED_MCORE_COMMIT


def gradient_group_check():
    """Four Gloo ranks check EP=2 with two expert-data replicas, without CUDA."""
    from nexttransformer.scaling.megatron_optimizer import MoEOptimizer
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 4:
            raise ValueError("gradient group experiment requires four CPU ranks")
        ep, expt_dp = None, None
        for ids in ([0, 1], [2, 3]):
            group = dist.new_group(ids, backend="gloo")
            if rank in ids:
                ep = group
        for ids in ([0, 2], [1, 3]):
            group = dist.new_group(ids, backend="gloo")
            if rank in ids:
                expt_dp = group
        dense = torch.nn.Parameter(torch.zeros(2, 2))
        expert = torch.nn.Parameter(torch.zeros(2, 2))
        # Dense local means 1,2,3,4 -> global mean 2.5.
        dense.grad = torch.full_like(dense, rank + 1.)
        # On EP group 0 the sums of sample-owner contributions are 3 and 7;
        # on its replica they are 11 and 15. After replica mean and /EP:
        # expert0 -> 3.5, expert1 -> 5.5. World averaging would mix expert IDs.
        expert.grad = torch.full_like(expert, (3., 7., 11., 15.)[rank])
        groups = [dict(kind="muon", expert=False, params=[dense]),
                  dict(kind="muon", expert=True, params=[expert])]
        opt = MoEOptimizer(groups, SimpleNamespace(dp_cp=dist.group.WORLD, ep=ep, expt_dp=expt_dp))
        opt.synchronize_gradients()
        torch.testing.assert_close(dense.grad, torch.full_like(dense, 2.5))
        value = 3.5 if rank % 2 == 0 else 5.5
        torch.testing.assert_close(expert.grad, torch.full_like(expert, value))
        dist.barrier()
        if rank == 0:
            print(json.dumps(dict(check="gradient_ownership", ranks=4, ep=2,
                                  expert_data_replicas=2, passed=True)))
    finally:
        dist.destroy_process_group()


def run_experiments():
    from nexttransformer.scaling.megatron_moe import CPUMathMoE, build_model, cpu_process_groups, set_aux_loss_scale
    from nexttransformer.scaling.megatron_optimizer import build_optimizer
    from nexttransformer.scaling.reference_blocks import RoutedFFN
    from nexttransformer.scaling.config import ScalingTrainConfig
    import megatron.core
    torch.set_num_threads(1)
    torch.manual_seed(42)
    r = replace(load_t3_recipe().debug(routed_experts=256, top_k=8), aux_loss_coeff=0., z_loss_coeff=0.)
    result = dict(megatron_version=megatron.core.__version__, tested_commit=TESTED_MCORE_COMMIT,
                  device="cpu", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  gpu_validated=False, formal_training_ready=False,
                  t3=[load_t3_recipe(f"moe-T3-L{d}").manifest() for d in (16, 20, 28, 32)])
    with cpu_process_groups() as pg:
        actual = CPUMathMoE(r, 2, pg)
        ref = RoutedFFN(SimpleNamespace(width=r.width), r.routed_experts, r.top_k, r.expert_width)
        with torch.no_grad():
            ref.router.weight.copy_(actual.router.weight)
            pairs = [(ref.shared, actual.shared_experts), *zip(ref.experts, actual.experts.local_experts)]
            for a, b in pairs:
                a.c_fc.weight.copy_(b.linear_fc1.weight)
                nn_out = torch.randn_like(b.linear_fc2.weight) * .1
                a.c_proj.weight.copy_(nn_out)
                b.linear_fc2.weight.copy_(nn_out)
        x = torch.randn(7, 2, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        out, _ = actual(x)
        expected = ref(y)
        weights = torch.randn_like(out)
        (out * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual.router.weight.grad, ref.router.weight.grad, rtol=2e-5, atol=2e-6)
        result["parity"] = dict(experts=256, top_k=8, tokens=14,
                                forward_max_abs=(out - expected).abs().max().item(),
                                input_grad_max_abs=(x.grad - y.grad).abs().max().item(),
                                router_grad_max_abs=(actual.router.weight.grad - ref.router.weight.grad).abs().max().item(),
                                empty_experts=(actual.last_load == 0).sum().item(), passed=True)
        actual.zero_grad(set_to_none=True)
        ref.zero_grad(set_to_none=True)
        bf_input = x.detach().to(torch.bfloat16).requires_grad_(True)
        bf_out, _ = actual(bf_input)
        bf_expected = ref(bf_input)
        bf_relative = ((bf_out.float() - bf_expected.float()).square().mean().sqrt() /
                       bf_expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
        bf_out.float().square().mean().backward()
        if bf_relative >= .02 or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.parameters()):
            raise RuntimeError("BF16 CPU cast-path check failed")
        result["bf16_cpu"] = dict(relative_rms_forward_error=bf_relative,
                                  fp32_master_weights=all(p.dtype == torch.float32 for p in actual.parameters()),
                                  finite_gradients=True, passed=True)
        # Small complete model, using actual MoE coefficients and native optimizer.
        small = load_t3_recipe().debug()
        model = build_model(small, pg_collection=pg, cpu_math=True)
        config = ScalingTrainConfig(protocol="nextt-scale-2k-v1", total_tokens=320, global_batch_tokens=32)
        opt = build_optimizer(model, config, pg_collection=pg)
        batches = [torch.randint(small.vocab_size, (4, small.sequence_len)) for _ in range(6)]
        def update(m, o, batch):
            o.zero_grad(set_to_none=True)
            set_aux_loss_scale(1.)
            loss = m(batch, targets=batch.roll(-1, dims=1))
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad):
                raise RuntimeError("nonfinite CPU smoke loss or gradient")
            o.step()
            return loss.item()
        losses = [update(model, opt, batch) for batch in batches[:5]]
        checkpoint = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state()), checkpoint)
        restored = build_model(small, pg_collection=pg, cpu_math=True)
        other = build_optimizer(restored, config, pg_collection=pg)
        checkpoint.seek(0)
        state = torch.load(checkpoint, weights_only=True)
        restored.load_state_dict(state["model"])
        other.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        next_loss = update(model, opt, batches[5])
        resumed_loss = update(restored, other, batches[5])
        max_delta = max((a - b).abs().max().item() for a, b in zip(model.parameters(), restored.parameters()))
        if next_loss != resumed_loss or max_delta != 0:
            raise RuntimeError("checkpoint continuation disagrees")
        result["small_model"] = dict(layers=small.layers, width=small.width, experts=small.routed_experts,
                                     top_k=small.top_k, parameters=sum(p.numel() for p in model.parameters()),
                                     losses=losses, next_loss=next_loss, resumed_loss=resumed_loss,
                                     resumed_parameter_max_abs=max_delta, passed=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gradient-groups", action="store_true")
    args = parser.parse_args()
    if args.gradient_groups:
        gradient_group_check()
        return
    result = run_experiments()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "t3"}, indent=2))


if __name__ == "__main__":
    main()
