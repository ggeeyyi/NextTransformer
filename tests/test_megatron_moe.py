"""CPU acceptance checks; MCore-specific tests skip without the optional stack."""
from dataclasses import replace
import importlib.util
import io
import unittest

import torch
from torch.nn import functional as F

from nexttransformer.scaling.moe_recipe import load_t3_recipe

HAVE_MCORE = importlib.util.find_spec("megatron") is not None


class RecipeTests(unittest.TestCase):
    def test_t3_manifest_agrees_with_published_tokens(self):
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parents[1] / "configs/scaling/t3.json").read_text())
        for cell in data["cells"]:
            if cell["family"] == "moe":
                r = load_t3_recipe(cell["id"])
                self.assertEqual(r.estimate()["estimated_tokens"], cell["estimatedTokens"])
                self.assertFalse(r.manifest()["gpu_validated"])
                self.assertFalse(r.estimate()["cost_proof_complete"])

    def test_counts_and_rejected_shapes(self):
        r = load_t3_recipe("moe-T3-L20")
        self.assertEqual(r.counts()["block_total"], 4150558720)
        self.assertEqual(r.counts()["block_active"], 290488320)
        self.assertRaises(ValueError, replace, r, top_k=257)
        self.assertRaises(ValueError, replace, r, expert_width=640)
        self.assertRaises(ValueError, load_t3_recipe, "dense-T3-L20")


@unittest.skipUnless(HAVE_MCORE, "optional Megatron-Core stack is not installed")
class MegatronCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nexttransformer.scaling.megatron_moe import cpu_process_groups
        cls.context = cpu_process_groups()
        cls.pg = cls.context.__enter__()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        torch.manual_seed(11)
        self.recipe = replace(load_t3_recipe().debug(), aux_loss_coeff=0., z_loss_coeff=0.)

    def ffn_pair(self, recipe=None):
        from types import SimpleNamespace
        from nexttransformer.scaling.megatron_moe import CPUMathMoE
        from nexttransformer.scaling.reference_blocks import RoutedFFN
        r = recipe or self.recipe
        actual = CPUMathMoE(r, 2, self.pg)
        reference = RoutedFFN(SimpleNamespace(width=r.width), r.routed_experts, r.top_k, r.expert_width)
        with torch.no_grad():
            reference.router.weight.copy_(actual.router.weight)
            pairs = [(reference.shared, actual.shared_experts)]
            pairs += list(zip(reference.experts, actual.experts.local_experts))
            for ref, other in pairs:
                ref.c_fc.weight.copy_(other.linear_fc1.weight)
                # Nonzero output weights prevent a vacuous all-zero parity check.
                nn_weight = torch.randn_like(other.linear_fc2.weight) * .1
                ref.c_proj.weight.copy_(nn_weight)
                other.linear_fc2.weight.copy_(nn_weight)
        return actual, reference

    def test_mcore_config_is_new_recipe(self):
        from nexttransformer.scaling.megatron_moe import transformer_config
        c = transformer_config(load_t3_recipe(), cpu=True)
        self.assertFalse(c.gated_linear_unit)
        self.assertEqual(c.num_query_groups, c.num_attention_heads)
        self.assertEqual(c.moe_shared_expert_intermediate_size, 320)
        self.assertEqual(c.moe_ffn_hidden_size, 320)
        self.assertEqual(c.moe_router_dtype, "fp32")
        self.assertFalse(c.moe_router_force_load_balancing)
        self.assertIsNone(c.moe_expert_capacity_factor)

    def test_nonzero_forward_and_all_parameter_gradients(self):
        actual, reference = self.ffn_pair()
        x = torch.randn(7, 2, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        out, _ = actual(x)
        expected = reference(y)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
        weights = torch.randn_like(out)
        (out * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual.router.weight.grad, reference.router.weight.grad, rtol=2e-5, atol=2e-6)
        for a, b in [(actual.shared_experts, reference.shared), *zip(actual.experts.local_experts, reference.experts)]:
            for ap, bp in [(a.linear_fc1.weight, b.c_fc.weight), (a.linear_fc2.weight, b.c_proj.weight)]:
                expected_grad = bp.grad if bp.grad is not None else torch.zeros_like(bp)
                torch.testing.assert_close(ap.grad, expected_grad, rtol=2e-5, atol=2e-6)

    def test_bf16_matmul_with_fp32_master_weights(self):
        actual, reference = self.ffn_pair()
        x = torch.randn(7, 2, 32, dtype=torch.bfloat16, requires_grad=True)
        out, _ = actual(x)
        expected = reference(x)
        # MCore weights expert activations before FC2; the native reference
        # weights outputs after FC2. Algebra agrees; BF16 rounding can differ.
        rms_error = (out.float() - expected.float()).square().mean().sqrt()
        rms_signal = expected.float().square().mean().sqrt().clamp_min(1e-8)
        self.assertLess((rms_error / rms_signal).item(), .02)
        out.float().square().mean().backward()
        self.assertTrue(all(p.dtype == torch.float32 for p in actual.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.parameters()))

    def test_256_experts_top8_and_empty_experts(self):
        r = replace(self.recipe, routed_experts=256, top_k=8)
        actual, reference = self.ffn_pair(r)
        x = torch.randn(1, 2, 32, requires_grad=True)
        out, _ = actual(x)
        torch.testing.assert_close(out, reference(x), rtol=1e-5, atol=1e-6)
        out.square().mean().backward()
        self.assertEqual(actual.last_load.sum().item(), 16)
        self.assertGreater((actual.last_load == 0).sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in actual.experts.parameters()))

    def test_full_backbone_and_parameter_accounting(self):
        from nexttransformer.scaling.megatron_moe import build_model
        model = build_model(self.recipe, pg_collection=self.pg, cpu_math=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), self.recipe.counts()["model_total"])
        self.assertEqual(model.transformer.h[0].mlp.c_fc.weight.shape, (128, 32))
        self.assertEqual(len(model.value_embeds), 0)
        self.assertTrue(all(b.attn.n_head == b.attn.n_kv_head == 2 for b in model.transformer.h))
        self.assertTrue(all(b.attn.ve_gate is None for b in model.transformer.h))
        self.assertTrue(all(w == (8, 0) for w in model.window_sizes))
        tokens = torch.randint(64, (2, 8))
        loss = model(tokens, targets=tokens)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_auxiliary_gradient_accumulation_scale(self):
        from nexttransformer.scaling.megatron_moe import CPUMathMoE, set_aux_loss_scale
        r = replace(self.recipe, aux_loss_coeff=.01, z_loss_coeff=.001)
        full = CPUMathMoE(r, 2, self.pg)
        split = CPUMathMoE(r, 2, self.pg)
        split.load_state_dict(full.state_dict())
        x = torch.randn(5, 2, 32)
        set_aux_loss_scale(1.)
        full(x)[0].sum().backward()
        set_aux_loss_scale(.5)
        for _ in range(2):
            (split(x)[0].sum() / 2).backward()
        torch.testing.assert_close(full.router.weight.grad, split.router.weight.grad)
        self.assertGreater(full.router.weight.grad.norm().item(), 0.)
        set_aux_loss_scale(1.)

    def test_optimizer_groups_and_resume_next_update(self):
        from nexttransformer.scaling.megatron_moe import build_model, set_aux_loss_scale
        from nexttransformer.scaling.megatron_optimizer import build_optimizer
        from nexttransformer.scaling.config import ScalingTrainConfig
        r = replace(self.recipe, aux_loss_coeff=.01, z_loss_coeff=.001)
        model = build_model(r, pg_collection=self.pg, cpu_math=True)
        config = ScalingTrainConfig(protocol="nextt-scale-2k-v1", total_tokens=320, global_batch_tokens=32)
        optimizer = build_optimizer(model, config, pg_collection=self.pg)
        names = {n: g["kind"] for g in optimizer.param_groups for n in g["parameter_names"]}
        self.assertTrue(all(kind == "adamw" for n, kind in names.items() if ".router." in n))
        self.assertTrue(all(kind == "muon" for n, kind in names.items() if ".local_experts." in n))
        tokens = torch.randint(64, (2, 8))
        def update(m, o):
            o.zero_grad(set_to_none=True)
            set_aux_loss_scale(1.)
            loss = m(tokens, targets=tokens)
            loss.backward()
            o.step()
            return loss.detach()
        update(model, optimizer)
        buf = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()), buf)
        restored = build_model(r, pg_collection=self.pg, cpu_math=True)
        other = build_optimizer(restored, config, pg_collection=self.pg)
        buf.seek(0)
        payload = torch.load(buf, weights_only=True)
        restored.load_state_dict(payload["model"])
        other.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng"])
        expected = update(model, optimizer)
        got = update(restored, other)
        torch.testing.assert_close(expected, got, rtol=0, atol=0)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
