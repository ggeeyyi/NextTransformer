"""The MoE comparison track: protocol, reference model, cost ledger, plugins, data order, evaluation, trainer."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from moe import compute, plugins, spec
from moe.data import Rows, micro_batches, update_rows
from moe.evaluation import score
from moe.model import Experts, MoEConfig, Router, build_model

ROOT = Path(__file__).resolve().parents[1]
TEST_PLUGINS = ROOT / "tests" / "moe_plugins"


@pytest.fixture
def test_plugins(monkeypatch):
    monkeypatch.setenv("NEXTT_MOE_METHOD_PATHS", str(TEST_PLUGINS))
    plugins.registry.cache_clear()
    plugins.module.cache_clear()
    yield
    plugins.registry.cache_clear()
    plugins.module.cache_clear()


def cl100k_available():
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


needs_tokenizer = pytest.mark.skipif(not cl100k_available(), reason="cl100k_base ranks are not cached/downloadable")


# --- protocol -------------------------------------------------------------------------------------------------

def test_tracks_follow_the_marin_cells():
    d768, d1024 = spec.TRACKS["d768"], spec.TRACKS["d1024"]
    assert (d768.layers, d768.heads, d768.kv_heads, d768.expert_width, d768.shared_expert_width) == (8, 6, 1, 384, 768)
    assert (d1024.layers, d1024.heads, d1024.kv_heads, d1024.expert_width, d1024.shared_expert_width) == (11, 8, 2, 512, 1024)
    assert d768.total_tokens == 16_875 * 64 * 4096 == 4_423_680_000
    assert d1024.total_tokens == 16_080 * 128 * 4096 == 8_430_551_040
    assert d768.validation_steps[0] == 0 and d768.validation_steps[-1] == 16_875 and len(d768.validation_steps) == 35
    assert d768.core_steps[-1] == d768.updates and d1024.core_steps[-1] == d1024.updates
    assert spec.long_layers(8) == (3, 7) and spec.long_layers(11) == (3, 7, 10)


def test_schedule_and_optimizer_hyperparameters_follow_marin():
    t = spec.TRACKS["d768"]
    assert t.warmup_steps == 168 and spec.TRACKS["d1024"].warmup_steps == 160
    assert spec.schedule(0, t) == 0.0 and spec.schedule(84, t) == pytest.approx(0.5)
    assert spec.schedule(168, t) == pytest.approx(1.0) and spec.schedule(t.updates, t) == pytest.approx(0.0)
    assert t.adam_lr == pytest.approx(0.00193222, rel=1e-5) and t.muonh_lr == pytest.approx(0.00837296, rel=1e-5)
    assert spec.TRACKS["d1024"].adam_lr == pytest.approx(0.00202863, rel=1e-5)
    assert t.beta2 == pytest.approx(0.999 ** 2) and spec.TRACKS["d1024"].beta2 == pytest.approx(0.999 ** 4)
    assert t.eps == pytest.approx(1.257e-15, rel=1e-3)
    assert all(spec.schedule(s, t) >= spec.schedule(s + 1, t) for s in range(168, t.updates, 97))


def test_config_roundtrip_and_tamper_detection(tmp_path):
    config = MoEConfig.baseline("d768")
    path = spec.write_config(tmp_path / "c.json", config, {"status": "candidate"})
    loaded, calibration = spec.load_config(path)
    assert loaded == config and calibration == {"status": "candidate"}
    data = json.loads(path.read_text())
    data["protocol"]["sequence_len"] = 2048
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        spec.load_config(path)
    data = json.loads(spec.write_config(tmp_path / "d.json", config, {}).read_text())
    data["track"]["updates"] = 1
    (tmp_path / "d.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="track"):
        spec.load_config(tmp_path / "d.json")


@pytest.mark.parametrize("track", ["d768", "d1024"])
def test_admitted_baseline_configs_match_the_protocol(track):
    path = spec.config_path("baseline", track)
    config, calibration = spec.load_config(path)
    assert config == MoEConfig.baseline(track)
    assert calibration["status"] == "admitted"
    record = json.loads((ROOT / calibration["cost_record"]).read_text())
    assert record["flops_per_token"] == pytest.approx(compute.analytic(config)["total"], rel=1e-9)
    assert record["source"]["sha256"] == compute.source_digest("baseline")["sha256"], \
        "the shared MoE core changed: rerun python -m moe.compute calibrate --method baseline --track " + track


# --- model and ledger -----------------------------------------------------------------------------------------

def debug_model(method="baseline", **overrides):
    config = plugins.config_for(method, "d768", debug=True)
    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
    torch.manual_seed(0)
    return build_model(config, "cpu")


def test_debug_model_trains_one_step():
    model = debug_model()
    idx = torch.randint(0, 512, (2, 64))
    loss = model(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())
    logits = model(idx)
    assert logits.shape == (2, 64, 512)


def test_quantile_balancing_router():
    config = MoEConfig.baseline("d768").debug()
    router = Router(config, 0)
    torch.nn.init.normal_(router.weight)
    router.tokens_per_row = 10
    x = torch.randn(100, config.width)
    index, gates, aux = router.train()(x)
    assert index.shape == gates.shape == (100, config.top_k) and aux is None
    torch.testing.assert_close(gates.sum(-1), torch.full((100,), config.routing_renorm_sum), rtol=1e-5, atol=1e-5)
    assert router.beta_count.item() == 10   # ten groups of one 10-token row
    beta = router.beta_sum / router.beta_count
    router.balance(router.beta_sum, router.beta_count)
    torch.testing.assert_close(router.bias, -beta + beta.mean())


def test_dropless_experts_match_a_dense_reference():
    config = MoEConfig.baseline("d768").debug()
    torch.manual_seed(1)
    experts = Experts(config, 0)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(10, config.width)
    index = torch.stack([torch.randperm(config.routed_experts)[:config.top_k] for _ in range(10)])
    gates = torch.rand(10, config.top_k)
    silu = torch.nn.functional.silu
    reference = torch.zeros_like(x)
    for n in range(10):
        for j in range(config.top_k):
            e = index[n, j]
            h = silu(experts.w_gate[e] @ x[n]) * (experts.w_up[e] @ x[n])
            reference[n] += gates[n, j] * (experts.w_down[e] @ h)
    torch.testing.assert_close(experts(x, index, gates), reference)
    # The grouped-GEMM layout (padded groups, transposed weights) computes the same function.
    flat = index.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=config.routed_experts)

    def grouped_mm(a, b, offs):   # torch._grouped_mm's contract
        out, start = [], 0
        for e, end in enumerate(offs.tolist()):
            out.append(a[start:end] @ b[e])
            start = end
        return torch.cat(out)

    original = getattr(torch, "_grouped_mm", None)
    torch._grouped_mm = grouped_mm
    try:
        ys = experts._grouped(x[order // config.top_k], counts, experts.w_gate, experts.w_up, experts.w_down)
    finally:
        if original is None:
            del torch._grouped_mm
        else:
            torch._grouped_mm = original
    loop = Experts.forward(experts, x, index, torch.ones_like(gates))
    combined = torch.zeros_like(x).index_add_(0, order // config.top_k, ys)
    torch.testing.assert_close(combined, loop)


@pytest.mark.parametrize("track, marin_active", [("d768", 58_641_920), ("d1024", 141_937_408)])
def test_parameter_counts_match_marin(track, marin_active):
    config = MoEConfig.baseline(track)
    counts = build_model(config, "meta").parameter_counts()
    # Marin counts every parameter except embedding and head; its router bias is a buffer here.
    assert counts["active_non_embedding"] == marin_active - config.layers * config.routed_experts


def test_document_mask_isolates_documents():
    model = debug_model().eval()
    a = torch.randint(1, 512, (1, 40))
    b = a.clone()
    b[0, :20] = torch.randint(1, 512, (20,))   # change only the first document
    a[0, 20] = b[0, 20] = 0                   # separator: the second document starts here
    torch.testing.assert_close(model(a)[0, 20:], model(b)[0, 20:])


def test_left_padding_is_exact_for_generation():
    model = debug_model().eval()
    prompt = torch.cat((torch.zeros(1, 1, dtype=torch.long), torch.randint(1, 512, (1, 30))), dim=1)
    padded = torch.cat((torch.zeros(1, 9, dtype=torch.long), prompt), dim=1)
    torch.testing.assert_close(model(padded)[0, -1], model(prompt)[0, -1], rtol=1e-4, atol=1e-4)


def test_evaluation_path_switches_off_flex_attention():
    from moe import model as model_lib
    assert model_lib.FLEX
    with model_lib.eager_attention():
        assert not model_lib.FLEX
        mask = model_lib.build_mask(torch.zeros(1, 8, dtype=torch.long), 3)
        assert mask.dtype is torch.bool and mask.shape == (1, 1, 8, 8)
    assert model_lib.FLEX


def test_resume_reads_events_without_a_step(tmp_path):
    from moe.train import recorded
    (tmp_path / "metrics.jsonl").write_text("\n".join([
        json.dumps(dict(event="val", step=0, val={"macro_bpb": 2.0})),
        json.dumps(dict(event="throughput", tokens_per_second=1.0)),   # older runs wrote no step
    ]) + "\n")
    assert recorded(tmp_path) == {("val", 0), ("throughput", None)}


def test_muonh_keeps_matrix_norms_and_assigns_marin_groups():
    from moe import optim
    model = debug_model()
    opt = optim.build(model, spec.TRACKS["d768"])
    kinds = {opt.names[p]: g["kind"] for g in opt.param_groups for p in g["params"]}
    assert kinds["embed.weight"] == "adam" and kinds["head.weight"] == "adamh"
    assert kinds["blocks.0.ffn.router.weight"] == "adam" and kinds["blocks.0.attn.head_gate.weight"] == "adam"
    assert kinds["blocks.0.attn.q.weight"] == kinds["blocks.0.ffn.experts.w_gate"] == "muonh"
    assert kinds["blocks.0.attn_gated.down.weight"] == "muonh" and kinds["final_norm.weight"] == "adam"
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.set_schedule(1.0)
    idx = torch.randint(0, 512, (2, 64))
    model(idx, idx).backward()
    opt.step()
    for name in ("blocks.0.attn.q.weight", "head.weight"):
        p = dict(model.named_parameters())[name]
        assert not torch.equal(p, before[name])
        torch.testing.assert_close(p.norm(), before[name].norm())
    w = dict(model.named_parameters())["blocks.0.ffn.experts.w_up"]
    torch.testing.assert_close(w.norm(dim=(1, 2)), before["blocks.0.ffn.experts.w_up"].norm(dim=(1, 2)))
    ortho = optim.newton_schulz(torch.randn(3, 16, 40))
    singular = torch.linalg.svdvals(ortho)
    assert singular.min() > 0.5 and singular.max() < 1.3


def test_attention_edges():
    assert compute.attention_edges(4, -1) == 10
    assert compute.attention_edges(4096, -1) == 4096 * 4097 // 2
    assert compute.attention_edges(4096, 2047) == 2048 * 2049 // 2 + 2048 * 2048


def test_measured_cost_equals_the_closed_form():
    config = MoEConfig.baseline("d768").debug()
    assert compute.measure(config, tokens=128)["flops_per_token"] == compute.analytic(config)["total"]


@pytest.mark.parametrize("track, flops, marin", [("d768", 936_884_736, 2.81e18), ("d1024", 1_694_558_208, 1.16e19)])
def test_track_budgets(track, flops, marin):
    row = compute.report()[track]
    assert row["flops_per_token"] == flops
    # Marin's own budget formula (no head, GatedNorms or gates; non-causal attention) reproduces its budget.
    assert row["marin_convention_budget"] == pytest.approx(marin, rel=0.01)


@pytest.mark.parametrize("track", ["d768", "d1024"])
def test_reference_numbers_follow_the_ledger(track):
    """moe/references.json is read by people, not by code, so nothing else catches it going stale."""
    from moe.train import PEAK_BF16_FLOPS
    row, reference = compute.report()[track], json.loads((ROOT / "moe" / "references.json").read_text())
    planned, measured = reference["tracks"][track]["protocol"], reference["tracks"][track]["throughput_measured"]
    stale = "moe/references.json is stale: rebuild it from python -m moe.compute report"
    assert planned["flops_per_token"] == row["flops_per_token"], stale
    assert planned["training_flops"] == row["training_flops"], stale
    assert planned["parameters"] == row["parameters"], stale
    assert reference["tracks"][track]["marin"]["budget_in_marin_convention_here"] == row["marin_convention_budget"], stale
    # The published MFU is the same anchor divided by the measured throughput, so it dates the measurement.
    mfu = row["flops_per_token"] * measured["tokens_per_second"] / (measured["devices"] * PEAK_BF16_FLOPS)
    assert measured["mfu"] == pytest.approx(mfu, abs=5e-4), "the throughput was measured against a different model"


# --- plugins --------------------------------------------------------------------------------------------------

def test_plugins_are_discovered_and_built(test_plugins):
    assert {"softmax_aux", "top1_wide"} <= set(plugins.methods())
    model = debug_model("softmax_aux")
    assert model.config.option("aux_coeff") == 0.1
    idx = torch.randint(0, 512, (2, 64))
    loss = model(idx, idx)
    assert model.last_aux is not None and model.last_aux > 0
    (loss + model.last_aux).backward()


def test_ffn_plugin_changes_measured_cost(test_plugins):
    base = MoEConfig.baseline("d768").debug()
    wide = plugins.config_for("top1_wide", "d768", debug=True)
    measured = compute.measure(wide, tokens=128)["flops_per_token"]
    assert measured < compute.measure(base, tokens=128)["flops_per_token"]


def test_plugin_declarations_are_validated(tmp_path, monkeypatch):
    bad = tmp_path / "Bad"
    bad.mkdir()
    (bad / "method.json").write_text(json.dumps({"name": "Bad"}))
    (bad / "__init__.py").write_text("")
    monkeypatch.setenv("NEXTT_MOE_METHOD_PATHS", str(tmp_path))
    plugins.registry.cache_clear()
    try:
        with pytest.raises(ValueError, match="lowercase"):
            plugins.registry()
    finally:
        plugins.registry.cache_clear()


# --- data order -----------------------------------------------------------------------------------------------

def test_rows_overlap_by_one_and_updates_ignore_the_layout(tmp_path):
    tokens = np.arange(10 * 16 + 1, dtype="<u4")
    tokens.tofile(tmp_path / "t.bin")
    rows = Rows(tmp_path / "t.bin", sequence_len=16)
    assert len(rows) == 10
    block = rows.get(2, 4)
    assert block[0, 0] == 32 and block[0, -1] == block[1, 0] == 48
    assert update_rows(1, 0, 1, 4) == (4, 8) and update_rows(1, 1, 2, 4) == (6, 8)
    whole = torch.cat([x for x, _ in micro_batches(rows, 1, 0, 1, 4, 4, torch.device("cpu"))])
    split = torch.cat([x for r in range(2) for x, _ in micro_batches(rows, 1, r, 2, 4, 1, torch.device("cpu"))])
    assert torch.equal(whole, split)


# --- evaluation -----------------------------------------------------------------------------------------------

@needs_tokenizer
def test_dolma2_tokenizer_matches_known_ids():
    from moe.tokenizer import EOT, Dolma2Tokenizer
    tok = Dolma2Tokenizer()
    assert tok.vocab_size == 100_278 and tok.get_bos_token_id() == EOT == 100_257
    assert tok.encode("call |||PHONE_NUMBER||| now") == [6797, 220, 100261, 1457]
    assert tok.encode(["a", "b"], prepend=EOT) == [[EOT, 64], [EOT, 65]]
    path = os.environ.get("DOLMA2_TOKENIZER_JSON")
    if path:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(path)
        text = "def f(x):\n\treturn x**2  # 1234567 |||EMAIL_ADDRESS||| <|endoftext|>"
        assert hf.encode(text, add_special_tokens=False).ids == tok.encode(text)


@needs_tokenizer
def test_ruler_samples_fit_their_lengths():
    from moe.evaluation import ruler
    from moe.tokenizer import Dolma2Tokenizer
    tok = Dolma2Tokenizer()
    table = ruler.samples()["samples"]
    assert set(table) == {"2048", "4096", "8192"} and all(len(v) == ruler.SAMPLES for v in table.values())
    for length, rows in table.items():
        for key, value, index, lines in rows[:5]:
            text = ruler.render(key, value, index, lines)
            assert f"One of the special magic numbers for {key} is: {value}." in text
            assert text.endswith(f"The special magic number for {key} mentioned in the provided text is")
            assert 1 + len(tok.encode(text)) + ruler.TOKENS_TO_GENERATE <= int(length)


def test_recall_scoring_is_lm_eval_contains():
    from moe.evaluation import recall
    assert recall.contains("Clearance of a New Device.", "clearance of a new device")
    assert not recall.contains("1982", "1983")


def test_moe_core_weights_every_entry_equally():
    short = dict(centered={t: 0.2 for t in score.SHORT})
    ruler_result = {"2048": 0.6, "4096": 0.4, "8192": 0.1}
    result = score.moe_core(short, ruler_result, {"fda": 0.3, "swde": 0.5})
    assert len(result["centered"]) == 14
    assert result["moe_core"] == pytest.approx((10 * 0.2 + 0.6 + 0.4 + 0.3 + 0.5) / 14)
    assert result["ruler_probe"] == {"8192": 0.1}
    with pytest.raises(ValueError):
        score.moe_core(short, {"2048": 0.6}, {"fda": 0.3, "swde": 0.5})


# --- trainer and audit ----------------------------------------------------------------------------------------

def fake_resources(root, rows=64, S=64):
    out = root / "moe"
    (out / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 512, rows * S + 1, dtype=np.uint32).astype("<u4").tofile(out / "train.bin")
    for source in spec.VALIDATION_SOURCES:
        rng.integers(0, 512, 8 * S + 1, dtype=np.uint32).astype("<u4").tofile(out / "val" / f"{source}.bin")
    (out / "manifest.json").write_text(json.dumps(dict(data_revision=spec.DATA_REVISION,
                                                       tokenizer_revision=spec.TOKENIZER_REVISION,
                                                       train_tokens=rows * S + 1, outputs={})))
    return root


def test_trainer_smoke_and_checkpoint_roundtrip(tmp_path):
    from moe import train
    resources = fake_resources(tmp_path / "res")
    out = tmp_path / "run"
    train.main(["--method", "baseline", "--track", "d768", "--debug", "--resources", str(resources),
                "--output", str(out), "--micro-batch", "2", "--smoke-steps", "3", "--no-eval"])
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    kinds = [(e["event"], e["step"]) for e in events]
    assert ("val", 0) in kinds and ("val", 3) in kinds and ("end", 3) in kinds
    assert json.loads((out / "ledger.json").read_text())["updates"] == 3
    assert not json.loads((out / "run.json").read_text())["formal"]
    model = debug_model()
    opt = train.optimizer(model, spec.TRACKS["d768"])
    model(torch.randint(0, 512, (1, 64)), torch.randint(0, 512, (1, 64))).backward()
    opt.step()
    train.save(model, opt, 7, tmp_path / "ck")
    fresh = debug_model()
    fresh_opt = train.optimizer(fresh, spec.TRACKS["d768"])
    assert train.load(fresh, fresh_opt, tmp_path / "ck" / "checkpoint_00007") == 7
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_formal_mode_refuses_research_shortcuts(tmp_path):
    from moe import train
    with pytest.raises(SystemExit):
        train.main(["--method", "baseline", "--track", "d768", "--debug", "--formal",
                    "--resources", str(tmp_path), "--output", str(tmp_path / "o")])


def test_audit_rejects_an_incomplete_run(tmp_path):
    from moe import audit
    (tmp_path / "run.json").write_text(json.dumps(dict(track="d768", formal=False, method="baseline")))
    (tmp_path / "metrics.jsonl").write_text(json.dumps(dict(event="val", step=0, val={"macro_bpb": 2.0})) + "\n")
    problems, summary = audit.audit(tmp_path)
    assert not summary["passed"]
    assert any("research mode" in p for p in problems) and any("step 500" in p for p in problems)
    assert audit.check() == []
