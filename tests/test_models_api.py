"""Public model API: rebuild a model from a saved checkpoint, and sample from it."""
from dataclasses import asdict

import pytest
import torch

from nexttransformer.generate import generate
from nexttransformer.models import build_model, config_from_dict, load_model
from nexttransformer.models.config import HYBRID, debug_config
from nexttransformer.models.native.config import NativeModelConfig, NativeTrainConfig
from nexttransformer.runtime.checkpoint import save_checkpoint


class Stream:
    def state_dict(self): return {}


class Tokenizer:
    def get_bos_token_id(self): return 0
    def encode(self, text, prepend=None): return [prepend, *[(ord(c) % 60)+1 for c in text]]
    def decode(self, ids): return " ".join(map(str, ids))


def tiny_native(architecture):
    return NativeModelConfig(architecture=architecture, vocab_size=128, width=32, heads=2, mlp_width=128,
        sequence_len=16, dense_layers=6, pre_layers=1, core_layers=2, post_layers=1, fourier_dim=16,
        circle_max_steps=6, circle_target_steps=4., circle_random_warmup=0, circle_route_transition=0)


def configs():
    return [debug_config("dense_full"), debug_config("dsa"), *[debug_config(m) for m in sorted(HYBRID)],
            tiny_native("mor"), tiny_native("circle")]


@pytest.mark.parametrize("config", configs(), ids=lambda c: getattr(c, "method", None) or c.architecture)
def test_load_model_restores_the_saved_model(config, tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = build_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    save_checkpoint(tmp_path, model, optimizer, Stream(), config, NativeTrainConfig(), dict(step=3, software=dict(torch=torch.__version__)), 0, 1)
    assert config_from_dict(asdict(config)) == config
    from environments.eval.checkpoint import read_model_payload
    checkpoint = tmp_path/"checkpoint_00000003"
    payload = read_model_payload(checkpoint)
    assert payload["model_config"] == asdict(config)
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        loaded, saved = load_model(checkpoint)
    assert saved["state"]["step"] == 3 and type(loaded.config) is type(config)
    for name, tensor in model.state_dict().items(): assert torch.equal(tensor, loaded.state_dict()[name]), name
    x = torch.randint(config.vocab_size, (1, config.sequence_len))
    model.eval(); loaded.eval()
    with torch.no_grad():
        a, b = model(x), loaded(x)
    a, b = (a["logits"], b["logits"]) if isinstance(a, dict) else (a, b)
    assert torch.equal(a, b)


def test_greedy_generation_is_deterministic():
    model = build_model(debug_config("dense_full"))
    out = [generate(model, Tokenizer(), "hello", max_new_tokens=5, temperature=0) for _ in range(2)]
    assert out[0] == out[1] and len(out[0].split()) == 5
    sampled = generate(model, Tokenizer(), "hello", max_new_tokens=5, temperature=1., seed=1)
    assert sampled == generate(model, Tokenizer(), "hello", max_new_tokens=5, temperature=1., seed=1)
