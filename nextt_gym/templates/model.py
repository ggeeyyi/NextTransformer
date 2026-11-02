"""Template submission: a plain causal Transformer that satisfies the gym model contract.

Copy this file and config.json, then change the architecture. Contract summary
(full text: nextt_gym/spec.md, resource nextt://spec):
  build(config) -> nn.Module           required; the module is fully initialized
  forward(idx, targets=None, loss_reduction="mean")
      targets None   -> logits [B, T, >=vocab]
      "mean"         -> scalar training objective (may include auxiliary losses)
      "none"         -> per-token cross-entropy [B, T] without auxiliary losses
  stage_for_step(config, step, total_steps) -> str     optional (default "joint")
  set_stage(model, stage, *, audit, step, total_steps)  optional
Attention must go through nanochat.flash_attention.flash_attn.flash_attn_func
(counted with exact causal-edge rules) or explicit matmul/softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotary(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), -1)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.c_q = nn.Linear(width, width, bias=False)
        self.c_k = nn.Linear(width, width, bias=False)
        self.c_v = nn.Linear(width, width, bias=False)
        self.c_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        shape = (b, t, self.heads, self.head_dim)
        q = rms_norm(rotary(self.c_q(x).view(shape), cos, sin))
        k = rms_norm(rotary(self.c_k(x).view(shape), cos, sin))
        v = self.c_v(x).view(shape)
        y = flash_attn.flash_attn_func(q, k, v, causal=True)
        return self.c_proj(y.reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.c_fc = nn.Linear(width, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.attn, self.mlp = Attention(width, heads), MLP(width, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        return x + self.mlp(rms_norm(x))


class TemplateLM(nn.Module):
    def __init__(self, vocab_size, sequence_len, width, heads, n_layer, mlp_width):
        super().__init__()
        self.vocab_size = vocab_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, width),
            h=nn.ModuleList(Block(width, heads, mlp_width) for _ in range(n_layer))))
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        head_dim = width // heads
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(sequence_len, dtype=torch.float32), inv)
        self.register_buffer("cos", angles.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, :, None, :], persistent=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, std=1.0)
        nn.init.normal_(self.lm_head.weight, std=0.001)
        for block in self.transformer.h:
            for linear in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.c_fc):
                nn.init.uniform_(linear.weight, -(3 ** .5) * linear.in_features ** -.5, (3 ** .5) * linear.in_features ** -.5)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        t = idx.shape[1]
        x = rms_norm(self.transformer.wte(idx))
        cos, sin = self.cos[:, :t].to(x.dtype), self.sin[:, :t].to(x.dtype)
        for block in self.transformer.h:
            x = block(x, cos, sin)
        logits = self.lm_head(rms_norm(x)).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction=loss_reduction)
        return loss.view(targets.shape) if loss_reduction == "none" else loss


def build(config):
    return TemplateLM(**config)
