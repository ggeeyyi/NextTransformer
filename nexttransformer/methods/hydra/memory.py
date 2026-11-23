"""Hashed n-gram embedding memory: zero-FLOP lookups of the trailing n-gram ending at each position.

For every order n and hash head h there is one table; a position looks up the n-gram of tokens t-n+1..t
(positions before the row start read a padding id), so the memory is causal. The looked-up vectors are
concatenated, RMS-normalized and projected to the model width. The tables live in the shell's `value_embeds`
ModuleDict (keys "ngram<order>_<head>"), so the optimizer treats them as embedding tables (AdamW), not as
Muon matrices; only the projection is a matrix.
"""
import torch
from torch import nn

from nexttransformer.models.backbone import Linear, norm

# Odd multipliers for the polynomial hash; one row per (order, head), one column per offset.
MULTIPLIERS = (
    (1000003, 999983, 999979, 999961), (1300021, 1299989, 1299971, 1299967),
    (1600033, 1599997, 1599991, 1599979), (1900009, 1899997, 1899991, 1899983),
    (2200013, 2199997, 2199971, 2199959), (2500009, 2499997, 2499989, 2499967),
    (2800001, 2799991, 2799971, 2799967), (3100007, 3099997, 3099991, 3099973),
)


def _largest_prime_at_most(n):
    def prime(m):
        if m < 2:
            return False
        i = 2
        while i * i <= m:
            if m % i == 0:
                return False
            i += 1
        return True
    while not prime(n):
        n -= 1
    return n


class NgramMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.orders = tuple(int(n) for n in config.option("ngram_orders"))
        self.heads = int(config.option("ngram_heads"))
        rows = int(config.option("ngram_rows"))
        self.dim = int(config.option("ngram_dim"))
        self.head_dim = self.dim // self.heads
        self.pad = config.vocab_size
        # Distinct prime table sizes decorrelate the collisions of different heads.
        self.sizes = []
        size = rows
        for _ in range(len(self.orders) * self.heads):
            size = _largest_prime_at_most(size)
            self.sizes.append(size)
            size -= 1
        self.ngram_proj = Linear(len(self.orders) * self.dim, config.width, bias=False)

    def keys(self):
        return [f"ngram{n}_{h}" for n in self.orders for h in range(self.heads)]

    def make_tables(self):
        return {key: nn.Embedding(size, self.head_dim) for key, size in zip(self.keys(), self.sizes)}

    def forward(self, idx, tables, dtype):
        B, T = idx.shape
        idx = idx.long()
        shifted = [idx]
        for back in range(1, max(self.orders)):
            if back < T:
                shifted.append(torch.cat((torch.full((B, back), self.pad, dtype=idx.dtype, device=idx.device),
                                          idx[:, :T - back]), 1))
            else:
                shifted.append(torch.full_like(idx, self.pad))
        parts = []
        slot = 0
        for n in self.orders:
            for h in range(self.heads):
                mult = MULTIPLIERS[slot % len(MULTIPLIERS)]
                code = shifted[0] * mult[0]
                for back in range(1, n):
                    code = code + (shifted[back] + 1) * mult[back]
                key = self.keys()[slot]
                parts.append(tables[key](code % self.sizes[slot]).to(dtype))
                slot += 1
        return self.ngram_proj(norm(torch.cat(parts, -1)))
