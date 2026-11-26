"""The dolma2 tokenizer (allenai/dolma2-tokenizer) reproduced with tiktoken.

dolma2 is cl100k_base plus 22 added tokens (ids 100256-100277). tiktoken's cl100k_base ranks with those added
tokens as special tokens, encoded with every special token allowed, reproduce the Hugging Face tokenizer id for
id on ordinary text and on the PII placeholders that occur in Dolma (|||EMAIL_ADDRESS||| and friends); the
test suite checks this against the pinned tokenizer.json when it is available. The Dolma 3 HQ token counts
were produced with this tokenizer, so the track's token budgets and the data card agree.

The interface matches what nanochat's CORE scorer calls: __call__/encode with `prepend`,
get_bos_token_id(), decode().
"""
from functools import lru_cache

import tiktoken

ADDED_TOKENS = {
    "<|extra_id_0|>": 100256, "<|endoftext|>": 100257, "<|fim_prefix|>": 100258, "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260, "|||PHONE_NUMBER|||": 100261, "|||EMAIL_ADDRESS|||": 100262,
    "|||IP_ADDRESS|||": 100263, "<|im_start|>": 100264, "<|im_end|>": 100265,
    **{f"<|extra_id_{i}|>": 100265 + i for i in range(1, 11)},
    "<|endofprompt|>": 100276, "<|pad|>": 100277,
}
EOT = ADDED_TOKENS["<|endoftext|>"]


@lru_cache(maxsize=1)
def encoding():
    base = tiktoken.get_encoding("cl100k_base")
    return tiktoken.Encoding("dolma2", pat_str=base._pat_str, mergeable_ranks=base._mergeable_ranks,
                             special_tokens=ADDED_TOKENS)


class Dolma2Tokenizer:
    def __init__(self):
        self.enc = encoding()
        self.vocab_size = self.enc.n_vocab   # 100,278

    def get_bos_token_id(self):
        return EOT

    def encode(self, text, prepend=None, append=None, num_threads=8):
        single = isinstance(text, str)
        rows = self.enc.encode_batch([text] if single else list(text), allowed_special="all", num_threads=num_threads)
        if prepend is not None:
            rows = [[prepend, *r] for r in rows]
        if append is not None:
            rows = [[*r, append] for r in rows]
        return rows[0] if single else rows

    __call__ = encode

    def decode(self, ids):
        return self.enc.decode([i for i in ids if 0 <= i < self.vocab_size])   # padding rows decode to nothing

    def token_bytes(self):
        """UTF-8 byte length of every id (special tokens count zero), for bits per byte."""
        import torch
        lengths = [0 if i in ADDED_TOKENS.values() else len(self.enc.decode_single_token_bytes(i))
                   for i in range(self.vocab_size)]
        return torch.tensor(lengths, dtype=torch.int32)
