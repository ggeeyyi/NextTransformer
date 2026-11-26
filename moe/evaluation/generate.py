"""Batched greedy decoding without a KV cache, for the generative tasks (RULER, FDA, SWDE).

Prompts are left-padded with the document separator. Under the track's document mask every padding token is
its own document, and attention scores depend only on relative positions (RoPE on half the channels, the other
half position-free), so a padded prompt produces exactly the logits of the unpadded one.
"""
import torch

from moe.tokenizer import EOT


@torch.no_grad()
def greedy(model, prompts, device, *, max_new_tokens, done=None, batch_size=8, pad=EOT, vocab=100_278):
    """Generated token ids for each prompt; `done(prompt_index, generated_ids) -> bool` may stop a row early."""
    outputs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        width = max(len(prompts[i]) for i in rows)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        for r, i in enumerate(rows):
            ids[r, width - len(prompts[i]):] = torch.tensor(prompts[i])
        ids = ids.to(device)
        generated = [[] for _ in rows]
        active = [True] * len(rows)
        for _ in range(max_new_tokens):
            nxt = model(ids)[:, -1, :vocab].argmax(-1)   # never pick an embedding padding row
            ids = torch.cat((ids, nxt[:, None]), dim=1)
            for r, token in enumerate(nxt.tolist()):
                if active[r]:
                    generated[r].append(token)
                    if token == EOT or (done is not None and done(rows[r], generated[r])):
                        active[r] = False
            if not any(active):
                break
        for r, i in enumerate(rows):
            outputs[i] = generated[r]
    return outputs
