"""RULER S-NIAH-1 (Hsieh et al., 2024, arXiv 2404.06654), as lm-evaluation-harness generates it.

lm_eval/tasks/ruler: niah_single_1 = a "repeat" haystack (the sentence below, joined by newlines), one needle
"One of the special magic numbers for {key} is: {value}." with an adjective-noun key (wonderwords) and a 7-digit
value, inserted before a random haystack line; the singular template; the generation prefix
"The special magic number for {key} mentioned in the provided text is"; scoring by string_match_all
(case-insensitive containment of the value in the stripped prediction). The haystack size per length follows
RULER's search (grow by 25 lines, or 5 below 4K, until input + answer + 128 generated tokens no longer fit),
measured with the dolma2 tokenizer.

The sampled parameters (key, value, needle line, haystack lines) are generated once by
`python -m moe.evaluation.ruler build` and committed in data/ruler_niah_single_1.json, so evaluation needs no
extra package and every method scores identical prompts. Lengths 2,048 and 4,096 are scored; 8,192 is a
length-extrapolation probe. Deviation from RULER: generation stops after 32 tokens (RULER allows 128), at the
end-of-text token, or as soon as the value appears; the 7-digit value is four tokens.
"""
import argparse
import json
from pathlib import Path
import random
import re

import torch
import torch.distributed as dist

from moe.evaluation.generate import greedy
from moe.tokenizer import EOT, Dolma2Tokenizer

DATA = Path(__file__).resolve().parent / "data" / "ruler_niah_single_1.json"
SCORED_LENGTHS = (2048, 4096)
PROBE_LENGTHS = (8192,)
SAMPLES = 500
TOKENS_TO_GENERATE = 128      # RULER's budget, used to size the haystack
MAX_NEW_TOKENS = 32
HAYSTACK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE = "One of the special magic numbers for {key} is: {value}."
TEMPLATE = ("A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you "
            "about the number afterwards.\n{context}\nWhat is the special magic number for {query} mentioned in the "
            "provided text?")
GEN_PREFIX = "The special magic number for {query} mentioned in the provided text is"


def render(key, value, index, lines):
    sentences = [HAYSTACK] * lines
    sentences.insert(index, NEEDLE.format(key=key, value=value))
    question = TEMPLATE.format(context="\n".join(sentences), query=key)
    return question + " " + GEN_PREFIX.format(query=key)


def postprocess(text):
    return re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip()


def build(tokenizer, words, lengths=SCORED_LENGTHS + PROBE_LENGTHS, samples=SAMPLES, seed=42):
    out = {}
    for length in lengths:
        rng = random.Random(f"ruler-niah-single-1/{length}/{seed}")
        incremental = 25 if length >= 4096 else 5

        def draw(lines):
            key = rng.choice(words)
            value = str(rng.randint(10 ** 6, 10 ** 7 - 1))
            return key, value, rng.randrange(lines)

        lines = incremental
        while True:   # RULER's haystack search on one draw
            key, value, index = draw(lines)
            total = 1 + len(tokenizer.encode(render(key, value, index, lines) + " " + value))
            if total + TOKENS_TO_GENERATE > length:
                lines -= incremental
                break
            lines += incremental
        rows = []
        for _ in range(samples):
            used = lines
            while used > 0:
                key, value, index = draw(used)
                if 1 + len(tokenizer.encode(render(key, value, index, used))) + TOKENS_TO_GENERATE <= length:
                    rows.append([key, value, index, used])
                    break
                used -= incremental
        out[str(length)] = rows
    return out


def samples():
    return json.loads(DATA.read_text())


@torch.no_grad()
def evaluate(model, device, *, lengths=SCORED_LENGTHS + PROBE_LENGTHS, limit=None, batch_size=8):
    tokenizer = Dolma2Tokenizer()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    capacity = model.cos.size(1)
    table = samples()["samples"]
    result = {}
    for length in lengths:
        if length > capacity:
            continue
        rows = table[str(length)][:limit]
        mine = list(range(rank, len(rows), world))
        prompts = [[EOT] + tokenizer.encode(render(*rows[i])) for i in mine]

        def found(j, generated):   # the value is already in the prediction: the score cannot change
            return rows[mine[j]][1].lower() in tokenizer.decode(generated).lower()
        outputs = greedy(model, prompts, device, max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size, done=found)
        correct = torch.zeros(len(rows), device=device)
        for i, ids in zip(mine, outputs):
            prediction = postprocess(tokenizer.decode([t for t in ids if t != EOT]))
            correct[i] = float(rows[i][1].lower() in prediction.lower())
        if world > 1:
            dist.all_reduce(correct)
        result[str(length)] = correct.mean().item()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("build",))
    args = p.parse_args(argv)
    import importlib.metadata
    import wonderwords   # build-time only: RULER's key vocabulary
    r = wonderwords.RandomWord()
    words = sorted({f"{a}-{n}" for a in r._categories["adjectives"] for n in r._categories["nouns"]})
    data = dict(task="ruler_niah_single_1", source="lm-evaluation-harness lm_eval/tasks/ruler (NVIDIA RULER)",
                tokenizer="dolma2", wonderwords=importlib.metadata.version("wonderwords"),
                fields=["key", "value", "needle_line", "haystack_lines"], samples=build(Dolma2Tokenizer(), words))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print({k: (len(v), v[0][3]) for k, v in data["samples"].items()})


if __name__ == "__main__":
    main()
