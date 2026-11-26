"""MoE-CORE: one score, the mean centered accuracy over 14 entries.

    short (10)  ARC-E, ARC-C, HellaSwag, PIQA, WinoGrande, CSQA, OBQA, LAMBADA, SQuAD, TriviaQA
    long  (4)   RULER S-NIAH-1 at 2,048 and at 4,096 tokens, FDA, SWDE

centered = (accuracy - chance) / (1 - chance), as in DCLM's CORE; generative and short-answer tasks have chance 0.
Every entry weighs the same. The RULER 8,192-token probe is reported but not scored.
"""
from moe.evaluation.ruler import SCORED_LENGTHS
from moe.evaluation.short import TASKS as SHORT

LONG = (*(f"ruler_niah_{n // 1024}k" for n in SCORED_LENGTHS), "fda", "swde")
ENTRIES = (*SHORT, *LONG)


def moe_core(short, ruler, recall):
    if any(str(n) not in ruler for n in SCORED_LENGTHS) or set(recall) != {"fda", "swde"}:
        raise ValueError("MoE-CORE needs every short and long entry")
    centered = dict(short["centered"])
    centered.update({f"ruler_niah_{n // 1024}k": ruler[str(n)] for n in SCORED_LENGTHS})
    centered.update({k: recall[k] for k in ("fda", "swde")})
    if tuple(centered) != ENTRIES:
        raise ValueError("MoE-CORE needs every short and long entry")
    long = [centered[k] for k in LONG]
    return dict(moe_core=sum(centered.values()) / len(centered), short=sum(short["centered"].values()) / len(SHORT),
                long=sum(long) / len(long), centered=centered,
                ruler_probe={k: v for k, v in ruler.items() if int(k) not in SCORED_LENGTHS})
