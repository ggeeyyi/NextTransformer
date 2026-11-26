"""Ten short-context tasks from the dense track's pinned DCLM eval bundle, scored by nanochat's CORE rules.

The tasks are the cloze/multiple-choice core shared by OLMoE (arXiv 2409.02060), OLMES (2406.08446), OLMo 3's
small-model suite (2512.13961) and DataDecide (2504.11393), plus the two short-answer tasks with the highest
signal-to-noise at small scale in Signal and Noise (2508.13144): TriviaQA and SQuAD. Formats, few-shot counts and
random baselines are DCLM's (core.yaml / eval_meta_data.csv); TriviaQA is DCLM's `triviaqa` file with its
3-shot setting. Multiple choice picks the lowest mean continuation loss; short answers count when every answer
token is the greedy argmax under teacher forcing. Prompts are left-truncated to the 4,096-token context.
"""
import csv
import json
import random
from pathlib import Path
import time

import torch
import yaml

from nanochat.core_eval import evaluate_task
from nexttransformer.evaluation.metrics import verify_core
from moe.spec import PROTOCOL
from moe.tokenizer import Dolma2Tokenizer

CORE_LABELS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "commonsense_qa", "openbook_qa",
               "lambada_openai", "squad")
TRIVIAQA = dict(label="triviaqa", dataset_uri="world_knowledge/triviaqa.jsonl", num_fewshot=[3],
                icl_task_type="language_modeling", continuation_delimiter=" ", baseline_key="triviaqa_sm_sub")
TASKS = (*CORE_LABELS, "triviaqa")


class Truncated(torch.nn.Module):
    """nanochat's scorer reads `max_seq_len` to left-crop long prompts."""

    def __init__(self, model, max_tokens):
        super().__init__()
        self.model, self.max_seq_len = model, max_tokens

    def forward(self, idx):
        return self.model(idx)


def definitions(resources):
    bundle = Path(resources) / "eval_bundle"
    entries = {t["label"]: t for t in yaml.safe_load((bundle / "core.yaml").read_text(encoding="utf-8"))["icl_tasks"]}
    with open(bundle / "eval_meta_data.csv", encoding="utf-8") as f:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) / 100 for row in csv.DictReader(f)}
    tasks = [dict(entries[label], baseline=baselines[label]) for label in CORE_LABELS]
    tasks.append(dict(TRIVIAQA, baseline=baselines[TRIVIAQA["baseline_key"]]))
    return bundle, tasks


def load(bundle, task):
    with open(bundle / "eval_data" / task["dataset_uri"], encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if task["label"] == "triviaqa":   # DCLM stores {context, answer}; the scorer reads {context, continuation}
        data = [dict(context=d["context"], continuation=d["answer"]) for d in data]
    random.Random(1337).shuffle(data)
    return data


@torch.no_grad()
def evaluate(model, resources, device, *, max_per_task=-1, max_tokens=PROTOCOL.sequence_len):
    verify_core(resources)
    bundle, tasks = definitions(resources)
    tokenizer = Dolma2Tokenizer()
    scorer = Truncated(model, max_tokens)
    accuracy, centered = {}, {}
    started = time.monotonic()
    for task in tasks:
        data = load(bundle, task)
        if max_per_task > 0:
            data = data[:max_per_task]
        meta = dict(task_type=task["icl_task_type"], dataset_uri=task["dataset_uri"],
                    num_fewshot=task["num_fewshot"][0], continuation_delimiter=task.get("continuation_delimiter", " "))
        acc = evaluate_task(scorer, tokenizer, data, device, meta)
        b = task["baseline"]
        accuracy[task["label"]], centered[task["label"]] = acc, (acc - b) / (1 - b)
    return dict(accuracy=accuracy, centered=centered, full=max_per_task <= 0, seconds=time.monotonic() - started)
