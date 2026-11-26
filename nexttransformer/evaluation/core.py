"""Run the pinned upstream scorer with an independent CORE-only context policy."""
from copy import copy

import torch
import torch.distributed as dist

from nanochat.base_eval import evaluate_core as upstream_core
from nexttransformer.evaluation.context import check_full_context, core_context
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.native.model import NativeExperiment


class _InferenceConfig:
    """The forward input limit for an evaluation view. All other settings come from training."""

    def __init__(self, trained, sequence_len):
        self.trained = trained
        self.sequence_len = sequence_len

    def __getattr__(self, name):
        return getattr(self.trained, name)


def inference_view(model, max_tokens):
    """Share weights and trained attention windows, isolate forward guards and runtime state."""
    if max_tokens is not None and max_tokens <= model.config.sequence_len:
        return model
    if not isinstance(model, (ReferenceLM, NativeExperiment)):
        return model
    view = copy(model)
    # Both model families prepare a rotary cache beyond their training context.
    capacity = model.cos.size(1)
    if isinstance(model, ReferenceLM):
        view.config = _InferenceConfig(model.config, capacity)
        # _window derives even short attention windows from the *training* sequence length.
        view._window = model._window
    else:
        view.max_seq_len = capacity
    return view


class CoreInputs(torch.nn.Module):
    def __init__(self, wrapped, max_tokens):
        super().__init__()
        self.wrapped = wrapped
        # Upstream's optional left crop is controlled exclusively by this adapter.
        self.max_seq_len = max_tokens
        self.longest = 0

    def forward(self, input_ids):
        self.longest = max(self.longest, input_ids.size(1))
        return self.wrapped(input_ids)


def score_core(model, wrapper, tokenizer, device, *, max_per_task, max_tokens):
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("core_max_tokens must be a positive integer or None")
    if max_per_task <= 0:
        check_full_context(model.config, max_tokens)
    scorer = CoreInputs(wrapper(inference_view(model, max_tokens)), max_tokens)
    result = upstream_core(scorer, tokenizer, device, max_per_task=max_per_task)
    longest = torch.tensor(scorer.longest, device=device)
    if dist.is_initialized():
        dist.all_reduce(longest, op=dist.ReduceOp.MAX)
    result["coreContext"] = core_context(model.config.sequence_len, max_tokens, int(longest.item()))
    return result
