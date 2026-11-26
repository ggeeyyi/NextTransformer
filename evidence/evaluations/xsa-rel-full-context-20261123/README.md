# Full-prompt CORE evaluation of the final XSA and Inkling REL checkpoints

This evaluation uses the published update-46,875 checkpoints and the pinned
CORE-22 bundle under the full-prompt policy introduced in main at
`1a0828a6b0eb8a078380ef00eb7250e0b237bf1c`. All examples and few-shot prompts are
retained, including BOS. Training context and checkpoint weights stay unchanged.
These are supplementary final-checkpoint measurements, not a new training run
or a replacement of the six historical left-cropped CORE events.

## Inkling REL beyond its distance bank

Inkling defines zero positional bias outside its learned relative-distance
bank. See `InklingRelativeLogits.forward` in the
[Transformers source](https://github.com/huggingface/transformers/blob/42efcfde46e668067a74bbd439d33f107b4a7c6c/src/transformers/models/inkling/modeling_inkling.py).
For causal query/key positions i,j:

```text
bias(i,j) = r(i) E[:, i-j]   if 0 <= i-j < 2048
          = 0              if i-j >= 2048
score(i,j) = dot(q(i), k(j)) / sqrt(head_dim) + bias(i,j)
```

This cutoff concerns relative distance, not the absolute token position.
Nearby pairs beyond position 2048 still use the learned bias. Distant pairs
retain their ordinary content attention. No RoPE, learned table extension,
clipping to the last entry, sliding-window mask, or prompt crop is introduced.

The published plugin rejects inputs longer than its table. The evaluation-only
adapter restores the original Inkling boundary behavior without editing that
fingerprinted training implementation. It pads an ephemeral table tensor with
zeros for the existing causal kernel; learned tensors and checkpoint bytes stay
unchanged. Inputs up to 2048 tokens take the exact original branch. Tests compare
the extended path with an independent direct expression, check causal prefixes,
and verify unchanged parameter state.

`evaluate.py` uses the pinned upstream scorer and shared input/context wrappers.
It avoids the current shared evaluator's blanket REL length rejection, records
the longest measured prompt and asserts the standard submitted context metadata.
The same scorer, tokenizer, answer handling, centering and task set are retained.
The trained attention policies are retained too: XSA uses its 2,048-token
backward window, while REL keeps full causal content attention. Full prompts
do not silently enlarge XSA's trained window.
The current shared `nexttransformer.evaluate` entry point still requires a cap
for this plugin; use this reproducible adapter for the supplementary evaluation.

## Reproduce

From the repository root, using a configured environment and complete checkpoint:

```bash
torchrun --nproc-per-node=8 \
  evidence/evaluations/xsa-rel-full-context-20261123/evaluate.py \
  --site-config site.local.json \
  --checkpoint RUN/checkpoint_00046875 --output full-core.json
```

The script handles XSA and REL. Passing `--core-max-tokens 2048` reproduces the
historical crop and disables the REL extension. It does not train or choose a
checkpoint. Model checksums, evaluator commit/file hashes, protocol/data/bundle
hashes and full task scores are recorded in the output. The actual measurement
uses 32 ranks; CORE splits examples across ranks without changing their prompts.

`verify_gpu.py` reproduces the synthetic GPU checks with the same four-node,
eight-rank-per-node launch and runtime configuration. `gpu-verification.json`
records rank zero's relative errors after every rank passed the independent
dense-oracle checks. CPU checks are in `tests/test_rel_full_context.py`.

The evaluation was requested after the methods were published. No held-out
scores were used to choose a seed, checkpoint or extrapolation rule; zero bias
outside the bank follows the primary Inkling implementation. Original results,
audits and cost proofs remain immutable.

The full-prompt runs use 32 ranks. The reproduction controls use 16 ranks per model on disjoint node pairs. Compute activations are BF16; the raw `dtype` field names the first checkpoint parameter, which may be FP32. Public full-prompt records also declare `computeDtype` explicitly.
