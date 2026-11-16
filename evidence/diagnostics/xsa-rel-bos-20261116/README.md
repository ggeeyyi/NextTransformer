# XSA, independent REL, and BOS-sensitive evaluation

These are completed fixed-protocol runs and post-run diagnostic evidence. Original scores are exported alongside model cards and run notes. Both methods completed 46,875 updates, 49.152B input tokens, 189 validation events, and six complete CORE-22 events with seed 42 and context 2048. Training used the shared eight-stream data order on 64 ranks (microbatch 8, accumulation 1), a layout since registered in the run audit by #57. No shared training/evaluation code or local launch infrastructure is included here.

## Original evaluation

| Method | Final validation BPB (lower) | CORE-22 (higher, ×100) | SQuAD (%) |
| --- | ---: | ---: | ---: |
| Dense baseline | 0.664460 | 33.0952 | 50.00 |
| XSA | 0.664079 | 31.7346 | 18.04 |
| Independent REL | 0.665655 | 33.1733 | 40.13 |

BPB uses the final 20,971,520-token split, not the periodic 41,943,040-token validation event. Each run's independent scorer uses the same final checkpoint. REL's independent evaluation reproduces every CORE task and validation BPB exactly; its training BPB difference is 5.8e-8. The first REL independent evaluation hit a transient shared-storage stale-file-handle error; automatic retry succeeded. A local finalizer initially treated the retryable error as terminal; this was corrected after completion and score comparison passed. No checkpoint, score, audit, or training source was rewritten.

Both methods use the shared Dense shell and fixed optimizer. XSA projects away the current effective-value direction after ordinary causal attention, retaining the self term in softmax and RoPE. REL independently replaces RoPE with a query-dependent distance bias covering 0 through 2047, retaining shared QK normalization, scaling and value embeddings. Neither implementation reproduces its source model's entire training recipe.

## Why BOS changes interpretation

`BestFitStream` starts every training row with BOS. CORE initially prepends BOS, then keeps the last 2048 tokens when an input is too long. Thus 6,895 of 10,570 SQuAD inputs (65.23%) lose their initial BOS. SQuAD here requires all answer-token argmax predictions to match under teacher forcing, using the original CORE scorer.

The diagnostic restores BOS by replacing the first retained token, keeping sequence length, remaining tokens, answer positions and weights fixed. It does not restore removed context or few-shot examples.

| Method | Original diagnostic SQuAD | Restore BOS only on cropped inputs | Gain (pp) |
| --- | ---: | ---: | ---: |
| Dense baseline | 50.0378% | 53.6329% | +3.5951 |
| XSA | 18.0416% | 50.4352% | +32.3936 |
| Independent REL | 40.1325% | 55.2223% | +15.0899 |

Original XSA and REL correct-answer counts reproduce the official evaluations exactly. Dense's earlier diagnostic original differs by four examples from the published 50.00%; its before/after comparison uses the same diagnostic run. Comparing a restored method against an unrestored baseline would be misleading.

For REL, cropped SQuAD accuracy increases from 31.6461% to 54.7788%, with 1,808 recovered answers and 213 regressions. Answer-token NLL falls from 0.9931 to 0.4634. Uncropped inputs score 56.0544%.

Matched REL controls establish sensitivity to BOS identity: on 674 cropped inputs, replacing the first retained token with a space leaves aggregate accuracy at 31.1573%, while BOS yields 55.4896%. On 350 uncropped inputs, replacing BOS with a space drops accuracy from 56.2857% to 31.1429%; deleting BOS gives 27.7143%. Same-length replacement controls distinguish BOS identity from simply changing a token or shifting sequence positions.

This effect is task-dependent. On 128 cropped BoolQ controls, REL BOS restoration reduces accuracy from 61.7188% to 53.9062%; Language ID changes from 23.4375% to 26.5625% on 128 examples with substantial paired uncertainty. LSAT-AR does not truncate and already retains BOS. BOS loss therefore cannot explain every weak task or the complete CORE ranking.

These interventions support learned dependence on the training input format. They do not identify an attention-sink circuit, prove a kernel defect, or establish a statistically reliable overall architectural gain from one seed. Diagnostic restored scores must not replace the original protocol scores. Any upstream BOS-preserving evaluator needs an explicit version and consistent evaluation across methods.

## Reproduction and review

`bos-summary.json` contains counts, paired recoveries/regressions, NLLs and descriptive intervals. `run-summaries.json` contains configuration, protocol/source fingerprints, final-checkpoint hashes, original task scores and independent comparisons. `artifact-hashes.json` identifies the retained local records without distributing datasets, machine paths or checkpoints.

`diagnose_bos.py` re-renders protocol prompts from prepared resources and supports either plugin's native final checkpoint. Run it on idle devices after activating the repository runtime:

```bash
PYTHONPATH=. NANOCHAT_DTYPE=bfloat16 torchrun --standalone --nproc-per-node=8 \
  evidence/diagnostics/xsa-rel-bos-20261116/diagnose_bos.py \
  --checkpoint CHECKPOINT --resources RESOURCES --output DIAGNOSTIC_OUTPUT \
  --tasks squad --all
```

For bounded matched strata and space/removal controls, omit `--all` and add `--controls`. This diagnostic does not modify the trainer, shared evaluator, or recorded run outputs. Model cards, original result exports and run notes accompany this evidence. Checkpoint hashes identify the locally retained checkpoints; weight hosting is separate from the original result records.

The published script was checked on the first eight SQuAD examples with all applicable controls. All 24 records, including correctness and answer-token NLL, exactly match the retained diagnostic records (`reproduction-check.json`). This bounded check verifies reproduction, not a new score estimate.
