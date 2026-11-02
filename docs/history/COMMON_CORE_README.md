# Historical shared-core study: MoR, LoopFormer, and Circle

> Historical record. Paths, module names, and commands are those in use at the time; the 2026-10-24
> reorganization maps them as described in [the repository guide](../guide/REPOSITORY.md).

The 2026-10-13 native revision uses **3 prefix + 9 shared-core + 3 suffix blocks** for all three recurrent models. Historical SSSL (`dense`) has 24 independent blocks. Width, heads, FFN, tokenizer, context, data, and input budget are shared; the independent parameter counts differ because routing/conditioning differs. See the [native table](NATIVE_D24.md#architecture-and-budget).

Sharing includes matrices, VE, and residual/x0 coefficients. Windows follow unfolded execution positions, with a long final suffix layer. Sparse compaction preserves original positions. Exited MoR/Circle tokens do not contribute later core K/V; the suffix processes all tokens again.

## MoR: decisions at loop boundaries

Every token executes prefix and the first nine-block traversal. Router 1 decides whether to enter loop 2; router 2 decides whether to enter loop 3. There is no intra-loop exit or reentry. Both train/eval use `q=sigmoid(linear(RMSNorm(h)))`, continue when q≥0.5, and apply `h+q*(F(h)-h)` on selected paths.

Survival targets 1/.64/.32 imply exit proportions .36/.32/.32, mean 1.96 loops, 17.64 core blocks, and total effective depth 23.64. The slight reduction from two loops offsets the historical rectangular-attention/padding estimate. A 1,2/3,1/3 target would be about 1.58% above the historical SSSL estimate.

```text
L = L_LM + sum(dual*e + 0.5*e^2)
e = hard_survival - target
dual <- clip(dual + 0.05*global_e, -10, 10)
```

Forward uses hard counts; backward uses an ST proxy. Update dual once per complete global update. This is a population constraint, not a document quota. Rejected candidates receive no direct counterfactual LM gradient; budget gradients do not guarantee optimal allocation.

## LoopFormer: fixed two traversals

Execution is `prefix3 → core9(t=0,dt=1/2) → core9(t=1/2,dt=1/2) → coda3`, total depth 24. There is no teacher, short path, or consistency objective. The roughly 90.45M extra conditioning parameters are evaluated once per recursion and broadcast over tokens; parameter and FLOP changes need not scale together.

Historical instability evidence: around step 2710 the earlier native run's loss rose 2.725→23.845; validation BPB rose 0.869 at 2500 to 2.209 at 2750. At checkpoint 5000, combined conditioning RMS was roughly 1103–1152 and attention raw offsets ranged approximately −17273 to 11503. This identifies an unbounded amplification path without proving it caused every anomaly.

The revised adaptation uses:

1. FP32 time/step MLPs and modulation, with RMSNorm on the combined condition.
2. Four offsets per core block, `0.5*tanh(raw)`, so input/residual multipliers stay in [0.5,1.5].
3. Zero modulation projections yielding multiplier one; initializing both multiplicative gate and output projection at zero would cut gradients.
4. A separate AdamW group, peak LR approximately .000424264 and norm-1 clipping after accumulation/DDP; backbone recipe unchanged.

Log preclip condition-gradient norm, absolute modulation maximum, and saturation fraction (absolute offset >.475). The optimizer's compile-cache limit is 64 for added shapes; the native update formulas remain unchanged.

The 256→257-step pilot/resume passed: final-32 mean loss 3.2876, maximum gradient norm 5.268. Saturation reached 91.85% at step 128 and fell to .135% at 257. Bounded multipliers exclude the earlier enormous scales but do not prove long-run stability; this pilot did not reach the old ~2710-step failure point.

## Circle and reproducibility

Circle uses the same physical core but makes per-block decisions after the first loop, with mean target 17.5 and allowed core depth 9–27. See [Circle](CIRCLE_SPARSE_README.md) for gradients, curriculum, and limits. Its distribution is neither MoR's three depth options nor LoopFormer's fixed depth.

All changed scientific configurations require new runs. Keep frozen source and checkpoints for old experiments. The [validation record](NATIVE_VALIDATION.md) and [cleanup history](CLEANUP.md) preserve evidence; the new [full-cost protocol](../../PROTOCOL.md) is a separate comparison contract.
