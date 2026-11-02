# Circle Loop: how an LLM-assisted design was researched

The first arena example. Circle Loop is the `circle` method of the native-v3 study ([model card](../../results/methods/circle.json),
[run record](../../results/runs/circle.json), [mechanism](../history/CIRCLE_SPARSE_README.md)). Its
[disclosure card](../../results/arena/circle-loop.json) classifies it as **L3**: the idea and several turning points
came from a person; LLMs assessed novelty, formalized the mechanism, designed the halter, budget controller, and loss,
and implemented, tested, and launched every version. It is a `legacy` entry: it predates `nextt-arena-v1` and does not
meet it, for the reasons at the end.

This history was reconstructed on 2026-10-26 from the repository's commits and the maintainer's ChatGPT and Codex
conversations. The conversations are private; dates are Asia/Shanghai.

## Timeline

| Date | Who | What happened | Evidence |
|---|---|---|---|
| 10-09 | Person | Asks ChatGPT whether a looped Transformer can be a *circle*: prefix and suffix blocks around a ring of blocks with no absolute depth, where each token decides where to leave the ring | ChatGPT conversation "Looped Transformer Circle Design" |
| 10-09 | ChatGPT | Compares the idea with Universal Transformer, Zero Token Transformer, and Mixture-of-Recursions; argues that the ring itself is not new and that the open question is online, per-block exit from computed state; formalizes a shared halter that receives no step count | Same conversation |
| 10-10 | Codex | Writes an implementation plan on the GPT-2 backbone: 2 prefix + 8 ring + 2 suffix blocks, per-block exit, one shared halter, exited tokens frozen but kept as context; estimates that a mean of about 18 ring steps matches the dense model's compute | Commit `e4bc6c6` |
| 10-10–12 | Codex | Surveys recent looped-model work and proposes a different research question, the value of continued computation for later tokens; the person keeps Circle as the main line | Codex research conversation |
| 10-11 | Codex | Circle-ST: straight-through halting with an EMA PI controller on mean depth; removes the earlier fixed cost penalty. Trained on 8 × A800 | Commits `9ea4aa3`, `e1e94b9` |
| 10-12 | Person | Moves all models to the native nanochat d24 shell with matched training FLOPs; accepts the agent's proposal of at most 20 ring steps with a mean of 18; asks whether Circle's algorithm can be improved | nanochat-loop conversation |
| 10-12 | Codex | Proposes an enhanced exit: halter features from the state and its change, plus depth biases; the person approves; launched with the other three models on 8 × A800 each | Commits `fdc22ab`, `0ea4cb1` |
| 10-13 | Person | Sees Circle trailing at 8,000 updates (subset CORE 19.94 against 21.98 for the model now called SSSL). Questions why exited tokens are still computed, proposes computing only surviving tokens as MoR does, a higher depth cap with a moderate mean, and three prefix and suffix blocks | nanochat-loop conversation |
| 10-13 | Codex | Proposes the final design: 3 + 9 + 3 blocks, 9–27 ring steps with a mean of 17.5, a halter with state-change features and 18 depth biases, a maximum-entropy depth target with a dual controller, and a random-route curriculum. The person approves it and has MoR and LoopFormer adopt the same 3 + 9 + 3 layout | Commits `ad97628`, `c42fdae`; [mechanism](../history/CIRCLE_SPARSE_README.md) |
| 10-13–14 | Person, Codex | Four formal launches on ModelArts: two exit before training (mount path, reused output directory), one stops at update 430 on a storage-quota error without a checkpoint, and `native-v3-core8k-20261014-r2` runs to completion | nanochat-loop conversation; [run record](../../results/runs/circle.json) |

## Result

At update 46,875: validation BPB **0.6912** and CORE **28.97**, against 0.6671 and 33.78 for SSSL in the same cohort and
0.6672 and 33.10 for the Dense Baseline. Its historical training estimate is 4.977 GFLOPs per input token, 4.2% above
the cohort's SSSL estimate of 4.775. Circle Loop did not beat the baselines; it is the example because its research
process is fully known.

## Why it is `legacy`

- It was researched and trained before `nextt-arena-v1` existed.
- Its run belongs to the native-v3 cohort, which charges the historical matrix-only FLOPs estimate, and its training
  compute is 4.2% above that cohort's SSSL. The arena requires an admitted reference configuration and a cumulative audit
  within 2%.
- Candidates were judged on periodic validation BPB and subset CORE, which the arena reserves for the formal run.
- No checkpoint or training logs are published.

An arena-v1 version would recalibrate Circle Loop under the operator ledger, select with the dev split, and train once
more under `nextt-reference-50b-v1`.

## What the example shows

- **Where the person mattered:** the original idea; the decision to keep Circle when the agent suggested another
  question; noticing that exited tokens were still computed; and the 3 + 9 + 3 layout constraint.
- **Where the agents mattered:** the novelty assessment, the halter and budget controller, the loss and curriculum,
  cost estimates, all code and tests, launch commands, and diagnosis of every failed launch.
- **What an L0 entry would have to do alone:** notice a semantic inefficiency in its own implementation and change the
  design after a disappointing intermediate result, without using reserved evaluation signals to decide.
