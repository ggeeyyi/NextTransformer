# AttnRes with one block per group (Full AttnRes)

Parent: `attnres` (att_000025).

Hypothesis: reading every earlier block as its own depth source (Full AttnRes, Kimi Team 2026) instead of eight group sums lets each sublayer retrieve finer-grained earlier representations, lowering BPB at the same training FLOPs. Only `attnres_groups` changes (8 -> 23); the extra depth reads cost about 0.1% of the budget.
