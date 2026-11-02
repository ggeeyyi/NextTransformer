# AttnRes with head-wise gated attention output

Parent: `attnres` (att_000025).

Hypothesis: a head-wise sigmoid gate on the attention output, computed from the attention input (Gated Attention, Qiu et al. 2025), adds input-dependent sparsity and removes the attention sink, lowering BPB at about 0.05% more training FLOPs. It is the only change to AttnRes.
