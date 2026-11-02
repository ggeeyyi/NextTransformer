# AttnRes with Canon layers on every sublayer input

Parent: `attnres` (att_000025).

Hypothesis: a residual causal depthwise convolution over the last four positions on every sublayer input (Canon-A/C layers, Allen-Zhu 2025) gives attention and MLP cheap local token mixing, as the short convolutions of the linear methods do, and lowers BPB at about 0.04% more training FLOPs.
