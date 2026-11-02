# AttnRes with a value embedding in every layer

Parent: `attnres` (att_000025).

Hypothesis: value embeddings cost almost no training FLOPs, so giving every layer one (instead of every other layer) adds about 0.55B embedding parameters for free and lowers BPB at matched training FLOPs.
