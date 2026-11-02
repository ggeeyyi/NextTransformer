# AttnRes with element-wise gated attention, depth recalibrated

Parent: `attnres` (att_000025).

Hypothesis: the element-wise output gate, the strongest variant in Gated Attention (Qiu et al. 2025), lowers BPB by more than the cost of the compute it displaces: its d x d projection adds about 7% to the training FLOPs, so the model is calibrated back to the anchor with 22 layers and an FFN width of 5760 (-0.5%).
