# Dense (full attention) with head-wise gated attention output

Parent: `dense_full` (att_000002).

Hypothesis: the same head-wise output gate as attnres_gate lowers BPB on the plain residual stream too; with attnres_gate it separates the gate's effect from its interaction with AttnRes.
