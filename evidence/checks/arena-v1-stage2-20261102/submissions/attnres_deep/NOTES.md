# Deeper, narrower AttnRes (FFN 3d, more layers)

Parent: `attnres` (att_000025).

Hypothesis: at matched training FLOPs, AttnRes benefits from depth more than from FFN width, because each added block is also an added depth source and sublayer. The FFN width drops to 4608 (3d, the lowest width in the formal grid) and the layer count is calibrated back to the anchor cost (about 27 instead of 23).
