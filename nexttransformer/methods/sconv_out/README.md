# Static ShortConv · MLP output

`sconv_out` uses the full-attention Dense shell with four static,
causal depthwise convolutions per layer. It retains ordinary residual additions,
RoPE, QK normalization, value embeddings, squared-ReLU MLPs, and the shared optimizer.

The four positions are:

- K projection output, before RoPE and QK normalization.
- V projection output, before the shell's value-embedding addition.
- Attention output, after its output projection and before residual addition.
- MLP output, after its down projection and before residual addition.

The MLP convolution is not on its expanded hidden activation. There is no Q
convolution, AttnRes, XSA, dynamic filter, or additional activation or attention gate.

Each filter computes `S(x) = x + DWConv(x; W)`, with kernel width four and zero
initial weights. Thus all filters begin as the identity and shared weights match
Dense at the same shape and seed. The trainable taps have shape `(channels, 1, 4)`
and use the existing non-matrix AdamW parameter rule. Each channel has its own
weights, shared across token positions, with no bias or extra residual scaling.

Convolution state resets between packed rows and batches. Tokens may mix across
BOS within a row, matching the fixed shell's row boundary policy. This differs
from Marin's document-masked convolution and is a disclosed adaptation. Training
and full-sequence evaluation are supported; incremental KV-cache decoding raises
an error because the shared cache does not store these convolution states.

The placement is inspired by [Inkling](https://thinkingmachines.ai/news/introducing-inkling/)
and the four-site ablation in [Marin's architecture report](https://marin.community/blog/2026/11/09/marin-535b-architecture/).
It adapts those sites to the fixed NextTransformer shell; it is not a reproduction
of either complete architecture. The independently calibrated configuration and
cost evidence determine admission. Research-prefix results are not formal results.
