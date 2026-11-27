# AttnRes with a mixture-of-experts MLP (8 experts, top-2)

Parent: `attnres` (att_000025).

Hypothesis: at matched training FLOPs, a top-2 mixture of 8 experts (capacity 1.25, so each token gets about the dense FFN's compute) reaches lower BPB than the dense FFN, because it holds about four times the FFN parameters (Shazeer et al. 2017; Fedus et al. 2022). Routing is causal (position priority within a sequence), shapes are static, and empty capacity slots are charged.
