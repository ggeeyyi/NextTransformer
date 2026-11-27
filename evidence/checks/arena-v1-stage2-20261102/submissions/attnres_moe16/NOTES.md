# AttnRes with a mixture-of-experts MLP (16 experts, top-4)

Parent: `attnres` (att_000025).

Hypothesis: at matched training FLOPs, a top-4 mixture of 16 experts (capacity 1.25, so each token gets about the dense FFN's compute) reaches lower BPB than the dense FFN, because it holds about four times the FFN parameters (Shazeer et al. 2017; Fedus et al. 2022). Routing is causal (position priority within a sequence), shapes are static, and empty capacity slots are charged. Finer experts (16 of half the width, top-4) keep the compute and the parameters of attnres_moe and give more expert combinations per token (fine-grained experts, Dai et al. 2024, DeepSeekMoE).
