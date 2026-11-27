<div align="center">

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/banner.jpg" alt="NextTransformer: toward the next generation of Transformers" width="100%"></a>

<h3>The comprehensive, cutting-edge open-source framework, benchmark<br>and auto-research platform for next-generation Transformers</h3>

<p>
<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/readme/button-observatory-en.webp" alt="Explore the observatory" height="60"></a>
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/"><img src=".github/assets/readme/button-docs-en.webp" alt="Read the docs" height="60"></a>
<a href="#quick-start"><img src=".github/assets/readme/button-quickstart-en.webp" alt="Quick start" height="60"></a>
</p>

<p>
<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-live-96C1F1?style=flat&amp;labelColor=D1E5FF" height="19"></a>&thinsp;
<a href="https://interactive-active-ai.github.io/NextT-Blogs/"><img alt="NextTransformer research blog" src="https://img.shields.io/badge/Blog-F6D7AB?style=flat" height="19"></a>&thinsp;
<a href="https://huggingface.co/NextTransformer"><img alt="Model weights on Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-models-F5CE70?style=flat&amp;logo=huggingface&amp;logoColor=79501F&amp;labelColor=FFF0BB" height="19"></a>&thinsp;
<a href="https://discord.gg/k2XPXW6Jz"><img alt="Join Discord" src="https://img.shields.io/badge/Discord-D3C2F3?style=flat&amp;logo=discord&amp;logoColor=574589" height="19"></a>&thinsp;
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/community/#wechat"><img alt="WeChat group" src="https://img.shields.io/badge/WeChat-F5CED9?style=flat&amp;logo=wechat&amp;logoColor=8B4161" height="19"></a>&thinsp;
<a href="https://github.com/Interactive-Active-AI/NextTransformer/discussions"><img alt="GitHub Discussions" src="https://img.shields.io/badge/Discussions-C4D4F6?style=flat&amp;logo=github&amp;logoColor=475D8A" height="19"></a>
<br>
<img alt="Python 3.10–3.13" src="https://img.shields.io/badge/Python-3.10--3.13-A4CBEB?style=flat&amp;logo=python&amp;logoColor=345B86&amp;labelColor=D7EBFA" height="19">&thinsp;
<img alt="PyTorch 2.9.1" src="https://img.shields.io/badge/PyTorch-2.9.1-F4AF98?style=flat&amp;logo=pytorch&amp;logoColor=9D4436&amp;labelColor=FFE0D2" height="19">&thinsp;
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/MIT-E3D7F4?style=flat" height="19"></a>&thinsp;
<a href="CITATION.cff"><img alt="Cite NextTransformer" src="https://img.shields.io/badge/Cite-F6D7AB?style=flat" height="19"></a>&thinsp;
<img alt="English (current language)" src="https://img.shields.io/badge/English-A9C7EB?style=flat" height="19">&nbsp;<a href="README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-E9EFF6?style=flat" height="19"></a>
</p>

</div>

<br>

**When the training budget is fixed, how should a language model spend its computation?** NextTransformer trains
recurrent depth, sparse and linear attention, residual-stream variants and hybrids on the same data, for the same
49.152B tokens, at the same theoretical training FLOPs per token, and evaluates them at the same checkpoints. Every
result is an audited, file-backed record, and AI agents can research new architectures under the very same rules.

## News

- <img src="https://img.shields.io/badge/NEW-E5607E?style=flat-square" alt="New" height="18"> **2026-11-26** · **240 TPP MoE residual results** are live across both tracks: all five arms (Baseline, HC, mHC, SimpleHC, iHC) on [d768-240tpp](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp), plus Baseline and iHC on [d1024-240tpp](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d1024-240tpp).
- <img src="https://img.shields.io/badge/NEW-E5607E?style=flat-square" alt="New" height="18"> **2026-11-23** · The **[MoE track](#the-moe-track)** is open: two Marin-derived backbones, a [240 TPP comparison](moe/PROTOCOL.md#240-tpp-comparison) at 14.07B and 34.06B tokens, MoE-CORE and a shared throughput harness. [Read the MoE protocol](https://interactive-active-ai.github.io/NextTransformer/docs/moe/protocol/).
- <img src="https://img.shields.io/badge/LIVE-8B5CF6?style=flat-square" alt="Live" height="18"> **2026-11-14** · **Arena v2** is live: agent-native research end to end, through a [research gym](gym/README.md) with MCP tools, metered compute and research traces.
- **2026-11-13** · The [2K scaling study is open](https://interactive-active-ai.github.io/NextTransformer/#scaling), starting with 16 T3 configurations across four families. [Choose a configuration, find reference code and contribute](https://interactive-active-ai.github.io/NextTransformer/#contribute).
- <img src="https://img.shields.io/badge/BLOG-F6D7AB?style=flat-square" alt="Blog" height="18"> **2026-11-11** · The research blog is live. The first post traces [how Opus found AttnRes-Canon](https://interactive-active-ai.github.io/NextT-Blogs/attnres-canon/en/). [Blog repository](https://github.com/Interactive-Active-AI/NextT-Blogs)
- <img src="https://img.shields.io/badge/COMMUNITY-2A78D6?style=flat-square" alt="Community" height="18"> **2026-11-04** · **[NextTransformer-V](https://github.com/Enderfga/NextTransformer-V)** by [@Enderfga](https://github.com/Enderfga): a community multimodal track that trains every NextTransformer method on interleaved image and text tokens under the same compute-matched contract. Its first study, pre-registered, asks where hybrid backbones get the 2D layout of an image.
- <img src="https://img.shields.io/badge/NEW-E5607E?style=flat-square" alt="New" height="18"> **2026-10-31** · NextTransformer is now open source: 31 finished runs, 36 model cards, the full protocol, the documentation and the observatory.

## Highlights

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/methods/zoo/"><img src=".github/assets/readme/card-framework-en.webp" alt="Many architectures, one framework: attention, sparse, linear, looped, residual and hybrid models such as GQA, NSA, MoBA, GDN, KDA, Ouro, AttnRes and GDN+SWA plug into one shared trainer, data pipeline and evaluation." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/"><img src=".github/assets/readme/card-protocol-en.webp" alt="A fair, explicit protocol: same data, tokens, optimizer, schedule and evaluation for every model; training compute matched to within 1% and audited over the whole run." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/#scaling"><img src=".github/assets/readme/card-scale-en.webp" alt="2K scaling study at fixed compute: Dense, MoE, KDA–DSA and Fixed Recursive each sweep four depths, 16 T3 configurations at 10²⁰ training FLOPs per run with a 2,048-token context. T3 comes first, then T4, T5 and a held-out C200 test." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/arena/task/"><img src=".github/assets/readme/card-agent-en.webp" alt="Agent-native research: hand the arena task to your agent; the full version, Arena v2, has been live since November 14, 2026." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp"><img src=".github/assets/readme/card-moe-en.webp" alt="MoE track: sparse experts under one contract. Two Marin-derived sizes, d768 at 14.07B tokens and d1024 at 34.06B tokens, both 256 experts with top-4 routing at 240 tokens per active parameter, scored by MoE-CORE." width="100%"></a>

## The observatory

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/readme/observatory.webp" alt="The NextTransformer observatory: learning curves, leaderboard, model atlas and arena" width="100%"></a>

Learning curves of every run, a replayable leaderboard, all 22 CORE tasks, architecture diagrams and model cards, in
your browser.

## Results

<!-- generated:leaderboard -->
<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard"><img src=".github/assets/readme/results-en.svg" alt="CORE at update 46,875 for all 47 runs, against the Dense Baseline" width="100%"></a>

[Full leaderboard](https://interactive-active-ai.github.io/NextTransformer/docs/results/leaderboard/) · [Training replay](https://interactive-active-ai.github.io/NextTransformer/#leaderboard)
<!-- /generated:leaderboard -->

## Quick start

**1. Install** (Linux, NVIDIA GPUs, [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Interactive-Active-AI/NextTransformer.git && cd NextTransformer
uv sync --frozen --extra gpu --extra upstream && source .venv/bin/activate
uv pip install --no-deps --require-hashes -r requirements/linear.txt   # linear-attention kernels
```

**2. Prepare the data** (ClimbMix at a pinned revision, about 600 GB, plus the tokenizer and CORE bundle):

```bash
cp site.native.example.json site.local.json        # set runtime_dir, resources and data_dir
hf download karpathy/climbmix-400b-shuffle --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 --local-dir /path/to/climbmix-400b-shuffle
source launch/env.sh --site-config site.local.json
python -m nexttransformer.prepare --site-config site.local.json
```

**3. Train on one node with 8 GPUs:**

```bash
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full --micro-batch 8 --formal --resume auto
```

**4. Multi-node training (2 nodes × 8 GPUs)**

Install `uv pip install accelerate==1.15.0` in the training environment. Submit one distributed job with the following per-node entry point. Map the platform's node rank, main node address and port to the three variables shown:

```bash
export FLA_DISABLE_BACKEND_DISPATCH=1   # registered FLA Triton path (linear and hybrid methods)
accelerate launch --config_file launch/accelerate.yaml \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  -m nexttransformer.train --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal
```

On Slurm, submit `sbatch launch/slurm.sbatch` for the same example. The batch script gets the launch parameters from the allocation. [Cluster setup and other node counts](https://interactive-active-ai.github.io/NextTransformer/docs/getting-started/quick-start/#multi-node)

Evaluation runs on schedule during training. The [quick start guide](https://interactive-active-ai.github.io/NextTransformer/docs/getting-started/quick-start/)
covers smoke tests, evaluation, auditing and exporting a result.

## Model zoo

The [NextTransformer organization on Hugging Face](https://huggingface.co/NextTransformer) is the home for model weights. Check its **Models** tab for available downloads. The [leaderboard](https://interactive-active-ai.github.io/NextTransformer/#leaderboard) lists evaluated runs, including those whose weights have not yet been released.

Contributor-hosted weights: [AttnRes-Canon (49B comparison)](https://huggingface.co/GY2233/nextt-arena-attnres-full-canon-abcd) · [Dense (200B-token scaling run)](https://huggingface.co/Enderfga/nextt-scaling-dense-200b-4k).

<details open>
<summary><b>Browse architectures by family</b></summary>

<!-- generated:model-zoo -->
| Family | Models |
|---|---|
| <img src="https://img.shields.io/badge/Baseline-7A8288?style=flat-square" alt="Baseline"> | Dense Baseline |
| <img src="https://img.shields.io/badge/Residual-33456E?style=flat-square" alt="Residual"> | Standard Residual · AttnRes · HC · mHC · MSAR · AttnRes-Canon · SimpleHC |
| <img src="https://img.shields.io/badge/Looped-8B5CF6?style=flat-square" alt="Looped"> | UT-ACT · Fixed Recursive · Huginn · Ouro · DeepLoop · MoR · LoopFormer · Circle Loop |
| <img src="https://img.shields.io/badge/Sparse-E5607E?style=flat-square" alt="Sparse"> | DSA · QSA · NSA · SeerAttention-style · MoBA |
| <img src="https://img.shields.io/badge/Linear-3F7FD9?style=flat-square" alt="Linear"> | DeltaNet · GSA + QK-norm · GLA · KDA · RetNet · ELU-Norm · GDN |
| <img src="https://img.shields.io/badge/Hybrid-F07856?style=flat-square" alt="Hybrid"> | GDN–MHA 3:1 · GDN–MHA 1:1 · GDN–SWA · DeltaNet–MHA · KDA–MLA-NoPE · GDN–QSA · MoBA–MHA |
| <img src="https://img.shields.io/badge/Others-D99A2B?style=flat-square" alt="Others"> | SSSL · MQA · GQA · MLA-QK · MLA-NoPE-QK · Gated Attention · elementwise · Gated Attention · headwise · XSA · Exclusive Self Attention · Inkling REL · Static ShortConv · MLP output · Dynamic-KV ShortConv · MLP output · Hydra |
<!-- /generated:model-zoo -->

\* Results coming soon. Papers, adaptations and configurations are in the [model zoo](https://interactive-active-ai.github.io/NextTransformer/docs/methods/zoo/).

</details>

## The protocol at a glance

| | |
|---|---|
| **Data** | ClimbMix at a pinned revision · one 32,768-token tokenizer · 2,048-token context · one global sample order |
| **Budget** | 49.152B input tokens · 46,875 updates of 1,048,576 tokens · seed 42 |
| **Compute** | 4.672 GFLOPs per input token (forward, backward, auxiliary and optimizer work); ±1% at admission, ±2% over the run |
| **Evaluation** | Validation BPB at 189 points · full CORE-22 at six checkpoints · an independent final BPB |
| **Baseline** | Dense: full causal attention in every layer, trained under the same budget |

Parameters, memory and speed are outcomes, not controls. The [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/)
explains every rule and the [compute accounting](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/compute/)
what is counted.

## The MoE track

Sparse models get the same treatment as dense ones, under their own contract: same data in the same order, same
training FLOPs per token, same evaluation. The two backbones reproduce Marin's compute-optimal Grug MoE cells, and
the [240 TPP comparison](moe/PROTOCOL.md#240-tpp-comparison) trains them at 240 tokens per active baseline parameter excluding the embedding and output head.

Runs train on Dolma 3 HQ with the MuonH optimizer and are scored by **MoE-CORE**, the mean centered accuracy over 14
published tasks: 10 short ones plus RULER S-NIAH-1 at 2K and 4K, FDA and SWDE. Every result also reports validation
BPB and its throughput relative to the baseline on the same node, and every configuration is admitted within 1% of
its track anchor and audited within 2% over the run.

Choose MoE in the observatory for the [model leaderboard](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp) and [training curves and task scores](https://interactive-active-ai.github.io/NextTransformer/#results?track=d768-240tpp).

<!-- generated:moe-results -->
| 240 TPP leaderboard | Updates | Input tokens | Observatory views |
|---|---:|---:|---|
| [d768](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp) | 53,686 | 14.07B | [Model atlas](https://interactive-active-ai.github.io/NextTransformer/#atlas?track=d768-240tpp) · [Curves and evaluations](https://interactive-active-ai.github.io/NextTransformer/#results?track=d768-240tpp) |
| [d1024](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d1024-240tpp) | 64,973 | 34.06B | [Model atlas](https://interactive-active-ai.github.io/NextTransformer/#atlas?track=d1024-240tpp) · [Curves and evaluations](https://interactive-active-ai.github.io/NextTransformer/#results?track=d1024-240tpp) |
<!-- /generated:moe-results -->

[MoE overview](https://interactive-active-ai.github.io/NextTransformer/docs/moe/overview/) ·
[MoE protocol](https://interactive-active-ai.github.io/NextTransformer/docs/moe/protocol/) ·
[`moe/`](moe/README.md) · [submission flow](moe/README.md#submitting-a-method)

## Add your method

New methods and Arena entries follow the same submission process.

1. [Open a proposal issue](https://github.com/Interactive-Active-AI/NextTransformer/issues/new/choose) to discuss the method and run plan.
2. Implement the plugin on your fork, calibrate it to the shared budget, and complete a `--formal` run.
3. Once you have audited and exported the result, open **one pull request** to upstream `main` with the implementation and result. Link the proposal issue.

Start with the [plugin guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Auto-research arena

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/arena/overview/"><img src=".github/assets/readme/arena-en.webp" alt="Arena v2: agent-native research, end to end" width="100%"></a>

> [!NOTE]
> **Arena v2 is live.** Autonomous agent research is supported end to end: a [research gym](gym/README.md) with MCP
> tools, metered compute and research traces.

Any LLM, with or without a research framework, can already enter an architecture today: it is trained under the same
protocol, ranked on the same leaderboard, and disclosed with how it was found. See the
[arena](https://interactive-active-ai.github.io/NextTransformer/docs/arena/overview/).

## Built on NextTransformer

Community projects that extend NextTransformer.

<table>
<tr>
<td width="50%" valign="top">
<a href="https://github.com/Enderfga/NextTransformer-V"><img src="https://raw.githubusercontent.com/Enderfga/NextTransformer-V/8cd729800cea4ab23fdc5c083e8e2c68f2e654f7/docs/assets/banner.webp" alt="NextTransformer-V: a multimodal track for NextTransformer" width="100%"></a>
<h3><a href="https://github.com/Enderfga/NextTransformer-V">NextTransformer-V</a></h3>
<p><img src="https://img.shields.io/badge/Multimodal-1BAF7A?style=flat-square" alt="Multimodal"> by <a href="https://github.com/Enderfga">@Enderfga</a></p>
<p>Every NextTransformer method on frozen SigLIP2 image tokens interleaved with text at an 8,192-token context, compute-matched within 1%. Its first study, pre-registered, asks where hybrid backbones read the 2D layout of an image.</p>
</td>
<td width="50%" valign="top">
<a href="https://github.com/DAGroup-PKU/ResearchBoost-Agent"><img src="https://raw.githubusercontent.com/DAGroup-PKU/ResearchBoost-Agent/600043885d7eb3e9326725b47e966e273712dfe2/docs/assets/readme/banner-en.jpg" alt="ResearchBoost-Agent: AI that accelerates the research that advances AI" width="100%"></a>
<h3><a href="https://github.com/DAGroup-PKU/ResearchBoost-Agent">ResearchBoost-Agent</a></h3>
<p><img src="https://img.shields.io/badge/Training%20speedups-8C0000?style=flat-square" alt="Training speedups"> by <a href="https://github.com/AI4AI-for-Anything">AI4AI for Anything</a>, contributed by <a href="https://github.com/henry-y">@henry-y</a></p>
<p>An agent that profiles, optimizes and validates AI training code. It contributed NextTransformer's opt-in <code>--speedups</code> (<a href="https://github.com/Interactive-Active-AI/NextTransformer/pull/26">#26</a>): fused kernels for the shell every method shares. Its <code>exact</code> mode reproduces the default path bit for bit, formal runs accept it, and it trains <code>dense_full</code> 1.38× faster on one eight-GPU node.</p>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<h3>Your project here</h3>
<p>Built something on NextTransformer's code or protocol? Share it in <a href="https://github.com/Interactive-Active-AI/NextTransformer/discussions/categories/show-and-tell">Show and tell</a> or open a pull request that adds a card with a wide banner image.</p>
<sub>Listed projects are public and say in their README that they build on NextTransformer.</sub>
</td>
</tr>
</table>

## Roadmap

- **2K scaling:** start with the [T3 size sweep](https://interactive-active-ai.github.io/NextTransformer/#scaling), refine the fits, check preregistered forecasts, then hold out C200 for the final test.
- **MoE track:** the [240 TPP leaderboard](https://interactive-active-ai.github.io/NextTransformer/#leaderboard?track=d768-240tpp) updates with each published run.
- **Hybrid GDN + QSA:** completed formal results, cumulative audit and [run notes](https://interactive-active-ai.github.io/NextTransformer/docs/results/runs/qsa_hybrid/).
- **Planned:** checkpoints and the tokenizer on the Hugging Face Hub.

More on the [roadmap](https://interactive-active-ai.github.io/NextTransformer/docs/roadmap/).

## Community

- [Discord](https://discord.gg/k2XPXW6Jz) — chat with the community in #general (permanent invite)
- [GitHub Discussions](https://github.com/Interactive-Active-AI/NextTransformer/discussions) for questions and ideas
- [WeChat group](https://interactive-active-ai.github.io/NextTransformer/docs/community/#wechat)

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
Community projects are listed under [Built on NextTransformer](#built-on-nexttransformer).

## Citation

```bibtex
@software{nexttransformer2026,
  title   = {NextTransformer: An Open Framework and Fair Benchmark for Next-Generation Transformer Architectures},
  author  = {{NextTransformer contributors}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Interactive-Active-AI/NextTransformer}
}
```

## Acknowledgments

NextTransformer builds on [nanochat](https://github.com/karpathy/nanochat) and ideas from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), uses
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) kernels, and trains on
[ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle). We thank the authors of every method in the
zoo. Released under the [MIT License](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md).
