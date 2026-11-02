# Models

Every built-in architecture in NextTransformer lives here, behind one interface. Contributed methods are plugins
under [`nexttransformer/methods/`](../methods) ([plugin guide](../../docs/guide/PLUGINS.md)); they are registered and
built through the same interface.

```python
from nexttransformer.models import build_model, load_model, METHODS, FAMILY
from nexttransformer.models.config import candidate_config, debug_config

model = build_model(debug_config("dsa"))                       # a registered method at debug size
model, saved = load_model("RUN/checkpoint_00046875", "cuda")   # a trained checkpoint of either track
```

| Module | Contents |
|---|---|
| [`registry.py`](registry.py) | Registered methods and their families: baseline, looped, sparse, linear, residual, hybrid, others |
| [`config.py`](config.py) | `ModelConfig`, family sets, hybrid layouts, candidate and debug configurations |
| [`backbone.py`](backbone.py) | The nanochat d24 GPT backbone (byte-pinned upstream file) |
| [`lm.py`](lm.py) | `ReferenceLM`: assembles every registered method from the backbone and the family mixers |
| [`looped.py`](looped.py) | UT-ACT, Huginn, Ouro, and DeepLoop components |
| [`sparse.py`](sparse.py), [`sparse_kernels.py`](sparse_kernels.py) | DSA, QSA, NSA, SeerAttention-style, and MoBA mixers and their kernel path |
| [`linear.py`](linear.py) | DeltaNet, GLA, KDA, RetNet, ELU+1 variants, and GDN |
| [`others.py`](others.py) | Shared MLA-NoPE attention for H04 and its independent control; MHA-noVE keeps native attention |
| [`residual.py`](residual.py) | Hyper-Connections, mHC (Sinkhorn-Knopp), and Block AttnRes |
| [`native/`](native) | Historical native track: SSSL Dense, MoR, LoopFormer, Circle Loop (`NativeModelConfig`) |

Admit a `ModelConfig` with `nexttransformer.protocol.admission.admit_config` before formal use; see
[configs/](../../configs) and [docs/methods/CONFIGURATIONS.md](../../docs/methods/CONFIGURATIONS.md). Parameter names and
configuration fields are part of checkpoint compatibility: changing them breaks loading of trained runs. Adaptations
from each original method are listed in [docs/methods/SOURCES.md](../../docs/methods/SOURCES.md).
