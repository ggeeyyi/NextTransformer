# Method plugins

A new method is a plugin: one directory under [`nexttransformer/methods/`](../../nexttransformer/methods) that
the registry discovers. It needs no change to shared files, so it cannot change another method. Its cost proof
fingerprints only its own package and the shared core ([COSTS.md](../protocol/COSTS.md)), so adding it never
invalidates another proof, and CI accepts the pull request when it adds only the plugin's own files.

## Layout

```text
nexttransformer/methods/<name>/
  method.json            declaration, read without importing code
  __init__.py            hooks (any other modules of the method may sit next to it)
configs/<family>/<name>.json                       calibrated configuration, written by calibrate
evidence/costs/<proof-set>/<name>.json, dense.json  cost proof and the SSSL anchor proof, written by calibrate
results/methods/<name>.json                        model card
tests/test_<name>.py                               tests
```

## Declaration

```json
{
 "name": "output_gate",
 "family": "others",
 "title": "Attention with a learned output gate",
 "paper": {"title": "<paper title>", "url": "<paper URL>"},
 "options": {"gate_bias": 0.0},
 "debug_options": {"gate_bias": 0.5},
 "shell": {"layers": 24, "mlp_width": 6144, "window_pattern": "L"},
 "layer_multiple": 1
}
```

| Field | Meaning |
|---|---|
| `name` | Lowercase letters, digits and `_`; equals the directory name; not a built-in method |
| `family` | Display family: `looped`, `sparse`, `linear`, `residual`, `hybrid` or `others` |
| `options` | The method's own parameters and their defaults; read with `config.option(name)` |
| `debug_options` | Values for the small CPU configuration of tests (`debug_config`) |
| `shell` | Starting `layers`, `mlp_width` and `window_pattern` of the native shell (default 24, 6144, `L`) |
| `layer_multiple` | Size of a repeating unit; calibration and validation keep the layer count a multiple of it |

## Hooks

Define one or more in `__init__.py`:

| Hook | Use |
|---|---|
| `mixer(config, layer)` | Returns the attention module of a layer, called as `(x, ve, cos_sin, window, kv_cache)` |
| `ffn(config, layer)` | Returns the feed-forward module of a layer, called as `(x)` |
| `block(block, config, layer)` | Wraps or replaces a block, called as `(x, ve, cos_sin, window, kv_cache)` |
| `model` | A subclass of `nexttransformer.models.lm.ReferenceLM`, for a method that changes the forward flow; override, for example, `_ordinary_hidden` or `set_training_stage` |
| `frozen` | Names of shell parameters the method leaves unused, such as `("backout_lambda",)` |
| `init(model)` | Initializes the method's own parameters after the shared initialization |
| `stages(config)` | Contiguous training stages in order, `[(name, updates), ...]` summing to 46,875; default one `joint` stage |
| `check(config)` | Raises `ValueError` for invalid option values |

The shared shell is fixed, and a plugin keeps to its contract:

- Every block keeps `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` as linear
  layers; the shell initializes them, and the projections into the residual stream start at zero.
- Every trainable parameter receives a gradient at every update; list unused shell parameters in `frozen`.
- A module may set a tensor `last_aux` (an auxiliary training loss added in training) and define
  `set_step(step)`, called with the optimizer step.
- Parameters keep stable names once a run exists: checkpoints store tensors by name.

A plugin cannot change a built-in or another plugin method, the cost ledger, the trainer, the optimizer, the data
or the evaluation, and it cannot add a display family. Those are maintainer changes, and a change to the shared
core requires a new audit of every configuration.

## Example

This `output_gate` plugin multiplies each layer's attention output by a learned sigmoid gate. It only illustrates
the hooks; a published method should follow its paper's exact formulation and document any adaptation in
[SOURCES.md](../methods/SOURCES.md).

```python
"""Causal attention whose output passes a learned sigmoid gate."""
import torch
from torch import nn

from nexttransformer.models.backbone import CausalSelfAttention, Linear


class GatedAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.gate = Linear(config.width, config.width, bias=False)
        self.bias = config.option("gate_bias")

    def forward(self, x, ve, cos_sin, window, kv_cache):
        return super().forward(x, ve, cos_sin, window, kv_cache) * torch.sigmoid(self.gate(x) + self.bias)


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    for block in model.transformer.h:
        nn.init.zeros_(block.attn.gate.weight)
```

The test plugins in [`tests/plugins/`](../../tests/plugins) cover a mixer, a feed-forward network with an option
check, and a whole-model plugin with two training stages and a frozen shell parameter.

## Cost

Calibrate the method to the shared budget, within 1% of the SSSL anchor:

```bash
python -m nexttransformer.protocol.calibrate --methods <name> --cache .agent-work/<task>/cost-cache \
    --output configs --results evidence/costs/<name>-<date>
```

Calibration searches the registered v2 grid: 16 to 32 layers in steps of `layer_multiple`, then FFN widths from
3d to 5d in multiples of 128. It writes the configuration and the proof, and the SSSL anchor proof next to it.

- Compute softmax attention with `nanochat.flash_attention.flash_attn.flash_attn_func`; the ledger counts its
  causal or windowed edges exactly.
- Express everything else with operators the ledger counts; the audit names any uncovered operator and refuses.
- A custom kernel needs a pure PyTorch path that the CPU audit runs, and a test that both compute the same function.

The proof fingerprints the plugin's package and the shared core. Formal results need the audited PyTorch build
(2.9.1+cu128); research mode trains with any build ([README](../../README.md#quick-start)).

## Before the pull request

```bash
python -m pytest tests/test_<name>.py -q && python -m pytest -q
python -m nexttransformer.protocol.fingerprint        # every published method still matches its proof
python -m nexttransformer.arena --scope origin/main   # the pull request adds only the method's own files
python -m nexttransformer.export_run --check
python -m nexttransformer.train --config configs/<family>/<name>.json --output RUN --micro-batch N --smoke-steps 2
```

A maintainer re-runs the method's audit in the audited environment and compares it with the submitted proof.
After the merge, the formal run uses `nexttransformer.train --formal`, and its result arrives in a separate pull
request ([CONTRIBUTING.md](../../CONTRIBUTING.md#submitting-a-run)).

To try plugins kept outside the repository, list their roots in `NEXTT_METHOD_PATHS`. Fingerprints cover only plugin
roots inside the repository, so such methods are for research.
