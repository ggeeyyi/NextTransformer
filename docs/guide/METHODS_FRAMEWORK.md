# Training, calibration, and validation

This guide owns the current build, admission, training and validation workflow. Machine paths, cache settings, tracking and launcher selection are maintained in [configuration](CONFIGURATION.md). The package provides scientific configurations, cost audits, protocol admission, optimizers, and trainers. Model implementations live in [`models/`](../../nexttransformer/models/README.md). Start with the [project overview](../../README.md), [protocol](../../PROTOCOL.md), [status](../../PLAN.md), and [v2 roadmap](../plans/V2.md).

| Guide | Purpose |
|---|---|
| [Method plugins](PLUGINS.md) | Adding a method without editing shared files: declaration, hooks, cost, checks |
| [Configurations](../methods/CONFIGURATIONS.md) | Registered dimensions, parameters, stages, and cost deviations |
| [Sources](../methods/SOURCES.md) | Original mechanisms, pinned sources, and explicit native adaptations |
| [Costs](../protocol/COSTS.md) | Operator policy, stage weighting, calibration, and admission |
| [Experiment history](../records/EXPERIMENTS.md) | Dated checks, failures, deployment observations, and limitations |
| [Evidence index](../../evidence/README.md) | Current proofs and immutable historical checks |

## Environment and resources

Use the root `pyproject.toml` and `uv.lock`. CPU development needs no FLA package. CUDA training uses the locked GPU environment; linear CUDA backends additionally require the hash-pinned dependencies:

```bash
uv sync --frozen --extra gpu --extra upstream --extra tracking
source .venv/bin/activate
uv pip install --no-deps --require-hashes -r requirements/linear.txt
```

Prepare a local site file as described in [configuration](CONFIGURATION.md). Data, tokenizer, byte table, CORE resources, cache, environment, and output paths are supplied through CLI/site settings. Reuse the same verified resources across methods. Never store credentials or private paths in scientific JSON.

CUDA runs using `fla_chunk64` set `NANOCHAT_DTYPE=bfloat16` and `FLA_DISABLE_BACKEND_DISPATCH=1`.
Match the proof's PyTorch build and the hash-pinned linear dependencies; the optimizer compiler also needs
`setuptools`. Verify training-shard contents and tokenizer/byte-table hashes against the shared baseline;
retraining a tokenizer with the same settings does not establish identity. Preserve frozen environments.

## Build a model

For formal use, admit the configuration before constructing the model:

```python
from nexttransformer.protocol.admission import admit_config
from nexttransformer.models import build_model
from nexttransformer.training.optimizer import build_optimizer

config, proof = admit_config("configs/linear/kda.json")
model = build_model(config, "cuda")
model.set_training_stage(0)  # Determine trainable parameters before optimizer/DDP.
optimizer = build_optimizer(model)
```

Admission rejects stale source/config/protocol fingerprints, a different PyTorch version, unknown operators, incomplete stages, and excessive cost error. It is a preflight, not a final-run verdict. The checked-in CUDA proofs are not admission evidence for a different CPU build.

`model(x, y)` returns the scalar training objective. `model(x)` returns float32 logits. `model(x, y, loss_reduction="none")` returns plain token CE without auxiliary loss; BPB uses this interface. `model.eval()` selects the registered evaluation-depth/exit policy. `last_lm_loss`, `last_aux_loss`, `last_stats`, and `last_trace` expose separate diagnostics.

Set the optimizer step with `set_step(step)` and each microbatch's first global row with `set_row_offset(row_id)`. Huginn's keyed randomness depends on these values. Use the shared `BestFitStream` and preserve its complete buffer/cursor on resume. Ouro's final gate-only stage requires `transition_stage`, preserved gate moments, and rebuilt DDP.

## Train

| Entry point | Supported use |
|---|---|
| `python -m nexttransformer.train` | Every registered method on the registered 8/16/32-rank shared-stream layouts: FLA backends, `--sparse-backend kernel` for sparse methods (CORE always uses the reference path), Ouro's gate-only stage, segmented resume |
| `python -m nexttransformer.native.train` (`launch/native_*.sh`) | The historical native track: SSSL, MoR, LoopFormer, Circle, with matrix-cost reporting |

The trainer refuses a prepared tokenizer that differs from the pinned one. By default it trains in research mode:
stale cost evidence (changed source, another PyTorch build) or a cost outside the budget is printed and recorded, and
the run cannot be exported as a result. `--formal` enforces cost admission and is required for results; a run keeps
its mode across segments. On ModelArts, `launch/modelarts.py` starts a formal `nexttransformer.train` with one admitted
configuration per node, each an independent eight-GPU run; see [configuration](CONFIGURATION.md#reference-modelarts-launcher).
A small bounded run on prepared resources:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m nexttransformer.train \
  --config configs/linear/gdn.json --site-config site.native.local.json \
  --output /persistent/runs/gdn-smoke --micro-batch 8 --smoke-steps 2

python -m torch.distributed.run --standalone --nproc_per_node=8 -m nexttransformer.train \
  --config configs/sparse/qsa.json --site-config site.native.local.json --sparse-backend kernel \
  --output /persistent/runs/qsa-smoke --micro-batch 8 --smoke-steps 2
```

Smoke outputs cannot become formal results. Formal runs require completed preflight/pilot gates and their own fresh
output directory. `--resume auto` continues complete checkpoints and checks source/software (including the sparse
backend), config, RNG, stage, and budget continuity. `--max-wall-hours` ends a segment at a checkpoint boundary.
Resume a run from the source it started with; evaluate or sample any checkpoint with `nexttransformer.evaluate` and
`nexttransformer.generate`.

All six registered hybrids use the same entry point with their files under `configs/hybrid/`. For a
bounded check that targets deterministic resume, pass `--deterministic-attention`: this rejects FA3 and disables SDPA flash. If FA3 is cached,
select fresh task-local `hf_home` and `hf_hub_cache` paths in the site file with `HF_HUB_OFFLINE=1`, and verify
the resulting SDPA backend. Keep the chosen backend throughout the run and preserve shared caches. This
option alone does not establish a bitwise guarantee; see the evidence and limits below.

## Recovery guarantees

Recovery has separate claims: restoration of training state, exact data/counter continuity, and numerical
agreement with uninterrupted execution. A checkpoint loading successfully establishes neither the last
claim nor formal completion. The evidence below is specific to its source, software, kernels, rank layout,
optimizer implementation and checkpoint boundary; it is not a universal guarantee for all configurations.

| Path / evidence | Observed guarantee | Conditions and remaining limit |
|---|---|---|
| GDN bounded 32-rank H20 check, [record 031](../records/EXPERIMENTS.md#record-031) | Bitwise losses, gradient norms and final weights across eight one-update resumed segments; exact stream/counter state | Fixed optimizer kernel configurations, DDP bucket warm-up and restored per-rank caches in the recorded environment. Agreement across rank counts is numerical, not bitwise. |
| Earlier KDA v2, RetNet and `elu_qknorm` formal segments, [records 031](../records/EXPERIMENTS.md#record-031) and [037](../records/EXPERIMENTS.md#record-037) | State restored; continuations are rounding-level consistent | Per-process optimizer benchmark choices could change across segments. Do not retrospectively label these runs bitwise. |
| H04 real-data H100 check, [record 045](../records/EXPERIMENTS.md#record-045) | Twelve continuous updates equal one update plus fresh-process resume, including weights, losses and all eight ranks' optimizer/RNG/data state | Deterministic SDPA-noflash/FLA path, deterministic PyTorch algorithms and recorded cuBLAS settings. The separate FA3 throughput smoke has no such resume claim. No 16/32-rank hardware guarantee. |
| H05/H06 real-data H100 smoke, [record 046](../records/EXPERIMENTS.md#record-046) | Two updates equal one plus fresh-process resume on four ranks; counters and next-row hashes exact | Four-stream smoke only; H05 stays in burn-in. Registered eight-stream acceptance and the actual 512-update transition remain pending. |
| HC/mHC actual A800 continuations, [record 051](../records/EXPERIMENTS.md#record-051) | Actual continuation is not bitwise; mHC bounded replay has disclosed numerical agreement | HC's first logged resumed loss differs by 0.0013. mHC has a registered source transition and separate proofs; it cannot be treated as an unchanged-source replay. |
| Gym research path | Resume support exists in its trainer; real-data end-to-end recovery acceptance is pending | See [Gym operations](../../nextt_gym/README.md) and [AR-01/Q-01](../../PLAN.md#approved-roadmap-refinements). Implementation and synthetic smoke do not establish formal-trainer equivalence or production recovery. |

For each new claimed recovery path, record the exact commit/configuration/proof, tokenizer/data identity,
PyTorch/CUDA/NCCL and kernel selection, GPU/rank/microbatch layout, compiler/cache policy, optimizer and
stage. Compare a bounded uninterrupted trajectory with save/exit/fresh-process continuation: restored
weights and optimizer, RNG and packing state, next-row identities, losses/gradients, final parameters and
token/FLOPs counters. Register tolerances before a numerical comparison; use an explicit result label
such as **bitwise within tested scope**, **numerically close within stated tolerance**, or **unverified**.
Failures and source transitions remain in the record. Targeted missing checks are tracked in
[Q-01](../../PLAN.md#approved-roadmap-refinements); no blanket rerun of historical experiments is required.

## Re-audit selected configurations

Run from the repository root in the target PyTorch environment. Use a new output/proof directory for each audit:

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 \
python -m nexttransformer.protocol.calibrate --workers 4 \
  --reuse-configs configs \
  --cache .agent-work/cost-audit/cache \
  --output .agent-work/cost-audit/configs \
  --results .agent-work/cost-audit/proofs
```

This executes shape/autograd accounting on CPU without data or GPU training. `--reuse-configs` holds existing structures fixed; omitting it invokes the registered legacy candidate search. Review new proofs before promoting configurations and curated evidence. Source, algorithm, software, or backend changes require re-audit. Do not overwrite historical proof directories.

## Required validation

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 \
  python -m pytest -q -m 'not slow'

# Bounded full-size BF16 architecture checks; ranks test disjoint models.
NANOCHAT_DTYPE=bfloat16 python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m nexttransformer.checks.validate --mode target --config-dir configs \
  --output .agent-work/architecture-check/full-context.json
```

The second command is not a synchronized global optimizer update. Use `nexttransformer.checks.distributed` for that separate check and `nexttransformer.checks.integration` for resource/data/evaluator integration; inspect each CLI's `--help`. `nexttransformer.checks.fla` compares actual linear backends with numerical oracles, including captured checkpoint inputs. It performs no optimizer update.

For hybrid deployment, [the bounded-check launcher](../../launch/koala/bounded_check.sh) defines the real-data
continuous-versus-resumed 12-update check; adapt its site-specific paths before using another cluster. Use
41,943,040 validation tokens per event, compare losses and weights bitwise on the registered deterministic
path, and verify stream states and counters exactly. Follow the [hybrid acceptance sequence](../plans/V2-linear-hybrids.md#7-validation-and-admission-per-row)
for the separate 512-update pilot and any intended 16/32-rank checks; pilot weights never seed formal runs.

Keep reusable algorithm, gradient, protocol, cost, optimizer, and resume regressions tracked. Local probes and raw output belong in `.agent-work/` or ignored local-test directories. Curated proofs and bounded evidence remain versioned. No random-initialization score from these checks is a model-quality result.
