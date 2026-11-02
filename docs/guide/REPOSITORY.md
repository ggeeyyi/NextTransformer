# Repository guide

This is the detailed directory and documentation map. [PLAN.md](../../PLAN.md) owns current project status;
this guide explains where maintained definitions, instructions and immutable evidence belong.

The 2026-10-24 reorganization gathered the code into one package, `nexttransformer`, and gave models, configurations,
published results, evidence, launchers, and the website one directory each. It changed module locations and source
fingerprints, not model mathematics: every configuration keeps its model and protocol hashes, the re-audited cost
proofs equal the previous ones in every count, and trained checkpoints load unchanged.

## Layout

| Path | Contents |
|---|---|
| `nexttransformer/models/` | Registry, `ModelConfig`, backbone, one module per family, the plugin interface, and the native track |
| `nexttransformer/methods/` | Contributed methods, one plugin package each ([plugin guide](PLUGINS.md)) |
| `nexttransformer/protocol/` | `Protocol`, configuration files, admission, the cost ledger, calibration |
| `nexttransformer/training/` | Optimizers (`optimizer.py`) and data-parallel optimizer sharding (`sharded.py`) |
| `nexttransformer/evaluation/` | Formal BPB/CORE (`formal.py`), native evaluators, routing metrics |
| `nexttransformer/runtime/` | Site settings, process groups and events, data streams, checkpoints, tracking |
| `nexttransformer/native/` | Trainer, preflight, and batch benchmark of the native track |
| `nexttransformer/checks/` | Bounded acceptance checks: fingerprints, distributed updates, FLA, integration |
| `nexttransformer/{train,evaluate,generate,prepare,export_run,arena}.py` | Command-line entry points |
| `configs/<family>/<method>.json` | Admitted configurations; `configs/native/` for the native track |
| `results/` | Published run records, model cards and Arena disclosure cards |
| `evidence/costs/`, `evidence/checks/` | The current cost proof set and acceptance checks |
| `launch/` | ModelArts (`modelarts.py`, `modelarts_native.py`), native shell launchers, Koala (`launch/koala/`) |
| `nanochat/` | Upstream runtime; six files are byte-pinned by the prepared-resource manifests |
| `nextt_gym/` | Research environment, submission/model contracts, agent tools and metering |
| `environments/eval/` | Gym runtime-image build and fixed-runtime scoring workflow |
| `adapters/harbor_eval/` | Optional Harbor controller for the same Gym scoring entry point |
| `website/` | The observatory, built from `results/` |
| `docs/` | `guide/`, `methods/`, `protocol/`, `plans/`, `records/`, `history/`, `upstream/` |
| `tests/` | Python regression tests; website tests live in `website/tests/` |

## Documentation responsibilities

Use one maintained source for each kind of information. Other entry points may summarize it and link here;
they should not maintain parallel status tables, parameter inventories or competing procedures.

| Source | Owns | Boundary |
|---|---|---|
| [PLAN.md](../../PLAN.md) | Current completion state, owners, evidence, next work and approved priorities | The live status index; dated records do not replace it |
| [docs/plans/](../plans/) | Design choices, dependencies, source analysis and acceptance criteria | Link to PLAN for progress; preserve registered scientific definitions |
| [PROTOCOL.md](../../PROTOCOL.md) | Binding scientific controls, cost thresholds, training and result acceptance | Documentation cleanup does not revise the protocol |
| [ARENA.md](../../ARENA.md) | Arena research/disclosure rules added to the scientific contract | Rule changes follow its versioning policy |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Method/result contribution workflow and required checks | Component/schema docs link here for submission steps |
| [AGENTS.md](../../AGENTS.md) and [Arena TASK](../arena/TASK.md) | Concise agent instructions and the self-contained research task | Summarize and link the binding rules; do not create independent thresholds |
| [CONFIGURATION.md](CONFIGURATION.md) | Site paths, caches, environment resolution, tracking and launchers | Machine settings remain separate from scientific configurations |
| [METHODS_FRAMEWORK.md](METHODS_FRAMEWORK.md) | Current model build, calibration, training, validation and resume workflow | Historical native workflows are explicitly identified |
| [WANDB_METRICS.md](WANDB_METRICS.md) | Tracking metric names and interpretation | Tracking is not the source of published website records |
| [SOURCES.md](../methods/SOURCES.md) | Source versions, attribution, retained mechanisms and native adaptations | Model names do not imply reproduction of a paper's full recipe |
| [CONFIGURATIONS.md](../methods/CONFIGURATIONS.md) | Registered structures, dimensions, stages and audited costs | Current proof-set selection is indexed in evidence/README |
| [COSTS.md](../protocol/COSTS.md) | Detailed accounting algorithms and calibration rationale | Shared thresholds remain in PROTOCOL; link current proofs rather than copy their status |
| [results/README.md](../../results/README.md) and [Arena card fields](../../results/arena/README.md) | Run/model record and disclosure-card schemas | Stored fields and semantics, not a second contribution workflow |
| [Models README](../../nexttransformer/models/README.md), [Gym README](../../nextt_gym/README.md), [Gym interface](../../nextt_gym/INTERFACE.md), [Gym model contract](../../nextt_gym/spec.md) | Component APIs and contracts | Keep model submission, tool interface and runtime setup distinct |
| [Website README](../../website/README.md) | Build/hosting commands, presentation and scoring/display semantics | Published data comes from results/; release status comes from PLAN |
| [EXPERIMENTS.md](../records/EXPERIMENTS.md), other [records](../records/), [history](../history/) and [evidence](../../evidence/README.md) | Dated observations, failures, manifests, hashes and immutable validation artifacts | Preserve original source/time boundaries; never rewrite them to match later progress |

## Earlier paths

| Before 2026-10-24 | Now |
|---|---|
| `models/backbone.py` (`nanochat/gpt.py`) | `nexttransformer/models/backbone.py` |
| `models/reference/model.py` (`reference/model.py`) | `nexttransformer/models/lm.py` |
| `models/reference/{looped,sparse,sparse_kernels,linear,residual}.py` | `nexttransformer/models/<same>.py` |
| `models/native/*` (`nanochat/native_experiment/<model modules>`) | `nexttransformer/models/native/*` |
| `reference/config.py` | `nexttransformer/models/config.py`; `Protocol`, `load_config`, `write_config` in `nexttransformer/protocol/spec.py` |
| `reference/protocol.py` | `nexttransformer/protocol/admission.py`; `evaluate_bpb`, `evaluate_core` in `nexttransformer/evaluation/formal.py` |
| `reference/{compute,calibrate}.py` | `nexttransformer/protocol/{compute,calibrate}.py` |
| `reference/optimizer.py` | `nexttransformer/training/optimizer.py` |
| `reference/train.py`, `scripts/reference_train.py` | `nexttransformer/train.py` (one trainer; `--sparse-backend`, Ouro gate-only stage) |
| `reference/{validate,distributed_check,fla_check,integration_check}.py` | `nexttransformer/checks/{validate,distributed,fla,integration}.py` |
| `reference/configs/<method>.json` | `configs/<family>/<method>.json` |
| `reference/results/` | `evidence/costs/` and `evidence/checks/`; superseded sets at tag `evidence-archive-20261024` |
| `reference/tests/`, `tests/` | `tests/` |
| `reference/koala/` | `launch/koala/` |
| `nanochat/experiment/{data,runtime,site,statistics,tracking,tracking_auth}.py` | `nexttransformer/runtime/{files,process,site,statistics,tracking,tracking_auth}.py` |
| `nanochat/experiment/{evaluation,optimizer}.py` | `nexttransformer/evaluation/metrics.py`, `nexttransformer/training/sharded.py` |
| `nanochat/native_experiment/{checkpoint,data,resources}.py` | `nexttransformer/runtime/{checkpoint,data,resources}.py` |
| `nanochat/native_experiment/{evaluation,optimizer}.py` | `nexttransformer/evaluation/native.py`, `nexttransformer/native/optimizer.py` |
| `scripts/native_{train,preflight,benchmark}.py` | `nexttransformer/native/{train,preflight,benchmark}.py` |
| `scripts/native_eval.py` | `nexttransformer/evaluate.py` (both tracks) |
| `scripts/native_prepare.py` | `nexttransformer/prepare.py` |
| `scripts/{base_eval,tok_train}.py` | `nanochat/{base_eval,tok_train}.py` (byte-identical upstream files) |
| `runs/train_reference_modelarts.py`, `runs/train_modelarts.py` | `launch/modelarts.py`, `launch/modelarts_native.py` |
| `runs/experiment_env.sh`, `runs/train_native_{common,a800,h200}.sh` | `launch/env.sh`, `launch/native_{train,a800,h200}.sh` |
| `website/data/{snapshot,xuyuan,linear-h20,methods}.json` | `results/runs/*.json`, `results/methods/*.json` |

Removed: the compatibility aliases, upstream chat/SFT/RL scripts and `tasks/`, upstream speedrun launchers, the
website's database submissions and tracking-service queries. All remain in Git history.

## Fingerprints and frozen runs

Admission compares each configuration's calibration with the `science_hash` of the model and protocol sources, and the
trainer records `science_hash`, `reference_source_hash`, and `native_source_sha256` in every segment. Moving or editing
those sources requires a new cost proof set for every configuration (`python -m nexttransformer.protocol.calibrate`).
A run resumes only on the source it started with: resume a run started before this reorganization from its frozen
checkout, and use `nexttransformer.models.load_model` to load its checkpoints for evaluation or generation.

## Tests and generated files

Python regressions live in `tests/` and website tests in `website/tests/`. Disposable checks go in ignored `tests/local/`
or `.agent-work/<task>/`. Curated cost proofs and acceptance evidence are versioned because admission and review depend
on them; never overwrite earlier evidence. Datasets, checkpoints, environments, credentials, and machine-local settings
stay outside Git.
