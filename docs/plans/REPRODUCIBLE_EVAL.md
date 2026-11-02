<a id="implementation-plan-fixed-gpu-runtime-for-isolated-gym-training-and-scoring"></a>
# Fixed GPU runtime: design and acceptance

This document defines the architecture and acceptance boundaries for the arena Gym runtime.
**[PLAN.md — Arena](../../PLAN.md#arena) owns P0–P6 progress, remaining work, owners and decisions.**
G3 and P6 refer to the same training/scoring isolation work. The binding scientific and entry rules
remain [PROTOCOL.md](../../PROTOCOL.md) and [ARENA.md](../../ARENA.md).

Ordinary training, evaluation and PR submissions, including Arena entries, do not require Docker.
They use the [configuration](../guide/CONFIGURATION.md) and
[training](../guide/METHODS_FRAMEWORK.md) guides. The fixed runtime serves Gym's separate training
and scoring roles; environment packaging does not create a new leaderboard submission route.

Use one operational entry point for each activity:

| Activity | Guide |
|---|---|
| Build and inspect the runtime image | [Build guide](../../environments/eval/README.md) |
| Capture inputs, select source, launch, validate and compare scoring | [Scoring workflow](../../environments/eval/SCORING.md) |
| Run the same scorer through optional Harbor orchestration | [Harbor adapter](../../adapters/harbor_eval/README.md) |
| Operate research tools and connect an agent | [Gym guide](../../nextt_gym/README.md) |

The former Dockerfile, shell and Harbor task templates in this document have been replaced by
these implemented guides. Historical versions remain in Git; do not mix their
`/opt/nextt-runtime` paths with the implementation's `/opt/venv` layout.

<a id="alignment-still-required-before-acceptance"></a>
### Alignment requirements

A release needs agreement between the accepted reference and the candidate runtime across source,
software, kernels, inputs, hardware and execution settings. The
[scoring alignment checklist](../../environments/eval/SCORING.md#outstanding-alignment-checklist)
defines the evidence to inspect. Implemented capture or comparison commands do not establish
acceptance without the corresponding run evidence.

### Decision: one fixed runtime image, separate training and scoring containers

Use one immutable runtime image, identified by digest, as the starting environment for both roles.
Start a fresh scoring container after training; never derive its environment from the modified
training container or mount the training virtualenv into it.

```text
Fixed GPU runtime image
  +-- Training: candidate code, train/dev inputs, checkpoint output
  +-- Fresh scorer: reviewed source, immutable checkpoint, reserved evaluation inputs
```

The roles have separate writable state, caches, commands and mounts. Reserved evaluation data
stays out of the common image and is mounted only to the scoring role. Agent/controller tools can
run outside the GPU runtime. If role-specific files later require different images, derive both
from the same pinned base instead of resolving two independent ML stacks, and record both digests.

Separate containers do not by themselves make arbitrary submitted Python safe. Model code must
execute inside the intended boundary during verification, training and scoring. Data access and
scorer integrity require explicit tests; API-level hiding of metrics is insufficient.

<a id="work-to-implement-in-order"></a>
### Dependency model

The consolidated plan tracks reference capture (P0), image build (P1), direct evaluation (P2),
paired validation (P3), distribution (P4), optional Harbor (P5), and isolation (P6/G3).
Direct execution separates runtime problems from orchestration problems. P6 needs the runtime
and direct path; it need not wait for optional Harbor work. A released scoring environment does
not establish training or resume equivalence.

<a id="1-what-already-exists-and-what-is-missing"></a>
## 1. Runtime and protocol boundaries

The runtime builds on the existing package lock, linear overlay, site configuration, resource
verification and evaluator. Preserve those contracts rather than changing algorithms, precision
or attention implementations for container convenience.

Standalone scoring evaluates a checkpoint. Gym runs architecture research through its research
trainer. Formal training additionally requires the complete schedule, all 189 validation events,
six full CORE events and cumulative cost acceptance. Gym's end-only CORE evaluation and a
standalone final checkpoint evaluation do not recreate that formal history.

The formal path remains `nexttransformer.train` followed by `nexttransformer.export_run`.
Any future direct admission of Gym results requires both scientific acceptance and a versioned
Arena policy decision, including who runs scoring and whether it replaces or agrees with the
run's own final evaluation. Environment tooling alone cannot grant that eligibility.

## 2. The environment contract: preserve what currently runs

Capture an identified reference run's source, interpreter distribution, complete software
inventory, GPU/driver, kernels, precision, backends and input identities. Resolve differences
against [uv.lock](../../uv.lock) and the hash-pinned
[linear overlay](../../requirements/linear.txt). A developer virtualenv is not automatically an
accepted reference, and a lock file alone does not identify every runtime artifact.

| Contract | Evidence |
|---|---|
| Interpreter and dependencies | Interpreter artifact hash; complete package inventory; root-lock and linear-overlay hashes; selected wheels |
| System runtime | Base platform digest, OS/compiler packages, CUDA runtime, cuDNN, NCCL and Triton |
| Source | Original training commit, reviewed scoring commit when different, source-file hashes and any source-layer digest |
| Inputs | Checkpoint completion and model identities; all required data contents; tokenizer/byte table; prepared manifests; CORE bundle |
| Execution | Per-phase backend, BF16/precision and compiler flags, rank/microbatch layout, data order and cache policy |

The source locks and build inventory own exact versions; do not maintain a second dependency
recipe in this document. A captured inventory such as `requirements/environment-daxiao.lock.txt`
is evidence, not a replacement lock. Controller dependencies belong outside the GPU image;
Harbor and vLLM are not evaluator dependencies.

Preserve phase-specific settings. Formal training switches sparse CORE evaluation to the
reference backend. Registered FLA execution fixes its dispatch policy. `sdpa-noflash` is distinct
from ordinary SDPA. Compare reference training with candidate training and reference scoring
with candidate scoring; sharing an image does not make these phases equivalent.

### Attention kernels are part of the environment

The [FA3 loader](../../nanochat/flash_attention.py) can download kernel repositories or fall back
to SDPA. Pinning the Python loader does not pin the downloaded revision or binary. Capture the
reference's actual repository revision and artifact hashes; distribute a verified snapshot and
seed a private cache. Scoring is offline and must reject unexpected backend fallback.

Do not choose SDPA, another FlashAttention package or different precision to bypass a missing
reference artifact. An empty kernel snapshot is appropriate only when SDPA is the declared
reference. The implemented scorer uses a separate read-only snapshot; its acquisition and hashes
must accompany any release that needs it.

Triton/Inductor caches are writable execution state. Key and isolate them by the accepted runtime,
GPU architecture, driver and backend. Resume must preserve the validated cache/autotuning policy.
An image digest alone does not imply bitwise equality across GPU types or reduction layouts.

### What the host still supplies

The container path requires Linux x86-64, an allocated NVIDIA GPU set, a compatible driver and
the NVIDIA container runtime. The host driver remains outside the image. Validate actual BF16,
intended kernels and NCCL on the selected layout; the CUDA version displayed by `nvidia-smi`
does not prove compatibility.

The implemented scorer targets registered models on one explicitly allocated eight-GPU node.
Formal multi-node layouts supported by the trainer are not automatically supported by this
wrapper. One-GPU informal smoke is not a formal evaluation. The release support matrix must name
the tested host class and measured memory, shared-memory, cache and runtime limits.

## 3. Downloads, mirrors, and offline operation

Build-time mirrors must deliver the same locked bytes. Retain platform/base-image digests,
interpreter artifact identity, wheel hashes, system-package inventory and the kernel snapshot.
Keep credentials in host configuration or build secrets, outside image layers and public
provenance. The [build guide](../../environments/eval/README.md) owns mirror variables and commands.

Consumers obtain a finished image by digest and verified inputs. They do not resolve packages,
install compilers, download kernels or retrain a tokenizer during scoring. Preserve prepared
resource symlinks or materialize their exact contents for container access. These evaluations
use NextTransformer checkpoints, not a downloaded base model.

Install the linear overlay after the final root environment sync; a later exact sync may remove
packages outside the root lock. Preserve the lock when a mirror lacks an artifact. For rebuild
reproducibility, record or fix the system-package snapshot as well as Python packages.
An offline image archive has its own hash in addition to the image identity.

<a id="4-planned-shared-runtime-dockerfile-and-scoring-build-procedure"></a>
## 4. Build identity and reviewed source

Use the [implemented Dockerfile](../../environments/eval/Dockerfile) and
[build guide](../../environments/eval/README.md). A release build uses a clean reviewed source
snapshot and records the exact packaged files, complete inventory and finished image digest.
A source-commit label alone does not prove that an image contains only committed files.

Keep original training source distinct from scoring source. A checkpoint may predate the scorer;
any newer scoring implementation needs a reviewed mapping and evidence that it preserves the
checkpoint's behavior. Never relabel the training commit or rewrite historical cost proofs.

The [source-layer tool](../../environments/eval/source_layer.py) allows reviewed model/configuration
changes on a fixed runtime without resolving dependencies. It checks protected runtime files and
locks against the base inventory, and records base and source-layer identities. Incompatible
lock, loader or runtime changes require a separately reviewed runtime release. Submitted Python
still requires review; the source-layer builder is not an arbitrary-code sandbox.

A training role also needs admitted configurations and their cost proofs. An evaluator image
that can load a checkpoint does not automatically satisfy training admission or resume.

<a id="5-planned-runtime-paths-and-direct-launch-script"></a>
## 5. Runtime paths and direct scoring

The implemented paths are:

| Path | Role |
|---|---|
| `/opt/nextt`, `/opt/venv`, `/opt/python` | Image-owned source, environment and interpreter |
| `/opt/nextt-manifest` | Source and runtime inventory |
| `/inputs/checkpoint` | Read-only complete selected checkpoint |
| `/inputs/resources`, `/inputs/data` | Verified resource bundle and required data |
| `/work` | Attempt-private writable configuration and caches |
| `/logs/verifier` | Retained results, diagnostics and validation evidence |

The [scoring guide](../../environments/eval/SCORING.md) owns launcher commands, site configuration,
the kernel-snapshot input and output details. Its input layout is for checkpoint scoring: the
evaluator needs training shards for final train BPB and inventory verification as well as reserved
evaluation data. Do not reuse these mounts unchanged for candidate training.

The scorer restricts checkpoint loading under the pinned PyTorch runtime and rejects unrecognized
objects without an unsafe fallback. That does not make tokenizer artifacts or model Python
untrusted-safe. Those inputs still need reviewed provenance.

For registered final BPB, each train/val split uses 20,971,520 tokens; full CORE includes all 22
tasks and all questions. A valid final result must be complete, finite and internally consistent.
A short smoke, partial CORE or a successful container exit is insufficient.

<a id="6-later-work-harbor-adapter-around-the-accepted-scoring-image"></a>
## 6. Optional Harbor orchestration

Harbor schedules the same scorer and retains its artifacts. It does not define a second
evaluation implementation or establish numerical equivalence. The
[adapter guide](../../adapters/harbor_eval/README.md) owns installation, generation and launch
commands for its pinned release.

### GPU access: automatic device requests, explicit host setup

The adapter targets Harbor 0.9.0. Its generator writes explicit Compose GPU device IDs because
that release does not provide the inspected newer upstream GPU-count translation. A task's GPU
count is not exclusive scheduling: the operator or scheduler must allocate the devices first.
The host still supplies the NVIDIA runtime. Cloud execution requires its own supported artifact
and resource adapter; local bind mounts do not transfer automatically.

### Task mounts and verification

The initial adapter uses the `nop` agent with reviewed checkpoints and source. It calls the
direct scorer's entry point and validator. Reward 1 means that evaluation completed and passed
validation; it is not a BPB or CORE quality score.

Harbor's verifier upload requires a writable container layer with the tested daemon; the direct
path uses a read-only root. Inputs stay read-only, and the adapter runs with the caller's non-root
UID and dropped capabilities. Treat that difference as part of acceptance. A CPU transport test
covers task upload, logs and reward handling, not GPU scoring or arbitrary-agent isolation.

A future research adapter must keep the controller, candidate training and independent verification
within their intended boundaries. Merely placing the existing Gym trainer in a container still
exposes its official data to submitted model code.

<a id="7-what-posttrainbench-actually-guarantees"></a>
## 7. Design precedent and limits

The original design review used
[PostTrainBench's separate agent/verifier architecture](https://github.com/aisa-group/PostTrainBench/blob/main/src/harbor_adapter/ARCHITECTURE.md)
as a precedent: transfer immutable model artifacts rather than copying a writable agent environment
into verification. NextTransformer starts from one pinned runtime because its training and scoring
share dependencies; role-specific derived images remain possible.

That precedent is not acceptance evidence for this repository. Upstream build recipes, version
pins and parity scripts do not establish installed artifact identity or runtime GPU equivalence
here. Record our finished digests, inputs and actual comparisons. A hash establishes identity
relative to a trusted manifest, not numerical correctness.

## 8. Acceptance: when can the gym use the image?

Acceptance is specific to the declared method/backend and supported host class:

1. **Clean build:** verify locked artifacts, complete inventories, packaged source and kernel inputs.
   Preserve the build recipe, logs and finished-image identity.
2. **GPU preflight:** exercise each allocated GPU, precision, actual kernels and distributed
   communication; reject fallback instead of silently changing the reference.
3. **Paired comparison:** use the same checkpoint, input rows, hardware and layout. Register
   tolerances before inspecting results; compare intermediate numerical behavior and complete final
   BPB/CORE outputs where required by the accepted scope.
4. **Complete scoring:** verify exact token counts, every CORE task, finite values, identities and
   aggregate consistency. Retain failures as well as successful attempts.
5. **Fresh consumer:** acquire the accepted image and inputs on a clean supported machine and run
   offline without undocumented caches or package resolution. Harbor adds its own parity check
   only when that optional path is included in the release.
6. **Training and isolation:** before claiming research execution support, verify full optimizer
   updates, accounting, data order and fresh-process resume, then test the training/scoring data
   and execution boundaries. Scoring acceptance alone does not establish these properties.

The release manifest identifies source, base/source-layer digests, interpreter and package
artifacts, system runtime, kernels, settings, inputs, supported hardware and the acceptance report.
Use [PLAN.md](../../PLAN.md#arena) for which gates are satisfied and what scope is approved.

Dated evidence remains separate from current status. The
[initial build record](../records/GPU_RUNTIME_BUILD.md) and
[tooling record](../records/SCORING_TOOLING.md) describe the early candidate, its removal by Harbor
cleanup and a stopped rebuild. A later image was built for the
[2026-10-28 A800 parity check](../../evidence/checks/scoring-dense-full-a800-20261028/README.md):
`dense_full` train/val BPB, CORE and all 22 task scores matched the reference and training-final
record exactly. It ran under udocker, so it does not establish Docker isolation, the direct Docker
launcher, Harbor GPU execution or other hardware/backend acceptance.
