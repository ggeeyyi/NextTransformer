# Fixed-runtime scoring workflow

This is the operational guide for the arena Gym's fixed-runtime scorer. Ordinary training,
evaluation and PR submissions do not require containers; their publication rules remain in
[CONTRIBUTING.md](../../CONTRIBUTING.md). This tool does not export leaderboard results.

[PLAN.md — Arena](../../PLAN.md#arena) owns implementation and acceptance status, release scope
and outstanding decisions. The [runtime design](../../docs/plans/REPRODUCIBLE_EVAL.md) defines
the boundaries and links the dated build/parity evidence. Check those references before treating
a local candidate as a supported release.

The implemented wrapper supports registered methods at step 46,875 on one explicitly allocated
eight-GPU node. Historical native-track results and multi-node scoring are outside this wrapper's
interface. Training and resume use their own entry points and acceptance.

<a id="outstanding-alignment-checklist"></a>
## Alignment checklist

Use this checklist when preparing or reviewing a scoring contract. It describes required evidence,
not a second progress tracker. The scripts enforce part of the contract; reviewed run evidence
must cover any settings they do not capture or check.

| Contract | Evidence to inspect |
|---|---|
| Reference identity | Accepted run and immutable checkpoint; original training source/interpreter/settings; reviewed reference manifest linked to the run evidence. The capture tool cannot certify acceptance by itself. |
| Software | Interpreter artifact, complete Python inventory, root lock plus linear overlay, CUDA/cuDNN/NCCL/Triton, compiler and system packages. Review every difference; do not edit a manifest to conceal a mismatch. Current runtime preflight requires exact inventory equality. |
| Training and scoring source | Keep the training commit distinct from reviewed scoring changes. The contract's scoring source must match the image inventory. Retain source mapping/diff and source-layer evidence when identities differ; preserve historical proofs. |
| Kernels and execution settings | Actual FA3 revision/artifacts, FLA dispatch, BPB and CORE sparse backends, precision/compiler flags, rank/microbatch, data order and cache policy. A pinned loader or package version does not establish these settings. |
| Evaluation inputs | Complete selected checkpoint; content hashes of all required data/resources, tokenizer/byte table and full CORE bundle; preserved or materialized symlinks. Restricted loading must work on the actual artifact without unsafe fallback. |
| Hardware and allocation | Named supported GPU/driver class, explicit device allocation, per-rank BF16/kernel checks and NCCL evidence; measured memory, shared-memory and cache limits for the workload. |
| Numerical behavior | Tolerances registered before comparison; paired intermediate numerical evidence where required by the accepted scope; complete BPB and 22-task CORE outputs. The comparison tool checks final metrics, not intermediate tensors. |
| Optional Harbor path | Same image/source, contract, inputs, devices, settings and validator as direct scoring. Validate its different writable-root behavior and retain a GPU comparison when this path is in release scope. |
| Rebuild and distribution | Clean build, finished image/platform digest, source-layer identity when used, input/kernel hashes and acquisition locations, system-package snapshot policy, and an offline consumer run on a clean supported machine. |

Scorer ownership and any future direct admission of Gym results are decisions for
[PLAN.md](../../PLAN.md#arena) and the versioned [Arena rules](../../ARENA.md), not settings an
operator can change in a scoring JSON.

## 1. Capture a real reference

Identify an accepted run, its source commit, interpreter, prepared resources, complete
checkpoint, data shards, attention backend, and sparse backend. Run these commands from
this checkout **using that reference interpreter**. They do not install dependencies.
Do not substitute a fresh candidate environment for the reference. The run-evidence file
is retained by hash as provenance; the tool cannot independently certify that the supplied
run was accepted. Review the generated manifest before using it.

For FA3, materialize the reference's existing kernel repository into a new directory:

```bash
"$REFERENCE_PYTHON" -m environments.eval.snapshot_kernels \
  --hub-cache "$REFERENCE_HF_HUB_CACHE" \
  --repository kernels-community/flash-attn3 --output "$KERNEL_SNAPSHOT"
```

Use `varunneal/flash-attention-3` for a reference that actually used that repository.
This copies only refs and snapshots, resolves blob symlinks, and excludes credentials and
unrelated model caches. Keep the snapshot read-only after capture. For a declared SDPA
reference, an empty snapshot directory is valid; do not choose SDPA to bypass missing FA3.

```bash
FLA_DISABLE_BACKEND_DISPATCH=1 "$REFERENCE_PYTHON" -m environments.eval.capture \
  --checkpoint "$CHECKPOINT" --resources "$RESOURCES" --data "$DATA" \
  --kernel-cache "$KERNEL_SNAPSHOT" --run-evidence "$RUN_EVIDENCE" \
  --source-commit "$REVIEWED_SOURCE_COMMIT" --attention "$EXPECTED_ATTENTION" \
  --sparse-bpb reference --micro-batch 4 --output "$REFERENCE_JSON"
```

`--attention` accepts `fa3`, `sdpa`, or `sdpa-noflash`. Preserve the reference's actual
BPB backend and microbatch. CORE uses the reference sparse path. Capture verifies prepared
resources, records every Parquet file's content hash (including training shards), all
resource/kernel files, checkpoint identity, all 22 CORE task names, Python packages,
CUDA/cuDNN, GPU/driver classes, BF16, and the selected backend. Hashing the full dataset can
take substantial I/O time. Inputs must not change during capture or scoring.

The kernel snapshot is currently a separate hash-verified, read-only input rather than
baked into the candidate image. A release must distribute it alongside the image or add
it to the accepted image; there are no runtime kernel downloads. Reference package
inventory differences fail preflight, including extra/missing packages. Resolve and record
such differences before creating an accepted scoring contract; do not edit the manifest
merely to suppress a failure.

## 2. Use a reviewed source and a fixed image

Build the candidate with the [mirror instructions](README.md). Its package inventory and
source hashes are under `/opt/nextt-manifest/`. The reference contract's source commit must
match the image inventory. A candidate built from uncommitted changes is useful for testing
but is not a clean release artifact.

Once a base runtime is published by digest, a reviewed model/configuration commit can be
layered onto it without resolving dependencies:

```bash
python3 -m environments.eval.source_layer --base-image "$BASE_IMAGE_DIGEST" \
  --commit "$REVIEWED_SOURCE_COMMIT" --tag nextt-scoring:source-candidate \
  --record "$NEW_SOURCE_BUILD_RECORD"
```

This tool checks protected runtime files, backbone, model loader, nanochat files, and locks
against the base inventory. Only model/configuration changes are allowed. Other changes
require a separately reviewed runtime release. It records both the base image identity
and the source-layer identity. Use only a base whose acceptance and release scope are recorded
in [PLAN.md](../../PLAN.md#arena).
The source-layer builder is not a sandbox for arbitrary submitted Python: accept reviewed source only.

## 3. Launch one isolated attempt

The host needs Docker, a validated NVIDIA driver/runtime, Python 3, and an explicit GPU
allocation. Copy [site.example.json](site.example.json) to an ignored local file and fill
all absolute paths. Pre-pull the released image through the approved registry mirror; the
launcher uses `--pull=never` and resolves the local image ID before starting. A local tag is
permitted for candidate tests and the exact resolved ID is saved in `launch.json`.

```bash
bash launch/eval_container.sh --config /absolute/scoring.local.json --dry-run
bash launch/eval_container.sh --config /absolute/scoring.local.json
```

The dry run validates local configuration and resolves the image, without allocating GPUs.
The live run uses eight explicit GPU indices/UUIDs, no network, a read-only filesystem and
inputs, the caller's UID/GID, private temporary caches, and a new retained output directory.
GPU allocation remains the operator/scheduler's responsibility. The launcher does not pick
apparently idle devices or reserve them against other jobs. The initial shared-memory and
cache limits are 16 GiB each; validate them for the target workload before release.

The container first verifies full input hashes, package/source inventory and the contract.
Every worker then checks its GPU/driver class, BF16, actual attention dispatch, a bounded
probe of the checkpoint's registered method/backend, and NCCL. It finally invokes the
existing evaluator with full train/val BPB and CORE. It preserves `sdpa-noflash` and separate
BPB/CORE sparse settings. Unexpected backend fallback fails the attempt.

The scorer sets `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`, so the pinned PyTorch runtime restricts
checkpoint loading even through legacy `weights_only=False` call sites. The preflight
inspector also explicitly uses `weights_only=True`. Unrecognized checkpoint objects are rejected,
not retried with unrestricted loading. The ordinary training/resume loader is unchanged.
Prepared tokenizer resources and submitted model source must still be trusted and reviewed.

Output includes `launch.json`, `inputs.json`, per-rank GPU reports, `evaluation.json`,
`validation.json`, logs, and status files. The validator requires the correct step/resource/
protocol identities, exactly 20,971,520 tokens in each BPB split, non-smoke BPB, all 22 CORE
tasks, finite values, and a consistent aggregate. Timeouts stop only this named container
and retain host diagnostics. Never reuse an output directory.

## 4. Compare and release

Before either run, commit a tolerance JSON with absolute tolerances for `bpb.train`,
`bpb.val`, `core_metric`, and `core.<task-name>` for each of the 22 tasks. Do not choose or
widen tolerances after seeing the results. Use the same checkpoint, data, GPU/driver,
rank layout, precision, and backend in reference and candidate runs.

```bash
python3 -m environments.eval.compare --reference "$REFERENCE_RESULT" \
  --candidate "$CANDIDATE_RESULT" --contract "$REFERENCE_JSON" \
  --tolerances "$PREREGISTERED_TOLERANCES" --output "$COMPARISON_REPORT"
```

This compares complete BPB and per-task CORE outputs; it does not yet compare paired
intermediate logits/loss tensors. A passing report is one part of acceptance, not automatic
release authorization. The [runtime acceptance criteria](../../docs/plans/REPRODUCIBLE_EVAL.md#8-acceptance-when-can-the-gym-use-the-image)
cover GPU evaluation, numerical checks, clean builds and a clean-machine consumer run. Record the
system-package snapshot policy and the limits of rebuild reproducibility with the release.
Publish the accepted image digest, source-layer identity if applicable, input/kernel hashes
and acquisition locations, supported host class, exact commands and acceptance report.
No registry publication is performed by these tools. Scorer ownership and its relationship to
training-final evaluation follow the decisions and versioning process linked above.

Harbor is an optional controller for the same entry point; see its [adapter instructions](../../adapters/harbor_eval/README.md).
