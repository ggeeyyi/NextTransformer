# First scoring-image parity check: `dense_full` on A800 — 2026-10-28

The [scoring image](../../../environments/eval/README.md) scored the final checkpoint of the accepted
full-attention Dense baseline (B01 `dense_full`, [record 038](../../../docs/records/EXPERIMENTS.md#record-038))
and reproduced the host reference and the training-final record **bit for bit**: train BPB, val BPB, CORE and
all 22 task scores are identical ([record 042](../../../docs/records/EXPERIMENTS.md#record-042)). This covers P0
(reference capture) and a first P3 comparison of the [plan](../../../docs/plans/REPRODUCIBLE_EVAL.md) for one
registered method on one hardware class. The image ran without a Docker daemon (see *Execution*), so the
direct Docker launcher and the Harbor path were not exercised.

| Metric | Training-final record | Host reference | Scoring image |
|---|---:|---:|---:|
| Train BPB (20,971,520 tokens) | 0.6665835136673783 | 0.6665835136673783 | 0.6665835136673783 |
| Val BPB (20,971,520 tokens) | 0.6644601775656058 | 0.6644601775656058 | 0.6644601775656058 |
| CORE (22 tasks) | 0.33095152996093863 | 0.33095152996093863 | 0.33095152996093863 |

The training-final values are the run's own step-46,875 evaluation, archived in
[`a800-dense-full-formal.json`](../a800-dense-full-formal.json); the per-task scores were compared against the
run directory's `core_final.json` (22 of 22 identical, raw and centered).

## Files

| File | Contents |
|---|---|
| [contract/reference.json](contract/reference.json) | Reference captured by `environments.eval.capture`: interpreter inventory, GPU/driver class, backend, checkpoint, resource and all 6,543 data-shard hashes |
| [contract/tolerances.json](contract/tolerances.json) | Absolute tolerances, registered at 2026-10-28 07:36:18 UTC before either evaluation: BPB ±1e-4, CORE ±1e-3, each task ±5e-3 |
| [reference/evaluation.json](reference/evaluation.json) | Host reference evaluation |
| [candidate/](candidate) | Scoring-image outputs from `/logs/verifier`: evaluation, validation, per-rank GPU preflight, status |
| [image/build.json](image/build.json) | The image's `/opt/nextt-manifest/build.json`; its SHA256 `3c222014…` equals the inventory hash the in-container preflight recorded |
| [comparison.json](comparison.json) | `environments.eval.compare` on the files above: passed, largest absolute difference 0.0 |

The only edit to any output is the `checkpoint` field of `reference/evaluation.json`, where the host path is
replaced by `RUNS/…`; `comparison.json` was computed on the committed files.

## What was compared

- **Checkpoint:** `checkpoint_00046875` of `nextt-v2-dense-20261023_dense_full_a800_seed42`, trained at `4dd6ebb`
  on 1×8 A800-SXM4-80GB with SDPA attention and microbatch 8. Its `model.pt` and `complete.json` hashes are in the contract.
- **Scoring source:** `640ceb5`, the PR #8 head `9259800` plus a scorer fix. Trainer checkpoints store
  `torch.__version__` (`torch.torch_version.TorchVersion`), and the scorer's forced weights-only loading rejected
  every real checkpoint, this one included. The fix allows only that `str` subclass in the preflight inspector and
  the evaluator. The science hash is unchanged (`a2bff1f3…`). The training commit and the scoring commit therefore
  differ, and the identical results show that this scoring source evaluates the `4dd6ebb` checkpoint unchanged.
- **Inputs:** the pinned ClimbMix directory (6,543 shards, inventory checked by `verify_resources`) and the
  prepared resources, copied out of their symlinked layout so they resolve inside a container; the contract
  records their content hashes. The FA3 snapshot is empty: A800 has no FA3, and `attention` is `sdpa`.
- **Settings:** BF16, eight ranks, microbatch 8 (the run's final-evaluation batch), reference sparse backends,
  full BPB and all 22 CORE tasks.

## Reference interpreter

The run's training environment cannot serve as the reference, because the preflight requires the image's
package inventory to equal the reference exactly. Every package the two share has the same version (including
Python 3.10.19, PyTorch 2.9.1+cu128, Triton 3.5.1, cuDNN 9.10.2, NCCL 2.27.5 and the CUDA wheels), but the
training environment adds `pytest`, `iniconfig`, `pluggy` and an installed copy of the project
(`nextt-nanochat` 0.2.0), and lacks the linear overlay (`fla-core`, `einops`).
The reference was therefore built on the same host from the image's recipe: uv 0.9.5, `uv export --frozen
--no-dev --extra gpu --extra upstream --extra tracking`, `uv pip sync --require-hashes`, then
`requirements/linear.txt`. Its `pip freeze` (107 packages) is byte-identical to the image's
`environment.complete.txt` (SHA256 `4f65233c…`), and its interpreter binary has the image's SHA256
(`365c8c3f…`). The training-final record bridges to the training environment: it matches as well.

## Image

Built on 2026-10-28 with `environments/eval/build.sh` from a clean checkout of `640ceb5` (buildx v0.38.1,
Docker 28.4.0, BuildKit docker driver, no GPU). `APT_MIRROR` was overridden to `https://repo.huaweicloud.com/ubuntu`:
the default Tsinghua mirror answered the builder's apt requests with HTTP 403. All other inputs are the
Dockerfile defaults. Image ID `sha256:a1cffa18fd2a68bb855ddc91ac0ef66ba752533677726872e72796831c467409`;
the transferred `docker save | gzip` archive has SHA256 `1075533fb8ccfef5e583b5d08a34c8096a6a39140bd0570648ea8777ad040477`.
Offline checks before transfer: `uv pip check` (107 compatible), imports of torch/triton/fla/evaluator without a GPU,
evaluator help. Apt packages remain unpinned (see the plan). The image is not published.

## Execution

The GPU host is a ModelArts notebook, itself a Kubernetes container without a Docker daemon, user namespaces or
`ptrace`. The image ran with [udocker](https://github.com/indigo-dc/udocker) 1.3.17 in fakechroot mode F3, with the
host NVIDIA driver libraries injected by `udocker setup --nvidia`. The command mirrors
`environments/eval/launch.py`: the same five inputs under `/inputs`, the `/logs/verifier`, `/work` and `/tmp`
mounts, `HOME=/work`, `PYTHONDONTWRITEBYTECODE=1`, working directory `/opt/nextt`, and entry point
`/opt/venv/bin/python -m environments.eval.run`. All eight GPUs of the node were allocated to this attempt.

`run.py` completed every stage inside the image: the input preflight (all contract hashes, image source files,
runtime inventory, image source commit), the eight-rank GPU preflight (device/driver class, BF16, SDPA dispatch, a
bounded `dense_full` probe, NCCL all-reduce), the evaluator, and the validator. Wall time: 11 min input preflight,
8 min GPU preflight and evaluation.

Not covered by this execution:

- Docker isolation: udocker does not enforce `--network none`, `--read-only`, `--cap-drop=ALL`, tmpfs limits or a
  GPU device subset. Mode F1 resolves the venv's interpreter symlink and loses the virtual environment, so F3 was
  used; F3 rewrites ELF loader paths in udocker's extracted copy (machine code unchanged).
- `launch.py`, the NVIDIA Container Toolkit, and Harbor's GPU trial.
- FA3 on Hopper, sparse kernels, linear (FLA) methods, multi-node scoring, a second clean machine, registry
  publication, and the official-scoring policy decisions in [SCORING.md](../../../environments/eval/SCORING.md).
