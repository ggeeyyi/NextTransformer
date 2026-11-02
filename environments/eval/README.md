# GPU runtime candidate

This is the build guide for the arena Gym runtime. The [runtime design](../../docs/plans/REPRODUCIBLE_EVAL.md)
defines its boundaries; [PLAN.md](../../PLAN.md#arena) owns acceptance state and release scope.
Ordinary training, evaluation and PR submissions do not require this image.
The image packages the repository's existing GPU lock and linear overlay. It excludes datasets,
checkpoints and a reference FA3 cache. A successful build does not establish numerical parity
for a method or hardware class.
The default command prints evaluator help; it does not start evaluation or training.

## Build through mirrors

From the repository root:

```bash
bash environments/eval/build.sh
```

The default local tag is `nextt-runtime:cu128-candidate`. Set `NEXTT_IMAGE_TAG` to change it and
`NEXTT_BUILDER` to select an existing BuildKit builder. The build loads the image into local Docker;
it never pushes to a registry. It needs no GPU, but requires sufficient disk for CUDA layers, package
downloads, the installed environment, and the final image.

| Input | Default source |
|---|---|
| CUDA 12.8.1 devel / Ubuntu 22.04, linux/amd64 | DaoCloud Docker Hub mirror, pinned platform digest in Dockerfile |
| uv 0.9.5, linux/amd64 | DaoCloud GHCR mirror, pinned platform digest in Dockerfile |
| CPython 3.10.19 standalone distribution | npmmirror Python standalone mirror |
| Ubuntu packages | Tsinghua Ubuntu mirror |
| Python packages, including Torch, NVIDIA libraries, and FLA | Huawei Cloud PyPI mirror |

The build exports `uv.lock` with `--frozen`, selecting `gpu`, `upstream`, and `tracking`, and installs
the exported requirements with `--require-hashes`. It then installs `requirements/linear.txt` with its
existing hashes. It does not re-lock or replace dependency versions, and it refuses source distributions
instead of silently downloading unpinned build dependencies. Project modules are imported from
`/opt/nextt` through `PYTHONPATH`; no separate project wheel build is necessary.

Mirrors can be overridden with `CUDA_IMAGE`, `UV_IMAGE`, `APT_MIRROR`, `PYPI_INDEX_URL`, and
`PYTHON_MIRROR`. Use immutable digest references for the two images. Use public or unauthenticated
internal mirror URLs for these build arguments: they are recorded in build provenance. Do not pass
credentials through build arguments. A mirror may redirect artifact bytes to its CDN; inspect build
logs when enforcing an organization's network allowlist. There is no configured upstream-index fallback.

The Tsinghua PyPI mirror is an alternative via `PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.
BuildKit keeps a dedicated uv download cache outside the final image so failed builds can reuse
completed downloads. Package hashes are still checked when installing from the cache.

If a mirror lacks a locked wheel, fix the mirror or stage the same hash-verified artifact. Do not change
the scientific lock to make a build pass. Apt packages are recorded but their repository snapshot is not
yet fixed; rebuilding at a later date may produce a different image. After acceptance, distribute the
finished image by digest rather than asking every user to rebuild it.

## Dedicated builder when the Docker daemon's proxy is broken

A shared daemon may have a stale proxy even when mirror hosts are reachable directly. Do not restart
or reconfigure that daemon to build this image. An operator can create an isolated builder using an
already available local BuildKit image:

```bash
docker buildx create --name nextt-env-builder --driver docker-container \
  --driver-opt image=moby/buildkit:buildx-stable-1 --driver-opt network=host \
  --driver-opt env.HTTP_PROXY= --driver-opt env.HTTPS_PROXY= --driver-opt 'env.NO_PROXY=*'
docker buildx inspect nextt-env-builder --bootstrap
NEXTT_BUILDER=nextt-env-builder bash environments/eval/build.sh
```

This example assumes the BuildKit image is already cached locally and direct mirror access works.
The builder is site infrastructure, not part of the released runtime; record its version with the build
log. It has its own build cache. Do not prune shared images or other builders to make space.

## Inspect without GPU allocation

After a successful build:

```bash
docker run --rm --network none nextt-runtime:cu128-candidate
docker run --rm --network none nextt-runtime:cu128-candidate \
  uv pip check --python /opt/venv/bin/python
docker run --rm --network none nextt-runtime:cu128-candidate \
  cat /opt/nextt-manifest/build.json
```

`/opt/nextt-manifest/` contains the exported hashed requirements, complete Python inventory, system
packages, source-file hashes, source base commit, Python executable hash, and download sources.
The source-file manifest identifies the packaged bytes, including uncommitted build tooling; the base
commit label alone does not imply a clean checkout. Preserve `docker image inspect` and build logs too.

Runtime code and the interpreter are at `/opt/nextt`, `/opt/venv`, and `/opt/python`. Site settings must
use `venv_dir=/opt/venv` and a separate writable runtime/cache directory. Use these implemented
paths consistently in scoring configuration.

Network access to Hugging Face is disabled by default, tracking is disabled, and the registered FLA
dispatch policy is set. No reference FA3 cache is bundled: a successful package build is not evidence
that a reference run's attention backend is available. Supply and verify the appropriate kernel
artifacts and reject unexpected fallback before evaluation. Continue with the
[scoring workflow](SCORING.md) for reference capture, input/GPU preflight, direct launch and comparison.
The optional [Harbor adapter](../../adapters/harbor_eval/README.md) uses that same scoring entry point.
Build and import checks alone do not establish GPU or training acceptance; use PLAN.md for the evidence level.

GPU tests require an explicit free-device allocation and a compatible host NVIDIA runtime. Validate
CUDA/BF16, the intended kernels, distributed communication, and paired checkpoint evaluation before
calling this image a supported training or scoring release.
