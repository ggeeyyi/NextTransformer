# The defaults are linux/amd64 manifests fetched through registry mirrors.
# No GPU is needed at build time. This image is a candidate until GPU parity passes.
ARG CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
ARG UV_IMAGE=ghcr.m.daocloud.io/astral-sh/uv@sha256:0f419824ea1810fe2a47af12aef8e8c39eabae8d19c728cd9612d18e17a98d17
FROM ${UV_IMAGE} AS uv_tool
FROM ${CUDA_IMAGE}

ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=uv_tool /uv /usr/local/bin/uv

# CUDA is already in the base image; no packages are fetched from NVIDIA's apt repo.
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && printf 'deb %s jammy main universe restricted multiverse\ndeb %s jammy-updates main universe restricted multiverse\ndeb %s jammy-security main universe restricted multiverse\n' \
         "$APT_MIRROR" "$APT_MIRROR" "$APT_MIRROR" > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PROJECT_ENVIRONMENT=/opt/venv
ARG PYTHON_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
RUN UV_PYTHON_INSTALL_MIRROR="$PYTHON_MIRROR" uv python install 3.10.19 \
    && uv venv --python /opt/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3 /opt/venv

WORKDIR /opt/nextt
COPY pyproject.toml uv.lock README.md ./
COPY requirements/ requirements/
# Export preserves locked versions and hashes, while installing via the mirror
# avoids fetching the original registry URLs embedded in uv.lock. Never re-lock.
ARG PYPI_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
RUN --mount=type=cache,id=nextt-eval-uv-cu128,target=/root/.cache/uv \
    mkdir -p /opt/nextt-manifest \
    && uv export --frozen --no-dev --extra gpu --extra upstream --extra tracking \
         --no-emit-project --format requirements-txt \
         --output-file /opt/nextt-manifest/runtime.requirements.txt > /dev/null \
    && uv pip sync --python /opt/venv/bin/python --require-hashes --only-binary :all: \
         --index-url "$PYPI_INDEX_URL" /opt/nextt-manifest/runtime.requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps --require-hashes \
         --only-binary :all: --index-url "$PYPI_INDEX_URL" -r requirements/linear.txt \
    && uv pip check --python /opt/venv/bin/python \
    && uv pip freeze --python /opt/venv/bin/python > /opt/nextt-manifest/environment.complete.txt

COPY nexttransformer/ nexttransformer/
COPY nanochat/ nanochat/
COPY nextt_gym/ nextt_gym/
COPY configs/ configs/
COPY launch/ launch/
COPY environments/eval/ environments/eval/
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/nextt PYTHONNOUSERSITE=1
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
ENV FLA_DISABLE_BACKEND_DISPATCH=1
ARG CUDA_IMAGE
ARG UV_IMAGE
ARG SOURCE_REVISION=unknown
RUN dpkg-query -W > /opt/nextt-manifest/system-packages.txt \
    && python environments/eval/inventory.py --source-revision "$SOURCE_REVISION" \
         --cuda-image "$CUDA_IMAGE" --uv-image "$UV_IMAGE" \
         --apt-mirror "$APT_MIRROR" --pypi-index "$PYPI_INDEX_URL" --python-mirror "$PYTHON_MIRROR" \
         > /opt/nextt-manifest/build.json
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
LABEL org.opencontainers.image.description="NextTransformer GPU runtime candidate; GPU parity and reference FA3 artifacts not yet validated"
CMD ["python", "-m", "nexttransformer.evaluate", "--help"]
