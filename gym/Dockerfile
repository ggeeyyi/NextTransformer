ARG BASE_IMAGE=python:3.12.13-slim-bookworm
ARG NODE_IMAGE=node:20-bookworm
FROM ${BASE_IMAGE} AS client-python
FROM ${NODE_IMAGE} AS client
COPY --from=client-python /usr/local/ /usr/local/
RUN pip install --no-cache-dir 'mcp==1.30.0'
RUN ldconfig && useradd --create-home --uid 1001 agent && mkdir -p /workspace /run/gym-runtime && chown agent:agent /workspace
ENV PYTHONPATH=/opt/client PYTHONDONTWRITEBYTECODE=1
COPY gym/src/nexttransformer_gym/__init__.py gym/src/nexttransformer_gym/client.py gym/src/nexttransformer_gym/common.py gym/src/nexttransformer_gym/mcp_server.py /opt/client/nexttransformer_gym/
COPY gym/examples/make_candidate.py gym/tasks/architecture-research/environment/runtime_snapshot.py /opt/gym/
WORKDIR /workspace
USER agent

# Codex client. The trial container drops every capability, so Harbor cannot install ripgrep or the CLI inside it;
# with the pinned CLI already present, Harbor skips installation.
FROM client AS client-codex
ARG CODEX_VERSION=0.167.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@${CODEX_VERSION} && npm cache clean --force
USER agent

FROM ${BASE_IMAGE} AS verifier
ENV PYTHONPATH=/opt/gym/src PYTHONDONTWRITEBYTECODE=1
# nexttransformer_gym.api imports jsonschema; versions from gym/uv.lock.
RUN pip install --no-cache-dir jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 \
    rpds-py==2026.9.1 attrs==26.1.0 typing-extensions==4.16.0
COPY gym/src /opt/gym/src
COPY gym/tasks/architecture-research/tests /tests
