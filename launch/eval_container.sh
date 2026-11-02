#!/usr/bin/env bash
# Host wrapper: never installs packages or chooses GPUs automatically.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 -m environments.eval.launch "$@"
