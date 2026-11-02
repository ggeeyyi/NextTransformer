#!/usr/bin/env bash
set -euo pipefail
export NATIVE_HARDWARE=h200
exec bash "$(dirname "${BASH_SOURCE[0]}")/native_train.sh" "$@"
