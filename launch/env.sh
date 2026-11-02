#!/usr/bin/env bash
# Source with optional --site-config/--runtime-dir; JSON is parsed as data, never sourced.
NEXTT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_nextt_exports="$("${NEXTT_BOOTSTRAP_PYTHON:-python3}" "$NEXTT_REPO_ROOT/nexttransformer/runtime/site.py" --shell "$@")" || return $?
eval "$_nextt_exports"
unset _nextt_exports
