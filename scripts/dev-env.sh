#!/usr/bin/env bash
# Source this before working on the project:  source scripts/dev-env.sh
#
# Why this exists: when the shell is started from the VS Code snap, XDG_DATA_HOME
# points at ~/snap/code/<revision>/.local/share, which is discarded on every snap
# update. Anything uv stores there -- managed interpreters in particular -- would
# silently disappear. Pin uv's directories to stable, snap-independent paths.

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PATH="$HOME/.local/bin:$PATH"

# The offline map extractor is .NET (ADR-0006). Installed per-user, same as uv,
# so nothing here needs root and nothing lands in a snap-versioned directory.
if [ -d "$HOME/.local/share/dotnet" ]; then
    export DOTNET_ROOT="$HOME/.local/share/dotnet"
    export DOTNET_CLI_TELEMETRY_OPTOUT=1
    export PATH="$DOTNET_ROOT:$PATH"
fi

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_repo_root/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$_repo_root/.venv/bin/activate"
else
    echo "No .venv yet -- run: uv sync --extra dev" >&2
fi
unset _repo_root
