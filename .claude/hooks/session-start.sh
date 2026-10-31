#!/bin/bash
# SessionStart hook: install drivesim Python dependencies. Idempotent and quiet.
# Cloud (CLAUDE_CODE_REMOTE=true): pip into the container's system Python.
# Local (macOS/Linux): a git-ignored .venv in the repo, reinstalled only when requirements change.
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

sha_hash() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}
sha_check() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -c --status "$1"; else shasum -a 256 -c "$1" >/dev/null 2>&1; fi
}

export PIP_ROOT_USER_ACTION=ignore
PIP_FLAGS=(-q --disable-pip-version-check)

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  PY=$(command -v python || command -v python3)
  # System Python in the cloud container: installed every start; pip is a fast no-op when satisfied.
  "$PY" -m pip install "${PIP_FLAGS[@]}" -r requirements.txt >/dev/null
else
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
  fi
  # The marker lives inside .venv (git-ignored), so it disappears with the packages.
  marker=.venv/.session-start.sha256
  if ! sha_check "$marker" 2>/dev/null; then
    rm -f "$marker"
    .venv/bin/python -m pip install "${PIP_FLAGS[@]}" -r requirements.txt >/dev/null
    sha_hash requirements.txt .claude/hooks/session-start.sh > "$marker"
  fi
  # Make `python -m pytest` in this session use the venv.
  if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    printf 'export VIRTUAL_ENV=%q\nexport PATH=%q\n' "$(pwd)/.venv" "$(pwd)/.venv/bin:$PATH" >> "$CLAUDE_ENV_FILE"
  fi
fi
