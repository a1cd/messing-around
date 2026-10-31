# drivesim setup

## Install locally

**macOS (Apple Silicon or Intel) or Linux**

```bash
cd messing-around
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

Optional RL training dependencies (torch, and mlx on Apple Silicon):

```bash
python -m pip install -r requirements-train.txt
```

For a CPU-only torch on Linux (smaller download), install torch first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## SessionStart hook

`.claude/settings.json` registers `.claude/hooks/session-start.sh`, which runs
at the start of every Claude Code session (startup, resume, clear, compact):

- **Cloud sessions** (`CLAUDE_CODE_REMOTE=true`): runs `python -m pip install -r requirements.txt`
  into the container's Python. pip is a fast no-op when everything is already installed.
- **Local sessions** (macOS/Linux): creates a git-ignored `.venv/` in the repo if it is missing,
  installs `requirements.txt` into it, and writes a checksum marker inside `.venv/`. Later sessions skip the install
  unless `requirements.txt` or the hook script changes. It also exports `VIRTUAL_ENV` and
  `PATH` into the session so `python -m pytest` uses the venv.

The hook is synchronous: the session waits for the install to finish. It is quiet on success
and exits non-zero (with pip's error output) on failure.

## Cloud environment setup script

The hook runs in each session, but the cloud environment's setup script runs before the
container starts, so it also gives you a ready environment. Set it in the Claude Code on the web
environment settings: open the environment menu in the session title bar, choose **Edit**, and
paste this into **Setup script**:

```bash
cd messing-around && python -m pip install -r requirements.txt
```

(The setup script starts one directory above the repository, so the `cd` is required.
Adjust `messing-around` if the repository is cloned under another name.)
