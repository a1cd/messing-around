"""Scripted-track option of scripts/drive.py and the showcase circuit (headless)."""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _run_drive(*args, timeout=900):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "drive.py"), *args],
                          capture_output=True, text=True, env=env, timeout=timeout, cwd=ROOT)


def test_drive_track_showcase_headless():
    r = _run_drive("--track", "showcase", "--headless-test", "60", "--no-viewer")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "headless: steps=60" in r.stdout


def test_drive_track_rejects_unknown_name():
    r = _run_drive("--track", "nope", "--headless-test", "5", "--no-viewer")
    assert r.returncode != 0
    assert "invalid choice" in r.stderr


@pytest.fixture(scope="module")
def showcase():
    from drivesim.track_script import build_showcase_world
    return build_showcase_world(0)


def test_showcase_registered_and_closed(showcase):
    from drivesim.track_script import NAMED_SCRIPTS
    assert "showcase" in NAMED_SCRIPTS
    terrain, track = showcase
    assert track.closed
    assert 2000.0 < track.length < 3500.0
    assert np.all(np.isfinite(track.point_at(0.0)[0]))
