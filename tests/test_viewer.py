"""Headless tests for the software 3D viewer (drivesim/viewer.py)."""
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from drivesim.vehicle import Controls, Vehicle
from drivesim.viewer import CAMERAS, Viewer, save_frames
from drivesim.world import generate_world

W, H = 800, 500


@pytest.fixture(scope="module")
def world():
    return generate_world(3, closed=True, difficulty=0.5)


@pytest.fixture(scope="module")
def viewer(world):
    terrain, track = world
    v = Viewer(terrain, track, width=W, height=H, headless=True)
    yield v
    v.close()


def _start_car(world):
    terrain, track = world
    car = Vehicle()
    pos, yaw = track.start_pose()
    car.reset(pos[:2], yaw, terrain)
    return car


def _pure_pursuit(track, car):
    la = track.lookahead(car.position[:2], car.yaw, [6.0])[0]
    curv = 2 * la[1] / max(la[0] ** 2 + la[1] ** 2, 1e-6)
    return float(np.clip(curv * 2.6 / 0.6, -1.0, 1.0))


def _drive(world, car, seconds):
    terrain, track = world
    for _ in range(int(seconds * 30)):
        thr = 0.35 if car.speed_kmh < 60 else 0.0
        car.step(Controls(steer=_pure_pursuit(track, car), throttle=thr), 1 / 30, terrain)


def _check_frame(frame):
    assert frame.shape == (H, W, 3)
    assert frame.dtype == np.uint8


def test_render_all_cameras_shape_dtype(world, viewer):
    car = _start_car(world)
    for cam in CAMERAS:
        _check_frame(viewer.render(car, camera=cam))


def test_render_with_hud_dict(world, viewer):
    car = _start_car(world)
    hud = {"speed_kmh": 42.0, "gear": 2, "rpm": 3100.0, "throttle": 0.5, "brake": 0.1,
           "steer": -0.3, "progress": 0.25, "reward": 1.5, "message": "lap 1",
           "custom_metric": 3.25}
    _check_frame(viewer.render(car, hud=hud, camera="chase"))
    # HUD values absent: speed, gear and rpm come from the vehicle
    _check_frame(viewer.render(car, hud={"message": ""}, camera="top"))
    _check_frame(viewer.render(car, hud=None, camera="hood"))


def test_frames_differ_when_car_moves(world, viewer):
    car = _start_car(world)
    rest = viewer.render(car, camera="chase")
    _drive(world, car, 4.0)
    moved = viewer.render(car, camera="chase")
    assert car.speed > 2.0
    assert np.any(rest != moved)
    # the top view follows the car too
    top_a = viewer.render(car, camera="top")
    _drive(world, car, 1.0)
    top_b = viewer.render(car, camera="top")
    assert np.any(top_a != top_b)


def test_camera_selection(world, viewer):
    car = _start_car(world)
    viewer.set_camera("top")
    a = viewer.render(car, camera=None)
    viewer.set_camera("hood")
    b = viewer.render(car, camera=None)
    assert np.any(a != b)
    viewer.set_camera("chase")
    assert viewer.next_camera() == "hood"
    assert viewer.next_camera() == "top"
    assert viewer.next_camera() == "chase"
    with pytest.raises(ValueError):
        viewer.set_camera("bird")
    with pytest.raises(ValueError):
        viewer.render(car, camera="bird")


def test_camera_extremes_no_exceptions(world, viewer):
    terrain, track = world
    car = _start_car(world)
    _drive(world, car, 1.0)

    # car flipped onto its roof (roll 180 deg about body X), and a sideways-on-wall pose
    flipped = car.quaternion.copy()
    car.q = [0.0, 1.0, 0.0, 0.0]
    assert car.is_flipped()
    for cam in CAMERAS:
        _check_frame(viewer.render(car, camera=cam))
    car.q = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]          # on its side
    for cam in CAMERAS:
        _check_frame(viewer.render(car, camera=cam))
    car.q = list(flipped)

    # car far off the terrain edge (heights are clamped, so the terrain runs out)
    for pos in ([5000.0, 0.0, 10.0], [-400.0, 900.0, 10.0]):
        saved = list(car.pos)
        car.pos = list(pos)
        for cam in CAMERAS:
            _check_frame(viewer.render(car, camera=cam))
        car.pos = saved

    # car at the very edge of the terrain
    half = terrain.size / 2.0 - 1.0
    car.pos = [half, half, float(terrain.height_at(half, half)) + 0.5]
    for cam in CAMERAS:
        _check_frame(viewer.render(car, camera=cam))


def test_save_frames_writes_png(world, viewer, tmp_path):
    car = _start_car(world)
    frames = [viewer.render(car, camera="chase"), viewer.render(car, camera="top")]
    paths = save_frames(frames, tmp_path / "frames")
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_fps_measurement(world, viewer):
    """Render-only throughput at 800x500 on a driven car, per camera."""
    car = _start_car(world)
    _drive(world, car, 1.0)
    fps = {}
    for cam in CAMERAS:
        viewer.render(car, camera=cam)            # warm-up
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            viewer.render(car, camera=cam)
        fps[cam] = n / (time.perf_counter() - t0)
    print("\nviewer fps (render only, 800x500):", {k: round(v, 1) for k, v in fps.items()})
    # sanity floor only; the target is >= 30 fps on a laptop CPU
    assert min(fps.values()) >= 15.0
