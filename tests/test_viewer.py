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



# ------------------------------------------------------------ wheel geometry
ORIENTATIONS = ("level", "roll30", "roll-30", "pitch15", "upside_down", "flipped")


def _set_orientation(car, terrain, name):
    """Set an orientation and step once. Setting car.q alone leaves the hubs stale until a step."""
    deg = {"level": 0.0, "roll30": 30.0, "roll-30": -30.0, "pitch15": 0.0,
           "upside_down": 170.0, "flipped": 180.0}[name]
    a = np.radians(deg)
    if name == "pitch15":
        car.q = [np.cos(a / 2 + np.radians(15) / 2), 0.0, np.sin(np.radians(15) / 2), 0.0]
    elif name == "level":
        pass
    else:
        car.q = [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
    car.step(Controls(), 1 / 30, terrain)


@pytest.mark.parametrize("name", ORIENTATIONS)
def test_wheel_geometry_matches_vehicle_hubs(world, viewer, name):
    """Drawn wheels sit on wheel_positions_world() and the hub offsets match the hardpoints."""
    from drivesim.viewer import CAR_BOXES, TIRE_CAP, TIRE_SIDE, WHEEL_SIDES, _Camera, _quat_axes

    terrain, _ = world
    car = _start_car(world)
    _set_orientation(car, terrain, name)
    X, Y, Z = _quat_axes(car.quaternion)
    M = np.stack([X, Y, Z], axis=1)
    hubs = np.asarray(car.wheel_positions_world(), float)
    local = (hubs - car.position) @ M               # hub offsets in the body frame

    p = car.p
    rear = -(p.wheelbase - p.cg_to_front)
    expect_x = np.array([p.cg_to_front, p.cg_to_front, rear, rear])
    expect_y = np.array([p.track_front, -p.track_front, p.track_rear, -p.track_rear]) / 2
    assert np.allclose(local[:, 0], expect_x, atol=0.05), local
    assert np.allclose(local[:, 1], expect_y, atol=0.05), local
    assert np.all(np.abs(local[:, 2]) < 0.6), local

    faces = viewer._car_faces(car.position, (X, Y, Z), car)
    caps = [pts for pts, n, col in faces if col == TIRE_CAP]
    sides = [pts for pts, n, col in faces if col == TIRE_SIDE]
    assert len(caps) == 8 and len(sides) == 4 * WHEEL_SIDES
    for i in range(4):
        cap_pair = caps[2 * i:2 * i + 2]
        mid = (cap_pair[0].mean(0) + cap_pair[1].mean(0)) / 2.0
        assert np.allclose(mid, hubs[i], atol=1e-9), (name, i)
        side_mean = np.concatenate(sides[i * WHEEL_SIDES:(i + 1) * WHEEL_SIDES]).mean(0)
        assert np.allclose(side_mean, hubs[i], atol=1e-6), (name, i)

    # screen position of the drawn wheel centre equals the projection of the hub
    cam = _Camera(car.position + X * -6 + Z * 3, X, Z, 300.0, W / 2, H / 2)
    for i in range(4):
        drawn = (caps[2 * i].mean(0) + caps[2 * i + 1].mean(0)) / 2.0
        assert np.allclose(cam.project(cam.to_cam(drawn)), cam.project(cam.to_cam(hubs[i])))

    # steered front wheels: axle = cross(Z, rolling direction), rolling dir = X rotated by steer
    steer = car.wheel_steer_angles
    for i in (0, 1):
        d = steer[i]
        f = np.cos(d) * X + np.sin(d) * Y
        axle = np.cross(Z, f)
        axle = axle / np.linalg.norm(axle)
        cap_normal = np.cross(Z, f)  # same direction as the drawn cap normal (up to sign)
        assert abs(abs(np.dot(axle, cap_normal / np.linalg.norm(cap_normal))) - 1.0) < 1e-9


def test_upside_down_wheels_are_above_body_centre(world):
    """Upside down, the wheel hubs are above the centre of gravity in world coordinates."""
    from drivesim.viewer import _quat_axes
    terrain, _ = world
    car = _start_car(world)
    _set_orientation(car, terrain, "flipped")
    X, Y, Z = _quat_axes(car.quaternion)
    local = (car.wheel_positions_world() - car.position) @ np.stack([X, Y, Z], axis=1)
    assert car.is_flipped()
    assert np.allclose(local[:, 0], [1.25, 1.25, -1.35, -1.35], atol=0.05)
    # the body's -Z axis points up in the world, so the wheels are above the centre of gravity
    assert np.all(local[:, 2] < 0.0)
    assert np.all(car.wheel_positions_world()[:, 2] > car.position[2])
