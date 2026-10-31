"""Tests for drivesim.human (HumanInput / RawInput) and scripts/drive.py.

Everything here is headless. Synthetic RawInput frames stand in for the keyboard and gamepad.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from types import SimpleNamespace as NS  # noqa: E402

from drivesim.human import DEFAULT_BINDINGS, HumanInput, HumanTuning, RawInput  # noqa: E402
from drivesim.vehicle import Controls, FlatTerrain, Vehicle, VehicleParams  # noqa: E402

DT = 1.0 / 30.0


def hold(h, held, seconds, speed=0.0, gear=1, beta=0.0, pressed=None):
    """Feed a constant key set for ``seconds``; return the list of Controls."""
    out = []
    n = max(1, int(round(seconds / DT)))
    for i in range(n):
        raw = RawInput(held=set(held), pressed=set(pressed or ()) if i == 0 else set())
        out.append(h.update(raw, DT, NS(speed=speed, gear=gear, beta=beta)))
    return out


def in_range(c: Controls):
    vals = [c.steer, c.throttle, c.brake, c.handbrake, c.clutch]
    return all(np.isfinite(v) for v in vals) and -1 <= c.steer <= 1 and all(
        0 <= v <= 1 for v in vals[1:])


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------
def test_steer_builds_progressively_not_instantly():
    h = HumanInput()
    cs = hold(h, {"a"}, 0.6)
    assert abs(cs[0].steer) < 0.1                          # first frame is gentle
    assert all(abs(b.steer) >= abs(a.steer) - 1e-9 for a, b in zip(cs, cs[1:]))  # monotone build
    assert abs(cs[-1].steer) == pytest.approx(1.0)         # reaches full lock within ~0.5 s
    assert cs[-1].steer > 0                                # 'a' is left (+)


def test_steer_recenters_fast_on_release():
    h = HumanInput()
    hold(h, {"d"}, 1.0)
    assert h.controls.steer == pytest.approx(-1.0)
    cs = hold(h, set(), 0.25)
    assert abs(cs[-1].steer) < 0.05                        # centred in about a quarter second
    assert cs[-1].steer == pytest.approx(0.0, abs=1e-9) or abs(cs[-1].steer) < 0.05


def test_release_is_faster_than_press():
    t = HumanTuning()
    assert t.steer_release_rate > t.steer_press_rate
    h = HumanInput()
    press = hold(h, {"a"}, 0.1)[-1].steer
    h2 = HumanInput()
    hold(h2, {"a"}, 1.0)
    rel = hold(h2, set(), 0.1)[-1].steer
    assert abs(press) > 0 and abs(rel) < 1.0 - 0.3       # released drops quickly


def test_flip_direction_is_quicker_than_build():
    h = HumanInput()
    hold(h, {"a"}, 1.0)
    cs = hold(h, {"d"}, 0.2)
    assert cs[-1].steer < 0.0                              # went through centre to the other side
    h2 = HumanInput()
    build = hold(h2, {"a"}, 0.2)[-1].steer                 # the same time spent building from centre
    assert abs(build) < 1.0 and build > 0.0


def test_steer_output_is_nonlinear_for_fine_control():
    h = HumanInput()
    cs = hold(h, {"a"}, 0.1)                               # small signal
    s = h._steer
    assert 0 < s < 0.5
    assert cs[-1].steer < s                                # curve compresses small inputs


def test_steer_limit_shrinks_with_speed():
    h = HumanInput()
    assert h.steer_limit(0.0) == pytest.approx(1.0)
    assert h.steer_limit(30.0) == pytest.approx(h.tuning.steer_limit_min)
    assert h.steer_limit(60.0) == pytest.approx(h.tuning.steer_limit_min)   # clamped beyond
    speeds = np.linspace(0, 30, 20)
    lims = [h.steer_limit(v) for v in speeds]
    assert all(b <= a + 1e-12 for a, b in zip(lims, lims[1:]))              # monotone decrease

    slow = HumanInput()
    fast = HumanInput()
    cs_slow = hold(slow, {"a"}, 1.0, speed=5.0)
    cs_fast = hold(fast, {"a"}, 1.0, speed=30.0)
    assert max(abs(c.steer) for c in cs_slow) == pytest.approx(h.steer_limit(5.0))
    assert h.steer_limit(5.0) > h.steer_limit(20.0) > h.tuning.steer_limit_min
    assert max(abs(c.steer) for c in cs_fast) == pytest.approx(h.tuning.steer_limit_min)


def test_gamepad_stick_deadzone_and_sign():
    h = HumanInput()
    c = h.update(RawInput(axes={"steer": 0.05}), DT, NS(speed=0))
    assert c.steer == 0.0
    c = h.update(RawInput(axes={"steer": 0.9}), DT, NS(speed=0))
    assert c.steer > 0.5                                   # stick right-positive from device -> left steer
    c = h.update(RawInput(axes={"steer": -1.0}), DT, NS(speed=0))
    assert c.steer == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# pedals and handbrake
# ---------------------------------------------------------------------------
def test_throttle_ramps_up_and_down():
    h = HumanInput()
    cs = hold(h, {"w"}, 1.5, gear=1)
    assert cs[0].throttle < 0.1
    assert 0.3 < cs[int(0.3 / DT)].throttle < 1.0
    assert cs[-1].throttle == pytest.approx(1.0)
    cs = hold(h, set(), 0.1)
    assert 0.0 < cs[-1].throttle < 1.0 or cs[-1].throttle == 0.0   # decays
    cs = hold(h, set(), 1.0)
    assert cs[-1].throttle == 0.0


def test_brake_ramps_and_overrides_gas():
    h = HumanInput()
    hold(h, {"w"}, 1.0)
    cs = hold(h, {"w", "s"}, 0.5, speed=10, gear=1)
    assert cs[-1].throttle == 0.0                          # gas is let off while braking
    assert cs[0].throttle > cs[-1].throttle                # ...as a ramp, not a cut
    h = HumanInput()
    cs = hold(h, {"s"}, 0.2, speed=10)
    assert 0.0 < cs[0].brake < 1.0 and cs[-1].brake > cs[0].brake
    cs = hold(h, {"s"}, 1.0, speed=10)
    assert cs[-1].brake == pytest.approx(1.0)


def test_handbrake_on_space_ramps():
    h = HumanInput()
    cs = hold(h, {"space"}, 0.5)
    assert cs[0].handbrake < 0.5
    assert cs[-1].handbrake == pytest.approx(1.0)
    cs = hold(h, set(), 0.5)
    assert cs[-1].handbrake == 0.0


# ---------------------------------------------------------------------------
# reverse convention
# ---------------------------------------------------------------------------
def test_in_reverse_gear_s_is_gas_and_w_is_brake():
    h = HumanInput()
    cs = hold(h, {"s"}, 1.0, speed=0.0, gear=-1)
    assert cs[-1].throttle == pytest.approx(1.0)
    assert cs[-1].brake == 0.0
    assert h.buttons["gas"] is True
    cs = hold(h, {"w"}, 1.0, speed=0.0, gear=-1)
    assert cs[-1].brake == pytest.approx(1.0)
    assert cs[-1].throttle == 0.0


def test_brake_hold_at_standstill_is_tracked():
    h = HumanInput()
    hold(h, {"s"}, 0.5, speed=0.0, gear=1)
    assert h.brake_hold > 0.2
    hold(h, {"s"}, 0.5, speed=10.0, gear=1)
    assert h.brake_hold == 0.0


def test_vehicle_reverses_when_brake_held_at_standstill_via_s_key():
    """Real Vehicle: holding S at standstill engages reverse and S then drives backwards."""
    terrain = FlatTerrain()
    car = Vehicle(VehicleParams.sports_car(), transmission="auto")
    car.reset([0.0, 0.0, 0.0], 0.0, terrain)
    h = HumanInput()
    for _ in range(int(2.0 / DT)):
        if car.gear == -1:
            break
        car.step(h.update(RawInput(held={"s"}), DT, car), DT, terrain)
    assert car.gear == -1
    start_x = float(car.position[0])
    for _ in range(int(2.0 / DT)):
        car.step(h.update(RawInput(held={"s"}), DT, car), DT, terrain)
    assert car.forward_speed < -0.5                        # driving backwards
    assert float(car.position[0]) < start_x


# ---------------------------------------------------------------------------
# manual transmission
# ---------------------------------------------------------------------------
def test_manual_clutch_ramps_and_shift_edges_fire_once():
    h = HumanInput(manual=True)
    cs = hold(h, {"lshift"}, 0.5)
    assert 0.0 < cs[0].clutch < 1.0
    assert cs[-1].clutch == pytest.approx(1.0)
    cs = hold(h, set(), 0.1)
    assert cs[-1].clutch > 0.5                             # clutch lets go slowly

    h = HumanInput(manual=True)
    c1 = h.update(RawInput(pressed={"e"}, held={"e"}), DT, NS(speed=0))
    c2 = h.update(RawInput(held={"e"}), DT, NS(speed=0))
    c3 = h.update(RawInput(pressed={"q"}, held={"q"}), DT, NS(speed=0))
    assert c1.shift_up and not c2.shift_up and c3.shift_down and not c3.shift_up


def test_automatic_mode_reports_no_shift_or_clutch():
    h = HumanInput(manual=False)
    c = h.update(RawInput(pressed={"e", "lshift"}, held={"e", "lshift"}), DT, NS(speed=0))
    assert not c.shift_up and not c.shift_down and c.clutch == 0.0


# ---------------------------------------------------------------------------
# assists
# ---------------------------------------------------------------------------
def test_assists_default_off_and_toggle_action():
    h = HumanInput()
    assert h.assists is False
    h.update(RawInput(pressed={"t"}, held={"t"}), DT, NS(speed=0))
    assert h.assists is True and "assist" in h.actions
    h.update(RawInput(), DT, NS(speed=0))
    h.update(RawInput(pressed={"t"}, held={"t"}), DT, NS(speed=0))
    assert h.assists is False


def test_counter_steer_assist_points_wheel_toward_slide():
    # beta < 0: the car is moving to the right of its heading, so the assist steers right (negative)
    h = HumanInput(assists=True)
    c = h.update(RawInput(), DT, NS(speed=15, beta=-0.3, gear=1))
    assert c.steer < -0.2
    h = HumanInput(assists=True)
    c = h.update(RawInput(), DT, NS(speed=15, beta=0.3, gear=1))
    assert c.steer > 0.2
    # inside the deadzone or below the minimum speed it does nothing
    h = HumanInput(assists=True)
    assert h.update(RawInput(), DT, NS(speed=15, beta=0.02, gear=1)).steer == 0.0
    h = HumanInput(assists=True)
    assert h.update(RawInput(), DT, NS(speed=1, beta=-0.3, gear=1)).steer == 0.0
    h = HumanInput(assists=False)
    assert h.update(RawInput(), DT, NS(speed=15, beta=-0.3, gear=1)).steer == 0.0


def test_throttle_trimmed_when_sliding_with_assist():
    h = HumanInput(assists=True)
    hold(h, {"w"}, 1.0, speed=15, beta=0.0)
    c = h.update(RawInput(held={"w"}), DT, NS(speed=15, beta=0.6, gear=1))
    assert c.throttle < 1.0


def rolling_car(terrain, speed):
    """Vehicle already rolling at ``speed`` in a gear matched to the wheels (as in test_vehicle)."""
    car = Vehicle(VehicleParams.sports_car(), "auto")
    car.reset([0.0, 0.0, 0.0], 0.0, terrain)
    car.vel = [speed, 0.0, 0.0]
    car.w = [speed / car.R] * 4
    gear = 1
    for gear in range(1, 7):
        G = car.p.gear_ratios[gear - 1] * car.p.final_drive
        if G * speed / car.R * 60.0 / (2 * np.pi) <= 6000.0:
            break
    car.pt.gear = gear
    car.pt.pending_gear = gear
    car.pt.engage = 1.0
    car.pt.engine_w = G * speed / car.R
    return car


def test_stability_assist_reduces_throttle_oversteer_on_real_vehicle():
    """Full throttle while steering left for 1.3 s at 20 m/s: the assist (counter-steer and gas trim)
    should cut the peak body slip. It is not a full ESC: a handbrake spin is not recovered."""
    terrain = FlatTerrain()

    def run(assists):
        car = rolling_car(terrain, 20.0)
        for _ in range(30):
            car.step(Controls(), DT, terrain)
        h = HumanInput(assists=assists)
        peak = 0.0
        for i in range(int(3.0 / DT)):
            held = {"w", "a"} if i < 40 else {"w"}
            c = h.update(RawInput(held=held), DT, car)
            car.step(c, DT, terrain)
            if i > 10:
                peak = max(peak, abs(car.body_slip_angle))
        return peak

    base = run(False)
    assisted = run(True)
    assert base > 0.25                                     # the setup really produces an oversteer slide
    assert assisted < 0.75 * base


# ---------------------------------------------------------------------------
# actions, env mapping, ranges
# ---------------------------------------------------------------------------
def test_action_roles_fire_once_per_press():
    h = HumanInput()
    h.update(RawInput(pressed={"c"}, held={"c"}), DT, NS(speed=0))
    assert "camera" in h.actions
    h.update(RawInput(held={"c"}), DT, NS(speed=0))
    assert "camera" not in h.actions
    h.update(RawInput(pressed={"p"}, held={"p"}), DT, NS(speed=0))
    assert "pause" in h.actions
    h.update(RawInput(pressed={"escape"}, held={"escape"}), DT, NS(speed=0))
    assert "quit" in h.actions


def test_gamepad_buttons_map_to_roles():
    h = HumanInput(manual=True)
    h.update(RawInput(pressed={"pad_5"}, held={"pad_5"}), DT, NS(speed=0))
    assert h.controls.shift_up
    h.update(RawInput(held={"pad_0"}), DT, NS(speed=0))
    assert h.controls.handbrake > 0.0


def test_discrete_mapping_matches_env_table():
    gym = pytest.importorskip("gymnasium")  # noqa: F841
    from drivesim.env import DISCRETE_ACTIONS
    cases = [
        (frozenset(), (1, 0, 0)),              # nothing held -> none, no gas, no brake
        (frozenset({"a", "w"}), (0, 1, 0)),    # left + gas
        (frozenset({"d", "s"}), (2, 0, 1)),    # right + brake
        (frozenset({"a", "s"}), (0, 0, 1)),
        (frozenset({"d", "w"}), (2, 1, 0)),
    ]
    for held, (steer_idx, gas, brake) in cases:
        h = HumanInput()
        h.update(RawInput(held=set(held)), DT, NS(speed=0, gear=1))
        idx = h.to_env_action("discrete")
        assert isinstance(idx, int)
        assert idx == steer_idx * 4 + gas * 2 + brake
        steer, g, b = DISCRETE_ACTIONS[idx]
        assert (g, b) == (float(gas), float(brake))
        assert np.sign(steer) == {0: 1, 1: 0, 2: -1}[steer_idx]
        assert np.sign(steer) == {0: 1, 1: 0, 2: -1}[steer_idx]


def test_env_action_shapes_and_order():
    h = HumanInput(manual=True)
    h.update(RawInput(held={"d", "w", "space", "lshift"}), DT, NS(speed=0))
    cont = h.to_env_action("continuous")
    full = h.to_env_action("full")
    assert cont.shape == (3,) and cont.dtype == np.float32
    assert full.shape == (7,) and full.dtype == np.float32
    assert np.allclose(full[:3], cont)
    assert full[3] == pytest.approx(h.controls.clutch)     # clutch at index 3
    assert full[4] == pytest.approx(h.controls.handbrake)  # handbrake at index 4
    with pytest.raises(ValueError):
        h.to_env_action("bogus")


def test_env_actions_are_accepted_by_drive_env():
    gym = pytest.importorskip("gymnasium")  # noqa: F841
    from drivesim.env import DriveEnv
    env = DriveEnv(closed=True, seed=1, control_mode="full")
    env.reset()
    h = HumanInput(manual=True)
    for _ in range(10):
        h.update(RawInput(held={"w"}), DT, NS(speed=0))
        obs, *_ = env.step(h.to_env_action("full"))
    assert np.all(np.isfinite(obs))
    env.close()


def test_controls_stay_in_range_under_random_input():
    rng = np.random.default_rng(0)
    keys = list(DEFAULT_BINDINGS)
    h = HumanInput(manual=True, assists=True)
    for i in range(3000):
        held = {k for k in keys if rng.random() < 0.2}
        pressed = {k for k in held if rng.random() < 0.1}
        axes = {"steer": float(rng.uniform(-1.5, 1.5)), "gas": float(rng.uniform(-0.5, 1.5)),
                "brake": float(rng.uniform(-0.5, 1.5))}
        c = h.update(RawInput(held=held, pressed=pressed, axes=axes), DT,
                     NS(speed=float(rng.uniform(0, 60)), beta=float(rng.uniform(-1, 1)),
                        gear=int(rng.integers(-1, 6))))
        assert in_range(c), (i, c)
        assert c.shift_up in (True, False)


def test_reset_clears_state():
    h = HumanInput()
    hold(h, {"d", "w"}, 1.0)
    h.reset()
    assert h.controls.steer == 0.0 and h.controls.throttle == 0.0


# ---------------------------------------------------------------------------
# drive.py
# ---------------------------------------------------------------------------
def _run_drive(*args, timeout=600):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "drive.py"), *args],
                          capture_output=True, text=True, env=env, timeout=timeout, cwd=ROOT)


def test_drive_headless_run_no_viewer():
    r = _run_drive("--headless-test", "150", "--seed", "2", "--no-viewer")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "headless: steps=150" in r.stdout


def test_drive_headless_run_manual_hatchback():
    r = _run_drive("--headless-test", "60", "--seed", "4", "--no-viewer", "--manual", "--car", "hatchback",
                   "--assists", "--open")
    assert r.returncode == 0, r.stderr[-2000:]


def test_drive_headless_run_with_viewer():
    pytest.importorskip("pygame")
    try:
        import drivesim.viewer  # noqa: F401
    except ImportError:
        pytest.skip("drivesim.viewer not available")
    r = _run_drive("--headless-test", "30", "--seed", "2")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "headless: steps=30" in r.stdout
