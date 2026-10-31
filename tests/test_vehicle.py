"""Tests for the vehicle dynamics model (drivesim.vehicle) on flat and synthetic terrain."""

import math
import time

import numpy as np
import pytest

from drivesim.drivetrain import RPM_PER_RADS
from drivesim.vehicle import Controls, FlatTerrain, Vehicle, VehicleParams, hatchback, sports_car

DT = 1.0 / 30.0
G = 9.81


class SlopeTerrain(FlatTerrain):
    """Plane z = grad_x * x (a constant slope along x)."""

    def __init__(self, grad_x: float):
        super().__init__()
        self.grad = float(grad_x)

    def height_at(self, x, y):
        x = np.asarray(x, dtype=float)
        return self.grad * x

    def normal_at(self, x, y):
        x = np.asarray(x, dtype=float)
        n = np.zeros(x.shape + (3,))
        norm = math.sqrt(1 + self.grad ** 2)
        n[..., 0] = -self.grad / norm
        n[..., 2] = 1 / norm
        return n


class LedgeTerrain(FlatTerrain):
    """Flat ground at z=0 for x < edge, and a drop of ``drop`` metres beyond it."""

    def __init__(self, edge: float, drop: float):
        super().__init__()
        self.edge = edge
        self.drop = drop

    def height_at(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.where(x < self.edge, 0.0, -self.drop)


class KerbTerrain(FlatTerrain):
    """Flat ground with a raised kerb (step) across the road at x in [x0, x0+0.3]."""

    def __init__(self, x0: float, height: float):
        super().__init__()
        self.x0, self.height = x0, height

    def height_at(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.where((x > self.x0) & (x < self.x0 + 0.3), self.height, 0.0)


@pytest.fixture
def flat():
    return FlatTerrain()


def make_car(terrain, params=None, transmission="auto", speed=0.0, heading=0.0):
    """Vehicle reset at the origin; with ``speed`` it starts rolling, engaged in a gear that
    matches the engine speed to the wheel speed (so the drivetrain starts in a consistent state)."""
    car = Vehicle(params or sports_car(), transmission)
    car.reset([0.0, 0.0, 0.0], heading, terrain)
    if speed:
        car.vel = [speed * math.cos(heading), speed * math.sin(heading), 0.0]
        car.w = [speed / car.R] * 4          # rolling, not locked
        gear, G = 1, 0.0
        for gear in range(1, 7):
            G = car.p.gear_ratios[gear - 1] * car.p.final_drive
            if G * speed / car.R * RPM_PER_RADS <= 6000.0:
                break
        car.pt.gear = gear
        car.pt.pending_gear = gear
        car.pt.engage = 1.0
        car.pt.engine_w = G * speed / car.R
    return car


def accel_time_0_100(params, transmission="auto"):
    t = FlatTerrain()
    car = make_car(t, params, transmission)
    for _ in range(30 * 60):
        car.step(Controls(throttle=1.0), DT, t)
        if car.speed_kmh >= 100.0:
            return car.time
    return float("inf")


# ---------------------------------------------------------------- API / basics
def test_api_shapes_and_types(flat):
    car = make_car(flat)
    assert car.position.shape == (3,)
    assert car.velocity_world.shape == (3,)
    assert car.velocity_local.shape == (3,)
    assert car.angular_velocity.shape == (3,)
    assert car.quaternion.shape == (4,)
    for name in ("wheel_slip_ratios", "wheel_slip_angles", "wheel_loads",
                 "wheel_in_contact", "suspension_compression"):
        assert getattr(car, name).shape == (4,), name
    assert car.wheel_positions_world().shape == (4, 3)
    assert isinstance(car.get_state_dict(), dict)
    assert car.is_flipped() is False
    assert isinstance(car.gear, int)


def test_controls_defaults_are_zero():
    c = Controls()
    assert (c.steer, c.throttle, c.brake, c.handbrake, c.clutch) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert c.shift_up is False and c.shift_down is False


def test_dict_controls_are_accepted(flat):
    car = make_car(flat)
    car.step({"throttle": 0.5, "steer": 0.2}, DT, flat)
    assert car.speed > 0.0


def test_presets_are_distinct():
    s, h = sports_car(), hatchback()
    assert s.mass == pytest.approx(1300.0) and h.mass == pytest.approx(1100.0)
    assert h.drive == "fwd" and s.drive == "rwd"


# ------------------------------------------------------- rest and settling
def test_car_at_rest_stays_at_rest_and_settles(flat):
    car = make_car(flat)
    p0 = car.position.copy()
    for _ in range(3 * 30):
        car.step(Controls(), DT, flat)
    assert np.linalg.norm(car.position[:2] - p0[:2]) < 1e-3
    assert car.speed < 1e-3
    assert abs(car.pitch) < 0.01 and abs(car.roll) < 0.01
    assert car.wheel_in_contact.all()


def test_ride_height_and_static_load_are_correct(flat):
    p = sports_car()
    car = make_car(flat, p)
    W = p.mass * G
    front_share = (p.wheelbase - p.cg_to_front) / p.wheelbase
    expected_load = [W * front_share / 2, W * front_share / 2,
                     W * (1 - front_share) / 2, W * (1 - front_share) / 2]
    loads = car.wheel_loads
    assert np.allclose(loads, expected_load, rtol=0.03)
    assert loads.sum() == pytest.approx(W, rel=0.01)
    # suspension compression equals the static sag spring/load
    sag_front = expected_load[0] / p.spring_rate_front
    assert car.suspension_compression[0] == pytest.approx(sag_front, abs=0.01)
    # ride height: CoG = ground + R + free spring length (compressed) - hardpoint height
    expected_cog = p.tire.radius + (p.suspension_rest_length - sag_front) - p.hardpoint_height
    assert car.position[2] == pytest.approx(expected_cog, abs=0.01)


def test_reset_is_independent_of_previous_state(flat):
    car = make_car(flat)
    for _ in range(60):
        car.step(Controls(throttle=1.0), DT, flat)
    car.reset([10.0, -5.0, 3.0], math.pi / 3, flat)
    assert car.position[:2] == pytest.approx([10.0, -5.0], abs=1e-4)
    assert car.yaw == pytest.approx(math.pi / 3, abs=1e-6)
    assert car.speed < 1e-3


# ------------------------------------------------------ longitudinal behaviour
def test_0_100_kmh_sports_car_is_plausible():
    t = accel_time_0_100(sports_car())
    assert 5.0 <= t <= 7.0, f"0-100 km/h took {t:.2f} s"


def test_hatchback_is_slower_than_sports_car():
    t_hatch = accel_time_0_100(hatchback())
    assert 9.0 <= t_hatch <= 13.0, f"hatchback 0-100 took {t_hatch:.2f} s"


def test_top_speed_is_plausible():
    t = FlatTerrain()
    car = make_car(t)
    for _ in range(30 * 240):
        car.step(Controls(throttle=1.0), DT, t)
        if car.speed_kmh > 300:
            break
    assert 240.0 <= car.speed_kmh <= 290.0, f"top speed {car.speed_kmh:.1f} km/h"
    assert 0 <= car.engine_rpm <= 7300


def test_full_brake_from_100_kmh_stops_in_plausible_distance():
    t = FlatTerrain()
    car = make_car(t, speed=100.0 / 3.6)
    x0 = car.position[0]
    for _ in range(30 * 30):
        car.step(Controls(brake=1.0), DT, t)
        if car.speed < 0.05:
            break
    dist = car.position[0] - x0
    assert 35.0 <= dist <= 45.0, f"braking distance {dist:.1f} m"
    assert car.speed < 0.05


def test_hard_braking_locks_wheels(flat):
    car = make_car(flat, speed=20.0)
    seen_lock = False
    for _ in range(60):
        car.step(Controls(brake=1.0), DT, flat)
        if car.wheel_slip_ratios.min() < -0.6:
            seen_lock = True
    assert seen_lock


def test_front_load_transfer_under_braking(flat):
    car = make_car(flat, speed=20.0)
    static = car.wheel_loads.copy()
    for _ in range(12):
        car.step(Controls(brake=1.0), DT, flat)
    loads = car.wheel_loads
    assert loads[:2].sum() > static[:2].sum() * 1.2
    assert loads[2:].sum() < static[2:].sum() * 0.85


def test_wheelspin_on_launch(flat):
    car = make_car(flat)
    car.step(Controls(), DT, flat)
    car.pt.gear = 1
    car.pt.engage = 0.0
    worst = 0.0
    for _ in range(60):
        car.step(Controls(throttle=1.0), DT, flat)
        worst = max(worst, car.wheel_slip_ratios[2:].max())
    assert worst > 0.08


# ------------------------------------------------------------ lateral dynamics
def _skidpad_lateral_g(a_des_g: float, speed: float, seconds: float = 40.0) -> float:
    """Closed-loop skidpad: track yaw rate for a desired lateral acceleration; return steady g."""
    t = FlatTerrain()
    car = make_car(t, speed=speed)
    integ = 0.0
    lat = []
    n = int(seconds * 30)
    for i in range(n):
        r_des = a_des_g * G / speed
        err = r_des - car.yaw_rate
        integ += err * DT
        steer_rad = r_des * 2.6 / speed * 4.0 + 0.5 * err + 0.2 * integ
        steer = max(-1.0, min(1.0, steer_rad / 0.6))
        throttle = 0.3 if car.forward_speed < speed else 0.0
        car.step(Controls(steer=steer, throttle=throttle), DT, t)
        if i > 0.6 * n:
            lat.append(car.lateral_accel)
    return float(np.mean(lat)) / G


def test_skidpad_reaches_about_one_g():
    g_achieved = _skidpad_lateral_g(1.05, 15.0)
    assert 0.85 <= g_achieved <= 1.1, f"skidpad {g_achieved:.3f} g"


def test_skidpad_tracks_moderate_lateral_accel():
    g_achieved = _skidpad_lateral_g(0.5, 15.0)
    assert g_achieved == pytest.approx(0.5, abs=0.05)


def test_handbrake_induces_oversteer_slide(flat):
    car = make_car(flat, speed=20.0)
    base_yaw = []
    for _ in range(30):
        car.step(Controls(), DT, flat)
    betas = []
    for i in range(60):
        hb = 1.0 if i < 15 else 0.0
        car.step(Controls(steer=0.2, handbrake=hb), DT, flat)
        betas.append(abs(car.body_slip_angle))
        base_yaw.append(abs(car.yaw_rate))
    assert max(betas) > 0.4, f"max body slip {max(betas):.2f} rad"
    assert max(base_yaw) > 0.8


def test_throttle_oversteer_grows_body_slip(flat):
    def max_slip(throttle):
        car = make_car(flat, speed=20.0)
        worst = 0.0
        for _ in range(60):
            car.step(Controls(steer=0.3, throttle=throttle), DT, flat)
            worst = max(worst, abs(car.body_slip_angle))
        return worst

    assert max_slip(1.0) > max_slip(0.0) + 0.2


def test_understeer_gradient_gives_positive_steering_response(flat):
    """Steady-state yaw rate is less than the kinematic prediction at speed (understeer)."""
    car = make_car(flat, speed=20.0)
    yaw_rates = []
    for _ in range(120):
        car.step(Controls(steer=0.1, throttle=0.2), DT, flat)
        yaw_rates.append(car.yaw_rate)
    kin = car.forward_speed * math.tan(car.steer_act) / 2.6
    assert 0.0 < np.mean(yaw_rates[-30:]) < kin * 0.98


def test_ackermann_inner_wheel_turns_more(flat):
    car = make_car(flat, speed=5.0)
    car.step(Controls(steer=1.0), DT, flat)
    for _ in range(60):
        car.step(Controls(steer=1.0), DT, flat)
    d = car.wheel_steer_angles
    assert d[0] > d[1] > 0.0          # left is inner in a left turn
    assert d[2] == 0.0 and d[3] == 0.0


def test_steering_rate_is_limited(flat):
    car = make_car(flat)
    car.step(Controls(steer=1.0), DT, flat)
    # four substeps of 1/120 s per env step, each limited to steer_rate * h
    assert car.steer_act <= car.p.steer_rate * DT + 1e-9


# ------------------------------------------------------------- manual / clutch
def test_manual_shifts_need_clutch_and_take_time(flat):
    car = make_car(flat, transmission="manual")
    assert car.gear == 0
    car.step(Controls(shift_up=True, clutch=0.0), DT, flat)
    assert car.gear == 0
    car.step(Controls(shift_up=True, clutch=1.0), DT, flat)
    assert car.shifting
    assert car.gear == 0
    for _ in range(int(0.3 * 30) + 1):
        car.step(Controls(clutch=1.0), DT, flat)
    assert car.gear == 1 and not car.shifting


def test_manual_launch_and_upshift(flat):
    car = make_car(flat, transmission="manual")
    car.step(Controls(shift_up=True, clutch=1.0), DT, flat)
    for _ in range(12):
        car.step(Controls(clutch=1.0), DT, flat)
    assert car.gear == 1
    for i in range(60):
        car.step(Controls(throttle=0.3, clutch=max(0.0, 1.0 - i / 40)), DT, flat)
    assert car.speed > 2.0 and car.engine_rpm > 800
    car.step(Controls(throttle=0.3, shift_up=True, clutch=1.0), DT, flat)
    for _ in range(12):
        car.step(Controls(throttle=0.0, clutch=1.0), DT, flat)
    assert car.gear == 2


def test_clutch_dump_at_low_rpm_stalls_engine(flat):
    car = make_car(flat, transmission="manual")
    car.step(Controls(shift_up=True, clutch=1.0), DT, flat)
    for _ in range(12):
        car.step(Controls(clutch=1.0), DT, flat)
    stalled = False
    for _ in range(90):
        car.step(Controls(throttle=0.05, clutch=0.0), DT, flat)
        if not car.pt.engine_on:
            stalled = True
            break
    assert stalled


def test_manual_engine_rpm_follows_clutch(flat):
    car = make_car(flat, transmission="manual")
    car.step(Controls(shift_up=True, clutch=1.0), DT, flat)
    for _ in range(12):
        car.step(Controls(clutch=1.0), DT, flat)
    for _ in range(30):
        car.step(Controls(throttle=1.0, clutch=1.0), DT, flat)
    free_rpm = car.engine_rpm
    assert free_rpm > 3000


def test_auto_reverse_toggle_by_holding_brake_at_standstill(flat):
    car = make_car(flat)
    for _ in range(40):                 # hold brake to a stop for 1.3 s
        car.step(Controls(brake=1.0), DT, flat)
    assert car.gear == -1
    for _ in range(40):                 # keeping the brake held does not toggle again
        car.step(Controls(brake=1.0), DT, flat)
    assert car.gear == -1
    for _ in range(40):                 # release, then hold the brake again toggles back to drive
        car.step(Controls(), DT, flat)
    for _ in range(40):
        car.step(Controls(brake=1.0), DT, flat)
    assert car.gear == 1


def test_auto_reverse_moves_backwards(flat):
    car = make_car(flat)
    for _ in range(40):
        car.step(Controls(brake=1.0), DT, flat)
    assert car.gear == -1
    for _ in range(90):
        car.step(Controls(throttle=0.5), DT, flat)
    assert car.gear == -1
    assert car.forward_speed < -0.5


def test_differential_limited_slip_and_awd_split():
    p_lsd = sports_car()
    p_lsd = VehicleParams(**{**p_lsd.__dict__, "diff": "lsd"})
    car = Vehicle(p_lsd, "auto")
    car.w = [0.0, 0.0, 100.0, 60.0]
    w = car._drive_weights()
    assert w[2] < 0.5 < w[3] + 1e-9 or w[2] < w[3]
    p_awd = VehicleParams(**{**sports_car().__dict__, "drive": "awd"})
    car = Vehicle(p_awd, "auto")
    w = car._drive_weights()
    assert sum(w) == pytest.approx(1.0)
    assert w[0] + w[1] == pytest.approx(p_awd.awd_front_split)


# ------------------------------------------------------------------- terrain
def test_flies_off_ledge_and_lands_without_nans():
    t = LedgeTerrain(edge=6.0, drop=1.5)
    car = make_car(t, speed=12.0)
    airborne = False
    for _ in range(30 * 6):
        car.step(Controls(throttle=0.3), DT, t)
        if not car.wheel_in_contact.any():
            airborne = True
        assert np.all(np.isfinite(car.position))
    assert airborne
    assert car.recoveries == 0


def test_kerb_strike_is_stable():
    t = KerbTerrain(x0=10.0, height=0.12)
    car = make_car(t, speed=8.0)
    for _ in range(30 * 4):
        car.step(Controls(throttle=0.2), DT, t)
    assert np.all(np.isfinite(car.position)) and np.all(np.isfinite(car.velocity_world))
    assert car.recoveries == 0
    assert car.speed < 30.0


def test_slope_holds_car_with_brake():
    t = SlopeTerrain(0.15)
    car = make_car(t)
    for _ in range(30 * 4):
        car.step(Controls(brake=1.0, handbrake=1.0), DT, t)
    assert car.speed < 0.2
    assert car.recoveries == 0


def test_car_rolls_on_steep_slope_without_brake():
    t = SlopeTerrain(0.15)
    car = make_car(t)
    x0 = car.position[0]
    for _ in range(30 * 4):
        car.step(Controls(), DT, t)
    assert car.position[0] < x0 - 0.5
    assert car.recoveries == 0


def test_low_friction_surface_reduces_braking():
    hi = FlatTerrain(friction=1.0)
    lo = FlatTerrain(friction=0.4)
    d = {}
    for name, t in (("hi", hi), ("lo", lo)):
        car = make_car(t, speed=20.0)
        x0 = car.position[0]
        for _ in range(30 * 20):
            car.step(Controls(brake=1.0), DT, t)
            if car.speed < 0.05:
                break
        d[name] = car.position[0] - x0
    assert d["lo"] > 1.8 * d["hi"]


# ------------------------------------------------------------- robustness
def test_ten_thousand_random_control_steps_stay_finite():
    rng = np.random.default_rng(1234)
    t = FlatTerrain()
    for mode in ("auto", "manual"):
        car = make_car(t, transmission=mode)
        for _ in range(10000):
            c = Controls(
                steer=rng.uniform(-1, 1),
                throttle=rng.uniform(0, 1),
                brake=float(rng.random() < 0.2) * rng.uniform(0, 1),
                handbrake=float(rng.random() < 0.1),
                clutch=float(rng.random() < 0.3),
                shift_up=bool(rng.random() < 0.05),
                shift_down=bool(rng.random() < 0.05),
            )
            car.step(c, DT, t)
            assert np.all(np.isfinite(car.position))
            assert np.all(np.isfinite(car.velocity_world))
            assert np.all(np.isfinite(car.angular_velocity))
        assert car.recoveries == 0
        assert car.speed < 150.0


def test_extreme_inputs_do_not_explode(flat):
    car = make_car(flat, speed=60.0)
    for _ in range(300):
        car.step(Controls(steer=1.0, throttle=1.0, brake=1.0, handbrake=1.0), DT, flat)
    assert np.all(np.isfinite(car.position)) and car.recoveries == 0
    assert car.speed < 150.0


def test_simulation_is_deterministic():
    def run():
        t = FlatTerrain()
        car = make_car(t)
        rng = np.random.default_rng(7)
        for _ in range(600):
            car.step(Controls(steer=rng.uniform(-1, 1), throttle=rng.uniform(0, 1),
                              brake=rng.uniform(0, 0.3)), DT, t)
        return car.position.copy(), car.velocity_world.copy()

    p1, v1 = run()
    p2, v2 = run()
    assert np.array_equal(p1, p2) and np.array_equal(v1, v2)


def test_ten_substeps_matches_coarser_substep_within_tolerance():
    """Integrator convergence check: 120 Hz and 240 Hz substepping should agree on a 3 s launch."""
    t = FlatTerrain()

    def run(hz):
        car = Vehicle(sports_car(), "auto", substep_hz=hz)
        car.reset([0.0, 0.0, 0.0], 0.0, t)
        for _ in range(90):
            car.step(Controls(throttle=0.8), DT, t)
        return car.position[0]

    assert run(120.0) == pytest.approx(run(240.0), rel=0.03)


# ------------------------------------------------------------------- speed
def test_speed_benchmark(capsys):
    t = FlatTerrain()
    car = make_car(t)
    n = 6000
    t0 = time.perf_counter()
    for i in range(n):
        car.step(Controls(throttle=0.7, steer=0.1 * math.sin(i / 30.0)), DT, t)
    elapsed = time.perf_counter() - t0
    rate = n / elapsed
    print(f"\nenv steps/s at dt=1/30, 120 Hz substeps (4 per step): {rate:.0f}")
    assert rate > 1000.0
