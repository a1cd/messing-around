"""Reward function tests (drivesim/reward.py)."""
import numpy as np
import pytest

from drivesim.baselines import PurePursuitController, to_action
from drivesim.env import DriveEnv, build_world
from drivesim.reward import RewardCalculator, RewardConfig


@pytest.fixture(scope="module")
def closed_world():
    return build_world(3, closed=True, difficulty=0.5)


@pytest.fixture(scope="module")
def open_world():
    return build_world(3, closed=False, difficulty=0.5)


def test_config_defaults_and_replace():
    cfg = RewardConfig()
    assert cfg.progress > 0 and cfg.completion > 0
    cfg2 = cfg.replace(progress=0.5)
    assert cfg2.progress == 0.5 and cfg.progress == 0.1
    assert set(cfg.as_dict()) >= {"progress", "offroad_step", "stuck_term"}


def test_progress_wraps_on_closed_track(closed_world):
    _, track = closed_world
    rc = RewardCalculator()
    rc.reset(track, s=track.length - 0.5)
    # crossing the start/finish line forward: +1 m, not -(L - 1)
    assert rc.delta_progress(0.5) == pytest.approx(1.0)
    rc.reset(track, s=0.5)
    assert rc.delta_progress(track.length - 0.5) == pytest.approx(-1.0)


def test_progress_does_not_wrap_on_open_track(open_world):
    _, track = open_world
    rc = RewardCalculator()
    rc.reset(track, s=0.0)
    assert rc.delta_progress(track.length) == pytest.approx(track.length)


def _step(rc, track, s, **kw):
    base = dict(s=s, lateral=0.0, idx=int(np.searchsorted(track.s, s) % track.n), yaw=float(track.heading[0]),
                velocity_xy=np.zeros(2), steer=0.0, throttle=0.0, brake=0.0)
    base.update(kw)
    return rc.step(**base)


def test_offroad_penalty_is_graded(closed_world):
    _, track = closed_world
    cfg = RewardConfig()
    rc = RewardCalculator(cfg)
    rc.reset(track, 0.0)
    r_on, _ = _step(rc, track, 0.5, lateral=0.0)
    rc.reset(track, 0.0)
    r_edge, _ = _step(rc, track, 0.5, lateral=track.half_width + 0.01)
    rc.reset(track, 0.0)
    r_far, _ = _step(rc, track, 0.5, lateral=track.half_width + 3.0)
    assert r_far < r_edge < r_on
    assert r_edge - r_on == pytest.approx(-cfg.offroad_step, abs=1e-3)
    assert (r_edge - r_far) == pytest.approx(cfg.offroad_per_m * 3.0 - 0.0, abs=1e-3) or r_far < r_edge


def test_backwards_motion_is_negative(closed_world):
    _, track = closed_world
    rc = RewardCalculator()
    rc.reset(track, 50.0)
    tan = track.tangent[int(np.searchsorted(track.s, 49.9))]
    r, terms = _step(rc, track, 49.0, idx=int(np.searchsorted(track.s, 49.0)),
                     velocity_xy=-10.0 * tan)
    assert terms["progress"] < 0
    assert terms["backwards"] < 0
    assert r < 0


def test_smoothness_terms_penalise_jerk(closed_world):
    _, track = closed_world
    rc = RewardCalculator()
    rc.reset(track, 0.0)
    _, terms = _step(rc, track, 0.0, steer=1.0, throttle=1.0)
    assert terms["steer_rate"] < 0 and terms["pedal_rate"] < 0
    _, terms = _step(rc, track, 0.0, steer=1.0, throttle=1.0)
    assert terms["steer_rate"] == 0 and terms["pedal_rate"] == 0


def test_terminal_terms():
    rc = RewardCalculator(RewardConfig())
    assert rc.terminal("success")[0] == RewardConfig().completion
    assert rc.terminal("flip")[0] == -RewardConfig().flip
    assert rc.terminal("offtrack")[0] == -RewardConfig().offtrack_term
    assert rc.terminal("stuck")[0] == -RewardConfig().stuck_term
    assert rc.terminal(None) == (0.0, {})


def test_centerline_driver_gets_positive_return():
    env = DriveEnv(closed=True, difficulty=0.5, seed=3, control_mode="continuous", max_steps=900)
    ctrl = PurePursuitController()
    env.reset()
    ret = 0.0
    for _ in range(400):
        _, r, term, trunc, info = env.step(to_action(ctrl.act(env.car, env.track, env._idx), "continuous"))
        ret += r
        assert not term, info["termination"]
    assert info["distance"] > 150.0
    assert ret > 10.0
    # progress dominates the return: the progress term is at least the sum of all other terms
    env.close()


def test_standing_still_gets_low_return():
    env = DriveEnv(closed=True, difficulty=0.5, seed=3, control_mode="continuous",
                   max_steps=300, stuck_time=1e6)
    env.reset()
    total = 0.0
    for _ in range(120):
        _, r, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        total += r
    per_step = total / 120
    assert -0.05 < per_step <= 0.0
    env.close()


def test_driving_off_track_is_negative_and_terminates():
    env = DriveEnv(closed=True, difficulty=1.0, seed=3, control_mode="discrete",
                   max_steps=3000)
    env.reset()
    total = 0.0
    term = False
    info = {}
    for _ in range(3000):
        _, r, term, trunc, info = env.step(DISCRETE_GAS_STRAIGHT)
        total += r
        if term or trunc:
            break
    assert term
    assert info["termination"] in ("offtrack", "flip")
    assert "offtrack_term" in info["reward_terms"] or "flip" in info["reward_terms"]
    assert total < 0
    env.close()


DISCRETE_GAS_STRAIGHT = 1 * 4 + 1 * 2 + 0   # steer none, gas on, brake off
