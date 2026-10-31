"""Gymnasium environment tests (drivesim/env.py)."""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import drivesim.env as envmod
from drivesim.baselines import PurePursuitController, run_episode
from drivesim.env import DISCRETE_ACTIONS, OBS_DIM, OBS_NAMES, DriveEnv, build_world


@pytest.mark.parametrize("mode", ["discrete", "continuous", "full"])
def test_gymnasium_env_checker(mode):
    env = DriveEnv(closed=True, difficulty=0.3, seed=2, control_mode=mode, max_steps=50)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_registered_id():
    env = gym.make("Drive-v0", closed=False, difficulty=0.2, seed=1, control_mode="continuous")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    env.close()


def test_spaces_per_mode():
    d = DriveEnv(control_mode="discrete")
    c = DriveEnv(control_mode="continuous")
    f = DriveEnv(control_mode="full")
    assert d.action_space.n == 12 and len(DISCRETE_ACTIONS) == 12
    assert c.action_space.shape == (3,) and f.action_space.shape == (7,)
    assert f.transmission == "manual" and c.transmission == "auto"
    assert d.observation_space.shape == (OBS_DIM,) == (len(OBS_NAMES),)
    assert c.observation_space.dtype == np.float32
    with pytest.raises(ValueError):
        DriveEnv(control_mode="continuous", transmission="manual")


def test_observation_is_finite_and_bounded():
    env = DriveEnv(closed=False, difficulty=0.5, seed=4, control_mode="full", max_steps=400)
    obs, _ = env.reset()
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs))
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 10.0 + 1e-6)
        assert np.isfinite(r)
        if term or trunc:
            break
    env.close()


def test_discrete_table_decoding():
    env = DriveEnv(control_mode="discrete")
    assert env._decode(0)[:3] == (1.0, 0.0, 0.0)    # left, no gas, no brake
    assert env._decode(11)[:3] == (-1.0, 1.0, 1.0)  # right, gas and brake
    assert env._decode(6)[:3] == (0.0, 1.0, 0.0)    # straight, gas
    env.close()


def test_seed_determinism_and_fresh_tracks():
    a = DriveEnv(closed=True, difficulty=0.5, seed=7, control_mode="continuous")
    b = DriveEnv(closed=True, difficulty=0.5, seed=7, control_mode="continuous")
    oa, _ = a.reset()
    ob, _ = b.reset()
    assert np.array_equal(oa, ob)
    for act in ([0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]):
        ra = a.step(np.array(act, np.float32))
        rb = b.step(np.array(act, np.float32))
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]
    # seed=None: each reset draws a new track
    c = DriveEnv(closed=True, difficulty=0.5, seed=None, control_mode="continuous")
    c.reset(seed=1)
    s1 = c.track_seed
    c.reset()
    assert c.track_seed != s1
    for e in (a, b, c):
        e.close()


def test_fixed_world_and_world_fn():
    world = build_world(5, closed=True, difficulty=0.2)
    env = DriveEnv(world=world, control_mode="continuous")
    env.reset()
    assert env.track is world[1]
    calls = []

    def fn(seed):
        calls.append(seed)
        return build_world(seed, closed=False, difficulty=0.0)

    env2 = DriveEnv(world_fn=fn, seed=11, control_mode="continuous", closed=False)
    env2.reset()
    assert calls == [11] and not env2.track.closed
    env.close()
    env2.close()


def test_truncation_at_max_steps():
    env = DriveEnv(closed=True, difficulty=0.5, seed=2, control_mode="continuous", max_steps=15)
    env.reset()
    for i in range(15):
        _, _, term, trunc, _ = env.step(np.zeros(3, np.float32))
    assert trunc and not term


def test_stuck_termination_when_idle():
    env = DriveEnv(closed=True, difficulty=0.5, seed=2, control_mode="continuous",
                   stuck_time=2.0, max_steps=3000)
    env.reset()
    for i in range(200):
        _, _, term, trunc, info = env.step(np.zeros(3, np.float32))
        if term:
            break
    assert term and info["termination"] == "stuck"
    assert i < 2.0 * 30 + 10
    env.close()


def test_auto_reverse_is_blocked_at_standstill():
    env = DriveEnv(closed=True, difficulty=0.5, seed=2, control_mode="continuous", max_steps=3000)
    env.reset()
    cleared = 0
    for _ in range(90):          # 3 s of brake at standstill would toggle reverse after 1 s
        _, _, term, _, info = env.step(np.array([0.0, 0.0, 1.0], np.float32))
        cleared += info["brake_cleared"]
        assert not term
    assert cleared > 0
    assert env.car.gear != -1
    env.close()


def test_manual_shift_edges():
    env = DriveEnv(closed=True, difficulty=0.5, seed=2, control_mode="full", max_steps=3000)
    env.reset()
    env.step(np.array([0, 0, 0, 1, 0, 0, 0], np.float32))           # clutch in
    env.step(np.array([0, 0, 0, 1, 0, 1, 0], np.float32))           # shift up request
    for _ in range(20):
        env.step(np.array([0, 0, 0, 1, 0, 0, 0], np.float32))
    assert env.car.gear == 1
    env.step(np.array([0, 0, 0, 1, 0, 1, 0], np.float32))           # held for two steps:
    env.step(np.array([0, 0, 0, 1, 0, 1, 0], np.float32))           # only the first is an edge
    for _ in range(20):
        env.step(np.array([0, 0, 0, 1, 0, 0, 0], np.float32))
    assert env.car.gear == 2
    env.close()


def test_respawn_keeps_episode_running():
    env = DriveEnv(closed=False, difficulty=0.3, seed=2, control_mode="continuous")
    env.reset()
    env.respawn_at(100.0)
    s = env._pr.s
    assert abs(s - 100.0) < 1.0
    _, _, term, _, _ = env.step(np.zeros(3, np.float32))
    assert not term
    env.close()


def test_open_track_success_with_baseline():
    env = DriveEnv(closed=False, difficulty=0.0, seed=0, control_mode="continuous", max_steps=3000)
    res = run_episode(env, PurePursuitController())
    assert res["success"], res
    assert res["termination"] == "success"
    env.close()


def test_render_without_viewer_raises_clearly():
    env = DriveEnv(render_mode="rgb_array", seed=1)
    env.reset()
    try:
        import drivesim.viewer  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="Viewer"):
            env.render()
    env.close()
