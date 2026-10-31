"""Fast tests for drivesim.rl: GAE, normalisers, vector env, a one-update PPO run, weight naming."""
import os
import pickle
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drivesim.rl.ppo import (PPOConfig, PPOTrainer, RunningMeanStd, NumpyPolicy,  # noqa: E402
                             TorchBackend, compute_gae, linear_lr)
from drivesim.rl.vec_env import DummyVecEnv, SubprocVecEnv, env_spec  # noqa: E402


def test_gae_hand_example_terminal():
    # three steps of reward 1, episode ends after step 2, zero values
    r = np.ones((3, 1))
    v = np.zeros((3, 1))
    d = np.array([[False], [False], [True]])
    adv, ret = compute_gae(r, v, d, np.zeros(1), gamma=0.5, lam=1.0)
    # t=2: 1 ; t=1: 1 + 0.5*1 = 1.5 ; t=0: 1 + 0.5*1.5 = 1.75
    np.testing.assert_allclose(adv[:, 0], [1.75, 1.5, 1.0])
    np.testing.assert_allclose(ret, adv)


def test_gae_bootstrap_from_last_value_and_lambda():
    r = np.zeros((2, 1))
    v = np.zeros((2, 1))
    d = np.zeros((2, 1), bool)
    adv, ret = compute_gae(r, v, d, np.array([4.0]), gamma=0.5, lam=1.0)
    # t=1: delta = 0.5*4 = 2 ; t=0: delta = 0, carried = gamma*lam*2 = 1
    np.testing.assert_allclose(adv[:, 0], [1.0, 2.0])
    adv2, _ = compute_gae(r, v, d, np.array([4.0]), gamma=0.5, lam=0.0)
    # lambda = 0 gives one-step TD errors only
    np.testing.assert_allclose(adv2[:, 0], [0.0, 2.0])


def test_gae_values_contribute():
    r = np.array([[1.0], [1.0]])
    v = np.array([[0.5], [0.5]])
    d = np.array([[False], [True]])
    adv, ret = compute_gae(r, v, d, np.zeros(1), gamma=1.0, lam=0.95)
    # t=1: delta = 1 - 0.5 = 0.5 ; t=0: delta = 1 + 0.5 - 0.5 = 1.0, plus 0.95 * 0.5
    np.testing.assert_allclose(adv[1, 0], 0.5)
    np.testing.assert_allclose(adv[0, 0], 1.0 + 0.95 * 0.5)
    np.testing.assert_allclose(ret, adv + v)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(500, 4))
    rms = RunningMeanStd((4,), eps=1e-8)
    for chunk in np.array_split(x, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, x.mean(0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rms.var, x.var(0), rtol=1e-5)


def test_linear_lr_schedule():
    cfg = PPOConfig(lr=1e-3, lr_final_frac=0.1, total_steps=100)
    assert linear_lr(cfg, 0) == pytest.approx(1e-3)
    assert linear_lr(cfg, 50) == pytest.approx(5.5e-4)
    assert linear_lr(cfg, 1000) == pytest.approx(1e-4)


def _spec():
    return env_spec(closed=False, difficulty=0.0, pool_size=1, max_steps=5)


def test_dummy_vec_env_two_envs_few_steps():
    venv = DummyVecEnv(_spec(), 2, seed=0)
    assert venv.initial_obs.shape == (2, venv.observation_dim)
    a = np.zeros((2, 3), np.float32)
    a[:, 1] = 0.5
    for _ in range(3):
        obs, rew, term, trunc, infos = venv.step(a)
    assert obs.shape == (2, venv.observation_dim) and np.all(np.isfinite(obs))
    assert rew.shape == (2,) and term.dtype == bool and trunc.dtype == bool
    venv.close()


def test_subproc_vec_env_two_envs_with_truncation_and_final_obs():
    venv = SubprocVecEnv(_spec(), 2, seed=1)
    try:
        assert venv.initial_obs.shape == (2, venv.observation_dim)
        a = np.zeros((2, 3), np.float32)
        a[:, 1] = 0.5
        seen_final = False
        for _ in range(7):           # max_steps=5 forces a truncation
            obs, rew, term, trunc, infos = venv.step(a)
            if trunc.any():
                seen_final = True
                assert all("final_obs" in infos[i] for i in np.where(trunc)[0])
        assert seen_final
        assert obs.shape == (2, venv.observation_dim)
    finally:
        venv.close()


def _tiny_cfg(**kw):
    base = dict(num_envs=2, rollout_len=16, epochs=1, minibatches=2, hidden=16, layers=1,
                total_steps=32, seed=0)
    base.update(kw)
    return PPOConfig(**base)


def test_one_update_ppo_smoke_torch(tmp_path):
    t0 = time.time()
    cfg = _tiny_cfg()
    venv = DummyVecEnv(_spec(), cfg.num_envs, seed=0)
    backend = TorchBackend(venv.observation_dim, 3, venv.action_space.low, venv.action_space.high, cfg)
    trainer = PPOTrainer(cfg, venv, backend)
    stats = trainer.train_iteration()
    venv.close()
    for k in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clipfrac", "lr"):
        assert np.isfinite(stats[k]), k
    assert stats["total_steps"] == cfg.batch_size() == 32
    assert time.time() - t0 < 30


def test_numpy_policy_matches_torch_actor(tmp_path):
    cfg = _tiny_cfg(squash="tanh")
    venv = DummyVecEnv(_spec(), cfg.num_envs, seed=0)
    backend = TorchBackend(venv.observation_dim, 3, venv.action_space.low, venv.action_space.high, cfg)
    trainer = PPOTrainer(cfg, venv, backend)
    trainer.train_iteration()
    venv.close()
    ck = {"weights": backend.get_weights(), "obs_rms": trainer.obs_rms.state(), "cfg": cfg.as_dict()}
    path = tmp_path / "ck.pt"
    with open(path, "wb") as f:
        pickle.dump(ck, f)
    from drivesim.rl.ppo import load_policy
    pol = load_policy(str(path))
    obs = venv.initial_obs[0]
    low, high = venv.action_space.low, venv.action_space.high
    a_np = pol.act(obs, low, high)
    # torch reference: normalise the same way, then the actor mean through tanh squash
    import torch
    from drivesim.rl.ppo import normalize_obs, deterministic_action
    on = normalize_obs(obs[None], trainer.obs_rms.mean, trainer.obs_rms.var, cfg.clip_obs)
    with torch.no_grad():
        mu = backend.actor(torch.as_tensor(on)).numpy()[0]
    np.testing.assert_allclose(a_np, deterministic_action(mu, low, high, "tanh"), rtol=1e-4, atol=1e-5)


def test_tanh_logp_matches_numerical_jacobian():
    import torch
    cfg = _tiny_cfg(squash="tanh")
    low, high = np.array([-1, 0, 0], np.float32), np.ones(3, np.float32)
    b = TorchBackend(56, 3, low, high, cfg)
    u = torch.tensor([[0.3, -0.7, 1.2]])
    mean = torch.zeros(1, 3)
    std = torch.ones(1, 3)
    lp = b._logp(u, mean, std)
    # change of variables: density of a = f(u) is N(u) / |det df/du| with f = low + (high-low)(tanh+1)/2
    eps = 1e-6
    jac = np.prod([(high[i] - low[i]) / 2 * (1 - np.tanh(u.numpy()[0, i]) ** 2) for i in range(3)])
    base = torch.distributions.Normal(mean, std).log_prob(u).sum().item()
    assert lp.item() == pytest.approx(base - np.log(jac), rel=1e-4)


def test_mlx_backend_one_update_if_available():
    pytest.importorskip("mlx.core")
    from drivesim.rl.ppo_mlx import MLXBackend
    cfg = _tiny_cfg()
    venv = DummyVecEnv(_spec(), cfg.num_envs, seed=0)
    backend = MLXBackend(venv.observation_dim, 3, venv.action_space.low, venv.action_space.high, cfg)
    trainer = PPOTrainer(cfg, venv, backend)
    stats = trainer.train_iteration()
    venv.close()
    assert np.isfinite(stats["pg_loss"]) and np.isfinite(stats["approx_kl"])
    w = backend.get_weights()
    assert w["actor.0.weight"].shape == (16, venv.observation_dim) and w["log_std"].shape == (3,)


def test_export_npz_roundtrip(tmp_path):
    import pickle
    import numpy as np
    from drivesim.rl.ppo import export_npz, numpy_mlp

    ck = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "ab_policy.pt"), "rb"))
    path = str(tmp_path / "p.npz")
    export_npz(ck, path)
    z = np.load(path)
    x = np.random.default_rng(0).normal(size=(3, ck["obs_dim"])).astype(np.float32)
    w_pkl = numpy_mlp(ck["weights"], "actor", x, int(ck["cfg"]["layers"]))
    w_npz = numpy_mlp({k: z[k] for k in ck["weights"]}, "actor", x, int(ck["cfg"]["layers"]))
    assert np.allclose(w_pkl, w_npz, atol=1e-6)
