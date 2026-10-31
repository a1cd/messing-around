"""Compact PPO for :class:`drivesim.env.DriveEnv` (continuous actions).

Structure
---------
* Shared numpy code (used by every backend): :class:`PPOConfig`, :class:`RunningMeanStd`,
  :func:`compute_gae`, :func:`numpy_mlp` and the rollout / update loop in :class:`PPOTrainer`.
* Backends implement ``act``, ``value``, ``train_minibatch``, ``set_lr``, ``get_weights``,
  ``get_optimizer_state`` and ``set_optimizer_state``. :class:`TorchBackend` is here;
  :mod:`drivesim.rl.ppo_mlx` provides the same interface on MLX.

Policy: Gaussian with a state-independent log-std, actor and critic are separate MLPs.
``squash='clip'`` (default) samples ``u ~ N(mu, sigma)`` and sends ``clip(u)`` to the env. The
log-prob uses the unclipped ``u``. ``squash='tanh'`` maps ``u`` into the action box with
``low + (high-low) * (tanh(u)+1)/2`` and includes the Jacobian in the log-prob. Both store
``u`` in the rollout, so the policy ratio is exact.

Weight names are shared by all backends: ``actor.{i}.weight``, ``actor.{i}.bias``,
``critic.{i}.weight``, ``critic.{i}.bias`` (PyTorch Linear layout, out x in), ``log_std``.
A checkpoint can therefore be run with numpy alone (:func:`numpy_mlp`), with no torch or MLX.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class PPOConfig:
    # --- rollout / batch: batch = num_envs * rollout_len
    num_envs: int = 8
    rollout_len: int = 256
    total_steps: int = 2_000_000
    # --- optimisation
    epochs: int = 8
    minibatches: int = 8
    lr: float = 3e-4
    lr_final_frac: float = 0.0          # linear decay from lr to lr * lr_final_frac
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None   # stop the epochs early when approx KL exceeds 1.5x this
    adam_eps: float = 1e-5
    # --- returns / advantages
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    adv_norm: bool = True
    reward_norm: bool = True            # divide rewards by the std of the discounted return
    # --- losses
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    # --- networks
    hidden: int = 64
    layers: int = 2
    init_log_std: float = -0.5
    init_act_bias: Optional[list] = None  # actor output bias at init, one per action dim
    squash: str = "clip"                # 'clip' or 'tanh'
    obs_norm: bool = True
    clip_obs: float = 10.0
    # --- misc
    seed: int = 0
    backend: str = "torch"              # 'torch' or 'mlx'

    def batch_size(self) -> int:
        return self.num_envs * self.rollout_len

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- numpy pieces
class RunningMeanStd:
    """Running mean and variance (Chan et al. parallel update)."""

    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load(self, st: dict) -> None:
        self.mean = np.asarray(st["mean"], np.float64).copy()
        self.var = np.asarray(st["var"], np.float64).copy()
        self.count = float(st["count"])


def normalize_obs(obs: np.ndarray, mean: np.ndarray, var: np.ndarray, clip_obs: float) -> np.ndarray:
    return np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip_obs, clip_obs).astype(np.float32)


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """Generalised advantage estimation for arrays shaped (T, N).

    ``dones[t]`` marks the end of an episode after step t (terminated or truncated). The next
    value is then cut off, so a truncated episode must already carry its bootstrap value in
    ``rewards``. Returns ``(advantages, returns)`` with ``returns = advantages + values``.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1:], np.float64)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t].astype(np.float64)
        next_v = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def explained_variance(y_pred: np.ndarray, y: np.ndarray) -> float:
    var_y = np.var(y)
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(y - y_pred) / var_y)


def linear_lr(cfg: PPOConfig, steps_done: int) -> float:
    frac = min(steps_done / max(cfg.total_steps, 1), 1.0)
    return cfg.lr * (1.0 - frac * (1.0 - cfg.lr_final_frac))


def numpy_mlp(weights: dict, prefix: str, x: np.ndarray, n_layers: int) -> np.ndarray:
    """Forward pass of ``n_layers`` tanh hidden layers plus a linear output, in numpy."""
    h = x
    for i in range(n_layers):
        h = np.tanh(h @ weights[f"{prefix}.{i}.weight"].T + weights[f"{prefix}.{i}.bias"])
    return h @ weights[f"{prefix}.{n_layers}.weight"].T + weights[f"{prefix}.{n_layers}.bias"]


def to_env_action(u: np.ndarray, low: np.ndarray, high: np.ndarray, squash: str) -> np.ndarray:
    if squash == "tanh":
        return (low + (high - low) * (np.tanh(u) + 1.0) / 2.0).astype(np.float32)
    return np.clip(u, low, high).astype(np.float32)


def deterministic_action(mean: np.ndarray, low, high, squash: str) -> np.ndarray:
    return to_env_action(mean, low, high, squash)


# --------------------------------------------------------------------------- trainer
class PPOTrainer:
    """Rollout collection, GAE and minibatch PPO updates. Backend-agnostic."""

    def __init__(self, cfg: PPOConfig, venv, backend):
        self.cfg = cfg
        self.venv = venv
        self.backend = backend
        self.n = venv.num_envs
        self.obs_dim = venv.observation_dim
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self.ret_acc = np.zeros(self.n)
        self.rng = np.random.default_rng(cfg.seed)
        self.obs = venv.initial_obs.astype(np.float32)
        self.ep_ret = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, np.int64)
        self.total_steps = 0
        self.updates = 0
        self.recent_eps = deque(maxlen=200)     # dicts: return, length, success, termination
        self.last_finished = []                 # episodes finished in the last rollout
        self._t0 = time.time()
        self._steps_at_start = 0

    # observations
    def _norm(self, obs):
        if not self.cfg.obs_norm:
            return obs.astype(np.float32)
        return normalize_obs(obs, self.obs_rms.mean, self.obs_rms.var, self.cfg.clip_obs)

    # rollout
    def collect(self) -> dict:
        cfg = self.cfg
        T, N, D = cfg.rollout_len, self.n, self.obs_dim
        A = self.backend.act_dim
        buf_obs = np.zeros((T, N, D), np.float32)
        buf_u = np.zeros((T, N, A), np.float32)
        buf_logp = np.zeros((T, N), np.float32)
        buf_val = np.zeros((T, N), np.float32)
        buf_rew = np.zeros((T, N), np.float64)
        buf_done = np.zeros((T, N), bool)
        self.last_finished = []
        for t in range(T):
            if cfg.obs_norm:
                self.obs_rms.update(self.obs)
            on = self._norm(self.obs)
            u, a_env, logp, v = self.backend.act(on)
            obs2, r, term, trunc, infos = self.venv.step(a_env)
            done = term | trunc
            r_s = r.copy()
            if cfg.reward_norm:
                self.ret_acc = self.ret_acc * cfg.gamma + r
                self.ret_rms.update(self.ret_acc)
                r_s = r / math.sqrt(float(self.ret_rms.var) + 1e-8)
                self.ret_acc[done] = 0.0
            # time-limit truncation: bootstrap from the value of the final observation
            tr_idx = np.where(trunc & ~term)[0]
            if len(tr_idx):
                finals = np.stack([infos[i]["final_obs"] for i in tr_idx]).astype(np.float32)
                vf = self.backend.value(self._norm(finals))
                r_s[tr_idx] += cfg.gamma * vf
            buf_obs[t], buf_u[t], buf_logp[t], buf_val[t] = on, u, logp, v
            buf_rew[t], buf_done[t] = r_s, done
            self.ep_ret += r
            self.ep_len += 1
            for i in np.where(done)[0]:
                ep = {"return": float(self.ep_ret[i]), "length": int(self.ep_len[i]),
                      "success": bool(infos[i]["success"]),
                      "termination": infos[i]["termination"] or "truncated"}
                self.recent_eps.append(ep)
                self.last_finished.append(ep)
                self.ep_ret[i] = 0.0
                self.ep_len[i] = 0
            self.obs = obs2.astype(np.float32)
            self.total_steps += N
        last_v = self.backend.value(self._norm(self.obs))
        adv, ret = compute_gae(buf_rew, buf_val.astype(np.float64), buf_done,
                               last_v.astype(np.float64), cfg.gamma, cfg.gae_lambda)
        return {"obs": buf_obs, "u": buf_u, "logp": buf_logp, "val": buf_val,
                "adv": adv.astype(np.float32), "ret": ret.astype(np.float32)}

    # update
    def update(self, batch: dict, steps_done: int) -> dict:
        cfg = self.cfg
        lr = linear_lr(cfg, steps_done)
        self.backend.set_lr(lr)
        obs = batch["obs"].reshape(-1, self.obs_dim)
        u = batch["u"].reshape(-1, self.backend.act_dim)
        logp = batch["logp"].reshape(-1)
        adv = batch["adv"].reshape(-1).astype(np.float64)
        ret = batch["ret"].reshape(-1)
        val = batch["val"].reshape(-1)
        ev = explained_variance(val, ret)
        if cfg.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.astype(np.float32)
        B = obs.shape[0]
        stats = {k: [] for k in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clipfrac")}
        stop, epochs_run = False, 0
        for _ in range(cfg.epochs):
            epochs_run += 1
            perm = self.rng.permutation(B)
            for idx in np.array_split(perm, cfg.minibatches):
                s = self.backend.train_minibatch(obs[idx], u[idx], logp[idx], adv[idx], ret[idx],
                                                 cfg)
                for k in stats:
                    stats[k].append(s[k])
            if cfg.target_kl is not None and np.mean(stats["approx_kl"][-cfg.minibatches:]) > 1.5 * cfg.target_kl:
                stop = True
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update(lr=lr, explained_var=ev, epochs_run=epochs_run, early_stop=stop)
        self.updates += 1
        return out

    def train_iteration(self, steps_target: Optional[int] = None) -> dict:
        """One rollout plus one update. Returns a flat dict of stats for logging."""
        t0 = time.time()
        batch = self.collect()
        t_roll = time.time() - t0
        stats = self.update(batch, self.total_steps)
        ep = list(self.recent_eps)
        stats.update(
            update=self.updates, total_steps=self.total_steps,
            ep_return=float(np.mean([e["return"] for e in ep])) if ep else float("nan"),
            ep_length=float(np.mean([e["length"] for e in ep])) if ep else float("nan"),
            success_rate=float(np.mean([e["success"] for e in ep])) if ep else float("nan"),
            n_episodes=len(ep), t_rollout=t_roll, t_total=time.time() - t0,
            fps=self.cfg.batch_size() / max(time.time() - t0, 1e-9),
            wall_s=time.time() - self._t0,
        )
        return stats

    # checkpoint
    def state(self) -> dict:
        return {"updates": self.updates, "total_steps": self.total_steps,
                "obs_rms": self.obs_rms.state(), "ret_rms": self.ret_rms.state(),
                "rng": self.rng.bit_generator.state}

    def load_state(self, st: dict) -> None:
        self.updates = int(st["updates"])
        self.total_steps = int(st["total_steps"])
        self.obs_rms.load(st["obs_rms"])
        self.ret_rms.load(st["ret_rms"])
        self.rng.bit_generator.state = st["rng"]


# --------------------------------------------------------------------------- torch backend
class TorchBackend:
    """Actor-critic MLPs and Adam in PyTorch (CPU is enough)."""

    name = "torch"

    def __init__(self, obs_dim: int, act_dim: int, low, high, cfg: PPOConfig):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.act_dim = act_dim
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.squash = cfg.squash
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        def mlp(out_dim, last_gain):
            layers, d = [], obs_dim
            for _ in range(cfg.layers):
                lin = nn.Linear(d, cfg.hidden)
                nn.init.orthogonal_(lin.weight, math.sqrt(2))
                nn.init.zeros_(lin.bias)
                layers += [lin, nn.Tanh()]
                d = cfg.hidden
            lin = nn.Linear(d, out_dim)
            nn.init.orthogonal_(lin.weight, last_gain)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            return nn.Sequential(*layers)

        self.actor = mlp(act_dim, 0.01)
        self.critic = mlp(1, 1.0)
        if cfg.init_act_bias is not None:
            with torch.no_grad():
                self.actor[-1].bias.copy_(torch.as_tensor(cfg.init_act_bias, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(cfg.init_log_std)))
        params = list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std]
        self.params = params
        self.opt = torch.optim.Adam(params, lr=cfg.lr, eps=cfg.adam_eps)
        self._half_range = torch.as_tensor(np.log((self.high - self.low) / 2.0), dtype=torch.float32)

    def _logp(self, u, mean, std):
        torch = self.torch
        lp = torch.distributions.Normal(mean, std).log_prob(u).sum(-1)
        if self.squash == "tanh":
            # log|da/du| = sum_i [log(range_i/2) + log(1 - tanh^2 u_i)],  log(1-tanh^2 u)=2(log2-u-softplus(-2u))
            log1m = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
            lp = lp - (self._half_range + log1m).sum(-1)
        return lp

    def act(self, obs: np.ndarray):
        torch = self.torch
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            mean = self.actor(o)
            std = self.log_std.exp().expand_as(mean)
            u = mean + std * torch.randn_like(mean)
            logp = self._logp(u, mean, std)
            v = self.critic(o).squeeze(-1)
        u_np = u.numpy()
        return u_np, to_env_action(u_np, self.low, self.high, self.squash), logp.numpy(), v.numpy()

    def value(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.critic(self.torch.as_tensor(obs, dtype=self.torch.float32)).squeeze(-1).numpy()

    def train_minibatch(self, obs, u, logp_old, adv, ret, cfg: PPOConfig) -> dict:
        torch = self.torch
        o = torch.as_tensor(obs, dtype=torch.float32)
        ut = torch.as_tensor(u, dtype=torch.float32)
        lo = torch.as_tensor(logp_old, dtype=torch.float32)
        at = torch.as_tensor(adv, dtype=torch.float32)
        rt = torch.as_tensor(ret, dtype=torch.float32)
        mean = self.actor(o)
        std = self.log_std.exp().expand_as(mean)
        logp = self._logp(ut, mean, std)
        logratio = logp - lo
        ratio = logratio.exp()
        pg = -torch.min(ratio * at, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * at).mean()
        v = self.critic(o).squeeze(-1)
        vf = 0.5 * ((v - rt) ** 2).mean()
        ent = (0.5 * math.log(2 * math.pi * math.e) + self.log_std).sum()   # state-independent
        loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
        self.opt.step()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item()
            clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
        return {"pg_loss": pg.item(), "vf_loss": vf.item(), "entropy": ent.item(),
                "approx_kl": approx_kl, "clipfrac": clipfrac}

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def get_weights(self) -> dict:
        w = {}
        for prefix, net in (("actor", self.actor), ("critic", self.critic)):
            for k, v in net.state_dict().items():
                # Sequential index 2j is linear layer j (each hidden layer is Linear + Tanh)
                idx, param = k.split(".")
                w[f"{prefix}.{int(idx) // 2}.{param}"] = v.detach().cpu().numpy().copy()
        w["log_std"] = self.log_std.detach().cpu().numpy().copy()
        return w

    def set_weights(self, w: dict) -> None:
        torch = self.torch
        with torch.no_grad():
            for prefix, net in (("actor", self.actor), ("critic", self.critic)):
                sd = {}
                for k in w:
                    if k.startswith(prefix + "."):
                        j, param = k[len(prefix) + 1:].split(".")
                        sd[f"{2 * int(j)}.{param}"] = torch.as_tensor(w[k])
                net.load_state_dict(sd)
            self.log_std.copy_(torch.as_tensor(w["log_std"]))

    def get_optimizer_state(self) -> dict:
        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, list):
                return [conv(v) for v in x]
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy().copy()
            return x
        return conv(self.opt.state_dict())

    def set_optimizer_state(self, st: dict) -> None:
        torch = self.torch

        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, list):
                return [conv(v) for v in x]
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x)
            return x
        self.opt.load_state_dict(conv(st))


# --------------------------------------------------------------------------- inference
class NumpyPolicy:
    """Deterministic policy from a checkpoint, numpy only (no torch or MLX needed)."""

    def __init__(self, ck: dict):
        self.w = ck["weights"]
        self.cfg = ck["cfg"]
        self.n_layers = int(self.cfg["layers"])
        self.squash = self.cfg["squash"]
        self.clip_obs = float(self.cfg["clip_obs"])
        self.obs_norm = bool(self.cfg["obs_norm"])
        self.mean = np.asarray(ck["obs_rms"]["mean"], np.float64)
        self.var = np.asarray(ck["obs_rms"]["var"], np.float64)

    def act(self, obs: np.ndarray, low, high) -> np.ndarray:
        x = normalize_obs(obs[None], self.mean, self.var, self.clip_obs) if self.obs_norm \
            else obs[None].astype(np.float32)
        mu = numpy_mlp(self.w, "actor", x, self.n_layers)[0]
        return deterministic_action(mu, low, high, self.squash)


def load_policy(path: str) -> NumpyPolicy:
    import pickle
    with open(path, "rb") as f:
        return NumpyPolicy(pickle.load(f))


# --------------------------------------------------------------------------- export
def export_npz(ck: dict, path: str) -> None:
    """Write a checkpoint's weights and observation normaliser as a plain ``.npz`` (no pickle).

    Keys: the weight names above, ``obs_mean``, ``obs_var``, and ``cfg_json`` (the PPO config as
    a JSON string). Load with ``np.load(path)``; :func:`numpy_mlp` takes the weight dict directly.
    """
    import json
    arrays = {k: np.asarray(v, np.float32) for k, v in ck["weights"].items()}
    arrays["obs_mean"] = np.asarray(ck["obs_rms"]["mean"], np.float32)
    arrays["obs_var"] = np.asarray(ck["obs_rms"]["var"], np.float32)
    arrays["cfg_json"] = np.array(json.dumps(ck["cfg"]))
    np.savez(path, **arrays)
