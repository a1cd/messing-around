"""MLX backend for :mod:`drivesim.rl.ppo` (Apple Silicon; also runs on Linux/CPU).

Only the network and optimiser differ from the torch backend. Rollouts, GAE, normalisers, the
minibatch loop and the checkpoint format are shared (see :class:`drivesim.rl.ppo.PPOTrainer`).
Weights use the same names and layout as the torch backend (``actor.{j}.weight`` with shape
(out, in), ``log_std``), so a checkpoint from either backend loads in numpy.

Selected with ``--backend mlx`` in ``scripts/train.py``. If MLX cannot be imported the script
falls back to torch. Tested here on Linux with ``pip install "mlx[cpu]"``. The Apple GPU
(Metal) path is the same code but has not been run on macOS.
"""
from __future__ import annotations

import math

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .ppo import PPOConfig, to_env_action


def _orthogonal(rng, out_dim: int, in_dim: int, gain: float) -> np.ndarray:
    a = rng.standard_normal((max(out_dim, in_dim), min(out_dim, in_dim)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if out_dim < in_dim:
        q = q.T
    return (gain * q[:out_dim, :in_dim]).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_layers, last_gain, rng):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers + [out_dim]
        self.layers = []
        for j in range(len(dims) - 1):
            lin = nn.Linear(dims[j], dims[j + 1])
            gain = last_gain if j == len(dims) - 2 else math.sqrt(2)
            lin.weight = mx.array(_orthogonal(rng, dims[j + 1], dims[j], gain))
            lin.bias = mx.zeros((dims[j + 1],))
            self.layers.append(lin)

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = mx.tanh(lin(x))
        return self.layers[-1](x)


class _Net(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig):
        super().__init__()
        rng = np.random.default_rng(cfg.seed)
        self.actor = _MLP(obs_dim, act_dim, cfg.hidden, cfg.layers, 0.01, rng)
        self.critic = _MLP(obs_dim, 1, cfg.hidden, cfg.layers, 1.0, rng)
        self.log_std = mx.full((act_dim,), float(cfg.init_log_std))


class MLXBackend:
    name = "mlx"

    def __init__(self, obs_dim: int, act_dim: int, low, high, cfg: PPOConfig):
        mx.random.seed(cfg.seed)
        self.act_dim = act_dim
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.squash = cfg.squash
        self.cfg = cfg
        self.net = _Net(obs_dim, act_dim, cfg)
        if cfg.init_act_bias is not None:
            self.net.actor.layers[-1].bias = mx.array(np.asarray(cfg.init_act_bias, np.float32))
        self.opt = optim.Adam(learning_rate=cfg.lr, eps=cfg.adam_eps)
        self._half_range = mx.array(np.log((self.high - self.low) / 2.0).astype(np.float32))
        self._vg = nn.value_and_grad(self.net, self._loss)

    # distribution helpers ---------------------------------------------------
    def _logp(self, u, mean, log_std):
        std = mx.exp(log_std)
        z = (u - mean) / std
        lp = mx.sum(-0.5 * z * z - log_std - 0.5 * math.log(2 * math.pi), axis=-1)
        if self.squash == "tanh":
            log1m = 2.0 * (math.log(2.0) - u - nn.softplus(-2.0 * u))
            lp = lp - mx.sum(self._half_range + log1m, axis=-1)
        return lp

    def act(self, obs: np.ndarray):
        o = mx.array(obs)
        mean = self.net.actor(o)
        std = mx.exp(self.net.log_std)
        u = mean + std * mx.random.normal(mean.shape)
        logp = self._logp(u, mean, self.net.log_std)
        v = self.net.critic(o).squeeze(-1)
        mx.eval(u, logp, v)
        u_np = np.array(u, dtype=np.float32)
        return u_np, to_env_action(u_np, self.low, self.high, self.squash), np.array(logp), np.array(v)

    def value(self, obs: np.ndarray) -> np.ndarray:
        v = self.net.critic(mx.array(obs)).squeeze(-1)
        mx.eval(v)
        return np.array(v)

    # training ----------------------------------------------------------------
    def _loss(self, obs, u, logp_old, adv, ret):
        cfg = self.cfg
        mean = self.net.actor(obs)
        logp = self._logp(u, mean, self.net.log_std)
        ratio = mx.exp(logp - logp_old)
        pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))
        v = self.net.critic(obs).squeeze(-1)
        vf = 0.5 * mx.mean((v - ret) ** 2)
        ent = mx.sum(0.5 * math.log(2 * math.pi * math.e) + self.net.log_std)
        return pg + cfg.vf_coef * vf - cfg.ent_coef * ent

    def train_minibatch(self, obs, u, logp_old, adv, ret, cfg: PPOConfig) -> dict:
        args = [mx.array(x) for x in (obs, u, logp_old, adv, ret)]
        loss, grads = self._vg(*args)
        grads, _ = optim.clip_grad_norm(grads, cfg.max_grad_norm)
        self.opt.update(self.net, grads)
        mx.eval(self.net.parameters(), self.opt.state)
        # statistics on the same minibatch, with the updated network (approximate)
        o, ut, lo, at, rt = args
        mean = self.net.actor(o)
        logp = self._logp(ut, mean, self.net.log_std)
        logratio = logp - lo
        ratio = mx.exp(logratio)
        v = self.net.critic(o).squeeze(-1)
        vals = [mx.mean(-(mx.minimum(ratio * at, mx.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * at))),
                0.5 * mx.mean((v - rt) ** 2),
                mx.sum(0.5 * math.log(2 * math.pi * math.e) + self.net.log_std),
                mx.mean((ratio - 1) - logratio),
                mx.mean((mx.abs(ratio - 1) > cfg.clip).astype(mx.float32))]
        mx.eval(*vals)
        pg, vf, ent, kl, cf = [float(x) for x in vals]
        return {"pg_loss": pg, "vf_loss": vf, "entropy": ent, "approx_kl": kl, "clipfrac": cf}

    def set_lr(self, lr: float) -> None:
        self.opt.learning_rate = float(lr)

    # checkpoints -------------------------------------------------------------
    def get_weights(self) -> dict:
        w = {}
        for k, v in tree_flatten(self.net.parameters()):
            parts = k.split(".")
            if parts[0] == "log_std":
                w["log_std"] = np.array(v, np.float32)
            else:
                # e.g. actor.layers.0.weight -> actor.0.weight
                w[f"{parts[0]}.{parts[2]}.{parts[3]}"] = np.array(v, np.float32)
        return w

    def set_weights(self, w: dict) -> None:
        flat = []
        for k in ("actor", "critic"):
            for name, arr in w.items():
                if name.startswith(k + "."):
                    j, param = name[len(k) + 1:].split(".")
                    flat.append((f"{k}.layers.{j}.{param}", mx.array(arr)))
        flat.append(("log_std", mx.array(w["log_std"])))
        self.net.update(tree_unflatten(flat))
        mx.eval(self.net.parameters())

    def get_optimizer_state(self) -> dict:
        return {k: np.array(v) for k, v in tree_flatten(self.opt.state)}

    def set_optimizer_state(self, st: dict) -> None:
        self.opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st.items()])
