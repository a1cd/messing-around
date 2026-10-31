"""Vectorised DriveEnv: one subprocess per env (spawn start method) or in-process.

Everything that crosses the process boundary is a plain dict/tuple of picklable values.
Environments are built by the top-level :func:`make_env` from a spec dict, so workers
work with the ``spawn`` start method used on macOS and Linux. Scripts that use the
subprocess env must guard their entry point with ``if __name__ == "__main__":``.

Auto-reset: when an episode ends, the worker resets the env and returns the new first
observation in ``obs``. The last observation of the finished episode is in
``infos[i]["final_obs"]`` (needed to bootstrap truncated episodes).

World sampling: the env is built with a :class:`WorldSampler` as its ``world_fn``. With
``pool_size=0`` every reset generates a fresh track from the env RNG. With ``pool_size=P``
the track seed is reduced modulo P, so each worker generates and caches at most P worlds
per difficulty (generation costs about 0.6 s per track).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import OrderedDict

import numpy as np

from ..env import DriveEnv
from ..reward import RewardConfig


class WorldSampler:
    """Callable ``track_seed -> (terrain, track)`` with an LRU cache and a mutable difficulty."""

    def __init__(self, closed: bool, difficulty: float, pool_size: int = 0, cache_size: int = 4):
        self.closed = bool(closed)
        self.difficulty = float(difficulty)
        self.pool_size = int(pool_size)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict = OrderedDict()

    def set(self, difficulty: float, pool_size: int) -> None:
        self.difficulty = float(difficulty)
        self.pool_size = int(pool_size)

    def __call__(self, track_seed: int):
        from ..world import generate_world

        seed = int(track_seed) % self.pool_size if self.pool_size > 0 else int(track_seed)
        key = (seed, self.closed, self.difficulty)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        world = generate_world(seed, closed=self.closed, difficulty=self.difficulty)
        self._cache[key] = world
        while len(self._cache) > max(self.cache_size, self.pool_size):
            self._cache.popitem(last=False)
        return world


def make_env(spec: dict) -> DriveEnv:
    """Build a DriveEnv from a picklable spec dict (see :func:`env_spec`)."""
    closed = bool(spec["closed"])
    sampler = WorldSampler(closed, spec["difficulty"], spec.get("pool_size", 0))
    reward = RewardConfig(**spec["reward"]) if spec.get("reward") else None
    env = DriveEnv(closed=closed, difficulty=spec["difficulty"], world_fn=sampler,
                   max_steps=spec["max_steps"], control_mode="continuous",
                   reward=reward, laps=spec.get("laps", 1))
    env.world_sampler = sampler
    return env


def env_spec(closed: bool = False, difficulty: float = 0.0, pool_size: int = 0,
             max_steps: int = 1500, reward: dict | None = None, laps: int = 1) -> dict:
    return {"closed": bool(closed), "difficulty": float(difficulty), "pool_size": int(pool_size),
            "max_steps": int(max_steps), "reward": dict(reward) if reward else None,
            "laps": int(laps)}


def _worker(remote, spec, seed):
    # one core per worker: avoid BLAS / OpenMP oversubscription
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    env = make_env(spec)
    low, high = env.action_space.low, env.action_space.high
    obs, _ = env.reset(seed=seed)
    remote.send(obs)                       # first observation (handshake)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                a = np.clip(np.asarray(data, dtype=np.float32), low, high)
                obs2, r, term, trunc, info = env.step(a)
                info = dict(info)
                if term or trunc:
                    info["final_obs"] = obs2
                    obs2, _ = env.reset()
                remote.send((obs2, float(r), bool(term), bool(trunc), info))
            elif cmd == "world":
                env.world_sampler.set(*data)
                remote.send(True)
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class SubprocVecEnv:
    """Pipe-based vector env. ``step`` sends all actions first, then collects all results."""

    def __init__(self, spec: dict, num_envs: int, seed: int = 0):
        ctx = mp.get_context("spawn")
        probe = make_env(spec)
        self.action_space = probe.action_space
        self.observation_dim = probe.observation_space.shape[0]
        probe.close()
        self.num_envs = int(num_envs)
        self.remotes, self.procs = [], []
        for i in range(self.num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, spec, seed + i), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self._closed = False
        self.initial_obs = np.stack([r.recv() for r in self.remotes]).astype(np.float32)

    def set_world(self, difficulty: float, pool_size: int) -> None:
        for r in self.remotes:
            r.send(("world", (difficulty, pool_size)))
        for r in self.remotes:
            r.recv()

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", np.asarray(a, dtype=np.float32)))
        return _collect([r.recv() for r in self.remotes])

    def close(self):
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class DummyVecEnv:
    """Same interface as :class:`SubprocVecEnv`, all envs in this process (tests, debugging)."""

    def __init__(self, spec: dict, num_envs: int, seed: int = 0):
        self.envs = [make_env(spec) for _ in range(int(num_envs))]
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_dim = self.envs[0].observation_space.shape[0]
        self.initial_obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.initial_obs = self.initial_obs.astype(np.float32)

    def set_world(self, difficulty: float, pool_size: int) -> None:
        for e in self.envs:
            e.world_sampler.set(difficulty, pool_size)

    def step(self, actions):
        low, high = self.action_space.low, self.action_space.high
        out = []
        for e, a in zip(self.envs, actions):
            obs2, r, term, trunc, info = e.step(np.clip(np.asarray(a, np.float32), low, high))
            info = dict(info)
            if term or trunc:
                info["final_obs"] = obs2
                obs2, _ = e.reset()
            out.append((obs2, float(r), bool(term), bool(trunc), info))
        return _collect(out)

    def close(self):
        for e in self.envs:
            e.close()


def _collect(out):
    obs = np.stack([o[0] for o in out]).astype(np.float32)
    rew = np.array([o[1] for o in out], dtype=np.float64)
    term = np.array([o[2] for o in out], dtype=bool)
    trunc = np.array([o[3] for o in out], dtype=bool)
    infos = [o[4] for o in out]
    return obs, rew, term, trunc, infos
