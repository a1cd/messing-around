"""Behaviour cloning (with DAgger) from the pure-pursuit controller, to warm-start PPO.

The pure-pursuit baseline drives A->B reliably, and its decisions are a function of the
observation (lookahead points, curvature ahead, speed, lateral offset), so an MLP can imitate
it. Plain BC drifts off the expert's state distribution, so DAgger rolls out the *learner*
(with probability 1 - beta per step) and labels every visited state with the expert action.

Output: a checkpoint in the same format as ``train.py`` (``weights``, ``obs_rms``, ``cfg``), so it
can be loaded with ``train.py --init-from`` or evaluated with ``eval.py --ckpt``.

Example::

    python scripts/bc_pretrain.py --out runs/bc1/bc.pt --episodes 240 --dagger-iters 4 \
        --workers 4 --difficulties 0,0.5,1.0
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def _rollout(job: dict) -> dict:
    """One episode. Executes the learner with prob 1-beta per step, else the expert.

    Labels are always the expert action for the visited observation.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from drivesim.env import DriveEnv
    from drivesim.baselines import PurePursuitController, to_action
    from drivesim.rl.ppo import numpy_mlp, normalize_obs, to_env_action

    env = DriveEnv(closed=job["closed"], difficulty=job["difficulty"], seed=None,
                   max_steps=job["max_steps"], control_mode="continuous")
    low, high = env.action_space.low, env.action_space.high
    pp = PurePursuitController()
    rng = np.random.default_rng(job["seed"])
    w, n_layers = job["weights"], job["n_layers"]
    obs, _ = env.reset(seed=job["seed"])
    O, Y = [], []
    term = trunc = False
    info = {"success": False, "termination": None}
    while not (term or trunc):
        expert = to_action(pp.act(env.car, env.track, env._idx), "continuous")
        O.append(obs.copy())
        Y.append(expert.copy())
        if w is None or rng.random() < job["beta"]:
            a = expert
        else:
            x = normalize_obs(obs[None], job["obs_mean"], job["obs_var"], job["clip_obs"])
            mu = numpy_mlp(w, "actor", x, n_layers)[0]
            a = to_env_action(mu, low, high, "clip")
        obs, r, term, trunc, info = env.step(a)
    env.close()
    return {"obs": np.asarray(O, np.float32), "act": np.asarray(Y, np.float32),
            "success": bool(info["success"]), "termination": info["termination"] or "truncated",
            "steps": len(O), "closed": job["closed"]}


def _collect(pool, n_episodes, weights, n_layers, beta, obs_mean, obs_var, clip_obs, difficulties,
             max_steps, seed_base, rng, closed_frac=0.0, closed_max_steps=3000):
    jobs = []
    for i in range(n_episodes):
        closed = bool(rng.random() < closed_frac)
        jobs.append(dict(
            difficulty=float(difficulties[rng.integers(len(difficulties))]), closed=closed,
            seed=int(seed_base + i), max_steps=closed_max_steps if closed else max_steps,
            weights=weights, n_layers=n_layers,
            beta=beta, obs_mean=obs_mean, obs_var=obs_var, clip_obs=clip_obs))
    return pool.map(_rollout, jobs, chunksize=1)


def _train(model, opt, O, Y, epochs, batch, rng, torch):
    n = len(O)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model(O[idx])
            loss = ((pred - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float(((model(O) - Y) ** 2).mean())


def _save(path, backend, rms, cfg, obs_dim, act_dim, args, it, n_data):
    ck = {"format": 1, "weights": backend.get_weights(), "obs_rms": rms.state(), "cfg": cfg.as_dict(),
          "obs_dim": obs_dim, "act_dim": act_dim, "updates": 0, "total_steps": 0, "backend": "torch",
          "bc": {"iters_done": it, "data": int(n_data), "args": vars(args)}}
    ck["weights"]["log_std"] = np.full((act_dim,), args.log_std, np.float32)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ck, f)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=240, help="expert episodes in iteration 0")
    ap.add_argument("--dagger-iters", type=int, default=4)
    ap.add_argument("--episodes-per-iter", type=int, default=120)
    ap.add_argument("--difficulties", default="0,0.5,1.0")
    ap.add_argument("--max-steps", type=int, default=1500, help="open-track episode limit")
    ap.add_argument("--closed-frac", type=float, default=0.0,
                    help="fraction of episodes on closed loops (one policy then drives both tasks)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-std", type=float, default=-2.0, help="log-std saved for PPO fine-tuning")
    ap.add_argument("--beta", type=float, default=0.5, help="DAgger: P(expert acts) in iteration 1")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--seed-base", type=int, default=100_000_000,
                    help="training track seeds start here (eval seeds are 900M+)")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from drivesim.rl.ppo import PPOConfig, RunningMeanStd, TorchBackend
    from drivesim.env import DriveEnv

    torch.set_num_threads(max(1, args.workers))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    difficulties = [float(x) for x in args.difficulties.split(",")]
    env = DriveEnv(closed=False, difficulty=0.0, seed=None, control_mode="continuous")
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    env.close()

    cfg = PPOConfig(hidden=args.hidden, layers=args.layers, init_log_std=args.log_std, squash="clip")
    backend = TorchBackend(obs_dim, act_dim, low, high, cfg)
    actor = backend.actor
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    rms = RunningMeanStd((obs_dim,))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    O_all, Y_all = [], []
    seed_cursor = args.seed_base
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for it in range(args.dagger_iters + 1):
            if it == 0:
                n_ep, weights, beta = args.episodes, None, 1.0
            else:
                n_ep, beta = args.episodes_per_iter, args.beta ** it
                weights = backend.get_weights()
            rows = _collect(pool, n_ep, weights, args.layers, beta,
                            rms.mean.copy() if it else None, rms.var.copy() if it else None,
                            cfg.clip_obs, difficulties, args.max_steps, seed_cursor, rng,
                            closed_frac=args.closed_frac)
            seed_cursor += n_ep
            succ = np.mean([r["success"] for r in rows])
            for r in rows:
                O_all.append(r["obs"])
                Y_all.append(r["act"])
            O = np.concatenate(O_all)
            Y = np.concatenate(Y_all)
            if it == 0:
                # observation normaliser: fixed from expert data (the PPO fine-tune keeps updating it)
                rms.update(O)
            On = np.clip((O - rms.mean) / np.sqrt(rms.var + 1e-8), -cfg.clip_obs, cfg.clip_obs)
            On_t = torch.as_tensor(On, dtype=torch.float32)
            Y_t = torch.as_tensor(Y, dtype=torch.float32)
            # actor output is the pre-squash mean; the expert targets are already in the action box
            mse = _train(actor, opt, On_t, Y_t, args.epochs, args.batch, rng, torch)
            succ_closed = [r["success"] for r in rows if r.get("closed")]
            _save(args.out, backend, rms, cfg, obs_dim, act_dim, args, it, len(O))
            print(f"iter {it} beta {beta:.3f} episodes {len(rows)} learner-run-success {succ:.2f} "
                  f"(closed {np.mean(succ_closed) if succ_closed else float('nan'):.2f}) "
                  f"data {len(O)} train-mse {mse:.5f} t {(time.time() - t0) / 60:.1f}m", flush=True)

    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
