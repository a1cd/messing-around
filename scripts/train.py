"""Train a PPO driving policy on drivesim.

Examples::

    python scripts/train.py --task ab --num-envs 4 --total-steps 200000 --save-dir runs/smoke
    python scripts/train.py --task ab --curriculum --difficulties 0,0.3,0.6,1.0 \
        --threshold 0.9 --save-dir runs/ab1
    python scripts/train.py --task loop --difficulties 0 --save-dir runs/loop1

Outputs in ``--save-dir``: ``metrics.csv`` (one row per update), ``config.json``,
``latest.pt`` (resumable: weights, optimizer, normalisers), ``best.pt`` (weights only),
``learning_curves.png``. Re-running with the same ``--save-dir`` resumes from ``latest.pt``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from drivesim.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402
from drivesim.rl.vec_env import DummyVecEnv, SubprocVecEnv, env_spec  # noqa: E402


class Curriculum:
    """Step through difficulty levels once the success rate since the stage start is high."""

    def __init__(self, difficulties, threshold: float, window: int, min_episodes: int, enabled: bool):
        self.difficulties = list(difficulties)
        self.threshold = threshold
        self.window = window
        self.min_episodes = min_episodes
        self.enabled = enabled and len(self.difficulties) > 1
        self.stage = 0
        self.hits: list = []

    @property
    def difficulty(self) -> float:
        return self.difficulties[self.stage]

    def observe(self, episodes) -> bool:
        """Feed finished episodes. Returns True when the stage advanced."""
        if not self.enabled:
            return False
        self.hits += [float(e["success"]) for e in episodes]
        self.hits = self.hits[-self.window:]
        if (self.stage < len(self.difficulties) - 1 and len(self.hits) >= self.min_episodes
                and np.mean(self.hits) >= self.threshold):
            self.stage += 1
            self.hits = []
            return True
        return False

    def state(self) -> dict:
        return {"stage": self.stage, "hits": self.hits}

    def load(self, st: dict) -> None:
        self.stage = int(st["stage"])
        self.hits = list(st["hits"])


def save_pickle(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_backend(name: str, obs_dim, act_dim, low, high, cfg: PPOConfig):
    if name == "mlx":
        try:
            from drivesim.rl.ppo_mlx import MLXBackend
            return MLXBackend(obs_dim, act_dim, low, high, cfg), "mlx"
        except ImportError as exc:
            print(f"[warn] MLX backend unavailable ({exc}); falling back to torch", flush=True)
    from drivesim.rl.ppo import TorchBackend
    return TorchBackend(obs_dim, act_dim, low, high, cfg), "torch"


def checkpoint(trainer, backend, cfg, curriculum, full: bool) -> dict:
    ck = {"format": 1, "weights": backend.get_weights(), "obs_rms": trainer.obs_rms.state(),
          "cfg": cfg.as_dict(), "obs_dim": trainer.obs_dim, "act_dim": backend.act_dim,
          "updates": trainer.updates, "total_steps": trainer.total_steps,
          "backend": backend.name}
    if full:
        ck.update(trainer_state=trainer.state(), optimizer=backend.get_optimizer_state(),
                  ret_rms=trainer.ret_rms.state(), curriculum=curriculum.state())
    return ck


METRIC_COLS = ["update", "total_steps", "wall_s", "fps", "t_rollout", "t_total", "ep_return",
               "ep_length", "success_rate", "n_episodes", "pg_loss", "vf_loss", "entropy",
               "approx_kl", "clipfrac", "explained_var", "lr", "epochs_run", "difficulty", "stage"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["ab", "loop"], default="ab")
    ap.add_argument("--difficulties", default="0.0", help="comma list; first is the start level")
    ap.add_argument("--curriculum", action="store_true", help="advance through --difficulties")
    ap.add_argument("--threshold", type=float, default=0.9, help="success rate to advance a stage")
    ap.add_argument("--window", type=int, default=100, help="episodes in the success window")
    ap.add_argument("--min-episodes", type=int, default=100)
    ap.add_argument("--pool", type=int, default=32,
                    help="distinct training tracks per difficulty (0 = a fresh random track every reset)")
    ap.add_argument("--max-steps", type=int, default=None, help="episode step limit (default 1500 ab, 3000 loop)")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--reward", nargs="*", default=[], metavar="KEY=VALUE",
                    help="RewardConfig overrides, e.g. time=0.02 completion=100")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-len", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final-frac", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--init-log-std", type=float, default=-0.5)
    ap.add_argument("--init-bias", default=None,
                    help="comma list: actor output bias at init per action (steer,throttle,brake)")
    ap.add_argument("--squash", choices=["clip", "tanh"], default="clip")
    ap.add_argument("--no-obs-norm", action="store_true")
    ap.add_argument("--no-reward-norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--in-process", action="store_true", help="no worker processes (debugging)")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "runs", "latest"))
    ap.add_argument("--save-every", type=int, default=10, help="updates between latest.pt saves")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt in --save-dir")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args(argv)


def parse_reward(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    os.makedirs(args.save_dir, exist_ok=True)
    closed = args.task == "loop"
    difficulties = [float(x) for x in args.difficulties.split(",")]
    max_steps = args.max_steps or (3000 if closed else 1500)
    cfg = PPOConfig(
        num_envs=args.num_envs, rollout_len=args.rollout_len, total_steps=args.total_steps,
        epochs=args.epochs, minibatches=args.minibatches, lr=args.lr, lr_final_frac=args.lr_final_frac,
        max_grad_norm=args.max_grad_norm, target_kl=args.target_kl, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip=args.clip, reward_norm=not args.no_reward_norm,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef, hidden=args.hidden, layers=args.layers,
        init_log_std=args.init_log_std, squash=args.squash,
        init_act_bias=[float(x) for x in args.init_bias.split(",")] if args.init_bias else None, obs_norm=not args.no_obs_norm,
        seed=args.seed, backend=args.backend)
    spec = env_spec(closed=closed, difficulty=difficulties[0], pool_size=args.pool,
                    max_steps=max_steps, reward=parse_reward(args.reward), laps=args.laps)
    VecCls = DummyVecEnv if args.in_process else SubprocVecEnv
    venv = VecCls(spec, cfg.num_envs, seed=cfg.seed * 1000)
    backend, used = make_backend(args.backend, venv.observation_dim, venv.action_space.shape[0],
                                 venv.action_space.low, venv.action_space.high, cfg)
    cfg.backend = used
    trainer = PPOTrainer(cfg, venv, backend)
    curriculum = Curriculum(difficulties, args.threshold, args.window, args.min_episodes, args.curriculum)

    latest = os.path.join(args.save_dir, "latest.pt")
    best_path = os.path.join(args.save_dir, "best.pt")
    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    best_key = (-1.0, -np.inf)
    if os.path.exists(latest) and not args.fresh:
        ck = load_pickle(latest)
        backend.set_weights(ck["weights"])
        backend.set_optimizer_state(ck["optimizer"])
        trainer.obs_rms.load(ck["obs_rms"])
        trainer.ret_rms.load(ck["ret_rms"])
        trainer.load_state(ck["trainer_state"])
        curriculum.load(ck["curriculum"])
        best_key = tuple(ck.get("best_key", best_key))
        print(f"resumed {latest} at update {trainer.updates}, {trainer.total_steps} steps", flush=True)
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(METRIC_COLS)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump({"cfg": cfg.as_dict(), "task": args.task, "difficulties": difficulties,
                   "curriculum": args.curriculum, "threshold": args.threshold, "window": args.window,
                   "pool": args.pool, "max_steps": max_steps, "reward": parse_reward(args.reward)},
                  f, indent=2)
    venv.set_world(curriculum.difficulty, args.pool)
    print(f"task={args.task} backend={used} envs={cfg.num_envs} batch={cfg.batch_size()} "
          f"difficulty={curriculum.difficulty} pool={args.pool or 'random'}", flush=True)

    t_start = time.time()
    try:
        while trainer.total_steps < cfg.total_steps:
            s = trainer.train_iteration()
            advanced = curriculum.observe(trainer.last_finished)
            if advanced:
                venv.set_world(curriculum.difficulty, args.pool)
                print(f"[curriculum] advanced to difficulty {curriculum.difficulty:.2f} "
                      f"at step {trainer.total_steps}", flush=True)
            s.update(difficulty=curriculum.difficulty, stage=curriculum.stage)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([s.get(c, "") for c in METRIC_COLS])
            print(f"upd {s['update']:5d} steps {s['total_steps']:9d} fps {s['fps']:6.0f} "
                  f"succ {s['success_rate']:.2f} ret {s['ep_return']:8.1f} len {s['ep_length']:6.0f} "
                  f"diff {curriculum.difficulty:.2f} | pg {s['pg_loss']:+.4f} vf {s['vf_loss']:.3f} "
                  f"ent {s['entropy']:+.2f} kl {s['approx_kl']:.4f} clip {s['clipfrac']:.2f} "
                  f"ev {s['explained_var']:+.2f} lr {s['lr']:.1e} t {s['wall_s']/60:5.1f}m",
                  flush=True)
            key = (s["success_rate"] if s["n_episodes"] >= args.window // 2 else -1.0, s["ep_return"])
            if key > best_key and s["n_episodes"] >= args.window // 2:
                best_key = key
                ck = checkpoint(trainer, backend, cfg, curriculum, full=False)
                ck.update(best_key=best_key, difficulty=curriculum.difficulty, stats=s)
                save_pickle(best_path, ck)
            if trainer.updates % args.save_every == 0:
                ck = checkpoint(trainer, backend, cfg, curriculum, full=True)
                ck.update(best_key=best_key)
                save_pickle(latest, ck)
    except KeyboardInterrupt:
        print("interrupted; saving latest", flush=True)
    finally:
        ck = checkpoint(trainer, backend, cfg, curriculum, full=True)
        ck.update(best_key=best_key)
        save_pickle(latest, ck)
        venv.close()
    print(f"done: {trainer.total_steps} steps, {trainer.updates} updates, "
          f"{(time.time() - t_start) / 60:.1f} min this session; best={best_key}", flush=True)
    if not args.no_plot:
        from drivesim.rl.plot import plot_run
        print("plot:", plot_run(args.save_dir), flush=True)


if __name__ == "__main__":
    main()
