"""Evaluate a trained PPO checkpoint on fresh tracks.

Examples::

    python scripts/eval.py --ckpt checkpoints/ab_policy.pt --task ab --difficulty 0.5 --episodes 20
    python scripts/eval.py --ckpt runs/ab1/best.pt --task ab --difficulty 1.0 --episodes 20 --workers 4
    python scripts/eval.py --ckpt checkpoints/ab_policy.pt --task ab --episodes 1 --render-dir frames/ab

Episodes use the deterministic policy (action = mean; clipped or tanh-squashed as trained) on
tracks generated from seeds ``--seed0 + i``. Those seeds are random 31-bit track seeds, so they
do not overlap the small training pool (seeds 0..pool-1). Frames are saved as PNG (headless
software renderer, ``SDL_VIDEODRIVER=dummy``) every ``--render-every`` steps.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def run_episode(job: dict) -> dict:
    from drivesim.env import DriveEnv
    from drivesim.rl.ppo import load_policy
    from drivesim.baselines import to_action  # noqa: F401  (used by the pure-pursuit baseline)

    policy = load_policy(job["ckpt"]) if job.get("ckpt") else None
    pp = None
    if job.get("baseline") == "pure_pursuit":
        from drivesim.baselines import PurePursuitController, to_action
        pp = PurePursuitController()
    env = DriveEnv(closed=job["closed"], difficulty=job["difficulty"], seed=None,
                   max_steps=job["max_steps"], control_mode="continuous")
    low, high = env.action_space.low, env.action_space.high
    frames_dir = job.get("render_dir")
    viewer = None
    if frames_dir:
        from drivesim.viewer import Viewer
        os.makedirs(frames_dir, exist_ok=True)
    t0 = time.time()
    obs, info = env.reset(seed=job["seed"])
    if frames_dir:
        viewer = Viewer(env.terrain, env.track, width=job["width"], height=job["height"], headless=True)
    ret, steps, done = 0.0, 0, False
    info = {"termination": None, "success": False, "time": 0.0, "distance": 0.0}
    frame_idx = 0
    while not done:
        if viewer is not None and steps % job["render_every"] == 0:
            import pygame
            frame = viewer.render(env.car, hud={"speed_kmh": env.car.speed_kmh})
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            pygame.image.save(surf, os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"))
            frame_idx += 1
        if pp is not None:
            a = to_action(pp.act(env.car, env.track, env._idx), "continuous")
        else:
            a = policy.act(obs, low, high)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        steps += 1
        done = term or trunc
    if viewer is not None:
        viewer.close()
    res = {"seed": job["seed"], "success": bool(info["success"]), "return": ret, "steps": steps,
           "termination": info["termination"] or "truncated", "sim_time_s": info["time"],
           "distance": info["distance"], "track_length": env.track.length,
           "wall_s": time.time() - t0, "frames": frame_idx}
    env.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="policy checkpoint (omit with --baseline)")
    ap.add_argument("--baseline", choices=["pure_pursuit"], default=None,
                    help="evaluate the hand-coded controller on the same seeds instead")
    ap.add_argument("--task", choices=["ab", "loop"], default="ab")
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=900_000_000, help="first track seed")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--render-dir", default=None, help="save PNG frames of the first episode here")
    ap.add_argument("--render-every", type=int, default=10)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--json", default=None, help="write per-episode results to this JSON file")
    args = ap.parse_args(argv)
    closed = args.task == "loop"
    max_steps = args.max_steps or (3000 if closed else 1500)
    if args.ckpt is None and args.baseline is None:
        ap.error("give --ckpt or --baseline")
    jobs = [dict(ckpt=os.path.abspath(args.ckpt) if args.ckpt else None, baseline=args.baseline, closed=closed, difficulty=args.difficulty,
                 max_steps=max_steps, seed=args.seed0 + i, render_every=args.render_every,
                 width=args.width, height=args.height,
                 render_dir=args.render_dir if (args.render_dir and i == 0) else None)
            for i in range(args.episodes)]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(run_episode, jobs, chunksize=1)
    else:
        rows = [run_episode(j) for j in jobs]
    wall = time.time() - t0
    succ = np.array([r["success"] for r in rows])
    ret = np.array([r["return"] for r in rows])
    name = args.ckpt or f"baseline {args.baseline}"
    print(f"policy {name}  task={args.task}  difficulty={args.difficulty}  episodes={len(rows)}")
    print(f"success      {succ.mean() * 100:.1f}%  ({succ.sum()}/{len(rows)})")
    print(f"return       mean {ret.mean():.1f}  std {ret.std():.1f}")
    if succ.any():
        st = np.array([r["sim_time_s"] for r in rows if r["success"]])
        print(f"sim time     mean {st.mean():.1f} s on successful episodes")
    print(f"steps        mean {np.mean([r['steps'] for r in rows]):.0f}")
    terms = {}
    for r in rows:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    print(f"terminations {terms}")
    print(f"wall time    {wall:.1f} s total  ({sum(r['steps'] for r in rows) / max(wall, 1e-9):.0f} steps/s)")
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump({"summary": {"success": float(succ.mean()), "return": float(ret.mean())},
                       "episodes": rows}, f, indent=2)
    return float(succ.mean())


if __name__ == "__main__":
    main()
