"""Hand-coded baseline controller and evaluation for :class:`drivesim.env.DriveEnv`.

:class:`PurePursuitController` steers toward a point on the centerline ``Ld`` metres ahead
(pure pursuit with the wheelbase from the vehicle parameters) and sets a target speed from
the lateral-acceleration limit on the curvature ahead (with a braking distance check to the
end of an open track). It uses only the track and the vehicle state, so it does not depend
on the env observation.

Run as a script to evaluate it::

    python -m drivesim.baselines              # closed + open, 5 seeds, 3 difficulties
    python -m drivesim.baselines --quick
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

from .env import DISCRETE_ACTIONS, DriveEnv
from .vehicle import PRESETS


@dataclass
class PurePursuitController:
    params: object = None               # VehicleParams; defaults to sports car
    lookahead_min: float = 5.0          # m
    lookahead_gain: float = 0.45        # m per m/s of speed
    lookahead_max: float = 25.0         # m
    a_lat: float = 7.0                  # m/s^2 comfortable lateral acceleration
    a_brake: float = 6.0                # m/s^2 planned braking deceleration
    speed_max: float = 45.0             # m/s cap
    speed_ahead: tuple = tuple(range(0, 101, 10))  # m, curvature look-ahead for speed

    def __post_init__(self):
        if self.params is None:
            self.params = PRESETS["sports_car"]()

    def act(self, car, track, hint=None):
        """Return a dict with ``steer`` [-1, 1], ``throttle`` [0, 1], ``brake`` [0, 1]."""
        p = self.params
        v = car.forward_speed
        pos = car.position[:2]
        yaw = car.yaw
        pr = track.project(pos, hint)
        s = pr.s
        # ---- steering: pure pursuit
        Ld = min(self.lookahead_max, max(self.lookahead_min, self.lookahead_gain * abs(v)))
        xyz, _ = track.point_at(s + Ld)
        dx, dy = xyz[0] - pos[0], xyz[1] - pos[1]
        c, sn = math.cos(yaw), math.sin(yaw)
        y_left = -sn * dx + c * dy
        curv = 2.0 * y_left / (Ld * Ld)
        road_angle = math.atan(p.wheelbase * curv)
        smax = p.steer_max_low + (p.steer_max_high - p.steer_max_low) * min(abs(v) / p.steer_speed_ref, 1.0)
        steer = max(-1.0, min(1.0, road_angle / smax))
        # ---- speed: lateral-acceleration limit on the curvature ahead
        d = np.array(self.speed_ahead, dtype=float)
        k = np.abs(track.curvature_at(s + d))
        v_target = self.speed_max
        if np.any(k > 1e-4):
            v_corner = np.sqrt(self.a_lat / np.maximum(k, 1e-4))
            # braking-distance-limited speed to reach each corner
            v_corner = np.minimum(v_corner, np.sqrt(np.maximum(v_corner ** 2 + 2 * self.a_brake * d, 0)))
            v_target = min(v_target, float(np.min(v_corner)))
        if not track.closed:
            rem = track.length - s
            v_target = min(v_target, math.sqrt(2.0 * self.a_brake * max(rem - 2.0, 0.0)))
        err = abs(v) - v_target
        if err > 0.5:
            throttle, brake = 0.0, min(1.0, err / 4.0)
        else:
            throttle, brake = max(0.0, min(1.0, 0.5 + (v_target - abs(v)) / 4.0)), 0.0
        return {"steer": steer, "throttle": throttle, "brake": brake}


def to_action(ctrl: dict, control_mode: str):
    """Map a controller output to the action space of a :class:`DriveEnv` control mode."""
    if control_mode == "discrete":
        steer_idx = 0 if ctrl["steer"] > 0.25 else (2 if ctrl["steer"] < -0.25 else 1)
        gas = 1 if ctrl["throttle"] > 0.5 else 0
        brake = 1 if ctrl["brake"] > 0.5 else 0
        return steer_idx * 4 + gas * 2 + brake
    if control_mode == "continuous":
        return np.array([ctrl["steer"], ctrl["throttle"], ctrl["brake"]], dtype=np.float32)
    return np.array([ctrl["steer"], ctrl["throttle"], ctrl["brake"], 0, 0, 0, 0], dtype=np.float32)


def run_episode(env: DriveEnv, controller: PurePursuitController, seed=None, max_steps=None):
    obs, info = env.reset(seed=seed)
    ret = 0.0
    n = 0
    hint = None
    term = trunc = False
    while not (term or trunc):
        ctrl = controller.act(env.car, env.track, env._idx)
        obs, r, term, trunc, info = env.step(to_action(ctrl, env.control_mode))
        ret += r
        n += 1
    return {"return": ret, "steps": n, "termination": info["termination"],
            "success": bool(info["success"]), "time_s": info["time"],
            "distance": info["distance"], "track_length": env.track.length}


def evaluate(seeds=range(5), difficulties=(0.0, 0.5, 1.0), closed_modes=(True, False),
             control_mode="continuous", max_steps=3000):
    """Run the baseline on each (closed, difficulty, seed) and summarise success rates."""
    rows = []
    ctrl = PurePursuitController()
    for closed in closed_modes:
        for diff in difficulties:
            for seed in seeds:
                env = DriveEnv(closed=closed, difficulty=diff, seed=seed,
                               control_mode=control_mode, max_steps=max_steps)
                res = run_episode(env, ctrl)
                res.update(closed=closed, difficulty=diff, seed=seed)
                rows.append(res)
                env.close()
    return rows


def summarise(rows):
    out = {}
    for closed in (True, False):
        for diff in sorted({r["difficulty"] for r in rows}):
            sub = [r for r in rows if r["closed"] == closed and r["difficulty"] == diff]
            if not sub:
                continue
            succ = np.mean([r["success"] for r in sub])
            out[("loop" if closed else "A->B", diff)] = {
                "n": len(sub), "success": float(succ),
                "mean_return": float(np.mean([r["return"] for r in sub])),
                "mean_time_s": float(np.mean([r["time_s"] for r in sub])),
                "terminations": {t: sum(1 for r in sub if r["termination"] == t)
                                 for t in {r["termination"] for r in sub}},
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="2 seeds, 2 difficulties")
    ap.add_argument("--mode", default="continuous", choices=["discrete", "continuous", "full"])
    args = ap.parse_args(argv)
    seeds = range(2) if args.quick else range(5)
    diffs = (0.0, 1.0) if args.quick else (0.0, 0.5, 1.0)
    t0 = time.time()
    rows = evaluate(seeds=seeds, difficulties=diffs, control_mode=args.mode)
    dt = time.time() - t0
    for key, v in summarise(rows).items():
        print(f"{key[0]:5s} difficulty={key[1]:.1f}  n={v['n']}  success={v['success']:.2f}  "
              f"return={v['mean_return']:8.1f}  time={v['mean_time_s']:6.1f}s  {v['terminations']}")
    steps = sum(r["steps"] for r in rows)
    print(f"total {len(rows)} episodes, {steps} env steps, {steps / dt:.0f} steps/s incl. world generation")


if __name__ == "__main__":
    main()
