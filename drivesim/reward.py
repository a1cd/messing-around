"""Reward function for :class:`drivesim.env.DriveEnv`.

Design
------
The main signal is **centerline progress**: the change in arc length ``s`` of the
car's projection onto the track, per metre. Reward therefore follows distance
along the road, not raw speed. Driving fast into a corner and leaving the road
loses progress and triggers penalties, so the policy learns to brake for corners.

Per physics step (``dt`` seconds) the total reward is the sum of:

=============== ==========================================================
term            definition (weight field in :class:`RewardConfig`)
=============== ==========================================================
progress        ``progress * delta_s`` with ``delta_s`` in metres. Closed
                tracks wrap at the start/finish line. Driving backwards
                gives a negative value.
speed           ``speed * clip(v_along / speed_ref, -1, 1)``. A small bonus
                for speed along the track tangent. It is an order of
                magnitude smaller than the progress rate at speed.
backwards       ``-backwards`` when ``v_along < -0.5 m/s``.
offroad         beyond the road edge: ``-(offroad_step + offroad_per_m * excess)``
                where ``excess`` is the distance beyond the edge (m).
heading         ``-heading * |heading error|`` (rad). Heading error is the
                track tangent minus the car yaw.
steer_rate      ``-steer_rate * |steer - previous steer|`` (smoothness).
pedal_rate      ``-pedal_rate * (|d throttle| + |d brake|)`` (smoothness).
steer_mag       ``-steer_mag * |steer|`` (discourages constant lock).
wheelspin       ``-wheelspin * sum_wheels max(0, |slip ratio| - 0.2)``
                (off by default, weight 0).
time            ``-time`` per step.
=============== ==========================================================

Terminal terms (added once when the episode ends):

* ``completion`` on success (goal reached or laps completed),
* ``-offtrack_term`` when the car is off track too long,
* ``-flip`` when the car flips,
* ``-stuck_term`` when there is no new best progress for ``stuck_time``.

The environment has no obstacles or walls, so "crash" means flipping or leaving
the road.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple

import math

import numpy as np


@dataclass
class RewardConfig:
    progress: float = 0.1          # per metre of centerline progress
    speed: float = 0.02            # per step, scaled by v_along / speed_ref
    speed_ref: float = 30.0        # m/s
    backwards: float = 0.1         # per step while moving backwards (v_along < -0.5)
    offroad_step: float = 0.2      # per step while beyond the road edge
    offroad_per_m: float = 0.05    # per step per metre beyond the road edge
    heading: float = 0.02          # per step per radian of heading error
    steer_rate: float = 0.05       # per unit change of steer per step
    pedal_rate: float = 0.03       # per unit change of throttle+brake per step
    steer_mag: float = 0.005       # per unit |steer| per step
    wheelspin: float = 0.0         # per step per unit of excess wheel slip (off by default)
    time: float = 0.01             # per step
    completion: float = 100.0      # terminal bonus on success
    offtrack_term: float = 10.0    # terminal penalty for leaving the track for too long
    flip: float = 20.0             # terminal penalty for flipping
    stuck_term: float = 5.0        # terminal penalty for no progress

    def replace(self, **kw) -> "RewardConfig":
        return replace(self, **kw)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RewardCalculator:
    """Stateful per-episode reward. Call :meth:`reset` then :meth:`step` once per physics step."""

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config if config is not None else RewardConfig()
        self.track = None
        self.prev_s = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

    def reset(self, track, s: float, steer: float = 0.0, throttle: float = 0.0,
              brake: float = 0.0) -> None:
        self.track = track
        self.prev_s = float(s)
        self.prev_steer = float(steer)
        self.prev_throttle = float(throttle)
        self.prev_brake = float(brake)

    def delta_progress(self, s: float) -> float:
        """Change of arc length since the last call, wrapped on closed tracks (metres)."""
        d = float(s) - self.prev_s
        if self.track.closed:
            L = self.track.length
            if d > 0.5 * L:
                d -= L
            elif d < -0.5 * L:
                d += L
        return d

    def step(self, *, s: float, lateral: float, idx: int, yaw: float, velocity_xy,
             steer: float, throttle: float, brake: float, wheel_slip_ratios=None,
             ) -> Tuple[float, Dict[str, float]]:
        """Reward for one physics step. Updates the internal previous-value state."""
        c = self.cfg
        track = self.track
        d_s = self.delta_progress(s)
        self.prev_s = float(s)

        tan = track.tangent[idx]
        v_along = float(velocity_xy[0] * tan[0] + velocity_xy[1] * tan[1])
        excess = max(0.0, abs(lateral) - track.half_width)
        h_err = abs(_wrap(float(track.heading[idx]) - yaw))
        terms = {
            "progress": c.progress * d_s,
            "speed": c.speed * max(-1.0, min(1.0, v_along / c.speed_ref)),
            "backwards": -c.backwards if v_along < -0.5 else 0.0,
            "offroad": -(c.offroad_step + c.offroad_per_m * excess) if excess > 0.0 else 0.0,
            "heading": -c.heading * h_err,
            "steer_rate": -c.steer_rate * abs(steer - self.prev_steer),
            "pedal_rate": -c.pedal_rate * (abs(throttle - self.prev_throttle)
                                           + abs(brake - self.prev_brake)),
            "steer_mag": -c.steer_mag * abs(steer),
            "wheelspin": 0.0,
            "time": -c.time,
        }
        if c.wheelspin and wheel_slip_ratios is not None:
            sl = np.abs(np.asarray(wheel_slip_ratios, dtype=float))
            terms["wheelspin"] = -c.wheelspin * float(np.sum(np.maximum(0.0, sl - 0.2)))
        self.prev_steer, self.prev_throttle, self.prev_brake = steer, throttle, brake
        total = float(sum(terms.values()))
        return total, terms

    def terminal(self, reason: str | None) -> Tuple[float, Dict[str, float]]:
        """Terminal reward for an episode end reason ('success', 'offtrack', 'flip', 'stuck')."""
        c = self.cfg
        table = {
            "success": ("completion", c.completion),
            "offtrack": ("offtrack_term", -c.offtrack_term),
            "flip": ("flip", -c.flip),
            "stuck": ("stuck_term", -c.stuck_term),
        }
        if reason not in table:
            return 0.0, {}
        name, val = table[reason]
        return float(val), {name: float(val)}
