"""Gymnasium environment for the drivesim road-vehicle simulator.

Quick use::

    import gymnasium as gym
    from drivesim.env import DriveEnv

    env = DriveEnv(closed=True, difficulty=0.5, seed=3, control_mode="continuous")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Also registered as ``gym.make("Drive-v0", ...)`` (import ``drivesim.env`` first).

Control modes
-------------
``'discrete'``   ``Discrete(12)``. Index ``= steer_idx * 4 + gas * 2 + brake`` with
                 ``steer_idx`` 0 = left, 1 = none, 2 = right. ``gas`` and ``brake`` are
                 0/1 (full throttle or full brake). Automatic transmission. Use
                 :data:`DISCRETE_ACTIONS` for the table.
``'continuous'`` ``Box([-1, 0, 0], [1, 1, 1])`` = ``[steer, throttle, brake]``, automatic.
``'full'``       ``Box([-1, 0, 0, 0, 0, 0, 0], 1)`` =
                 ``[steer, throttle, brake, clutch, handbrake, shift_up, shift_down]``.
                 Shift entries are edge-triggered (a value > 0.5 that was <= 0.5 on the
                 previous step). Transmission is manual by default. Pass
                 ``transmission='auto'`` to keep the handbrake but ignore the clutch/shifts.

Automatic reverse: the vehicle's automatic gearbox selects reverse when the brake is held
at standstill for about 1 s. The env clears the brake while the car is nearly stopped
(|v| < 0.5 m/s), so a policy cannot select reverse by accident. Reversing is still possible
by driving backwards on a slope or with manual reverse in ``'full'`` mode. Such events are
reported in ``info['brake_cleared']`` and are penalised by the reward ``backwards`` term.

Observation: see :data:`OBS_GROUPS` and :data:`OBS_NAMES` (float32, normalised, clipped to
[-10, 10]).

Termination: success (goal reached on an open track, ``laps`` completed on a closed one),
flip, off track beyond ``offtrack_margin`` for ``offtrack_time`` seconds, or stuck (no new
best progress for ``stuck_time`` seconds). Truncation: ``max_steps``.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .reward import RewardCalculator, RewardConfig
from .vehicle import Controls, Vehicle, VehicleParams, PRESETS
from .world import generate_world

# ---------------------------------------------------------------------------
# action / observation layout
# ---------------------------------------------------------------------------
DISCRETE_ACTIONS = tuple(
    (steer, gas, brake)
    for steer in (1.0, 0.0, -1.0)          # steer_idx 0 = left (+1), 1 = none, 2 = right (-1)
    for gas in (0.0, 1.0)
    for brake in (0.0, 1.0)
)
# order of entries in DISCRETE_ACTIONS: index = steer_idx * 4 + gas * 2 + brake

LOOKAHEAD_DISTANCES = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0)   # m along the track
CURVATURE_DISTANCES = (10.0, 30.0, 60.0)                          # m along the track
VEL_SCALE = 40.0          # m/s
YAW_RATE_SCALE = 3.0      # rad/s
ACC_SCALE = 20.0          # m/s^2
LOOK_SCALE = 50.0         # m
CURV_REF = 30.0           # m: curvature * CURV_REF is ~1 at a 30 m radius
RPM_SCALE = 7000.0
OBS_CLIP = 10.0

# (name, size) groups in observation order
OBS_GROUPS = (
    ("forward_speed", 1), ("lateral_speed", 1), ("yaw_rate", 1), ("body_slip", 1),
    ("engine_rpm", 1), ("gear", 1), ("longitudinal_accel", 1), ("lateral_accel", 1),
    ("wheel_slip_ratio", 4),     # FL FR RL RR
    ("wheel_slip_angle", 4),
    ("steer_angle", 1),          # mean front road-wheel angle / max at standstill
    ("prev_steer", 1), ("prev_throttle", 1), ("prev_brake", 1),
    ("prev_handbrake", 1), ("prev_clutch", 1),
    ("lookahead", 2 * len(LOOKAHEAD_DISTANCES)),   # (forward, left) per distance
    ("lateral_offset", 1),       # / road half width
    ("heading_error", 1),        # / pi
    ("curvature_ahead", len(CURVATURE_DISTANCES)),
    ("surface_friction", 4),     # per wheel
    ("wheel_contact", 4),
    ("progress", 1),             # s / length
    ("laps_travelled", 1),       # unwrapped distance / length
    ("is_closed", 1),
    ("goal_vector", 2),          # open track: (forward, left) to B / 100 m; closed: 0
    ("goal_distance", 1),        # open track: distance to B / length; closed: 0
    ("on_track", 1),
)
OBS_NAMES = tuple(
    n for name, size in OBS_GROUPS
    for n in ([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
)
OBS_DIM = sum(size for _, size in OBS_GROUPS)

# default cache of generated worlds, keyed by (seed, closed, difficulty)
_WORLD_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_WORLD_CACHE_MAX = 8


def build_world(seed: int, closed: bool = True, difficulty: float = 0.5):
    """Cached :func:`drivesim.world.generate_world` (about 1 s per uncached call)."""
    key = (int(seed), bool(closed), float(difficulty))
    if key in _WORLD_CACHE:
        _WORLD_CACHE.move_to_end(key)
        return _WORLD_CACHE[key]
    world = generate_world(int(seed), closed=closed, difficulty=difficulty)
    _WORLD_CACHE[key] = world
    while len(_WORLD_CACHE) > _WORLD_CACHE_MAX:
        _WORLD_CACHE.popitem(last=False)
    return world


def _rot_local(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


class DriveEnv(gym.Env):
    """Driving environment on procedurally generated terrain and tracks.

    Parameters
    ----------
    closed : True for a loop circuit, False for point-to-point A -> B.
    difficulty : track difficulty in [0, 1] (more and tighter corners).
    seed : ``None`` regenerates the track at every reset (the seed is drawn from the env RNG,
        which ``reset(seed=...)`` sets). An int gives a fixed track.
    world : optional fixed ``(terrain, track)`` pair. Overrides generation. Use this for
        scripted tracks.
    world_fn : optional callable ``seed -> (terrain, track)`` used instead of the default
        generator. Use this for scripted tracks that vary with the seed.
    dt : physics time per action repeat (default 1/30 s).
    action_repeat : number of ``dt`` physics steps each action is held for per env step.
    max_steps : env steps before truncation.
    control_mode : ``'discrete'``, ``'continuous'`` or ``'full'``.
    transmission : ``'auto'`` or ``'manual'``. Default: manual for ``'full'``, auto otherwise.
    vehicle : ``'sports_car'`` or ``'hatchback'``, or a :class:`VehicleParams`.
    reward : :class:`RewardConfig` (defaults to the documented weights).
    laps : laps to complete on a closed track before success.
    goal_radius : success radius around B on an open track (m).
    offtrack_margin : metres beyond the road edge that count as off track.
    offtrack_time : seconds off track before termination.
    stuck_time : seconds without a new best progress before termination.
    render_mode : ``None``, ``'human'`` or ``'rgb_array'``. Rendering needs
        :class:`drivesim.viewer.Viewer` and is imported lazily.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        closed: bool = True,
        difficulty: float = 0.5,
        seed: Optional[int] = None,
        world=None,
        world_fn: Optional[Callable] = None,
        dt: float = 1.0 / 30.0,
        action_repeat: int = 1,
        max_steps: int = 3000,
        control_mode: str = "continuous",
        transmission: Optional[str] = None,
        vehicle="sports_car",
        reward: Optional[RewardConfig] = None,
        laps: int = 1,
        goal_radius: float = 6.0,
        offtrack_margin: float = 2.0,
        offtrack_time: float = 4.0,
        stuck_time: float = 20.0,
        render_mode: Optional[str] = None,
    ):
        if control_mode not in ("discrete", "continuous", "full"):
            raise ValueError("control_mode must be 'discrete', 'continuous' or 'full'")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'")
        if transmission is None:
            transmission = "manual" if control_mode == "full" else "auto"
        if transmission not in ("auto", "manual"):
            raise ValueError("transmission must be 'auto' or 'manual'")
        if control_mode != "full" and transmission == "manual":
            raise ValueError("manual transmission needs control_mode='full'")
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")

        self.closed = bool(closed)
        self.difficulty = float(difficulty)
        self.seed_value = seed
        self._world_fixed = world
        self._world_fn = world_fn
        self.dt = float(dt)
        self.action_repeat = int(action_repeat)
        self.max_steps = int(max_steps)
        self.control_mode = control_mode
        self.transmission = transmission
        if isinstance(vehicle, VehicleParams):
            self.vehicle_params = vehicle
        else:
            if vehicle not in PRESETS:
                raise ValueError(f"unknown vehicle {vehicle!r}")
            self.vehicle_params = PRESETS[vehicle]()
        self.reward_cfg = reward if reward is not None else RewardConfig()
        self.laps = int(laps)
        self.goal_radius = float(goal_radius)
        self.offtrack_margin = float(offtrack_margin)
        self.offtrack_time = float(offtrack_time)
        self.stuck_time = float(stuck_time)
        self.render_mode = render_mode

        if control_mode == "discrete":
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        elif control_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], np.float32),
                high=np.ones(3, np.float32), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=np.array([-1, 0, 0, 0, 0, 0, 0], np.float32),
                high=np.ones(7, np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32)

        self.car = Vehicle(self.vehicle_params, transmission=transmission)
        self.reward_calc = RewardCalculator(self.reward_cfg)
        self.terrain = None
        self.track = None
        self._viewer = None
        self._last_reward = 0.0
        self.track_seed = None

    # ------------------------------------------------------------------ worlds
    def _make_world(self, track_seed: int):
        if self._world_fixed is not None:
            return self._world_fixed
        if self._world_fn is not None:
            return self._world_fn(track_seed)
        return build_world(track_seed, closed=self.closed, difficulty=self.difficulty)

    # ------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._world_fixed is not None:
            track_seed = -1
        elif self.seed_value is not None:
            track_seed = int(self.seed_value)
        else:
            track_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.track_seed = track_seed
        self.terrain, self.track = self._make_world(track_seed)
        if self.track.closed != self.closed and self._world_fixed is None and self._world_fn is None:
            raise RuntimeError("generated track does not match closed flag")

        start, yaw = self.track.start_pose()
        self.car.reset(start[:2], yaw, self.terrain)
        self._steps = 0
        self._dist = 0.0              # unwrapped centerline distance travelled (m)
        self._best = 0.0
        self._time_since_best = 0.0
        self._off_time = 0.0
        self._prev_up = False
        self._prev_down = False
        self._last_action = np.zeros(5, dtype=float)   # steer, throttle, brake, hb, clutch
        self._last_reward = 0.0
        self._ended = False
        pr = self.track.project(self.car.position[:2], None)
        self._pr = pr
        self._idx = pr.index
        self._last_s = pr.s
        self.reward_calc.reset(self.track, pr.s)
        self.reward_calc.prev_steer = 0.0
        return self._obs(), self._info(extra={"track_seed": self.track_seed})

    def respawn_at(self, s: float, speed: float = 0.0) -> None:
        """Place the car at arc length ``s`` on the track, at rest, facing along the track.

        Used for checkpoints and human resets. It does not end the episode and keeps the
        distance counters. A reset also clears the off-track and stuck timers.
        """
        s = float(s)
        xyz, heading = self.track.point_at(s)
        self.car.reset(xyz[:2], float(heading), self.terrain)
        pr = self.track.project(self.car.position[:2], None)
        self._pr = pr
        self._idx = pr.index
        self._last_s = pr.s
        self.reward_calc.reset(self.track, pr.s)
        self._off_time = 0.0
        self._time_since_best = 0.0
        self._ended = False
        self._last_action = np.zeros(5, dtype=float)

    # -------------------------------------------------------------------- step
    def _decode(self, action):
        """Return (steer, throttle, brake, handbrake, clutch, shift_up, shift_down)."""
        if self.control_mode == "discrete":
            steer, gas, brake = DISCRETE_ACTIONS[int(action)]
            return steer, gas, brake, 0.0, 0.0, False, False
        a = np.asarray(action, dtype=float).reshape(-1)
        steer = float(np.clip(a[0], -1.0, 1.0))
        throttle = float(np.clip(a[1], 0.0, 1.0))
        brake = float(np.clip(a[2], 0.0, 1.0))
        if self.control_mode == "continuous":
            return steer, throttle, brake, 0.0, 0.0, False, False
        clutch = float(np.clip(a[3], 0.0, 1.0))
        hb = float(np.clip(a[4], 0.0, 1.0))
        up = bool(a[5] > 0.5)
        down = bool(a[6] > 0.5)
        return steer, throttle, brake, hb, clutch, up, down

    def step(self, action):
        if self._ended:
            raise RuntimeError("step() called after the episode ended; call reset()")
        steer, throttle, brake, hb, clutch, up, down = self._decode(action)
        if self.control_mode == "full":
            # edge-triggered shifts: only the first physics step of this action sees a request
            up_req = up and not self._prev_up
            down_req = down and not self._prev_down
            self._prev_up, self._prev_down = up, down
        else:
            up_req = down_req = False
        cleared = 0
        total = 0.0
        terminated = False
        reason = None
        terms_sum: dict = {}
        for k in range(self.action_repeat):
            b = brake
            if self.car.speed < 0.5 and b > 0.0 and self.transmission == "auto":
                b = 0.0           # avoid the auto reverse toggle at standstill
                cleared += 1
            ctrl = Controls(steer=steer, throttle=throttle, brake=b, handbrake=hb,
                            clutch=clutch, shift_up=(up_req and k == 0),
                            shift_down=(down_req and k == 0))
            self.car.step(ctrl, self.dt, self.terrain)
            self._time_since_best += self.dt
            r, terms, reason = self._physics_reward(steer, throttle, b)
            total += r
            for kk, vv in terms.items():
                terms_sum[kk] = terms_sum.get(kk, 0.0) + vv
            if reason is not None:
                terminated = True
                break
        self._steps += 1
        self._last_action = np.array([steer, throttle, brake, hb, clutch], dtype=float)
        if terminated:
            tr, tterms = self.reward_calc.terminal(reason)
            total += tr
            terms_sum.update(tterms)
            self._ended = True
        truncated = (not terminated) and self._steps >= self.max_steps
        self._last_reward = total
        obs = self._obs()
        info = self._info(extra={"reward_terms": terms_sum, "brake_cleared": cleared,
                                 "termination": reason})
        return obs, float(total), terminated, truncated, info

    def _physics_reward(self, steer, throttle, brake):
        car, track = self.car, self.track
        pr = track.project(car.position[:2], self._idx)
        self._idx = pr.index
        self._pr = pr
        d = self.reward_calc.delta_progress(pr.s)
        self._dist += d
        if self._dist > self._best + 1.0:
            self._best = self._dist
            self._time_since_best = 0.0
        if abs(pr.lateral) > track.half_width + self.offtrack_margin:
            self._off_time += self.dt
        else:
            self._off_time = 0.0
        r, terms = self.reward_calc.step(
            s=pr.s, lateral=pr.lateral, idx=pr.index, yaw=car.yaw,
            velocity_xy=car.velocity_world[:2], steer=steer, throttle=throttle, brake=brake,
            wheel_slip_ratios=car.wheel_slip_ratios)
        reason = None
        if not np.all(np.isfinite(car.position)):
            reason = "flip"
        elif car.is_flipped():
            reason = "flip"
        elif self._is_success(pr):
            reason = "success"
        elif self._off_time >= self.offtrack_time:
            reason = "offtrack"
        elif self._time_since_best >= self.stuck_time:
            reason = "stuck"
        return r, terms, reason

    def _is_success(self, pr) -> bool:
        if self.track.closed:
            return self._dist >= self.laps * self.track.length - 1e-6
        goal = self.track.goal
        dist = float(np.hypot(self.car.position[0] - goal[0], self.car.position[1] - goal[1]))
        return dist <= self.goal_radius and pr.s >= self.track.length - 3.0 * self.goal_radius

    # ------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        car, track, pr = self.car, self.track, self._pr
        yaw = car.yaw
        s = pr.s
        L = track.length
        steer_max_low = self.vehicle_params.steer_max_low
        cw = self.vehicle_params.steer_max_high
        fr = car.wheel_steer_angles
        parts = [
            [car.forward_speed / VEL_SCALE, car.lateral_speed / VEL_SCALE,
             car.yaw_rate / YAW_RATE_SCALE, car.body_slip_angle / 0.5,
             car.engine_rpm / RPM_SCALE, car.gear / 6.0,
             car.longitudinal_accel / ACC_SCALE, car.lateral_accel / ACC_SCALE],
            np.clip(car.wheel_slip_ratios, -2.0, 2.0),
            np.clip(car.wheel_slip_angles / 0.5, -3.0, 3.0),
            [(0.5 * (fr[0] + fr[1])) / steer_max_low],
            self._last_action,
        ]
        d = np.array(LOOKAHEAD_DISTANCES, dtype=float)
        xyz, _ = track.point_at(s + d)
        fwd, left = _rot_local(xyz[:, 0] - car.position[0], xyz[:, 1] - car.position[1], yaw)
        parts.append(np.clip(np.column_stack([fwd, left]).reshape(-1) / LOOK_SCALE, -2.0, 2.0))
        half = track.half_width
        parts.append([pr.lateral / half, _wrap_angle(track.heading[pr.index] - yaw) / math.pi])
        cd = np.array(CURVATURE_DISTANCES, dtype=float)
        parts.append(np.clip(track.curvature_at(s + cd) * CURV_REF, -3.0, 3.0))
        wp = car.wheel_positions_world()
        parts.append(self._friction(wp))
        parts.append(car.wheel_in_contact.astype(float))
        if track.closed:
            goal_vec = [0.0, 0.0]
            goal_dist = 0.0
        else:
            g = track.goal
            gf, gl = _rot_local(g[0] - car.position[0], g[1] - car.position[1], yaw)
            goal_vec = [gf / 100.0, gl / 100.0]
            goal_dist = math.hypot(g[0] - car.position[0], g[1] - car.position[1]) / L
        parts.append([s / L, self._dist / L, 1.0 if track.closed else 0.0])
        parts.append(goal_vec)
        parts.append([goal_dist, 1.0 if abs(pr.lateral) <= half else 0.0])
        flat = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])
        out = np.clip(np.nan_to_num(flat, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP),
                      -OBS_CLIP, OBS_CLIP).astype(np.float32)
        return out

    def _friction(self, wheel_xyz):
        ids = self.terrain.surface_at(wheel_xyz[:, 0], wheel_xyz[:, 1])
        return np.asarray(ids, dtype=float).reshape(-1)

    def _info(self, extra=None):
        car = self.car
        info = {
            "steps": self._steps,
            "time": self._steps * self.dt * self.action_repeat,
            "speed_kmh": car.speed_kmh,
            "gear": car.gear,
            "progress": float(self._pr.s / self.track.length),
            "distance": float(self._dist),
            "lap": float(self._dist / self.track.length) if self.track.closed else None,
            "lateral": float(self._pr.lateral),
            "on_track": bool(abs(self._pr.lateral) <= self.track.half_width),
            "success": False,
            "termination": None,
        }
        if extra:
            info.update(extra)
            if extra.get("termination") == "success":
                info["success"] = True
        return info

    # ------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None or self._viewer_world is not self.track:
            try:
                from .viewer import Viewer  # imported lazily: rendering is optional
            except ImportError as exc:
                raise ImportError(
                    "render() needs drivesim.viewer.Viewer, which could not be imported: "
                    f"{exc}") from exc
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = Viewer(self.terrain, self.track, width=800, height=500,
                                  headless=(self.render_mode == "rgb_array"))
            self._viewer_world = self.track
        hud = {
            "speed_kmh": self.car.speed_kmh, "gear": self.car.gear, "rpm": self.car.engine_rpm,
            "throttle": float(self._last_action[1]), "brake": float(self._last_action[2]),
            "steer": float(self._last_action[0]), "progress": float(self._pr.s / self.track.length),
            "lap": (self._dist / self.track.length) if self.track.closed else None,
            "reward": self._last_reward, "steps": self._steps,
        }
        return self._viewer.render(self.car, hud)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


try:  # register Drive-v0; re-registration of the same id is ignored
    gym.register(id="Drive-v0", entry_point="drivesim.env:DriveEnv", max_episode_steps=None)
except Exception:  # pragma: no cover
    pass
