"""6-DOF road vehicle dynamics for the drivesim RL environment.

Overview
--------
* Chassis: rigid body (mass, diagonal inertia), quaternion orientation, world-frame
  linear velocity, body-frame angular velocity.  Integrated with semi-implicit Euler
  at a fixed substep rate (default 120 Hz).  One ``Vehicle.step(controls, dt, terrain)``
  call performs ``round(dt * substep_hz)`` substeps.
* Suspension: each corner is a spring-damper (asymmetric bump/rebound damping) between a
  chassis hardpoint and the wheel hub.  The hub is placed on the terrain by a ray along
  the body's -Z axis (the "raycast vehicle" approach), so the wheel never penetrates
  the ground and the suspension force is simply spring + damper + bump stop.  Anti-roll
  bars couple left/right corners of each axle.  When the ground is further than the
  droop limit the wheel is airborne and carries no load.
* Tires: Pacejka Magic Formula shape, combined slip (see ``drivesim.tires``), load
  sensitive friction, surface friction multiplier from ``terrain.surface_at``, lateral
  relaxation length, implicit wheel-speed update for stiff longitudinal slip.
* Driveline: engine (torque curve, idle, rev limiter, friction/engine braking), friction
  clutch with slip, 6-speed gearbox + reverse, open or limited-slip differential, RWD /
  FWD / AWD.  Manual or automatic transmission (see ``drivesim.drivetrain``).
* Steering: Ackermann geometry, speed-sensitive maximum angle, rate limited actuator.
* Brakes: front/rear bias, optional ABS-like slip regulation, handbrake on rear wheels.

Frames and conventions
----------------------
* World: Z up.  Body: X forward, Y left, Z up.  Yaw is a rotation about world Z.
* ``position`` is the chassis centre of gravity (CoG) in world coordinates.
* Wheel order everywhere is FL, FR, RL, RR (index 0..3).
* Quaternion is (w, x, y, z), body-to-world.
* Angular velocity is stored in the body frame internally; ``angular_velocity`` returns the
  world-frame value.  Controls are in the ranges given in ``Controls``.

Performance: the inner loop uses Python floats and ``math``; numpy is only used at the
terrain-query boundary and for the public accessors.  Vehicle state for the
renderer is available via ``wheel_positions_world``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .drivetrain import Powertrain, RPM_PER_RADS
from .tires import TireModel, TireParams

GRAVITY = 9.81
AIR_DENSITY = 1.225

_V_DEN = 0.3        # floor for slip-ratio / slip-angle denominators [m/s]
_V_RELAX = 0.5      # floor for relaxation speed [m/s]
_V_RR = 0.3         # scale of rolling-resistance smoothing [m/s]
_MAX_DAMPER_V = 8.0  # clamp on suspension compression rate [m/s]
_MAX_SPEED = 150.0   # safety clamp on chassis speed [m/s]
_MAX_OMEGA = 40.0    # safety clamp on chassis body angular rate [rad/s]
_MIN_HUB_L = 0.02    # lower clamp on suspension length [m]
_ABS_RATE = 25.0    # ABS brake-factor rate [1/s]
# Inverted / on-side handling.  When the body up axis has world-z component Z2 <= _FLIP_Z2 the
# suspension ray (along body -Z) cannot reach the ground, so wheels get no contact.  Instead the
# body box corners below act as penalty contacts (spring-damper along the terrain normal, with
# stick-slip friction).  Offsets are in body coordinates from the CoG [m].
_FLIP_Z2 = 0.05
_NO_RAY_L = 1e4     # suspension length reported when the ray has no ground intersection
_BODY_BOX_Z = (-0.3, 0.85)              # underbody and roof heights above CoG [m]
_BODY_BOX_HALF_W = 0.75                 # half width of the body box [m]
_BODY_K = 1.0e6     # penalty stiffness of a body corner against the ground [N/m]
_BODY_C = 2.5e4     # penalty damping along the ground normal [N s/m]
_BODY_KT = 1.0e6    # tangential (stick) stiffness of a body contact [N/m]
_BODY_CT = 2.0e3    # tangential damping of a body contact [N s/m]
_BODY_MU = 0.7      # body-contact friction coefficient (scaled by terrain surface friction)


# ---------------------------------------------------------------------------
# public data types
# ---------------------------------------------------------------------------
@dataclass
class Controls:
    """Driver inputs, all held constant over one ``Vehicle.step`` call.

    steer      [-1, 1]  left (+) / right (-), scaled by the speed-dependent max steer angle
    throttle   [0, 1]
    brake      [0, 1]   service brakes (front/rear bias from params)
    handbrake  [0, 1]   rear-wheel brake, independent of service brake
    clutch     [0, 1]   MANUAL mode only: clutch pedal, 1 = fully pressed (disengaged)
    shift_up   bool     MANUAL mode only: edge-triggered shift up request
    shift_down bool     MANUAL mode only: edge-triggered shift down request
    Clutch and shift fields are ignored in automatic mode.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: float = 0.0
    clutch: float = 0.0
    shift_up: bool = False
    shift_down: bool = False


@dataclass
class VehicleParams:
    """All physical parameters of a vehicle.  Defaults describe a ~250 hp RWD sports car."""

    name: str = "sports_car"
    # --- mass / inertia / geometry
    mass: float = 1300.0                                 # [kg]
    inertia: Tuple[float, float, float] = (550.0, 2300.0, 2700.0)  # Ixx, Iyy, Izz about CoG [kg m^2]
    wheelbase: float = 2.6                               # [m]
    cg_to_front: float = 1.25                            # CoG to front axle [m]
    track_front: float = 1.56                            # [m]
    track_rear: float = 1.56                             # [m]
    hardpoint_height: float = 0.05                       # spring top height above CoG [m]
    tire: TireParams = field(default_factory=TireParams)

    # --- suspension (per corner; front/rear values)
    suspension_rest_length: float = 0.33                 # free length spring top -> hub [m]
    spring_rate_front: float = 42000.0                   # [N/m]
    spring_rate_rear: float = 39000.0
    damper_bump_front: float = 2800.0                    # compression damping [N s/m]
    damper_bump_rear: float = 3000.0
    damper_rebound_front: float = 3800.0                 # extension damping [N s/m]
    damper_rebound_rear: float = 4000.0
    bump_travel: float = 0.12                            # compression before bump stop [m]
    droop_travel: float = 0.10                           # extension before wheel leaves ground [m]
    bump_stop_rate: float = 150000.0                     # extra stiffness beyond bump travel [N/m]
    anti_roll_front: float = 14000.0                     # roll stiffness per axle [N/m]
    anti_roll_rear: float = 8000.0

    # --- aerodynamics
    drag_coefficient: float = 0.32
    frontal_area: float = 2.0                            # [m^2]
    downforce_area: float = 0.8                          # Cl*A magnitude, pushes car down [m^2]

    # --- steering
    steer_max_low: float = 0.60                          # max wheel angle at standstill [rad]
    steer_max_high: float = 0.30                         # max wheel angle at steer_speed_ref [rad]
    steer_speed_ref: float = 25.0                        # [m/s]
    steer_rate: float = 3.0                              # actuator rate limit [rad/s]
    ackermann: float = 1.0                               # 0 = parallel steer, 1 = full Ackermann

    # --- brakes
    brake_torque_max: float = 4200.0                     # total service brake torque at brake=1 [N m]
    brake_front_bias: float = 0.62                       # fraction of service torque at front axle
    handbrake_torque: float = 1000.0                     # per rear wheel at handbrake=1 [N m]
    abs: bool = False                                    # simple slow ABS (experimental, see README)
    abs_slip: float = 0.12                               # slip ratio above which brake is released

    # --- drivetrain
    drive: str = "rwd"                                   # 'rwd' | 'fwd' | 'awd'
    awd_front_split: float = 0.4                         # front share of torque in AWD
    diff: str = "open"                                   # 'open' | 'lsd' (both axles)
    lsd_lock: float = 0.35                               # max torque bias of LSD (fraction)
    lsd_ramp: float = 8.0                                # speed difference for full LSD lock [rad/s]
    engine_inertia: float = 0.25                         # [kg m^2]
    torque_curve: Tuple[Tuple[float, float], ...] = (
        (1000.0, 180.0), (2000.0, 260.0), (3000.0, 300.0), (4000.0, 320.0),
        (5000.0, 315.0), (6000.0, 290.0), (6500.0, 274.0), (7000.0, 250.0),
    )                                                    # (rpm, N m) at full throttle
    idle_rpm: float = 850.0
    redline_rpm: float = 7000.0                          # fuel cut / auto shift reference
    stall_rpm: float = 350.0
    friction_c0: float = 12.0                            # engine friction, constant [N m]
    friction_c1: float = 0.035                           # engine friction per rad/s [N m s/rad]
    idle_gain: float = 4.0                               # idle controller gain [N m per rad/s]
    idle_torque_max: float = 60.0
    starter_torque: float = 80.0
    gear_ratios: Tuple[float, ...] = (3.6, 2.2, 1.6, 1.25, 1.0, 0.82)
    reverse_ratio: float = 3.4
    final_drive: float = 3.5
    clutch_torque: float = 520.0                         # clutch capacity at full engagement [N m]
    shift_time: float = 0.25                             # [s]

    @classmethod
    def sports_car(cls) -> "VehicleParams":
        return cls()

    @classmethod
    def hatchback(cls) -> "VehicleParams":
        """~1100 kg, ~110 hp front-wheel-drive hatchback."""
        return cls(
            name="hatchback",
            mass=1100.0,
            inertia=(400.0, 1500.0, 1800.0),
            wheelbase=2.5,
            cg_to_front=1.05,
            track_front=1.50,
            track_rear=1.48,
            spring_rate_front=34000.0,
            spring_rate_rear=25000.0,
            damper_bump_front=2000.0,
            damper_bump_rear=1900.0,
            damper_rebound_front=2800.0,
            damper_rebound_rear=2700.0,
            anti_roll_front=12000.0,
            anti_roll_rear=6000.0,
            drag_coefficient=0.31,
            frontal_area=2.1,
            downforce_area=0.0,
            brake_torque_max=2800.0,
            brake_front_bias=0.66,
            handbrake_torque=700.0,
            drive="fwd",
            diff="open",
            engine_inertia=0.18,
            torque_curve=(
                (1000.0, 95.0), (2000.0, 125.0), (3000.0, 140.0), (4000.0, 148.0),
                (5000.0, 140.0), (6000.0, 125.0), (6800.0, 108.0),
            ),
            idle_rpm=800.0,
            redline_rpm=6800.0,
            friction_c0=9.0,
            friction_c1=0.03,
            gear_ratios=(3.5, 2.0, 1.4, 1.1, 0.92, 0.78),
            reverse_ratio=3.4,
            final_drive=4.1,
            clutch_torque=240.0,
        )


def sports_car() -> VehicleParams:
    """Preset: ~1300 kg, ~250 hp rear-wheel-drive sports car."""
    return VehicleParams.sports_car()


def hatchback() -> VehicleParams:
    """Preset: ~1100 kg, ~110 hp front-wheel-drive hatchback."""
    return VehicleParams.hatchback()


PRESETS = {"sports_car": sports_car, "hatchback": hatchback}


class FlatTerrain:
    """Minimal flat terrain implementing the interface Vehicle needs (for tests / demos)."""

    def __init__(self, height: float = 0.0, friction: float = 1.0):
        self.height = float(height)
        self.friction = float(friction)

    def height_at(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.full(x.shape, self.height)

    def normal_at(self, x, y):
        x = np.asarray(x, dtype=float)
        n = np.zeros(x.shape + (3,))
        n[..., 2] = 1.0
        return n

    def surface_at(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.full(x.shape, self.friction)


def _read_controls(c):
    if isinstance(c, dict):
        g = lambda k, d: c.get(k, d)
    else:
        g = lambda k, d: getattr(c, k, d)
    steer = float(g("steer", 0.0))
    throttle = float(g("throttle", 0.0))
    brake = float(g("brake", 0.0))
    hb = float(g("handbrake", 0.0))
    clutch = float(g("clutch", 0.0))
    up = bool(g("shift_up", False))
    down = bool(g("shift_down", False))
    steer = -1.0 if steer < -1.0 else (1.0 if steer > 1.0 else steer)
    throttle = 0.0 if throttle < 0.0 else (1.0 if throttle > 1.0 else throttle)
    brake = 0.0 if brake < 0.0 else (1.0 if brake > 1.0 else brake)
    hb = 0.0 if hb < 0.0 else (1.0 if hb > 1.0 else hb)
    clutch = 0.0 if clutch < 0.0 else (1.0 if clutch > 1.0 else clutch)
    if not math.isfinite(steer + throttle + brake + hb + clutch):
        steer = throttle = brake = hb = clutch = 0.0
    return steer, throttle, brake, hb, clutch, up, down


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# the vehicle
# ---------------------------------------------------------------------------
class Vehicle:
    """Driveable road vehicle.  See the module docstring for conventions.

    Parameters
    ----------
    params : VehicleParams, optional
        Defaults to ``VehicleParams.sports_car()``.
    transmission : {'auto', 'manual'}
        ``'auto'`` handles clutch and shifting; reverse is selected by holding the brake
        at standstill for about one second, drive by applying throttle at standstill.
    substep_hz : float, keyword-only
        Internal integration rate (default 120 Hz).
    """

    def __init__(self, params: VehicleParams | None = None, transmission: str = "auto", *,
                 substep_hz: float = 120.0):
        self.p = params if params is not None else VehicleParams.sports_car()
        p = self.p
        if p.drive not in ("rwd", "fwd", "awd"):
            raise ValueError("drive must be 'rwd', 'fwd' or 'awd'")
        if p.diff not in ("open", "lsd"):
            raise ValueError("diff must be 'open' or 'lsd'")
        self.mode = transmission
        self.tire = TireModel(p.tire)
        self.pt = Powertrain(p, transmission)
        self.substep_hz = float(substep_hz)
        self.R = p.tire.radius
        self.Iw = p.tire.wheel_inertia
        self.Ix, self.Iy, self.Iz = p.inertia
        self.m = p.mass
        self.Ie = p.engine_inertia
        self.clutch_cap = p.clutch_torque

        lf = p.cg_to_front
        lr = p.wheelbase - p.cg_to_front
        hz = p.hardpoint_height
        self.hp = [
            (lf, +p.track_front / 2, hz),   # FL
            (lf, -p.track_front / 2, hz),   # FR
            (-lr, +p.track_rear / 2, hz),   # RL
            (-lr, -p.track_rear / 2, hz),   # RR
        ]
        front_share = lr / p.wheelbase
        W = p.mass * GRAVITY
        k_f, k_r = p.spring_rate_front, p.spring_rate_rear
        self.k = [k_f, k_f, k_r, k_r]
        self.cb = [p.damper_bump_front] * 2 + [p.damper_bump_rear] * 2
        self.cr = [p.damper_rebound_front] * 2 + [p.damper_rebound_rear] * 2
        self.arb = [p.anti_roll_front] * 2 + [p.anti_roll_rear] * 2
        self.L0 = p.suspension_rest_length
        self.droop = p.droop_travel
        self.travel = p.bump_travel
        self.kbs = p.bump_stop_rate
        self.Lmax = self.L0 + p.droop_travel
        # body box corners (underbody and roof, at the axle lines) in body coordinates
        zb, zr = _BODY_BOX_Z
        self._box = (
            (lf, lf, lf, lf, -lr, -lr, -lr, -lr),
            tuple(s * _BODY_BOX_HALF_W for s in (1, -1, 1, -1, 1, -1, 1, -1)),
            tuple(z for z in (zb, zb, zr, zr, zb, zb, zr, zr)),
        )
        share = [front_share, front_share, 1 - front_share, 1 - front_share]
        self.Lstatic = [self.L0 - (W * share[i] / 2.0) / self.k[i] for i in range(4)]
        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self):
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.om = [0.0, 0.0, 0.0]               # body angular velocity
        self.w = [0.0, 0.0, 0.0, 0.0]           # wheel spin rates [rad/s]
        self.wheel_rot = [0.0, 0.0, 0.0, 0.0]
        self.Lprev = list(self.Lstatic)
        self.comp = [0.0, 0.0, 0.0, 0.0]
        self.contact = [False, False, False, False]
        self.load = [0.0, 0.0, 0.0, 0.0]
        self.kappa = [0.0, 0.0, 0.0, 0.0]
        self.abs_f = [1.0, 1.0, 1.0, 1.0]
        self.alpha = [0.0, 0.0, 0.0, 0.0]      # relaxed slip angle used by the tire
        self.steer_act = 0.0
        self.wheel_steer = [0.0, 0.0, 0.0, 0.0]
        self.hub = [(0.0, 0.0, 0.0)] * 4
        self._body_anchor = [None] * 8          # body-contact friction anchors (world points)
        self.a_local = [0.0, 0.0, 0.0]
        self.time = 0.0
        self.steps = 0
        self.recoveries = 0

    def reset(self, position_xyz, yaw, terrain) -> None:
        """Place the car at rest on the ground at (x, y) with heading ``yaw`` [rad].

        The z component of ``position_xyz`` is ignored; the car is put on the terrain
        surface and the suspension is allowed to settle for about 1.2 s of simulated time.
        """
        x, y = float(position_xyz[0]), float(position_xyz[1])
        h0 = float(np.asarray(terrain.height_at(np.array([x]), np.array([y]))).reshape(-1)[0])
        self._init_state()
        self.q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mean_L = sum(self.Lstatic) / 4.0
        self.pos = [x, y, h0 + self.R + mean_L - self.p.hardpoint_height]
        self.Lprev = list(self.Lstatic)
        self.pt.reset()
        self.pt.gear = 0          # settle in neutral with no brakes
        self.pt.pending_gear = 0
        h = 1.0 / self.substep_hz
        for _ in range(int(round(1.2 * self.substep_hz))):
            self._substep(h, 0.0, 0.0, 0.0, 0.0, 0.0, terrain)
        self.vel = [0.0, 0.0, 0.0]
        self.om = [0.0, 0.0, 0.0]
        self.w = [0.0, 0.0, 0.0, 0.0]
        self.pt.reset()
        self.a_local = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ step
    def step(self, controls, dt: float, terrain) -> None:
        """Advance the vehicle by ``dt`` seconds (split into fixed substeps)."""
        steer, throttle, brake, hb, clutch, up, down = _read_controls(controls)
        if not (dt > 0.0) or not math.isfinite(dt):
            return
        n = max(1, int(round(dt * self.substep_hz)))
        h = dt / n
        self.pt.request_shift(up, down, clutch)
        v0 = self.vel[:]
        for _ in range(n):
            self._substep(h, steer, throttle, brake, hb, clutch, terrain)
        self.time += dt
        self.steps += 1
        # body-frame acceleration averaged over this step
        X, Y, Z = self._axes()
        a = [(self.vel[i] - v0[i]) / dt for i in range(3)]
        self.a_local = [a[0] * X[0] + a[1] * X[1] + a[2] * X[2],
                        a[0] * Y[0] + a[1] * Y[1] + a[2] * Y[2],
                        a[0] * Z[0] + a[1] * Z[1] + a[2] * Z[2]]
        if not self._finite():
            self._recover()

    # --------------------------------------------------------------- helpers
    def _axes(self):
        q0, q1, q2, q3 = self.q
        X = (1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 + q3 * q0), 2 * (q1 * q3 - q2 * q0))
        Y = (2 * (q1 * q2 - q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 + q1 * q0))
        Z = (2 * (q1 * q3 + q2 * q0), 2 * (q2 * q3 - q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2))
        return X, Y, Z

    def _finite(self) -> bool:
        vals = self.pos + self.vel + self.q + self.om + self.w
        for v in vals:
            if not math.isfinite(v):
                return False
        return True

    def _recover(self):
        """Last-resort guard: keep position/orientation if sane, zero the rates."""
        self.recoveries += 1
        if not all(math.isfinite(v) for v in self.pos + self.q):
            self._init_state()
            self.pos = [0.0, 0.0, 1.0]
            self.q = [1.0, 0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.om = [0.0, 0.0, 0.0]
        self.w = [0.0, 0.0, 0.0, 0.0]
        self.kappa = [0.0] * 4
        self.alpha = [0.0] * 4
        self.a_local = [0.0, 0.0, 0.0]
        self.pt.engine_w = _clamp(self.pt.engine_w if math.isfinite(self.pt.engine_w) else self.pt.idle_w,
                                  0.0, self.pt.limiter_w)

    def _drive_weights(self):
        p = self.p
        if p.drive == "rwd":
            fs = 0.0
        elif p.drive == "fwd":
            fs = 1.0
        else:
            fs = p.awd_front_split
        rs = 1.0 - fs
        w = [0.0, 0.0, 0.0, 0.0]
        lsd = p.diff == "lsd"
        for (a, b, share, out_a, out_b) in ((0, 1, fs, 0, 1), (2, 3, rs, 2, 3)):
            if share <= 0.0:
                continue
            bias = 0.0
            if lsd:
                s = _clamp((self.w[a] - self.w[b]) / p.lsd_ramp, -1.0, 1.0)
                bias = 0.5 * p.lsd_lock * s
            w[out_a] = share * (0.5 - bias)
            w[out_b] = share * (0.5 + bias)
        return w

    def _brake_torques(self, brake, handbrake, h):
        """Per-wheel brake torque [N m].  Service brake is scaled by an ABS factor per wheel.

        The ABS factor is an integrator: it is reduced while the wheel slips beyond
        ``abs_slip`` and recovers otherwise.  Handbrake torque is never modulated.
        """
        p = self.p
        tb = p.brake_torque_max * brake
        front_each = tb * p.brake_front_bias * 0.5
        rear_each = tb * (1.0 - p.brake_front_bias) * 0.5
        hb = handbrake * p.handbrake_torque
        out = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            svc = front_each if i < 2 else rear_each
            f = self.abs_f[i]
            if p.abs and brake > 0.0 and svc > 0.0:
                if -self.kappa[i] > p.abs_slip:
                    f -= _ABS_RATE * h
                else:
                    f += _ABS_RATE * h
                f = _clamp(f, 0.0, 1.0)
            else:
                f = 1.0
            self.abs_f[i] = f
            out[i] = svc * f + (hb if i >= 2 else 0.0)
        return out

    # --------------------------------------------------------------- substep
    def _substep(self, h, steer_cmd, throttle, brake, handbrake, clutch_pedal, terrain):
        p = self.p
        R = self.R
        tire = self.tire
        (X0, X1, X2), (Y0, Y1, Y2), (Z0, Z1, Z2) = self._axes()
        px, py, pz = self.pos
        vx, vy, vz = self.vel
        ox, oy, oz = self.om
        # world angular velocity
        wx = ox * X0 + oy * Y0 + oz * Z0
        wy = ox * X1 + oy * Y1 + oz * Z1
        wz = ox * X2 + oy * Y2 + oz * Z2
        v_fwd = vx * X0 + vy * X1 + vz * X2

        # ---- steering actuator and Ackermann wheel angles
        smax = p.steer_max_low + (p.steer_max_high - p.steer_max_low) * min(abs(v_fwd) / p.steer_speed_ref, 1.0)
        target = steer_cmd * smax
        dsteer = target - self.steer_act
        lim = p.steer_rate * h
        self.steer_act += _clamp(dsteer, -lim, lim)
        dc = self.steer_act
        if abs(dc) > 1e-6:
            tc = math.tan(dc)
            half_t = 0.5 * p.track_front
            d_in = math.atan(p.wheelbase * tc / (p.wheelbase - half_t * tc))
            d_out = math.atan(p.wheelbase * tc / (p.wheelbase + half_t * tc))
            dpar = dc
            if dc > 0:   # left turn: left wheel is inner
                dL = dpar + p.ackermann * (d_in - dpar)
                dR = dpar + p.ackermann * (d_out - dpar)
            else:
                dL = dpar + p.ackermann * (d_out - dpar)
                dR = dpar + p.ackermann * (d_in - dpar)
        else:
            dL = dR = 0.0
        deltas = (dL, dR, 0.0, 0.0)
        self.wheel_steer = [dL, dR, 0.0, 0.0]

        # ---- suspension geometry: raycast from hardpoints along -Z to the ground
        H = [None] * 4
        for i in range(4):
            hx, hy, hzb = self.hp[i]
            H[i] = (px + hx * X0 + hy * Y0 + hzb * Z0,
                    py + hx * X1 + hy * Y1 + hzb * Z1,
                    pz + hx * X2 + hy * Y2 + hzb * Z2)
        Lp = self.Lprev
        xs = np.array([H[i][0] - Z0 * Lp[i] for i in range(4)])
        ys = np.array([H[i][1] - Z1 * Lp[i] for i in range(4)])
        hg = np.asarray(terrain.height_at(xs, ys), dtype=float).reshape(4).tolist()
        ng = np.asarray(terrain.normal_at(xs, ys), dtype=float).reshape(4, 3)
        sf = np.asarray(terrain.surface_at(xs, ys), dtype=float).reshape(4).tolist()
        nxs = ng[:, 0].tolist()
        nys = ng[:, 1].tolist()
        nzs = ng[:, 2].tolist()

        # The ray along -Z only reaches the ground when Z2 > 0 (body up points up). When the body is
        # inverted or on its side there is no wheel contact; the body box takes over below.
        ray_ok = Z2 > _FLIP_Z2
        zz = Z2 if ray_ok else 1.0
        Ln = [0.0] * 4
        Lraw = [0.0] * 4
        comp_c = [0.0] * 4
        Fraw = [0.0] * 4
        contact = [False] * 4
        hub = [None] * 4
        for i in range(4):
            nzi = nzs[i] if nzs[i] > 0.2 else 0.2
            ground_z = hg[i] + R / nzi
            L = (H[i][2] - ground_z) / zz if ray_ok else _NO_RAY_L
            Lraw[i] = L
            Lmax = self.Lmax
            contact[i] = L <= Lmax
            Lc = L if L > _MIN_HUB_L else _MIN_HUB_L
            if Lc > Lmax:
                Lc = Lmax
            Ln[i] = Lc
            hub[i] = (H[i][0] - Z0 * Lc, H[i][1] - Z1 * Lc, H[i][2] - Z2 * Lc)
            rate = -(Lc - Lp[i]) / h          # compression rate, +ve when compressing
            if rate > _MAX_DAMPER_V:
                rate = _MAX_DAMPER_V
            elif rate < -_MAX_DAMPER_V:
                rate = -_MAX_DAMPER_V
            comp = self.L0 - L
            comp_c[i] = comp if comp > -self.droop else -self.droop
            c = self.cb[i] if rate > 0.0 else self.cr[i]
            F = self.k[i] * comp + c * rate
            if comp > self.travel:
                F += self.kbs * (comp - self.travel)
            Fraw[i] = F
        Fs = [0.0] * 4
        for i in range(4):
            if contact[i]:
                F = Fraw[i] + self.arb[i] * (comp_c[i] - comp_c[i ^ 1])
                Fs[i] = F if F > 0.0 else 0.0
        self.Lprev = Ln
        self.hub = hub

        # ---- tires, wheel dynamics (implicit longitudinal update)
        Fz = [0.0] * 4
        Dl = [0.0] * 4
        Kl = [0.0] * 4
        Den = [self.Iw] * 4
        den_s = [1.0] * 4
        w0 = [0.0] * 4
        Dinv = [0.0] * 4
        Fx0 = [0.0] * 4
        Fy0 = [0.0] * 4
        fvec = [None] * 4
        lvec = [None] * 4
        Pt = [None] * 4
        kap = [0.0] * 4
        wold = list(self.w)
        relax = tire.p.relax_length
        rr = tire.p.rolling_resistance
        for i in range(4):
            if not contact[i]:
                self.alpha[i] = 0.0
                self.kappa[i] = 0.0
                w0[i] = self.w[i]
                Dinv[i] = h / self.Iw
                continue
            nx, ny, nz = nxs[i], nys[i], nzs[i]
            d = deltas[i]
            cd, sd = math.cos(d), math.sin(d)
            fx = cd * X0 + sd * Y0
            fy = cd * X1 + sd * Y1
            fz_ = cd * X2 + sd * Y2
            dot = fx * nx + fy * ny + fz_ * nz
            fx -= dot * nx
            fy -= dot * ny
            fz_ -= dot * nz
            nf = math.sqrt(fx * fx + fy * fy + fz_ * fz_)
            if nf < 1e-9:
                fx, fy, fz_ = X0, X1, X2
                nf = 1.0
            fx /= nf
            fy /= nf
            fz_ /= nf
            lx = ny * fz_ - nz * fy
            ly = nz * fx - nx * fz_
            lz = nx * fy - ny * fx
            fvec[i] = (fx, fy, fz_)
            lvec[i] = (lx, ly, lz)
            hub_i = hub[i]
            Pp = (hub_i[0], hub_i[1], hg[i])
            Pt[i] = Pp
            rx, ry, rz = Pp[0] - px, Pp[1] - py, Pp[2] - pz
            # contact-point velocity = v + w x r
            vpx = vx + (wy * rz - wz * ry)
            vpy = vy + (wz * rx - wx * rz)
            vpz = vz + (wx * ry - wy * rx)
            vlong = vpx * fx + vpy * fy + vpz * fz_
            vlat = vpx * lx + vpy * ly + vpz * lz
            Fn = Fs[i] * max(nx * Z0 + ny * Z1 + nz * Z2, 0.0)
            Fz[i] = Fn
            D = tire.friction_limit(Fn, sf[i])
            Dl[i] = D
            wi = self.w[i]
            avl = abs(vlong)
            vabs_den = avl if avl > _V_DEN else _V_DEN
            wr = wi * R
            aw = abs(wr)
            den = avl if avl > aw else aw
            if den < _V_DEN:
                den = _V_DEN
            k_ = (wr - vlong) / den
            kap[i] = k_
            self.kappa[i] = k_
            alpha_t = math.atan2(vlat, vabs_den)
            vr = avl if avl > _V_RELAX else _V_RELAX
            a_relax = 1.0 - math.exp(-vr * h / relax)
            a_s = self.alpha[i] + a_relax * (alpha_t - self.alpha[i])
            self.alpha[i] = a_s
            ffx, ffy = tire.combined_forces(D, k_, a_s)
            rrf = -rr * Fn * _clamp(vlong / _V_RR, -1.0, 1.0)
            ffx += rrf
            Fx0[i] = ffx
            Fy0[i] = ffy
            K = D * tire.k0 / tire.p.sigma_kappa
            Kl[i] = K
            den_s[i] = den
            Dn = self.Iw + h * K * R * R / den
            Den[i] = Dn
            w0[i] = wi + h * (-R * ffx) / Dn
            Dinv[i] = h / Dn

        # ---- drivetrain: clutch coupling, engine
        pt = self.pt
        G = pt.ratio()
        weights = self._drive_weights()
        # engine speed implied by the chassis speed in the current gear (not the wheel speed,
        # which brakes can lock); used by the automatic clutch launch logic
        pt.update(h, throttle, brake, clutch_pedal, v_fwd, abs(G) * abs(v_fwd) / R)
        Te, Fe = pt.engine_torque(throttle)
        we = pt.engine_w
        wwa0 = 0.0
        Ieff = 0.0
        for i in range(4):
            wt = weights[i]
            if wt != 0.0:
                wwa0 += wt * w0[i]
                Ieff += wt * wt * Dinv[i]
        cap = self.clutch_cap * pt.engage_eff() if (pt.engine_on and G != 0.0) else 0.0
        Tc = 0.0
        if cap > 0.0:
            num = we - G * wwa0 + h * (Te - Fe) / self.Ie
            dd = h / self.Ie + G * G * Ieff
            Tlock = num / dd
            if Tlock > cap:
                Tc = cap
            elif Tlock < -cap:
                Tc = -cap
            else:
                Tc = Tlock
        new_we = we + h * (Te - Tc - Fe) / self.Ie
        if new_we < 0.0:
            new_we = 0.0
        elif new_we > pt.limiter_w * 1.1:
            new_we = pt.limiter_w * 1.1
        pt.engine_w = new_we

        # ---- wheel speeds (drive, brake, impulse-limited stick) and tire forces
        brakes = self._brake_torques(brake, handbrake, h)
        Fx_f = [0.0] * 4
        Fy_f = [0.0] * 4
        for i in range(4):
            wn = w0[i] + Dinv[i] * (Tc * G * weights[i])
            imp = h * brakes[i] / self.Iw
            if abs(wn) <= imp:
                wf = 0.0
            else:
                wf = wn - math.copysign(imp, wn)
            self.w[i] = wf
            if contact[i]:
                Fx = Fx0[i] + Kl[i] * R / den_s[i] * (wf - wold[i])
                Fy = Fy0[i]
                D = Dl[i]
                mag = math.sqrt(Fx * Fx + Fy * Fy)
                if D <= 0.0:
                    Fx = Fy = 0.0
                elif mag > D:
                    s_ = D / mag
                    Fx *= s_
                    Fy *= s_
                Fx_f[i] = Fx
                Fy_f[i] = Fy
            self.wheel_rot[i] += wf * h

        # ---- chassis forces and torques (world frame)
        Fx_t = 0.0
        Fy_t = 0.0
        Fz_t = -p.mass * GRAVITY
        Tq_x = 0.0
        Tq_y = 0.0
        Tq_z = 0.0
        for i in range(4):
            if not contact[i]:
                self.load[i] = 0.0
                self.contact[i] = False
                self.comp[i] = self.L0 - Lraw[i] if Lraw[i] < 1e3 else 0.0
                continue
            self.contact[i] = True
            self.load[i] = Fz[i]
            self.comp[i] = self.L0 - Lraw[i]
            Pp = Pt[i]
            rx, ry, rz = Pp[0] - px, Pp[1] - py, Pp[2] - pz
            # suspension (ground reaction) force along the contact normal, at the contact point
            Fs_i = Fs[i]
            nx, ny, nz = nxs[i], nys[i], nzs[i]
            sx, sy, sz = Fs_i * nx, Fs_i * ny, Fs_i * nz
            Fx_t += sx
            Fy_t += sy
            Fz_t += sz
            Tq_x += ry * sz - rz * sy
            Tq_y += rz * sx - rx * sz
            Tq_z += rx * sy - ry * sx
            # tire forces at contact point
            fvx, fvy, fvz = fvec[i]
            lvx, lvy, lvz = lvec[i]
            tfx = Fx_f[i] * fvx + Fy_f[i] * lvx
            tfy = Fx_f[i] * fvy + Fy_f[i] * lvy
            tfz = Fx_f[i] * fvz + Fy_f[i] * lvz
            Fx_t += tfx
            Fy_t += tfy
            Fz_t += tfz
            Tq_x += ry * tfz - rz * tfy
            Tq_y += rz * tfx - rx * tfz
            Tq_z += rx * tfy - ry * tfx
        # body-box penalty contacts, only when the body is inverted or on its side (no wheel ray)
        if not ray_ok:
            anchors = self._body_anchor
            bxs, bys, bzs = self._box
            Px = [px + a * X0 + b * Y0 + c * Z0 for a, b, c in zip(bxs, bys, bzs)]
            Py = [py + a * X1 + b * Y1 + c * Z1 for a, b, c in zip(bxs, bys, bzs)]
            Pz = [pz + a * X2 + b * Y2 + c * Z2 for a, b, c in zip(bxs, bys, bzs)]
            qx = np.array(Px)
            qy = np.array(Py)
            gh = np.asarray(terrain.height_at(qx, qy), dtype=float).reshape(8).tolist()
            gn = np.asarray(terrain.normal_at(qx, qy), dtype=float).reshape(8, 3).tolist()
            gs = np.asarray(terrain.surface_at(qx, qy), dtype=float).reshape(8).tolist()
            for k in range(8):
                nx_, ny_, nz_ = gn[k]
                # signed distance of the corner above the terrain plane through its ground point
                dist = nx_ * (Px[k] - qx[k]) + ny_ * (Py[k] - qy[k]) + nz_ * (Pz[k] - gh[k])
                if dist >= 0.0:
                    anchors[k] = None
                    continue
                pen = -dist
                rx, ry, rz = Px[k] - px, Py[k] - py, Pz[k] - pz
                pvx = vx + (wy * rz - wz * ry)
                pvy = vy + (wz * rx - wx * rz)
                pvz = vz + (wx * ry - wy * rx)
                vn = pvx * nx_ + pvy * ny_ + pvz * nz_
                Fn = _BODY_K * pen - _BODY_C * vn
                if Fn <= 0.0:
                    anchors[k] = None
                    continue
                # stick-slip friction: a tangential spring to an anchor point (static friction),
                # capped at mu * Fn (sliding friction); slipping moves the anchor
                tvx = pvx - vn * nx_
                tvy = pvy - vn * ny_
                tvz = pvz - vn * nz_
                an = anchors[k]
                if an is None:
                    an = (Px[k], Py[k], Pz[k])
                dx_ = Px[k] - an[0]
                dy_ = Py[k] - an[1]
                dz_ = Pz[k] - an[2]
                dn = dx_ * nx_ + dy_ * ny_ + dz_ * nz_
                dx_ -= dn * nx_
                dy_ -= dn * ny_
                dz_ -= dn * nz_
                sx_ = -(_BODY_KT * dx_ + _BODY_CT * tvx)
                sy_ = -(_BODY_KT * dy_ + _BODY_CT * tvy)
                sz_ = -(_BODY_KT * dz_ + _BODY_CT * tvz)
                cap = _BODY_MU * gs[k] * Fn
                ss = math.sqrt(sx_ * sx_ + sy_ * sy_ + sz_ * sz_)
                if ss > cap:
                    sc = cap / ss
                    sx_ *= sc
                    sy_ *= sc
                    sz_ *= sc
                    # slip: re-anchor so that the spring carries exactly the capped force
                    ex_ = -(sx_ + _BODY_CT * tvx) / _BODY_KT
                    ey_ = -(sy_ + _BODY_CT * tvy) / _BODY_KT
                    ez_ = -(sz_ + _BODY_CT * tvz) / _BODY_KT
                    anchors[k] = (Px[k] - ex_, Py[k] - ey_, Pz[k] - ez_)
                else:
                    anchors[k] = an
                fx_ = Fn * nx_ + sx_
                fy_ = Fn * ny_ + sy_
                fz_ = Fn * nz_ + sz_
                Fx_t += fx_
                Fy_t += fy_
                Fz_t += fz_
                Tq_x += ry * fz_ - rz * fy_
                Tq_y += rz * fx_ - rx * fz_
                Tq_z += rx * fy_ - ry * fx_

        else:
            self._body_anchor = [None] * 8

        # aerodynamics and rolling (chassis-level part)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        qa = 0.5 * AIR_DENSITY
        drag = qa * p.drag_coefficient * p.frontal_area * speed
        Fx_t -= drag * vx
        Fy_t -= drag * vy
        Fz_t -= drag * vz
        df = qa * p.downforce_area * speed * speed
        Fx_t -= df * Z0
        Fy_t -= df * Z1
        Fz_t -= df * Z2

        # ---- integrate chassis (semi-implicit Euler)
        im = 1.0 / self.m
        vx += h * Fx_t * im
        vy += h * Fy_t * im
        vz += h * Fz_t * im
        if speed > _MAX_SPEED:
            s_ = _MAX_SPEED / speed
            vx *= s_
            vy *= s_
            vz *= s_
        self.vel[0] = vx
        self.vel[1] = vy
        self.vel[2] = vz
        px += h * vx
        py += h * vy
        pz += h * vz
        self.pos[0] = px
        self.pos[1] = py
        self.pos[2] = pz

        # torques into body frame
        tb0 = Tq_x * X0 + Tq_y * X1 + Tq_z * X2
        tb1 = Tq_x * Y0 + Tq_y * Y1 + Tq_z * Y2
        tb2 = Tq_x * Z0 + Tq_y * Z1 + Tq_z * Z2
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
        ox = ox + h * (tb0 - (Iz - Iy) * oy * oz) / Ix
        oy = oy + h * (tb1 - (Ix - Iz) * oz * ox) / Iy
        oz = oz + h * (tb2 - (Iy - Ix) * ox * oy) / Iz
        ox = _clamp(ox, -_MAX_OMEGA, _MAX_OMEGA)
        oy = _clamp(oy, -_MAX_OMEGA, _MAX_OMEGA)
        oz = _clamp(oz, -_MAX_OMEGA, _MAX_OMEGA)
        self.om[0] = ox
        self.om[1] = oy
        self.om[2] = oz

        # quaternion: q <- normalise(q + h/2 * q (0, w_body))
        q0, q1, q2, q3 = self.q
        dq0 = -q1 * ox - q2 * oy - q3 * oz
        dq1 = q0 * ox + q2 * oz - q3 * oy
        dq2 = q0 * oy + q3 * ox - q1 * oz
        dq3 = q0 * oz + q1 * oy - q2 * ox
        hh = 0.5 * h
        q0 += hh * dq0
        q1 += hh * dq1
        q2 += hh * dq2
        q3 += hh * dq3
        nq = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if nq > 0.0:
            q0 /= nq
            q1 /= nq
            q2 /= nq
            q3 /= nq
        self.q = [q0, q1, q2, q3]

    # --------------------------------------------------------------- accessors
    @property
    def position(self) -> np.ndarray:
        return np.array(self.pos, dtype=float)

    @property
    def quaternion(self) -> np.ndarray:
        return np.array(self.q, dtype=float)

    def _euler(self):
        q0, q1, q2, q3 = self.q
        yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        s = 2 * (q0 * q2 - q3 * q1)
        s = _clamp(s, -1.0, 1.0)
        pitch = math.asin(s)
        roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        return yaw, pitch, roll

    @property
    def yaw(self) -> float:
        return self._euler()[0]

    @property
    def pitch(self) -> float:
        return self._euler()[1]

    @property
    def roll(self) -> float:
        return self._euler()[2]

    @property
    def velocity_world(self) -> np.ndarray:
        return np.array(self.vel, dtype=float)

    @property
    def velocity_local(self) -> np.ndarray:
        """(forward, lateral, vertical) body-frame velocity [m/s]."""
        X, Y, Z = self._axes()
        v = self.vel
        return np.array([v[0] * X[0] + v[1] * X[1] + v[2] * X[2],
                         v[0] * Y[0] + v[1] * Y[1] + v[2] * Y[2],
                         v[0] * Z[0] + v[1] * Z[1] + v[2] * Z[2]], dtype=float)

    @property
    def angular_velocity(self) -> np.ndarray:
        """World-frame angular velocity [rad/s]."""
        X, Y, Z = self._axes()
        o = self.om
        return np.array([o[0] * X[0] + o[1] * Y[0] + o[2] * Z[0],
                         o[0] * X[1] + o[1] * Y[1] + o[2] * Z[1],
                         o[0] * X[2] + o[1] * Y[2] + o[2] * Z[2]], dtype=float)

    @property
    def angular_velocity_body(self) -> np.ndarray:
        return np.array(self.om, dtype=float)

    @property
    def speed(self) -> float:
        v = self.vel
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def forward_speed(self) -> float:
        return float(self.velocity_local[0])

    @property
    def lateral_speed(self) -> float:
        return float(self.velocity_local[1])

    @property
    def body_slip_angle(self) -> float:
        """Body slip angle beta = atan2(v_lateral, v_forward) [rad]; positive when moving left of heading."""
        vl = self.velocity_local
        return math.atan2(vl[1], vl[0])

    @property
    def beta(self) -> float:
        return self.body_slip_angle

    @property
    def engine_rpm(self) -> float:
        return self.pt.engine_w * RPM_PER_RADS

    @property
    def gear(self) -> int:
        return int(self.pt.gear)

    @property
    def clutch_engagement(self) -> float:
        return float(self.pt.engage_eff())

    @property
    def shifting(self) -> bool:
        return self.pt.shift_timer > 0.0

    @property
    def wheel_slip_ratios(self) -> np.ndarray:
        return np.array(self.kappa, dtype=float)

    @property
    def wheel_slip_angles(self) -> np.ndarray:
        return np.array(self.alpha, dtype=float)

    @property
    def wheel_loads(self) -> np.ndarray:
        return np.array(self.load, dtype=float)

    @property
    def wheel_in_contact(self) -> np.ndarray:
        return np.array(self.contact, dtype=bool)

    @property
    def suspension_compression(self) -> np.ndarray:
        return np.array(self.comp, dtype=float)

    @property
    def wheel_speeds(self) -> np.ndarray:
        """Wheel spin rates [rad/s] for FL, FR, RL, RR."""
        return np.array(self.w, dtype=float)

    @property
    def wheel_steer_angles(self) -> np.ndarray:
        """Road-wheel steer angles [rad] for FL, FR, RL, RR (rear = 0)."""
        return np.array(self.wheel_steer, dtype=float)

    @property
    def wheel_rotations(self) -> np.ndarray:
        """Integrated wheel spin angle [rad] for the renderer."""
        return np.array(self.wheel_rot, dtype=float)

    @property
    def yaw_rate(self) -> float:
        """Yaw rate about world Z [rad/s]."""
        return float(self.angular_velocity[2])

    @property
    def longitudinal_accel(self) -> float:
        """Body-frame forward acceleration averaged over the last step [m/s^2] (kinematic, no gravity)."""
        return float(self.a_local[0])

    @property
    def lateral_accel(self) -> float:
        """Body-frame lateral acceleration (positive left) averaged over the last step [m/s^2]."""
        return float(self.a_local[1])

    def wheel_positions_world(self) -> np.ndarray:
        """(4, 3) wheel hub centres in world coordinates, FL FR RL RR."""
        return np.array(self.hub, dtype=float)

    def is_flipped(self) -> bool:
        """True when the body's up axis points downwards (car on its roof or side past vertical)."""
        return (1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2)) < 0.0

    def get_state_dict(self) -> dict:
        yaw, pitch, roll = self._euler()
        return {
            "position": self.position,
            "quaternion": self.quaternion,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "velocity_world": self.velocity_world,
            "velocity_local": self.velocity_local,
            "angular_velocity": self.angular_velocity,
            "speed": self.speed,
            "speed_kmh": self.speed_kmh,
            "forward_speed": self.forward_speed,
            "lateral_speed": self.lateral_speed,
            "body_slip_angle": self.body_slip_angle,
            "yaw_rate": self.yaw_rate,
            "longitudinal_accel": self.longitudinal_accel,
            "lateral_accel": self.lateral_accel,
            "engine_rpm": self.engine_rpm,
            "gear": self.gear,
            "clutch_engagement": self.clutch_engagement,
            "shifting": self.shifting,
            "wheel_speeds": self.wheel_speeds,
            "wheel_slip_ratios": self.wheel_slip_ratios,
            "wheel_slip_angles": self.wheel_slip_angles,
            "wheel_loads": self.wheel_loads,
            "wheel_in_contact": self.wheel_in_contact,
            "suspension_compression": self.suspension_compression,
            "wheel_steer_angles": self.wheel_steer_angles,
            "is_flipped": self.is_flipped(),
            "time": self.time,
        }
