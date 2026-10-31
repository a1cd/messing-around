# drivesim vehicle dynamics

`drivesim/vehicle.py` (with `tires.py` and `drivetrain.py`) is a road-vehicle
model: a 6-DOF rigid body with raycast suspension, Magic-Formula tires with combined slip, a
clutch/gearbox/differential driveline, Ackermann steering and brakes. It is written for RL
throughput: the inner loop uses plain Python floats; numpy only appears at the terrain query and in
the public accessors.

## Quick start

```python
from drivesim.vehicle import Vehicle, VehicleParams, Controls, hatchback

car = Vehicle(VehicleParams.sports_car(), transmission="auto")   # or hatchback(), "manual"
car.reset(position_xyz=[0, 0, 0], yaw=0.0, terrain=terrain)      # z is ignored; car is settled on the ground
car.step(Controls(throttle=1.0), dt=1/30, terrain=terrain)
print(car.speed_kmh, car.engine_rpm, car.gear)
```

`Vehicle` lives in `drivesim.vehicle`. (`drivesim/__init__.py` belongs to the terrain code and does not
re-export it.)

## Public API

```python
class Vehicle:
    def __init__(self, params: VehicleParams | None = None, transmission: str = "auto", *,
                 substep_hz: float = 120.0)
    def reset(self, position_xyz, yaw: float, terrain) -> None
    def step(self, controls, dt: float, terrain) -> None      # controls: Controls or dict
    # state (numpy arrays unless noted)
    position: (3,)            CoG, world [m]
    quaternion: (4,)          (w, x, y, z), body -> world
    yaw, pitch, roll: float   [rad], Z-Y-X convention
    velocity_world: (3,)      [m/s]
    velocity_local: (3,)      (forward, lateral, vertical) [m/s]
    angular_velocity: (3,)    world frame [rad/s]
    angular_velocity_body: (3,)
    speed, speed_kmh, forward_speed, lateral_speed: float
    body_slip_angle (alias beta): float   atan2(v_lat, v_fwd) [rad]
    yaw_rate: float           world Z component [rad/s]
    longitudinal_accel, lateral_accel: float  body frame, averaged over the last step [m/s^2]
    engine_rpm: float
    gear: int                 -1 reverse, 0 neutral, 1..6
    clutch_engagement: float  effective engagement used in the last substep
    shifting: bool
    wheel_slip_ratios: (4,)   FL FR RL RR
    wheel_slip_angles: (4,)   [rad]
    wheel_loads: (4,)         normal load [N]
    wheel_in_contact: (4,)    bool
    suspension_compression: (4,)  [m], positive = compressed
    wheel_speeds, wheel_steer_angles, wheel_rotations: (4,)
    def get_state_dict(self) -> dict
    def wheel_positions_world(self) -> (4, 3)   wheel hub centres, for rendering
    def is_flipped(self) -> bool                 body Z axis points below horizontal
```

`Controls` (dataclass, all defaults zero/False):

| field | range | meaning |
|---|---|---|
| `steer` | [-1, 1] | left (+) / right (-); mapped to a road-wheel angle that shrinks with speed, rate limited |
| `throttle` | [0, 1] | |
| `brake` | [0, 1] | service brake, front/rear bias from params |
| `handbrake` | [0, 1] | rear-wheel brake, not modulated by ABS |
| `clutch` | [0, 1] | **manual only**: pedal, 1 = fully pressed (disengaged) |
| `shift_up`, `shift_down` | bool | **manual only**: edge requests, applied at the start of `step` |

Clutch and shift fields are ignored in automatic mode. Out-of-range values are clamped.

`VehicleParams` (dataclass). Presets: `VehicleParams.sports_car()` (default), `VehicleParams.hatchback()`,
and module functions `sports_car()`, `hatchback()`.

| group | fields (defaults = sports car) |
|---|---|
| mass / geometry | `mass` 1300 kg, `inertia` (Ixx, Iyy, Izz) = (550, 2300, 2700), `wheelbase` 2.6, `cg_to_front` 1.25, `track_front/rear` 1.56, `hardpoint_height` 0.05 |
| tires (`TireParams`) | `radius` 0.33, `wheel_inertia` 1.2, `peak_mu` 1.05, `load_sensitivity` 0.12, `nominal_load` 3500 N, `sigma_kappa` 0.12, `sigma_alpha` 0.14 rad, `relax_length` 0.5 m, `B/C/E` 1.2 / 1.35 / -3.0, `rolling_resistance` 0.012 |
| suspension | `suspension_rest_length` 0.33, `spring_rate_front/rear` 42k / 39k N/m, `damper_bump_*` 2.8k / 3.0k, `damper_rebound_*` 3.8k / 4.0k Ns/m, `bump_travel` 0.12, `droop_travel` 0.10, `bump_stop_rate` 150k N/m, `anti_roll_front/rear` 14k / 8k N/m |
| aero | `drag_coefficient` 0.32, `frontal_area` 2.0 m^2, `downforce_area` 0.8 (Cl*A) |
| steering | `steer_max_low` 0.60 rad, `steer_max_high` 0.30 rad (at `steer_speed_ref` 25 m/s), `steer_rate` 3 rad/s, `ackermann` 1.0 |
| brakes | `brake_torque_max` 4200 N m, `brake_front_bias` 0.62, `handbrake_torque` 1000 N m per rear wheel, `abs` False (see limitations) |
| drivetrain | `drive` 'rwd' / 'fwd' / 'awd', `awd_front_split` 0.4, `diff` 'open' / 'lsd', `lsd_lock` 0.35, `lsd_ramp` 8 rad/s, `engine_inertia` 0.25, `torque_curve` ((rpm, N m), ...) peaking 320 N m, `idle_rpm` 850, `redline_rpm` 7000, `stall_rpm` 350, `friction_c0/c1` 12 N m + 0.035 N m s/rad, `gear_ratios` (3.6, 2.2, 1.6, 1.25, 1.0, 0.82), `reverse_ratio` 3.4, `final_drive` 3.5, `clutch_torque` 520 N m, `shift_time` 0.25 s |

Hatchback: 1100 kg, 110 hp, FWD, open diff, different tire/suspension/gearing values (see `VehicleParams.hatchback`).

## Modelling notes

* **Suspension.** Each corner is a spring plus asymmetric damper between a chassis hardpoint and the
  wheel. The hub is found by casting a ray from the hardpoint along the body -Z axis to the terrain
  (using `height_at`/`normal_at` at the hub position), so wheels never penetrate. The ground reaction
  acts along the terrain normal at the contact point (the Bullet raycast-vehicle convention). Bump
  stops, droop limit and anti-roll bars are included. Wheels with the ground beyond the droop limit are
  airborne and carry no load.
  Unsprung mass and tire radial compliance are not modelled (see limitations).
* **Tires.** Magic-Formula shape normalised to peak 1, with a single combined-slip vector
  `s = sqrt((kappa/sigma_k)^2 + (alpha/sigma_a)^2)`: `F = mu(Fz) Fz M(s)`, `Fx = F kappa/(sigma_k s)`,
  `Fy = -F alpha/(sigma_a s)`. Friction is load-sensitive and scaled by `terrain.surface_at`. Lateral
  slip angle relaxes with `relax_length`. Longitudinal slip is updated implicitly, which keeps the
  wheel-tire mode stable at 120 Hz. Wheel spin has its own inertia and brake torque.
* **Drivetrain.** The clutch is solved in closed form each substep: if the clutch can hold, the engine
  and driven wheels are locked to a common speed; otherwise it slips at its torque capacity. The
  engine has a torque curve, idle controller, friction and engine braking, a rev limiter, and stalls
  below `stall_rpm`. Shifts take `shift_time` with the clutch open.
* **Manual transmission.** The clutch engagement is `1 - clutch`. A shift is accepted only when the
  clutch pedal is at least half pressed. The H-pattern order is R, N, 1..6, and each shift moves one
  step.
* **Automatic transmission.** Upshifts happen at 3000 rpm (zero throttle) rising to 6500 rpm (full
  throttle). Downshifts happen below 2100 rpm, and only if the predicted post-downshift rpm stays
  under 90% of redline. Shifts happen only while the clutch is engaged, to avoid cascading. Launch
  engages the clutch from throttle, and the clutch opens if the engine nears idle. **Reverse:** hold the
  brake at standstill (|v| < 0.5 m/s) for 1 s to toggle between drive and reverse. The brake must be
  released before the next toggle. In reverse, throttle drives the car backwards.
* **Steering.** Ackermann geometry for the front wheels, with `ackermann` blending to parallel steer.
  The maximum angle decreases with speed. The actuator is rate limited.
* **Numerics.** Chassis integration is semi-implicit Euler at `substep_hz` (120 Hz default, one
  env step at dt=1/30 is 4 substeps). Quaternions are renormalised every substep. Speed, angular rate
  and suspension compression rate are clamped. If a non-finite state appears, `Vehicle` resets the rates,
  keeps the position where possible, and increments `vehicle.recoveries` (should stay 0).
* **Determinism.** No randomness is used; identical inputs give bit-identical states.

## Performance

Measured on the dev container (single core, Python 3.13), dt = 1/30 s:

* 120 Hz substeps (4 per env step): about 3100-4100 env steps/s.
* 240 Hz substeps (8 per env step): about 1800 env steps/s.

## Known limitations

* No unsprung-mass dynamics or tire radial compliance. Kerb hits are absorbed by the spring-damper
  and the ground-following hub, not by a wheel-bounce mode.
* The combined-slip shape is one Magic-Formula curve shared by the longitudinal and lateral directions
  (peaks set by `sigma_kappa`/`sigma_alpha`). Real tires have different shapes per direction. Tire
  properties are not differentiated per axle.
* No gyroscopic wheel effects, no aerodynamic yaw or wind, no suspension geometry changes (camber,
  toe, caster, bump steer), and no differential/driveshaft inertia beyond the lumped engine inertia.
* `abs=True` enables a slow integrating ABS, which is experimental. In tests it lengthened the braking
  distance, so it is off by default.
* Auto reverse is a simple brake-hold toggle, so a driver (or agent) that holds the brake for 1 s at a
  stop will flip to reverse.
* Engine braking can lock driven wheels under hard lift-off (realistic for RWD cars, but very strong
  with the default gearing).
* The engine stall model is simple (torque-based, rpm-threshold). No clutch wear, no torque converter.
* `reset` ignores the z component of `position_xyz` and settles the car for 1.2 s of simulated time.

## Tests and demo

* `tests/test_vehicle.py` (38 tests, about 12 s): rest and settling, static loads and ride height,
  0-100 km/h (sports car about 6.0 s; hatchback about 11.5 s), top speed, 100-0 km/h braking (about
  41 m, no ABS), locked-wheel braking and load transfer, wheelspin, skidpad (about 0.94 g at 15 m/s
  saturated), handbrake and throttle oversteer, understeer gradient, Ackermann, steering rate,
  manual shift rules and clutch stall, auto reverse, LSD and AWD splits, ledge landing, kerb strike,
  slopes, low-friction braking, 10k random-input stability, determinism, 120 vs 240 Hz convergence,
  and a speed benchmark.
* `scripts/vehicle_demo.py [--out PATH]` runs three scenarios (launch, braking, handbrake slide) and
  writes a PNG of speed, rpm/gear, slip ratio and body slip/yaw rate.
