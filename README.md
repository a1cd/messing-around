# messing-around: drivesim

A 3D driving simulator for reinforcement learning: procedural terrain and tracks, a 6-DOF
vehicle model, a Gymnasium environment with a documented reward, and a hand-coded baseline.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Run the environment:

```python
from drivesim.env import DriveEnv

env = DriveEnv(closed=True, difficulty=0.5, seed=3, control_mode="continuous")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.0, 1.0, 0.0])  # steer, throttle, brake
```

### Minimal RL loop

```python
import numpy as np
from drivesim.env import DriveEnv

env = DriveEnv(closed=False, difficulty=0.3, control_mode="continuous")  # A -> B
for episode in range(3):
    obs, info = env.reset(seed=episode)
    done, ret = False, 0.0
    while not done:
        action = env.action_space.sample()          # replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        done = terminated or truncated
    print(episode, round(ret, 1), info["termination"], info["success"])
```

Baseline check (pure pursuit + curvature speed control):

```bash
python -m drivesim.baselines            # 5 seeds x 3 difficulties x {loop, A->B}
```

### Driving yourself

_To be added (keyboard/gamepad driving with `scripts/drive.py`)._

### Viewer

_To be added (pygame software 3D renderer; `render_mode='human'` / `'rgb_array'`)._

## Environment

`drivesim.env.DriveEnv(closed=True, difficulty=0.5, seed=None, world=None, world_fn=None,
dt=1/30, action_repeat=1, max_steps=3000, control_mode="continuous", transmission=None,
vehicle="sports_car", reward=None, laps=1, goal_radius=6.0, offtrack_margin=2.0,
offtrack_time=4.0, stuck_time=20.0, render_mode=None)`

Also available as `gym.make("Drive-v0", ...)` after `import drivesim.env`.

* `closed=True` loop circuit, `False` point-to-point A -> B.
* `seed=None`: a new track every reset (seed drawn from the env RNG, set by `reset(seed=...)`).
  `seed=int`: fixed track. `world=(terrain, track)` or `world_fn(seed) -> (terrain, track)`
  plug in scripted tracks.
* `dt` physics time per step; `action_repeat` holds each action for that many `dt` physics steps.

### Action spaces

| `control_mode` | action space | meaning | transmission |
|---|---|---|---|
| `discrete` | `Discrete(12)` | index `= steer_idx*4 + gas*2 + brake`, `steer_idx`: 0 left, 1 none, 2 right; gas/brake 0 or 1 | auto |
| `continuous` | `Box([-1,0,0], [1,1,1])` | `[steer, throttle, brake]` | auto |
| `full` | `Box([-1,0,0,0,0,0,0], 1)` | `[steer, throttle, brake, clutch, handbrake, shift_up, shift_down]` | manual (or `transmission="auto"`) |

Shift entries in `full` are edge-triggered: a value above 0.5 after a value at or below 0.5 is one request.
In auto mode the env clears the brake while the car is nearly stopped (|v| < 0.5 m/s), so a policy
cannot toggle into reverse by holding the brake. `info["brake_cleared"]` counts these events.

### Observation (float32, 56 values, clipped to [-10, 10])

| group | size | scaling |
|---|---|---|
| forward speed, lateral speed | 2 | / 40 m/s |
| yaw rate | 1 | / 3 rad/s |
| body slip angle | 1 | / 0.5 rad |
| engine rpm, gear | 2 | / 7000, / 6 |
| longitudinal and lateral acceleration | 2 | / 20 m/s^2 |
| per-wheel slip ratio (FL FR RL RR) | 4 | clipped to [-2, 2] |
| per-wheel slip angle | 4 | / 0.5 rad |
| steer angle (front road wheels) | 1 | / max steer at standstill |
| previous steer, throttle, brake, handbrake, clutch | 5 | applied values |
| lookahead centerline points at 5, 10, 20, 30, 45, 60, 80 m | 14 | (forward, left) in car frame, / 50 m |
| lateral offset from centerline | 1 | / road half width |
| heading error (track tangent minus yaw) | 1 | / pi |
| track curvature at 10, 30, 60 m ahead | 3 | curvature * 30 m |
| surface friction under each wheel | 4 | 1.0 asphalt ... 0.2 ice |
| wheel in contact (per wheel) | 4 | 0 or 1 |
| progress along track | 1 | s / length |
| laps travelled | 1 | unwrapped distance / length |
| is closed track | 1 | 0 or 1 |
| goal vector (open track only) | 2 | (forward, left) to B / 100 m, zero on loops |
| goal distance (open track only) | 1 | / length, zero on loops |
| on track | 1 | 0 or 1 |

`drivesim.env.OBS_NAMES` lists the names in order.

### Termination and truncation

| event | flag | `info["termination"]` |
|---|---|---|
| goal reached (open) or `laps` completed (closed) | terminated | `success` |
| flipped | terminated | `flip` |
| off track beyond `offtrack_margin` (2 m past the road edge) for `offtrack_time` (4 s) | terminated | `offtrack` |
| no new best progress for `stuck_time` (20 s) | terminated | `stuck` |
| `max_steps` reached | truncated | `None` |

### Reward (`drivesim/reward.py`, weights in `RewardConfig`)

| term | per step | default weight |
|---|---|---|
| progress along the centerline | `progress * delta_s` (m, loop-aware) | 0.1 per m |
| speed along track tangent | `speed * clip(v/30, -1, 1)` | 0.02 |
| driving backwards (v < -0.5 m/s) | `-backwards` | 0.1 |
| beyond road edge | `-(offroad_step + offroad_per_m * excess_m)` | 0.2 + 0.05/m |
| heading error | `-heading * abs(error_rad)` | 0.02 |
| steer change | `-steer_rate * abs(d steer)` | 0.05 |
| throttle + brake change | `-pedal_rate * (abs(d throttle) + abs(d brake))` | 0.03 |
| steer magnitude | `-steer_mag * abs(steer)` | 0.005 |
| wheel slip (optional) | `-wheelspin * sum(max(0, abs(slip) - 0.2))` | 0 (off) |
| time | `-time` | 0.01 |
| success (terminal) | `+completion` | 100 |
| off track (terminal) | `-offtrack_term` | 10 |
| flip (terminal) | `-flip` | 20 |
| stuck (terminal) | `-stuck_term` | 5 |

Progress is measured along the centerline, not as raw speed. Leaving the road costs more per
step than the progress gained, so the policy learns to brake for corners.

### Baseline results

`python -m drivesim.baselines` (pure pursuit, continuous actions, 5 seeds per row):

| track | difficulty | episodes | success | mean return | mean time |
|---|---|---|---|---|---|
| loop | 0.0 | 5 | 100% | 183.5 | 33.9 s |
| loop | 0.5 | 5 | 100% | 174.2 | 40.4 s |
| loop | 1.0 | 5 | 100% | 165.4 | 50.9 s |
| A -> B | 0.0 | 5 | 100% | 138.1 | 16.9 s |
| A -> B | 0.5 | 5 | 100% | 114.2 | 31.9 s |
| A -> B | 1.0 | 5 | 100% | 139.8 | 35.0 s |

Discrete-action version of the same controller (3 seeds, difficulty 0.5): loop 0% success (all
episodes leave the road), A -> B 100%.

## Known limitations

* Environment throughput is about 450-500 steps/s (single core, dt = 1/30). The vehicle model alone
  runs about 3400 steps/s, so most of the time goes to observation and projection.
* Track generation costs about 0.6-1 s per new seed. Fixed seeds are cached (8 worlds).
* The discrete baseline (binary steer, throttle and brake) succeeds on A -> B tracks but leaves the road
  on loops. Coarse steering, not the vehicle, limits it.
* There are no obstacles, so "crash" means flipping or leaving the road.
* Reversing is not in the action space. The env blocks auto-reverse at standstill, and manual reverse
  is available only through the gearbox in `full` mode.
