# CLAUDE.md

Orientation for Claude Code sessions in this repo. Check `git log` and `drivesim/README_*.md` for the latest state.

## Project overview

`drivesim` is a 3D driving simulator for reinforcement learning. It has procedural terrain and tracks,
a 6-DOF vehicle model (suspension, Pacejka-style tires, clutch/gearbox), a headless software viewer,
a Gymnasium environment (`DriveEnv`, also `gym.make("Drive-v0")`) with a documented reward, a
pure-pursuit baseline, and a human driving layer (keyboard/gamepad). Python 3.13 in the dev container.

## Repo layout

Package `drivesim/` (one line per module):

- `terrain.py`: heightfield with surface types (asphalt, grass, gravel, mud, ice), fBm noise, `height_at`/`normal_at`.
- `track.py`: closed loop or A->B centreline, projection, lookahead, curvature, `start_pose`.
- `world.py`: `generate_world(seed, closed, difficulty)` and `carve_track`, which grades terrain under the road.
- `vehicle.py`: `Vehicle`, `VehicleParams` (presets `sports_car`, `hatchback`), `Controls`, integration.
- `tires.py`: Magic-Formula tire forces with combined slip.
- `drivetrain.py`: engine, clutch, gearbox (manual and auto), differential.
- `env.py`: `DriveEnv`, action decoding, 56-value observation, termination, `OBS_NAMES`, world cache.
- `reward.py`: `RewardConfig` (weights) and `RewardCalculator`.
- `baselines.py`: `PurePursuitController`, `evaluate`, `main` (`python -m drivesim.baselines`).
- `viewer.py`: numpy + pygame software 3D renderer, chase/hood/top cameras, `save_frames`.
- `human.py`: `HumanInput`, `HumanTuning`, `DeviceReader`, `RawInput`; never imports pygame.
- `track_script.py`: scripted segment-based track builder (`TrackScript`, `NAMED_SCRIPTS` incl. `showcase`); see `drivesim/README_track_script.md`.
- `rl/`: PPO training (`ppo.py` torch, `ppo_mlx.py` MLX, `vec_env.py`, `plot.py`) and the shipped DAgger-warm-started policy. See `rl/README_rl.md`.

Other directories:

- `scripts/`: `drive.py` (drive it yourself, `--track showcase` for the scripted circuit), `vehicle_demo.py` (writes a PNG), `preview_world.py` (writes a world PNG), `plot_showcase_track.py` (writes `docs/showcase_track.png`), `train.py` (PPO), `bc_pretrain.py` (behaviour cloning + DAgger warm start for PPO).
- `tests/`: `conftest.py` plus `test_env`, `test_human`, `test_reward`, `test_vehicle`, `test_viewer`, `test_world`.
- `drivesim/README_{world,vehicle,viewer,driving,track_script}.md`: design docs with API and limitations sections.
- `.claude/`: SessionStart hook (`hooks/session-start.sh`) and `settings.json`.

## Setup and commands

Setup (macOS or Linux, from the repo root):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt          # numpy, scipy, matplotlib, pytest, gymnasium, pygame
python -m pip install -r requirements-train.txt    # optional: torch, plus mlx on Apple Silicon
```

The SessionStart hook creates `.venv/` and installs requirements automatically in local sessions.

Commands:

```bash
python -m pytest                       # use `python -m pytest`, not bare `pytest` (may use a system Python without numpy)
python -m pytest tests/test_env.py -q  # one file
python -m pytest --co -q               # list tests (237 collected)
python scripts/drive.py                # drive yourself (keyboard/gamepad); --help lists flags
python scripts/drive.py --headless-test 300 --no-viewer   # scripted smoke run, no window
python scripts/drive.py --track showcase --headless-test 300 --no-viewer  # scripted showcase circuit
python -m drivesim.baselines           # pure-pursuit baseline, 5 seeds x 3 difficulties x {loop, A->B}
python -m drivesim.baselines --quick --mode discrete      # faster, 2 seeds, other action mode
python scripts/vehicle_demo.py --out <png>             # vehicle scenarios to a PNG
python scripts/preview_world.py <out.png> --seed 3     # world preview PNG
```

Train/eval: `python scripts/train.py --help` (PPO; examples in its docstring), `python scripts/bc_pretrain.py` (DAgger warm start, writes a `train.py`-compatible checkpoint), `python scripts/plot_showcase_track.py` (layout PNG). Details in `drivesim/rl/README_rl.md`.

## Key conventions

- **World frame:** x east, y north, z up, all in metres.
- **Yaw:** 0 is +x, counter-clockwise positive (rotation about +z).
- **Lateral offsets:** positive is left of the track direction.
- **Heading error:** track heading minus car yaw, wrapped to (-pi, pi]. Positive means the track points left.
- **Lookahead:** `track.lookahead(pos, yaw, distances)` returns `(K, 2)` columns `(forward, left)` in the car frame.
- **Controls** (`vehicle.Controls`): `steer` [-1, 1] (+ left), `throttle` and `brake` [0, 1], `handbrake` [0, 1].
  `clutch`, `shift_up` and `shift_down` apply only in manual mode and are ignored in auto.
- **Action modes** (`DriveEnv(control_mode=...)`):
  - `discrete`: `Discrete(12)`, index `steer_idx*4 + gas*2 + brake`, auto transmission.
  - `continuous`: `Box([-1,0,0],[1,1,1])` = `[steer, throttle, brake]`, auto transmission.
  - `full`: 7 values `[steer, throttle, brake, clutch, handbrake, shift_up, shift_down]`, manual unless `transmission="auto"`.
- **Observation:** 56 float32 values, clipped to [-10, 10]. Names are in `drivesim.env.OBS_NAMES`.
- **Auto-transmission reverse:** the env clears the brake when |v| < 0.5 m/s, so a policy cannot enter reverse by holding brake. `info["brake_cleared"]` counts these events. The vehicle itself toggles reverse after holding brake at standstill for 1 s.
- **Seeds:** the same seed and arguments give bit-identical worlds and vehicle states. The vehicle uses no randomness. `DriveEnv(seed=int)` fixes the track; `seed=None` draws a new track on each reset from the env RNG (set via `reset(seed=...)`).
- **Caches:** `env.build_world` caches fixed-seed worlds (8 entries). Generating a new world takes about 0.6-1 s.

## Performance notes

- Vehicle model alone: about 3400 steps/s (`dt = 1/30`, 120 Hz substeps).
- `DriveEnv`: about 450-500 steps/s, single core. Most of the time goes to observation and track projection, not physics.
- Current figures are in `README.md` (Known limitations) and `drivesim/README_vehicle.md`. Check `git log` for speed-up work.

## Testing rules

- Run `python -m pytest` before committing. Tests must pass.
- Never skip, xfail, or weaken a test to get green. Fix the code, or explain the failure to the user.
- Headless: set `SDL_VIDEODRIVER=dummy`. The viewer and `drive.py` set it themselves when unset, but tests and scripts should not need a display.

## Git workflow

- Develop on `claude/confident-einstein-xqdtxy`. Do not work on the default branch.
- Stage with explicit paths: `git add <paths>`. Never `git add -A` or `git add .`.
- Never commit `.claude/worktrees/` (ignored by `.gitignore`; keep it out of `git add` anyway).
- Ignored by `.gitignore`: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.png`, `.venv/`, `.claude/worktrees/`. Generated PNG and output files stay uncommitted, except `docs/showcase_track.png`, which is tracked (force-added).
- Do not open pull requests unless the user asks.
- Pull and push: `git pull --rebase --autostash origin <branch>`, then `git push -u origin <branch>`.

## macOS notes

- Pure Python, numpy, pygame. Runs on Apple Silicon and Intel.
- MLX is in `requirements-train.txt` only for `darwin`/`arm64`. Torch is optional. On Linux, a CPU-only torch wheel is enough.
- `drive.py` needs a real Terminal or iTerm window to get keyboard focus. Do not run it from a headless shell on macOS.
- If you add `multiprocessing` for training or rollouts, use the `spawn` start method. macOS defaults to it, and `fork` is risky once pygame or torch/MLX are loaded.
- Gamepad button and axis numbers differ between controllers. See the macOS section of `drivesim/README_driving.md`.

## Known limitations

Read the limitation sections before changing a module. Do not repeat them here.

- `README.md`: "Known limitations" (throughput, world generation time, discrete baseline, no obstacles, no reverse action).
- `drivesim/README_vehicle.md`: "Known limitations" (no unsprung mass, ABS experimental, auto reverse is a brake-hold toggle).
- `drivesim/README_driving.md`: "Limitations" (stability assist, reset to last checkpoint, manual mode stalls, no force feedback).
- `drivesim/README_viewer.md`: "Limitations" (no depth buffer, painter's sorting, near-plane dropping, car is a box).
- `drivesim/README_world.md`: "Limitations".

Note: the "Driving yourself" and "Viewer" sections in `README.md` are placeholders. The real docs are `README_driving.md` and `README_viewer.md`.

## Working style

- Visual style is not a priority. Bare-bones rendering is fine, as long as it is readable.
- The project must run easily on macOS. Prefer dependencies that install with pip, and avoid Linux-only tooling.
- The owner prefers delegating work to small, focused Haiku subagents, one concern per agent: world generation, vehicle, env/reward, viewer, human controls, training.
- The coordinating agent does not write code itself. It splits the task, hands it out, and reviews the results.
