# Driving the simulator yourself

`scripts/drive.py` runs a generated world with the vehicle model and lets you drive it from the
keyboard or a gamepad. `drivesim/human.py` turns raw device state into smooth controls; the same
code produces actions for `DriveEnv` in all three control modes.

```
python scripts/drive.py                       # sports car, closed circuit, seed 0
python scripts/drive.py --seed 7 --difficulty 0.8 --open --car hatchback
python scripts/drive.py --manual --assists    # manual gearbox, stability assist on
python scripts/drive.py --headless-test 300 --no-viewer   # scripted smoke run, no window
```

Options: `--seed`, `--difficulty` (0..1), `--open` / `--closed`, `--manual`,
`--car sports_car|hatchback`, `--assists`, `--fps` (display cap, default 30), `--no-viewer`,
`--headless-test N`, `--width`, `--height`.

The simulation advances a fixed `dt = 1/30 s` per frame. `Vehicle` substeps internally (120 Hz).

## Controls

| Action | Keyboard | Gamepad (pygame index, Xbox layout) |
|---|---|---|
| Steer left / right | `A` / `D` or Left / Right | left stick X (deadzone 0.12) |
| Gas | `W` or Up | RT (axis 5) |
| Brake; then reverse | `S` or Down | LT (axis 4) |
| Handbrake | `Space` | A (button 0) |
| Clutch (manual only) | `Left Shift` (held) | B (button 1) |
| Shift up / down (manual only) | `E` / `Q` | RB (5) / LB (4) |
| Camera (chase, hood, top) | `C` | Back (6) |
| Reset to last checkpoint | `R` | Y (3) |
| Stability assist on/off | `T` | X (2) |
| Pause | `P` | Start (7) |
| Help overlay | `H` | |
| Quit | `Esc` | window close |

**Reverse.** Hold `S` (brake) at a standstill for about 1 s and the automatic gearbox engages reverse.
Once in reverse, the pedals swap: `S` is gas (backwards) and `W` is brake, so holding `W` to stop and
then again for about 1 s returns to forward. In manual mode, shift down from neutral to reverse.

**Manual mode** starts in neutral. Hold `Left Shift`, press `E` to go into first, release the clutch
(the release is slow, so it can be bitten). The shift only takes when the clutch is at least half
pressed, as in the vehicle model.

**Checkpoints.** The last safe point is recorded every 20 m while all four wheels are on the road and
the car is upright. `R` puts the car there, facing down the track, at rest.

## Feel

`HumanInput` keeps a linear steering signal `s` in [-1, 1] that moves toward the held direction
and returns toward zero when released. The output is `sign(s) * |s|^1.5`, so small inputs give fine
control and a full press reaches full lock. The lock is then limited by speed (`steer_limit`), from
1.0 at standstill to `steer_limit_min` at `steer_limit_speed`. The vehicle model already shrinks its
own lock with speed, so the two effects stack.

| Input | Behaviour |
|---|---|
| Steer | builds at 2.4 units/s (full lock in about 0.4 s); returns at 4.5/s; reverses at 6/s |
| Gas | rises at 2.2/s (about 0.45 s to full), falls at 5/s |
| Brake | rises at 4/s, falls at 6/s; overrides gas |
| Handbrake | rises and falls at 6/s |
| Clutch (manual) | rises at 6/s, falls at 3/s |

All of these are fields of `HumanTuning`. Pass a customised instance to `HumanInput(tuning=...)`:

```python
from drivesim.human import HumanInput, HumanTuning
tuning = HumanTuning(steer_curve=2.0, steer_limit_min=0.4, steer_press_rate=1.8)
human = HumanInput(tuning=tuning, manual=False, assists=False)
```

Useful knobs:

* `steer_press_rate` / `steer_release_rate`: lower the press rate for finer steering, raise the
  release rate for quicker self-centering.
* `steer_curve`: 1.0 is linear; higher values soften the centre.
* `steer_limit_min` and `steer_limit_speed`: how much lock is removed at speed.
* `pad_deadzone`, `pad_curve`: gamepad stick feel.
* `throttle_rise`, `brake_rise`: how quickly the pedals reach full travel.

## Stability assist

Off by default. `T` toggles it, or start with `--assists`. When active (above 4 m/s) it does two
things, both scaled by body slip angle beyond a 3 degree deadzone:

1. **Counter-steer.** It adds steering toward the slide, up to 0.6 of full lock
   (`counter_steer_gain` 2.5 per rad).
2. **Gas trim.** Above 0.3 rad of body slip, gas is reduced by `throttle_cut_gain` per rad.

It is not a full ESC. It catches throttle-induced oversteer (a steady left turn at 20 m/s with
full throttle has its peak slip cut by more than half in the tests). A handbrake spin already past
about 1 rad is not recovered.

## Using it from code

```python
from types import SimpleNamespace
from drivesim.human import DeviceReader, HumanInput

reader = DeviceReader()        # keyboard + first gamepad, auto-detected
reader.open()                  # call after pygame.joystick is available
human = HumanInput(manual=False)

raw = reader.read(pygame.event.get())            # or a synthetic RawInput(held={"w"})
controls = human.update(raw, dt=1/30, vehicle=car)   # car: anything with speed, beta, gear
car.step(controls, 1/30, terrain)

human.actions          # one-shot roles pressed this frame: camera, reset, pause, help, quit, assist
human.to_env_action("discrete")   # int, index into env.DISCRETE_ACTIONS
human.to_env_action("continuous") # float32 [steer, throttle, brake]
human.to_env_action("full")       # float32 [steer, throttle, brake, clutch, handbrake, shift_up, shift_down]
```

`to_env_action("discrete")` uses thresholded raw buttons (gas on/off, brake on/off, steer -1/0/+1
from the keys or the stick past 0.5), not the ramped values. The other two modes use the ramped
controls, as a human on a pedal box would.

`HumanInput` never imports pygame. Tests and scripted drivers can pass `RawInput(held=..., pressed=...,
axes=...)` directly. `pressed` holds names that went down since the last frame, so taps shorter than a
frame are not lost.

## macOS notes

* `pygame` 2.x works on macOS. Run `drive.py` from Terminal or iTerm, not a headless shell. The
  window needs focus to receive keys.
* Gamepads: connect over Bluetooth or USB before starting. Button and axis numbers come from SDL and
  differ between pads. The defaults assume an Xbox-style layout. For other pads, override
  `DeviceReader(axes=...)` or `bindings`, and check the numbers with a short pygame script.
* Trigger axes are read as -1 at rest by default (`DeviceReader(trigger_rest=-1.0)`). If a pad's
  triggers rest at 0, pass `trigger_rest=0.0`.
* Only the left Shift key is bound to the clutch.
* Headless smoke tests: `SDL_VIDEODRIVER=dummy python scripts/drive.py --headless-test 300 --no-viewer`.
  The same is true on Linux.

## Limitations

* The stability assist is a simple slip-angle controller, not an ESC. It does nothing for a handbrake
  spin already past about 1 rad of slip.
* Reset puts the car on the most recent checkpoint, not on the closest point to where it crashed.
  On a long off-track excursion it may drop the car well back.
* `P` pauses by freezing the simulation and the ramps. The viewer keeps rendering and the HUD
  message reads PAUSED.
* The help overlay replaces the HUD message while it is open.
* Manual mode has no auto-clutch or rev-matching: the car stalls if you release the clutch too
  quickly at low rpm, as in the vehicle model.
* No force feedback. Racing wheels work only if pygame exposes them as joysticks with the same axis
  layout as the defaults.
* Gamepad steering is applied directly with a curve, so the stick does not use the keyboard ramp.
  If both are used at once, the stick wins while it is outside its deadzone.
