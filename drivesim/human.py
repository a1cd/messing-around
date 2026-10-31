"""Human driver input layer: raw keyboard / gamepad state -> smooth ``vehicle.Controls``.

The pipeline is

    device (pygame keys, joystick, or synthetic RawInput)
        -> RawInput  (held names, newly-pressed names, analog axes)
        -> HumanInput.update(raw, dt, vehicle)  (ramps, curves, assists)
        -> Controls  (for Vehicle.step)            or  to_env_action(mode)

``HumanInput`` never touches pygame, so it can be driven headlessly with
synthetic ``RawInput`` objects. ``DeviceReader`` is the only pygame-facing part
and imports pygame lazily.

Default keys (see ``DEFAULT_BINDINGS``):
    A/Left, D/Right  steer          W/Up   gas      S/Down  brake / reverse
    Space            handbrake      LShift clutch (manual)
    E / Q            shift up/down  (manual)
    C camera, R reset, T assist toggle, P pause, H help, Esc quit

Gamepad (pygame button indices, Xbox layout on SDL): left stick X steers,
RT gas, LT brake, A handbrake, B clutch (manual), LB/RB shift down/up,
Back camera, Start pause, X assist, Y reset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import numpy as np

from .vehicle import Controls

# ---------------------------------------------------------------------------
# bindings
# ---------------------------------------------------------------------------
ROLES = (
    "steer_left", "steer_right", "gas", "brake", "handbrake", "clutch",
    "shift_up", "shift_down", "camera", "reset", "assist", "pause", "help", "quit",
)
# Actions that are reported once per press in HumanInput.actions (shifts are reported through Controls).
ACTION_ROLES = ("camera", "reset", "assist", "pause", "help", "quit")

DEFAULT_BINDINGS: Dict[str, str] = {
    "a": "steer_left", "left": "steer_left",
    "d": "steer_right", "right": "steer_right",
    "w": "gas", "up": "gas",
    "s": "brake", "down": "brake",
    "space": "handbrake",
    "lshift": "clutch",
    "e": "shift_up", "q": "shift_down",
    "c": "camera", "r": "reset", "t": "assist", "p": "pause", "h": "help", "escape": "quit",
    # gamepad buttons (pygame joystick indices)
    "pad_0": "handbrake", "pad_1": "clutch", "pad_2": "assist", "pad_3": "reset",
    "pad_4": "shift_down", "pad_5": "shift_up", "pad_6": "camera", "pad_7": "pause",
}

# pygame.key.key_code() names for the binding names that differ from pygame's own spelling
_PYGAME_KEY_NAMES = {"lshift": "left shift", "escape": "escape", "space": "space",
                     "up": "up", "down": "down", "left": "left", "right": "right"}

# gamepad analog axes (pygame joystick indices, Xbox layout on SDL)
DEFAULT_AXES = {"steer": 0, "gas": 5, "brake": 4}


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
@dataclass
class RawInput:
    """One frame of device state.

    held     names of keys / pad buttons currently down (e.g. "w", "left", "pad_0")
    pressed  names that went down since the previous frame (for taps shorter than a frame)
    axes     analog values: "steer" in [-1, 1] (left +), "gas" and "brake" in [0, 1]
    """

    held: Set[str] = field(default_factory=set)
    pressed: Set[str] = field(default_factory=set)
    axes: Dict[str, float] = field(default_factory=dict)


@dataclass
class HumanTuning:
    """Feel parameters. All rates are per second of simulated time."""

    # steering: the keyboard produces a linear signal s in [-1, 1] that eases toward the
    # held direction, and the output is a nonlinear curve of s, limited by speed.
    steer_press_rate: float = 2.4      # build toward lock while a key is held (full lock in ~0.4 s)
    steer_release_rate: float = 4.5    # return to center when released (fast self-centering)
    steer_flip_rate: float = 6.0       # when switching direction (quick counter-flick)
    steer_curve: float = 1.5           # output = sign(s) |s|^curve: fine control near center
    steer_limit_min: float = 0.6       # normalized lock available at steer_limit_speed and above
    steer_limit_speed: float = 30.0    # m/s (108 km/h) where the lock limit is reached
    pad_deadzone: float = 0.12         # stick deadzone
    pad_curve: float = 1.8             # stick response curve

    # pedals
    throttle_rise: float = 2.2         # gas pedal press (about 0.45 s to full)
    throttle_fall: float = 5.0         # gas released
    brake_rise: float = 4.0            # brake press
    brake_fall: float = 6.0            # brake released
    handbrake_rate: float = 6.0        # handbrake lever applied / released
    clutch_rise: float = 6.0           # clutch pedal pressed (manual)
    clutch_fall: float = 3.0           # clutch released slowly (manual, so it can be bitten)

    # stability assist (off by default, toggled with the assist key)
    counter_steer_gain: float = 2.5    # normalized steer per rad of body slip beyond the deadzone
    counter_steer_max: float = 0.6     # cap on the assist steer contribution
    slip_deadzone: float = 0.05        # rad (about 3 deg) of body slip ignored by the assist
    throttle_cut_slip: float = 0.30    # rad of body slip above which gas is trimmed
    throttle_cut_gain: float = 2.0     # gas multiplier falls by this per rad above throttle_cut_slip
    assist_min_speed: float = 4.0      # m/s; the assist is inactive below this speed

    # reverse
    reverse_speed_eps: float = 0.5     # m/s; "standstill" for the reverse hint


def _approach(x: float, target: float, rate_up: float, rate_down: float, dt: float) -> float:
    if target > x:
        return min(target, x + rate_up * dt)
    return max(target, x - rate_down * dt)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _curve(x: float, exponent: float) -> float:
    return math.copysign(abs(x) ** exponent, x) if x != 0.0 else 0.0


# ---------------------------------------------------------------------------
# the input model
# ---------------------------------------------------------------------------
class HumanInput:
    """Turns RawInput frames into smooth Controls.

    Call ``update(raw, dt, vehicle)`` once per simulation step. ``vehicle`` is optional and only
    read through ``speed``, ``beta`` (body slip) and ``gear``; any object with those attributes works.

    After each update:
      ``controls``  the Controls returned
      ``actions``   set of one-shot roles pressed this step (camera, reset, assist, pause, help, quit)
      ``buttons``   thresholded raw buttons (used by ``to_env_action('discrete')``)
      ``brake_hold``  seconds the brake has been held at standstill (for a reverse hint)
    """

    def __init__(self, tuning: Optional[HumanTuning] = None, *, manual: bool = False,
                 assists: bool = False, bindings: Optional[Dict[str, str]] = None):
        self.tuning = tuning or HumanTuning()
        self.manual = bool(manual)
        self.assists = bool(assists)
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)
        self._role_of = self.bindings
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self._steer = 0.0       # linear steer signal s in [-1, 1]
        self._throttle = 0.0
        self._brake = 0.0
        self._handbrake = 0.0
        self._clutch = 0.0
        self._prev_held: Set[str] = set()
        self.brake_hold = 0.0
        self.controls = Controls()
        self.actions: Set[str] = set()
        self.buttons = {"gas": False, "brake": False, "steer": 0, "handbrake": False, "clutch": False}

    def toggle_assists(self) -> bool:
        self.assists = not self.assists
        return self.assists

    def steer_limit(self, speed: float) -> float:
        """Normalized steering lock available at ``speed`` [m/s]. 1.0 at standstill, shrinking to steer_limit_min."""
        t = self.tuning
        f = _clamp(abs(speed) / max(t.steer_limit_speed, 1e-6), 0.0, 1.0)
        return 1.0 + (t.steer_limit_min - 1.0) * f

    # -- main --------------------------------------------------------------
    def update(self, raw: RawInput, dt: float, vehicle=None) -> Controls:
        t = self.tuning
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
        beta = float(getattr(vehicle, "beta", 0.0) or 0.0)
        gear = int(getattr(vehicle, "gear", 0) or 0)
        dt = max(float(dt), 0.0)

        # --- classify the held and pressed names into roles
        roles_held: Set[str] = set()
        roles_pressed: Set[str] = set()
        for name in raw.held:
            r = self._role_of.get(name)
            if r is not None:
                roles_held.add(r)
        edges = set(raw.pressed) | (set(raw.held) - self._prev_held)
        self._prev_held = set(raw.held) | set(raw.pressed)
        for name in edges:
            r = self._role_of.get(name)
            if r is not None:
                roles_pressed.add(r)
        self.actions = {r for r in roles_pressed if r in ACTION_ROLES}
        if "assist" in self.actions:
            self.toggle_assists()

        # --- keyboard intents
        key_steer = (1 if "steer_left" in roles_held else 0) - (1 if "steer_right" in roles_held else 0)
        key_gas = "gas" in roles_held
        key_brake = "brake" in roles_held
        key_hb = "handbrake" in roles_held
        key_clutch = "clutch" in roles_held

        # --- gamepad analog
        axes = raw.axes
        pad_steer = _clamp(float(axes.get("steer", 0.0)), -1.0, 1.0)
        pad_gas = _clamp(float(axes.get("gas", 0.0)), 0.0, 1.0)
        pad_brake = _clamp(float(axes.get("brake", 0.0)), 0.0, 1.0)

        # --- reverse: once the car is in reverse gear the pedals swap, S is gas and W is brake
        in_reverse = gear == -1
        if in_reverse:
            key_gas, key_brake = key_brake, key_gas
            gas_btn = "brake" in roles_held
            brake_btn = "gas" in roles_held
        else:
            gas_btn, brake_btn = key_gas, key_brake

        gas_target = max(1.0 if key_gas else 0.0, pad_gas)
        brake_target = max(1.0 if key_brake else 0.0, pad_brake)
        if brake_target > 0.0:
            gas_target = 0.0       # brake overrides gas, as on a real pedal box
        self._throttle = _approach(self._throttle, gas_target, t.throttle_rise, t.throttle_fall, dt)
        self._brake = _approach(self._brake, brake_target, t.brake_rise, t.brake_fall, dt)
        # hold-to-reverse hint: brake held at standstill
        if self._brake > 0.5 and abs(speed) < t.reverse_speed_eps:
            self.brake_hold += dt
        else:
            self.brake_hold = 0.0

        hb_target = 1.0 if key_hb else 0.0
        self._handbrake = _approach(self._handbrake, hb_target, t.handbrake_rate, t.handbrake_rate, dt)

        # --- steering
        limit = self.steer_limit(speed)
        if abs(pad_steer) > t.pad_deadzone:
            mag = (abs(pad_steer) - t.pad_deadzone) / (1.0 - t.pad_deadzone)
            self._steer = math.copysign(min(mag, 1.0), pad_steer)
            steer_out = _curve(self._steer, t.pad_curve) * limit
        else:
            if key_steer == 0:
                self._steer = _approach(self._steer, 0.0, t.steer_release_rate, t.steer_release_rate, dt)
            else:
                flip = self._steer * key_steer < 0.0
                rate = t.steer_flip_rate if flip else t.steer_press_rate
                self._steer = _approach(self._steer, float(key_steer), rate, rate, dt)
            steer_out = _curve(self._steer, t.steer_curve) * limit

        # --- stability assist (optional)
        if self.assists and speed > t.assist_min_speed:
            excess = abs(beta) - t.slip_deadzone
            if excess > 0.0:
                assist = math.copysign(min(t.counter_steer_gain * excess, t.counter_steer_max), beta)
                steer_out += assist
            over = abs(beta) - t.throttle_cut_slip
            if over > 0.0:
                self._throttle *= _clamp(1.0 - t.throttle_cut_gain * over, 0.0, 1.0)

        # --- manual transmission
        shift_up = shift_down = False
        clutch_out = 0.0
        if self.manual:
            clutch_target = 1.0 if key_clutch else 0.0
            self._clutch = _approach(self._clutch, clutch_target, t.clutch_rise, t.clutch_fall, dt)
            clutch_out = self._clutch
            shift_up = "shift_up" in roles_pressed
            shift_down = "shift_down" in roles_pressed

        self.controls = Controls(
            steer=_clamp(steer_out, -1.0, 1.0),
            throttle=_clamp(self._throttle, 0.0, 1.0),
            brake=_clamp(self._brake, 0.0, 1.0),
            handbrake=_clamp(self._handbrake, 0.0, 1.0),
            clutch=_clamp(clutch_out, 0.0, 1.0),
            shift_up=shift_up,
            shift_down=shift_down,
        )

        # thresholded raw buttons for the discrete env mode
        steer_btn = key_steer
        if abs(pad_steer) > t.pad_deadzone:
            steer_btn = 1 if pad_steer > 0 else -1
        self.buttons = {
            "gas": bool(gas_btn or pad_gas > 0.5),
            "brake": bool(brake_btn or pad_brake > 0.5),
            "steer": int(steer_btn),
            "handbrake": bool(key_hb),
            "clutch": bool(key_clutch),
        }
        return self.controls

    # -- env actions --------------------------------------------------------
    def to_env_action(self, mode: str = "continuous"):
        """Action for ``drivesim.env.DriveEnv`` in the given control mode, from the most recent update.

        discrete    int index into ``env.DISCRETE_ACTIONS``: ``steer_idx * 4 + gas * 2 + brake``
                    with steer_idx 0 = left, 1 = none, 2 = right. Built from thresholded raw
                    buttons, so gas/brake are fully on or off.
        continuous  float32 [steer, throttle, brake]
        full        float32 [steer, throttle, brake, clutch, handbrake, shift_up, shift_down]
                    (the order used by DriveEnv._decode).
        """
        c = self.controls
        if mode == "discrete":
            b = self.buttons
            steer_idx = {1: 0, 0: 1, -1: 2}[int(b["steer"])]
            return int(steer_idx * 4 + int(b["gas"]) * 2 + int(b["brake"]))
        if mode == "continuous":
            return np.array([c.steer, c.throttle, c.brake], dtype=np.float32)
        if mode == "full":
            return np.array([c.steer, c.throttle, c.brake, c.clutch, c.handbrake,
                             float(c.shift_up), float(c.shift_down)], dtype=np.float32)
        raise ValueError(f"unknown env mode {mode!r}; expected 'discrete', 'continuous' or 'full'")


# ---------------------------------------------------------------------------
# pygame-facing device reader (imported lazily)
# ---------------------------------------------------------------------------
class DeviceReader:
    """Reads the keyboard and the first gamepad through pygame.

    ``read(events)`` takes the events already fetched by the caller (``pygame.event.get()``),
    so the same loop can also handle QUIT and window events. Call ``open()`` after
    ``pygame.init()``. Gamepad support is auto-detected: if there is no joystick, the reader
    just reports the keyboard.
    """

    def __init__(self, bindings: Optional[Dict[str, str]] = None, axes: Optional[Dict[str, int]] = None,
                 trigger_rest: float = -1.0):
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)
        self.axes = dict(DEFAULT_AXES if axes is None else axes)
        self.trigger_rest = float(trigger_rest)
        self._pg = None
        self._joy = None
        self._key_codes: Dict[str, int] = {}

    def open(self) -> bool:
        import pygame
        self._pg = pygame
        pygame.joystick.init()          # the keyboard needs no init; avoid starting the audio device
        for name in self.bindings:
            if name.startswith("pad_"):
                continue
            pg_name = _PYGAME_KEY_NAMES.get(name, name)
            try:
                self._key_codes[name] = pygame.key.key_code(pg_name)
            except (ValueError, KeyError):
                pass
        self._open_joystick()
        return self._joy is not None

    @property
    def gamepad_name(self) -> Optional[str]:
        return self._joy.get_name() if self._joy is not None else None

    def _open_joystick(self) -> None:
        pg = self._pg
        if self._joy is None and pg.joystick.get_count() > 0:
            try:
                self._joy = pg.joystick.Joystick(0)
            except pg.error:
                self._joy = None

    def read(self, events=()) -> RawInput:
        pg = self._pg
        if pg is None:
            raise RuntimeError("DeviceReader.open() must be called first")
        raw = RawInput()
        for ev in events:
            if ev.type == pg.KEYDOWN:
                for name, code in self._key_codes.items():
                    if ev.key == code:
                        raw.pressed.add(name)
            elif ev.type == pg.JOYBUTTONDOWN:
                raw.pressed.add(f"pad_{ev.button}")
            elif ev.type == pg.JOYDEVICEADDED:
                self._open_joystick()
            elif ev.type == pg.JOYDEVICEREMOVED:
                self._joy = None
        keys = pg.key.get_pressed()
        for name, code in self._key_codes.items():
            try:
                if keys[code]:
                    raw.held.add(name)
            except IndexError:
                pass
        if self._joy is not None:
            joy = self._joy
            try:
                for b in range(joy.get_numbuttons()):
                    if joy.get_button(b):
                        raw.held.add(f"pad_{b}")
                if "steer" in self.axes:
                    raw.axes["steer"] = float(joy.get_axis(self.axes["steer"]))
                for role in ("gas", "brake"):
                    if role in self.axes:
                        v = float(joy.get_axis(self.axes[role]))
                        if self.trigger_rest < 0.0:
                            v = (v + 1.0) * 0.5
                        raw.axes[role] = _clamp(v, 0.0, 1.0)
            except pg.error:
                self._joy = None
        return raw
