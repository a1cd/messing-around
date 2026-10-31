"""Engine, friction clutch, gearbox and shift logic.

Conventions
-----------
* Engine speed is kept internally in rad/s; ``RPM_PER_RADS`` converts for display.
* ``gear``: -1 = reverse, 0 = neutral, 1..6 = forward gears.
* ``engage`` in [0, 1] is the clutch engagement (1 = fully engaged).
* Manual mode: ``engage = 1 - clutch_pedal`` (pedal 1 = pressed = disengaged).
* Automatic mode: the clutch engagement is an internal state that is ramped
  up/down by a launch rule, and shifts are chosen from engine speed.

The clutch coupling itself (locking vs. slipping) is solved in Vehicle._substep
because it depends on the wheel dynamics; this module provides the engine
torque/friction, the gear ratio and the shift state machine.
"""

from __future__ import annotations

import bisect
import math
from typing import Sequence, Tuple

RPM_PER_RADS = 60.0 / (2.0 * math.pi)


def rpm_to_rads(rpm: float) -> float:
    return rpm / RPM_PER_RADS


class TorqueCurve:
    """Piecewise-linear engine torque curve, evaluated in rad/s.

    ``points`` is a sequence of (rpm, torque[N m]) pairs.  Outside the table the
    end values are held.
    """

    def __init__(self, points: Sequence[Tuple[float, float]]):
        pts = sorted((float(r), float(t)) for r, t in points)
        if len(pts) < 2:
            raise ValueError("torque curve needs at least two points")
        self._w = [rpm_to_rads(r) for r, _ in pts]
        self._t = [t for _, t in pts]
        self._n = len(pts)

    def __call__(self, w: float) -> float:
        ws, ts = self._w, self._t
        if w <= ws[0]:
            return ts[0]
        if w >= ws[-1]:
            return ts[-1]
        i = bisect.bisect_right(ws, w)
        w0, w1 = ws[i - 1], ws[i]
        f = (w - w0) / (w1 - w0)
        return ts[i - 1] + f * (ts[i] - ts[i - 1])


class Powertrain:
    """Engine state, clutch engagement, gear selection and shifting.

    Created and owned by :class:`drivesim.vehicle.Vehicle`.
    """

    # auto-shift thresholds (rpm)
    AUTO_UPSHIFT_BASE = 3000.0      # upshift at this rpm with zero throttle ...
    AUTO_UPSHIFT_SPAN = 3500.0      # ... rising to base+span at full throttle
    AUTO_DOWNSHIFT_RPM = 2100.0
    AUTO_LAUNCH_RPM = 1800.0        # wheel-side engine-equivalent speed above which clutch is fully engaged
    AUTO_ENGAGE_UP_RATE = 3.0       # 1/s
    AUTO_ENGAGE_DOWN_RATE = 10.0    # 1/s
    AUTO_REVERSE_HOLD_TIME = 1.0    # s of brake at standstill before reverse is selected
    STANDSTILL_SPEED = 0.5          # m/s

    def __init__(self, p, transmission: str):
        if transmission not in ("auto", "manual"):
            raise ValueError("transmission must be 'auto' or 'manual'")
        self.p = p
        self.mode = transmission
        self.curve = TorqueCurve(p.torque_curve)
        self.idle_w = rpm_to_rads(p.idle_rpm)
        self.redline_w = rpm_to_rads(p.redline_rpm)
        self.stall_w = rpm_to_rads(p.stall_rpm)
        self.limiter_w = rpm_to_rads(p.redline_rpm * 1.05)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.engine_on = True
        self.engine_w = self.idle_w
        self.gear = 1 if self.mode == "auto" else 0
        self.pending_gear = self.gear
        self.shift_timer = 0.0
        self.engage = 0.0
        self.hold_timer = 0.0
        self.brake_latched = False

    # ----------------------------------------------------------- gear ratios
    def ratio(self) -> float:
        """Total ratio from engine to the propeller side of the differential (0 in neutral)."""
        g = self.gear
        if g == 0:
            return 0.0
        if g < 0:
            return -self.p.reverse_ratio * self.p.final_drive
        return self.p.gear_ratios[g - 1] * self.p.final_drive

    # ------------------------------------------------------------ engine
    def engine_torque(self, throttle: float):
        """Return (engine torque [N m], engine friction torque [N m]) at the current speed."""
        p = self.p
        w = self.engine_w
        friction = p.friction_c0 + p.friction_c1 * w
        if not self.engine_on:
            return 0.0, friction
        if w > self.limiter_w:                # hard fuel cut
            return 0.0, friction
        torque = throttle * self.curve(w) if w < self.redline_w else 0.0
        if w < self.idle_w:                   # idle speed controller
            idle_t = (self.idle_w - w) * p.idle_gain
            if idle_t > p.idle_torque_max:
                idle_t = p.idle_torque_max
            torque += idle_t
        return torque, friction

    def starter_torque(self) -> float:
        return self.p.starter_torque

    # ------------------------------------------------------------ clutch
    def engage_eff(self) -> float:
        """Effective clutch engagement used this substep (0 while shifting or stalled)."""
        if self.shift_timer > 0.0 or not self.engine_on:
            return 0.0
        return self.engage

    # --------------------------------------------------------- manual requests
    def request_shift(self, up: bool, down: bool, clutch_pedal: float) -> None:
        """Handle manual shift-up / shift-down edges (call once per env step).

        A shift is only accepted when the clutch pedal is at least half pressed,
        and while no other shift is in progress.  H-pattern order:
        R <-> N <-> 1 <-> 2 <-> ... <-> 6.  Shifts take ``shift_time`` seconds,
        during which the clutch is open.
        """
        if self.mode != "manual" or not (up or down):
            return
        if self.shift_timer > 0.0 or clutch_pedal < 0.5:
            return
        g = self.gear
        if up and not down:
            target = {-1: 0, 0: 1}.get(g, min(6, g + 1))
        elif down and not up:
            if g == -1:
                return
            target = {0: -1, 1: 0}.get(g, g - 1)
        else:  # both pressed: ignore
            return
        if target != g:
            self.pending_gear = target
            self.shift_timer = self.p.shift_time

    # ------------------------------------------------------- per-substep update
    def update(self, h: float, throttle: float, brake: float, clutch_pedal: float,
               v_long: float, in_gear_engine_w: float) -> None:
        """Advance gear/clutch/engine-start state by one substep of length ``h``.

        ``in_gear_engine_w`` is |total ratio| * |driven wheel speed| (the engine
        speed the wheels would impose if the clutch were fully engaged).
        """
        p = self.p
        # --- engine stall / starter
        if self.engine_on and self.engine_w < self.stall_w:
            self.engine_on = False
        if not self.engine_on:
            can_start = throttle > 0.05 and (self.mode == "auto" or self.gear == 0 or clutch_pedal >= 0.5)
            if can_start:
                self.engine_w += h * (self.starter_torque() / p.engine_inertia)
                if self.engine_w >= self.idle_w:
                    self.engine_w = self.idle_w
                    self.engine_on = True

        # --- shift timer
        if self.shift_timer > 0.0:
            self.shift_timer -= h
            if self.shift_timer <= 0.0:
                self.shift_timer = 0.0
                self.gear = self.pending_gear

        if self.mode == "manual":
            self.engage = 1.0 - clutch_pedal
            return

        # ================= automatic transmission =================
        moving_slow = abs(v_long) < self.STANDSTILL_SPEED
        # --- direction toggle: hold the brake at standstill for AUTO_REVERSE_HOLD_TIME to flip D <-> R.
        # The brake must be released before the next toggle (a latch prevents repeated flipping).
        if brake < 0.05:
            self.hold_timer = 0.0
            self.brake_latched = False
        elif (moving_slow and self.shift_timer <= 0.0 and throttle < 0.05
              and not self.brake_latched):
            self.hold_timer += h
            if self.hold_timer >= self.AUTO_REVERSE_HOLD_TIME:
                self._instant_gear(-1 if self.gear != -1 else 1)
                self.hold_timer = 0.0
                self.brake_latched = True
                return
        else:
            self.hold_timer = 0.0

        # --- gear selection (forward gears only)
        # only shift while the clutch is engaged, otherwise the engine has not followed the gears
        if self.shift_timer <= 0.0 and self.engine_on and self.gear >= 1 and self.engage > 0.95:
            rpm = self.engine_w * RPM_PER_RADS
            if self.gear < 6 and rpm > self.AUTO_UPSHIFT_BASE + self.AUTO_UPSHIFT_SPAN * throttle:
                self._start_shift(self.gear + 1)
            elif self.gear > 1 and rpm < self.AUTO_DOWNSHIFT_RPM:
                g = self.gear
                ratio_new = p.gear_ratios[g - 2]
                ratio_cur = p.gear_ratios[g - 1]
                predicted = rpm * ratio_new / ratio_cur
                if predicted < 0.9 * p.redline_rpm:
                    self._start_shift(g - 1)

        # --- clutch engagement target
        if self.shift_timer > 0.0:
            self.engage = 0.0
            return
        target = 0.0
        if self.engine_on and self.ratio() != 0.0:
            if in_gear_engine_w > rpm_to_rads(self.AUTO_LAUNCH_RPM):
                target = 1.0
            elif self.engine_w > 1.1 * self.idle_w:
                target = min(1.0, 1.3 * throttle)
        if self.engine_w < 1.1 * self.idle_w:
            # engine about to stall (e.g. locked wheels dragging it down): open the clutch at once
            self.engage = 0.0
            return
        if target > self.engage:
            self.engage = min(target, self.engage + self.AUTO_ENGAGE_UP_RATE * h)
        else:
            self.engage = max(target, self.engage - self.AUTO_ENGAGE_DOWN_RATE * h)

    # ------------------------------------------------------------ helpers
    def _start_shift(self, target: int) -> None:
        self.pending_gear = target
        self.shift_timer = self.p.shift_time
        self.engage = 0.0

    def _instant_gear(self, target: int) -> None:
        self.gear = target
        self.pending_gear = target
        self.shift_timer = 0.0
        self.engage = 0.0
