"""Scripted (segment-based) track builder.

A circuit is written as an ordered list of distinctive segments, like a rally
pace-note sheet or a track designer's storyboard::

    S200 R4L L2 HAIRPIN-L ZIGZAG4 DROP-15/80 SWEEP-R90 CLOSE

or with the Python builder::

    TrackScript().straight(200).right(severity=3, length="long").hairpin("left") \\
                 .zigzag(n=4).drop(15, 80).sweep("right", 90).close()

The builder integrates the segments into a planar centreline (curvature
continuous: every corner is entered and left through clothoid-style linear
curvature ramps), attaches an elevation profile (explicit for drop / climb /
crest / dip / grade segments, terrain-following elsewhere) and returns an
ordinary :class:`drivesim.track.Track`.

Public API
----------
``TrackScript`` builder (chainable), ``TrackScript.from_dsl(text)``,
``script.to_dsl()``, ``script.build(terrain=None, **options) -> Track``,
``build_world_from_script(script, seed, ...) -> (Terrain, Track)``,
``parse_track_script(text)``, ``EXAMPLES`` / ``example_script(name)``.
See ``drivesim/README_track_script.md`` for the DSL grammar and conventions.
"""
from __future__ import annotations

import itertools
import math
import re
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .terrain import Terrain
from .track import EMBANKMENT_WIDTH, SHOULDER_WIDTH, Track, wrap_angle
from .world import carve_track

FINE_DS = 0.1                 # internal integration step (m)
MIN_RADIUS = 8.0              # tightest allowed corner radius (m)
DEFAULT_MAX_GRADE = 0.25      # |dz/ds| limit for scripted tracks
DEFAULT_WIDTH = 10.0
DEFAULT_SPACING = 1.5
HEADING_TOLERANCE = math.radians(1.0)
GRADE_SMOOTHING_M = 30.0      # terrain-following grade is taken from terrain smoothed over this scale

# Pace-note severity -> corner radius (m). 1 = hairpin-tight, 6 = flat-out kink.
RADIUS_BY_SEVERITY = {1: 9.0, 2: 16.0, 3: 28.0, 4: 45.0, 5: 75.0, 6: 120.0}
# Pace-note length -> corner angle (degrees).
ANGLE_BY_SPAN = {"short": 40.0, "normal": 75.0, "long": 120.0}
TIGHTEN_FACTOR = 0.6          # "tightens": exit radius = entry radius * 0.6
OPEN_FACTOR = 1.5             # "opens": exit radius = entry radius * 1.5
HAIRPIN_RADIUS = 9.0
SWITCHBACK_LEG = 30.0         # straight between the two hairpins of a switchback (m)
TURN_RADIUS = 25.0
SWEEP_RADIUS = 80.0
ZIGZAG_SEVERITY = 3
ESSES_SEVERITY = 5
RAMP_M = 15.0                 # length of the smooth start / end of an elevation change (m)

KINDS = ("straight", "corner", "hairpin", "switchback", "zigzag", "turn", "sweep", "elev", "steep")
ELEV_SHAPES = ("drop", "climb", "crest", "dip")

SegmentInfo = namedtuple(
    "SegmentInfo", "index kind text s0 s1 heading_change elevation_change caution"
)
SegmentInfo.__doc__ = """One scripted segment as it appears on the finished track.

index : position in the script (0-based). kind : segment kind.
text : canonical DSL token. s0, s1 : arc-length range (m).
heading_change : signed heading change over the segment (rad).
elevation_change : net height change over the segment (m).
caution : True if the segment carries the caution flag (advisory only)."""

_SPAN_FROM_LETTER = {"S": "short", "N": "normal", "L": "long"}
_LETTER_FROM_SPAN = {v: k for k, v in _SPAN_FROM_LETTER.items()}


# ---------------------------------------------------------------- normalising
def _r(x):
    return round(float(x), 6)


def _num(x):
    """Compact number formatting used by the DSL (round-trips through float())."""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ("%.6f" % x).rstrip("0").rstrip(".")


def _dir(d):
    s = str(d).strip().upper()
    if s in ("L", "LEFT"):
        return "L"
    if s in ("R", "RIGHT"):
        return "R"
    raise ValueError(f"direction must be 'left' or 'right' (or 'L'/'R'), got {d!r}")


def _span(length):
    key = str(length).strip()
    if key.lower() in ("short", "normal", "long"):
        return key.lower()
    if key.upper() in _SPAN_FROM_LETTER:
        return _SPAN_FROM_LETTER[key.upper()]
    raise ValueError(f"length must be 'short', 'normal' or 'long' (or S/N/L), got {length!r}")


def _severity(sev):
    if int(sev) != sev or not 1 <= int(sev) <= 6:
        raise ValueError(f"severity must be an integer 1..6, got {sev!r}")
    return int(sev)


def _positive(name, value, upper=20000.0):
    v = float(value)
    if not (0.0 < v <= upper) or not math.isfinite(v):
        raise ValueError(f"{name} must be > 0 and <= {upper:g} m, got {value!r}")
    return _r(v)


def _radius(value):
    v = float(value)
    if not v >= MIN_RADIUS:
        raise ValueError(f"radius {value!r} m is below the minimum corner radius {MIN_RADIUS:g} m")
    return _r(v)


# -------------------------------------------------------------------- segment

@dataclass(frozen=True)
class Segment:
    """One script segment. Build these through :class:`TrackScript`.

    Fields that do not apply to a kind are left at their defaults so that
    equal segments compare equal however they were written.
    """

    kind: str
    direction: str = ""
    length: float = 0.0
    severity: int = 0
    span: str = ""
    count: int = 0
    angle: float = 0.0
    radius: Optional[float] = None
    leg: Optional[float] = None
    height: float = 0.0
    shape: str = ""
    grade: float = 0.0          # signed fraction (0.12 = 12 % climb)
    tightens: bool = False
    opens: bool = False
    caution: bool = False
    high_speed: bool = False

    def __post_init__(self):
        put = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        k = self.kind
        if k not in KINDS:
            raise ValueError(f"unknown segment kind {k!r}")
        f = dict(direction="", length=0.0, severity=0, span="", count=0, angle=0.0,
                 radius=None, leg=None, height=0.0, shape="", grade=0.0,
                 tightens=False, opens=False, caution=False, high_speed=False)
        if k == "straight":
            f["length"] = _positive("straight length", self.length)
            f["high_speed"] = bool(self.high_speed)
        elif k == "corner":
            f["direction"] = _dir(self.direction)
            f["severity"] = _severity(self.severity)
            f["span"] = _span(self.span or "normal")
            f["tightens"], f["opens"] = bool(self.tightens), bool(self.opens)
            if f["tightens"] and f["opens"]:
                raise ValueError("a corner cannot both tighten and open")
            f["caution"] = bool(self.caution)
        elif k == "hairpin":
            f["direction"] = _dir(self.direction)
            f["radius"] = _radius(HAIRPIN_RADIUS if self.radius is None else self.radius)
        elif k == "switchback":
            f["direction"] = _dir(self.direction)
            f["radius"] = _radius(HAIRPIN_RADIUS if self.radius is None else self.radius)
            leg = SWITCHBACK_LEG if self.leg is None else float(self.leg)
            if not 0.0 <= leg <= 20000.0:
                raise ValueError(f"switchback leg must be 0..20000 m, got {self.leg!r}")
            f["leg"] = _r(leg)
        elif k == "zigzag":
            f["direction"] = _dir(self.direction or "L")
            f["severity"] = _severity(self.severity or ZIGZAG_SEVERITY)
            f["span"] = _span(self.span or "short")
            n = int(self.count)
            if int(self.count) != self.count or not 1 <= n <= 50:
                raise ValueError(f"zigzag count must be an integer 1..50, got {self.count!r}")
            f["count"] = n
        elif k == "turn" or k == "sweep":
            f["direction"] = _dir(self.direction)
            a = float(self.angle)
            if not 0.0 < a <= 360.0:
                raise ValueError(f"turn angle must be > 0 and <= 360 degrees, got {self.angle!r}")
            f["angle"] = _r(a)
            default = TURN_RADIUS if k == "turn" else SWEEP_RADIUS
            f["radius"] = _radius(default if self.radius is None else self.radius)
        elif k == "elev":
            if self.shape not in ELEV_SHAPES:
                raise ValueError(f"elevation shape must be one of {ELEV_SHAPES}, got {self.shape!r}")
            f["shape"] = self.shape
            f["height"] = _positive("elevation change", self.height, upper=5000.0)
            f["length"] = _positive("elevation run length", self.length)
        elif k == "steep":
            g = float(self.grade)
            if g == 0.0 or abs(g) > 1.0 or not math.isfinite(g):
                raise ValueError(f"steep grade must be a nonzero fraction with |grade| <= 1, got {self.grade!r}")
            f["shape"] = "grade"
            f["grade"] = _r(g)
            f["length"] = _positive("steep length", self.length)
        for name, value in f.items():
            put(name, value)


# ---------------------------------------------------------------------- DSL
_NUM = r"(\d+(?:\.\d+)?)"
_RE = {
    "hs": re.compile(rf"HS{_NUM}"),
    "straight": re.compile(rf"S{_NUM}"),
    "corner": re.compile(r"([LR])([1-6])([SNL]?)([<>!]*)"),
    "hairpin": re.compile(rf"HAIRPIN-([LR])(?:/{_NUM})?"),
    "switchback": re.compile(rf"SWITCHBACK-([LR])(?:/{_NUM}(?:/{_NUM})?)?"),
    "zigzag": re.compile(r"ZIGZAG(\d+)(?:-([LR])([1-6])([SNL]?))?"),
    "esses": re.compile(r"ESSES(\d+)(?:-([LR]))?"),
    "turn": re.compile(rf"TURN-([LR]){_NUM}(?:/{_NUM})?"),
    "sweep": re.compile(rf"SWEEP-([LR]){_NUM}(?:/{_NUM})?"),
    "elev": re.compile(rf"(DROP|CLIMB|CREST|DIP)-{_NUM}/{_NUM}"),
    "steep": re.compile(rf"GRADE([+-]){_NUM}/{_NUM}"),
}
_DSL_HELP = (
    "tokens: S<m> HS<m> <L|R><1-6>[S|N|L][<|>|!] HAIRPIN-<L|R>[/<r>] "
    "SWITCHBACK-<L|R>[/<r>[/<leg>]] ZIGZAG<n>[-<L|R><1-6>[S|N|L]] ESSES<n>[-<L|R>] "
    "TURN-<L|R><deg>[/<r>] SWEEP-<L|R><deg>[/<r>] DROP|CLIMB|CREST|DIP-<h>/<m> "
    "GRADE<+|-><pct>/<m> CLOSE"
)


def _parse_token(tok):
    """Return a Segment, or the string 'CLOSE'."""
    if tok == "CLOSE":
        return "CLOSE"
    m = _RE["hs"].fullmatch(tok)
    if m:
        return Segment("straight", length=float(m.group(1)), high_speed=True)
    m = _RE["straight"].fullmatch(tok)
    if m:
        return Segment("straight", length=float(m.group(1)))
    m = _RE["corner"].fullmatch(tok)
    if m:
        d, sev, span, mods = m.groups()
        if mods.count("<") + mods.count(">") > 1 or mods.count("!") > 1:
            raise ValueError("repeated modifier")
        return Segment("corner", direction=d, severity=int(sev), span=_SPAN_FROM_LETTER.get(span, "normal"),
                       tightens=">" in mods, opens="<" in mods, caution="!" in mods)
    m = _RE["hairpin"].fullmatch(tok)
    if m:
        r = float(m.group(2)) if m.group(2) else None
        return Segment("hairpin", direction=m.group(1), radius=r)
    m = _RE["switchback"].fullmatch(tok)
    if m:
        r = float(m.group(2)) if m.group(2) else None
        leg = float(m.group(3)) if m.group(3) else None
        return Segment("switchback", direction=m.group(1), radius=r, leg=leg)
    m = _RE["zigzag"].fullmatch(tok)
    if m:
        n, d, sev, span = m.groups()
        if d is None:
            d, sev, span = "L", str(ZIGZAG_SEVERITY), ""
        return Segment("zigzag", count=int(n), direction=d, severity=int(sev),
                       span=_SPAN_FROM_LETTER.get(span, "short") if span else "short")
    m = _RE["esses"].fullmatch(tok)
    if m:
        return Segment("zigzag", count=int(m.group(1)), direction=m.group(2) or "L",
                       severity=ESSES_SEVERITY, span="short")
    m = _RE["turn"].fullmatch(tok)
    if m:
        r = float(m.group(3)) if m.group(3) else None
        return Segment("turn", direction=m.group(1), angle=float(m.group(2)), radius=r)
    m = _RE["sweep"].fullmatch(tok)
    if m:
        r = float(m.group(3)) if m.group(3) else None
        return Segment("sweep", direction=m.group(1), angle=float(m.group(2)), radius=r)
    m = _RE["elev"].fullmatch(tok)
    if m:
        return Segment("elev", shape=m.group(1).lower(), height=float(m.group(2)), length=float(m.group(3)))
    m = _RE["steep"].fullmatch(tok)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return Segment("steep", grade=sign * float(m.group(2)) / 100.0, length=float(m.group(3)))
    raise ValueError(f"unrecognised token {tok!r}")


def _token(seg):
    """Canonical DSL token for a segment."""
    k = seg.kind
    if k == "straight":
        return ("HS" if seg.high_speed else "S") + _num(seg.length)
    if k == "corner":
        letter = "" if seg.span == "normal" else _LETTER_FROM_SPAN[seg.span]
        mods = (">" if seg.tightens else "") + ("<" if seg.opens else "") + ("!" if seg.caution else "")
        return f"{seg.direction}{seg.severity}{letter}{mods}"
    if k == "hairpin":
        return f"HAIRPIN-{seg.direction}/{_num(seg.radius)}"
    if k == "switchback":
        return f"SWITCHBACK-{seg.direction}/{_num(seg.radius)}/{_num(seg.leg)}"
    if k == "zigzag":
        letter = "" if seg.span == "short" else _LETTER_FROM_SPAN[seg.span]
        return f"ZIGZAG{seg.count}-{seg.direction}{seg.severity}{letter}"
    if k == "turn":
        return f"TURN-{seg.direction}{_num(seg.angle)}/{_num(seg.radius)}"
    if k == "sweep":
        return f"SWEEP-{seg.direction}{_num(seg.angle)}/{_num(seg.radius)}"
    if k == "elev":
        return f"{seg.shape.upper()}-{_num(seg.height)}/{_num(seg.length)}"
    if k == "steep":
        sign = "-" if seg.grade < 0 else "+"
        return f"GRADE{sign}{_num(abs(seg.grade) * 100.0)}/{_num(seg.length)}"
    raise ValueError(f"cannot write kind {k!r}")  # pragma: no cover


# --------------------------------------------------------------- TrackScript
class TrackScript:
    """An ordered list of segments plus a closed/open intent. Chainable.

    ``closed`` is set by :meth:`close` (or ``CLOSE`` in the DSL). It is the
    default for :meth:`build`; pass ``closed=`` explicitly to override.
    """

    def __init__(self, segments: Sequence[Segment] = (), closed: bool = False):
        self.segments: List[Segment] = []
        self.closed = bool(closed)
        for seg in segments:
            if not isinstance(seg, Segment):
                raise TypeError("segments must be Segment objects")
            self.segments.append(seg)

    # ------------------------------------------------------------ builder
    def _add(self, seg: Segment) -> "TrackScript":
        self.segments.append(seg)
        return self

    def straight(self, length: float, *, high_speed: bool = False) -> "TrackScript":
        """Straight of ``length`` metres. ``high_speed=True`` marks a long/fast section."""
        return self._add(Segment("straight", length=length, high_speed=high_speed))

    def long_straight(self, length: float) -> "TrackScript":
        """High-speed straight (DSL ``HS<m>``)."""
        return self.straight(length, high_speed=True)

    def corner(self, direction, severity: int = 3, length="normal", *,
               tightens: bool = False, opens: bool = False, caution: bool = False) -> "TrackScript":
        """Pace-note corner. Severity 1 (hairpin-tight) .. 6 (flat-out kink) sets the radius;
        ``length`` ('short' | 'normal' | 'long') sets the angle (40 / 75 / 120 degrees).
        ``tightens`` / ``opens`` change the radius through the corner; ``caution`` is an advisory flag."""
        return self._add(Segment("corner", direction=direction, severity=severity, span=_span(length),
                                 tightens=tightens, opens=opens, caution=caution))

    def left(self, severity: int = 3, length="normal", **kwargs) -> "TrackScript":
        return self.corner("L", severity, length, **kwargs)

    def right(self, severity: int = 3, length="normal", **kwargs) -> "TrackScript":
        return self.corner("R", severity, length, **kwargs)

    def hairpin(self, direction="left", radius: Optional[float] = None) -> "TrackScript":
        """180 degree turn, default radius 9 m."""
        return self._add(Segment("hairpin", direction=direction, radius=radius))

    def switchback(self, direction="left", radius: Optional[float] = None,
                   leg: Optional[float] = None) -> "TrackScript":
        """Hairpin, a straight ``leg`` (default 30 m), then a hairpin in the opposite direction."""
        return self._add(Segment("switchback", direction=direction, radius=radius, leg=leg))

    def zigzag(self, n: int = 4, direction="left", severity: int = ZIGZAG_SEVERITY,
               length="short") -> "TrackScript":
        """``n`` consecutive corners alternating direction, each as a pace-note corner."""
        return self._add(Segment("zigzag", count=n, direction=direction, severity=severity, span=_span(length)))

    def esses(self, n: int = 3, direction="left") -> "TrackScript":
        """Fast alternating corners: zigzag of severity 5, short."""
        return self.zigzag(n, direction, ESSES_SEVERITY, "short")

    def chicane(self, direction="left", severity: int = ZIGZAG_SEVERITY, length="short") -> "TrackScript":
        """Two alternating corners (an S-bend)."""
        return self.zigzag(2, direction, severity, length)

    def turn(self, angle: float, direction="left", radius: Optional[float] = None) -> "TrackScript":
        """Explicit-angle turn in degrees, default radius 25 m."""
        return self._add(Segment("turn", direction=direction, angle=angle, radius=radius))

    def sweep(self, direction="left", angle: float = 90.0, radius: Optional[float] = None) -> "TrackScript":
        """Long, fast bend of ``angle`` degrees, default radius 80 m."""
        return self._add(Segment("sweep", direction=direction, angle=angle, radius=radius))

    def drop(self, height: float, length: float) -> "TrackScript":
        """Descend ``height`` m over a ``length`` m straight (smooth start and end)."""
        return self._add(Segment("elev", shape="drop", height=height, length=length))

    def climb(self, height: float, length: float) -> "TrackScript":
        """Climb ``height`` m over a ``length`` m straight."""
        return self._add(Segment("elev", shape="climb", height=height, length=length))

    def crest(self, height: float, length: float) -> "TrackScript":
        """Rise ``height`` m then fall back to the entry level over ``length`` m (hill crest)."""
        return self._add(Segment("elev", shape="crest", height=height, length=length))

    def dip(self, height: float, length: float) -> "TrackScript":
        """Fall ``height`` m then rise back (compression / dip)."""
        return self._add(Segment("elev", shape="dip", height=height, length=length))

    def steep(self, grade: float, length: float) -> "TrackScript":
        """Constant grade over ``length`` m. ``grade`` is a signed fraction (0.12 = 12 % climb)."""
        return self._add(Segment("steep", grade=grade, length=length))

    def close(self) -> "TrackScript":
        """Mark the script as a loop. The end is blended back onto the start when built."""
        self.closed = True
        return self

    # ---------------------------------------------------------------- text
    @classmethod
    def from_dsl(cls, text: str) -> "TrackScript":
        """Parse the compact DSL (see README_track_script.md). Raises ValueError with the bad token."""
        if not isinstance(text, str):
            raise TypeError("DSL text must be a string")
        toks = text.upper().split()
        if not toks:
            raise ValueError("empty track script")
        script = cls()
        for i, tok in enumerate(toks):
            try:
                item = _parse_token(tok)
            except ValueError as exc:
                raise ValueError(f"track script token {i + 1} {tok!r}: {exc}. {_DSL_HELP}") from None
            if item == "CLOSE":
                script.closed = True
            else:
                script.segments.append(item)
        if not script.segments:
            raise ValueError("track script has no segments (only CLOSE)")
        return script

    def to_dsl(self) -> str:
        """Canonical DSL text. ``TrackScript.from_dsl(s.to_dsl()) == s``."""
        toks = [_token(seg) for seg in self.segments]
        if self.closed:
            toks.append("CLOSE")
        return " ".join(toks)

    def copy(self) -> "TrackScript":
        return TrackScript(self.segments, self.closed)

    def __eq__(self, other):
        if not isinstance(other, TrackScript):
            return NotImplemented
        return self.segments == other.segments and self.closed == other.closed

    def __len__(self):
        return len(self.segments)

    def __repr__(self):
        return f"TrackScript({self.to_dsl()!r})"

    # --------------------------------------------------------------- build
    def build(self, terrain: Optional[Terrain] = None, *, closed: Optional[bool] = None,
              width: float = DEFAULT_WIDTH, spacing: float = DEFAULT_SPACING,
              max_grade: float = DEFAULT_MAX_GRADE, min_radius: float = MIN_RADIUS,
              close_tolerance: float = 0.5, close_gap_max: float = 50.0,
              close_blend: Optional[float] = None, self_intersection: str = "error",
              heading0: float = 0.0, strict: bool = False, goal=None) -> Track:
        """Build the :class:`Track`. See :func:`build_world_from_script` for the options.

        ``terrain`` is optional. With a terrain, unspecified elevation follows
        it; without one it is flat (0 m) there.
        """
        return _build(self, terrain, closed=closed, width=width, spacing=spacing, max_grade=max_grade,
                      min_radius=min_radius, close_tolerance=close_tolerance,
                      close_gap_max=close_gap_max, close_blend=close_blend,
                      self_intersection=self_intersection, heading0=heading0, strict=strict, goal=goal)


def parse_track_script(text: str) -> TrackScript:
    """Same as ``TrackScript.from_dsl(text)``."""
    return TrackScript.from_dsl(text)


def _as_script(script) -> TrackScript:
    if isinstance(script, TrackScript):
        return script
    if isinstance(script, str):
        return TrackScript.from_dsl(script)
    raise TypeError("script must be a TrackScript or a DSL string")


# ------------------------------------------------------------------ geometry
@dataclass
class _Piece:
    length: float
    k: Callable            # curvature (1/m) at u in [0, length]
    g: Optional[Callable] = None   # explicit grade at u, or None for terrain-following


def _zero_k(u):
    return np.zeros_like(np.asarray(u, dtype=float))


def _corner_piece(theta: float, R0: float, R1: Optional[float] = None) -> _Piece:
    """Corner turning ``theta`` radians with radius R0 at entry and R1 at exit.

    Curvature ramps linearly from 0 to K0 (clothoid entry), varies linearly
    from K0 to K1 over the hold, and ramps linearly back to 0 (clothoid
    exit). Curvature is continuous everywhere and the heading change is exactly
    ``theta``. The transition length is ``T = min(max(0.3 * arc, 6), 40, 0.4 * arc)``.
    """
    R1 = R0 if R1 is None else R1
    sgn = 1.0 if theta >= 0 else -1.0
    K0, K1 = sgn / R0, sgn / R1
    arc = abs(theta) * 0.5 * (R0 + R1)
    T = min(max(0.3 * arc, 6.0), 40.0, 0.4 * arc)
    Lh = 2.0 * theta / (K0 + K1) - T
    if Lh < 0.2:
        raise ValueError(f"corner of {math.degrees(theta):.1f} degrees at radius {R0:g} m is too short "
                         "to hold its transitions")
    L = Lh + 2.0 * T

    def k(u):
        u = np.asarray(u, dtype=float)
        out = np.zeros_like(u)
        a = u < T
        b = (~a) & (u < T + Lh)
        c = ~(a | b)
        out[a] = K0 * u[a] / T
        out[b] = K0 + (K1 - K0) * (u[b] - T) / Lh
        out[c] = K1 * np.clip(1.0 - (u[c] - T - Lh) / T, 0.0, 1.0)
        return out

    return _Piece(L, k)


def _grade_shape(L: float):
    """Unit-mean smooth bump on v in [0, 1] with zero slope at both ends."""
    r = min(RAMP_M / L, 0.2)

    def w(v):
        a = np.clip(v / r, 0.0, 1.0)
        b = np.clip((1.0 - v) / r, 0.0, 1.0)
        return (a * a * (3 - 2 * a)) * (b * b * (3 - 2 * b))

    grid = (np.arange(4000) + 0.5) / 4000.0
    mean = float(np.mean(w(grid)))
    return lambda v: w(np.asarray(v, dtype=float)) / mean


def _elev_piece(shape: str, height: float, length: float, grade: float = 0.0) -> _Piece:
    """Elevation change over a straight of ``length`` metres.

    drop / climb / grade: grade = (+-H/L) times a unit-mean smooth bump, so the
    net change is exactly H. crest / dip: z = +-H sin^2(pi v), net change 0.
    """
    L = float(length)
    if shape == "crest":
        g = lambda u: (math.pi * height / L) * np.sin(2 * math.pi * np.asarray(u, dtype=float) / L)  # noqa: E731
    elif shape == "dip":
        g = lambda u: -(math.pi * height / L) * np.sin(2 * math.pi * np.asarray(u, dtype=float) / L)  # noqa: E731
    else:
        amp = grade if shape == "grade" else (-height / L if shape == "drop" else height / L)
        m = _grade_shape(L)
        g = lambda u: amp * m(np.asarray(u, dtype=float) / L)  # noqa: E731
    return _Piece(L, _zero_k, g)


def _pieces_for(seg: Segment):
    k = seg.kind
    sgn = 1.0 if seg.direction == "L" else -1.0
    if k == "straight":
        return [_Piece(seg.length, _zero_k)]
    if k == "corner":
        R0 = RADIUS_BY_SEVERITY[seg.severity]
        R1 = R0 * (TIGHTEN_FACTOR if seg.tightens else OPEN_FACTOR if seg.opens else 1.0)
        return [_corner_piece(sgn * math.radians(ANGLE_BY_SPAN[seg.span]), R0, R1)]
    if k == "hairpin":
        return [_corner_piece(sgn * math.pi, seg.radius)]
    if k == "switchback":
        return [_corner_piece(sgn * math.pi, seg.radius), _Piece(seg.leg, _zero_k),
                _corner_piece(-sgn * math.pi, seg.radius)]
    if k == "zigzag":
        out = []
        for i in range(seg.count):
            s_i = sgn * (-1.0) ** i
            out.append(_corner_piece(s_i * math.radians(ANGLE_BY_SPAN[seg.span]), RADIUS_BY_SEVERITY[seg.severity]))
        return out
    if k in ("turn", "sweep"):
        return [_corner_piece(sgn * math.radians(seg.angle), seg.radius)]
    if k == "elev":
        return [_elev_piece(seg.shape, seg.height, seg.length)]
    if k == "steep":
        return [_elev_piece("grade", 0.0, seg.length, grade=seg.grade)]
    raise ValueError(f"unknown kind {k!r}")  # pragma: no cover


class _Plan:
    """Planar path and explicit elevation request, before terrain is applied."""


def _sample(segments, max_grade, step=FINE_DS):
    """Sample every piece at ``step`` metres. Returns per-sample K, DS, explicit G (nan = terrain), owner."""
    Ks, DSs, Gs, owners = [], [], [], []
    for i, seg in enumerate(segments):
        for piece in _pieces_for(seg):
            if piece.length <= 1e-9:
                continue
            n = max(1, int(round(piece.length / step)))
            ds = piece.length / n
            u = (np.arange(n) + 0.5) * ds
            Ks.append(np.asarray(piece.k(u), dtype=float))
            DSs.append(np.full(n, ds))
            if piece.g is None:
                g = np.full(n, np.nan)
            else:
                g = np.asarray(piece.g(u), dtype=float)
                peak = float(np.max(np.abs(g)))
                if peak > max_grade + 1e-9:
                    raise ValueError(
                        f"segment {i + 1} ({_token(seg)}) needs a grade of up to {peak * 100:.1f} %, "
                        f"above max_grade {max_grade * 100:.0f} %. Lengthen the elevation change or "
                        "reduce its height."
                    )
            Gs.append(g)
            owners.append(np.full(n, i))
    if not Ks:
        raise ValueError("track script has no segments")
    return (np.concatenate(Ks), np.concatenate(DSs), np.concatenate(Gs), np.concatenate(owners))


def _integrate_from(K, DS, x0, y0, th0):
    """Planar integration with midpoint headings from a start pose. Returns node x, y, heading."""
    KD = K * DS
    cum = np.cumsum(KD)
    th_mid = th0 + cum - 0.5 * KD
    x = np.concatenate([[x0], x0 + np.cumsum(np.cos(th_mid) * DS)])
    y = np.concatenate([[y0], y0 + np.cumsum(np.sin(th_mid) * DS)])
    th = np.concatenate([[th0], th0 + cum])
    return x, y, th


def _integrate(K, DS, th0):
    return _integrate_from(K, DS, 0.0, 0.0, th0)


def _blend_region(DS, close_blend):
    """Final stretch used for closing: (sample indices, v in [0, 1], length)."""
    Lt = float(DS.sum())
    S_mid = np.cumsum(DS) - 0.5 * DS
    Lb = close_blend if close_blend is not None else min(max(30.0, 0.15 * Lt), 120.0)
    Lb = min(float(Lb), 0.9 * Lt)
    sb = Lt - Lb
    idx = np.nonzero(S_mid >= sb)[0]
    return idx, (S_mid[idx] - sb) / Lb, Lb


def _blend_solve(K, DS, idx, v, heading0, close_tolerance, min_radius):
    """Least-squares blend of curvature over the closing region. Returns (K, gap, heading error)."""
    B = np.column_stack([math.comb(5, j) * v ** j * (1.0 - v) ** (5 - j) for j in range(1, 5)])

    def resid(c):
        Kc = K.copy()
        Kc[idx] += B @ c
        xx, yy, tt = _integrate(Kc, DS, heading0)
        return np.concatenate([[xx[-1], yy[-1], wrap_angle(tt[-1] - heading0)], 1e-3 * c])

    sol = least_squares(resid, np.zeros(4), method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    Kc = K.copy()
    Kc[idx] += B @ sol.x
    x, y, th = _integrate(Kc, DS, heading0)
    gap = float(np.hypot(x[-1], y[-1]))
    dth = float(wrap_angle(th[-1] - heading0))
    if gap > close_tolerance or abs(dth) > HEADING_TOLERANCE:
        raise ValueError(
            f"could not close the loop by blending the final stretch: {gap:.2f} m / "
            f"{math.degrees(dth):.2f} deg remain (tolerance {close_tolerance:g} m / "
            f"{math.degrees(HEADING_TOLERANCE):.1f} deg)"
        )
    kmax = float(np.max(np.abs(Kc)))
    if kmax > 1.0 / min_radius + 1e-9:
        raise ValueError(f"the closing blend tightens the track to {1.0 / kmax:.1f} m radius, "
                         f"below min_radius {min_radius:g} m")
    return Kc, gap, dth


def _crossing(x, y, s, closed, width, total):
    """True if two points closer than the road clearance are far apart along the route."""
    clear = width + 2.0 * SHOULDER_WIDTH
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(clear, output_type="ndarray")
    if len(pairs) == 0:
        return False
    d = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
    if closed:
        d = np.minimum(d, total - d)
    return bool(np.any(d > 3.0 * clear))


def _route_is_clear(segs, h0, closed, width):
    K, DS, _, _ = _sample(segs, np.inf)
    x, y, _ = _integrate_from(K, DS, 0.0, 0.0, h0)
    S = np.concatenate([[0.0], np.cumsum(DS)])
    stride = max(1, int(round(1.5 / FINE_DS)))
    idx = np.arange(0, len(x), stride)
    return not _crossing(x[idx], y[idx], S[idx], closed, width, float(S[-1]))


_TEMPLATE_TRIES = 4


def _complete(segs, K, DS, h0, *, closed, goal, min_radius, width):
    """Procedurally append three corners and two straights that bring the route to its target.

    closed: target is the start pose (position and heading). Open with ``goal``: target is
    the goal position, heading free. Solutions are checked for self-intersection and
    the cheapest valid one is returned as a list of Segments (or None if none is found).
    """
    x, y, th = _integrate_from(K, DS, 0.0, 0.0, h0)
    ex, ey, eth = float(x[-1]), float(y[-1]), float(th[-1])
    tx, ty = (0.0, 0.0) if closed else (float(goal[0]), float(goal[1]))
    lo = np.array([0.1, 0.1, 0.1, 0.5, 0.5] + [1.2 * min_radius] * 3)
    hi = np.array([2 * np.pi] * 3 + [300.0, 300.0] + [120.0] * 3)
    rng = np.random.default_rng(0)

    def tmpl(p, signs):
        d = [("L" if s > 0 else "R") for s in signs]
        return [
            Segment("turn", direction=d[0], angle=math.degrees(p[0]), radius=float(p[5])),
            Segment("straight", length=float(p[3])),
            Segment("turn", direction=d[1], angle=math.degrees(p[1]), radius=float(p[6])),
            Segment("straight", length=float(p[4])),
            Segment("turn", direction=d[2], angle=math.degrees(p[2]), radius=float(p[7])),
        ]

    found = []
    for signs in itertools.product((1.0, -1.0), repeat=3):
        def resid(p, signs=signs):
            Kt, DSt, _, _ = _sample(tmpl(p, signs), np.inf, step=0.5)
            xe, ye, te = _integrate_from(Kt, DSt, ex, ey, eth)
            r = [xe[-1] - tx, ye[-1] - ty]
            if closed:
                r.append(float(wrap_angle(te[-1] - h0)))
            return np.array(r)

        for _ in range(_TEMPLATE_TRIES):
            p0 = lo + rng.uniform(size=8) * (hi - lo)
            sol = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                                xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=80)
            if np.max(np.abs(sol.fun[:2])) > 0.005 or (closed and abs(sol.fun[2]) > 1e-4):
                continue
            full = list(segs) + tmpl(sol.x, signs)
            if _route_is_clear(full, h0, closed, width):
                Kt, DSt, _, _ = _sample(tmpl(sol.x, signs), np.inf, step=0.5)
                found.append((float(DSt.sum()), tmpl(sol.x, signs)))
            if found:
                break
        if found:
            break
    if not found:
        return None
    return min(found, key=lambda f: f[0])[1]


def _plan(segments, *, closed, heading0, min_radius, max_grade, close_tolerance, close_gap_max,
          close_blend, strict=False, goal=None, width=DEFAULT_WIDTH):
    """Plan the planar route. Closes or extends it (unless strict) and returns a _Plan."""
    if not segments:
        raise ValueError("track script has no segments")
    segs = list(segments)
    K, DS, G, owner = _sample(segs, max_grade)
    kmax = float(np.max(np.abs(K)))
    if kmax > 1.0 / min_radius + 1e-9:
        i = int(owner[int(np.argmax(np.abs(K)))])
        raise ValueError(f"segment {i + 1} ({_token(segs[i])}) has radius {1.0 / kmax:.1f} m, "
                         f"below min_radius {min_radius:g} m")

    closure = dict(applied=False, completed=False, completion_length=0.0, gap_before=0.0,
                   heading_before=0.0, gap_after=0.0, heading_after=0.0, blend_length=0.0,
                   vertical_residual=0.0)
    x, y, th = _integrate(K, DS, heading0)

    if closed:
        gap0 = float(np.hypot(x[-1], y[-1]))
        dth0 = float(wrap_angle(th[-1] - heading0))
        closure.update(gap_before=gap0, heading_before=dth0, gap_after=gap0, heading_after=dth0)
        if gap0 > 1e-6 or abs(dth0) > 1e-9:
            K0, DS0 = K, DS
            problem = None
            if gap0 > close_gap_max:
                problem = (f"the script ends {gap0:.1f} m and {math.degrees(dth0):.1f} deg away from its "
                           f"start, more than close_gap_max = {close_gap_max:g} m")
            else:
                try:
                    idx, v, _ = _blend_region(DS, close_blend)
                    K, gap1, dth1 = _blend_solve(K, DS, idx, v, heading0, close_tolerance, min_radius)
                    closure.update(applied=True, gap_after=gap1, heading_after=dth1)
                except ValueError as exc:
                    K = K0
                    problem = str(exc)
            if problem is not None:
                if strict:
                    raise ValueError(f"{problem}. With strict=True the route is not completed; "
                                     "adjust the segments or use strict=False.")
                extra = _complete(segs, K0, DS0, heading0, closed=True, goal=None,
                                  min_radius=min_radius, width=width)
                if extra is None:
                    raise ValueError(f"could not auto-complete the route ({problem}). Add segments that "
                                     "bring it back toward the start, or use closed=False.")
                segs = segs + extra
                K, DS, G, owner = _sample(segs, max_grade)
                # the completion was solved on a coarse step; a light blend removes the residual at full resolution
                idx, v, _ = _blend_region(DS, close_blend)
                try:
                    K, gap1, dth1 = _blend_solve(K, DS, idx, v, heading0, close_tolerance, min_radius)
                except ValueError as exc:
                    raise ValueError(f"auto-completed route could not be closed exactly: {exc}") from None
                closure.update(completed=True, completion_length=float(DS.sum() - DS0.sum()),
                               gap_after=gap1, heading_after=dth1)
    elif goal is not None:
        gx, gy = float(goal[0]), float(goal[1])
        d = float(np.hypot(x[-1] - gx, y[-1] - gy))
        if d > close_tolerance:
            if strict:
                raise ValueError(f"the route ends {d:.1f} m from the goal; with strict=True it is not "
                                 "extended. Adjust the segments or use strict=False.")
            extra = _complete(segs, K, DS, heading0, closed=False, goal=(gx, gy),
                              min_radius=min_radius, width=width)
            if extra is None:
                raise ValueError(f"could not extend the route to the goal ({d:.1f} m away). Add segments.")
            segs = segs + extra
            K, DS, G, owner = _sample(segs, max_grade)
            x, y, th = _integrate(K, DS, heading0)
            closure.update(completed=True, completion_length=float(DS.sum()) - float(sum(
                DS[owner == i].sum() for i in range(len(segments)))))

    if closed:
        _, _, closure["blend_length"] = _blend_region(DS, close_blend)
        blend_idx, blend_v, _ = _blend_region(DS, close_blend)
    else:
        blend_idx, blend_v = np.zeros(0, dtype=int), np.zeros(0)

    x, y, th = _integrate(K, DS, heading0)
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    plan = _Plan()
    plan.x = x - cx
    plan.y = y - cy
    plan.th = th
    plan.K = K
    plan.DS = DS
    plan.G = G
    plan.owner = owner
    plan.s_nodes = np.concatenate([[0.0], np.cumsum(DS)])
    plan.closed = bool(closed)
    plan.closure = closure
    plan.blend_idx = blend_idx
    plan.blend_v = blend_v
    plan.segments = segs
    plan.n_user = len(segments)
    return plan


def _materialise(plan, terrain, script, width, spacing, max_grade, min_radius, self_intersection):
    N = len(plan.DS)
    DS = plan.DS
    if terrain is not None:
        zt = terrain.height_at(plan.x, plan.y)
        zs = gaussian_filter1d(zt, GRADE_SMOOTHING_M / FINE_DS, mode="nearest")
        gt = np.clip(np.diff(zs) / DS, -max_grade, max_grade)
        z0 = float(zt[0])
    else:
        gt = np.zeros(N)
        z0 = 0.0
    G = np.where(np.isnan(plan.G), gt, plan.G)

    def integrate_z(GG):
        return z0 + np.concatenate([[0.0], np.cumsum(GG * DS)])

    Z = integrate_z(G)
    if plan.closed and len(plan.blend_idx):
        res = float(Z[-1] - Z[0])
        plan.closure["vertical_residual"] = res
        if abs(res) > 1e-9:
            vv = plan.blend_v
            w = 1.0 - vv ** 5 - (1.0 - vv) ** 5
            a = -res / float(np.sum(w * DS[plan.blend_idx]))
            G = G.copy()
            G[plan.blend_idx] += a * w
            Z = integrate_z(G)
    if np.max(np.abs(G)) > max_grade + 1e-6:
        raise ValueError(f"the track needs a grade of {np.max(np.abs(G)) * 100:.1f} %, above max_grade "
                         f"{max_grade * 100:.0f} %; the closing blend is too steep. Adjust the final segments.")

    X, Y, S = plan.x, plan.y, plan.s_nodes
    if plan.closed:
        gap = float(np.hypot(X[-1] - X[0], Y[-1] - Y[0]))
        X = np.concatenate([X, [X[0]]])
        Y = np.concatenate([Y, [Y[0]]])
        Z = np.concatenate([Z, [Z[0]]])
        S = np.concatenate([S, [S[-1] + gap]])
        L = float(S[-1])
        n_out = max(8, int(round(L / spacing)))
        t = np.arange(n_out) * (L / n_out)
        ds = L / n_out
    else:
        L = float(S[-1])
        n_out = max(4, int(round(L / spacing)) + 1)
        t = np.linspace(0.0, L, n_out)
        ds = L / (n_out - 1)
    P = np.column_stack([np.interp(t, S, X), np.interp(t, S, Y), np.interp(t, S, Z)])

    if terrain is not None:
        margin = width / 2.0 + SHOULDER_WIDTH + EMBANKMENT_WIDTH + 5.0
        half = terrain.size / 2.0 - margin
        reach = float(np.max(np.abs(P[:, :2])))
        if reach > half:
            raise ValueError(f"track reaches {reach:.0f} m from its centre but the terrain only allows "
                             f"{half:.0f} m. Use a larger terrain size.")

    _check_self_intersection(P, plan.closed, ds, width, self_intersection)

    track = Track(P, width, ds, plan.closed, min_radius=min_radius)
    track.script = script.copy()
    track.closure = dict(plan.closure)
    track.segments = _segment_table(plan, S, np.asarray(plan.th), Z)
    track.completion_segments = [_token(seg) for seg in plan.segments[plan.n_user:]]
    return track


def _segment_table(plan, S, th, Z):
    rows = []
    for i, seg in enumerate(plan.segments):
        idx = np.nonzero(plan.owner == i)[0]
        if len(idx) == 0:
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        rows.append(SegmentInfo(
            index=i, kind=seg.kind, text=_token(seg), s0=float(S[a]), s1=float(S[b]),
            heading_change=float(th[b] - th[a]), elevation_change=float(Z[b] - Z[a]),
            caution=bool(seg.caution),
        ))
    return rows


def _check_self_intersection(P, closed, ds, width, mode):
    if mode not in ("error", "warn", "ignore"):
        raise ValueError("self_intersection must be 'error', 'warn' or 'ignore'")
    if mode == "ignore":
        return
    clear = width + 2.0 * SHOULDER_WIDTH
    n = len(P)
    pairs = cKDTree(P[:, :2]).query_pairs(clear, output_type="ndarray")
    if len(pairs) == 0:
        return
    d = np.abs(pairs[:, 0] - pairs[:, 1])
    if closed:
        d = np.minimum(d, n - d)
    bad = pairs[d * ds > 3.0 * clear]
    if len(bad) == 0:
        return
    i, j = bad[0]
    msg = (f"track crosses or touches itself near ({P[i, 0]:.0f}, {P[i, 1]:.0f}) m, where arc positions "
           f"{i * ds:.0f} m and {j * ds:.0f} m meet. Change the loop geometry, or pass "
           "self_intersection='warn' to keep it.")
    if mode == "error":
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=3)


def _make_plan(script, *, closed, width, spacing, max_grade, min_radius, close_tolerance,
               close_gap_max, close_blend, self_intersection, heading0, strict, goal):
    script = _as_script(script)
    closed_flag = script.closed if closed is None else bool(closed)
    plan = _plan(script.segments, closed=closed_flag, heading0=heading0, min_radius=min_radius,
                 max_grade=max_grade, close_tolerance=close_tolerance, close_gap_max=close_gap_max,
                 close_blend=close_blend, strict=strict, goal=goal, width=width)
    return script, plan


def _build(script, terrain, *, closed, width, spacing, max_grade, min_radius, close_tolerance,
           close_gap_max, close_blend, self_intersection, heading0, strict=False, goal=None):
    script, plan = _make_plan(script, closed=closed, width=width, spacing=spacing, max_grade=max_grade,
                              min_radius=min_radius, close_tolerance=close_tolerance,
                              close_gap_max=close_gap_max, close_blend=close_blend,
                              self_intersection=self_intersection, heading0=heading0, strict=strict, goal=goal)
    return _materialise(plan, terrain, script, width, spacing, max_grade, min_radius, self_intersection)


def build_world_from_script(script, seed: int = 0, closed: Optional[bool] = None,
                            size: Optional[float] = None, resolution: float = 1.0,
                            width: float = DEFAULT_WIDTH, amplitude: float = 10.0,
                            spacing: float = DEFAULT_SPACING, patches: bool = True,
                            max_grade: float = DEFAULT_MAX_GRADE, min_radius: float = MIN_RADIUS,
                            close_tolerance: float = 0.5, close_gap_max: float = 50.0,
                            close_blend: Optional[float] = None, self_intersection: str = "error",
                            heading0: float = 0.0, strict: bool = False, goal=None):
    """Build a terrain and a scripted track on it, carved like :func:`drivesim.world.generate_world`.

    Parameters
    ----------
    script : :class:`TrackScript` or DSL string.
    seed : terrain seed (non-negative int). The track geometry is fully determined by the script.
    closed : loop (True) or point-to-point (False). ``None`` uses the script's CLOSE flag.
    size : terrain side length (m). ``None`` uses 600 m, or the smallest multiple of 100 m
        that fits the route. An explicit size that is too small raises ValueError.
    resolution, amplitude, patches : terrain options (as in ``generate_world``).
    width, spacing : road width and centreline sample spacing (m).
    max_grade : limit on |dz/ds| (default 0.25). Explicit elevation changes above it raise.
    min_radius : smallest allowed corner radius (default 8 m).
    close_tolerance : after closing, the end must lie within this distance of the start (m).
    close_gap_max : a loop whose end is further than this from the start raises (m).
    close_blend : length of the final closing blend (m). Default 30..120 m, scaled to the route.
    self_intersection : 'error' (default), 'warn' or 'ignore'.
    heading0 : initial heading (rad, 0 = +x).
    strict : False (default) completes an incomplete route automatically (see below).
        True raises ValueError instead of completing or extending.
    goal : open tracks only, (x, y) in metres from the start (heading0 frame). If given and the
        script ends elsewhere, the route is extended to it (unless strict).

    Auto-completion: a closed route whose end is more than ``close_tolerance`` from the start
    is first closed by blending its final stretch (only if the gap is within ``close_gap_max``).
    If that fails or the gap is too large, the builder appends three corners and two straights
    whose radii and angles are solved by least squares so the route returns to the start pose
    and heading, keeping every corner at least ``min_radius`` and avoiding self-intersection.
    ``track.closure['completed']`` reports it and ``track.completion_segments`` lists the added
    segments. ``track.script`` is always the script as written.

    Returns ``(terrain, track)``. ``track.segments`` lists each segment's
    arc-length range, heading change and elevation change; ``track.closure``
    reports the closing blend.
    """
    script, plan = _make_plan(script, closed=closed, width=width, spacing=spacing, max_grade=max_grade,
                              min_radius=min_radius, close_tolerance=close_tolerance,
                              close_gap_max=close_gap_max, close_blend=close_blend,
                              self_intersection=self_intersection, heading0=heading0, strict=strict, goal=goal)
    margin = width / 2.0 + SHOULDER_WIDTH + EMBANKMENT_WIDTH + 5.0
    reach = float(max(np.abs(plan.x).max(), np.abs(plan.y).max()))
    need = 2.0 * (reach + margin)
    if size is None:
        size = 600.0 if need <= 600.0 else float(np.ceil(need / 100.0) * 100.0)
    elif need > size:
        raise ValueError(f"route needs a terrain of at least {need:.0f} m, got size={size:g}")
    terrain = Terrain.generate(int(seed), size=size, resolution=resolution,
                               amplitude=amplitude, patches=patches)
    track = _materialise(plan, terrain, script, width, spacing, max_grade, min_radius, self_intersection)
    carve_track(terrain, track)
    return terrain, track


# ------------------------------------------------------------------ examples
# Two identical halves, each turning through 180 degrees, close exactly: the second half is the
# first rotated by 180 degrees, so their displacements cancel. This keeps the examples closed
# without relying on the blend.
_RALLY_HALF = ("S120 L2N S60 R4S S40 HAIRPIN-L/9 S80 R2N S70 L2S S100 DROP-12/70 S40 CLIMB-12/70 "
               "S60 ZIGZAG4-R3S S120")
_OVAL_HALF = "HS400 SWEEP-L180/70 S60 SWEEP-R90/60 SWEEP-L90/60 ZIGZAG2-L3S ZIGZAG2-R4N S40"

EXAMPLES = {
    # Technical rally stage: pace-note corners, a hairpin, esses, a descent and a climb.
    "technical_rally": f"{_RALLY_HALF} {_RALLY_HALF} CLOSE",
    # High-speed circuit: long straights, sweepers and chicanes.
    "high_speed_oval": f"{_OVAL_HALF} {_OVAL_HALF} CLOSE",
    # Point-to-point hill climb: a climb with stacked switchbacks, ending on a crest.
    "switchback_hillclimb": (
        "S40 CLIMB-18/110 SWITCHBACK-L/9/30 S20 SWITCHBACK-L/9/30 S20 "
        "SWITCHBACK-L/9/30 S20 GRADE+10/60 SWEEP-R90/60 CREST-3/50 S30"
    ),
}


# Showcase circuit for a human driver (about 2.8 km, closed, default 10 m road). Two identical
# halves, each turning 180 degrees, so it closes exactly. Each half has a long fast straight into
# a hard hairpin, a sweeper pair, a zigzag, two 90 degree corners, a tightening corner, a crest,
# a drop and a climb, then a high-speed run back.
_SHOWCASE_HALF = ("HS200 HAIRPIN-L/9 S30 SWEEP-R50/90 SWEEP-L50/90 S30 ZIGZAG4-R3S S30 "
                  "TURN-L90/30 S30 TURN-R90/30 S30 R3N> S20 L3N S30 CREST-5/70 S20 DROP-6/80 "
                  "S20 CLIMB-6/80 S20 HS150")
SHOWCASE = f"{_SHOWCASE_HALF} {_SHOWCASE_HALF} CLOSE"

NAMED_SCRIPTS = {**EXAMPLES, "showcase": SHOWCASE}


def build_showcase_world(seed: int = 0):
    """Terrain and track for the ``showcase`` circuit (about 2.8 km). Returns ``(terrain, track)``."""
    return build_world_from_script(NAMED_SCRIPTS["showcase"], seed=seed)


def example_script(name: str) -> TrackScript:
    """Return one of the named example scripts (see ``EXAMPLES``)."""
    if name not in EXAMPLES:
        raise KeyError(f"unknown example {name!r}; choose from {sorted(EXAMPLES)}")
    return TrackScript.from_dsl(EXAMPLES[name])
