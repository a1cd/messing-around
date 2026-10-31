"""Procedural race track (circuit or point-to-point) and track queries.

Frames and conventions
----------------------
* World frame as in :mod:`drivesim.terrain` (x east, y north, z up).
* Yaw / heading: radians, 0 along +x, increasing counter-clockwise.
* ``lateral`` offset: metres, **positive = left** of the track direction.
* Car local frame for lookahead: ``forward`` along the car's heading and
  ``left`` to its left, i.e. ``forward = c*dx + s*dy``, ``left = -s*dx + c*dy``
  with ``c, s = cos(yaw), sin(yaw)`` and ``(dx, dy)`` the world offset.
* ``heading_error`` = track heading minus car yaw, wrapped to (-pi, pi].
  Positive means the track points to the car's left.

The centerline is a polyline with (nearly) uniform spacing ``spacing``
(metres). ``s`` is arc length from the first sample. Closed tracks wrap
(``s`` in ``[0, length)``); open tracks run from ``s=0`` (start A) to
``s=length`` (goal B) and queries are clamped to that range.

Public API
----------
``Track.generate(terrain, rng, closed, difficulty, width, spacing)``
``track.project(pos, hint=None) -> Projection``
``track.project_many(xy) -> dict`` of arrays (used for terrain carving)
``track.point_at(s) -> (xyz, heading)``
``track.curvature_at(s)``  signed curvature 1/m (positive = left turn)
``track.start_pose() -> (xyz, yaw)``
``track.heading_error(pos, yaw, hint=None) -> float``
``track.lookahead(pos, yaw, distances, hint=None) -> (K, 2) array of (forward, left)``
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .terrain import Terrain

# Road geometry shared with terrain carving (see drivesim.world).
SHOULDER_WIDTH = 2.0      # gravel/dirt shoulder on each side of the asphalt (m)
EMBANKMENT_WIDTH = 25.0   # width over which terrain blends into the graded track (m)

LOCAL_WINDOW = 40         # samples searched around a projection hint
TRUST_FACTOR = 3.0        # hint search is abandoned if the car is this many widths away


class Projection(NamedTuple):
    """Result of :meth:`Track.project`."""

    index: int          # nearest centerline sample index
    s: float            # arc length of the projected point (wrapped for closed tracks)
    progress: float     # s / length, in [0, 1]
    lateral: float      # signed offset, positive = left of track direction (m)
    heading: float      # track tangent yaw at the projection (rad)
    point: np.ndarray   # projected centerline point, (3,)
    on_track: bool      # |lateral| <= half width (and within open-track ends)


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _segment_headings(P, closed):
    if closed:
        D = np.roll(P, -1, axis=0) - P
    else:
        D = np.diff(P, axis=0)
    return np.arctan2(D[:, 1], D[:, 0])


def _curvature(th_seg, closed, ds):
    """Signed discrete curvature per node from segment headings."""
    if closed:
        return wrap_angle(th_seg - np.roll(th_seg, 1)) / ds
    k = np.zeros(len(th_seg) + 1)
    k[1:-1] = wrap_angle(np.diff(th_seg)) / ds
    k[0] = k[1]
    k[-1] = k[-2]
    return k


def _resample(P, spacing, closed):
    """Resample a polyline to (nearly) uniform arc-length spacing."""
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    if closed:
        N = max(8, int(round(L / spacing)))
        t = np.arange(N) * (L / N)
        ds = L / N
    else:
        N = max(4, int(round(L / spacing)) + 1)
        t = np.linspace(0.0, L, N)
        ds = L / (N - 1)
    out = np.column_stack([np.interp(t, cum, Q[:, 0]), np.interp(t, cum, Q[:, 1])])
    return out, ds


class Track:
    """A racing track centerline with edges and query helpers.

    Build with :meth:`Track.generate`. Attributes:

    centerline : (N, 3) xyz samples, spacing ``spacing`` metres.
    left_edge, right_edge : (N, 3) edge points at +/- half width.
    width, half_width : road width (m).
    spacing : sample spacing (m).
    length : track length (m).
    closed : True for loop circuits.
    s : (N,) arc length of each sample.
    heading : (N,) tangent yaw at each sample (rad).
    tangent : (N, 2) unit tangent in xy.
    curvature : (N,) signed curvature (1/m) at each sample.
    min_radius : curvature limit used at generation (m); None if unknown.
    goal : (3,) finish point. For point-to-point tracks, the end point B.
           For closed circuits, the start/finish line point (sample 0).
    """

    def __init__(self, centerline, width, spacing, closed, min_radius=None):
        P = np.asarray(centerline, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 4:
            raise ValueError("centerline must be (N, 3) with N >= 4")
        N = P.shape[0]
        self.centerline = P
        self.width = float(width)
        self.half_width = self.width / 2.0
        self.spacing = float(spacing)
        self.closed = bool(closed)
        self.n = N
        self.min_radius = min_radius
        self.length = N * self.spacing if closed else (N - 1) * self.spacing
        self.s = np.arange(N) * self.spacing

        th_seg = _segment_headings(P[:, :2], self.closed)
        self._th = th_seg if closed else np.append(th_seg, th_seg[-1])
        self.heading = self._th
        self.tangent = np.column_stack([np.cos(self._th), np.sin(self._th)])
        self.curvature = _curvature(th_seg, self.closed, self.spacing)
        normal = np.column_stack([-np.sin(self._th), np.cos(self._th)])
        offset = normal * self.half_width
        self.left_edge = P + np.column_stack([offset, np.zeros(N)])
        self.right_edge = P - np.column_stack([offset, np.zeros(N)])
        self.goal = P[0].copy() if closed else P[-1].copy()

        self._xy = P[:, :2].copy()
        self._tree = cKDTree(self._xy)

    # ----------------------------------------------------------- construction
    @classmethod
    def generate(
        cls,
        terrain: Terrain,
        rng: Union[int, np.random.Generator] = 0,
        closed: bool = True,
        difficulty: float = 0.5,
        width: float = 10.0,
        spacing: float = 1.5,
        max_grade: float = 0.10,
        span: float = 0.6,
        attempts: int = 200,
    ) -> "Track":
        """Generate a random racing track that fits inside ``terrain``.

        Parameters
        ----------
        terrain : the Terrain the track is placed on (used for bounds and the
            base elevation profile).
        rng : int seed or numpy Generator.
        closed : loop circuit if True, point-to-point (A -> B) otherwise.
        difficulty : 0..1. Higher means more control points (more corners),
            a tighter minimum corner radius (35 m -> 15 m) and more lateral
            wiggle for point-to-point tracks.
        width : road width in metres (default 10).
        spacing : centerline sample spacing in metres (1-2 m).
        max_grade : maximum longitudinal grade |dz/ds| of the track.
        span : point-to-point only: distance A->B as a fraction of terrain size.
        attempts : random retries before giving up (RuntimeError).
        """
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        if width <= 0 or not 0.3 <= spacing <= 5.0:
            raise ValueError("width must be > 0 and spacing in [0.3, 5]")
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        size = terrain.size
        hw = width / 2.0
        margin = hw + SHOULDER_WIDTH + EMBANKMENT_WIDTH + 5.0
        half_extent = size / 2.0 - margin
        clear = width + 2.0 * SHOULDER_WIDTH + 2.0   # min centre-to-centre gap
        min_radius = 35.0 - 20.0 * difficulty
        kmax = 1.0 / min_radius

        for _ in range(attempts):
            if closed:
                ctrl = _closed_control(rng, 0.25 * size, difficulty)
                dense = _dense_closed(ctrl)
            else:
                ctrl = _open_control(rng, size, span, difficulty)
                dense = _dense_open(ctrl)

            P, ds = _resample(dense, spacing, closed)
            P = _enforce_curvature(P, closed, ds, kmax, spacing)
            if P is None:
                continue
            if not _geometry_ok(P, closed, ds, half_extent, clear):
                continue
            if not closed and np.linalg.norm(P[-1] - P[0]) < 0.3 * size:
                continue

            z = terrain.height_at(P[:, 0], P[:, 1])
            z = _elevation_profile(z, closed, ds, max_grade)
            centerline = np.column_stack([P, z])
            return cls(centerline, width, ds, closed, min_radius=min_radius)

        raise RuntimeError(
            f"could not generate a valid track in {attempts} attempts "
            f"(closed={closed}, difficulty={difficulty}, width={width})"
        )

    # ---------------------------------------------------------------- queries
    def start_pose(self):
        """Start position (xyz, shape (3,)) and yaw (rad) at sample 0."""
        return self.centerline[0].copy(), float(self.heading[0])

    def point_at(self, s):
        """Centerline point and tangent yaw at arc length ``s``.

        Returns ``(xyz, heading)``; for scalar ``s`` these are (3,) and float,
        otherwise arrays of shape (..., 3) and (...).
        """
        s_arr = np.asarray(s, dtype=float)
        P = self.centerline
        N, ds, L = self.n, self.spacing, self.length
        if self.closed:
            f = np.mod(s_arr, L) / ds
            i0 = np.floor(f).astype(int) % N
            t = f - np.floor(f)
            i1 = (i0 + 1) % N
        else:
            f = np.clip(s_arr, 0.0, L) / ds
            i0 = np.minimum(np.floor(f).astype(int), N - 2)
            t = f - i0
            i1 = i0 + 1
        xyz = P[i0] + t[..., None] * (P[i1] - P[i0])
        heading = self.heading[i0]
        if s_arr.ndim == 0:
            return xyz, float(heading)
        return xyz, heading

    def curvature_at(self, s):
        """Signed curvature (1/m, positive = left turn) at arc length ``s``."""
        s_arr = np.asarray(s, dtype=float)
        N, ds, L = self.n, self.spacing, self.length
        k = self.curvature
        if self.closed:
            f = np.mod(s_arr, L) / ds
            i0 = np.floor(f).astype(int) % N
            t = f - np.floor(f)
            i1 = (i0 + 1) % N
        else:
            f = np.clip(s_arr, 0.0, L) / ds
            i0 = np.minimum(np.floor(f).astype(int), N - 2)
            t = f - i0
            i1 = i0 + 1
        out = k[i0] + t * (k[i1] - k[i0])
        return float(out) if out.ndim == 0 else out

    def project(self, pos, hint: Optional[int] = None) -> Projection:
        """Project a position onto the centerline.

        Parameters
        ----------
        pos : (2,) or (3,) world position; only x and y are used.
        hint : index of the sample found by the previous call (e.g.
            ``previous.index``). Gives an O(1) local search, falling back to a
            global KD-tree search when the car is far from the hint.
        """
        p = np.asarray(pos, dtype=float).reshape(-1)[:2]
        idx = self._nearest_index(p, hint)
        r = self._refine(p[None, :], np.array([idx]))
        return Projection(
            index=int(idx),
            s=float(r["s"][0]),
            progress=float(r["s"][0] / self.length),
            lateral=float(r["lateral"][0]),
            heading=float(r["heading"][0]),
            point=r["point"][0].copy(),
            on_track=bool(r["on_track"][0]),
        )

    def project_many(self, xy):
        """Vectorised global projection of many xy points.

        Returns a dict of arrays (length M): ``s``, ``lateral``, ``heading``,
        ``distance`` (unsigned Euclidean distance to the centerline), ``point``
        (M, 3), ``on_track``, ``index``.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        _, idx = self._tree.query(xy)
        out = self._refine(xy, np.asarray(idx))
        out["index"] = np.asarray(idx)
        return out

    def heading_error(self, pos, yaw: float, hint: Optional[int] = None) -> float:
        """Track heading minus ``yaw`` at ``pos``, wrapped to (-pi, pi].

        Positive means the track direction is to the car's left.
        """
        return float(wrap_angle(self.project(pos, hint).heading - yaw))

    def lookahead(self, pos, yaw: float, distances, hint: Optional[int] = None) -> np.ndarray:
        """Centerline points ahead of the car, in the car's local frame.

        ``distances`` are arc-length offsets (m) measured along the track from
        the car's projection (``s + d``). Points beyond the end of an open
        track repeat the goal point.

        Returns an array of shape (K, 2) with columns ``(forward, left)``.
        """
        pr = self.project(pos, hint)
        d = np.atleast_1d(np.asarray(distances, dtype=float))
        xyz, _ = self.point_at(pr.s + d)
        p = np.asarray(pos, dtype=float).reshape(-1)[:2]
        dx = xyz[:, 0] - p[0]
        dy = xyz[:, 1] - p[1]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    # --------------------------------------------------------------- internals
    def _nearest_index(self, p, hint):
        if hint is not None:
            N = self.n
            offs = np.arange(-LOCAL_WINDOW, LOCAL_WINDOW + 1)
            if self.closed:
                cand = (int(hint) + offs) % N
            else:
                cand = np.clip(int(hint) + offs, 0, N - 1)
            d2 = np.sum((self._xy[cand] - p) ** 2, axis=1)
            b = int(np.argmin(d2))
            at_window_edge = b in (0, len(cand) - 1)
            at_track_end = (not self.closed) and int(cand[b]) in (0, N - 1)
            far = np.sqrt(d2[b]) > TRUST_FACTOR * self.width
            if not ((at_window_edge and not at_track_end) or far):
                return int(cand[b])
        _, idx = self._tree.query(p)
        return int(idx)

    def _refine(self, p, idx):
        """Exact projection onto the two polyline segments adjacent to ``idx``."""
        N, ds, L = self.n, self.spacing, self.length
        xy = self._xy
        Z = self.centerline[:, 2]
        if self.closed:
            prev = (idx - 1) % N
            nxt = (idx + 1) % N
        else:
            prev = np.maximum(idx - 1, 0)
            nxt = np.minimum(idx + 1, N - 1)

        def seg_proj(a0, a1):
            d = xy[a1] - xy[a0]
            L2 = np.sum(d * d, axis=1)
            safe = np.maximum(L2, 1e-18)
            t = np.where(L2 > 1e-18, np.sum((p - xy[a0]) * d, axis=1) / safe, 0.0)
            t = np.clip(t, 0.0, 1.0)
            q = xy[a0] + t[:, None] * d
            return t, q, np.linalg.norm(p - q, axis=1)

        tA, qA, dA = seg_proj(prev, idx)
        tB, qB, dB = seg_proj(idx, nxt)
        use_a = dA <= dB
        a0 = np.where(use_a, prev, idx)
        a1 = np.where(use_a, idx, nxt)
        t = np.where(use_a, tA, tB)
        q = np.where(use_a[:, None], qA, qB)
        dist = np.where(use_a, dA, dB)

        s = (a0 + t) * ds
        if self.closed:
            s = np.mod(s, L)
        th = self._th[a0]
        u = np.column_stack([np.cos(th), np.sin(th)])
        normal = np.column_stack([-u[:, 1], u[:, 0]])
        diff = p - q
        lateral = np.sum(diff * normal, axis=1)
        along = np.sum(diff * u, axis=1)
        z = Z[a0] + t * (Z[a1] - Z[a0])
        on_track = np.abs(lateral) <= self.half_width
        if not self.closed:
            beyond = ((s <= 1e-9) & (along < -1e-9)) | ((s >= L - 1e-9) & (along > 1e-9))
            on_track &= ~beyond
        return {
            "s": s,
            "lateral": lateral,
            "heading": th,
            "distance": dist,
            "point": np.column_stack([q, z]),
            "on_track": on_track,
        }


# ------------------------------------------------------------------ generation
def _closed_control(rng, radius, difficulty):
    n = 5 + int(round(6 * difficulty))
    angles = 2.0 * np.pi * (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n
    spread = 0.3 + 0.4 * difficulty
    radii = radius * (1.0 - spread * rng.uniform(0.0, 1.0, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _open_control(rng, size, span, difficulty):
    D = span * size
    theta = rng.uniform(0.0, 2.0 * np.pi)
    u = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-u[1], u[0]])
    centre = rng.uniform(-0.1, 0.1, 2) * size
    A = centre - 0.5 * D * u
    B = centre + 0.5 * D * u
    n_mid = 2 + int(round(4 * difficulty))
    ts = np.linspace(0.0, 1.0, n_mid + 2)[1:-1]
    amp = D * (0.08 + 0.12 * difficulty)
    offsets = rng.uniform(-amp, amp, n_mid)
    mids = A + ts[:, None] * (B - A) + offsets[:, None] * perp
    return np.vstack([A, mids, B])


def _dense_closed(ctrl):
    C = np.vstack([ctrl, ctrl[:1]])
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    cs = CubicSpline(u, C, bc_type="periodic")
    M = int(max(400, u[-1] / 0.25))
    return cs(np.linspace(0.0, u[-1], M, endpoint=False))


def _dense_open(ctrl):
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))])
    cs = CubicSpline(u, ctrl, bc_type="natural")
    M = int(max(400, u[-1] / 0.25))
    return cs(np.linspace(0.0, u[-1], M))


def _enforce_curvature(P, closed, ds, kmax, spacing):
    """Smooth the path until |curvature| <= kmax everywhere (or give up)."""
    sigma = 1.0
    for _ in range(120):
        if np.max(np.abs(_curvature(_segment_headings(P, closed), closed, ds))) <= kmax:
            return P
        if sigma > len(P) / 8.0:
            return None
        if closed:
            Q = gaussian_filter1d(P, sigma, axis=0, mode="wrap")
        else:
            Q = gaussian_filter1d(P, sigma, axis=0, mode="nearest")
            # keep the start and goal fixed while smoothing
            off0 = P[0] - Q[0]
            offN = P[-1] - Q[-1]
            w = np.linspace(0.0, 1.0, len(Q))[:, None]
            Q = Q + (1.0 - w) * off0 + w * offN
        P, ds = _resample(Q, spacing, closed)
        sigma *= 1.15
    return None


def _geometry_ok(P, closed, ds, half_extent, clear):
    if np.any(np.abs(P) > half_extent):
        return False
    N = len(P)
    tree = cKDTree(P)
    pairs = tree.query_pairs(clear, output_type="ndarray")
    if len(pairs):
        sep = np.abs(pairs[:, 0] - pairs[:, 1])
        if closed:
            sep = np.minimum(sep, N - sep)
        if np.any(sep * ds > 5.0 * clear):
            return False  # two parts of the road cross or nearly touch
    return True


def _elevation_profile(z, closed, ds, max_grade):
    """Smooth terrain-following elevation whose grade stays under max_grade."""
    sigma_m = 30.0
    for _ in range(40):
        sig = sigma_m / ds
        if closed:
            zs = gaussian_filter1d(z, sig, mode="wrap")
            grade = np.max(np.abs(np.diff(np.append(zs, zs[0])))) / ds
        else:
            zs = gaussian_filter1d(z, sig, mode="nearest")
            grade = np.max(np.abs(np.diff(zs))) / ds
        if grade <= max_grade:
            return zs
        sigma_m *= 1.3
    return zs
