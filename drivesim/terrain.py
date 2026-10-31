"""Procedural heightfield terrain.

Coordinate conventions
----------------------
* World frame: x east, y north, z up (metres). The terrain is centred on the
  origin, so it spans ``[-size/2, +size/2]`` in both x and y.
* ``heights`` is a 2D array indexed ``heights[iy, ix]``; ``ix`` runs along +x
  and ``iy`` along +y (so ``imshow(heights, origin="lower")`` is top-down
  correct). Node ``(ix, iy)`` sits at ``(origin + ix*res, origin + iy*res)``.
* Queries outside the terrain are clamped to the border.

Public API
----------
``Terrain.generate(seed, ...)``        -> deterministic procedural terrain
``Terrain.height_at(x, y)``            -> bilinear height (z), vectorised
``Terrain.normal_at(x, y)``            -> unit surface normals, shape (..., 3)
``Terrain.surface_type_at(x, y)``      -> integer surface ids (SURFACE_*)
``Terrain.surface_at(x, y)``           -> friction multiplier per point
"""
from __future__ import annotations

import numpy as np

# Surface ids stored in ``Terrain.surface_ids``. Friction multipliers are
# relative to dry asphalt (1.0) and are looked up via SURFACE_FRICTION.
SURFACE_ASPHALT = 0
SURFACE_GRASS = 1
SURFACE_GRAVEL = 2  # dirt / gravel shoulder
SURFACE_MUD = 3
SURFACE_ICE = 4

SURFACE_NAMES = ("asphalt", "grass", "gravel", "mud", "ice")
SURFACE_FRICTION = np.array([1.0, 0.6, 0.5, 0.4, 0.2], dtype=float)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(lattice, u, v):
    """Smooth value noise on a lattice.

    ``lattice`` has shape (nv, nu); ``u`` (nx,) and ``v`` (ny,) are coordinates
    in lattice units. Returns an array of shape (ny, nx).
    """
    iu = np.floor(u).astype(int)
    iv = np.floor(v).astype(int)
    su = _smoothstep(u - iu)
    sv = _smoothstep(v - iv)
    a = lattice[np.ix_(iv, iu)]
    b = lattice[np.ix_(iv, iu + 1)]
    c = lattice[np.ix_(iv + 1, iu)]
    d = lattice[np.ix_(iv + 1, iu + 1)]
    top = a + (b - a) * su[None, :]
    bot = c + (d - c) * su[None, :]
    return top + (bot - top) * sv[:, None]


def fbm_noise(rng, xs, ys, octaves, base_wavelength, persistence, lacunarity=2.0):
    """Fractal (fBm) value noise evaluated on the grid ``xs`` x ``ys``.

    ``xs``/``ys`` are 1D coordinates in metres measured from the terrain corner.
    Returns an array (len(ys), len(xs)) with values roughly in [-1, 1].
    """
    total = np.zeros((len(ys), len(xs)))
    amp, norm, wl = 1.0, 0.0, float(base_wavelength)
    x_max, y_max = float(np.max(xs)), float(np.max(ys))
    for _ in range(octaves):
        cells_u = int(np.ceil(x_max / wl)) + 2
        cells_v = int(np.ceil(y_max / wl)) + 2
        lattice = rng.uniform(-1.0, 1.0, size=(cells_v, cells_u))
        total += amp * _value_noise(lattice, xs / wl, ys / wl)
        norm += amp
        amp *= persistence
        wl /= lacunarity
    return total / norm


class Terrain:
    """Square heightfield with per-cell surface types.

    Prefer :meth:`Terrain.generate` to build one procedurally.

    Parameters
    ----------
    heights : (n, n) array of node heights in metres.
    size : side length in metres (nodes span exactly ``size``).
    resolution : node spacing in metres (informational; derived from size/(n-1)).
    seed : seed used to generate (informational).
    surface_ids : (n, n) uint8 array of ``SURFACE_*`` ids. Default: grass.
    """

    def __init__(self, heights, size, resolution=None, seed=None, surface_ids=None):
        heights = np.asarray(heights, dtype=float)
        if heights.ndim != 2 or heights.shape[0] != heights.shape[1] or heights.shape[0] < 2:
            raise ValueError("heights must be a square 2D array with at least 2 rows")
        n = heights.shape[0]
        self.heights = heights
        self.size = float(size)
        self.origin = -self.size / 2.0
        self.resolution = self.size / (n - 1)
        if resolution is not None and abs(resolution - self.resolution) > 1e-6 * self.size:
            raise ValueError(
                f"resolution {resolution} inconsistent with size {size} and {n} nodes"
            )
        self.seed = seed
        if surface_ids is None:
            surface_ids = np.full((n, n), SURFACE_GRASS, dtype=np.uint8)
        self.surface_ids = np.asarray(surface_ids, dtype=np.uint8)
        if self.surface_ids.shape != heights.shape:
            raise ValueError("surface_ids must have the same shape as heights")

    @property
    def n(self) -> int:
        """Number of nodes along each axis."""
        return self.heights.shape[0]

    @property
    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the terrain in metres."""
        h = self.size / 2.0
        return (-h, h, -h, h)

    def grid_xy(self):
        """Node coordinates ``(X, Y)``, each of shape (n, n), indexed like heights."""
        xs = self.origin + self.resolution * np.arange(self.n)
        return np.meshgrid(xs, xs)

    # ------------------------------------------------------------------ queries
    def height_at(self, x, y):
        """Bilinearly interpolated terrain height (z) at (x, y).

        Accepts scalars or broadcastable arrays. Returns a float for scalar
        input, otherwise an array of the broadcast shape.
        """
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        n = self.n
        fx = np.clip((x - self.origin) / self.resolution, 0.0, n - 1)
        fy = np.clip((y - self.origin) / self.resolution, 0.0, n - 1)
        ix = np.minimum(np.floor(fx).astype(int), n - 2)
        iy = np.minimum(np.floor(fy).astype(int), n - 2)
        tx = fx - ix
        ty = fy - iy
        h = self.heights
        h00 = h[iy, ix]
        h01 = h[iy, ix + 1]
        h10 = h[iy + 1, ix]
        h11 = h[iy + 1, ix + 1]
        top = h00 + (h01 - h00) * tx
        bot = h10 + (h11 - h10) * tx
        out = top + (bot - top) * ty
        return float(out) if out.ndim == 0 else out

    def normal_at(self, x, y):
        """Unit surface normals at (x, y), shape (..., 3). Points up (+z)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        e = self.resolution
        dzdx = (self.height_at(x + e, y) - self.height_at(x - e, y)) / (2 * e)
        dzdy = (self.height_at(x, y + e) - self.height_at(x, y - e)) / (2 * e)
        normal = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        return normal

    def surface_type_at(self, x, y):
        """Nearest-cell surface id (``SURFACE_*``) at (x, y)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ix = np.clip(np.rint((x - self.origin) / self.resolution).astype(int), 0, self.n - 1)
        iy = np.clip(np.rint((y - self.origin) / self.resolution).astype(int), 0, self.n - 1)
        out = self.surface_ids[iy, ix]
        return int(out) if out.ndim == 0 else out

    def surface_at(self, x, y):
        """Friction multiplier (1.0 = dry asphalt) at (x, y)."""
        ids = self.surface_type_at(x, y)
        out = SURFACE_FRICTION[ids]
        return float(out) if np.ndim(out) == 0 else out

    # -------------------------------------------------------------- generation
    @classmethod
    def generate(
        cls,
        seed: int,
        size: float = 600.0,
        resolution: float = 1.0,
        amplitude: float = 10.0,
        octaves: int = 4,
        base_wavelength: float = 300.0,
        persistence: float = 0.4,
        patches: bool = True,
        mud_fraction: float = 0.06,
        ice_fraction: float = 0.02,
    ) -> "Terrain":
        """Generate a deterministic rolling-hill terrain.

        Parameters
        ----------
        seed : non-negative int. Same seed + parameters -> identical terrain.
        size : side length in metres.
        resolution : approximate node spacing in metres (~1 m default).
        amplitude : max absolute height offset from the mean, in metres.
        octaves, base_wavelength, persistence : fBm shape. The largest hills
            have wavelength ``base_wavelength``.
        patches : if True, scatter mud and ice patches over the grass.
        mud_fraction, ice_fraction : approximate area fractions of mud / ice.
        """
        if seed < 0:
            raise ValueError("seed must be non-negative")
        n = int(round(size / resolution)) + 1
        xs = np.linspace(0.0, size, n)
        rng = np.random.default_rng(seed)
        field = fbm_noise(rng, xs, xs, octaves, base_wavelength, persistence)
        field = field - field.mean()
        peak = np.max(np.abs(field))
        heights = amplitude * field / max(peak, 1e-12)

        ids = np.full((n, n), SURFACE_GRASS, dtype=np.uint8)
        if patches:
            pnoise = fbm_noise(np.random.default_rng([seed, 2]), xs, xs, 3, 80.0, 0.5)
            if mud_fraction > 0:
                ids[pnoise >= np.quantile(pnoise, 1.0 - mud_fraction)] = SURFACE_MUD
            if ice_fraction > 0:
                ids[pnoise <= np.quantile(pnoise, ice_fraction)] = SURFACE_ICE
        return cls(heights, size, resolution=size / (n - 1), seed=seed, surface_ids=ids)
