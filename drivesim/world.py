"""Convenience constructor for a consistent (terrain, track) pair.

Public API
----------
``generate_world(seed, closed=True, difficulty=0.5, ...) -> (Terrain, Track)``
``carve_track(terrain, track)``  flatten/grade the terrain under a track and
                                 assign surface types (called by generate_world).
"""
from __future__ import annotations

import numpy as np

from .terrain import (
    SURFACE_ASPHALT,
    SURFACE_GRASS,
    SURFACE_GRAVEL,
    Terrain,
)
from .track import EMBANKMENT_WIDTH, SHOULDER_WIDTH, Track


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def carve_track(terrain: Terrain, track: Track,
                shoulder: float = SHOULDER_WIDTH,
                embankment: float = EMBANKMENT_WIDTH) -> None:
    """Modify ``terrain`` in place so that it matches ``track``.

    * Within ``half_width + shoulder`` of the centerline the terrain is set to
      the track's elevation profile (flat across the road, graded along it).
    * Over the next ``embankment`` metres the terrain blends smoothly
      (smoothstep) from the graded height back to the natural terrain.
    * Surface ids: asphalt on the road, gravel on the shoulder, grass on the
      embankment; natural mud/ice patches are kept beyond that.
    """
    X, Y = terrain.grid_xy()
    xy = np.column_stack([X.ravel(), Y.ravel()])
    pr = track.project_many(xy)
    dist = pr["distance"]
    target = track.point_at(pr["s"])[0][:, 2]

    flat_r = track.half_width + shoulder
    w = 1.0 - _smoothstep((dist - flat_r) / embankment)
    new_h = w * target + (1.0 - w) * terrain.heights.ravel()

    ids = terrain.surface_ids.ravel().copy()
    ids[dist <= track.half_width] = SURFACE_ASPHALT
    ids[(dist > track.half_width) & (dist <= flat_r)] = SURFACE_GRAVEL
    ids[(dist > flat_r) & (dist <= flat_r + embankment)] = SURFACE_GRASS

    n = terrain.n
    terrain.heights = new_h.reshape(n, n)
    terrain.surface_ids = ids.reshape(n, n).astype(np.uint8)


def generate_world(seed: int, closed: bool = True, difficulty: float = 0.5,
                   size: float = 600.0, resolution: float = 1.0,
                   width: float = 10.0, amplitude: float = 10.0,
                   spacing: float = 1.5, patches: bool = True):
    """Generate a terrain and a track on it, fully consistent.

    Parameters
    ----------
    seed : non-negative int. Same arguments -> identical world.
    closed : True for a loop circuit, False for point-to-point (A -> B).
    difficulty : 0..1, more and tighter corners as it grows.
    size : terrain side length (m). resolution : terrain node spacing (m).
    width : road width (m). amplitude : terrain relief (m).
    spacing : track centerline sample spacing (m).
    patches : scatter mud/ice patches on the natural terrain.

    Returns ``(terrain, track)``. The terrain has already been carved under
    the track, so ``terrain.height_at`` matches ``track.centerline`` z along
    the road.
    """
    seed = int(seed)
    terrain = Terrain.generate(seed, size=size, resolution=resolution,
                               amplitude=amplitude, patches=patches)
    track = Track.generate(terrain, rng=np.random.default_rng([seed, 1]),
                           closed=closed, difficulty=difficulty,
                           width=width, spacing=spacing)
    carve_track(terrain, track)
    return terrain, track
