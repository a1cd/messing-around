"""Driving simulator package for reinforcement learning.

World generation lives in :mod:`drivesim.terrain`, :mod:`drivesim.track` and
:mod:`drivesim.world`; see ``drivesim/README_world.md``.
"""
from .terrain import Terrain
from .track import Track, Projection
from .world import generate_world, carve_track

__all__ = ["Terrain", "Track", "Projection", "generate_world", "carve_track"]
