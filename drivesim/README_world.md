# drivesim world generation

Procedural terrain and racing tracks for the driving simulator.

```python
from drivesim.world import generate_world
from drivesim.track import Track

terrain, track = generate_world(seed=3, closed=True, difficulty=0.5)  # or closed=False for A -> B
pos, yaw = track.start_pose()
pr = track.project(pos, hint=None)          # pr.s, pr.lateral, pr.heading, pr.on_track, pr.index
la = track.lookahead(pos, yaw, [5, 10, 20]) # (3, 2) array of (forward, left) in the car frame
```

Conventions: x east, y north, z up (metres). Yaw is radians, 0 along +x, CCW positive.
`lateral` is positive to the left of the track. The terrain is centred on the origin.

## Public API

- `generate_world(seed, closed=True, difficulty=0.5, size=600.0, resolution=1.0, width=10.0, amplitude=10.0, spacing=1.5, patches=True) -> (Terrain, Track)`
  Deterministic for a given seed and arguments. The terrain is already carved under the track.
- `Terrain.generate(seed, size, resolution, amplitude, octaves, base_wavelength, persistence, patches, mud_fraction, ice_fraction)`
- `Terrain.height_at(x, y)`: bilinear height, vectorised, clamped at the border.
- `Terrain.normal_at(x, y)`: unit normals, shape (..., 3).
- `Terrain.surface_type_at(x, y)`: `SURFACE_*` id. `Terrain.surface_at(x, y)`: friction multiplier.
  Friction: asphalt 1.0, grass 0.6, gravel/dirt 0.5, mud 0.4, ice 0.2 (`SURFACE_FRICTION`).
- `Track.generate(terrain, rng, closed=True, difficulty=0.5, width=10.0, spacing=1.5, max_grade=0.10, span=0.6)`
- `track.project(pos, hint=None) -> Projection(index, s, progress, lateral, heading, point, on_track)`
  Pass the previous `index` as `hint` for an O(1) local search. The search falls back to a global
  KD-tree query when the car is far from the hint.
- `track.heading_error(pos, yaw, hint=None)`: track heading minus yaw, wrapped to (-pi, pi].
- `track.lookahead(pos, yaw, distances, hint=None) -> (K, 2)` of `(forward, left)`. Distances are arc-length offsets
  from the car's projection. Beyond the goal of an open track, the goal point repeats.
- `track.point_at(s) -> (xyz, heading)`, `track.curvature_at(s)`, `track.start_pose() -> (xyz, yaw)`.
- `track.project_many(xy)`: vectorised global projection, returns a dict of arrays.
- Track attributes: `centerline` (N,3), `left_edge`, `right_edge`, `width`, `half_width`, `spacing`,
  `length`, `closed`, `s`, `heading`, `tangent`, `curvature`, `min_radius`, `goal`.

## Design notes

- Track curvature is bounded by `1 / min_radius`, where `min_radius` goes from 35 m (difficulty 0) to 15 m (difficulty 1).
- Track grade is limited to `max_grade` by smoothing the terrain-following elevation profile.
- Terrain under the road is flattened to the track elevation within `width/2 + 2 m` (the shoulder).
  Over the next 25 m it blends back to natural terrain with a smoothstep. Shoulders are gravel, the blend zone is grass.
- The centerline is a polyline with uniform spacing (about 1.5 m). Projection is exact onto the segments.

## Preview

    python scripts/preview_world.py /path/to/out.png --seed 3 [--open] [--difficulty 0.5]

## Tests

    python -m pytest tests/test_world.py

## Limitations

- Terrain is a square heightfield with no overhangs, banking or tunnels. Track z is taken from the terrain, so the track
  can float above or dip below the natural ground where the embankment is steep.
- Surface types are a single id per grid cell (nearest-cell lookup). There is no per-vertex blending.
- Lateral and heading values are exact for the polyline. They differ from a smooth-spline ideal by under 0.2% of the
  sample spacing, which is negligible.
- Point-to-point tracks are placed roughly across the terrain centre (`span` of the terrain size). The generator does not
  check whether a route is 'interesting' beyond curvature and self-clearance.
