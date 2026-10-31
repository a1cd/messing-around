# Scripted (segment-based) tracks

Plan a circuit as an ordered list of distinctive segments, like a rally pace-note sheet
or a track designer's storyboard. The result is an ordinary `drivesim.track.Track`, so
`project`, `lookahead`, `start_pose`, `curvature_at` and the Gym env work unchanged.

```python
from drivesim import TrackScript, build_world_from_script
from drivesim.track_script import EXAMPLES

script = TrackScript.from_dsl("S200 R4L L2 HAIRPIN-L ZIGZAG4 DROP-15/80 SWEEP-R90 CLOSE")
# or the Python builder, equivalent to the same script:
script = (TrackScript().straight(200).right(severity=4, length="long").left(severity=2)
          .hairpin("left").zigzag(n=4).drop(15, 80).sweep("right", 90).close())

terrain, track = build_world_from_script(script, seed=3)   # terrain carved under the road
track = script.build()                                     # track only, no terrain (flat elevation)
```

Conventions are those of `drivesim/README_world.md`: x east, y north, z up, yaw CCW.

## Segment vocabulary

| DSL token | Python | Meaning |
|---|---|---|
| `S200` | `straight(200)` | straight, 200 m |
| `HS400` | `long_straight(400)` | high-speed straight (flag only, same geometry) |
| `R4L`, `L2`, `L3S>!` | `right(4, "long")`, `left(2)` | pace-note corner (see below) |
| `HAIRPIN-L/9` | `hairpin("left", radius=9)` | 180 degree turn. Radius defaults to 9 m |
| `SWITCHBACK-R/9/30` | `switchback("right", 9, 30)` | hairpin, straight leg, hairpin the other way |
| `ZIGZAG4-L3S` | `zigzag(4, "left", 3, "short")` | 4 alternating corners (chicane, esses) |
| `ESSES3-L` | `esses(3, "left")` | zigzag of severity 5, short |
| (none) | `chicane("left")` | 2 alternating corners (an S-bend). Python only; DSL: `ZIGZAG2-L3S` |
| `TURN-R45/30` | `turn(45, "right", radius=30)` | explicit angle in degrees and radius (default 25 m) |
| `SWEEP-R90/80` | `sweep("right", 90, radius=80)` | long fast bend (default radius 80 m) |
| `DROP-15/80` | `drop(15, 80)` | descend 15 m over an 80 m straight |
| `CLIMB-20/120` | `climb(20, 120)` | climb 20 m over 120 m |
| `CREST-4/80` | `crest(4, 80)` | rise 4 m then fall back over 80 m (hill crest) |
| `DIP-4/80` | `dip(4, 80)` | fall 4 m then rise back (compression) |
| `GRADE+12/60` | `steep(0.12, 60)` | constant 12 % climb over 60 m (`-` for descent) |
| `CLOSE` | `close()` | the route is a loop |

### Pace-note corners

`<dir><severity>[<length>][<modifiers>]`

- direction `L` or `R`.
- severity `1`..`6` sets the radius: 1 = 9 m (hairpin-tight), 2 = 16 m, 3 = 28 m, 4 = 45 m,
  5 = 75 m, 6 = 120 m (flat-out kink).
- length `S` / `N` / `L` (short / normal / long; default normal) sets the angle:
  40 / 75 / 120 degrees.
- modifiers, in any order:
  - `>` tightens: exit radius = 0.6 x entry radius, so the corner gets tighter.
  - `<` opens: exit radius = 1.5 x entry radius.
  - `!` caution: an advisory flag. It changes no geometry and only appears in `track.segments`.
  - `>` and `<` cannot both be given.

Examples: `R4L` is right, severity 4, long. `L2` is left, severity 2, normal. `R3S>` is right,
severity 3, short, tightening.

### Defaults and rules

- Keywords are case-insensitive, and tokens are separated by whitespace.
- Numbers are metres unless stated (angles in degrees, grades as percent in the DSL).
- Corner radius must be at least 8 m (`MIN_RADIUS`). Smaller values raise ValueError.
- A DSL token that cannot be parsed raises ValueError naming the token and listing the grammar.

## Geometry

- Each corner is entered and left through linear curvature ramps (clothoid transitions). Curvature
  is continuous, and the heading change is exactly the stated angle. The transition length is
  `T = min(max(0.3 * arc, 6), 40, 0.4 * arc)`. A corner's arc length is `R * angle + T`.
- Elevation changes (`DROP`, `CLIMB`, `CREST`, `DIP`, `GRADE`) use a grade profile that eases in and
  out, so there are no vertical jumps. Their net change is exact (drop 15 m gives 15 m).
- Unspecified elevation follows the terrain: grade = the terrain's gradient along the route, smoothed
  over 30 m. Explicit elevation changes add to the running height. With no terrain, unspecified
  elevation is flat.
- `max_grade` (default 0.25) limits |dz/ds|. An explicit change whose peak grade exceeds it raises
  ValueError naming the segment. The terrain-following grade is clamped to it.

## Closing and completing a route

Scripts do not have to be complete. With the default `strict=False`:

1. **Closed route** (`closed=True`, or `CLOSE` in the script). If the end is within
   `close_gap_max` (default 50 m) of the start, the final stretch (30 to 120 m) is re-shaped by a
   least-squares blend of curvature so the end meets the start position and heading (within
   `close_tolerance`, 0.5 m and 1 degree). If the blend cannot do it, or the gap is larger, the
   builder appends **three corners and two straights**, whose angles, radii and lengths are solved
   by least squares so the route returns to the start pose and heading. Candidates must keep corners
   at least `min_radius` and must not cross the route. The first valid candidate is used.
2. **Open route** (`closed=False`). The route is point-to-point. With `goal=(x, y)`, metres from the
   start in the `heading0` frame, it is extended to the goal in the same way (heading free).

The completion is reported in `track.closure['completed']` and the appended segments in
`track.completion_segments`. `track.script` is always the script as written.

With `strict=True` nothing is added. The builder raises ValueError with the gap and heading error.

Limits of completion: the template is three corners and two straights. It handles simple cases
(for example `S100 L3 S100 CLOSE`, about 200 m out) but can fail for larger gaps, or where the
return would have to cross the route. Such routes raise an error saying the route could not be
completed. For example `S200 R4L L2 HAIRPIN-L ZIGZAG4
DROP-15/80 SWEEP-R90 CLOSE` ends about 188 m and 45 degrees from its start and cannot be completed.

Point-symmetric routes close exactly. Two identical halves that each turn through 180 degrees
(`EXAMPLES` use this) cancel their displacements, so no blend is needed.

## Validation

- **Self-intersection** (`self_intersection='error'` by default): two route points closer than the
  road clearance (`width + 4 m`) but more than `3 x clearance` apart along the route raise ValueError
  naming the point. Use `'warn'` to keep such a route, or `'ignore'`.
- **Terrain bounds**: the route is centred on the origin. `build_world_from_script` chooses the
  terrain size: 600 m, or the smallest multiple of 100 m that fits. A `size` that is too small
  raises ValueError. `build(terrain)` raises if the route lies outside the terrain.
- **Minimum radius** and **max grade** raise ValueError with the offending segment.
- `track.segments` lists each segment's `s0`, `s1`, `heading_change`, `elevation_change`, `text`
  and `caution`. `track.closure` reports the blend or completion.

## Public API

- `TrackScript(segments=(), closed=False)`: chainable builder methods (above). `from_dsl(text)`,
  `to_dsl()` (canonical, `from_dsl(s.to_dsl()) == s`), `copy()`, `close()`, `build(terrain=None,
  *, closed=None, width=10, spacing=1.5, max_grade=0.25, min_radius=8, close_tolerance=0.5,
  close_gap_max=50, close_blend=None, self_intersection='error', heading0=0.0, strict=False,
  goal=None) -> Track`.
- `build_world_from_script(script, seed=0, closed=None, size=None, resolution=1.0, width=10,
  amplitude=10, spacing=1.5, patches=True, ...same options...) -> (Terrain, Track)`. The track is
  carved into the terrain with `carve_track`, as `generate_world` does.
- `parse_track_script(text)`, `Segment`, `EXAMPLES` (DSL strings) and `example_script(name)`.
- Named examples: `technical_rally` (closed, about 2160 m, pace-note corners, esses, a descent,
  a crest and a switchback), `high_speed_oval` (closed, about 2420 m, long straights and sweepers),
  `switchback_hillclimb` (point-to-point, about 780 m, a climb with stacked switchbacks).

## Preview

    python scripts/preview_script_track.py OUT_DIR [--seed 0] [--example NAME ...] [--dsl "S100 L3 S100 CLOSE"]

## Tests

    python -m pytest tests/test_track_script.py

## Limitations

- Corners have constant radius, or a linear radius change for `tightens` / `opens`. There are no
  general curvature profiles.
- Caution flags are advisory metadata. Nothing in the simulator reads them.
- The completion template is limited (see above). Large far-off gaps need more segments.
- Closure is a planar blend and a grade bump. Heading and position close exactly, but a route
  far from its start can need a steep closing grade, which then fails `max_grade`.
- Elevation follows terrain only where the script does not specify it. Explicit changes are
  not reshaped to match the terrain.
- Terrain is a square heightfield (see `README_world.md`). The route floats where the terrain
  falls away steeply, and carving then flattens the terrain under the road.
