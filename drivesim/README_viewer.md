# drivesim 3D viewer

`drivesim/viewer.py` is a bare-bones software renderer for a generated world and a `Vehicle`.
It uses numpy for projection, culling and clipping, and pygame polygons for drawing (painter's
algorithm). No GPU or OpenGL is needed. It runs on macOS and Linux.

## Quick start

```python
from drivesim.world import generate_world
from drivesim.vehicle import Vehicle, Controls
from drivesim.viewer import Viewer, save_frames

terrain, track = generate_world(seed=3, closed=True, difficulty=0.5)
car = Vehicle()
pos, yaw = track.start_pose()
car.reset(pos[:2], yaw, terrain)

viewer = Viewer(terrain, track, width=800, height=500, headless=True)   # headless=False opens a window
for _ in range(60):
    car.step(Controls(throttle=0.5), 1 / 30, terrain)
    frame = viewer.render(car, hud={"progress": 0.1, "message": "go"}, camera="chase")  # (500, 800, 3) uint8
save_frames([frame], "/tmp/frames")
viewer.close()
```

`headless=True` works with `SDL_VIDEODRIVER=dummy` and no display. The viewer sets that variable
itself if it is not already set.

## Public API

```python
class Viewer:
    def __init__(self, terrain, track, width=800, height=500, headless=False, view_radius=120.0)
    def render(self, vehicle, hud: dict | None = None, camera: str | None = "chase") -> np.ndarray
    def set_camera(self, name: str) -> None        # 'chase' | 'hood' | 'top'; ValueError otherwise
    def next_camera(self) -> str                   # cycles chase -> hood -> top -> chase
    def close(self) -> None
    closed: bool                                    # True after the window closes

def save_frames(frames, out_dir, fps: int = 15) -> list[Path]
```

* `render` always returns an `(H, W, 3)` uint8 RGB array. When `headless=False` it also draws the
  frame in a window and processes events. The **C** key cycles the camera, and closing the window
  sets `closed`.
* `camera` names the camera for that call. Pass `camera=None` to use the one chosen with
  `set_camera` or **C**.
* `hud` keys: `speed_kmh`, `gear`, `rpm`, `throttle`, `brake`, `steer` ([-1, 1]), `progress`
  (fraction), `reward`, `message`. Any other numeric key is shown as `key value`. Speed, gear and
  rpm come from the vehicle when absent. Steer falls back to the front road-wheel angle.
* `save_frames` writes `frame_00000.png`, and `episode.gif` if `imageio` is installed.

## What is drawn

* Terrain as flat-shaded quads: colour by surface type (asphalt, grass, gravel, mud, ice), Lambert
  shade from the surface normal, and a faint checker so speed is visible. Distant terrain fades to
  the horizon colour.
* Level of detail: 6 m cells out to `view_radius`, 3 m cells in the middle ring, and 1 m cells in a
  core of about `0.12 * view_radius`. The rings are nested, so there are no cracks.
* Track: white road edges and a dashed yellow centreline, drawn over the terrain.
* Car: a box body with a cabin, and four wheels (rolling prisms with a spin spoke). Wheels visibly
  steer and spin. Faces are back-face culled and depth sorted.
* Cameras:
  * `chase`: 7.5 m behind and 2.6 m up, smoothed in position and yaw. Snaps when the car jumps.
  * `hood`: from the bonnet, with a simple bonnet silhouette and the car body hidden.
  * `top`: straight down, heading up, covering about 0.6 * `view_radius` in half the frame height.
* Minimap (top right) with the whole track, the car and its heading. Goal marked for point-to-point
  tracks.
* HUD: speed, gear, rpm bar with redline zone, throttle and brake bars, steering indicator, camera
  name, and the extra hud items.

## Performance

Render-only throughput at 800x500 on the dev container (single core, Python 3.13, `headless=True`),
after warm-up on a driven car:

| camera | fps |
|---|---|
| chase | ~48 |
| hood | ~52 |
| top | ~36 |

These figures come from `tests/test_viewer.py::test_fps_measurement`, which prints them with `-s`.
The test only asserts a loose floor of 15 fps so it stays stable on slow machines.

## Limitations

* Terrain normals are cached when the viewer is built. Heights and surface ids are read live.
* No depth buffer. Painter's sorting by quad centre is exact for most heightfield views but can
  misorder quads at steep ridges. Thin seams between polygons are possible.
* Hood view hides the car body and wheels and draws a simple bonnet shape.
* The car is a box. It has no lights, shading beyond Lambert, or suspension travel.
* Near-plane handling drops quads and polygons that cross the near plane rather than clipping them.
  This is invisible in the chase and top views, but a quad right under a low camera can disappear.
* Track edges are drawn as lines on top of the terrain. They are not occluded by hills.

## Tests

    python -m pytest tests/test_viewer.py -s

Covers shape and dtype for all cameras, frame change while driving, HUD keys and fallbacks, camera
selection, camera extremes (car flipped, on its side, far off the terrain, at the terrain edge),
PNG output, and fps.
