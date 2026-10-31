"""Bare-bones software 3D viewer for the drivesim environment.

Rendering is done with numpy (perspective projection, vectorised culling and
clipping) and pygame polygons (painter's algorithm). No GPU or OpenGL is used.

Public API
----------
``Viewer(terrain, track, width=800, height=500, headless=False, view_radius=120.0)``
``viewer.render(vehicle, hud=None, camera='chase') -> (H, W, 3) uint8``
    Always returns the frame. Also shows a window when ``headless`` is False.
    ``camera=None`` uses the camera chosen with :meth:`set_camera`.
``viewer.set_camera(name)``  name in ``CAMERAS`` = ('chase', 'hood', 'top')
``viewer.next_camera()``     cycle to the next camera, returns its name
``viewer.close()``
``save_frames(frames, out_dir, fps=15) -> list[Path]``
    Writes ``frame_00000.png`` ... and ``episode.gif`` when imageio is installed.

Frames and conventions follow :mod:`drivesim.terrain` (x east, y north, z up)
and :mod:`drivesim.vehicle` (body X forward, Y left, Z up, quaternion w,x,y,z).

Level of detail: the terrain is drawn as three nested square rings of quads
(1 m cells in a core of about 0.12 x ``view_radius``, 3 m cells in the middle
ring, 6 m cells out to ``view_radius``).
Fog fades distant terrain to the horizon colour, so the view radius is not
obvious.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .terrain import SURFACE_ASPHALT, SURFACE_GRASS, SURFACE_GRAVEL, SURFACE_ICE, SURFACE_MUD  # noqa: E402

CAMERAS = ("chase", "hood", "top")

NEAR = 0.5                  # near clipping distance, camera space [m]
CHASE_FOV_DEG = 60.0
HOOD_FOV_DEG = 70.0
TOP_HEIGHT = 130.0          # top camera height above the car [m]
TOP_SPAN_FRAC = 0.6         # top view shows this fraction of view_radius in half-height
FOLLOW_ALPHA = 0.45         # chase camera smoothing per render call
FOLLOW_SNAP = 30.0          # jump distance [m] above which the chase camera snaps
MINIMAP_SIZE = 150
REDLINE_RPM = 7000.0
STEER_FULL_RAD = 0.6        # road-wheel angle shown as full deflection (fallback)

SURFACE_RGB = np.zeros((5, 3))
SURFACE_RGB[SURFACE_ASPHALT] = (58, 60, 66)
SURFACE_RGB[SURFACE_GRASS] = (86, 142, 62)
SURFACE_RGB[SURFACE_GRAVEL] = (170, 146, 104)
SURFACE_RGB[SURFACE_MUD] = (112, 82, 54)
SURFACE_RGB[SURFACE_ICE] = (198, 228, 242)

SKY_TOP = np.array([64, 114, 196], float)
SKY_HORIZON = np.array([196, 218, 236], float)
FOG_COLOR = SKY_HORIZON
LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

CAR_BODY = (196, 40, 40)
CAR_CABIN = (120, 22, 24)
TIRE_SIDE = (34, 34, 38)
TIRE_CAP = (150, 150, 156)
SPOKE = (225, 225, 230)
EDGE_COLOR = (245, 245, 245)
CENTRE_COLOR = (250, 210, 60)

# Car geometry in the body frame, relative to the centre of gravity [m].
# Each box: (min corner, max corner, colour).
CAR_BOXES = (
    ((-2.2, -0.9, -0.2), (2.0, 0.9, 0.7), CAR_BODY),
    ((-1.1, -0.8, 0.7), (0.8, 0.8, 1.25), CAR_CABIN),
)
Z_UP = np.array([0.0, 0.0, 1.0])
WHEEL_SIDES = 12
WHEEL_WIDTH = 0.24
WHEEL_RADIUS_DEFAULT = 0.33


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _quat_axes(q):
    """Body X, Y, Z unit vectors in world coordinates from a (w, x, y, z) quaternion."""
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    X = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    Y = np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)])
    Z = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    return X, Y, Z


class _Camera:
    """Pinhole camera. ``M`` maps world offsets to (right, up, forward) components."""

    def __init__(self, pos, forward, up, focal, cx, cy):
        f = _unit(np.asarray(forward, float))
        u = np.asarray(up, float)
        u = _unit(u - f * np.dot(u, f))
        r = np.cross(f, u)
        self.pos = np.asarray(pos, float)
        self.f = f
        self.u = u
        self.r = r
        self.M = np.stack([r, u, f])
        self.focal = float(focal)
        self.cx = float(cx)
        self.cy = float(cy)

    def to_cam(self, P):
        return (P - self.pos) @ self.M.T

    def project(self, C):
        """Camera-space (..., 3) -> screen (..., 2). Caller must ensure C[..., 2] > 0."""
        z = C[..., 2]
        sx = self.cx + self.focal * C[..., 0] / z
        sy = self.cy - self.focal * C[..., 1] / z
        return sx, sy


class Viewer:
    """Software perspective renderer for a terrain + track + vehicle.

    Parameters
    ----------
    terrain : drivesim.terrain.Terrain (node heights, surface ids, resolution).
        Heights and surface ids are read live; surface normals are cached at construction.
    track : drivesim.track.Track, used for the road edges and the minimap.
    width, height : frame size in pixels.
    headless : True renders off-screen only (works with SDL_VIDEODRIVER=dummy).
    view_radius : terrain is drawn within this distance (m) of the car.
    """

    def __init__(self, terrain, track, width=800, height=500, headless=False, view_radius=120.0):
        self.terrain = terrain
        self.track = track
        self.width = int(width)
        self.height = int(height)
        self.headless = bool(headless)
        self.view_radius = float(view_radius)
        self.camera = "chase"
        self._follow_pos = None      # smoothed car position for the chase/top cameras
        self._follow_yaw = 0.0       # smoothed car heading for the chase/top cameras
        self._window = None
        self._closed = False

        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.font.init()
        if not self.headless:
            pygame.display.init()
            self._window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("drivesim viewer")
        self._surf = pygame.Surface((self.width, self.height))
        self._font = pygame.font.Font(None, 22)
        self._font_small = pygame.font.Font(None, 18)

        self._normals = self._node_normals()
        self._sky = self._make_sky()
        self._minimap = self._make_minimap()
        self._hud_panel = pygame.Surface((236, 96), pygame.SRCALPHA)
        self._hud_panel.fill((10, 12, 16, 150))

    # ------------------------------------------------------------ public API
    def set_camera(self, name: str) -> None:
        if name not in CAMERAS:
            raise ValueError(f"unknown camera {name!r}; expected one of {CAMERAS}")
        self.camera = name

    def next_camera(self) -> str:
        i = CAMERAS.index(self.camera)
        self.camera = CAMERAS[(i + 1) % len(CAMERAS)]
        return self.camera

    def render(self, vehicle, hud: dict | None = None, camera: str | None = "chase"):
        """Render one frame and return it as an (H, W, 3) uint8 RGB array."""
        name = self.camera if camera is None else camera
        if name not in CAMERAS:
            raise ValueError(f"unknown camera {name!r}; expected one of {CAMERAS}")
        self._pump_events()

        car_pos = np.asarray(vehicle.position, float)
        q = np.asarray(vehicle.quaternion, float)
        axes = _quat_axes(q)
        cam = self._make_camera(name, car_pos, axes, vehicle)
        car_xy = self._follow_pos[:2] if name in ("chase", "top") else car_pos[:2]

        surf = self._surf
        surf.blit(self._sky, (0, 0))
        self._draw_terrain(surf, cam, car_xy)
        self._draw_track_lines(surf, cam, car_xy)
        if name == "hood":
            self._draw_hood(surf)
        else:
            self._draw_car(surf, cam, car_pos, axes, vehicle)
        self._draw_minimap(surf, car_xy, self._follow_yaw)
        self._draw_hud(surf, vehicle, hud, name)

        frame = np.ascontiguousarray(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        if self._window is not None:
            self._window.blit(surf, (0, 0))
            pygame.display.flip()
        return frame

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        self._closed = True

    @property
    def closed(self) -> bool:
        """True after the window was closed by the user or :meth:`close` was called."""
        return self._closed

    # ------------------------------------------------------------- cameras
    def _make_camera(self, name, car_pos, axes, vehicle):
        W, H = self.width, self.height
        cx, cy = W / 2.0, H / 2.0
        X, Y, Z = axes
        if name == "chase":
            self._update_follow(car_pos, vehicle.yaw)
            pos = self._follow_pos
            yaw = self._follow_yaw
            fxy = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            cam_pos = pos - fxy * 7.5 + np.array([0.0, 0.0, 2.6])
            ground = float(self.terrain.height_at(cam_pos[0], cam_pos[1]))
            cam_pos[2] = max(cam_pos[2], ground + 1.2)
            look = pos + fxy * 4.0 + np.array([0.0, 0.0, 0.8])
            focal = (H / 2.0) / math.tan(math.radians(CHASE_FOV_DEG) / 2.0)
            return _Camera(cam_pos, look - cam_pos, Z_UP, focal, cx, cy)
        if name == "hood":
            cam_pos = car_pos + X * 1.2 + Z * 1.0
            focal = (H / 2.0) / math.tan(math.radians(HOOD_FOV_DEG) / 2.0)
            return _Camera(cam_pos, X, Z, focal, cx, cy)
        # top: looking straight down, heading up; half the frame height spans TOP_SPAN_FRAC * view_radius
        self._update_follow(car_pos, vehicle.yaw)
        pos = self._follow_pos
        yaw = self._follow_yaw
        cam_pos = np.array([pos[0], pos[1], pos[2] + TOP_HEIGHT])
        focal = H * TOP_HEIGHT / (2.0 * TOP_SPAN_FRAC * self.view_radius)
        up = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        return _Camera(cam_pos, np.array([0.0, 0.0, -1.0]), up, focal, cx, cy)

    def _update_follow(self, car_pos, yaw):
        if self._follow_pos is None or np.linalg.norm(car_pos - self._follow_pos) > FOLLOW_SNAP:
            self._follow_pos = car_pos.copy()
            self._follow_yaw = float(yaw)
            return
        self._follow_pos = self._follow_pos + FOLLOW_ALPHA * (car_pos - self._follow_pos)
        self._follow_yaw = self._follow_yaw + FOLLOW_ALPHA * float(_wrap(yaw - self._follow_yaw))
        self._follow_yaw = float(_wrap(self._follow_yaw))

    # ------------------------------------------------------------- terrain
    def _node_normals(self):
        H = self.terrain.heights
        res = self.terrain.resolution
        dzdy, dzdx = np.gradient(H, res)
        N = np.stack([-dzdx, -dzdy, np.ones_like(H)], axis=-1)
        return N / np.linalg.norm(N, axis=-1, keepdims=True)

    def _draw_terrain(self, surf, cam, car_xy):
        t = self.terrain
        n, res, org = t.n, t.resolution, t.origin
        R = self.view_radius
        cx, cy = car_xy
        # (cell size, outer half-size, exclusion half-size) coarse -> fine
        levels = ((6, R, 0.45 * R), (3, 0.45 * R, 0.12 * R), (1, 0.12 * R, None))
        Hz = t.heights
        W, Hh = self.width, self.height
        for k, outer, excl in levels:
            span = outer / res
            fx = (cx - org) / res
            fy = (cy - org) / res
            i0 = (int(math.floor(fx - span)) // k) * k
            j0 = (int(math.floor(fy - span)) // k) * k
            i1 = int(math.ceil(fx + span)) + k
            j1 = int(math.ceil(fy + span)) + k
            I = np.arange(i0, i1, k)
            J = np.arange(j0, j1, k)
            I = I[(I >= 0) & (I + k <= n - 1)]
            J = J[(J >= 0) & (J + k <= n - 1)]
            if I.size == 0 or J.size == 0:
                continue
            II, JJ = np.meshgrid(I, J)
            II = II.ravel()
            JJ = JJ.ravel()
            ci = np.stack([II, II + k, II + k, II], axis=1)   # (Q, 4)
            cj = np.stack([JJ, JJ, JJ + k, JJ + k], axis=1)
            xs = org + ci * res
            ys = org + cj * res
            if excl is not None:
                inside = (np.abs(xs - cx) <= excl) & (np.abs(ys - cy) <= excl)
                keep = ~np.all(inside, axis=1)
                ci, cj, xs, ys, II, JJ = ci[keep], cj[keep], xs[keep], ys[keep], II[keep], JJ[keep]
            zs = Hz[cj, ci]
            V = np.stack([xs, ys, zs], axis=-1)
            C = cam.to_cam(V)
            zc = C[..., 2]
            ok = np.all(zc > NEAR, axis=1)
            if not np.any(ok):
                continue
            C, zc = C[ok], zc[ok]
            ci, cj, II, JJ, xs, ys = ci[ok], cj[ok], II[ok], JJ[ok], xs[ok], ys[ok]
            sx = cam.cx + cam.focal * C[..., 0] / zc
            sy = cam.cy - cam.focal * C[..., 1] / zc
            minx, maxx = sx.min(1), sx.max(1)
            miny, maxy = sy.min(1), sy.max(1)
            vis = (maxx >= 0) & (minx <= W) & (maxy >= 0) & (miny <= Hh)
            if not np.any(vis):
                continue
            sx, sy, zc = sx[vis], sy[vis], zc[vis]
            ci, cj, II, JJ, xs, ys = ci[vis], cj[vis], II[vis], JJ[vis], xs[vis], ys[vis]

            # flat colour per quad: surface of the centre node, Lambert shade, checker, fog
            mi = np.minimum(II + k // 2, n - 1)
            mj = np.minimum(JJ + k // 2, n - 1)
            sid = t.surface_ids[mj, mi]
            base = SURFACE_RGB[sid]
            checker = 1.0 + 0.035 * (((II // k + JJ // k) & 1) * 2 - 1)
            nq = self._normals[cj, ci].mean(axis=1)
            nq = nq / np.linalg.norm(nq, axis=1, keepdims=True)
            shade = 0.62 + 0.38 * np.maximum(nq @ LIGHT, 0.0)
            dxy = np.hypot(xs.mean(1) - cx, ys.mean(1) - cy)
            fog = np.clip((dxy - 0.55 * R) / (0.45 * R), 0.0, 1.0)[:, None]
            col = base * (checker * shade)[:, None]
            col = col * (1.0 - fog) + FOG_COLOR * fog

            order = np.argsort(-zc.mean(1))
            pts = np.stack([sx, sy], axis=-1)[order].astype(np.int32).tolist()
            cols = np.clip(col[order], 0, 255).astype(np.int32).tolist()
            draw = pygame.draw.polygon
            for p, c in zip(pts, cols):
                draw(surf, c, p)

    # --------------------------------------------------------- track lines
    def _draw_track_lines(self, surf, cam, car_xy):
        tr = self.track
        R = self.view_radius * 1.05
        lift = np.array([0.0, 0.0, 0.08])
        if tr.closed:
            a_idx = np.arange(tr.n)
            b_idx = (a_idx + 1) % tr.n
        else:
            a_idx = np.arange(tr.n - 1)
            b_idx = a_idx + 1
        cxy = tr.centerline[:, :2]
        near = np.hypot(cxy[:, 0] - car_xy[0], cxy[:, 1] - car_xy[1]) <= R
        seg = near[a_idx] & near[b_idx]
        if not np.any(seg):
            return
        a = a_idx[seg]
        b = b_idx[seg]
        draw = pygame.draw.line
        for name, pts, colour, width in (
            ("left", tr.left_edge, EDGE_COLOR, 2),
            ("right", tr.right_edge, EDGE_COLOR, 2),
        ):
            P = pts + lift
            self._draw_segments(surf, cam, P[a], P[b], colour, width)
        # dashed centreline: every other segment
        dash = (a % 4) < 2
        P = tr.centerline + lift
        self._draw_segments(surf, cam, P[a[dash]], P[b[dash]], CENTRE_COLOR, 1)

    def _draw_segments(self, surf, cam, A, B, colour, width):
        CA = cam.to_cam(A)
        CB = cam.to_cam(B)
        ok = (CA[:, 2] > NEAR) & (CB[:, 2] > NEAR)
        if not np.any(ok):
            return
        sa = np.stack(cam.project(CA[ok]), axis=-1).astype(np.int32).tolist()
        sb = np.stack(cam.project(CB[ok]), axis=-1).astype(np.int32).tolist()
        for p, q in zip(sa, sb):
            pygame.draw.line(surf, colour, p, q, width)

    # ------------------------------------------------------------ the car
    def _draw_car(self, surf, cam, car_pos, axes, vehicle):
        X, Y, Z = axes
        M = np.stack([X, Y, Z], axis=1)     # body -> world
        prims = []                          # (depth, world points (k,3), colour)

        def add_face(world_pts, normal, colour, cam_pos_):
            centre = world_pts.mean(0)
            if np.dot(normal, cam_pos_ - centre) <= 0.0:
                return
            shade = 0.5 + 0.5 * max(float(np.dot(normal, LIGHT)), 0.0)
            col = tuple(int(c * shade) for c in colour)
            prims.append((world_pts, col))

        def box_faces(lo, hi, colour):
            lo = np.asarray(lo, float)
            hi = np.asarray(hi, float)
            c = (lo + hi) / 2.0
            h = (hi - lo) / 2.0
            for axis in range(3):
                for sgn in (-1.0, 1.0):
                    ax_b = [a for a in range(3) if a != axis]
                    nloc = np.zeros(3)
                    nloc[axis] = sgn
                    fc = c + nloc * h[axis]
                    corners = []
                    for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                        p = fc.copy()
                        p[ax_b[0]] += s1 * h[ax_b[0]]
                        p[ax_b[1]] += s2 * h[ax_b[1]]
                        corners.append(p)
                    yield np.array(corners), nloc, colour

        cam_pos = cam.pos
        for lo, hi, colour in CAR_BOXES:
            for loc_pts, nloc, col in box_faces(lo, hi, colour):
                wp = car_pos + loc_pts @ M.T
                add_face(wp, M @ nloc, col, cam_pos)

        hubs = np.asarray(vehicle.wheel_positions_world(), float).reshape(4, 3)
        steer = np.asarray(vehicle.wheel_steer_angles, float).reshape(4)
        spin = np.asarray(vehicle.wheel_rotations, float).reshape(4)
        tire = getattr(getattr(vehicle, "p", None), "tire", None)
        radius = float(getattr(tire, "radius", WHEEL_RADIUS_DEFAULT))
        if not np.all(np.isfinite(hubs)):
            hubs = np.zeros((4, 3))
        hw = WHEEL_WIDTH / 2.0
        ang = np.linspace(0.0, 2.0 * np.pi, WHEEL_SIDES, endpoint=False)
        for i in range(4):
            d = steer[i]
            f = math.cos(d) * X + math.sin(d) * Y          # rolling direction
            axle = _unit(np.cross(Z, f))                    # lateral axis of this wheel
            rim_dir = np.cos(ang)[:, None] * f + np.sin(ang)[:, None] * Z
            hub = hubs[i]
            rim = hub + radius * rim_dir                    # (S, 3)
            for side in (1.0, -1.0):
                off = hub + side * hw * axle
                add_face(off + radius * rim_dir, side * axle, TIRE_CAP, cam_pos)
                # spoke on the cap, rotating with wheel spin
                sp = math.cos(spin[i]) * f + math.sin(spin[i]) * Z
                lat = _unit(np.cross(axle, sp))
                sp_pts = np.array([
                    off + 0.9 * radius * sp + 0.05 * radius * lat,
                    off + 0.9 * radius * sp - 0.05 * radius * lat,
                    off - 0.9 * radius * sp - 0.05 * radius * lat,
                    off - 0.9 * radius * sp + 0.05 * radius * lat,
                ])
                add_face(sp_pts, side * axle, SPOKE, cam_pos)
            for s in range(WHEEL_SIDES):
                a_, b_ = s, (s + 1) % WHEEL_SIDES
                quad = np.array([
                    hub + hw * axle + radius * rim_dir[a_],
                    hub + hw * axle + radius * rim_dir[b_],
                    hub - hw * axle + radius * rim_dir[b_],
                    hub - hw * axle + radius * rim_dir[a_],
                ])
                nrad = (rim_dir[a_] + rim_dir[b_])
                nrad = _unit(nrad)
                add_face(quad, nrad, TIRE_SIDE, cam_pos)

        # project, cull behind the near plane, painter's sort
        polys, depths, cols = [], [], []
        for wp, col in prims:
            C = cam.to_cam(wp)
            if np.any(C[:, 2] <= NEAR):
                continue
            s = np.stack(cam.project(C), axis=-1)
            polys.append(s.astype(np.int32).tolist())
            depths.append(float(C[:, 2].mean()))
            cols.append(col)
        if not polys:
            return
        for k in np.argsort(-np.asarray(depths)):
            pygame.draw.polygon(surf, cols[k], polys[k])

    def _draw_hood(self, surf):
        """Dark bonnet silhouette along the bottom of the hood view."""
        W, H = self.width, self.height
        pts = [(0, H), (W, H), (int(W * 0.78), int(H * 0.70)), (int(W * 0.22), int(H * 0.70))]
        pygame.draw.polygon(surf, (40, 42, 48), pts)
        pygame.draw.line(surf, (90, 92, 100), pts[2], pts[3], 3)

    # ---------------------------------------------------------- minimap
    def _make_minimap(self):
        S = MINIMAP_SIZE
        surf = pygame.Surface((S, S))
        surf.fill((22, 34, 24))
        size = self.terrain.size
        P = self.track.centerline[:, :2]
        pix = self._map_pts(P, size, S)
        if self.track.closed:
            pygame.draw.lines(surf, (150, 150, 150), True, pix.tolist(), 3)
        else:
            pygame.draw.lines(surf, (150, 150, 150), False, pix.tolist(), 3)
            g = self._map_pts(self.track.goal[None, :2], size, S)[0]
            pygame.draw.circle(surf, (60, 220, 90), (int(g[0]), int(g[1])), 4)
        s0 = self._map_pts(P[:1], size, S)[0]
        pygame.draw.circle(surf, (240, 240, 240), (int(s0[0]), int(s0[1])), 3)
        pygame.draw.rect(surf, (200, 200, 200), surf.get_rect(), 1)
        return surf

    @staticmethod
    def _map_pts(xy, size, S):
        xy = np.asarray(xy, float)
        px = (xy[:, 0] + size / 2.0) / size * (S - 1)
        py = (size / 2.0 - xy[:, 1]) / size * (S - 1)
        return np.stack([px, py], axis=-1)

    def _draw_minimap(self, surf, car_xy, yaw):
        S = MINIMAP_SIZE
        x0 = self.width - S - 10
        y0 = 10
        surf.blit(self._minimap, (x0, y0))
        size = self.terrain.size
        cx, cy = self._map_pts(np.asarray(car_xy, float)[None, :], size, S)[0]
        cx, cy = x0 + cx, y0 + cy
        tip = (cx + 9 * math.cos(yaw), cy - 9 * math.sin(yaw))
        pygame.draw.line(surf, (255, 90, 90), (cx, cy), tip, 2)
        pygame.draw.circle(surf, (255, 40, 40), (int(cx), int(cy)), 4)

    # -------------------------------------------------------------- HUD
    def _draw_hud(self, surf, vehicle, hud, camera_name):
        hud = dict(hud or {})
        W, H = self.width, self.height
        speed = float(hud.get("speed_kmh", vehicle.speed_kmh))
        gear = int(hud.get("gear", vehicle.gear))
        rpm = float(hud.get("rpm", vehicle.engine_rpm))
        throttle = float(np.clip(hud.get("throttle", 0.0), 0.0, 1.0))
        brake = float(np.clip(hud.get("brake", 0.0), 0.0, 1.0))
        if "steer" in hud:
            steer = float(np.clip(hud["steer"], -1.0, 1.0))
        else:
            steer = float(np.clip(vehicle.wheel_steer_angles[0] / STEER_FULL_RAD, -1.0, 1.0))

        surf.blit(self._hud_panel, (8, 8))
        gear_txt = {-1: "R", 0: "N"}.get(gear, str(gear))
        lines = [f"{speed:6.1f} km/h", f"gear {gear_txt}", f"rpm {rpm:6.0f}"]
        for i, txt in enumerate(lines):
            surf.blit(self._font.render(txt, True, (240, 240, 240)), (16, 14 + i * 18))
        # rpm bar with redline zone
        bx, by, bw = 16, 74, 200
        pygame.draw.rect(surf, (60, 60, 64), (bx, by, bw, 8))
        frac = float(np.clip(rpm / REDLINE_RPM, 0.0, 1.0))
        bar_col = (90, 200, 90) if frac < 0.85 else (230, 70, 60)
        pygame.draw.rect(surf, bar_col, (bx, by, int(bw * frac), 8))
        # throttle and brake bars
        self._hbar(surf, 16, 88, 96, throttle, (80, 210, 80), "thr")
        self._hbar(surf, 122, 88, 96, brake, (220, 70, 60), "brk")

        # extra HUD items on the right, under the minimap
        y = 10 + MINIMAP_SIZE + 10
        for key in ("progress", "reward"):
            if key in hud:
                v = float(hud[key])
                txt = f"progress {100 * v:5.1f}%" if key == "progress" else f"reward {v:+.2f}"
                surf.blit(self._font.render(txt, True, (250, 250, 250)), (W - MINIMAP_SIZE - 10, y))
                y += 18
        for key, val in hud.items():
            if key in ("speed_kmh", "gear", "rpm", "throttle", "brake", "steer",
                       "progress", "reward", "message"):
                continue
            if isinstance(val, (int, float, np.floating, np.integer)):
                txt = f"{key} {float(val):.3f}"
                surf.blit(self._font_small.render(txt, True, (230, 230, 230)), (W - MINIMAP_SIZE - 10, y))
                y += 16

        # steering indicator
        sx0, sw, sy = W // 2 - 120, 240, H - 22
        pygame.draw.rect(surf, (50, 50, 54), (sx0, sy, sw, 8))
        pygame.draw.line(surf, (200, 200, 200), (sx0 + sw // 2, sy - 3), (sx0 + sw // 2, sy + 11), 1)
        mx = int(sx0 + sw / 2 + steer * sw / 2)
        pygame.draw.rect(surf, (250, 210, 60), (mx - 3, sy - 3, 6, 14))
        surf.blit(self._font_small.render("L", True, (220, 220, 220)), (sx0 - 14, sy - 4))
        surf.blit(self._font_small.render("R", True, (220, 220, 220)), (sx0 + sw + 6, sy - 4))

        if "message" in hud and hud["message"]:
            msg = str(hud["message"])
            surf.blit(self._font.render(msg, True, (255, 255, 255)),
                      (W // 2 - self._font.size(msg)[0] // 2, H - 52))
        surf.blit(self._font_small.render(f"camera: {camera_name}", True, (230, 230, 230)), (12, H - 18))

    def _hbar(self, surf, x, y, w, frac, colour, label):
        surf.blit(self._font_small.render(label, True, (230, 230, 230)), (x, y - 1))
        bx = x + 30
        pygame.draw.rect(surf, (60, 60, 64), (bx, y, w - 30, 8))
        pygame.draw.rect(surf, colour, (bx, y, int((w - 30) * frac), 8))

    # ------------------------------------------------------- sky, window
    def _make_sky(self):
        H, W = self.height, self.width
        t = np.linspace(0.0, 1.0, H)[:, None]
        col = SKY_TOP * (1.0 - t ** 0.7) + SKY_HORIZON * (t ** 0.7)
        arr = np.repeat(col[:, None, :], W, axis=1)     # (H, W, 3)
        return pygame.surfarray.make_surface(np.ascontiguousarray(arr.transpose(1, 0, 2)))

    def _pump_events(self):
        if self._window is None:
            return
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._closed = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                self.next_camera()



def save_frames(frames, out_dir, fps: int = 15):
    """Write frames (iterable of (H, W, 3) uint8) as PNGs, plus a GIF if imageio is installed.

    Returns the list of PNG paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    paths = []
    for i, f in enumerate(frames):
        p = out / f"frame_{i:05d}.png"
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(f.transpose(1, 0, 2)))
        pygame.image.save(surf, str(p))
        paths.append(p)
    try:
        import imageio.v2 as imageio  # optional
    except ImportError:
        imageio = None
    if imageio is not None and frames:
        imageio.mimsave(out / "episode.gif", frames, duration=1.0 / max(fps, 1))
    return paths
