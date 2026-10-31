"""Tests for procedural terrain, track generation and world assembly."""
import numpy as np
import pytest

from drivesim.terrain import (
    SURFACE_ASPHALT,
    SURFACE_FRICTION,
    SURFACE_GRASS,
    Terrain,
)
from drivesim.track import Track, wrap_angle
from drivesim.world import generate_world

SIZE = 600.0


@pytest.fixture(scope="module")
def world_closed():
    return generate_world(11, closed=True, difficulty=0.5)


@pytest.fixture(scope="module")
def world_open():
    return generate_world(11, closed=False, difficulty=0.5)


# --------------------------------------------------------------- determinism
def test_terrain_deterministic_by_seed():
    a = Terrain.generate(42)
    b = Terrain.generate(42)
    c = Terrain.generate(43)
    assert np.array_equal(a.heights, b.heights)
    assert np.array_equal(a.surface_ids, b.surface_ids)
    assert not np.array_equal(a.heights, c.heights)


def test_world_deterministic_by_seed():
    for closed in (True, False):
        t1, k1 = generate_world(7, closed=closed, difficulty=0.8)
        t2, k2 = generate_world(7, closed=closed, difficulty=0.8)
        assert np.array_equal(t1.heights, t2.heights)
        assert np.array_equal(t1.surface_ids, t2.surface_ids)
        assert np.array_equal(k1.centerline, k2.centerline)
        assert k1.length == k2.length


def test_terrain_shape_and_bounds():
    t = Terrain.generate(1, size=600.0, resolution=1.0)
    assert t.heights.shape == (601, 601)
    assert t.size == 600.0
    assert abs(t.resolution - 1.0) < 1e-9
    assert np.isfinite(t.heights).all()
    assert np.ptp(t.heights) > 5.0  # actually hilly


# ----------------------------------------------------------- track continuity
@pytest.mark.parametrize("closed", [True, False])
def test_track_continuity(world_closed, world_open, closed):
    _, tr = world_closed if closed else world_open
    c = tr.centerline
    steps = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)
    if closed:
        steps = np.append(steps, np.linalg.norm(c[0, :2] - c[-1, :2]))
    assert np.all(np.abs(steps - tr.spacing) < 0.01 * tr.spacing)
    assert 0.5 <= tr.spacing <= 2.0
    # heading does not jump between samples
    dth = np.abs(wrap_angle(np.diff(tr.heading)))
    assert dth.max() < 0.2
    if closed:
        assert abs(wrap_angle(tr.heading[0] - tr.heading[-1])) < 0.2
    # edges are exactly half a width from the centreline
    off = np.linalg.norm(tr.left_edge[:, :2] - c[:, :2], axis=1)
    assert np.allclose(off, tr.half_width)
    assert np.allclose(tr.width, 10.0)


def test_track_closed_loop_geometry(world_closed):
    _, tr = world_closed
    assert tr.closed
    assert abs(tr.length - tr.n * tr.spacing) < 1e-9
    assert tr.length > 300


# ------------------------------------------------------------ curvature bounds
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("closed", [True, False])
def test_curvature_bounded(closed, difficulty):
    terrain = Terrain.generate(5)
    tr = Track.generate(terrain, rng=123, closed=closed, difficulty=difficulty)
    assert np.all(np.isfinite(tr.curvature))
    assert np.abs(tr.curvature).max() <= 1.0 / tr.min_radius + 1e-9
    # curvature_at agrees with sampled values at nodes
    s_nodes = tr.s[5:-5]
    assert np.allclose(tr.curvature_at(s_nodes), tr.curvature[5:-5])


def test_curvature_at_wraps_for_closed(world_closed):
    _, tr = world_closed
    assert np.isclose(tr.curvature_at(0.0), tr.curvature_at(tr.length))


# ------------------------------------------------------------- projection
def test_centerline_points_project_to_zero_lateral(world_closed):
    _, tr = world_closed
    for i in range(0, tr.n, 17):
        pr = tr.project(tr.centerline[i], hint=i)
        assert abs(pr.lateral) < 1e-6
        assert pr.on_track
        assert abs(pr.s - tr.s[i]) < 1e-6 or abs(abs(pr.s - tr.s[i]) - tr.length) < 1e-6


def test_projection_lateral_sign_and_magnitude(world_open):
    _, tr = world_open
    i = tr.n // 3
    t = tr.tangent[i]
    left = np.array([-t[1], t[0]])
    p = tr.centerline[i, :2]
    pl = tr.project(np.append(p + 3.0 * left, 0.0), hint=i)
    pr_ = tr.project(p - 3.0 * left, hint=i)
    # tolerance: the projection uses the exact polyline chord, not the node normal
    assert abs(pl.lateral - 3.0) < 2e-3
    assert abs(pr_.lateral + 3.0) < 2e-3
    assert not tr.project(p + 6.0 * left).on_track
    assert tr.project(p + 3.0 * left).on_track


def test_projection_progress_and_heading(world_closed):
    _, tr = world_closed
    i = tr.n // 2
    pr = tr.project(tr.centerline[i] + np.array([0.0, 0.0, 0.0]), hint=i)
    # heading is the chord direction at the projection; node heading is the
    # average of two chords, so they differ by up to half a turn per sample
    assert abs(wrap_angle(pr.heading - tr.heading[i])) < 3e-3
    assert 0.0 <= pr.progress < 1.0


def test_hint_and_global_search_agree(world_closed):
    _, tr = world_closed
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = int(rng.integers(0, tr.n))
        p = tr.centerline[i, :2] + rng.uniform(-3, 3, 2)
        hinted = tr.project(p, hint=max(0, i - 5))  # hint slightly behind
        glob = tr.project(p)
        assert abs(hinted.s - glob.s) < 1e-6 or abs(abs(hinted.s - glob.s) - tr.length) < 1e-6
        assert abs(hinted.lateral - glob.lateral) < 1e-6


def test_hint_far_away_falls_back_to_global(world_closed):
    _, tr = world_closed
    i = tr.n // 4
    j = (i + tr.n // 2) % tr.n  # far along the loop
    pr = tr.project(tr.centerline[j], hint=i)
    assert pr.index == j or abs(pr.s - tr.s[j]) < 1e-6


def test_projection_and_heading_error(world_open):
    _, tr = world_open
    i = 10
    yaw = tr.heading[i] + 0.3
    err = tr.heading_error(tr.centerline[i], yaw)
    assert abs(err - (-0.3)) < 3e-3  # track is 0.3 rad to the car's right
    assert abs(tr.heading_error(tr.centerline[i], tr.heading[i])) < 3e-3


# -------------------------------------------------------------- lookahead
def test_lookahead_frame_correctness(world_closed):
    _, tr = world_closed
    rng = np.random.default_rng(3)
    dists = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    for _ in range(30):
        i = int(rng.integers(0, tr.n))
        yaw = rng.uniform(-np.pi, np.pi)
        pos = tr.centerline[i, :2] + rng.uniform(-2, 2, 2)
        la = tr.lookahead(pos, yaw, dists)
        assert la.shape == (len(dists), 2)
        # invert the car-frame transform and compare with the true track points
        pr = tr.project(pos)
        xyz, _ = tr.point_at(pr.s + dists)
        c, s = np.cos(yaw), np.sin(yaw)
        fwd, lft = la[:, 0], la[:, 1]
        wx = pos[0] + c * fwd - s * lft
        wy = pos[1] + s * fwd + c * lft
        assert np.allclose(wx, xyz[:, 0], atol=1e-6)
        assert np.allclose(wy, xyz[:, 1], atol=1e-6)


def test_lookahead_left_right_sign(world_open):
    _, tr = world_open
    i = 40
    t = tr.tangent[i]
    right = np.array([t[1], -t[0]])
    # Car 2 m to the right of the centreline, facing along the track.
    pos = tr.centerline[i, :2] + 2.0 * right
    la = tr.lookahead(pos, tr.heading[i], [0.0, 5.0])
    assert abs(la[0, 0]) < 0.1
    assert abs(la[0, 1] - 2.0) < 0.05          # track is 2 m to the car's left
    assert la[1, 0] > 4.0                      # and ahead of it


def test_lookahead_beyond_goal_repeats_goal(world_open):
    _, tr = world_open
    la = tr.lookahead(tr.centerline[-1, :2], tr.heading[-1], [50.0])
    assert abs(la[0, 0]) < 1e-6 and abs(la[0, 1]) < 1e-6


# -------------------------------------------------------- terrain under track
@pytest.mark.parametrize("closed", [True, False])
def test_terrain_flattened_under_track(world_closed, world_open, closed):
    terrain, tr = world_closed if closed else world_open
    idx = np.arange(0, tr.n, 3)
    nrm = np.column_stack([-tr.tangent[idx, 1], tr.tangent[idx, 0]])
    worst = 0.0
    for frac in (-0.9, -0.5, 0.0, 0.5, 0.9):
        p = tr.centerline[idx, :2] + frac * tr.half_width * nrm
        dz = np.abs(terrain.height_at(p[:, 0], p[:, 1]) - tr.centerline[idx, 2])
        worst = max(worst, dz.max())
    assert worst < 0.05, f"terrain under road deviates by {worst:.3f} m"
    # the road itself is asphalt, friction 1.0
    assert np.all(terrain.surface_type_at(tr.centerline[idx, 0], tr.centerline[idx, 1]) == SURFACE_ASPHALT)
    assert np.allclose(terrain.surface_at(tr.centerline[idx, 0], tr.centerline[idx, 1]), 1.0)


def test_track_grade_is_limited(world_closed, world_open):
    for _, tr in (world_closed, world_open):
        grade = np.abs(np.diff(tr.centerline[:, 2])) / tr.spacing
        assert grade.max() < 0.12


def test_embankment_blends_to_grass(world_closed):
    terrain, tr = world_closed
    i = tr.n // 2
    t = tr.tangent[i]
    nrm = np.array([-t[1], t[0]])
    far = tr.centerline[i, :2] + 30.0 * nrm
    assert terrain.surface_type_at(far[0], far[1]) == SURFACE_GRASS
    assert terrain.surface_at(far[0], far[1]) == SURFACE_FRICTION[SURFACE_GRASS]


# ------------------------------------------------------- point to point
def test_point_to_point_start_not_goal(world_open):
    _, tr = world_open
    start, _ = tr.start_pose()
    assert not tr.closed
    assert np.linalg.norm(start[:2] - tr.goal[:2]) > 0.3 * SIZE
    assert np.allclose(tr.goal, tr.centerline[-1])


def test_start_pose_matches_centerline(world_closed):
    _, tr = world_closed
    pos, yaw = tr.start_pose()
    assert np.allclose(pos, tr.centerline[0])
    assert abs(wrap_angle(yaw - tr.heading[0])) < 1e-12


# ------------------------------------------------------- terrain queries
def test_terrain_queries_vectorised():
    t = Terrain.generate(9)
    x = np.linspace(-200, 200, 50)
    y = np.linspace(-150, 150, 50)
    h = t.height_at(x, y)
    assert h.shape == (50,)
    assert t.height_at(1.5, 2.5) == pytest.approx(float(t.height_at(np.array(1.5), np.array(2.5))))
    # exact at grid nodes
    node_x = t.origin + 10 * t.resolution
    node_y = t.origin + 20 * t.resolution
    assert t.height_at(node_x, node_y) == pytest.approx(t.heights[20, 10])
    # clamped outside the terrain
    assert t.height_at(1e6, 1e6) == pytest.approx(t.heights[-1, -1])
    n = t.normal_at(x, y)
    assert n.shape == (50, 3)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.all(n[:, 2] > 0)
    assert t.normal_at(0.0, 0.0).shape == (3,)
    fr = t.surface_at(x, y)
    assert fr.shape == (50,)
    assert set(np.unique(fr)).issubset(set(SURFACE_FRICTION))


def test_terrain_rejects_bad_input():
    with pytest.raises(ValueError):
        Terrain(np.zeros((3, 4)), size=10.0)
    with pytest.raises(ValueError):
        Terrain.generate(-1)


def test_bad_parameters_rejected():
    t = Terrain.generate(2, size=300.0, resolution=1.0, patches=False)
    with pytest.raises(ValueError):
        Track.generate(t, rng=0, difficulty=1.5)


# ------------------------------------------------------- many seeds
def test_many_track_seeds_generate_without_error():
    terrain = Terrain.generate(0)  # track generation depends on terrain only via its shape/heights
    for seed in range(60):
        for closed in (True, False):
            d = (seed % 11) / 10.0
            tr = Track.generate(terrain, rng=seed, closed=closed, difficulty=d)
            assert tr.n > 10
            assert np.abs(tr.curvature).max() <= 1.0 / tr.min_radius + 1e-9


@pytest.mark.parametrize("seed", list(range(12)))
def test_many_full_worlds_generate_without_error(seed):
    for closed in (True, False):
        terrain, tr = generate_world(seed, closed=closed, difficulty=(seed % 5) / 4.0, size=400.0)
        assert np.isfinite(terrain.heights).all()
        assert tr.n > 10
        pr = tr.project(tr.centerline[tr.n // 2])
        assert abs(pr.lateral) < 1e-6
