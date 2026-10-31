"""Tests for the scripted (segment-based) track builder."""
import math
import warnings

import numpy as np
import pytest

from drivesim.terrain import SURFACE_ASPHALT, Terrain
from drivesim.track import Track, wrap_angle
from drivesim.track_script import (
    EXAMPLES,
    MIN_RADIUS,
    RADIUS_BY_SEVERITY,
    Segment,
    TrackScript,
    build_world_from_script,
    example_script,
    parse_track_script,
)

R_CORNER = RADIUS_BY_SEVERITY[3]


# ----------------------------------------------------------------- helpers
def corner_length(R, deg):
    """Expected arc length of a corner: R*theta plus the clothoid transition T (documented formula)."""
    theta = math.radians(deg)
    arc = R * theta
    T = min(max(0.3 * arc, 6.0), 40.0, 0.4 * arc)
    return arc + T


def assert_track_continuity(tr):
    """Same checks as tests/test_world.py::test_track_continuity, for any Track."""
    c = tr.centerline
    steps = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)
    if tr.closed:
        steps = np.append(steps, np.linalg.norm(c[0, :2] - c[-1, :2]))
    assert np.all(np.abs(steps - tr.spacing) < 0.01 * tr.spacing)
    dth = np.abs(wrap_angle(np.diff(tr.heading)))
    assert dth.max() < 0.2
    if tr.closed:
        assert abs(wrap_angle(tr.heading[0] - tr.heading[-1])) < 0.2
    off = np.linalg.norm(tr.left_edge[:, :2] - c[:, :2], axis=1)
    assert np.allclose(off, tr.half_width)


def build(dsl_or_script, **kw):
    script = dsl_or_script if isinstance(dsl_or_script, TrackScript) else TrackScript.from_dsl(dsl_or_script)
    return script.build(**kw)


def seg_row(tr, index=0):
    return tr.segments[index]


@pytest.fixture(scope="module")
def rally_world():
    return build_world_from_script(EXAMPLES["technical_rally"], seed=0)


@pytest.fixture(scope="module")
def oval_world():
    return build_world_from_script(EXAMPLES["high_speed_oval"], seed=0)


@pytest.fixture(scope="module")
def hill_world():
    return build_world_from_script(EXAMPLES["switchback_hillclimb"], seed=0)


# ------------------------------------------------ per-segment heading & length
def test_straight_length_and_no_heading_change():
    tr = build(TrackScript().straight(200))
    row = seg_row(tr)
    assert row.kind == "straight"
    assert row.s1 - row.s0 == pytest.approx(200.0, abs=1e-6)
    assert row.heading_change == pytest.approx(0.0, abs=1e-9)
    assert tr.length == pytest.approx(200.0, abs=tr.spacing)
    assert not tr.closed


def test_long_straight_is_high_speed_and_same_geometry():
    a = TrackScript().long_straight(300)
    assert a.segments[0].high_speed
    assert TrackScript.from_dsl("HS300") == a
    tr = a.build()
    assert tr.length == pytest.approx(300.0, abs=tr.spacing)


@pytest.mark.parametrize("sev,length,deg", [(3, "normal", 75.0), (1, "short", 40.0), (6, "long", 120.0)])
def test_pace_note_corner_heading_and_length(sev, length, deg):
    tr = build(TrackScript().right(severity=sev, length=length))
    row = seg_row(tr)
    R = RADIUS_BY_SEVERITY[sev]
    assert row.heading_change == pytest.approx(-math.radians(deg), abs=1e-4)
    assert row.s1 - row.s0 == pytest.approx(corner_length(R, deg), abs=0.05)


def test_left_corner_turns_positive_and_right_negative():
    assert seg_row(build(TrackScript().left(3))).heading_change > 0
    assert seg_row(build(TrackScript().right(3))).heading_change < 0


def test_tightens_and_opens_keep_total_heading_change():
    base = build(TrackScript().left(3))
    tight = build(TrackScript().left(3, tightens=True))
    opened = build(TrackScript().left(3, opens=True))
    for tr in (tight, opened):
        assert seg_row(tr).heading_change == pytest.approx(seg_row(base).heading_change, abs=1e-4)
    # tightening ends with a smaller radius, so the curvature at the end of the hold is higher
    assert np.max(np.abs(tight.curvature)) > np.max(np.abs(base.curvature))
    assert np.max(np.abs(opened.curvature)) < np.max(np.abs(base.curvature))


def test_caution_flag_is_recorded_only():
    tr = build(TrackScript().left(2, caution=True).straight(50))
    assert tr.segments[0].caution is True
    assert tr.segments[1].caution is False
    assert TrackScript().left(2, caution=True) != TrackScript().left(2)  # flag is part of the script


def test_hairpin_turns_pi():
    tr = build(TrackScript().hairpin("left"))
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(math.pi, abs=1e-4)
    assert row.s1 - row.s0 == pytest.approx(corner_length(9.0, 180.0), abs=0.05)


def test_switchback_is_two_opposite_hairpins():
    tr = build(TrackScript().switchback("left", radius=9.0, leg=30.0))
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(0.0, abs=1e-4)  # opposite hairpins cancel
    assert row.s1 - row.s0 == pytest.approx(2 * corner_length(9.0, 180.0) + 30.0, abs=0.1)
    # the two hairpins sit on opposite sides of the leg: the track reaches both signs of curvature
    assert tr.curvature.max() > 0 > tr.curvature.min()


@pytest.mark.parametrize("n,expected", [(4, 0.0), (3, math.radians(40.0)), (2, 0.0)])
def test_zigzag_net_heading_and_length(n, expected):
    tr = build(TrackScript().zigzag(n=n, direction="left", severity=3, length="short"))
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(expected, abs=1e-4)
    assert row.s1 - row.s0 == pytest.approx(n * corner_length(R_CORNER, 40.0), abs=0.2)


def test_zigzag_alternates_curvature_sign():
    tr = build(TrackScript().zigzag(n=4, direction="left", severity=3, length="normal"))
    signs = []
    k = tr.curvature
    peaks = [k[i] for i in range(len(k)) if abs(k[i]) > 0.5 / R_CORNER]
    # group consecutive samples of equal sign
    for v in peaks:
        if not signs or np.sign(v) != signs[-1]:
            signs.append(np.sign(v))
    assert signs == [1.0, -1.0, 1.0, -1.0]  # four corners, alternating


def test_esses_and_chicane_are_zigzags():
    assert TrackScript().esses(3) == TrackScript().zigzag(3, "left", 5, "short")
    assert TrackScript().chicane() == TrackScript().zigzag(2, "left", 3, "short")


def test_turn_by_explicit_angle_and_radius():
    tr = build(TrackScript().turn(90, "left", radius=25.0))
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(math.pi / 2, abs=1e-4)
    assert row.s1 - row.s0 == pytest.approx(corner_length(25.0, 90.0), abs=0.05)
    k_hold = np.max(np.abs(tr.curvature))
    assert k_hold == pytest.approx(1 / 25.0, rel=0.05)


def test_sweep_is_a_long_fast_bend():
    tr = build(TrackScript().sweep("right", 90))
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(-math.pi / 2, abs=1e-4)
    assert row.s1 - row.s0 == pytest.approx(corner_length(80.0, 90.0), abs=0.05)


@pytest.mark.parametrize("builder,shape_h,net", [
    (lambda s: s.drop(15, 80), 15.0, -15.0),
    (lambda s: s.climb(15, 80), 15.0, 15.0),
    (lambda s: s.crest(4, 80), 4.0, 0.0),
    (lambda s: s.dip(4, 80), -4.0, 0.0),
])
def test_elevation_segments(builder, shape_h, net):
    tr = builder(TrackScript()).build()
    row = seg_row(tr)
    assert row.heading_change == pytest.approx(0.0, abs=1e-6)
    assert row.elevation_change == pytest.approx(net, abs=0.05)
    z = tr.centerline[:, 2] - tr.centerline[0, 2]
    if shape_h > 0 and net == 0.0:
        assert z.max() == pytest.approx(shape_h, abs=0.1)  # crest peak
    if shape_h < 0:
        assert z.min() == pytest.approx(shape_h, abs=0.1)  # dip trough


def test_steep_section_net_rise():
    tr = build(TrackScript().steep(0.12, 60))
    row = seg_row(tr)
    assert row.elevation_change == pytest.approx(0.12 * 60, abs=0.05)
    assert np.max(np.abs(np.diff(tr.centerline[:, 2]))) / tr.spacing <= 0.25  # within max_grade


def test_segment_table_covers_whole_route():
    tr = build("S50 L3 S40 HAIRPIN-L/9 S30")
    assert len(tr.segments) == 5
    assert tr.segments[0].s0 == pytest.approx(0.0, abs=1e-6)
    for a, b in zip(tr.segments[:-1], tr.segments[1:]):
        assert a.s1 == pytest.approx(b.s0, abs=1e-6)


# ------------------------------------------------------ smoothness and bounds
def test_curvature_changes_continuously_no_jumps():
    # the route may cross itself here; only curvature is under test
    tr = TrackScript.from_dsl("S40 L3 S30 HAIRPIN-L/9 S30 ZIGZAG3-R4S SWEEP-R90/70 S40").build(self_intersection="ignore")
    dk = np.abs(np.diff(tr.curvature))
    assert dk.max() < 0.05  # clothoid ramps: no step changes in curvature
    assert np.abs(tr.curvature).max() <= 1.0 / MIN_RADIUS + 1e-9


def test_continuity_and_track_type():
    tr = build("S100 L3 S80 R4 S60")
    assert isinstance(tr, Track)
    assert_track_continuity(tr)
    assert tr.min_radius == MIN_RADIUS


def test_min_radius_limit_is_enforced():
    with pytest.raises(ValueError, match="below the minimum"):
        TrackScript().turn(40, "left", radius=3.0)
    with pytest.raises(ValueError, match="below min_radius"):
        TrackScript().left(3).build(min_radius=30.0)


def test_max_grade_is_enforced_with_clear_error():
    with pytest.raises(ValueError, match="grade of up to .* above max_grade"):
        TrackScript().straight(50).drop(15, 50).build()
    # the same drop is fine with a larger allowance
    tr = TrackScript().straight(50).drop(15, 50).build(max_grade=0.5)
    assert tr.length > 0


def test_grade_stays_within_limit_for_examples_and_closure(rally_world, hill_world):
    for _, tr in (rally_world, hill_world):
        grade = np.abs(np.diff(tr.centerline[:, 2])) / tr.spacing
        assert grade.max() <= 0.25 + 1e-3


def test_self_intersection_error_and_warning():
    # a 360 degree turn returns to its own start: the route touches itself
    with pytest.raises(ValueError, match="crosses or touches itself"):
        TrackScript().turn(360, "left", radius=30).build()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tr = TrackScript().turn(360, "left", radius=30).build(self_intersection="warn")
    assert any("touches itself" in str(x.message) for x in w)
    assert tr.n > 4


def test_terrain_too_small_is_an_error():
    with pytest.raises(ValueError, match="at least"):
        build_world_from_script(EXAMPLES["high_speed_oval"], seed=0, size=300.0)


# ------------------------------------------------------------ closure
def square_script(last_angle=90.0):
    s = TrackScript()
    for i in range(4):
        s.straight(100).turn(last_angle if i == 3 else 90, "left", radius=20)
    return s


def test_closure_by_blending_small_gap():
    s = square_script(last_angle=86.0).close()
    tr = s.build()
    assert tr.closed
    assert tr.closure["applied"] is True
    assert tr.closure["gap_after"] < 0.5
    assert_track_continuity(tr)


def test_exact_closed_square_needs_no_completion():
    tr = square_script().close().build()
    assert tr.closure["completed"] is False
    assert_track_continuity(tr)


def test_closed_false_is_point_to_point_even_with_close_token():
    s = square_script(last_angle=45.0).close()
    tr = s.build(closed=False)
    assert not tr.closed
    assert tr.length > 0


def test_two_segment_script_auto_completes_into_closed_circuit():
    tr = TrackScript.from_dsl("S100 L3 S100 CLOSE").build()
    assert tr.closed
    assert tr.closure["completed"] is True
    assert len(tr.completion_segments) > 0
    assert tr.script == TrackScript.from_dsl("S100 L3 S100 CLOSE")  # script is kept as written
    assert_track_continuity(tr)
    assert tr.closure["gap_after"] < 0.05  # the route ends on its start to within 5 cm
    assert abs(wrap_angle(tr.heading[0] - tr.heading[-1])) < 0.02


def test_strict_mode_errors_instead_of_completing():
    with pytest.raises(ValueError, match="strict"):
        TrackScript.from_dsl("S100 L3 S100").build(closed=True, strict=True)
    with pytest.raises(ValueError, match="strict"):
        TrackScript.from_dsl("S100 L3 S100 CLOSE").build(strict=True)


def test_completion_failure_has_clear_message():
    with pytest.raises(ValueError, match="could not auto-complete"):
        TrackScript.from_dsl("S200 L3 S150 R3").build(closed=True)


def test_point_to_point_goal_extension():
    base = TrackScript.from_dsl("S200 L3 S100").build(closed=False)
    end = base.centerline[-1, :2] - base.centerline[0, :2]
    goal = end + np.array([60.0, 80.0])  # 100 m beyond where the script ends
    tr = TrackScript.from_dsl("S200 L3 S100").build(closed=False, goal=tuple(goal))
    assert tr.closure["completed"] is True
    # the goal is measured from the start, which the builder centres; compare relative offsets
    assert np.allclose(tr.goal[:2] - tr.centerline[0, :2], goal, atol=0.5)
    with pytest.raises(ValueError, match="strict"):
        TrackScript.from_dsl("S200 L3 S100").build(closed=False, goal=tuple(goal), strict=True)


def test_documented_dsl_example_completes():
    tr = TrackScript.from_dsl("S100 L3 S100 CLOSE").build()
    assert tr.closed and tr.closure["completed"]
    assert_track_continuity(tr)


def test_far_open_route_fails_clearly_when_completion_cannot_reach():
    # a 188 m gap with a 45 degree heading error is beyond the completion template
    with pytest.raises(ValueError, match="could not auto-complete"):
        TrackScript.from_dsl("S200 R4L L2 HAIRPIN-L ZIGZAG4 DROP-15/80 SWEEP-R90 CLOSE").build()


# ------------------------------------------------------------ builder & DSL
def test_builder_chain_matches_dsl():
    chain = (TrackScript().straight(200).right(severity=3, length="long").hairpin("left")
             .zigzag(n=4).drop(15, 80).close())
    dsl = TrackScript.from_dsl("S200 R3L HAIRPIN-L/9 ZIGZAG4-L3 DROP-15/80 CLOSE")
    assert chain == dsl
    assert chain.to_dsl() == "S200 R3L HAIRPIN-L/9 ZIGZAG4-L3 DROP-15/80 CLOSE"


def test_dsl_round_trip_for_examples_and_builder():
    for name, text in EXAMPLES.items():
        s = TrackScript.from_dsl(text)
        assert TrackScript.from_dsl(s.to_dsl()) == s, name
        assert parse_track_script(s.to_dsl()) == s
    s = (TrackScript().long_straight(300).left(2, "short", tightens=True, caution=True)
         .right(6, "long", opens=True).switchback("right", 10, 25).sweep("left", 45, 60)
         .turn(12.5, "right", 30).crest(3, 40).dip(2, 30).steep(-0.08, 50).esses(3, "right").close())
    assert TrackScript.from_dsl(s.to_dsl()) == s


def test_dsl_parses_every_token_kind():
    s = TrackScript.from_dsl("S100 HS200 L2 R4L> R5S< L3! HAIRPIN-R SWITCHBACK-L/9/30 ZIGZAG3-R4S "
                             "ESSES2-L TURN-R45/30 SWEEP-L90 DROP-10/60 CLIMB-10/60 CREST-4/50 DIP-4/50 "
                             "GRADE+12/60 GRADE-8/50 CLOSE")
    kinds = [x.kind for x in s.segments]
    assert kinds == ["straight", "straight", "corner", "corner", "corner", "corner", "hairpin",
                     "switchback", "zigzag", "zigzag", "turn", "sweep", "elev", "elev", "elev",
                     "elev", "steep", "steep"]
    assert s.closed
    assert s.segments[3].tightens and s.segments[4].opens and s.segments[5].caution
    assert s.segments[16].grade == pytest.approx(0.12)
    assert s.segments[17].grade == pytest.approx(-0.08)


def test_dsl_is_case_insensitive_and_whitespace_tolerant():
    assert TrackScript.from_dsl("  s200\tr4l  ") == TrackScript.from_dsl("S200 R4L")


@pytest.mark.parametrize("bad,fragment", [
    ("Q5", "Q5"),
    ("R7L", "R7L"),                  # severity out of range
    ("R4X", "R4X"),
    ("DROP-15", "DROP-15"),          # missing length
    ("DROP-15/0", "DROP-15/0"),      # zero length
    ("HAIRPIN-L/3", "HAIRPIN-L/3"),  # radius below minimum
    ("GRADE15/60", "GRADE15/60"),    # missing sign
    ("ZIGZAG0-L3", "ZIGZAG0-L3"),    # no turns
    ("TURN-L400/30", "TURN-L400/30"),  # angle above 360
    ("S-100", "S-100"),              # negative length
    ("L3 L3>!!", "L3>!!"),           # bad modifier count (only '!' allowed twice)
])
def test_dsl_rejects_bad_input(bad, fragment):
    with pytest.raises(ValueError, match="track script token|token|radius|grade|angle|must|zigzag|modifier"):
        TrackScript.from_dsl(bad)


def test_dsl_rejects_empty_and_close_only():
    with pytest.raises(ValueError, match="empty"):
        TrackScript.from_dsl("   ")
    with pytest.raises(ValueError, match="no segments"):
        TrackScript.from_dsl("CLOSE")


def test_segment_constructor_validates_directly():
    with pytest.raises(ValueError):
        Segment("corner", direction="L", severity=7)
    with pytest.raises(ValueError):
        Segment("corner", direction="L", severity=2, tightens=True, opens=True)
    with pytest.raises(ValueError):
        Segment("elev", shape="drop", height=5, length=0)
    with pytest.raises(ValueError):
        Segment("steep", grade=0.0, length=10)


def test_track_script_repr_and_len():
    s = TrackScript.from_dsl("S10 R2")
    assert len(s) == 2
    assert "S10" in repr(s)


# ------------------------------------------------ examples carved into terrain
@pytest.mark.parametrize("name", list(EXAMPLES))
def test_examples_are_valid_tracks(name):
    tr = example_script(name).build()
    assert isinstance(tr, Track)
    assert tr.length > 500
    assert_track_continuity(tr)


def test_rally_example_carves_into_terrain(rally_world):
    terrain, tr = rally_world
    assert tr.closed and tr.closure["gap_after"] < 0.05
    idx = np.arange(0, tr.n, 9)
    nrm = np.column_stack([-tr.tangent[idx, 1], tr.tangent[idx, 0]])
    worst = 0.0
    for frac in (-0.9, 0.0, 0.9):
        p = tr.centerline[idx, :2] + frac * tr.half_width * nrm
        worst = max(worst, np.abs(terrain.height_at(p[:, 0], p[:, 1]) - tr.centerline[idx, 2]).max())
    assert worst < 0.05
    assert np.all(terrain.surface_type_at(tr.centerline[idx, 0], tr.centerline[idx, 1]) == SURFACE_ASPHALT)


def test_oval_example_carves_into_terrain(oval_world):
    terrain, tr = oval_world
    assert tr.closed
    assert isinstance(terrain, Terrain)
    idx = np.arange(0, tr.n, 9)
    assert np.all(terrain.surface_type_at(tr.centerline[idx, 0], tr.centerline[idx, 1]) == SURFACE_ASPHALT)


def test_hill_example_is_point_to_point_and_climbs(hill_world):
    terrain, tr = hill_world
    assert not tr.closed
    assert tr.centerline[-1, 2] > tr.centerline[0, 2] + 5.0
    assert np.isfinite(terrain.heights).all()


def test_world_build_is_deterministic():
    a = build_world_from_script(EXAMPLES["switchback_hillclimb"], seed=4)
    b = build_world_from_script(EXAMPLES["switchback_hillclimb"], seed=4)
    assert np.array_equal(a[1].centerline, b[1].centerline)
    assert np.array_equal(a[0].heights, b[0].heights)


def test_world_accepts_dsl_string_and_script():
    t1, k1 = build_world_from_script("S120 L3 S80 R3", seed=1)
    t2, k2 = build_world_from_script(TrackScript.from_dsl("S120 L3 S80 R3"), seed=1)
    assert np.array_equal(k1.centerline, k2.centerline)


# ------------------------------------------------- project and lookahead
def test_project_and_lookahead_on_scripted_track(rally_world):
    _, tr = rally_world
    rng = np.random.default_rng(3)
    for i in range(0, tr.n, 97):
        pr = tr.project(tr.centerline[i], hint=i)
        assert abs(pr.lateral) < 1e-6 and pr.on_track
    for _ in range(20):
        i = int(rng.integers(0, tr.n))
        yaw = rng.uniform(-np.pi, np.pi)
        pos = tr.centerline[i, :2] + rng.uniform(-2, 2, 2)
        dists = np.array([0.0, 5.0, 20.0])
        la = tr.lookahead(pos, yaw, dists)
        pr = tr.project(pos)
        xyz, _ = tr.point_at(pr.s + dists)
        c, s = np.cos(yaw), np.sin(yaw)
        wx = pos[0] + c * la[:, 0] - s * la[:, 1]
        wy = pos[1] + s * la[:, 0] + c * la[:, 1]
        assert np.allclose(wx, xyz[:, 0], atol=1e-6)
        assert np.allclose(wy, xyz[:, 1], atol=1e-6)
