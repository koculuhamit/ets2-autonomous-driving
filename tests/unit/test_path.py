"""Tests for reference paths.

Half of these pin the turn conventions. A lateral controller built on an
inverted sign steers away from the path, and the symptom -- a truck leaving
the road -- says nothing about which sign is wrong, so each convention is
asserted explicitly and traced back to the measurement it came from in the
module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.localization.path import ReferencePath, TrailerTracker


def straight(length: float = 120.0, spacing: float = 1.0) -> ReferencePath:
    """A path running along +z, so heading is zero."""
    z = np.arange(0.0, length, spacing)
    return ReferencePath(np.column_stack([np.zeros_like(z), z]), spacing=2.0)


def arc(curvature: float, length: float = 100.0) -> ReferencePath:
    """A constant-curvature path starting at the origin heading +z.

    Integrating `dx/ds = sin(ks)`, `dz/ds = cos(ks)` gives the closed form
    below, so the tests compare against an exact answer rather than another
    approximation.
    """
    s = np.arange(0.0, length, 1.0)
    points = np.column_stack(
        [(1 - np.cos(curvature * s)) / curvature, np.sin(curvature * s) / curvature]
    )
    return ReferencePath(points, spacing=2.0)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_length_matches_the_geometry() -> None:
    assert straight(100.0).length == pytest.approx(98.0, abs=2.0)


def test_resampling_gives_uniform_spacing() -> None:
    path = straight(100.0)
    step = np.hypot(np.diff(path.points[:, 0]), np.diff(path.points[:, 1]))
    assert step[:-1] == pytest.approx(2.0, abs=1e-6)
    assert 0.0 < step[-1] <= 2.0, "the last step to the end, however short"


def test_a_path_ends_where_its_points_do() -> None:
    """Cut to the spacing, a path to a gas pump out of Torino ended 1.8 m short
    of it, and the truck stopped where the game would not fill the tank."""
    path = ReferencePath(np.array([[0.0, 0.0], [0.0, 11.4]]), spacing=2.0)
    assert path.length == pytest.approx(11.4)
    end = path.sample_at(path.length)
    assert (end.x, end.z) == pytest.approx((0.0, 11.4))


def test_a_stationary_stretch_does_not_distort_the_path() -> None:
    """A truck waiting at a junction contributes thousands of identical points,
    and every quantity derived by differencing divides by roughly zero there."""
    moving = np.column_stack([np.zeros(50), np.arange(0.0, 50.0)])
    parked = np.tile(moving[-1], (500, 1))
    onwards = np.column_stack([np.zeros(50), np.arange(50.0, 100.0)])
    path = ReferencePath(np.vstack([moving, parked, onwards]), spacing=2.0)
    assert path.length == pytest.approx(98.0, abs=2.0)
    assert np.all(np.isfinite(path.points))


def test_too_few_points_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ReferencePath(np.array([[0.0, 0.0], [0.0, 0.001]]), spacing=2.0)


def test_a_spacing_below_the_floor_is_rejected() -> None:
    """Below half a metre the heading is differencing noise."""
    with pytest.raises(ValueError, match="spacing"):
        straight(spacing=1.0).__class__(np.zeros((10, 2)), spacing=0.1)


def test_wrong_shaped_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        ReferencePath(np.zeros((10, 3)), spacing=2.0)


# --------------------------------------------------------------------------- #
# Conventions -- each one is a way for a controller to steer the wrong way
# --------------------------------------------------------------------------- #


def test_heading_zero_means_travelling_towards_positive_z() -> None:
    assert straight().sample_at(50.0).heading == pytest.approx(0.0, abs=1e-6)


def test_positive_lateral_error_means_the_vehicle_is_left_of_the_path() -> None:
    """`+x` is the left-hand side: turning right decreases heading, and from
    heading zero that swings the vehicle towards `-x`."""
    path = straight()
    sample = path.nearest(3.0, 60.0, near=60.0)
    assert path.lateral_error(3.0, 60.0, sample) == pytest.approx(3.0, abs=1e-3)
    sample = path.nearest(-3.0, 60.0, near=60.0)
    assert path.lateral_error(-3.0, 60.0, sample) == pytest.approx(-3.0, abs=1e-3)


def test_a_right_turn_decreases_heading_and_has_negative_curvature() -> None:
    path = arc(-1 / 200)
    early = path.sample_at(5.0)
    late = path.sample_at(path.length - 5.0)
    assert late.heading < early.heading
    assert path.sample_at(path.length / 2).curvature < 0.0
    assert path.points[-1, 0] < 0.0, "a right turn should move towards -x"


def test_a_left_turn_increases_heading_and_has_positive_curvature() -> None:
    path = arc(+1 / 200)
    early = path.sample_at(5.0)
    late = path.sample_at(path.length - 5.0)
    assert late.heading > early.heading
    assert path.sample_at(path.length / 2).curvature > 0.0
    assert path.points[-1, 0] > 0.0, "a left turn should move towards +x"


@pytest.mark.parametrize("curvature", [-1 / 100, -1 / 400, 1 / 150, 1 / 500])
def test_curvature_matches_the_analytic_value(curvature: float) -> None:
    path = arc(curvature)
    measured = path.sample_at(path.length / 2).curvature
    assert measured == pytest.approx(curvature, rel=0.02)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_sampling_is_clamped_to_the_ends() -> None:
    path = straight()
    assert path.sample_at(-50.0).s == 0.0
    assert path.sample_at(1e6).s == pytest.approx(path.length)


def test_nearest_finds_the_foot_of_the_perpendicular() -> None:
    """Returning the closest vertex instead would be up to a metre out at a
    2 m spacing -- the same order as the error being controlled."""
    path = straight()
    sample = path.nearest(0.0, 51.0, near=51.0)
    assert sample.s == pytest.approx(51.0, abs=0.05)


def test_nearest_reproduces_the_points_it_was_built_from() -> None:
    z = np.arange(0.0, 200.0, 0.5)
    points = np.column_stack([np.sin(z / 40.0) * 8.0, z])
    path = ReferencePath(points, spacing=2.0)
    errors = []
    previous = None
    for x, z_value in points[::7]:
        sample = path.nearest(float(x), float(z_value), near=previous)
        previous = sample.s
        errors.append(path.lateral_error(float(x), float(z_value), sample))
    assert np.abs(errors).max() < 0.05


def test_the_search_window_follows_the_vehicle() -> None:
    """A route that crosses itself has two nearest points, and the one in
    space is not necessarily the one the vehicle is on."""
    z = np.arange(0.0, 100.0, 1.0)
    out = np.column_stack([np.zeros_like(z), z])
    back = np.column_stack([np.full_like(z, 4.0), z[::-1]])
    path = ReferencePath(np.vstack([out, back]), spacing=2.0)

    query = (0.5, 50.0)
    outbound = path.nearest(*query, near=50.0)
    returning = path.nearest(*query, near=path.length - 50.0)
    assert abs(outbound.s - returning.s) > 50.0


def test_heading_interpolation_takes_the_short_way_round() -> None:
    """Across the wrap a naive average points backwards."""
    z = np.arange(0.0, 60.0, 1.0)
    points = np.column_stack([np.zeros_like(z), -z])  # travelling towards -z
    path = ReferencePath(points, spacing=2.0)
    headings = [path.sample_at(s).heading for s in np.arange(0.0, path.length, 1.0)]
    assert np.abs(np.diff(headings)).max() < 0.1


# --------------------------------------------------------------------------- #
# Trailer projection
# --------------------------------------------------------------------------- #


def _closed_loop() -> ReferencePath:
    """A 100 m circle, whose end is its beginning."""
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    radius = 100.0 / (2.0 * np.pi)
    points = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
    return ReferencePath(points, spacing=1.0)


def test_trailer_is_not_reported_before_it_joins_the_route() -> None:
    path = _closed_loop()
    tracker = TrailerTracker(path, behind=8.0)
    assert tracker.update(0.0, path.sample_at(0.0).z, tractor_progress=3.0) is None
    assert tracker.progress is None


def test_trailer_on_a_closed_loop_is_not_found_at_the_far_end() -> None:
    """The bug: a trailer behind the start is nearest to the path's end, and
    an unseeded search says so -- 227 m of lateral error in a live report."""
    path = _closed_loop()
    tracker = TrailerTracker(path, behind=8.0)
    trailer = path.sample_at(12.0)
    error = tracker.update(trailer.x, trailer.z, tractor_progress=20.0)
    assert error is not None
    assert abs(error) < 0.1
    assert tracker.progress == pytest.approx(12.0, abs=1.0)


def test_trailer_error_is_signed_like_the_tractor_s() -> None:
    path = straight()
    tracker = TrailerTracker(path, behind=8.0)
    point = path.sample_at(20.0)
    left = tracker.update(point.x + 1.5, point.z, tractor_progress=28.0)
    assert left == pytest.approx(1.5, abs=0.05)


def test_trailer_search_is_seeded_from_its_own_last_position() -> None:
    """Once it is on the route the tractor's progress stops being the seed,
    so a self-crossing route resolves the same way it does for the tractor."""
    path = straight()
    tracker = TrailerTracker(path, behind=8.0)
    tracker.update(0.0, 20.0, tractor_progress=28.0)
    assert tracker.progress == pytest.approx(20.0, abs=1.0)
    tracker.update(0.0, 24.0, tractor_progress=32.0)
    assert tracker.progress == pytest.approx(24.0, abs=1.0)


@pytest.mark.parametrize("behind", [0.0, -1.0])
def test_a_trailer_length_must_be_positive(behind: float) -> None:
    with pytest.raises(ValueError, match="behind"):
        TrailerTracker(straight(), behind=behind)
