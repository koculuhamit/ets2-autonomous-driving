"""Tests for the curvature speed profile."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.localization.path import ReferencePath
from core.planning.speed_profile import SpeedProfile


def straight(length: float = 600.0) -> ReferencePath:
    z = np.arange(0.0, length, 1.0)
    return ReferencePath(np.column_stack([np.zeros_like(z), z]), spacing=2.0)


def straight_then_arc(radius: float, straight_len: float = 400.0, arc_len: float = 300.0):
    z = np.arange(0.0, straight_len, 1.0)
    first = np.column_stack([np.zeros_like(z), z])
    curvature = -1.0 / radius
    s = np.arange(0.0, arc_len, 1.0)
    second = np.column_stack(
        [
            (1 - np.cos(curvature * s)) / curvature,
            straight_len + np.sin(curvature * s) / curvature,
        ]
    )
    return ReferencePath(np.vstack([first, second]), spacing=2.0)


def test_a_straight_path_is_limited_only_by_the_ceiling() -> None:
    profile = SpeedProfile(straight(), max_speed=25.0)
    assert profile.speeds.max() == pytest.approx(25.0)
    assert profile.speed_at(300.0) == pytest.approx(25.0, abs=0.5)


def test_a_corner_is_limited_by_lateral_acceleration() -> None:
    """v = sqrt(a / k): a 50 m radius at 2.5 m/s^2 is 11.2 m/s."""
    profile = SpeedProfile(straight_then_arc(50.0), max_lateral_acceleration=2.5, max_speed=30.0)
    in_corner = profile.speed_at(600.0)
    assert in_corner == pytest.approx(math.sqrt(2.5 * 50.0), rel=0.15)


def test_a_junction_turn_is_taken_at_walking_pace_for_the_wheel() -> None:
    """North of Zurich a 9-10 m turn at 16 km/h, as the lateral limit
    allowed, left the truck 2-3 m wide while its wheel came round."""
    from core.planning.speed_profile import TIGHT_SPEED_MPS

    tight = SpeedProfile(straight_then_arc(9.5, arc_len=15.0), max_speed=30.0)
    assert tight.slowest() <= TIGHT_SPEED_MPS + 1e-6
    roomy = SpeedProfile(straight_then_arc(30.0), max_speed=30.0)
    assert roomy.slowest() == pytest.approx(math.sqrt(2.5 * 30.0), rel=0.15), (
        "a wide bend as before"
    )


def test_a_tighter_corner_is_slower() -> None:
    wide = SpeedProfile(straight_then_arc(200.0), max_speed=30.0).slowest()
    tight = SpeedProfile(straight_then_arc(30.0), max_speed=30.0).slowest()
    assert tight < wide


def test_braking_starts_before_the_corner() -> None:
    """The point of the backward pass. A limit that only applies once you are
    in the corner is a statement, not a plan -- the same argument the brake's
    dead zone forced on the longitudinal controller."""
    profile = SpeedProfile(
        straight_then_arc(30.0),
        max_lateral_acceleration=2.5,
        max_deceleration=1.5,
        max_speed=25.0,
    )
    # The corner allows sqrt(2.5 * 30) = 8.66 m/s, and shedding 25 down to that
    # at 1.5 m/s^2 takes (25^2 - 8.66^2) / 3 = 183 m. So the profile must leave
    # the ceiling around s = 217 and not before.
    corner_speed = math.sqrt(2.5 * 30.0)
    braking_distance = (25.0**2 - corner_speed**2) / (2 * 1.5)
    assert profile.speed_at(395.0) == pytest.approx(corner_speed, rel=0.2)
    assert profile.speed_at(400.0 - braking_distance * 0.5) < 25.0 - 1.0
    assert profile.speed_at(400.0 - braking_distance * 1.5) == pytest.approx(25.0, abs=0.5)


def test_the_deceleration_between_points_is_achievable() -> None:
    """Every step of the profile has to be one the vehicle could actually
    perform, or the plan asks for braking that does not exist."""
    profile = SpeedProfile(straight_then_arc(30.0), max_deceleration=1.5, max_speed=30.0)
    speeds, positions = profile.speeds, profile.positions
    for index in range(len(speeds) - 1):
        span = positions[index + 1] - positions[index]
        if span <= 0:
            continue
        required = (speeds[index] ** 2 - speeds[index + 1] ** 2) / (2 * span)
        assert required <= 1.5 + 1e-6


def test_a_harder_braking_budget_lets_it_brake_later() -> None:
    gentle = SpeedProfile(straight_then_arc(30.0), max_deceleration=0.5, max_speed=25.0)
    firm = SpeedProfile(straight_then_arc(30.0), max_deceleration=3.0, max_speed=25.0)
    assert firm.speed_at(380.0) > gentle.speed_at(380.0)


def test_the_profile_never_falls_below_the_floor() -> None:
    """A profile that reaches zero in a corner leaves the vehicle stuck."""
    profile = SpeedProfile(straight_then_arc(5.0), min_speed=3.0, max_speed=25.0)
    assert profile.speeds.min() >= 3.0


def test_speed_at_is_clamped_outside_the_path() -> None:
    profile = SpeedProfile(straight(), max_speed=20.0)
    assert profile.speed_at(-100.0) == pytest.approx(profile.speeds[0])
    assert profile.speed_at(1e6) == pytest.approx(profile.speeds[-1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lateral_acceleration": 0.0},
        {"max_deceleration": -1.0},
        {"min_speed": 0.0},
        {"min_speed": 20.0, "max_speed": 5.0},
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SpeedProfile(straight(), **kwargs)  # type: ignore[arg-type]


def test_coming_to_a_corner_it_stops_gaining_speed_well_before_it_brakes() -> None:
    from core.planning.speed_profile import COAST_DECELERATION

    profile = SpeedProfile(
        straight_then_arc(30.0), max_lateral_acceleration=2.5, max_deceleration=1.5, max_speed=25.0
    )
    corner_speed = profile.speed_at(400.0)
    lift_off = (25.0**2 - corner_speed**2) / (2 * COAST_DECELERATION)
    braking = (25.0**2 - corner_speed**2) / (2 * 1.5)
    assert profile.coasting_at(400.0 - lift_off * 1.2) == pytest.approx(25.0, abs=0.5)
    assert (
        profile.coasting_at(400.0 - braking * 1.2) < profile.speed_at(400.0 - braking * 1.2) - 3.0
    )
    assert all(
        profile.coasting_at(s) <= profile.speed_at(s) + 1e-9 for s in np.arange(0.0, 700.0, 5.0)
    )


def test_on_a_straight_road_there_is_nothing_to_lift_off_for() -> None:
    profile = SpeedProfile(straight(), max_speed=25.0)
    assert profile.coasting_at(300.0) == pytest.approx(profile.speed_at(300.0))


def test_it_does_not_gain_speed_a_bend_just_ahead_takes_away_again() -> None:
    """Between two bends out of Milan the profile rose by 5 km/h for 30 m."""
    profile = SpeedProfile(straight_then_arc(30.0), max_speed=25.0)
    corner = profile.speed_at(420.0)
    near = 400.0 - 30.0
    assert profile.coasting_at(near, speed=10.0) == pytest.approx(corner, abs=0.3)
    assert profile.coasting_at(near, speed=0.0) > corner + 1.0
