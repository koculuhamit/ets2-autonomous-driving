"""Swinging wide in tight turns, so the trailer follows the lane."""

from __future__ import annotations

import numpy as np
import pytest

from core.localization.path import WIDEN_MAX_M, widen_for_trailer


def turn(radius: float, left: bool) -> tuple[np.ndarray, np.ndarray]:
    """80 m north along x = 0, a quarter turn, 80 m on. Returns the line and
    the turn's centre. +x is to the left heading north (core.control.lateral)."""
    side = 1.0 if left else -1.0
    centre = np.array([side * radius, 80.0])
    angles = np.linspace(0.0, np.pi / 2, 60)
    arc = np.column_stack(
        [centre[0] - side * radius * np.cos(angles), centre[1] + radius * np.sin(angles)]
    )
    before = np.column_stack([np.zeros(40), np.linspace(0.0, 78.0, 40)])
    after = np.column_stack(
        [arc[-1, 0] + side * np.linspace(2.0, 80.0, 40), np.full(40, arc[-1, 1])]
    )
    return np.vstack([before, arc, after]), centre


def mid_turn_radius(line: np.ndarray, centre: np.ndarray, left: bool) -> float:
    """Distance from the centre where the line is half way round the turn."""
    side = 1.0 if left else -1.0
    angle = np.arctan2(line[:, 1] - centre[1], side * (centre[0] - line[:, 0]))
    half_way = np.abs(angle - np.pi / 4) < np.radians(5)
    return float(np.hypot(*(line[half_way] - centre).T).mean())


@pytest.mark.parametrize("left", [True, False])
def test_a_tight_turn_is_taken_wide(left: bool) -> None:
    """12 m: the trailer's axles run 3.1 m inside; the tractor goes out."""
    line, centre = turn(12.0, left)
    wide = widen_for_trailer(line)
    assert mid_turn_radius(wide, centre, left) == pytest.approx(12.0 + WIDEN_MAX_M, abs=0.3)


def test_straights_well_away_from_a_turn_are_left_alone() -> None:
    line, _ = turn(12.0, True)
    wide = widen_for_trailer(line)
    start = wide[wide[:, 1] < 40.0]
    assert np.abs(start[:, 0]).max() < 0.05


def test_a_gentle_bend_is_barely_moved() -> None:
    """300 m: the trailer is 0.1 m inside; nothing worth swinging out for."""
    line, centre = turn(300.0, True)
    wide = widen_for_trailer(line)
    assert mid_turn_radius(wide, centre, True) == pytest.approx(300.0, abs=0.2)


def test_in_a_stretch_asked_for_it_turns_are_taken_wider() -> None:
    """A border's customs lanes: trucks wait on the inside of the bends."""
    line, centre = turn(12.0, True)
    wide = widen_for_trailer(line)
    wider = widen_for_trailer(line, wider=[(np.array([0.0, 30.0]), np.array([92.0, 92.0]))])
    assert mid_turn_radius(wider, centre, True) > mid_turn_radius(wide, centre, True) + 1.0
    assert np.abs(wider[wider[:, 1] < 10.0, 0]).max() < 0.05, "outside the stretch, as before"


def test_a_driver_s_own_line_is_left_as_driven() -> None:
    """A taught crossing already swings wide where the driver needed it to."""
    line, _ = turn(12.0, True)
    plain = widen_for_trailer(line, max_shift=0.0)
    whole = [(line[0], line[-1])]
    exact = widen_for_trailer(line, exact=whole)
    assert np.hypot(*(exact - plain).T).max() < 0.05
    assert np.hypot(*(widen_for_trailer(line) - plain).T).max() > 1.0, "otherwise it would move"


def test_with_no_room_outside_a_turn_it_swings_only_as_far_as_allowed() -> None:
    """Milan: swung 1.2 m wide in a T-junction, the tractor hit a guard
    house's island on the outside."""
    line, centre = turn(12.0, True)

    def no_room(points, outward, wanted):
        return np.full(len(points), 0.6)

    tight = widen_for_trailer(line, room=no_room)
    assert mid_turn_radius(tight, centre, True) == pytest.approx(12.0 + 0.6, abs=0.25)
    roomy = widen_for_trailer(line, room=lambda p, o, w: np.full(len(p), np.inf))
    assert mid_turn_radius(roomy, centre, True) == pytest.approx(12.0 + WIDEN_MAX_M, abs=0.3)


def test_the_room_asked_about_is_on_the_outside_of_the_turn() -> None:
    line, centre = turn(12.0, True)
    seen = []

    def room(points, outward, wanted):
        seen.extend((p, o) for p, o, w in zip(points, outward, wanted, strict=True) if w > 0.6)
        return np.full(len(points), np.inf)

    widen_for_trailer(line, room=room)
    assert seen
    for point, out in seen:
        assert np.hypot(*(point + out - centre)) > np.hypot(*(point - centre)), "away from it"


def test_a_path_shorter_than_the_widening_s_windows_is_left_as_it_is() -> None:
    # 8 m of a tight turn, the whole route to a depot by Uppsala.
    a = np.linspace(0.0, 1.2, 9)
    points = np.column_stack([6.0 * np.sin(a), 6.0 - 6.0 * np.cos(a)])
    line = widen_for_trailer(points, room=lambda p, outward, wanted: np.full(len(p), 9.0))
    assert line.shape[1] == 2
    assert np.allclose(line[0], points[0])
