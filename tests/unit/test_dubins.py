"""Dubins paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.planning.dubins import dubins_length, dubins_path


@pytest.mark.parametrize(
    "end",
    [(0.0, 30.0, 0.0), (20.0, 20.0, math.pi / 2), (-15.0, 5.0, math.pi), (5.0, -10.0, -2.0)],
)
def test_the_path_ends_where_asked_at_the_length_it_says(end) -> None:
    start = (0.0, 0.0, 0.0)
    path = dubins_path(start, end, 8.0, step=0.05)
    assert np.hypot(path[-1][0] - end[0], path[-1][1] - end[1]) < 0.05
    assert abs((path[-1][2] - end[2] + math.pi) % (2 * math.pi) - math.pi) < 0.01
    steps = np.hypot(*np.diff(np.array([p[:2] for p in path]), axis=0).T)
    assert steps.sum() == pytest.approx(dubins_length(start, end, 8.0), rel=0.01)


def test_straight_ahead_is_the_distance_and_behind_is_a_turn_round() -> None:
    assert dubins_length((0.0, 0.0, 0.0), (0.0, 30.0, 0.0), 8.0) == pytest.approx(30.0)
    back = dubins_length((0.0, 0.0, 0.0), (0.0, -30.0, math.pi), 8.0)
    assert back > 30.0 + math.pi * 8.0 - 1.0


def test_a_left_quarter_circle_turns_with_positive_curvature() -> None:
    """Heading increases to the left: from heading 0 (along +z), left is +x."""
    path = dubins_path((0.0, 0.0, 0.0), (10.0, 10.0, math.pi / 2), 10.0, step=0.1)
    assert all(p[3] > 0 for p in path[1:])
    assert dubins_length((0.0, 0.0, 0.0), (10.0, 10.0, math.pi / 2), 10.0) == pytest.approx(
        math.pi * 10.0 / 2
    )
