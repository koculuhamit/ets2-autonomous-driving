"""Following a vehicle along the lane graph.

Built on small hand-made networks, each the shape of a way the nearest-lane
guess went wrong on the recorded drives: a fork whose branches start at the
same point, a lane alongside, two lanes of a junction side by side that the
graph does not join, and traffic the other way.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from core.map import LaneMatcher
from tests.unit.lane_networks import network, polyline


def drive(matcher: LaneMatcher, path: np.ndarray, step: float = 1.0) -> list:
    """Feed positions every `step` metres along `path`, heading along it."""
    matches = []
    for a, b in itertools.pairwise(path):
        heading = math.atan2(b[0] - a[0], b[2] - a[2])
        count = max(1, round(math.hypot(b[0] - a[0], b[2] - a[2]) / step))
        for t in np.linspace(0.0, 1.0, count, endpoint=False):
            matches.append(
                matcher.update(a[0] + t * (b[0] - a[0]), a[2] + t * (b[2] - a[2]), heading)
            )
    return matches


def test_it_follows_a_lane_and_ignores_the_oncoming_one() -> None:
    """Two lanes 4.5 m apart; the heading, not the distance, tells them apart."""
    forward = polyline((0, 0), (0, 100))
    oncoming = polyline((4.5, 100), (4.5, 0))
    matcher = LaneMatcher(network([forward, oncoming]))

    # Drive a metre off the forward lane towards the oncoming one.
    matches = drive(matcher, polyline((1.0, 0), (1.0, 100)))
    assert all(m is not None and m.lane == 0 for m in matches)


def test_at_a_fork_it_ends_on_the_branch_actually_driven() -> None:
    """Both branches start at the same point, so the first metres cannot decide."""
    approach = polyline((0, 0), (0, 50))
    straight = polyline((0, 50), (0, 100))
    right = polyline((0, 50), (-30, 90))
    matcher = LaneMatcher(network([approach, straight, right], successors=[(0, 1), (0, 2)]))

    matches = drive(matcher, polyline((0, 0), (0, 50), (-30, 90)))
    assert matches[-1].lane == 2
    lanes = [m.lane for m in matches]
    assert lanes.index(2) > lanes.index(0)
    assert 1 not in lanes[lanes.index(2) :]


def test_it_changes_lanes_with_the_vehicle() -> None:
    inner = polyline((0, 0), (0, 200))
    outer = polyline((-4.5, 0), (-4.5, 200))
    matcher = LaneMatcher(network([inner, outer], changes=[(0, 1)]))

    matches = drive(matcher, polyline((0, 0), (0, 80), (-4.5, 120), (-4.5, 200)))
    assert matches[10].lane == 0
    assert matches[-1].lane == 1


def test_it_does_not_change_lanes_for_a_moment_s_drift() -> None:
    """A lane change costs something, so a brief swerve past the midline is not one.

    For three metres the vehicle is 2.6 m off its lane and 1.9 m from the next:
    nearer the other lane, but not for long enough to pay for changing to it.
    """
    inner = polyline((0, 0), (0, 200))
    outer = polyline((-4.5, 0), (-4.5, 200))
    matcher = LaneMatcher(network([inner, outer], changes=[(0, 1)]))

    swerve = polyline((0, 0), (0, 90), (-2.6, 90.5), (-2.6, 92.5), (0, 93), (0, 200), spacing=0.5)
    matches = drive(matcher, swerve)
    assert all(m.lane == 0 for m in matches)


def test_it_follows_a_vehicle_into_a_lane_alongside_the_graph_does_not_join() -> None:
    """Inside junctions two approach lanes can run side by side with no lane change.

    On the recorded drives the truck used the other one, and a matcher that
    only moved along the graph lost it there, in the same place, every lap.
    """
    right = polyline((0, 0), (0, 60))
    left = polyline((-4.7, 0), (-4.7, 60))
    matcher = LaneMatcher(network([right, left]))

    matches = drive(matcher, polyline((0, 0), (0, 15), (-4.7, 30), (-4.7, 60)))
    assert matches[5].lane == 0
    assert matches[-1].lane == 1
    assert all(m is not None for m in matches)


def test_off_the_network_it_says_so_and_then_finds_its_way_back() -> None:
    lane = polyline((0, 0), (0, 200))
    matcher = LaneMatcher(network([lane]))

    assert matcher.update(0.0, 10.0, 0.0) is not None
    assert matcher.update(60.0, 20.0, 0.0) is None
    found = matcher.update(0.2, 30.0, 0.0)
    assert found is not None and found.lane == 0


def test_the_lateral_error_is_signed_like_the_path_s() -> None:
    """Positive to the left of the lane, which for a lane along +z is +x."""
    matcher = LaneMatcher(network([polyline((0, 0), (0, 100))]))
    match = matcher.update(1.2, 50.0, 0.0)
    assert match is not None
    assert match.lateral == pytest.approx(1.2, abs=1e-3)
    assert match.s == pytest.approx(50.0, abs=0.01)
