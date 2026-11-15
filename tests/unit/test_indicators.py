"""Indicating before turns and lane changes."""

from __future__ import annotations

import numpy as np
import pytest

from core.common.controls import Button
from core.control.indicators import (
    CHANGE_LEAD_M,
    LEFT,
    PRESS_S,
    RIGHT,
    SETTLE_S,
    TURN_LEAD_M,
    Indicators,
    Signal,
    route_signals,
    wanted,
)
from core.control.lane_change import LaneChange
from core.localization.path import ReferencePath
from core.map import Lane, LaneGraph, RoadNetwork
from core.planning.route import LANE_CHANGE_LENGTH_M, ONWARD, Route
from tests.unit.lane_networks import polyline


def turning_right() -> RoadNetwork:
    """300 m north, then a junction lane that turns right, as the map marks it."""
    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(
        Lane(
            points=polyline((0, 300), (-20, 320)),
            source="prefab",
            owner="2",
            blinker="RightBlinker",
        )
    )
    graph.connect(0, 1)
    return RoadNetwork.compile(graph)


def test_a_turn_the_map_marks_is_indicated_from_before_it_until_it_is_done() -> None:
    net = turning_right()
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    (signal,) = route_signals(route, path)
    assert signal.side == RIGHT
    assert signal.start == pytest.approx(300.0 - TURN_LEAD_M, abs=2.0)
    assert signal.end == pytest.approx(300.0 + float(net.lane_length[1]), abs=2.0)


def test_a_lane_change_is_indicated_to_its_side() -> None:
    net = turning_right()
    route = Route(net, (0,), ())
    path = ReferencePath(np.column_stack([np.zeros(150), np.arange(150.0) * 2]), spacing=2.0)
    change = LaneChange(100.0, -1, np.zeros((2, 2)))
    (signal,) = route_signals(route, path, [change])
    assert (signal.start, signal.end, signal.side) == (
        100.0 - CHANGE_LEAD_M,
        100.0 + LANE_CHANGE_LENGTH_M,
        LEFT,
    )


def test_of_two_at_once_the_one_that_ends_first_is_indicated() -> None:
    """A change to the left, then a turn to the right: the change first."""
    signals = [Signal(60.0, 220.0, LEFT), Signal(180.0, 400.0, RIGHT)]
    assert wanted(signals, 190.0) == LEFT
    assert wanted(signals, 230.0) == RIGHT
    assert wanted(signals, 500.0) == 0


def test_the_button_is_pressed_once_and_the_switch_given_time_to_follow() -> None:
    indicators = Indicators([Signal(0.0, 100.0, RIGHT)])
    step = dict(progress=50.0, left_on=False, right_on=False)
    assert indicators.buttons(**step, now=0.0) == {Button.INDICATOR_RIGHT}
    assert indicators.buttons(**step, now=PRESS_S / 2) == {Button.INDICATOR_RIGHT}
    assert indicators.buttons(**step, now=PRESS_S + 0.1) == frozenset(), "released, waiting"
    on = dict(progress=50.0, left_on=False, right_on=True)
    assert indicators.buttons(**on, now=PRESS_S + SETTLE_S + 0.1) == frozenset()


def test_the_wrong_side_is_switched_over_and_a_finished_one_cancelled() -> None:
    indicators = Indicators([Signal(0.0, 100.0, RIGHT)])
    assert indicators.buttons(progress=50.0, left_on=True, right_on=False, now=0.0) == {
        Button.INDICATOR_RIGHT
    }
    later = PRESS_S + SETTLE_S + 1.0
    assert indicators.buttons(progress=150.0, left_on=False, right_on=True, now=later) == {
        Button.INDICATOR_RIGHT
    }
