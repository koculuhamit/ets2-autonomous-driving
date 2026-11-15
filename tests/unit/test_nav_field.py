"""The game's quickest route to a destination, from every lane at once."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from core.map.ferries import FerryLink
from core.planning.nav_field import (
    ACROSS_STEP,
    ONWARD_STEP,
    SEA_STEP,
    NavField,
    nav_length,
)
from tests.unit.lane_networks import network, polyline

# A fork at (0, 100): a short local road straight on to (0, 1100), or a
# longer motorway round by the east; both end at the destination lane.
APPROACH = polyline((0, 0), (0, 100))
LOCAL = polyline((0, 100), (0, 1100))
MOTORWAY = polyline((0, 100), (300, 400), (300, 800), (0, 1100))
GOAL = polyline((0, 1100), (0, 1200))


def fork(local_kmh: float, motorway_kmh: float):
    net = network([APPROACH, LOCAL, MOTORWAY, GOAL], successors=[(0, 1), (0, 2), (1, 3), (2, 3)])
    speeds = np.array([60.0, local_kmh, motorway_kmh, 60.0], dtype=np.float32)
    return replace(net, lane_nav_speed=speeds)


def test_the_game_takes_the_quicker_way_not_the_shorter() -> None:
    net = fork(local_kmh=40.0, motorway_kmh=72.0)
    field = NavField.to_goals(net, {3: 0.0})
    assert field.way(0)[0] == [0, 2, 3]


def test_and_the_shorter_when_it_is_also_quicker() -> None:
    net = fork(local_kmh=64.0, motorway_kmh=72.0)
    field = NavField.to_goals(net, {3: 0.0})
    assert field.way(0)[0] == [0, 1, 3]


def test_the_distance_is_that_of_the_way_taken() -> None:
    net = fork(local_kmh=40.0, motorway_kmh=72.0)
    field = NavField.to_goals(net, {3: 0.0})
    expected = sum(nav_length(net, lane) for lane in (0, 2))
    assert field.distance[0] == pytest.approx(expected)
    assert field.time[0] == pytest.approx(
        nav_length(net, 0) * 3.6 / 60.0 + nav_length(net, 2) * 3.6 / 72.0
    )


def test_part_way_along_a_lane_what_is_behind_is_not_counted() -> None:
    net = fork(local_kmh=40.0, motorway_kmh=72.0)
    field = NavField.to_goals(net, {3: 0.0})
    assert field.distance_at(0, 50.0) == pytest.approx(field.distance[0] - nav_length(net, 0) / 2)


def test_a_road_the_game_s_map_leaves_out_is_not_routed_over() -> None:
    net = fork(local_kmh=64.0, motorway_kmh=72.0)
    hidden = np.array([False, True, False, False])
    field = NavField.to_goals(replace(net, lane_hidden=hidden), {3: 0.0})
    assert field.way(0)[0] == [0, 2, 3]


def test_from_a_hidden_road_the_way_goes_onto_the_roads_the_map_shows() -> None:
    """Off the train at Folkestone, the way out of the terminal is hidden."""
    net = fork(local_kmh=64.0, motorway_kmh=72.0)
    hidden = np.array([True, True, False, False])
    field = NavField.to_goals(replace(net, lane_hidden=hidden), {3: 0.0})
    assert math.isfinite(field.time[0])
    assert field.way(0)[0][-1] == 3


def test_a_road_the_gps_avoids_is_gone_round_if_there_is_a_way() -> None:
    net = fork(local_kmh=64.0, motorway_kmh=72.0)
    avoid = np.array([False, True, False, False])
    field = NavField.to_goals(replace(net, lane_gps_avoid=avoid), {3: 0.0})
    assert field.way(0)[0] == [0, 2, 3]


def test_moving_across_costs_next_to_nothing() -> None:
    left = polyline((-3.5, 0), (-3.5, 300))
    right = polyline((0, 0), (0, 300))
    on = polyline((0, 300), (0, 400))
    net = network([left, right, on], successors=[(1, 2)], changes=[(0, 1)])
    field = NavField.to_goals(net, {2: 0.0})
    assert field.way(0) == ([0, 1, 2], [ACROSS_STEP, ONWARD_STEP])
    assert field.distance[0] == pytest.approx(field.distance[1])


def test_a_ferry_is_weighed_by_its_crossing_time() -> None:
    """Across the water in 5 minutes, or round by road: the road is 19 km
    of the game's at 72 km/h, 16 minutes."""
    shore = polyline((0, 0), (0, 100))
    road = polyline((0, 100), (1000, 100))
    far = polyline((1000, 100), (1000, 200))
    net = network([shore, road, far], successors=[(0, 1), (1, 2)])
    net = replace(net, lane_nav_speed=np.full(3, 72.0, dtype=np.float32))
    for minutes, expected in ((5.0, SEA_STEP), (60.0, ONWARD_STEP)):
        link = FerryLink(0, 2, 1000.0, "west", "east", False, minutes)
        field = NavField.to_goals(replace(net, ferry_links={0: (link,)}), {2: 0.0})
        assert int(field.step[0]) == expected


def test_a_crossing_counts_as_far_as_the_game_states() -> None:
    """The Channel Tunnel's train: 60 km by the game, 125 of its kilometres
    between the terminals as the crow flies."""
    shore = polyline((0, 0), (0, 100))
    far = polyline((6500, 100), (6500, 200))
    net = network([shore, far])
    link = FerryLink(0, 1, 6500.0, "tunnel_c", "tunnel_d", True, 35.0, 60.0)
    field = NavField.to_goals(replace(net, ferry_links={0: (link,)}), {1: 0.0})
    assert field.distance[0] == pytest.approx(60_000.0 + nav_length(net, 0))


def test_where_the_destination_cannot_be_reached_the_time_is_infinite() -> None:
    net = fork(local_kmh=64.0, motorway_kmh=72.0)
    field = NavField.to_goals(net, {0: 0.0})
    assert math.isinf(field.time[3])


def test_quiet_roads_in_an_avoided_place_cost_as_the_games_avoided_lanes() -> None:
    """By a border in Denmark the way went through a station's service road
    by the main road, and the truck hit a tree at its side, twice."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning.nav_field import _avoided
    from tests.unit.lane_networks import polyline

    graph = LaneGraph()
    for points, ai in (
        (polyline((0, 0), (0, 100)), True),  # the main road
        (polyline((10, 0), (10, 100)), False),  # the service road beside it
        (polyline((500, 0), (500, 100)), False),  # a quiet road elsewhere
    ):
        graph.lanes.append(Lane(points=points, source="road", owner="1", ai_vehicles=ai))
    net = RoadNetwork.compile(graph)
    avoided = _avoided(net, places=[(5.0, 50.0, 30.0)])
    assert avoided.tolist() == [False, True, False]
