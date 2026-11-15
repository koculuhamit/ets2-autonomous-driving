"""Crossing the sea: ferry lines as links between lanes, and routes over them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.map import CompanyPlace, RoadNetwork
from core.map.ferries import GAME_KM_PER_MAP_KM, link_ferries, load_links, save_links
from core.map.sector import Ferry, FerryLine
from core.planning.route import FERRY, ONWARD, Route, reroute_from
from tests.unit.lane_networks import network, polyline

# Two coasts 5 km apart: a road to a quay in the west, a quay and a road on
# in the east. The quays' ferry items stand in the water, off the lanes.
WEST_ROAD = polyline((0, 0), (0, 500))
WEST_QUAY = polyline((0, 500), (0, 560))
EAST_QUAY = polyline((5000, 560), (5000, 620))
EAST_ROAD = polyline((5000, 620), (5000, 1500))
COASTS = network([WEST_ROAD, WEST_QUAY, EAST_QUAY, EAST_ROAD], successors=[(0, 1), (2, 3)])
PORTS = [
    Ferry("west", np.array([0.0, 0.0, 600.0]), np.zeros(3), train=False),
    Ferry("east", np.array([5000.0, 0.0, 540.0]), np.zeros(3), train=False),
]
LINES = [FerryLine("west", "east", 190.0), FerryLine("east", "north", 50.0)]


def with_links(net):
    by_lane: dict[int, list] = {}
    for link in link_ferries(net, PORTS, LINES):
        by_lane.setdefault(link.board, []).append(link)
    return replace(net, ferry_links={lane: tuple(found) for lane, found in by_lane.items()})


def test_a_line_links_the_dead_end_at_one_quay_to_the_road_on_from_the_other() -> None:
    (link,) = link_ferries(COASTS, PORTS, LINES)
    assert (link.board, link.land) == (1, 2)
    assert (link.start, link.end) == ("west", "east")
    assert link.cost == pytest.approx(190.0 * 1000.0 / GAME_KM_PER_MAP_KM)


def test_a_link_keeps_the_crossing_s_length_as_the_game_states_it() -> None:
    (link,) = link_ferries(COASTS, PORTS, LINES)
    assert link.km == pytest.approx(190.0)


def terminal(hidden_exit: bool) -> RoadNetwork:
    """A train's terminal: the platform lanes run 60 m on to where trucks
    drive onto the train, and end there; the way out starts 150 m off,
    beside the platform, and leads 1 km away -- or is a hidden road."""
    from core.map import Lane, LaneGraph

    graph = LaneGraph()
    pieces = [
        (((5000, 500), (5000, 560)), False),  # platform
        (((5000, 560), (5000, 580)), False),  # onto the train: a dead end
        (((5150, 540), (5150, 700)), hidden_exit),  # the way out
        (((5150, 700), (5150, 1700)), False),  # and on
    ]
    for index, (points, hidden) in enumerate(pieces):
        graph.lanes.append(
            Lane(points=polyline(*points), source="road", owner=f"{index + 1:016X}", hidden=hidden)
        )
    graph.connect(0, 1)
    graph.connect(2, 3)
    return RoadNetwork.compile(graph)


def test_the_truck_is_put_ashore_where_a_road_leads_out_not_onto_the_platform() -> None:
    """Folkestone: landed on the platform's lanes, which go nowhere, the
    planner never took the tunnel."""
    from core.map.ferries import _landing

    assert _landing(terminal(hidden_exit=False), np.array([5000.0, 540.0])) == [2]


def test_a_way_out_by_a_hidden_road_is_not_the_gps_s() -> None:
    from core.map.ferries import _landing

    assert _landing(terminal(hidden_exit=True), np.array([5000.0, 540.0])) == [3]


def test_a_line_to_a_port_not_on_the_map_is_left_out() -> None:
    assert all(link.end != "north" for link in link_ferries(COASTS, PORTS, LINES))


def test_a_ferry_never_costs_less_than_the_straight_line() -> None:
    """The planner's estimate is the straight line; a line priced below it
    would make that estimate overshoot."""
    short = [FerryLine("west", "east", 1.0)]
    (link,) = link_ferries(COASTS, PORTS, short)
    assert link.cost >= 5000.0


def test_links_come_back_as_they_went(tmp_path) -> None:
    links = link_ferries(COASTS, PORTS, LINES)
    save_links(tmp_path, links)
    assert load_links(tmp_path) == {1: tuple(links)}


def test_a_depot_over_the_sea_is_reached_by_the_ferry() -> None:
    """Göteborg to Dover: with roads alone there was no route at all."""
    depot = CompanyPlace("lkwlog", "dover", (5000.0, 1490.0), ((5000.0, 1490.0),), ())
    route = reroute_from(with_links(COASTS), (0,), (), depot)
    assert route.lanes == (0, 1, 2, 3)
    assert route.steps == (ONWARD, FERRY, ONWARD)
    assert route.ferry == 1


def test_a_route_is_driven_as_far_as_its_ferry() -> None:
    """To the port: the quay the ship is boarded from is the port's, and
    out on it the truck hit a bollard in Kristiansand."""
    route = Route(with_links(COASTS), (0, 1, 2, 3), (ONWARD, FERRY, ONWARD))
    driven = route.to_ferry()
    assert driven.lanes == (0,)
    assert driven.steps == ()
    assert driven.points()[-1, 2] == pytest.approx(500.0)
    assert Route(COASTS, (0, 1), (ONWARD,)).to_ferry().lanes == (0, 1), "no ferry: all of it"


def port_approach(booth: bool) -> RoadNetwork:
    """2 km of road, 100 m of booth lane (or road), 400 m to the quay, a ship."""
    from core.map import Lane, LaneGraph
    from core.map.sector import Signal

    graph = LaneGraph()
    pieces = [((0, 0), (0, 2000)), ((0, 2000), (0, 2100)), ((0, 2100), (0, 2500))]
    for index, (a, b) in enumerate(pieces):
        signal = Signal("barrier_manual", "b_gen_sh_cm") if booth and index == 1 else None
        graph.lanes.append(
            Lane(points=polyline(a, b), source="road", owner=f"{index + 1:016X}", signal=signal)
        )
    graph.lanes.append(Lane(points=polyline((5000, 0), (5000, 500)), source="road", owner="9"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    return RoadNetwork.compile(graph)


def test_a_drive_to_a_ship_ends_before_the_port_s_booths() -> None:
    """At Calais the truck caught a booth's island; past the booths a port is
    queues and ramps the map does not show."""
    net = port_approach(booth=True)
    route = Route(net, (0, 1, 2, 3), (ONWARD, ONWARD, FERRY)).to_ferry()
    assert route.lanes == (0,)


def test_with_no_booths_it_ends_short_of_the_ship() -> None:
    net = port_approach(booth=False)
    route = Route(net, (0, 1, 2, 3), (ONWARD, ONWARD, FERRY)).to_ferry()
    assert route.lanes == (0, 1)


def into_a_port() -> RoadNetwork:
    """A road into a port's junction, three lanes of it on to the end of a
    quay its ship is boarded from, and a road out of it."""
    from core.map import Lane, LaneGraph
    from core.map.ferries import FerryLink

    graph = LaneGraph()
    pieces = [
        (((0, 0), (0, 500)), "road", "1"),
        (((0, 500), (0, 600)), "prefab", "F"),
        (((0, 600), (0, 700)), "prefab", "F"),
        (((0, 700), (0, 760)), "prefab", "F"),
        (((10, 500), (10, 0)), "road", "2"),
    ]
    for (a, b), source, owner in pieces:
        graph.lanes.append(Lane(points=polyline(a, b), source=source, owner=f"{owner:>016}"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    graph.connect(2, 3)
    net = RoadNetwork.compile(graph)
    link = FerryLink(
        board=3, land=4, cost=1000.0, start="kristiansand", end="hirtshals", train=False
    )
    return replace(net, ferry_links={3: (link,)})


def test_a_way_into_a_port_without_its_crossing_ends_where_the_port_begins() -> None:
    """Into Kristiansand's port the minimap's red line, followed the
    straightest way on, ran over the quay to its end, and the truck hit a
    bollard by the ship."""
    net = into_a_port()
    route = Route(net, (0, 1, 2, 3), (ONWARD, ONWARD, ONWARD)).to_ferry()
    assert route.lanes == (0,)


def test_a_way_out_of_a_port_off_a_ship_is_driven_all_of_it() -> None:
    net = into_a_port()
    route = Route(net, (1, 2, 3), (ONWARD, ONWARD))
    assert route.to_ferry().lanes == (1, 2, 3)


def test_a_drive_to_a_ship_ends_where_its_port_begins_if_that_is_sooner() -> None:
    """100 m short of the ship in Kristiansand was out on the quay, 143 m
    into the port, and the truck hit a bollard there."""
    net = into_a_port()
    route = Route(net, (0, 1, 2, 3, 4), (ONWARD, ONWARD, ONWARD, FERRY)).to_ferry()
    assert route.lanes == (0,)


def test_a_drive_to_a_ship_ends_where_the_port_s_own_roads_begin() -> None:
    """At Calais the passport booths have no barrier the map knows: the truck
    drove into them and hit an island. The port's own roads have none of the
    game's traffic; the driver takes them by hand (12 November)."""
    from core.map import Lane, LaneGraph

    graph = LaneGraph()
    pieces = [
        ((0, 0), (0, 2000), True),
        ((0, 2000), (0, 2600), False),
        ((0, 2600), (0, 2700), False),
    ]
    for index, (a, b, traffic) in enumerate(pieces):
        graph.lanes.append(
            Lane(
                points=polyline(a, b), source="road", owner=f"{index + 1:016X}", ai_vehicles=traffic
            )
        )
    graph.lanes.append(Lane(points=polyline((5000, 0), (5000, 500)), source="road", owner="9"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    net = RoadNetwork.compile(graph)
    route = Route(net, (0, 1, 2, 3), (ONWARD, ONWARD, FERRY)).to_ferry()
    assert route.lanes == (0,)
