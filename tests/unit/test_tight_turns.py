"""Junction turns too tight for the rig: found, shaped and driven."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pytest

from core.localization.path import ReferencePath
from core.map import Lane, LaneGraph, RoadNetwork
from core.planning import tight_turns
from core.planning.route import ONWARD, Route
from core.planning.tight_turns import between, shaped_turn, splice, stretches
from tests.unit.lane_networks import polyline


def arc(radius: float, count: int = 20) -> np.ndarray:
    """From (0, 0) heading +z, a quarter turn onto +x at (radius, radius)."""
    angles = np.linspace(math.pi, math.pi / 2, count)
    return np.column_stack(
        [radius + radius * np.cos(angles), np.zeros(count), radius * np.sin(angles)]
    )


def t_junction(radius: float, far_lane_against: bool = False) -> Route:
    """A road up to a main road, turning onto it along a junction's lane of
    `radius`; the main road's two lanes across the junction, the far one going
    the same way as the near, or `far_lane_against` it: coming out of the
    road turned into."""
    graph = LaneGraph()

    def add(points: np.ndarray, source: str) -> int:
        # A junction's lanes are all its prefab's; each road's its own.
        owner = "00000000000000FF" if source == "prefab" else f"{len(graph.lanes) + 1:016X}"
        graph.lanes.append(Lane(points=points, source=source, owner=owner))
        return len(graph.lanes) - 1

    stem = add(polyline((0, -150), (0, 0)), "road")
    turn = add(arc(radius), "prefab")
    out = add(polyline((radius, radius), (150, radius)), "road")
    across = add(polyline((-40, radius), (radius, radius)), "prefab")
    far = ((150, radius + 3.5), (-40, radius + 3.5))
    add(polyline(*(far if far_lane_against else far[::-1])), "prefab")
    graph.connect(stem, turn)
    graph.connect(turn, out)
    graph.connect(across, out)
    network = RoadNetwork.compile(graph)
    return Route(network, (stem, turn, out), (ONWARD, ONWARD))


@pytest.fixture(scope="module")
def shaped() -> tuple[Route, np.ndarray, np.ndarray | None]:
    route = t_junction(9.0)
    (stretch,) = stretches(route)
    points = route.points()[:, [0, 2]]
    first, last = between(points, stretch.approach[0], stretch.exit_[-1])
    own = points[first : last + 1]
    return (
        route,
        own,
        shaped_turn(route.network, stretch.approach, stretch.exit_, own, stretch.junction),
    )


def stray(route: Route, line: np.ndarray) -> float:
    (stretch,) = stretches(route)
    about = np.vstack([stretch.approach, stretch.exit_])
    points = route.points()[:, [0, 2]]
    first, last = between(points, stretch.approach[0], stretch.exit_[-1])
    outside, origin = tight_turns._ground(
        route.network, about, points[first : last + 1], stretch.junction
    )
    return tight_turns._stray(
        ReferencePath(np.asarray(line, float), spacing=1.0),
        stretch.approach,
        outside,
        origin,
        tight_turns.DESIGN_TRAILER_M,
    )


def test_a_junction_turn_on_a_9_m_radius_is_tight_and_one_on_60_m_is_not() -> None:
    """Into the depot road at Duisburg: 9 m, and a car carrier's trailer over
    the kerb on the inside."""
    tight = t_junction(9.0)
    assert [(s.begins, s.key) for s in stretches(tight)] == [(0, tight.lanes)]
    assert tight_turns.tight_turns(t_junction(60.0)) == []


def test_the_shaped_line_keeps_the_rig_on_the_junction_s_ground(shaped) -> None:
    route, own, line = shaped
    assert line is not None
    assert stray(route, own) > 2.0, "the lane's middle: the trailer far over the inside"
    assert stray(route, line) < stray(route, own) - 1.5


def test_the_shaped_line_goes_on_into_the_junction_before_turning(shaped) -> None:
    """Deeper than the lane, as a driver takes such a turn."""
    _, own, line = shaped
    assert line is not None

    def furthest_on(points: np.ndarray) -> float:
        """How far up the stem the line is still heading straight on."""
        steps = np.diff(points, axis=0)
        straight = (np.abs(steps[:, 0]) < 0.05 * np.hypot(*steps.T)) & (points[:-1, 1] > -50)
        return float(points[:-1][straight][:, 1].max())

    assert furthest_on(line) > furthest_on(own) + 2.0


def test_a_turn_is_not_shaped_over_the_lane_coming_out_of_the_road_turned_into() -> None:
    """In Norway, at night, the line went on into the junction and swung over
    the mouth of the oncoming lane of the road it turned into, where a lorry
    stood waiting, and hit it."""
    route = t_junction(9.0, far_lane_against=True)
    (stretch,) = stretches(route)
    points = route.points()[:, [0, 2]]
    first, last = between(points, stretch.approach[0], stretch.exit_[-1])
    own = points[first : last + 1]
    line = shaped_turn(route.network, stretch.approach, stretch.exit_, own, stretch.junction)
    if line is not None:
        assert float(line[:, 1].max()) < 9.0 + 3.5 - 2.0, "over the oncoming lane"


def test_the_shaped_line_starts_and_ends_on_the_route(shaped) -> None:
    _, own, line = shaped
    assert line is not None
    assert np.allclose(line[0], own[0])
    assert np.allclose(line[-1], own[-1])


def test_splicing_puts_the_line_in_place_of_its_stretch_only() -> None:
    points = np.column_stack([np.zeros(101), np.arange(101.0)])
    line = np.column_stack([np.array([0.0, 1.0, 1.0, 0.0]), np.array([40.0, 45.0, 55.0, 60.0])])
    spliced = splice(points, line)
    assert np.allclose(spliced[:40], points[:40])
    assert np.allclose(spliced[40:44], line)
    assert np.allclose(spliced[44:], points[61:])


def test_route_path_drives_a_shaped_turn_and_without_one_the_lane() -> None:
    from tools.follow import route_path

    route = t_junction(9.0, far_lane_against=True)
    (stretch,) = stretches(route)
    line = np.vstack(
        [stretch.approach, [[-1.0, 1.5], [-1.5, 3.0]], stretch.exit_[2:]]
    )  # any line from the stretch's start to its end
    plain = route_path(route, 1.0)
    driven = route_path(route, 1.0, shapes={stretch.key: line})
    assert route_path(route, 1.0, shapes={stretch.key: None}).length == pytest.approx(plain.length)
    near = driven.nearest(-1.5, 3.0)
    assert math.hypot(near.x + 1.5, near.z - 3.0) < 0.4
    lane = plain.nearest(-1.5, 3.0)
    assert math.hypot(lane.x + 1.5, lane.z - 3.0) > 1.5


def test_the_gps_worker_s_turns_are_shaped_beside_it_and_kept() -> None:
    from tools.follow import TurnShaping

    route = t_junction(9.0)
    shaping = TurnShaping(route.network)
    shaping.shape(route, route.points()[:, [0, 2]], at=-1)
    deadline = time.perf_counter() + 30.0
    while not shaping.newly_shaped(at=0):
        assert time.perf_counter() < deadline, "never shaped"
        time.sleep(0.05)
    assert shaping.lines[route.lanes] is not None
    shaping.shape(route, route.points()[:, [0, 2]], at=-1)
    time.sleep(0.2)
    assert not shaping.newly_shaped(at=0), "a turn shaped is not shaped again"


def test_a_turn_the_truck_is_already_on_the_way_into_is_left_as_it_is() -> None:
    from tools.follow import TurnShaping

    route = t_junction(9.0)
    shaping = TurnShaping(route.network)
    shaping.shape(route, route.points()[:, [0, 2]], at=0)
    time.sleep(0.2)
    assert not shaping.newly_shaped(at=0)
    assert route.lanes not in shaping.lines


def test_ground_is_the_junction_and_the_roads_going_the_way_the_turn_goes() -> None:
    """A lane coming the other way beyond the junction is no ground: with
    every lane for ground, the search cut a corner over an oncoming
    carriageway to the road beyond."""
    graph = LaneGraph()
    for points, owner in (
        (polyline((0, -60), (0, 0)), "01"),  # the way in, north
        (polyline((4.0, 0), (4.0, -60)), "02"),  # beside it, coming south
        (polyline((-30, 4.0), (30, 4.0)), "FF"),  # the junction's, west
    ):
        graph.lanes.append(Lane(points=points, source="road", owner=f"{owner:>016}"))
    network = RoadNetwork.compile(graph)
    own = np.column_stack([np.zeros(30), np.linspace(-60.0, 0.0, 30)])
    outside, origin = tight_turns._ground(network, own, own, junction=(2,))

    def off(x: float, z: float) -> float:
        return tight_turns._off(outside, origin, np.array([[x, z]]))

    assert off(0.0, -30.0) == 0.0
    assert off(4.0, -30.0) > 1.0, "the oncoming lane"
    assert off(-20.0, 4.0) == 0.0, "the junction's, whichever way"


def right_t_junction(radius: float, gap: float = 3.5) -> Route:
    """A road up to a main road, turning right onto it on a junction lane of
    `radius`; the main road's other lane, `gap` beyond, coming out of it."""
    graph = LaneGraph()

    def add(points: np.ndarray, source: str) -> int:
        owner = "00000000000000FF" if source == "prefab" else f"{len(graph.lanes) + 1:016X}"
        graph.lanes.append(Lane(points=points, source=source, owner=owner))
        return len(graph.lanes) - 1

    turning = arc(radius)
    turning[:, 0] *= -1.0  # +x is the left: towards -x, a right turn
    stem = add(polyline((0, -150), (0, 0)), "road")
    turn = add(turning, "prefab")
    out = add(polyline((-radius, radius), (-150, radius)), "road")
    add(polyline((-150, radius + gap), (40, radius + gap)), "prefab")
    graph.connect(stem, turn)
    graph.connect(turn, out)
    return Route(RoadNetwork.compile(graph), (stem, turn, out), (ONWARD, ONWARD))


def test_a_right_turn_over_the_other_way_s_lane_is_kept_in_its_own() -> None:
    """Twice on the way to Bergen, round one right turn as tight as a hairpin,
    the cab came up to the mouth of the lane coming out of the road turned
    into, where vehicles stood waiting: the driver's rule (11 November), keep
    to the truck's own lane round it."""
    route = right_t_junction(14.0)
    (stretch,) = stretches(route)
    points = route.points()[:, [0, 2]]
    first, last = between(points, stretch.approach[0], stretch.exit_[-1])
    own = points[first : last + 1]
    assert stray(route, own) >= tight_turns.OVER_AGAINST_M, "the lane's own goes over it"
    line = shaped_turn(route.network, stretch.approach, stretch.exit_, own, stretch.junction)
    assert line is not None
    assert stray(route, line) < tight_turns.OVER_AGAINST_M, "kept off it"
    assert np.allclose(line[0], own[0], atol=0.5) and np.allclose(line[-1], own[-1], atol=0.5)


def test_only_the_hairpin_of_a_long_run_of_junction_lanes_is_a_tight_turn() -> None:
    """Near Calais a 644 m run of a junction's lanes, with one tight bend,
    was all of it slowed to 8 km/h, on the straight."""
    from core.planning.tight_turns import tight_turn_spans

    graph = LaneGraph()
    bend = arc(9.0)
    bend[:, 2] += 400.0
    graph.lanes.append(
        Lane(points=polyline((0, -150), (0, 0)), source="road", owner="0000000000000001")
    )
    graph.lanes.append(
        Lane(points=polyline((0, 0), (0, 400)), source="prefab", owner="00000000000000FF")
    )
    graph.lanes.append(Lane(points=bend, source="prefab", owner="00000000000000FF"))
    graph.lanes.append(
        Lane(points=polyline((9, 409), (150, 409)), source="road", owner="0000000000000002")
    )
    for a, b in ((0, 1), (1, 2), (2, 3)):
        graph.connect(a, b)
    route = Route(RoadNetwork.compile(graph), (0, 1, 2, 3), (ONWARD, ONWARD, ONWARD))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=1.0)
    ((start, end, _),) = tight_turn_spans(route, path)
    assert start > 150.0 + 350.0, "not from the junction's first lane"
    assert end - start < 50.0


def test_two_tight_turns_far_apart_are_each_found_on_its_bend() -> None:
    """Looked for within 3 km of the last, a turn further on was placed on
    another part of the path, km off; and by Dover a span's end, looked for
    on the path, was where the route came back past it 500 m on: the truck
    crept at 8 km/h all that way, on an empty road."""
    from core.planning.tight_turns import tight_turn_spans

    radius = 8.0
    first = arc(radius)
    angles = np.linspace(-np.pi / 2, 0.0, 20)  # heading +x, then on towards +z
    second = np.column_stack(
        [5000.0 + radius * np.cos(angles), np.zeros(20), 2 * radius + radius * np.sin(angles)]
    )
    graph = LaneGraph()
    pieces = [
        (polyline((0, -150), (0, 0)), "road", "0000000000000001"),
        (first, "prefab", "00000000000000F1"),
        (polyline((radius, radius), (5000, radius)), "road", "0000000000000002"),
        (second, "prefab", "00000000000000F2"),
        (polyline((5000 + radius, 2 * radius), (5000 + radius, 300)), "road", "0000000000000003"),
    ]
    for points, source, owner in pieces:
        graph.lanes.append(Lane(points=points, source=source, owner=owner))
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        graph.connect(a, b)
    route = Route(RoadNetwork.compile(graph), (0, 1, 2, 3, 4), (ONWARD,) * 4)
    path = ReferencePath(route.points()[:, [0, 2]], spacing=1.0)
    spans = tight_turn_spans(route, path)
    assert len(spans) == 2
    for start, end, points in spans:
        at = path.sample_at(start)
        assert np.hypot(at.x - points[0][0], at.z - points[0][1]) < 3.0, "on its bend"
        assert end - start < 40.0
    assert spans[1][0] > 5000.0


def test_a_line_pulled_in_never_hooks_back_where_lanes_join() -> None:
    """By Birmingham the line pulled in hooked back where a junction's lanes
    joined, points on top of one another; following it the truck put full
    lock on and went over a pavement."""
    turning = arc(10.0)
    own = np.vstack(
        [
            np.column_stack([np.zeros(40), np.linspace(-80.0, 0.0, 40)]),
            turning[:, [0, 2]],
            turning[-1:, [0, 2]],  # a join: the same point twice
            turning[-1:, [0, 2]] + [0.05, 0.0],  # and a step of 5 cm
            np.column_stack([np.linspace(10.2, 90.0, 40), np.full(40, 10.0)]),
        ]
    )
    pulled = tight_turns._pulled_in(own, 1.0, 80.0, 96.0)
    steps = np.diff(pulled, axis=0)
    assert np.all(np.sum(steps[1:] * steps[:-1], axis=1) > 0.0), "never back on itself"
    heading = np.unwrap(np.arctan2(steps[:, 0], steps[:, 1]))
    assert np.degrees(np.max(np.abs(np.diff(heading)))) < 12.0


def test_which_side_traffic_keeps_to_is_read_off_the_roads_not_the_junction() -> None:
    """A junction's lanes are never marked: asked of one, every turn in
    Britain was taken for one to the right-hand side."""
    route = right_t_junction(12.0)
    net = route.network
    marked = np.zeros(len(net), dtype=bool)
    sources = np.asarray(net.lane_source)
    marked[sources == 0] = True  # the roads keep left; the junction's lanes say nothing
    keeping_left = replace(net, lane_left_hand=marked)
    (stretch,) = stretches(route)
    points = route.points()[:, [0, 2]]
    first, last = between(points, stretch.approach[0], stretch.exit_[-1])
    line = points[first : last + 1]
    assert tight_turns._keeps_left(keeping_left, line)
    assert not tight_turns._to_exit_side(keeping_left, line, stretch.junction), (
        "a right turn where traffic keeps left crosses the other way"
    )


def test_only_lanes_near_the_way_out_count_as_coming_out_of_it() -> None:
    """Near Verona the route bent before its turn, and the way in's own
    oncoming lane, heading against the way out, was taken for one coming out
    of the road turned into: every line went over it in simulation, the line
    was pulled in its furthest, and the trailer cut into a bollard."""
    graph = LaneGraph()
    for points, owner in (
        (polyline((-60, 0), (0, 0)), "01"),  # the way in, east
        (polyline((0, 4.0), (-60, 4.0)), "02"),  # beside it, west
        (polyline((60, 43.5), (0, 43.5)), "03"),  # beside the way out, west
    ):
        graph.lanes.append(Lane(points=points, source="road", owner=f"{owner:>016}"))
    network = RoadNetwork.compile(graph)
    exit_ = np.column_stack([np.linspace(0.0, 60.0, 30), np.full(30, 40.0)])
    own = np.vstack([np.column_stack([np.linspace(-60.0, 0.0, 30), np.zeros(30)]), exit_])
    outside, origin = tight_turns._ground(network, own, exit_, exit_=exit_)

    def off(x: float, z: float) -> float:
        return tight_turns._off(outside, origin, np.array([[x, z]]))

    assert off(-30.0, 4.0) < tight_turns.OVER_AGAINST_M, "the way in's own"
    assert off(30.0, 43.5) >= tight_turns.OVER_AGAINST_M, "coming out of the way out"
