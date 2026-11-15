"""Planning a route on the lane graph, and drawing it as a line to follow."""

from __future__ import annotations

import numpy as np
import pytest

from core.planning import route as routing
from core.planning.route import CHANGE, ONWARD, RouteError, plan
from tests.unit.lane_networks import network, polyline


def test_a_route_follows_the_lanes_in_order() -> None:
    lanes = [polyline((0, 0), (0, 100)), polyline((0, 100), (0, 200)), polyline((0, 200), (0, 300))]
    net = network(lanes, successors=[(0, 1), (1, 2)])

    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 290.0)])
    assert route.lanes == (0, 1, 2)
    assert route.steps == (ONWARD, ONWARD)
    assert route.length == pytest.approx(300.0, abs=0.5)


def test_it_changes_lane_when_only_the_other_lane_leads_there() -> None:
    inner, outer = polyline((0, 0), (0, 200)), polyline((-4.5, 0), (-4.5, 200))
    exit_ramp = polyline((-4.5, 200), (-30, 260))
    net = network([inner, outer, exit_ramp], successors=[(1, 2)], changes=[(0, 1)])

    route = plan(net, start=(0.0, 5.0, 0.0), through=[(-30.0, 258.0)])
    assert route.lanes == (0, 1, 2)
    assert route.steps == (CHANGE, ONWARD)
    assert route.lane_changes == 1


def test_a_lane_change_is_drawn_as_a_move_across_not_a_step() -> None:
    """The lane sequence says 'jump 4.5 m'; a controller must not be told that."""
    inner, outer = polyline((0, 0), (0, 200)), polyline((-4.5, 0), (-4.5, 200))
    exit_ramp = polyline((-4.5, 200), (-30, 260))
    net = network([inner, outer, exit_ramp], successors=[(1, 2)], changes=[(0, 1)])
    points = plan(net, start=(0.0, 5.0, 0.0), through=[(-30.0, 258.0)]).points()

    gaps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 2]))
    assert gaps.max() < 2.1
    assert points[0, 0] == pytest.approx(0.0)
    # Across within the lane-change length, and on the new lane after it.
    done = routing.LANE_CHANGE_LENGTH_M + 1.0
    on_outer = points[(points[:, 2] > done) & (points[:, 2] < 200)]
    assert len(on_outer) > 10
    assert on_outer[:, 0] == pytest.approx(-4.5, abs=1e-6)


def test_the_move_across_leaves_and_joins_its_lanes_tangentially() -> None:
    a, b = polyline((0, 0), (0, 100)), polyline((-4.5, 0), (-4.5, 100))
    eased = routing._ease_across(a, b, 50.0)
    x = eased[:, 0]
    assert x[0] == pytest.approx(0.0) and x[-1] == pytest.approx(-4.5)
    # A smoothstep has zero slope at both ends of the move: the first and last
    # steps across are a small fraction of the steepest one in the middle.
    steps = np.abs(np.diff(x[:26]))
    assert steps[0] < 0.1 * steps.max()
    assert steps[-1] < 0.1 * steps.max()
    assert np.all(np.diff(x) <= 1e-9)


def test_a_waypoint_in_a_junction_is_reached_ahead_not_by_going_round() -> None:
    """The bug: the lane nearest a waypoint can be a branch the other way.

    Here the waypoint sits 1.0 m from lane 3, which only leads away, and
    1.5 m from lane 1, which is straight ahead. Nearest-first went the long
    way round the loop to reach lane 3.
    """
    approach = polyline((0, 0), (0, 100))
    ahead = polyline((0, 100), (0, 200))
    around = polyline((0, 200), (200, 200), (200, -200), (-2.5, -200), (-2.5, 150))
    other_way = polyline((-2.5, 150), (-2.5, 250))
    net = network([approach, ahead, around, other_way], successors=[(0, 1), (1, 2), (2, 3)])

    route = plan(net, start=(0.0, 10.0, 0.0), through=[(-1.5, 180.0)])
    assert route.lanes == (0, 1)


def test_a_start_off_the_network_is_refused() -> None:
    net = network([polyline((0, 0), (0, 100))])
    with pytest.raises(RouteError, match="not on the network"):
        plan(net, start=(500.0, 500.0, 0.0), through=[(0.0, 90.0)])


def test_a_destination_nothing_leads_to_is_refused() -> None:
    """Two lanes that do not join: a road that ends, for instance."""
    net = network([polyline((0, 0), (0, 100)), polyline((0, 200), (0, 300))])
    with pytest.raises(RouteError, match="nothing leads"):
        plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 290.0)])


def test_a_saved_route_comes_back_as_its_line(tmp_path) -> None:
    net = network([polyline((0, 0), (0, 100)), polyline((0, 100), (0, 200))], successors=[(0, 1)])
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 190.0)])
    routing.save(route, tmp_path / "route.npz", note="test")

    assert routing.load_points(tmp_path / "route.npz") == pytest.approx(route.points())


def test_a_file_that_is_not_a_route_is_refused(tmp_path) -> None:
    np.savez(tmp_path / "other.npz", points=np.zeros((3, 3)))
    with pytest.raises(RouteError, match="not a saved route"):
        routing.load_points(tmp_path / "other.npz")


def test_a_route_counts_what_the_stack_cannot_see_yet() -> None:
    """Traffic lights and turns come from the map, so they can be counted before driving.

    The taught loop the truck had already driven autonomously turned out to
    have six traffic lights on it, the first 207 m in.
    """
    from core.map import Lane, LaneGraph, RoadNetwork, Signal

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 200)), source="road", owner="00A1"))
    graph.lanes.append(
        Lane(
            points=polyline((0, 200), (0, 220)),
            source="prefab",
            owner="00B1",
            semaphore=2,
            signal=Signal("traffic_light_major", "cr_2x1"),
            blinker="RightBlinker",
        )
    )
    graph.lanes.append(Lane(points=polyline((0, 220), (0, 400)), source="road", owner="00A2"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    net = RoadNetwork.compile(graph)

    hazards = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)]).hazards()
    assert hazards.junctions == 1
    assert hazards.traffic_lights == 1
    assert hazards.first_traffic_light_m == pytest.approx(200.0, abs=0.5)
    assert hazards.right_turns == 1
    assert hazards.left_turns == 0


def test_the_road_ahead_takes_the_straightest_way_on() -> None:
    from core.planning.route import plan_ahead

    approach = polyline((0, 0), (0, 100))
    turn = polyline((0, 100), (-60, 140))
    straight = polyline((0, 100), (2, 200))
    net = network([approach, turn, straight], successors=[(0, 1), (0, 2)])

    route = plan_ahead(net, (0.0, 10.0, 0.0), 150.0)
    assert route.lanes == (0, 2)


def test_the_road_ahead_ends_before_a_traffic_light() -> None:
    """Nothing in the stack can see one, so a route for a first drive stops short."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning.route import plan_ahead

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 100)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(points=polyline((0, 100), (0, 120)), source="prefab", owner="0B", semaphore=0)
    )
    graph.connect(0, 1)
    net = RoadNetwork.compile(graph)

    assert plan_ahead(net, (0.0, 10.0, 0.0), 500.0).lanes == (0,)
    assert plan_ahead(net, (0.0, 10.0, 0.0), 500.0, stop_before_lights=False).lanes == (0, 1)


def test_the_road_ahead_ends_where_the_road_does() -> None:
    from core.planning.route import plan_ahead

    net = network([polyline((0, 0), (0, 100))])
    route = plan_ahead(net, (0.0, 10.0, 0.0), 1000.0)
    assert route.lanes == (0,)
    assert route.length == pytest.approx(100.0, abs=0.5)


def _junction_with(signal) -> object:
    """A road into a junction lane governed by `signal`, and on out of it."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 200)), source="road", owner="00A1"))
    graph.lanes.append(
        Lane(
            points=polyline((0, 200), (0, 202)),
            source="prefab",
            owner="00B1",
            semaphore=1,
            signal=signal,
        )
    )
    graph.lanes.append(
        Lane(
            points=polyline((0, 202), (0, 220)),
            source="prefab",
            owner="00B1",
            semaphore=1,
            signal=signal,
        )
    )
    graph.lanes.append(Lane(points=polyline((0, 220), (0, 400)), source="road", owner="00A2"))
    for a, b in ((0, 1), (1, 2), (2, 3)):
        graph.connect(a, b)
    return RoadNetwork.compile(graph)


def test_a_light_stop_line_is_where_the_route_enters_its_lanes() -> None:
    from core.map import Signal

    light = Signal("traffic_light_major", "cr_2x1")
    net = _junction_with(light)
    lines = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)]).stop_lines()

    assert len(lines) == 1, "two lanes of one light are one stop"
    assert lines[0].position[[0, 2]] == pytest.approx([0.0, 200.0])
    assert lines[0].signal == light


def test_a_light_the_driver_says_the_road_has_not_is_no_stop(monkeypatch) -> None:
    """At the bridge in Geneva the map has a light 52 m after a real one,
    and the truck stopped at it every time; the road has none."""
    from core.map import Signal
    from core.planning import route as routing

    net = _junction_with(Signal("traffic_light_major", "cr_2x1"))
    monkeypatch.setattr(routing, "load_no_lights", lambda: np.array([[1.0, 201.0]]))
    assert plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)]).stop_lines() == []
    monkeypatch.setattr(routing, "load_no_lights", lambda: np.array([[0.0, 260.0]]))
    assert len(plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)]).stop_lines()) == 1


def test_a_barrier_is_not_a_light_to_stop_at() -> None:
    from core.map import Signal

    net = _junction_with(Signal("barrier_manual", "ab_toll"))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)])

    assert route.stop_lines() == []
    assert route.hazards().barriers == 1
    assert route.hazards().traffic_lights == 0


def test_a_light_of_unknown_kind_is_still_stopped_at() -> None:
    """A profile the definitions did not have: a light until shown otherwise."""
    from core.map import Signal

    net = _junction_with(Signal("", "cr_new"))
    lines = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)]).stop_lines()
    assert len(lines) == 1


def test_a_semaphore_the_junction_does_not_define_is_no_light() -> None:
    """In city-01 such a lane crossed a junction with no light on that approach."""
    net = _junction_with(None)
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)])
    assert route.stop_lines() == []
    assert route.hazards().traffic_lights == 0


def test_a_saved_route_keeps_its_stop_lines(tmp_path) -> None:
    from core.map import Signal

    light = Signal("traffic_light_minor", "cr_1x1", (1410, 180))
    route = plan(_junction_with(light), start=(0.0, 10.0, 0.0), through=[(0.0, 390.0)])
    routing.save(route, tmp_path / "route.npz")

    back = routing.load_stop_lines(tmp_path / "route.npz")
    assert len(back) == 1
    assert back[0].position == pytest.approx(route.stop_lines()[0].position)
    assert back[0].signal == light


def test_a_route_saved_before_stop_lines_has_none(tmp_path) -> None:
    np.savez(
        tmp_path / "old.npz",
        format=np.array(routing.ROUTE_FORMAT),
        points=np.zeros((3, 3)),
    )
    assert routing.load_stop_lines(tmp_path / "old.npz") == []


def test_a_hidden_road_is_not_taken_when_a_shown_one_goes_there() -> None:
    """In Milan the planner's shortcut was a road the game's map hides,
    closed off by yellow crosses; the GPS went round."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 100)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(points=polyline((0, 100), (0, 200)), source="road", owner="0B", hidden=True)
    )
    graph.lanes.append(
        Lane(points=polyline((0, 100), (60, 150), (0, 200)), source="road", owner="0C")
    )
    graph.lanes.append(Lane(points=polyline((0, 200), (0, 300)), source="road", owner="0D"))
    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        graph.connect(a, b)
    net = RoadNetwork.compile(graph)

    assert plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 290.0)]).lanes == (0, 2, 3)


def test_a_hidden_road_is_taken_when_it_is_the_only_way() -> None:
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(
        Lane(points=polyline((0, 0), (0, 100)), source="road", owner="0A", hidden=True)
    )
    graph.lanes.append(Lane(points=polyline((0, 100), (0, 200)), source="road", owner="0B"))
    graph.connect(0, 1)
    net = RoadNetwork.compile(graph)

    assert plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 190.0)]).lanes == (0, 1)


def test_a_toll_gate_is_found_to_be_slowed_for() -> None:
    from core.map import Signal

    route = plan(
        _junction_with(Signal("barrier_manual", "ab_toll")),
        start=(0.0, 10.0, 0.0),
        through=[(0.0, 390.0)],
    )
    gates = route.barriers()
    assert len(gates) == 1
    assert gates[0].position[[0, 2]] == pytest.approx([0.0, 200.0])


def test_a_road_without_traffic_is_not_taken_as_a_shortcut() -> None:
    """At a border the planner went through the customs yard, a road the
    game's traffic does not use; the GPS kept to the main road round it."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 100)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(points=polyline((0, 100), (0, 200)), source="road", owner="0B", ai_vehicles=False)
    )
    graph.lanes.append(
        Lane(points=polyline((0, 100), (40, 150), (0, 200)), source="road", owner="0C")
    )
    graph.lanes.append(Lane(points=polyline((0, 200), (0, 300)), source="road", owner="0D"))
    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        graph.connect(a, b)
    net = RoadNetwork.compile(graph)

    assert plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 290.0)]).lanes == (0, 2, 3)


def _with_quiet_detour(detour: list[tuple[float, float]]) -> object:
    """1 km north, then a road without traffic along `detour`, then 1 km on."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 1000)), source="road", owner="0A"))
    graph.lanes.append(Lane(points=polyline(*detour), source="road", owner="0B", ai_vehicles=False))
    end = detour[-1]
    graph.lanes.append(
        Lane(points=polyline(end, (end[0], end[1] + 1000)), source="road", owner="0C")
    )
    graph.connect(0, 1)
    graph.connect(1, 2)
    return RoadNetwork.compile(graph)


def test_a_border_s_customs_stretch_is_found_and_kept_as_mapped() -> None:
    """Italy to Switzerland: the customs lanes, to be driven slowly on the
    mapped road -- straight across instead, the truck hit the plaza."""
    net = _with_quiet_detour([(0, 1000), (40, 1100), (40, 1250), (0, 1350)])
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 2340.0)])
    assert route.quiet_stretches() == [(1, 1)]
    line = route.points()
    assert np.abs(line[:, 0]).max() == pytest.approx(40.0, abs=0.5), "round, as mapped"


def test_a_quiet_loop_is_left_as_mapped() -> None:
    """A stretch that turns back on itself is not a way through."""
    net = _with_quiet_detour([(0, 1000), (60, 1100), (60, 1000), (10, 1030)])
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(10.0, 2020.0)])
    assert route.quiet_stretches() == []


def test_a_customs_stretch_just_ahead_of_the_start_is_still_driven_straight() -> None:
    """Started from a save 170 m short of the border, the stretch was taken
    for a yard at the start."""
    net = _with_quiet_detour([(0, 1000), (40, 1100), (40, 1250), (0, 1350)])
    route = plan(net, start=(0.0, 900.0, 0.0), through=[(0.0, 2340.0)])
    assert len(route.quiet_stretches()) == 1


def test_a_route_that_starts_in_a_yard_does_not_straighten_it() -> None:
    net = _with_quiet_detour([(0, 1000), (40, 1100), (40, 1250), (0, 1350)])
    route = plan(net, start=(40.0, 1150.0, 0.0), through=[(0.0, 2340.0)])
    assert route.quiet_stretches() == []


def test_a_lane_change_inside_a_customs_stretch_does_not_count_twice() -> None:
    """At the border the route changed lanes in the customs stretch, and the
    two lanes' lengths together made it look like a detour."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 1000)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(points=polyline((0, 1000), (0, 1100)), source="road", owner="0B", ai_vehicles=False)
    )
    graph.lanes.append(
        Lane(points=polyline((4, 1000), (4, 1100)), source="road", owner="0B", ai_vehicles=False)
    )
    graph.lanes.append(
        Lane(points=polyline((4, 1100), (4, 1300)), source="road", owner="0C", ai_vehicles=False)
    )
    graph.lanes.append(Lane(points=polyline((4, 1300), (4, 2300)), source="road", owner="0D"))
    graph.connect(0, 1)
    graph.lane_changes[1].append(2)
    graph.lane_changes[2].append(1)
    graph.connect(2, 3)
    graph.connect(3, 4)
    route = plan(RoadNetwork.compile(graph), start=(0.0, 10.0, 0.0), through=[(4.0, 2290.0)])
    assert CHANGE in route.steps
    assert len(route.quiet_stretches()) == 1


@pytest.mark.parametrize("offset", [2.27, -2.27])
def test_a_customs_stretch_s_lanes_are_drawn_further_from_the_centre(offset: float) -> None:
    """Italian-Swiss customs: booths on islands on the road's centre line, and
    in the middle of the left lane as mapped the truck was over one."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning.route import CUSTOMS_SPREAD_M

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 1000)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(
            points=polyline((0, 1000), (0, 1200)),
            source="road",
            owner="0B",
            offset=offset,
            ai_vehicles=False,
        )
    )
    graph.lanes.append(Lane(points=polyline((0, 1200), (0, 2200)), source="road", owner="0C"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    route = plan(RoadNetwork.compile(graph), start=(0.0, 10.0, 0.0), through=[(0.0, 2190.0)])
    line = route.points()
    middle = line[(line[:, 2] > 1080) & (line[:, 2] < 1120)]
    # Heading north, left is +x: a left lane (positive offset) moves further left.
    assert middle[:, 0].mean() == pytest.approx(np.sign(offset) * CUSTOMS_SPREAD_M, abs=0.05)
    assert np.abs(line[line[:, 2] < 990, 0]).max() < 0.01, "not before the stretch"
    ends = line[(line[:, 2] > 999) & (line[:, 2] < 1003)]
    assert np.abs(ends[:, 0]).max() < 0.4, "eased in, not a step"


def test_a_taught_crossing_replaces_the_customs_stretch(monkeypatch) -> None:
    """At the Italian-Swiss border the mapped customs road led into a booth; the
    driver's own line through is taken instead, joined on smoothly."""
    from core.planning import route as routing

    net = _with_quiet_detour([(0, 1000), (40, 1100), (40, 1250), (0, 1350)])
    driven = np.column_stack([np.full(120, -5.0), np.zeros(120), np.linspace(1000, 1350, 120)])
    crossing = routing.Crossing(np.array([0.0, 1000.0]), np.array([0.0, 1350.0]), driven)
    monkeypatch.setattr(routing, "load_crossings", lambda: (crossing,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 2340.0)])
    line = route.points()
    middle = line[(line[:, 2] > 1050) & (line[:, 2] < 1300)]
    assert middle[:, 0] == pytest.approx(-5.0, abs=0.01), "the driven line"
    steps = np.hypot(*np.diff(line[:, [0, 2]], axis=0).T)
    assert steps.max() < 4.0, "joined on, not jumped to"


def test_a_crossing_taught_elsewhere_is_not_used(monkeypatch) -> None:
    from core.planning import route as routing

    net = _with_quiet_detour([(0, 1000), (40, 1100), (40, 1250), (0, 1350)])
    elsewhere = routing.Crossing(
        np.array([500.0, 500.0]), np.array([600.0, 600.0]), np.zeros((5, 3))
    )
    monkeypatch.setattr(routing, "load_crossings", lambda: (elsewhere,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 2340.0)])
    assert route.taught_crossings() == []


def test_on_a_long_lane_the_move_across_comes_late() -> None:
    """Moved across at the lane's start, the truck changed into an exit lane
    400 m before the exit; it finishes 100 m before the lane's end instead."""
    a, b = polyline((0, 0), (0, 600)), polyline((-4.5, 0), (-4.5, 600))
    eased = routing._ease_across(a, b, routing.LANE_CHANGE_LENGTH_M)
    begins = routing.change_begins(600.0)
    assert begins == pytest.approx(600.0 - routing.LANE_CHANGE_FINISH_M - 120.0)
    z, x = eased[:, 2], eased[:, 0]
    assert np.all(x[z <= begins] == pytest.approx(0.0))
    assert np.all(x[z >= 600.0 - routing.LANE_CHANGE_FINISH_M] == pytest.approx(-4.5))


def test_a_lane_change_is_made_on_a_long_lane_not_a_short_one() -> None:
    """On an 18 m lane the move across was a swerve. Here the lane on is a
    little longer than the one beside it, so without the penalty the move
    would be made at once, on the 40 m lane."""
    own_short, own_long = polyline((0, 0), (0, 40)), polyline((0, 40), (4, 270), (0, 500))
    next_short, next_long = polyline((-4.5, 0), (-4.5, 40)), polyline((-4.5, 40), (-4.5, 500))
    exit_ramp = polyline((-4.5, 500), (-40, 560))
    net = network(
        [own_short, own_long, next_short, next_long, exit_ramp],
        successors=[(0, 1), (2, 3), (3, 4)],
        changes=[(0, 2), (1, 3)],
    )
    route = plan(net, start=(0.0, 5.0, 0.0), through=[(-40.0, 558.0)])
    assert route.lanes == (0, 1, 3, 4)
    assert route.steps == (ONWARD, CHANGE, ONWARD)


def two_lanes(left_hand: bool = False):
    """A 2 km road of two lanes, x = 0 and x = -4 (to the right heading +z)."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    for index, x in enumerate((0.0, -4.0)):
        graph.lanes.append(
            Lane(
                points=polyline((x, 0), (x, 2000)),
                source="road",
                owner=f"{index + 1:016X}",
                left_hand=left_hand,
            )
        )
    graph.lane_changes[0].append(1)
    graph.lane_changes[1].append(0)
    for index, x in enumerate((0.0, -4.0), start=2):
        graph.lanes.append(
            Lane(
                points=polyline((x, 2000), (x, 4000)),
                source="road",
                owner=f"{index + 1:016X}",
                left_hand=left_hand,
            )
        )
    graph.connect(0, 2)
    graph.connect(1, 3)
    graph.lane_changes[2].append(3)
    graph.lane_changes[3].append(2)
    return RoadNetwork.compile(graph)


def test_on_a_long_road_the_route_keeps_out_of_the_exit_lane() -> None:
    """In the outer lane an exit the GPS did not take was taken before the
    minimap showed it; from the other lane leaving takes a change first."""
    route = plan(two_lanes(), start=(-4.0, 5.0, 0.0), through=[(-2.0, 3990.0)])
    assert route.lanes[:2] == (1, 0), "moved across from the outer lane at once"
    assert route.lanes[-1] == 2


def test_a_new_network_is_not_taken_for_one_gone_before_it(monkeypatch) -> None:
    """Whether a lane is outermost is kept per network; a network made where
    a collected one was, at the same id, must not be given its answers."""
    from core.planning import route as routing

    left = two_lanes(left_hand=True)
    assert routing._outermost(left, 0)
    gone = id(left)
    monkeypatch.setattr(routing, "id", lambda _obj: gone, raising=False)
    right = two_lanes()
    assert not routing._outermost(right, 0)
    assert routing._outermost(right, 1)


def test_where_traffic_keeps_left_the_exit_lane_is_the_left_one() -> None:
    route = plan(two_lanes(left_hand=True), start=(-4.0, 5.0, 0.0), through=[(-2.0, 3990.0)])
    assert route.lanes == (1, 3), "the right lane is the inner one in Britain"


def test_where_two_road_pieces_meet_a_lane_apart_the_line_eases_across() -> None:
    """At a motorway where the median widens from 2 m to 4 m, the lanes of
    the next piece start 1 m further out; driven as drawn it was a step, the
    speed profile braked for it to 20 km/h and the steering hunted."""
    before = polyline((0, 0), (0, 300))
    after = polyline((1.0, 300), (1.0, 600))
    net = network([before, after], successors=[(0, 1)])
    line = routing.Route(net, (0, 1), (ONWARD,)).points()
    x, z = line[:, 0], line[:, 2]
    assert np.max(np.abs(np.diff(x))) < 0.1, "no step"
    assert x[np.argmin(np.abs(z - 300.0))] == pytest.approx(0.0, abs=0.05)
    assert x[np.argmin(np.abs(z - (300.0 + routing.JOIN_EASE_M + 10.0)))] == pytest.approx(1.0)


def _passage_net():
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 500)), source="road", owner="0A"))
    graph.lanes.append(Lane(points=polyline((0, 500), (0, 1000)), source="road", owner="0B"))
    graph.connect(0, 1)
    return RoadNetwork.compile(graph)


def test_a_taught_passage_is_driven_wherever_a_route_passes_it(monkeypatch) -> None:
    """Out of a roundabout in Bern the mapped lane ran over an island; the
    line the driver took there is taken by every route that passes it."""
    from core.planning import route as routing

    net = _passage_net()
    z = np.linspace(400, 600, 60)
    driven = np.column_stack([np.where((z > 450) & (z < 550), -3.0, 0.0), np.zeros(60), z])
    passage = routing.Crossing(driven[0, [0, 2]], driven[-1, [0, 2]], driven, routing.PASSAGE)
    monkeypatch.setattr(routing, "load_crossings", lambda: (passage,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 990.0)])
    ((first, last, _),) = route.taught_crossings()
    assert (first, last) == (0, 1)
    line = route.points()
    middle = line[(line[:, 2] > 470) & (line[:, 2] < 530)]
    assert middle[:, 0] == pytest.approx(-3.0, abs=0.01), "the driven line"


def test_where_a_taught_line_runs_along_the_path_is_known(monkeypatch) -> None:
    """The lane guard keeps off a driver's line: in the customs north of
    Basel it kept the truck 1.5 m off it, after the paint, and it hit
    something."""
    from core.localization.path import ReferencePath
    from core.planning import route as routing
    from tools.follow import taught_spans_on

    net = _passage_net()
    z = np.linspace(400, 600, 60)
    driven = np.column_stack([np.where((z > 450) & (z < 550), -3.0, 0.0), np.zeros(60), z])
    passage = routing.Crossing(driven[0, [0, 2]], driven[-1, [0, 2]], driven, routing.PASSAGE)
    monkeypatch.setattr(routing, "load_crossings", lambda: (passage,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 990.0)])
    path = ReferencePath(route.points()[:, [0, 2]], spacing=1.0)
    ((s0, s1),) = taught_spans_on(route, path)
    assert s0 == pytest.approx(path.nearest(0.0, 400.0).s, abs=2.0)
    assert s1 == pytest.approx(path.nearest(0.0, 600.0).s, abs=2.0)


def test_a_route_joining_a_taught_passage_part_way_drives_it_from_there(monkeypatch) -> None:
    """Into France at Basel, by a road joining the taught way half through
    the customs, the truck kept to the map's lanes, and they ran through the
    island of the 'Douane France' sign."""
    from core.planning import route as routing

    net = _passage_net()
    z = np.linspace(300, 600, 151)
    # Taught coming in from a road 20 m to the side, onto the route's at 440.
    x = np.interp(z, [300, 420, 440, 470, 550, 580, 600], [20, 20, 0, -3, -3, 0, 0])
    driven = np.column_stack([x, np.zeros(len(z)), z])
    passage = routing.Crossing(driven[0, [0, 2]], driven[-1, [0, 2]], driven, routing.PASSAGE)
    monkeypatch.setattr(routing, "load_crossings", lambda: (passage,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 990.0)])
    ((first, last, joined),) = route.taught_crossings()
    assert (first, last) == (0, 1)
    assert 430.0 < joined[0, 2] < 445.0, "from where it joins the route"
    line = route.points()
    middle = line[(line[:, 2] > 490) & (line[:, 2] < 530)]
    assert middle[:, 0] == pytest.approx(-3.0, abs=0.01), "the driven line"
    assert line[np.argmin(np.abs(line[:, 2] - 350.0)), 0] == pytest.approx(0.0), "not the side road"


def test_a_passage_taught_the_other_way_is_not_driven(monkeypatch) -> None:
    from core.planning import route as routing

    net = _passage_net()
    z = np.linspace(600, 400, 60)
    driven = np.column_stack([np.full(60, -3.0), np.zeros(60), z])
    passage = routing.Crossing(driven[0, [0, 2]], driven[-1, [0, 2]], driven, routing.PASSAGE)
    monkeypatch.setattr(routing, "load_crossings", lambda: (passage,))
    route = plan(net, start=(0.0, 10.0, 0.0), through=[(0.0, 990.0)])
    assert route.taught_crossings() == []


def test_the_part_of_a_route_ahead_ends_with_the_lane_that_reaches_that_far() -> None:
    """What the GPS worker sends first when the red line turns the route: a
    route's whole path took 6 s to make, and the junction was passed."""
    from core.planning.route import Route

    lanes = [polyline((0, 100 * k), (0, 100 * (k + 1))) for k in range(10)]
    net = network(lanes, successors=[(k, k + 1) for k in range(9)])
    whole = Route(net, tuple(range(10)), (ONWARD,) * 9)
    part = whole.ahead(2, 250.0)
    assert part.lanes == (0, 1, 2, 3, 4), "from lane 2: 100 + 100 + 100 >= 250"
    assert part.steps == (ONWARD,) * 4
    assert whole.ahead(0, 5_000.0) is whole
