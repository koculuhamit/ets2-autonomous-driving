"""Driving the game's route: planned as its GPS plans, and put right where
the game shows otherwise."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np

from core.map import CompanyPlace
from core.planning import game_route
from core.planning.game_route import GameRouter
from core.planning.nav_field import nav_length
from tests.unit.lane_networks import network, polyline
from tests.unit.test_gps import minimap_with_route_along

# Two ways round to one road on, the same length: which one is quicker is
# the game's call (the lanes' speeds).
APPROACH = polyline((0, 0), (0, 100))
LEFT = polyline((0, 100), (40, 140))
RIGHT = polyline((0, 100), (-40, 140))
LEFT_ON = polyline((40, 140), (40, 400))
RIGHT_ON = polyline((-40, 140), (-40, 400))
LEFT_JOIN = polyline((40, 400), (0, 440))
RIGHT_JOIN = polyline((-40, 400), (0, 440))
ON = polyline((0, 440), (0, 700))
LANES = [APPROACH, LEFT, RIGHT, LEFT_ON, RIGHT_ON, LEFT_JOIN, RIGHT_JOIN, ON]
DEPOT = CompanyPlace("transinet", "torino", (0.0, 700.0), ((0.0, 690.0),), ())
POSE = (0.0, 40.0, 0.0)


def ways(left_kmh: float = 72.0, right_kmh: float = 72.0):
    net = network(
        LANES,
        successors=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)],
    )
    speeds = np.full(len(LANES), 72.0, dtype=np.float32)
    speeds[[1, 3, 5]] = left_kmh
    speeds[[2, 4, 6]] = right_kmh
    return replace(net, lane_nav_speed=speeds)


def by_class(left_kmh: float, right_kmh: float):
    """The left way a local road, the right a motorway."""
    net = ways(left_kmh, right_kmh)
    classes = np.full(len(LANES), 2, dtype=np.uint8)
    classes[[1, 3, 5]] = 0
    return replace(net, lane_speed_class=classes)


def twice(router: GameRouter, mask, **kwargs):
    router.update(POSE, mask, now=0.0, **kwargs)
    return router.update(POSE, mask, now=0.5, **kwargs)


def test_the_route_is_the_game_s_quickest_to_the_depot() -> None:
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=40.0), DEPOT, POSE)
    assert router.route.lanes == (0, 1, 3, 5, 7)
    router = GameRouter.plan(ways(left_kmh=40.0, right_kmh=72.0), DEPOT, POSE)
    assert router.route.lanes == (0, 2, 4, 6, 7)


def test_between_two_ways_as_quick_the_red_line_decides() -> None:
    router = GameRouter.plan(ways(), DEPOT, POSE)
    taken = router.route.lanes[1]
    other = (RIGHT, RIGHT_ON) if taken == 1 else (LEFT, LEFT_ON)
    route = twice(router, minimap_with_route_along(APPROACH, *other, pose=POSE))
    assert route is not None
    assert route.lanes[1] != taken
    assert route.lanes[-1] == 7


def test_a_red_line_on_a_way_minutes_slower_is_taken_for_a_misread() -> None:
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=2.0), DEPOT, POSE)
    assert router.route.lanes[1] == 1
    route = twice(router, minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE))
    assert route is None
    assert router.route.lanes[1] == 1


def test_a_red_line_staying_on_a_way_minutes_slower_is_the_game_s_and_taken() -> None:
    """Out of Duisburg, the map's reckoning 36 km off the game's, the truck
    took the red line's way twice for a misread and went its own: the game
    planned anew each time."""
    from core.planning.game_route import DOUBTED_SEEN_S, DOUBTED_SIGHTINGS

    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=2.0), DEPOT, POSE)
    assert router.route.lanes[1] == 1
    red = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    route = None
    for k in range(DOUBTED_SIGHTINGS + 2):
        route = router.update(POSE, red, now=k * DOUBTED_SEEN_S / DOUBTED_SIGHTINGS)
        if route is not None:
            break
    assert route is not None and route.lanes[1] == 2, "the red line's way, right"
    assert route.lanes[-1] == 7


def test_where_the_game_goes_a_way_of_its_own_a_slower_red_line_is_taken_at_once() -> None:
    """To Luxembourg by two borders marked on the map, the game's route
    21.8 km off the nearest model's: a red line on a way minutes slower,
    watched, was passed, and the game planned anew twice."""
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=2.0), DEPOT, POSE, nav_distance=20_000.0)
    assert router.own_way
    assert router.route.lanes[1] == 1
    red = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    route = twice(router, red, nav_distance=20_000.0)
    assert route is not None and route.lanes[1] == 2, "the red line's way, right"


def test_where_the_game_s_distance_fits_its_way_is_not_its_own() -> None:
    planned = GameRouter.plan(ways(), DEPOT, POSE)
    router = GameRouter.plan(ways(), DEPOT, POSE, nav_distance=planned.nav_distance(POSE))
    assert not router.own_way


def test_one_odd_minimap_does_not_turn_it() -> None:
    router = GameRouter.plan(ways(), DEPOT, POSE)
    taken = router.route.lanes[1]
    other = (RIGHT, RIGHT_ON) if taken == 1 else (LEFT, LEFT_ON)
    assert router.update(POSE, minimap_with_route_along(APPROACH, *other, pose=POSE)) is None
    assert router.route.lanes[1] == taken


def test_when_the_game_plans_anew_so_does_the_route() -> None:
    router = GameRouter.plan(ways(), DEPOT, POSE)
    assert router.update(POSE, nav_distance=10_000.0) is None
    assert router.update(POSE, nav_distance=9_990.0) is None
    assert router.update(POSE, nav_distance=12_000.0) is not None


def test_off_the_route_for_a_moment_is_not_lost() -> None:
    router = GameRouter.plan(ways(), DEPOT, POSE)
    away = (15.0, 40.0, 0.0)
    assert router.update(away) is None
    assert router.update(away) is None


def test_the_route_is_extended_as_it_is_driven(monkeypatch) -> None:
    monkeypatch.setattr(game_route, "HORIZON_M", 300.0)
    monkeypatch.setattr(game_route, "EXTEND_WHEN_M", 500.0)
    monkeypatch.setattr(game_route, "KEEP_M", 100.0)
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=40.0), DEPOT, POSE)
    before = router.route
    assert before.lanes[-1] != 7
    route = router.update(POSE)
    assert route is not None
    assert route.lanes[: len(before.lanes)] == before.lanes
    assert route.length > before.length


def far_ways():
    """6 km straight on local roads, or round by a motorway 7 km long; each
    in pieces, most of them further than a model's free radius from both
    ends."""
    lanes = [polyline((0, 0), (0, 100))]
    successors = []
    local = [polyline((0, 100 + 500 * k), (0, 600 + 500 * k)) for k in range(12)]
    bend = [(0, 100), (1500, 1500), (1500, 2500), (1500, 3500), (1500, 4600), (0, 6100)]
    motorway = [polyline(a, b) for a, b in itertools.pairwise(bend)]
    for way in (local, motorway):
        first = len(lanes)
        lanes.extend(way)
        successors.append((0, first))
        successors.extend((first + k, first + k + 1) for k in range(len(way) - 1))
    lanes.append(polyline((0, 6100), (0, 6300)))
    goal = len(lanes) - 1
    successors += [(len(local), goal), (goal - 1, goal)]
    net = network(lanes, successors=successors)
    classes = np.full(len(lanes), 2, dtype=np.uint8)
    classes[1 : 1 + len(local)] = 0
    speeds = np.full(len(lanes), 72.0, dtype=np.float32)
    speeds[1 : 1 + len(local)] = 64.0
    depot = CompanyPlace("transinet", "torino", (0.0, 6300.0), ((0.0, 6290.0),), ())
    return replace(net, lane_speed_class=classes, lane_nav_speed=speeds), depot, len(local)


def test_where_the_game_s_route_is_longer_the_model_that_fits_it_is_driven() -> None:
    """Antwerp to Nynäshamn: the game kept to the motorway, though by its own
    reckoning a way on local roads was quicker; its distance said so."""
    net, depot, locals_ = far_ways()
    plain = GameRouter.plan(net, depot, POSE)
    assert plain.route.lanes[1] == 1  # local, and quicker
    motorway_length = sum(nav_length(net, lane) for lane in range(1 + locals_, len(net) - 1))
    game = nav_length(net, 0) * 0.6 + motorway_length + nav_length(net, len(net) - 1) * 0.95
    fitted = GameRouter.plan(net, depot, POSE, nav_distance=game)
    assert fitted.route.lanes[1] == 1 + locals_
    assert abs(fitted.nav_distance(POSE) - game) < 1000.0


def test_without_the_game_s_distance_the_plain_quickest_is_driven() -> None:
    net, depot, _ = far_ways()
    assert GameRouter.plan(net, depot, POSE, nav_distance=None).route.lanes[1] == 1


def test_a_route_saved_is_driven_as_it_is_if_it_fits() -> None:
    planned = GameRouter.plan(ways(), DEPOT, POSE)
    router = GameRouter.along(ways(), DEPOT, planned.route)
    assert router.update(POSE, nav_distance=planned.nav_distance(POSE)) is None
    assert router.route.lanes == planned.route.lanes


def test_a_red_line_on_a_slow_way_is_taken_when_the_game_s_distance_says_so() -> None:
    """Out of Helsingborg the red line took a way minutes slower by the map's
    reckoning; it was the game's, and turned away, the game planned anew
    twice. The length the game gives fits that way, and not the planned."""
    net, depot, locals_ = far_ways()
    speeds = np.array(net.lane_nav_speed)
    speeds[1 + locals_ : len(net) - 1] = 3.0  # the motorway at walking pace, by the map
    net = replace(net, lane_nav_speed=speeds)
    first_motorway = 1 + locals_
    game = (
        nav_length(net, 0) * 0.6
        + sum(nav_length(net, lane) for lane in range(first_motorway, len(net) - 1))
        + nav_length(net, len(net) - 1) * 0.95
    )
    router = GameRouter.plan(net, depot, POSE, nav_distance=game)
    assert router.route.lanes[1] == 1, "the map's quickest is the local road"
    red = minimap_with_route_along(
        polyline((0, 0), (0, 100)), polyline((0, 100), (1500, 1500)), pose=POSE
    )
    route = twice(router, red, nav_distance=game)
    assert route is not None
    assert route.lanes[1] == first_motorway


def test_a_truck_beside_a_hidden_road_starts_on_the_road_the_map_shows() -> None:
    """Near Brussels the truck was matched to a hidden road beside the
    motorway; a hidden road is where it starts only when there is no other."""
    from core.planning.game_route import _start_lane
    from core.planning.nav_field import NavField

    hidden_way = polyline((0, 0), (0, 300))
    shown_way = polyline((4, 0), (4, 300))
    on = polyline((4, 300), (4, 700))
    net = network([hidden_way, shown_way, on], successors=[(0, 2), (1, 2)])
    net = replace(net, lane_hidden=np.array([True, False, False]))
    field = NavField.to_goals(net, {2: 0.0})
    assert _start_lane(net, field, (0.0, 100.0, 0.0)) == 1
    alone = replace(net, lane_hidden=np.array([True, True, False]))
    field = NavField.to_goals(alone, {2: 0.0})
    assert _start_lane(alone, field, (0.0, 100.0, 0.0)) == 0


def test_on_a_long_way_a_red_line_a_small_share_slower_is_the_game_s(monkeypatch) -> None:
    """Dover, 26 hours from Oslo: the ferry and the Channel Tunnel 4 minutes
    apart by the map. Turned away, the red line to the ferry sent the truck
    round and round; the map is off the game's reckoning by more than that."""
    monkeypatch.setattr(game_route, "CLOSE_CALL_S", 0.0)
    monkeypatch.setattr(game_route, "CLOSE_CALL_SHARE", 0.2)
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=64.0), DEPOT, POSE)
    assert router.route.lanes[1] == 1
    route = twice(router, minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE))
    assert route is not None and route.lanes[1] == 2


GATE_POSE = (0.0, 30.0, 0.0)


def two_gates():
    """A road on to a fork: the near gate down the left way, the far one
    60 m further down the right."""
    lanes = [
        polyline((0, 0), (0, 50)),
        polyline((0, 50), (-40, 70), (-40, 90)),
        polyline((0, 50), (40, 70), (40, 150)),
    ]
    net = network(lanes, successors=[(0, 1), (0, 2)])
    depot = CompanyPlace("ika_bohag", "milano", (0.0, 100.0), ((-40.0, 88.0), (40.0, 148.0)), ())
    return net, depot


def to_far_gate(net) -> float:
    """The game's metres from `GATE_POSE` to the far gate."""
    from core.planning.nav_field import nav_along

    on = nav_length(net, 0) - nav_along(net, 0, 30.0)
    return on + nav_along(net, 2, net.project(2, 40.0, 148.0).s)


def test_the_gate_driven_to_is_the_one_the_game_s_distance_fits() -> None:
    """At ika_bohag in Milan the game went to the far gate; the route went to
    the near one, and turned round in a junction to come back."""
    net, depot = two_gates()
    assert GameRouter.plan(net, depot, GATE_POSE).route.lanes == (0, 1)
    assert to_far_gate(net) < game_route.GATE_WITHIN_M
    router = GameRouter.plan(net, depot, GATE_POSE, nav_distance=to_far_gate(net) + 20.0)
    assert router.route.lanes == (0, 2)
    assert any("fits the gate" in note for note in router.notes)


def test_a_distance_that_fits_no_gate_well_leaves_the_route_as_it_is() -> None:
    net, depot = two_gates()
    off = to_far_gate(net) + 2 * game_route.GATE_FIT_M
    assert GameRouter.plan(net, depot, GATE_POSE, nav_distance=off).route.lanes == (0, 1)


def test_driving_on_the_gate_is_chosen_once_the_game_s_distance_fits_it() -> None:
    net, depot = two_gates()
    router = GameRouter.plan(net, depot, GATE_POSE)
    assert router.route.lanes == (0, 1)
    assert router.update(GATE_POSE, now=0.0, nav_distance=to_far_gate(net)) is not None
    assert router.route.lanes == (0, 2)


def test_on_a_taught_crossing_far_from_the_mapped_lanes_the_truck_is_not_lost(
    monkeypatch,
) -> None:
    """Out of the Italian-Swiss customs, in a booth's lane 6 m off the mapped
    ones, the truck was taken to be lost; the route was planned again over
    the booths' islands, and the truck was steered into one."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning import route as routing

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 1000)), source="road", owner="0A"))
    detour = polyline((0, 1000), (40, 1100), (40, 1250), (0, 1350))
    graph.lanes.append(Lane(points=detour, source="road", owner="0B", ai_vehicles=False))
    graph.lanes.append(Lane(points=polyline((0, 1350), (0, 2350)), source="road", owner="0C"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    net = RoadNetwork.compile(graph)
    driven = np.column_stack([np.full(120, -5.0), np.zeros(120), np.linspace(1000, 1350, 120)])
    crossing = routing.Crossing(np.array([0.0, 1000.0]), np.array([0.0, 1350.0]), driven)
    monkeypatch.setattr(routing, "load_crossings", lambda: (crossing,))
    depot = CompanyPlace("rt_log", "bern", (0.0, 2340.0), ((0.0, 2340.0),), ())
    router = GameRouter.along(net, depot, routing.Route(net, (0, 1, 2), ("onward", "onward")))
    assert router.locate(-5.0, 1200.0) == 1
    assert router.locate(-30.0, 1200.0) is None, "off the driven line too"


def test_a_red_line_read_onto_one_lane_then_another_of_its_road_is_one_sighting(
    monkeypatch,
) -> None:
    """Read onto the road's lanes in turn, the count of sightings began anew
    at each minimap, and the junction was passed before the way was taken."""
    from core.planning.game_route import DOUBTED_SEEN_S, DOUBTED_SIGHTINGS

    net = ways(left_kmh=72.0, right_kmh=2.0)
    owners = np.array(net.lane_owner).copy()
    owners[4] = owners[2]  # RIGHT and RIGHT_ON as lanes of one road
    router = GameRouter.plan(replace(net, lane_owner=owners), DEPOT, POSE)
    reads = iter([(0, (), 4), (0, (), 2)] * DOUBTED_SIGHTINGS)
    monkeypatch.setattr(game_route, "red_line_turn", lambda *a, **k: next(reads))
    red = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    route = None
    for k in range(DOUBTED_SIGHTINGS):
        route = router.update(POSE, red, now=k * DOUBTED_SEEN_S / (DOUBTED_SIGHTINGS - 1))
    assert route is not None and route.lanes[1] == 2


def test_a_slower_red_line_way_back_or_already_passed_is_not_taken() -> None:
    """Out of Duisburg a roundabout's ring drawn red was read as a way round
    and back; taken while in it, the truck went back to the depot it left."""
    router = GameRouter.plan(ways(left_kmh=72.0, right_kmh=2.0), DEPOT, POSE)
    ahead = router._still_to_take(POSE, 0, (), 4)  # RIGHT_ON runs north, as the truck
    assert ahead
    turned = (POSE[0], POSE[1], math.pi)  # the truck heading south
    assert not router._still_to_take(turned, 0, (), 4), "back the way it came"
    router.at = 1
    assert not router._still_to_take(POSE, 0, (), 4), "in the junction already"


def test_a_red_line_read_onto_a_way_round_and_back_is_never_taken() -> None:
    """Off a pump near Bremen the station's way in runs beside its way out;
    the red line out was read onto the way back in, 43 s slower by the map,
    and taken: the truck went round the station back to the pumps."""
    # Past the junction a way off to the right goes round and back onto the
    # approach; it is as quick for the map as the way on, by its speeds.
    loop_out = polyline((0, 100), (-30, 130))
    loop_round = polyline((-30, 130), (-30, 20))
    loop_in = polyline((-30, 20), (0, 30))
    lanes = [polyline((0, 0), (0, 30)), polyline((0, 30), (0, 100)), polyline((0, 100), (0, 700))]
    net = network(
        [*lanes, loop_out, loop_round, loop_in],
        successors=[(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 1)],
    )
    net = replace(net, lane_nav_speed=np.full(6, 72.0, dtype=np.float32))
    router = GameRouter.plan(net, DEPOT, POSE)
    red = minimap_with_route_along(lanes[1], loop_out, loop_round, pose=POSE)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):  # drawn there, and staying drawn
        assert router.update(POSE, red, now=t) is None, "round and back: a misread"
    assert 3 not in router.route.lanes


def test_asked_to_plan_again_from_here_it_does_on_the_next_update() -> None:
    """Near Birmingham the route driven ran out on the road, a lane change
    given up and the red line's new ways refused: the truck stopped there.
    Asked (tools/follow.py, REPLAN_LEFT_M), the router plans afresh."""
    router = GameRouter.plan(ways(), DEPOT, POSE)
    assert router.update(POSE) is None
    router.replan_here()
    route = router.update(POSE)
    assert route is not None
    assert any("planned again from here" in note for note in router.notes)
    assert router.update(POSE) is None, "once"


def test_a_move_across_from_a_lane_too_short_is_put_off_to_the_next() -> None:
    # Three lanes side by side, 15 m long past a junction, then 95 m: the
    # way on two lanes over is moved across to in the 95 m ones.
    short = [polyline((x, 0), (x, 15)) for x in (0.0, 3.5, 7.0)]
    long = [polyline((x, 15), (x, 110)) for x in (0.0, 3.5, 7.0)]
    net = network(
        [*short, *long],
        successors=[(0, 3), (1, 4), (2, 5)],
        changes=[(0, 1), (1, 2), (3, 4), (4, 5)],
    )
    router = GameRouter.__new__(GameRouter)
    router.network = net
    router.route = game_route.Route(net, (0, 3), (game_route.ONWARD,))
    assert router._across_later(0, (1, 2), 5) == (1, (4,))
    # Where the lane is long enough, the move across stays where it is.
    roomy = [polyline((x, 0), (x, 90)) for x in (0.0, 3.5, 7.0)]
    net = network(
        [*roomy, *long],
        successors=[(0, 3), (1, 4), (2, 5)],
        changes=[(0, 1), (1, 2), (3, 4), (4, 5)],
    )
    router.network = net
    router.route = game_route.Route(net, (0, 3), (game_route.ONWARD,))
    assert router._across_later(0, (1, 2), 5) is None
