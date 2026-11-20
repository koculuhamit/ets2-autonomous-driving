"""Following the GPS off the control loop's process, as tools/follow.py does."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from core.localization.path import ReferencePath
from core.map import CompanyPlace
from core.planning.gps import GpsRouter
from tests.unit.test_gps import (
    APPROACH,
    LEFT,
    LEFT_ON,
    NET,
    PLANNED_LEFT,
    POSE,
    RIGHT,
    RIGHT_ON,
    minimap_with_route_along,
)
from tools.follow import EASE_ONTO_M, GpsWorker, eased_onto, path_marks, route_path

DEPOT = CompanyPlace("transinet", "torino", (-40.0, 395.0), ((-40.0, 395.0),), ())


def packed(mask: np.ndarray) -> bytes:
    return np.packbits(mask).tobytes()


def worker() -> GpsWorker:
    gps = GpsWorker(GpsRouter(NET, DEPOT, PLANNED_LEFT), spacing=2.0, lateral=1.5, max_speed=10.0)
    gps.INTERVAL_S = 0.0
    return gps


def offered_until_taken(gps: GpsWorker, pose, mask: bytes, seconds: float = 10.0):
    """Offer the minimap over and over, as the loop does, until a route comes
    back: the worker looks at the newest only, and one seen twice running is
    believed."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        gps.offer(pose, mask, b"")
        taken = taken_within(gps, 0.2)
        if taken is not None:
            return taken
    return None


def taken_within(gps: GpsWorker, seconds: float):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        taken = gps.take()
        if taken is not None:
            return taken
        time.sleep(0.02)
    return None


def test_where_the_gps_turns_off_a_route_ready_to_drive_comes_back() -> None:
    """The route, its path and its profile are all made in the worker: made
    in the loop, the profile alone for 80 km held it up for over a second."""
    gps = worker()
    try:
        mask = packed(minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON))
        taken = offered_until_taken(gps, POSE, mask)
    finally:
        gps.stop()
    assert taken is not None
    route, path, profile, _ = taken
    assert route.lanes == (0, 2, 4)
    assert route.network is NET
    assert np.allclose(path.points, route_path(route, 2.0).points)
    assert profile is not None


def test_a_route_comes_back_marked_along_its_path() -> None:
    """Its lights, junctions and roundabouts are found in the worker too:
    found in the loop for a new route of 30 km they held it up for 1.3 s, in
    a roundabout in Torino, and the plugin stopped the truck."""
    gps = worker()
    try:
        mask = packed(minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON))
        taken = offered_until_taken(gps, POSE, mask)
    finally:
        gps.stop()
    assert taken is not None
    route, path, _, marks = taken
    expected = path_marks(route, path)
    assert marks.lights == expected.lights
    assert marks.barriers == expected.barriers
    assert [change.start for change in marks.changes] == [c.start for c in expected.changes]
    assert marks.junctions == expected.junctions
    assert marks.turns_across == expected.turns_across
    assert marks.rings == expected.rings


def test_where_the_gps_agrees_nothing_comes_back() -> None:
    gps = worker()
    try:
        mask = packed(minimap_with_route_along(APPROACH, LEFT, LEFT_ON))
        gps.offer(POSE, mask, b"")
        time.sleep(0.2)
        gps.offer(POSE, mask, b"")
        assert taken_within(gps, 1.0) is None
    finally:
        gps.stop()


def test_offers_come_at_most_twice_a_second() -> None:
    gps = GpsWorker(GpsRouter(NET, DEPOT, PLANNED_LEFT), spacing=2.0, lateral=1.5, max_speed=10.0)
    try:
        mask = packed(minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON))
        gps.offer(POSE, mask, b"")
        gps.offer(POSE, mask, b"")  # too soon: dropped, so the GPS is seen only once
        assert taken_within(gps, 1.0) is None
    finally:
        gps.stop()


class SlowRouter:
    """A router whose first look takes a while, as a new 30 km route's path
    did (6 s); it says which pose it looked from."""

    def __init__(self) -> None:
        self.network = NET
        self.route = PLANNED_LEFT
        self.notes: list[str] = []

    def update(self, pose, mask, roads, now):
        self.notes = [f"looked from {pose[1]:.0f}"]
        if pose[1] == 0.0:
            time.sleep(1.5)
        return None


def test_a_worker_that_was_busy_looks_next_at_the_newest_minimap() -> None:
    """Out of Bern the worker was still on minimaps from seconds before when
    the red line left the route 150 m ahead: offers queued behind a slow
    look, and on Windows the pipe holds less than one of them."""
    gps = GpsWorker(SlowRouter(), spacing=2.0, lateral=1.5, max_speed=10.0)
    gps.INTERVAL_S = 0.0
    mask = packed(np.zeros((270, 335), dtype=bool))
    seen: list[str] = []
    try:
        deadline = time.perf_counter() + 20.0  # the worker started, spawned on Windows
        while not seen and time.perf_counter() < deadline:
            gps.offer((0.0, -1.0, 0.0), mask, mask)
            time.sleep(0.05)
            gps.take()
            seen = gps.notes()
        assert seen == ["looked from -1"]
        seen = []
        gps.offer((0.0, 0.0, 0.0), mask, mask)
        time.sleep(0.3)  # the worker is in its slow look
        for z in (10.0, 20.0, 30.0):
            gps.offer((0.0, z, 0.0), mask, mask)
            time.sleep(0.1)
        deadline = time.perf_counter() + 5.0
        while len(seen) < 2 and time.perf_counter() < deadline:
            gps.take()
            seen += gps.notes()
            time.sleep(0.02)
    finally:
        gps.stop()
    assert seen[:2] == ["looked from 0", "looked from 30"]


def long_road():
    from core.planning.route import ONWARD, Route
    from tests.unit.lane_networks import network, polyline

    lanes = [polyline((0, 500 * k), (0, 500 * (k + 1))) for k in range(20)]
    net = network(lanes, successors=[(k, k + 1) for k in range(19)])
    return Route(net, tuple(range(20)), (ONWARD,) * 19)


class TurningRouter:
    """A router that, at its first look, turns onto a route 10 km long."""

    def __init__(self) -> None:
        self.route = long_road()
        self.network = self.route.network
        self.at = 0
        self.notes: list[str] = []

    def update(self, pose, mask, roads, now):
        return self.route


def test_a_new_route_comes_first_as_its_first_part_then_whole() -> None:
    """Out of Bern the red line turned the route 125 m before its junction;
    the new 30 km route's path took 6 s, and the junction was passed."""
    from tools.follow import FIRST_PART_M

    gps = GpsWorker(TurningRouter(), spacing=2.0, lateral=1.5, max_speed=10.0)
    gps.INTERVAL_S = 0.0
    lengths: list[float] = []
    try:
        mask = packed(np.zeros((270, 335), dtype=bool))
        deadline = time.perf_counter() + 30.0
        while len(lengths) < 2 and time.perf_counter() < deadline:
            gps.offer((0.0, 10.0, 0.0), mask, mask)
            # one at a time, as the loop does, so both are seen
            while (result := gps._received.get(timeout=30.0))[0] != "route":
                pass
            lengths.append(float(result[5].length))
    finally:
        gps.stop()
    assert FIRST_PART_M <= lengths[0] < FIRST_PART_M + 600.0
    assert lengths[1] == pytest.approx(10_000.0, abs=5.0)


def test_a_new_route_off_to_one_side_is_joined_from_where_the_truck_is() -> None:
    """Moving back out of an exit lane, the new line eases across from the
    lane's start behind the truck, 3.75 m over where it is; driven as it
    stands that is a swerve."""
    old = ReferencePath(np.column_stack([np.full(300, -3.75), np.arange(300.0)]), spacing=2.0)
    new = ReferencePath(np.column_stack([np.zeros(300), np.arange(300.0)]), spacing=2.0)
    joined, begin = eased_onto(old, new, 100.0, 2.0)
    assert begin == pytest.approx(100.0, abs=2.0)
    here = joined.nearest(-3.75, 100.0, near=100.0)
    assert math.hypot(here.x + 3.75, here.z - 100.0) < 0.3, "the line starts where the truck is"
    later = joined.sample_at(100.0 + EASE_ONTO_M + 10.0)
    assert later.x == pytest.approx(0.0, abs=0.05)


def test_a_new_route_that_parts_ahead_is_taken_as_it_is() -> None:
    old = ReferencePath(np.column_stack([np.zeros(300), np.arange(300.0)]), spacing=2.0)
    new = ReferencePath(
        np.vstack([np.column_stack([np.zeros(200), np.arange(200.0)]), [[40.0, 260.0]]]),
        spacing=2.0,
    )
    joined, begin = eased_onto(old, new, 100.0, 2.0)
    assert begin is None
    assert joined is new


def test_the_game_s_route_is_planned_in_the_worker_and_told_the_game_s_distance() -> None:
    """The models are weighed where the route is driven from; when the game
    plans anew, its distance grows, and a new route comes back, with word."""
    from core.planning.game_route import GameRouter
    from tests.unit.test_game_route import DEPOT as GAME_DEPOT
    from tests.unit.test_game_route import POSE as GAME_POSE
    from tests.unit.test_game_route import ways

    planned = GameRouter.plan(ways(), GAME_DEPOT, GAME_POSE)
    router = GameRouter.along(ways(), GAME_DEPOT, planned.route)
    gps = GpsWorker(router, spacing=2.0, lateral=1.5, max_speed=10.0)
    gps.INTERVAL_S = 0.0
    blank = packed(np.zeros((270, 335), dtype=bool))
    try:
        # Until the worker has weighed the models on it: it looks at the
        # newest minimap only, and one offered before it is up is never seen.
        weighed, deadline = [], time.perf_counter() + 30.0
        while not any("by local roads" in note for note in weighed):
            assert time.perf_counter() < deadline, "the models were never weighed"
            gps.offer(GAME_POSE, blank, b"", planned.nav_distance(GAME_POSE))
            gps.take()
            weighed += gps.notes()
            time.sleep(0.05)
        gps.offer(GAME_POSE, blank, b"", planned.nav_distance(GAME_POSE) + 5_000.0)
        taken = taken_within(gps, 10.0)
        notes = gps.notes()
    finally:
        gps.stop()
    assert taken is not None
    assert any("planned anew" in note for note in notes)


def test_on_a_route_that_starts_at_the_truck_the_truck_is_found_at_its_start() -> None:
    """Planned anew by the game, the route began under the truck, 1,700 m
    short of where the old route had it; looked for about the old progress,
    the truck was put 9 m off its line and the drive stopped."""
    from tools.follow import where_on

    old = ReferencePath(np.column_stack([np.zeros(1000), np.arange(0.0, 2000.0, 2.0)]), spacing=2.0)
    new = ReferencePath(
        np.column_stack([np.linspace(0.0, 9.0, 300), 1700.0 + np.arange(300.0)]), spacing=2.0
    )
    at = where_on(new, 0.0, 1700.0, progress=1700.0)
    assert at.s < 5.0
    assert math.hypot(at.x - 0.0, at.z - 1700.0) < 0.5
    _, begin = eased_onto(old, new, 1700.0, 2.0)
    assert begin is None or begin < 5.0


def test_a_route_refused_is_forgotten_by_the_worker_too() -> None:
    """Out of Calais a route 4.1 m from the truck was refused, and the
    worker's router planned on from it all the same: the drive and the
    router went two ways. Refused, the worker goes back to the route driven,
    and a red line along that one is agreed with."""
    from core.planning.game_route import GameRouter
    from tests.unit import test_game_route as game

    planned = GameRouter.plan(game.ways(), game.DEPOT, game.POSE)
    router = GameRouter.along(game.ways(), game.DEPOT, planned.route)
    gps = GpsWorker(router, spacing=2.0, lateral=1.5, max_speed=10.0)
    gps.INTERVAL_S = 0.0
    taken = planned.route.lanes[1]
    ways = {1: (game.LEFT, game.LEFT_ON), 2: (game.RIGHT, game.RIGHT_ON)}
    other = packed(minimap_with_route_along(game.APPROACH, *ways[3 - taken], pose=game.POSE))
    same = packed(minimap_with_route_along(game.APPROACH, *ways[taken], pose=game.POSE))
    try:
        turned = offered_until_taken(gps, game.POSE, other)
        assert turned is not None and turned[0].lanes[1] != taken
        gps.refuse()
        for _ in range(3):
            gps.offer(game.POSE, same, b"")
            time.sleep(0.3)
        assert taken_within(gps, 1.5) is None
    finally:
        gps.stop()


def test_why_a_drive_ended_is_written_for_the_mission(tmp_path) -> None:
    import json

    from tools.follow import write_outcome

    write_outcome(tmp_path / "drive" / "outcome.json", "unsafe", "-6.9 m from the path")
    written = json.loads((tmp_path / "drive" / "outcome.json").read_text())
    assert written == {"reason": "unsafe", "detail": "-6.9 m from the path"}


def test_a_truck_a_little_off_its_route_drives_onto_it_from_where_it_stands() -> None:
    """Off the pump out of Frederikshavn the truck stood 6.3 m beside the
    forecourt's lane, and the drive stopped at once."""
    from tools.follow import JOIN_LENGTHS_M, JOIN_MAX_M, joined_from

    lane = ReferencePath(np.column_stack([np.zeros(200), np.arange(0.0, 400.0, 2.0)]), spacing=2.0)
    joined = joined_from(lane, 6.3, 30.0, 0.0, 2.0)
    assert joined is not None and joined is not lane
    start = joined.sample_at(0.0)
    assert math.hypot(start.x - 6.3, start.z - 30.0) < 0.5, "it starts where the truck stands"
    later = joined.sample_at(JOIN_LENGTHS_M[1] + 4.0)
    assert abs(later.x) < 0.1, "and is on the lane within the easing"
    assert joined_from(lane, 0.4, 30.0, 0.0, 2.0) is lane
    assert joined_from(lane, JOIN_MAX_M + 2.0, 30.0, 0.0, 2.0) is None


def test_a_new_route_one_lane_over_is_taken_and_one_further_is_not() -> None:
    """Half way into a lane, the red line turning back: the new route runs
    down the lane being left, 4.5 m off; another junction lane, 9.4 m."""
    from tools.follow import NEW_ROUTE_OFF_M

    assert 4.5 < NEW_ROUTE_OFF_M < 9.4


def test_a_truck_part_way_along_its_route_s_first_lane_is_on_it() -> None:
    """Milan: the route's first lane began 115 m behind the truck, and the
    truck, on it, was taken for 14.7 m off the road."""
    from tools.follow import joined_from

    lane = ReferencePath(np.column_stack([np.zeros(200), np.arange(0.0, 400.0, 2.0)]), spacing=2.0)
    assert joined_from(lane, 0.7, 115.0, 0.0, 2.0) is lane


def test_a_truck_a_little_off_a_route_that_turns_at_once_is_not_led_straight_on() -> None:
    """A Milan depot: 2.1 m off a route turning right at once, eased over
    80 m from straight ahead, the truck drove on into a railing."""
    from tools.follow import joined_from

    arc = np.radians(np.linspace(0.0, 90.0, 60))
    bend = np.column_stack([20.0 - 20.0 * np.cos(arc), 20.0 * np.sin(arc)])
    route = ReferencePath(bend, spacing=2.0)
    assert joined_from(route, -2.1, 0.0, 0.0, 2.0) is route, "the steering closes 2 m itself"
    joined = joined_from(route, -5.0, 0.0, 0.0, 2.0)
    assert joined is not None
    end_of_easing = joined.sample_at(30.0)
    on_route = route.nearest(end_of_easing.x, end_of_easing.z)
    assert math.hypot(on_route.x - end_of_easing.x, on_route.z - end_of_easing.z) < 1.0


def test_a_turn_swings_wide_only_where_the_map_has_a_lane_outside_it() -> None:
    from tests.unit.lane_networks import network, polyline
    from tools.follow import WITHIN_LANE_M, room_outside

    road_beside = network([polyline((0, 0), (0, 100)), polyline((4.5, 0), (4.5, 100))])
    alone = network([polyline((0, 0), (0, 100))])
    points = np.array([[0.0, 50.0]])
    outward = np.array([[1.0, 0.0]])
    wanted = np.array([1.2])
    assert room_outside(road_beside, points, outward, wanted)[0] == np.inf
    assert room_outside(alone, points, outward, wanted)[0] == WITHIN_LANE_M


def test_on_a_roundabouts_way_out_the_turn_is_not_swung_wide() -> None:
    """In Italy, at night, the tractor swung 0.4 m wide out of a roundabout,
    drifted 0.3 m more, and its cab hit the sign on the island by the way out."""
    from core.planning.route import ONWARD, Route
    from tests.unit.lane_networks import network, polyline
    from tests.unit.test_acc import roundabout
    from tools.follow import RING_EXIT_SWING_M, room_outside, ways_out_of_rings

    road_beside = network([polyline((0, 0), (0, 100)), polyline((4.5, 0), (4.5, 100))])
    points = np.array([[0.0, 50.0], [0.0, 90.0]])
    outward = np.array([[1.0, 0.0]] * 2)
    wanted = np.array([1.2] * 2)
    way_out = np.array([[0.0, 88.0], [0.0, 92.0]])
    room = room_outside(road_beside, points, outward, wanted, way_out)
    assert room[0] == np.inf
    assert room[1] == RING_EXIT_SWING_M
    ring = roundabout(way_out=True)
    out = ways_out_of_rings(Route(ring, (0, 1, 2, 6), (ONWARD,) * 3))
    assert len(out) and np.allclose(out[-1], np.asarray(ring.lane_points(6))[-1, [0, 2]])


def test_a_lane_beside_coming_the_other_way_is_no_room_to_swing_wide_onto() -> None:
    """Turning right into the depot road at Duisburg the oncoming lane,
    beyond a splitter island, was taken for room: the cab hit the island's
    sign."""
    from tests.unit.lane_networks import network, polyline
    from tools.follow import ONCOMING_SWING_M, room_outside

    same_way = network([polyline((0, 0), (0, 100)), polyline((4.5, 0), (4.5, 100))])
    other_way = network([polyline((0, 0), (0, 100)), polyline((4.5, 100), (4.5, 0))])
    points = np.array([[0.0, 48.0], [0.0, 50.0], [0.0, 52.0]])
    outward = np.array([[1.0, 0.0]] * 3)
    wanted = np.array([1.2] * 3)
    assert room_outside(same_way, points, outward, wanted)[1] == np.inf
    assert room_outside(other_way, points, outward, wanted)[1] == ONCOMING_SWING_M, "a little"


def test_with_a_lane_coming_the_other_way_on_the_outside_the_turn_is_not_swung_wide() -> None:
    """Out of Hamburg, turning right with the oncoming lane 6 m off beyond a
    median, the tractor swung 0.6 m within its lane and its cab hit the
    island's sign."""
    from tests.unit.lane_networks import network, polyline
    from tools.follow import ONCOMING_SWING_M, WITHIN_LANE_M, room_outside

    own = polyline((0, 0), (0, 100))
    median = network([own, polyline((6.0, 100), (6.0, 0))])
    inside = network([own, polyline((-6.0, 100), (-6.0, 0))])
    alone = network([own])
    points = np.array([[0.0, 48.0], [0.0, 50.0], [0.0, 52.0]])
    outward = np.array([[1.0, 0.0]] * 3)
    wanted = np.array([0.5] * 3)
    assert room_outside(median, points, outward, wanted)[1] == ONCOMING_SWING_M
    assert room_outside(inside, points, outward, wanted)[1] == np.inf, "on the inside: no matter"
    assert room_outside(alone, points, outward, np.array([1.2] * 3))[1] == WITHIN_LANE_M


def test_a_loop_that_stood_still_as_long_as_the_plugin_waits_ends_the_drive() -> None:
    """The plugin stops the truck itself after `SAFETY_STOP_S` of silence;
    a loop that comes back after that is steering a truck it no longer has."""
    from core.bridge.command import SAFETY_STOP_S
    from tools.follow import LoopStalled, check_loop

    check_loop(SAFETY_STOP_S * 0.9)  # a hiccup the plugin rides out, holding the steering
    with pytest.raises(LoopStalled):
        check_loop(SAFETY_STOP_S)


def test_a_loop_begun_on_the_first_part_is_sent_the_whole_route_at_once() -> None:
    """Making the path of a whole 30 km route took 6.3 s, and the truck stood
    waiting for it when the beacon was switched on: the loop begins on the
    route's first part, and the worker makes the whole and sends it unasked."""
    router = GpsRouter(NET, DEPOT, PLANNED_LEFT)
    gps = GpsWorker(router, spacing=2.0, lateral=1.5, max_speed=10.0, whole=True)
    try:
        taken = taken_within(gps, 10.0)
    finally:
        gps.stop()
    assert taken is not None, "nothing was sent"
    route, path, _, _ = taken
    assert route.lanes == PLANNED_LEFT.lanes
    assert np.allclose(path.points, route_path(PLANNED_LEFT, 2.0).points)


def test_off_a_pump_the_truck_goes_straight_on_before_turning_onto_the_route() -> None:
    """Out of a station in Sweden, steered at once onto a lane turning right,
    the trailer cut into the canopy's pillar."""
    from core.localization.path import ReferencePath
    from tools.follow import OFF_PUMP_STRAIGHT_M, joined_from

    angles = np.linspace(0.0, np.pi / 2, 60)
    turning = np.column_stack([-(20.0 - 20.0 * np.cos(angles)), 20.0 * np.sin(angles)])
    path = ReferencePath(turning, spacing=1.0)
    joined = joined_from(path, 0.0, 0.0, 0.0, 1.0, straight=OFF_PUMP_STRAIGHT_M)
    assert joined is not None
    ahead = joined.points[joined.points[:, 1] < OFF_PUMP_STRAIGHT_M - 1.0]
    assert np.abs(ahead[:, 0]).max() < 0.05, "straight on first"
    assert joined_from(path, 0.0, 0.0, 0.0, 1.0) is path, "elsewhere, the route as it is"


def test_the_driver_is_told_to_fill_up_once_the_tank_is_low_and_again_as_it_runs_down() -> None:
    from tools.follow import FUEL_TELL_BELOW_KM, FUEL_TELL_EVERY_KM, fuel_to_tell

    assert not fuel_to_tell(FUEL_TELL_BELOW_KM + 1.0, None)
    assert fuel_to_tell(FUEL_TELL_BELOW_KM - 1.0, None)
    told = FUEL_TELL_BELOW_KM - 1.0
    assert not fuel_to_tell(told - FUEL_TELL_EVERY_KM + 1.0, told), "not at every look"
    assert fuel_to_tell(told - FUEL_TELL_EVERY_KM, told)


def test_an_electric_truck_braking_with_its_motor_takes_the_pedal() -> None:
    """DAF XF Electric: brake read 0.25, applied 0.03; not 'nothing applied'."""
    from tools.follow import pedals_taken

    truck = {
        "effective_throttle": 0.0,
        "effective_brake": 0.03,
        "input_throttle": 0.0,
        "input_brake": 0.25,
    }
    assert pedals_taken(truck) == 0.25
    unfocused = dict.fromkeys(truck, 0.0)
    assert pedals_taken(unfocused) == 0.0, "a game that reads nothing still says so"


def test_no_speed_is_gathered_while_the_wheel_is_turned_for_a_corner() -> None:
    """Berlin: wheel near lock, planned speed up to 47 km/h, a lamp post at 20."""
    import math

    from core.control.lateral import ACTROS_GEOMETRY
    from tools.follow import TURNING_LATERAL_MPS2, turning_cap

    corner = turning_cap(ACTROS_GEOMETRY, -0.55)
    angle = ACTROS_GEOMETRY.angle_for_command(0.55)
    radius = ACTROS_GEOMETRY.wheelbase / math.tan(angle)
    assert corner == pytest.approx(math.sqrt(TURNING_LATERAL_MPS2 * radius))
    assert corner * 3.6 < 20.0
    assert turning_cap(ACTROS_GEOMETRY, 0.01) is None, "a motorway's bend"
    assert turning_cap(ACTROS_GEOMETRY, 0.0) is None
