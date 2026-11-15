"""Changing lanes only into a lane that is clear."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.control.lane_change import (
    BEHIND_CLEAR_M,
    DECELERATION,
    HOLD_M,
    LATEST_FINISH_M,
    STOP_BELOW_MPS,
    LaneChangeGate,
    lane_changes,
)
from core.localization.path import ReferencePath
from core.perception.camera import GroundCamera
from core.perception.service import TrackReport
from core.planning.route import CHANGE, ONWARD, Route, change_begins
from tests.unit.lane_networks import network, polyline

CAMERA = GroundCamera(focal_px=819.0, horizon_row=438.0, centre_col=960.0, height_m=2.88)

# Northwards along +z; +x is to the left at course 0, so the lane on the
# right runs along x = -3.75.
APPROACH = polyline((0, 0), (0, 300))
OWN = polyline((0, 300), (0, 600))
RIGHT = polyline((-3.75, 300), (-3.75, 600))
NET = network([APPROACH, OWN, RIGHT], successors=[(0, 1)], changes=[(1, 2)])
ROUTE = Route(NET, (0, 1, 2), (ONWARD, CHANGE))
PATH = ReferencePath(ROUTE.points()[:, [0, 2]], spacing=2.0)
START = 300.0 + change_begins(300.0)
SPEED = 20.0


def beside(distance: float, lateral: float = 3.75, closing: float = 0.0, track_id: int = 7):
    return TrackReport(track_id, "truck", True, True, distance, closing, lateral, 2.5)


def gate() -> LaneChangeGate:
    return LaneChangeGate(lane_changes(ROUTE, PATH), CAMERA, PATH)


def step(
    g: LaneChangeGate,
    progress: float,
    tracks=(),
    unseen=(),
    now: float = 0.0,
    speed: float = SPEED,
    x: float = 0.0,
):
    return g.step(
        progress=progress,
        speed=speed,
        pose=(x, progress, 0.0),
        tracks=tracks,
        unseen=unseen,
        now=now,
    )


def test_a_lane_change_starts_late_in_the_lane_it_leaves() -> None:
    (change,) = lane_changes(ROUTE, PATH)
    assert change.start == pytest.approx(380.0, abs=2.0)
    across = PATH.sample_at(change.start + 60.0)
    assert -3.75 < across.x < 0.0, "the line is moving across there"
    before = PATH.sample_at(change.start - 5.0)
    assert before.x == pytest.approx(0.0, abs=0.05), "and not before"
    assert change.side == 1


def test_a_vehicle_in_the_lane_moved_into_holds_the_truck_short_of_it() -> None:
    """On a Danish motorway the line eased onto a lorry being overtaken. The
    lane here ends, so the change cannot be missed: the truck slows to wait
    where it must begin at the latest, to be across 30 m before the end."""
    (change,) = lane_changes(ROUTE, PATH)
    assert not change.can_miss
    assert change.last_start == pytest.approx(600.0 - LATEST_FINISH_M - 120.0, abs=3.0)
    decision = step(gate(), 150.0, [beside(20.0)])
    room = change.last_start - 150.0
    assert decision.target_speed == pytest.approx(math.sqrt(2 * DECELERATION * room), abs=0.1)
    assert "right" in decision.reason


def test_a_clear_lane_is_moved_into_at_speed() -> None:
    assert step(gate(), 150.0).target_speed is None
    assert step(gate(), 150.0, [beside(20.0, lateral=0.0)]).target_speed is None, "own lane"
    assert step(gate(), 150.0, [beside(20.0, lateral=-3.75)]).target_speed is None, "other side"


def test_a_vehicle_well_ahead_in_the_other_lane_is_followed_after_the_change() -> None:
    assert step(gate(), 150.0, [beside(80.0)]).target_speed is None


def test_far_from_a_change_nothing_is_watched() -> None:
    assert step(gate(), 0.0, [beside(20.0)]).target_speed is None


def test_once_across_the_hold_point_the_change_goes_on() -> None:
    g = gate()
    assert step(g, START + 1.0).path is None, "clear at its start: begun, as planned"
    later = step(g, START + HOLD_M + 1.0, [beside(20.0)])
    assert later.target_speed is None and later.path is None


def test_a_vehicle_gone_out_of_view_beside_the_truck_is_still_there() -> None:
    """Being overtaken at 5 m/s from 10 m ahead, it falls back past the trailer
    in about 6.4 s; until then the lane is taken."""
    g = gate()
    step(g, 150.0, [beside(10.0, closing=-5.0)], now=0.0)
    assert step(g, 151.0, now=5.0).target_speed is not None
    gone = (10.0 + BEHIND_CLEAR_M) / 5.0
    assert step(g, 152.0, now=gone + 0.5).target_speed is None


def test_a_close_vehicle_with_no_distance_on_that_side_takes_the_lane() -> None:
    assert step(gate(), 150.0, unseen=[(3, 1500.0)]).target_speed is not None
    assert step(gate(), 150.0, unseen=[(3, 400.0)]).target_speed is None, "the other side"


def test_a_lorry_beside_the_truck_takes_the_lane_by_its_near_edge() -> None:
    """Seen side-on 10 m ahead, 9.4 m of box whose middle was 7.2 m to the
    right; its near edge was in the lane, and the truck moved into it."""
    lorry = TrackReport(3, "truck", True, True, 10.2, 0.0, 7.2, 9.4)
    assert step(gate(), 150.0, [lorry]).target_speed is not None


def test_a_wide_vehicle_ahead_in_the_truck_s_own_lane_leaves_the_other_lane_free() -> None:
    lead = TrackReport(4, "truck", True, True, 20.0, 0.0, 0.0, 6.0)
    assert step(gate(), 150.0, [lead]).target_speed is None


# A motorway fork: the truck's lane goes on north, the lane on its right
# leads off to the right after z = 600.
ON = polyline((0, 600), (0, 1500))
OFF = polyline((-3.75, 600), (-40, 900), (-40, 1500))
FORK = network(
    [APPROACH, OWN, RIGHT, ON, OFF], successors=[(0, 1), (1, 3), (2, 4)], changes=[(1, 2)]
)
FORK_ROUTE = Route(FORK, (0, 1, 2, 4), (ONWARD, CHANGE, ONWARD))
FORK_PATH = ReferencePath(FORK_ROUTE.points()[:, [0, 2]], spacing=2.0)


def fork_gate() -> LaneChangeGate:
    return LaneChangeGate(lane_changes(FORK_ROUTE, FORK_PATH), CAMERA, FORK_PATH)


def x_at(path: ReferencePath, z: float) -> float:
    return path.nearest(-1.0, z, near=z, window=40.0).x


def test_at_a_fork_a_change_can_wait_until_the_lanes_part() -> None:
    (change,) = lane_changes(FORK_ROUTE, FORK_PATH)
    assert change.can_miss
    assert change.last_start > change.start + 50.0
    assert change.last_start < 612.0 - LATEST_FINISH_M - 120.0 + 5.0


def test_on_a_fast_road_a_taken_lane_holds_the_line_and_slows_a_little() -> None:
    """On the A1 the gate braked for 14 km/h on the motorway; not slowed for
    at all, a change was missed twice running in England, a car beside."""
    from core.control.lane_change import MISS_SLOW_MPS

    g = fork_gate()
    approaching = step(g, START - 50.0, [beside(14.0)])
    assert "taken" in approaching.reason
    assert approaching.target_speed is None or approaching.target_speed >= MISS_SLOW_MPS
    held = step(g, START + 1.0, [beside(14.0)])
    assert held.target_speed is not None and held.target_speed >= MISS_SLOW_MPS
    late = step(g, START + 200.0, [beside(14.0)], speed=MISS_SLOW_MPS)
    assert late.target_speed is None or late.target_speed >= MISS_SLOW_MPS, "never stopped for"
    assert held.path is not None
    assert abs(x_at(held.path, START + 80.0)) < 0.05, "kept in the lane it is in"


def test_once_clear_the_change_is_made_from_there_over_its_whole_length() -> None:
    g = fork_gate()
    step(g, START + 1.0, [beside(14.0)])
    begin = START + 40.0
    begun = step(g, begin, now=20.0)
    assert begun.path is not None
    assert abs(x_at(begun.path, begin + 2.0)) < 0.05
    assert x_at(begun.path, begin + 60.0) == pytest.approx(-3.75 / 2, abs=0.2)
    assert x_at(begun.path, begin + 125.0) == pytest.approx(-3.75, abs=0.05)
    xs = [x_at(begun.path, z) for z in np.arange(begin, begin + 120.0, 2.0)]
    planned = [x_at(FORK_PATH, z) for z in np.arange(START, START + 120.0, 2.0)]
    steepest = np.abs(np.diff(planned)).max()
    assert np.abs(np.diff(xs)).max() <= steepest * 1.05, "no steeper than the planned ease"
    (signal,) = g.signalled()
    assert signal.start == pytest.approx(begin)


def test_a_lane_never_clear_on_a_fast_road_is_missed_and_the_lane_driven_on() -> None:
    g = fork_gate()
    (change,) = g.changes
    step(g, START + 1.0, [beside(14.0)])
    missed = step(g, change.last_start + 1.0, [beside(14.0)], now=1.0)
    assert missed.missed and missed.target_speed is None
    assert g.signalled() == [], "no indicator for a change not made"
    later = step(g, change.last_start + 2.0, now=2.0)
    assert later.path is None, "a lane clear after it is let be"


def _parted(change) -> float:
    return change.latest + LATEST_FINISH_M + change.span


def test_a_clear_lane_past_the_latest_start_is_crossed_in_the_room_left() -> None:
    """On a delivery to Amsterdam a route taken 7 s after the last held a change begun
    just behind the truck at 50 km/h; the lane was empty and it was missed on
    the spot."""
    g = fork_gate()
    (change,) = g.changes
    step(g, START + 1.0, [beside(14.0)])
    begin = change.last_start + 10.0
    late = step(g, begin, now=20.0)
    assert not late.missed and late.target_speed is None
    assert late.path is not None
    room = _parted(change) - LATEST_FINISH_M - begin
    assert abs(x_at(late.path, begin + 2.0)) < 0.05
    assert x_at(late.path, begin + room / 2.0) == pytest.approx(-3.75 / 2, abs=0.2)
    assert x_at(late.path, begin + room + 2.0) == pytest.approx(-3.75, abs=0.05)
    (signal,) = g.signalled()
    assert signal.start == pytest.approx(begin)


def test_a_late_change_is_never_quicker_than_its_least_time() -> None:
    from core.control.lane_change import LATE_CHANGE_MIN_M, LATE_CHANGE_MIN_S

    g = fork_gate()
    (change,) = g.changes
    room = LATE_CHANGE_MIN_S * 25.0 - 1.0
    begin = _parted(change) - LATEST_FINISH_M - room
    assert room > LATE_CHANGE_MIN_M and begin > change.last_start
    step(g, START + 1.0, [beside(14.0)], speed=25.0)
    missed = step(g, begin, now=20.0, speed=25.0)
    assert missed.missed
    assert "too late" in missed.reason and "never clear" not in missed.reason
    assert g.signalled() == []


def test_a_late_change_is_never_shorter_than_its_least_length() -> None:
    from core.control.lane_change import LATE_CHANGE_MIN_M

    g = fork_gate()
    (change,) = g.changes
    begin = _parted(change) - LATEST_FINISH_M - (LATE_CHANGE_MIN_M - 1.0)
    step(g, START + 1.0, [beside(14.0)])
    assert step(g, begin, now=20.0, speed=STOP_BELOW_MPS + 0.5).missed


def test_a_lane_taken_past_the_latest_start_says_so_when_missed() -> None:
    g = fork_gate()
    (change,) = g.changes
    step(g, START + 1.0, [beside(14.0)])
    missed = step(g, change.last_start + 10.0, [beside(14.0)], now=1.0)
    assert missed.missed and "taken" in missed.reason


def test_a_late_change_seen_taken_eases_back_over_its_own_length() -> None:
    g = fork_gate()
    (change,) = g.changes
    step(g, START + 1.0, [beside(14.0)])
    begin = change.last_start + 10.0
    assert step(g, begin, now=20.0).path is not None
    back = step(g, begin + 10.0, [beside(14.0)], now=21.0)
    assert back.path is not None
    assert abs(x_at(back.path, begin + 10.0 + 70.0)) < 0.05
    assert -0.6 < x_at(back.path, begin + 12.0) <= 0.0


def test_slow_a_taken_lane_is_waited_for_at_the_latest_start() -> None:
    g = fork_gate()
    (change,) = g.changes
    slow = STOP_BELOW_MPS - 1.0
    held = step(g, START + 1.0, [beside(14.0)], speed=slow)
    room = change.last_start - START - 1.0
    assert held.target_speed == pytest.approx(math.sqrt(2 * DECELERATION * room), abs=0.1)
    waiting = step(g, change.last_start + 1.0, [beside(14.0)], speed=0.5, now=1.0)
    assert not waiting.missed and waiting.target_speed == 0.0


def test_a_lane_taken_for_long_is_given_up_on_and_the_truck_goes_on() -> None:
    """At a light in Aberdeen a lorry stood through two greens in the lane the
    route wanted, and the truck stood beside it at the green, for good."""
    from core.control.lane_change import WAIT_AT_MOST_S

    g = fork_gate()
    (change,) = g.changes
    at = change.last_start + 1.0
    assert not step(g, at, [beside(14.0)], speed=0.0, now=1.0).missed
    held = step(g, at, [beside(14.0)], speed=0.0, now=1.0 + WAIT_AT_MOST_S - 1.0)
    assert not held.missed and held.target_speed == 0.0
    given_up = step(g, at, [beside(14.0)], speed=0.0, now=1.0 + WAIT_AT_MOST_S + 1.0)
    assert given_up.missed and given_up.target_speed is None
    assert abs(x_at(given_up.path or g.path, at + 60.0)) < 0.05 or given_up.path is None
    assert step(g, at, [beside(14.0)], speed=0.0, now=2.0 + WAIT_AT_MOST_S).target_speed is None


def test_moving_with_a_lane_taken_is_not_standing_for_it() -> None:
    from core.control.lane_change import WAIT_AT_MOST_S

    g = fork_gate()
    (change,) = g.changes
    at = change.last_start - 5.0
    assert not step(g, at, [beside(14.0)], speed=0.0, now=1.0).missed
    step(g, at, [beside(14.0)], speed=3.0, now=1.0 + WAIT_AT_MOST_S / 2.0)
    late = step(g, at, [beside(14.0)], speed=0.0, now=2.0 + WAIT_AT_MOST_S)
    assert not late.missed, "the wait began again"


def test_a_lane_seen_taken_just_after_the_change_began_is_eased_back_to() -> None:
    g = fork_gate()
    assert step(g, START + 1.0).path is None
    back = step(g, START + 10.0, [beside(14.0)], now=1.0)
    assert back.path is not None
    assert abs(x_at(back.path, START + 10.0 + 70.0)) < 0.05
    assert -0.3 < x_at(back.path, START + 12.0) <= 0.0


def test_a_route_whose_change_began_behind_the_truck_is_held_in_its_lane() -> None:
    """On the A1 the red line took the fork's other branch with the change's
    start 33 m behind the truck: eased across in 80 m, onto a car."""
    g = fork_gate()
    held = g.adopt(START + 33.0, 0.0, START + 33.0)
    assert held is not None
    assert abs(x_at(held, START + 60.0)) < 0.05
    clear = step(g, START + 35.0)
    assert clear.path is not None, "and across from there once the lane is clear"
    assert x_at(clear.path, START + 35.0 + 60.0) == pytest.approx(-3.75 / 2, abs=0.2)


def test_a_route_taken_with_the_truck_already_moving_across_is_not_pulled_back() -> None:
    g = fork_gate()
    assert g.adopt(START + 60.0, -1.9, START + 60.0) is None
    assert step(g, START + 70.0, [beside(14.0)]).path is None


def test_a_route_taken_with_a_change_wholly_behind_the_truck_leaves_it_be() -> None:
    """On a delivery to Amsterdam a route taken anew kept the old start; a
    change 1.5 km behind the truck was found taken and held, and the truck was
    put on its lane's line, 669 m from the path: the drive stopped."""
    g = fork_gate()
    (change,) = g.changes
    behind = change.start + change.span + 30.0
    assert g.adopt(behind, -3.75, behind) is None
    assert g.watching(behind) == 0, "no mirror watched for it"
    taken = step(g, behind, [beside(14.0)])
    assert taken.path is None and not taken.missed and taken.target_speed is None
    assert g.signalled() == [change], "the route's own line, across where it was"


def test_a_change_wholly_behind_on_a_new_gate_is_left_be_without_adopting() -> None:
    g = fork_gate()
    (change,) = g.changes
    behind = change.start + change.span + 30.0
    taken = step(g, behind, [beside(14.0)])
    assert taken.path is None and not taken.missed and taken.target_speed is None


def test_a_change_a_metre_across_goes_on_whatever_is_ahead_in_the_new_lane() -> None:
    """At Malmö, half way into the right lane, a car 14 m ahead in it turned
    the truck back into a mixer come up beside it in the lane it had left."""
    g = fork_gate()
    assert step(g, START + 1.0).path is None
    on = step(g, START + 10.0, [beside(14.0)], now=1.0, x=-1.2)
    assert on.path is None, "not eased back"
    assert "held" not in on.reason


def test_a_change_whose_lane_left_runs_into_a_hidden_road_cannot_be_missed() -> None:
    """Near Nuremberg the lane left at a motorway exit ran on into a road
    hidden from the game's map, closed by the barrier at the fork: the change
    missed, the truck hit the barrier."""
    from core.map import Lane, LaneGraph, RoadNetwork

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(Lane(points=polyline((0, 300), (0, 600)), source="road", owner="2"))
    graph.lanes.append(Lane(points=polyline((-3.75, 300), (-3.75, 600)), source="road", owner="2"))
    graph.lanes.append(
        Lane(points=polyline((0, 600), (0, 1200)), source="road", owner="3", hidden=True)
    )
    graph.lanes.append(Lane(points=polyline((-3.75, 600), (-60, 1200)), source="road", owner="4"))
    graph.connect(0, 1)
    graph.connect(1, 3)
    graph.connect(2, 4)
    graph.lane_changes[1].append(2)
    graph.lane_changes[2].append(1)
    net = RoadNetwork.compile(graph)
    route = Route(net, (0, 1, 2, 4), (ONWARD, CHANGE, ONWARD))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    (change,) = lane_changes(route, path)
    assert not change.can_miss

    net.lane_hidden[:] = False
    (open_road,) = lane_changes(Route(net, route.lanes, route.steps), path)
    assert open_road.can_miss, "the same lane left going on, the change can be missed"


def test_a_vehicle_coming_up_in_the_mirror_holds_the_change() -> None:
    """On the E6 by Helsingborg and in London a lorry coming up in the lane
    moved into hit the truck as it moved across: the camera ahead never sees
    it; the mirror on that side does."""
    from core.control.lane_change import MIRROR_NEAR

    g = gate()
    assert g.watching(150.0) == 1, "the mirror on the right"
    assert g.watching(0.0) == 0, "far from the change, none"
    near = (1, MIRROR_NEAR + 0.1)
    first = g.step(
        progress=150.0, speed=SPEED, pose=(0.0, 150.0, 0.0), tracks=(), now=0.0, mirror=near
    )
    assert first.target_speed is None, "once is not enough: a box can flicker"
    second = g.step(
        progress=151.0, speed=SPEED, pose=(0.0, 151.0, 0.0), tracks=(), now=0.2, mirror=near
    )
    assert second.target_speed is not None
    assert "mirror" in second.reason


def test_the_mirror_is_not_read_slow_nor_for_a_far_vehicle_nor_on_the_other_side() -> None:
    """In Bern the mirrors took trams and buses at a stop for traffic behind."""
    from core.control.lane_change import MIRROR_FROM_MPS, MIRROR_NEAR

    for speed, mirror in (
        (MIRROR_FROM_MPS - 1.0, (1, MIRROR_NEAR + 0.1)),
        (SPEED, (1, MIRROR_NEAR / 2)),
        (SPEED, (-1, MIRROR_NEAR + 0.1)),
    ):
        g = gate()
        for k in range(4):
            decision = g.step(
                progress=150.0,
                speed=speed,
                pose=(0.0, 150.0, 0.0),
                tracks=(),
                now=0.2 * k,
                mirror=mirror,
            )
        assert "mirror" not in decision.reason


def _mirror_step(g, progress, now, size, side=1, x=0.0, speed=SPEED):
    return g.step(
        progress=progress,
        speed=speed,
        pose=(x, progress, 0.0),
        tracks=(),
        now=now,
        mirror_far=(side, size),
    )


def test_a_vehicle_coming_up_fast_far_back_in_the_mirror_holds_the_change() -> None:
    """On a delivery to Amsterdam a car came up at speed in the lane moved
    into: small in the mirror, never `MIRROR_NEAR`, and the truck moved
    across into it."""
    g = gate()
    seen = ((-1.79, 0.026), (-1.54, 0.032), (-1.30, 0.033), (-1.10, 0.053))  # as it came
    decision = None
    for k, (now, size) in enumerate(seen):
        decision = _mirror_step(g, 150.0 + k, now, size)
    assert decision is not None and decision.target_speed is not None
    assert "coming up fast" in decision.reason


def test_a_vehicle_keeping_its_distance_in_the_mirror_holds_nothing() -> None:
    g = gate()
    decision = None
    for k, size in enumerate((0.06, 0.062, 0.059, 0.061, 0.06, 0.062)):
        decision = _mirror_step(g, 150.0 + k, 0.2 * k, size)
    assert decision is not None and "mirror" not in decision.reason


def test_a_vehicle_coming_up_fast_turns_a_change_back_past_the_hold_point() -> None:
    g = gate()
    assert step(g, START + 1.0).path is None, "begun"
    at = START + HOLD_M + 15.0
    back = None
    for k, size in enumerate((0.03, 0.04, 0.052, 0.068, 0.088)):
        back = _mirror_step(g, at + k, 1.0 + 0.25 * k, size, x=-1.2)
        if back.path is not None:
            break
    assert back is not None and back.path is not None, "eased back"
    assert "coming up fast" in back.reason
    assert abs(x_at(back.path, at + 80.0)) < 0.05, "back in the lane left"


def test_well_across_a_vehicle_coming_up_fast_does_not_turn_it_back() -> None:
    from core.control.lane_change import MIRROR_BACK_M

    g = gate()
    assert step(g, START + 1.0).path is None
    at = START + HOLD_M + 40.0
    for k, size in enumerate((0.03, 0.04, 0.052, 0.068, 0.088)):
        decision = _mirror_step(g, at + k, 1.0 + 0.25 * k, size, x=-(MIRROR_BACK_M + 0.2))
        assert decision.path is None


def test_past_the_hold_point_a_vehicle_ahead_in_the_new_lane_does_not_turn_it_back() -> None:
    g = gate()
    assert step(g, START + 1.0).path is None
    later = step(g, START + HOLD_M + 5.0, [beside(14.0)], x=-0.2)
    assert later.path is None and later.target_speed is None


def test_the_mirror_is_watched_until_the_change_is_across() -> None:
    g = gate()
    (change,) = g.changes
    assert step(g, START + 1.0).path is None
    assert g.watching(START + HOLD_M + 10.0) == change.side
    step(g, START + change.span + 1.0)
    assert g.watching(START + change.span + 1.0) == 0


def test_a_box_jumping_to_another_vehicle_in_the_mirror_is_not_one_coming_up() -> None:
    """Looser, the rule fired on a box that went from a far car to a near
    one in a frame: two vehicles, neither closing fast."""
    g = gate()
    decision = None
    for k, size in enumerate((0.033, 0.036, 0.035, 0.038, 0.094)):
        decision = _mirror_step(g, 150.0 + k, 0.25 * k, size)
    assert decision is not None and "coming up fast" not in decision.reason
