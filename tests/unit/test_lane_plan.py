"""Which lane to be in along the game's road: the one the truck is in, and
a lane change only where the lane is about to leave the road, early enough."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from core.map.ferries import FerryLink
from core.planning.lanes import LEAD_PER_LANE_M, corridor, plan_lanes
from core.planning.nav_field import NavField
from core.planning.route import CHANGE, FERRY, ONWARD
from tests.unit.lane_networks import network, polyline

PIECE_M = 250.0
PIECES = 12  # 3 km
WIDTH_M = 3.75


def motorway(lanes: int, branches: list[tuple[int, np.ndarray]]):
    """`lanes` lanes northwards in pieces of `PIECE_M`, lane 0 the leftmost
    (x = 0) and the last the rightmost -- the exit side, at negative x --
    then a branch from the end of each lane named in `branches`, and a
    road on straight ahead from the end of every lane."""
    polylines, successors, changes = [], [], []

    def index(piece: int, lane: int) -> int:
        return piece * lanes + lane

    for piece in range(PIECES):
        for lane in range(lanes):
            x = -WIDTH_M * lane
            polylines.append(polyline((x, piece * PIECE_M), (x, (piece + 1) * PIECE_M)))
            if piece:
                successors.append((index(piece - 1, lane), index(piece, lane)))
            if lane:
                changes.append((index(piece, lane - 1), index(piece, lane)))
    end = PIECES * PIECE_M
    ahead = []
    for lane in range(lanes):
        x = -WIDTH_M * lane
        polylines.append(polyline((x, end), (x, end + 2000.0)))
        ahead.append(len(polylines) - 1)
        successors.append((index(PIECES - 1, lane), ahead[-1]))
        if lane:
            changes.append((ahead[-2], ahead[-1]))
    ends = []
    for lane, points in branches:
        polylines.append(points)
        ends.append(len(polylines) - 1)
        successors.append((index(PIECES - 1, lane), ends[-1]))
    return network(polylines, successors=successors, changes=changes), index, ahead, ends


END = PIECES * PIECE_M
RIGHT_EXIT = polyline((-7.5, END), (-60.0, END + 100), (-400.0, END + 200))
LEFT_EXIT = polyline((0.0, END), (60.0, END + 100), (400.0, END + 200))


def changes_at(route, net) -> list[float]:
    """World metres from the start at which each lane change's lane ends."""
    at, out = 0.0, []
    for i, lane in enumerate(route.lanes):
        if i == 0 or route.steps[i - 1] != CHANGE:
            at += float(net.lane_length[lane])
        if i < len(route.steps) and route.steps[i] == CHANGE:
            out.append(at)
    return out


def test_in_the_exit_side_lane_an_exit_ahead_takes_no_lane_change() -> None:
    net, index, _, (exit_lane,) = motorway(3, [(2, RIGHT_EXIT)])
    field = NavField.to_goals(net, {exit_lane: 0.0})
    route = plan_lanes(net, field, index(0, 2), horizon=10_000.0)
    assert route.lane_changes == 0
    assert route.lanes[-1] == exit_lane


def test_straight_on_the_truck_stays_in_the_exit_side_lane_past_exits() -> None:
    """The fault driven: changes for nothing, each one braking at the gate."""
    net, index, ahead, _ = motorway(3, [(2, RIGHT_EXIT), (0, LEFT_EXIT)])
    field = NavField.to_goals(net, {ahead[2]: 0.0, ahead[1]: 0.0, ahead[0]: 0.0})
    route = plan_lanes(net, field, index(0, 2), horizon=10_000.0)
    assert route.lane_changes == 0


@pytest.mark.parametrize("lane", [0, 1])
def test_with_nothing_to_leave_it_stays_in_the_lane_it_is_in(lane: int) -> None:
    """The driver's rule (11 November): no lane change unless the way calls
    for one. Each is blind to what comes up behind; moving back to the exit
    lane after a junction had been the rule since 24 October."""
    net, index, ahead, _ = motorway(3, [])
    field = NavField.to_goals(net, {ahead[2]: 0.0, ahead[1]: 0.0, ahead[0]: 0.0})
    route = plan_lanes(net, field, index(0, lane), horizon=10_000.0)
    assert route.lane_changes == 0
    assert route.lanes[-1] == ahead[lane]


def test_a_change_needed_past_a_fork_does_not_bring_the_fork_s_forward() -> None:
    """By Helsingborg: the right lane forked off 1.4 km on, and the lane on
    past the fork ended later still. Counted as two lanes to cross before
    the fork, the move across began 1.2 km early -- and a vehicle coming up
    behind in the lane moved into hit the trailer."""
    fork, on = 1500.0, 3000.0
    polylines, successors, changes = [], [], []
    left = right = later = None
    for k in range(int(fork // PIECE_M)):  # two lanes up to the fork
        polylines += [polyline((0, k * PIECE_M), (0, (k + 1) * PIECE_M))]
        polylines += [polyline((-WIDTH_M, k * PIECE_M), (-WIDTH_M, (k + 1) * PIECE_M))]
        if left is not None:
            successors += [(left, len(polylines) - 2), (right, len(polylines) - 1)]
        left, right = len(polylines) - 2, len(polylines) - 1
        changes.append((left, right))
    polylines.append(polyline((-WIDTH_M, fork), (-300.0, fork + 200)))  # the right lane forks off
    successors.append((right, len(polylines) - 1))
    first_right = 1
    for k in range(int(fork // PIECE_M), int(on // PIECE_M)):  # two lanes again past it
        polylines += [polyline((0, k * PIECE_M), (0, (k + 1) * PIECE_M))]
        polylines += [polyline((-WIDTH_M, k * PIECE_M), (-WIDTH_M, (k + 1) * PIECE_M))]
        successors.append((left, len(polylines) - 2))
        if later is not None:
            successors.append((later, len(polylines) - 1))
        left, later = len(polylines) - 2, len(polylines) - 1
        changes.append((left, later))
    polylines.append(polyline((0, on), (300.0, on + 200)))  # and the left lane then forks off
    successors.append((left, len(polylines) - 1))
    polylines.append(polyline((-WIDTH_M, on), (-WIDTH_M, on + 500)))
    goal = len(polylines) - 1
    successors.append((later, goal))
    net = network(polylines, successors=successors, changes=changes)
    field = NavField.to_goals(net, {goal: 0.0})
    route = plan_lanes(net, field, first_right, horizon=10_000.0)
    assert route.lanes[-1] == goal
    first, second = changes_at(route, net)
    assert fork - first >= LEAD_PER_LANE_M, "across in good time"
    assert fork - first <= LEAD_PER_LANE_M + PIECE_M, "and not before"
    assert on - second <= LEAD_PER_LANE_M + PIECE_M


def test_for_a_branch_off_the_far_side_it_moves_across_in_good_time_and_not_before() -> None:
    net, index, _, (branch,) = motorway(3, [(0, LEFT_EXIT)])
    field = NavField.to_goals(net, {branch: 0.0})
    route = plan_lanes(net, field, index(0, 2), horizon=10_000.0)
    assert route.lane_changes == 2
    assert route.lanes[-1] == branch
    first, second = changes_at(route, net)
    assert END - second >= LEAD_PER_LANE_M  # the last done 500 m before the split
    assert END - first >= 2 * LEAD_PER_LANE_M
    assert END - first <= 2 * LEAD_PER_LANE_M + 2 * PIECE_M  # and kept right till then


def test_one_lane_change_per_stretch_of_lane() -> None:
    net, index, _, (branch,) = motorway(3, [(0, LEFT_EXIT)])
    field = NavField.to_goals(net, {branch: 0.0})
    route = plan_lanes(net, field, index(0, 2), horizon=10_000.0)
    assert all(not (a == CHANGE and b == CHANGE) for a, b in itertools.pairwise(route.steps))


def test_the_corridor_is_the_game_s_road_and_not_a_slower_one_beside_it() -> None:
    net, index, _, (right, left) = motorway(2, [(1, RIGHT_EXIT), (0, LEFT_EXIT)])
    speeds = np.full(len(net), 72.0, dtype=np.float32)
    speeds[left] = 20.0
    net = replace(net, lane_nav_speed=speeds)
    field = NavField.to_goals(net, {right: 0.0})
    road = corridor(net, field, index(0, 1), horizon=10_000.0)
    assert right in road.lanes
    assert left not in road.lanes


def test_the_plan_ends_at_the_horizon() -> None:
    net, index, ahead, _ = motorway(2, [])
    field = NavField.to_goals(net, {ahead[1]: 0.0, ahead[0]: 0.0})
    route = plan_lanes(net, field, index(0, 1), horizon=1000.0)
    assert route.length == pytest.approx(1000.0, abs=PIECE_M + 5.0)


def test_where_the_road_goes_on_by_sea_the_plan_ends_with_the_crossing() -> None:
    shore = polyline((0, 0), (0, 300))
    quay = polyline((0, 300), (0, 360))
    landing = polyline((5000, 0), (5000, 300))
    net = network([shore, quay, landing], successors=[(0, 1)])
    link = FerryLink(1, 2, 3000.0, "west", "east", False, 30.0)
    net = replace(net, ferry_links={1: (link,)})
    field = NavField.to_goals(net, {2: 0.0})
    route = plan_lanes(net, field, 0, horizon=10_000.0)
    assert route.lanes == (0, 1, 2)
    assert route.steps == (ONWARD, FERRY)


def test_where_only_two_changes_in_one_stretch_reach_the_road_it_makes_them() -> None:
    """Three lanes in one stretch, only the far one going on: at job5-06's
    start the plan found no way at all while one change a stretch was law."""
    lanes = [polyline((-WIDTH_M * k, 0), (-WIDTH_M * k, 300)) for k in range(3)]
    on = polyline((0, 300), (0, 600))
    net = network([*lanes, on], successors=[(0, 3)], changes=[(0, 1), (1, 2)])
    field = NavField.to_goals(net, {3: 0.0})
    route = plan_lanes(net, field, 2, horizon=10_000.0)
    assert route.lanes == (2, 1, 0, 3)
    assert route.steps == (CHANGE, CHANGE, ONWARD)


def test_a_slower_branch_is_another_road_whatever_the_slack() -> None:
    """In Bern the way once round a roundabout and back, 10 s slower than
    the game's, was in the corridor; the plan took it, and turned the truck
    back on its way."""
    from core.planning.lanes import BRANCH_SLACK_S, SLACK_S

    lanes = [
        polyline((0, 0), (0, 100)),
        polyline((0, 100), (0, 300)),
        polyline((0, 100), (20, 200), (0, 300)),  # 5 s slower
        polyline((0, 300), (0, 400)),
    ]
    net = network(lanes, successors=[(0, 1), (0, 2), (1, 3), (2, 3)])
    net = replace(net, lane_nav_speed=np.full(len(lanes), 50.0, dtype=np.float32))
    field = NavField.to_goals(net, {3: 0.0})
    slower = float(field.time[2]) - float(field.time[1])
    assert BRANCH_SLACK_S < slower < SLACK_S, "within the slack, but another way"
    road = corridor(net, field, 0, horizon=10_000.0)
    assert 2 not in road.lanes
    assert plan_lanes(net, field, 0, horizon=10_000.0).lanes == (0, 1, 3)


def test_where_traffic_keeps_left_it_stays_in_its_lane_too() -> None:
    net, index, ahead, _ = motorway(3, [])
    net = replace(net, lane_left_hand=np.ones(len(net), dtype=bool))
    field = NavField.to_goals(net, {ahead[2]: 0.0, ahead[1]: 0.0, ahead[0]: 0.0})
    route = plan_lanes(net, field, index(0, 2), horizon=10_000.0)
    assert route.lane_changes == 0
