"""Following the minimap's red line, with no route of our own."""

from __future__ import annotations

from core.map import CompanyPlace
from core.planning.gps_follow import GpsFollower
from core.planning.route import CHANGE, ONWARD
from tests.unit.test_gps import (
    APPROACH,
    EXIT_LANE,
    EXIT_ON,
    LEFT,
    LEFT_ON,
    NET,
    RIGHT,
    RIGHT_ON,
    SPLIT,
    THROUGH_LANE,
    THROUGH_ON,
    minimap_with_route_along,
)


def twice(follower: GpsFollower, pose, mask):
    follower.update(pose, mask, now=0.0)
    return follower.update(pose, mask, now=0.5)


def test_at_a_junction_the_route_goes_where_the_red_line_goes() -> None:
    pose = (0.0, 60.0, 0.0)
    follower = GpsFollower.ahead_of(NET, None, pose)
    for red, lanes in (((RIGHT, RIGHT_ON), (0, 2, 4)), ((LEFT, LEFT_ON), (0, 1, 3))):
        mask = minimap_with_route_along(APPROACH, *red, pose=pose)
        twice(follower, pose, mask)
        assert follower.route.lanes == lanes


def test_one_odd_minimap_does_not_turn_it() -> None:
    pose = (0.0, 60.0, 0.0)
    follower = GpsFollower.ahead_of(NET, None, pose)
    before = follower.route.lanes
    follower.update(pose, minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=pose))
    follower.update(pose, minimap_with_route_along(APPROACH, LEFT, LEFT_ON, pose=pose))
    assert follower.route.lanes == before


def test_where_the_red_line_leaves_from_the_lane_beside_the_route_moves_across() -> None:
    """In the middle lane, an exit off the outer one: across to it first."""
    pose = (0.0, 150.0, 0.0)
    follower = GpsFollower.ahead_of(SPLIT, None, pose)
    assert follower.route.lanes[:2] == (0, 2), "straight on, until the minimap says"
    twice(follower, pose, minimap_with_route_along(EXIT_LANE, EXIT_ON, pose=pose))
    assert follower.route.lanes[:3] == (0, 1, 3)
    assert follower.route.steps[:2] == (CHANGE, ONWARD)


def test_and_back_again_if_the_red_line_stays() -> None:
    pose = (0.0, 170.0, 0.0)
    follower = GpsFollower.ahead_of(SPLIT, None, pose)
    twice(follower, pose, minimap_with_route_along(EXIT_LANE, EXIT_ON, pose=pose))
    twice(follower, pose, minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose))
    assert follower.route.lanes[:2] == (0, 2)


def test_the_route_ends_at_the_depot() -> None:
    depot = CompanyPlace("transinet", "torino", (-40.0, 400.0), ((-40.0, 400.0),), ())
    pose = (0.0, 60.0, 0.0)
    follower = GpsFollower.ahead_of(NET, depot, pose)
    twice(follower, pose, minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=pose))
    assert follower.route.lanes == (0, 2, 4)
    assert follower.finished


def test_a_turn_lane_that_runs_beside_the_road_before_it_turns_is_taken() -> None:
    """Near Calais a left turn ran beside the straight lane for 50 m before
    turning; compared over their first 50 m, both lay on the red line."""
    from tests.unit.lane_networks import network, polyline

    approach = polyline((0, 0), (0, 100))
    straight = polyline((0, 100), (0, 400))
    turn = polyline((0, 100), (3.0, 110), (3.0, 160), (80.0, 200))
    turn_on = polyline((80.0, 200), (300.0, 200))
    net = network([approach, straight, turn, turn_on], successors=[(0, 1), (0, 2), (2, 3)])
    pose = (0.0, 40.0, 0.0)
    follower = GpsFollower.ahead_of(net, None, pose)
    assert follower.route.lanes[:2] == (0, 1), "straight on, until the minimap says"
    twice(follower, pose, minimap_with_route_along(approach, turn, turn_on, pose=pose))
    assert follower.route.lanes[:3] == (0, 2, 3)


def test_a_branch_two_lanes_over_is_reached_by_moving_across_twice() -> None:
    """A Belgian motorway split: the red line took the branch off the
    outermost of three lanes; from the middle-left one, one lane over never
    reached it."""
    from tests.unit.lane_networks import network, polyline

    lanes = [polyline((-3.75 * k, 0), (-3.75 * k, 300)) for k in range(3)]
    straight_on = [polyline((-3.75 * k, 300), (-3.75 * k, 700)) for k in range(2)]
    branch = polyline((-7.5, 300), (-90.0, 420), (-90.0, 700))
    net = network(
        [*lanes, *straight_on, branch],
        successors=[(0, 3), (1, 4), (2, 5)],
        changes=[(0, 1), (1, 2)],
    )
    pose = (0.0, 150.0, 0.0)
    follower = GpsFollower.ahead_of(net, None, pose)
    assert follower.route.lanes[:2] == (0, 3)
    twice(follower, pose, minimap_with_route_along(lanes[2], branch, pose=pose))
    assert follower.route.lanes[:4] == (0, 1, 2, 5)
    assert follower.route.steps[:3] == (CHANGE, CHANGE, ONWARD)
