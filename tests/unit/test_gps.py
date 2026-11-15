"""Following the game's GPS: the minimap's route, and turning where it turns."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from core.map import CompanyPlace
from core.planning.gps import (
    MINIMAP,
    REGION,
    GpsRouter,
    gps_turn,
    on_route,
    route_mask,
)
from core.planning.route import CHANGE, ONWARD, Route
from tests.unit.lane_networks import network, polyline

# Northwards along +z; +x is to the left at course 0 (core.control.lateral).
APPROACH = polyline((0, 0), (0, 100))
LEFT = polyline((0, 100), (40, 140))
RIGHT = polyline((0, 100), (-40, 140))
LEFT_ON = polyline((40, 140), (40, 400))
RIGHT_ON = polyline((-40, 140), (-40, 400))
NET = network(
    [APPROACH, LEFT, RIGHT, LEFT_ON, RIGHT_ON], successors=[(0, 1), (0, 2), (1, 3), (2, 4)]
)
POSE = (0.0, 60.0, 0.0)
PLANNED_LEFT = Route(NET, (0, 1, 3), (ONWARD, ONWARD))


def minimap_with_route_along(*lines: np.ndarray, pose=POSE) -> np.ndarray:
    """What `route_mask` would find if the GPS route ran along `lines`."""
    left, top, right, bottom = REGION
    mask = np.zeros((bottom - top, right - left), np.uint8)
    x, z, course = pose
    for line in lines:
        dx, dz = line[:, 0] - x, line[:, 2] - z
        forward = dx * math.sin(course) + dz * math.cos(course)
        rightward = -(dx * math.cos(course) - dz * math.sin(course))
        uv, ok = MINIMAP.project(rightward, forward)
        cv2.polylines(mask, [uv[ok].astype(np.int32)], False, 1, 9)
    return mask.astype(bool)


def test_the_route_s_red_is_told_from_the_roads_yellow() -> None:
    image = np.full((1080, 1920, 3), 30, np.uint8)
    left, top, _, _ = REGION
    image[top + 100 : top + 120, left + 50 : left + 70] = (0, 60, 230)  # route red
    image[top + 100 : top + 120, left + 200 : left + 220] = (60, 200, 255)  # road yellow
    mask = route_mask(image)
    assert mask[110, 60]
    assert not mask[110, 210]


def test_a_line_along_the_route_is_on_it() -> None:
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON)
    dx, dz = RIGHT[:, 0] - POSE[0], RIGHT[:, 2] - POSE[1]
    assert on_route(mask, -dx, dz) == pytest.approx(1.0)
    dx, dz = LEFT[:, 0] - POSE[0], LEFT[:, 2] - POSE[1]
    assert on_route(mask, -dx, dz) < 0.35


def test_where_the_gps_turns_the_other_way_it_is_found() -> None:
    """The first delivery: the route went left where the game's GPS went right."""
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON)
    turn = gps_turn(NET, PLANNED_LEFT, 0, POSE, mask)
    assert turn is not None
    assert (turn.index, turn.lane) == (0, 2)


def test_where_the_gps_agrees_nothing_is_changed() -> None:
    mask = minimap_with_route_along(APPROACH, LEFT, LEFT_ON)
    assert gps_turn(NET, PLANNED_LEFT, 0, POSE, mask) is None


def test_with_no_route_drawn_the_plan_stands() -> None:
    """No route on the minimap -- a menu over it, no job -- is no reason to turn."""
    mask = np.zeros((REGION[3] - REGION[1], REGION[2] - REGION[0]), bool)
    assert gps_turn(NET, PLANNED_LEFT, 0, POSE, mask) is None


def test_the_route_is_changed_only_when_the_gps_says_so_twice() -> None:
    depot = CompanyPlace("transinet", "torino", (-40.0, 395.0), ((-40.0, 395.0),), ())
    router = GpsRouter(NET, depot, PLANNED_LEFT)
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON)
    assert router.update(POSE, mask) is None
    rerouted = router.update(POSE, mask)
    assert rerouted is not None
    assert rerouted.lanes == (0, 2, 4)
    assert router.update(POSE, mask) is None, "and then it agrees"


def test_one_odd_minimap_does_not_change_the_route() -> None:
    depot = CompanyPlace("transinet", "torino", (-40.0, 395.0), ((-40.0, 395.0),), ())
    router = GpsRouter(NET, depot, PLANNED_LEFT)
    router.update(POSE, minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON))
    assert router.update(POSE, minimap_with_route_along(APPROACH, LEFT, LEFT_ON)) is None
    assert router.route.lanes == (0, 1, 3)


def test_a_drive_resumed_half_way_along_the_route_still_follows_the_gps() -> None:
    """Resumed 26 km in, the router looked for the truck only near the
    route's first lanes, never found it, and missed the GPS's turn."""
    straight = [polyline((0, 20 * k), (0, 20 * k + 20)) for k in range(20)]
    left = polyline((0, 400), (40, 440))
    right = polyline((0, 400), (-40, 440))
    left_on, right_on = polyline((40, 440), (40, 700)), polyline((-40, 440), (-40, 700))
    net = network(
        [*straight, left, right, left_on, right_on],
        successors=[(k, k + 1) for k in range(19)] + [(19, 20), (19, 21), (20, 22), (21, 23)],
    )
    planned = Route(net, (*range(20), 20, 22), (ONWARD,) * 21)
    depot = CompanyPlace("transinet", "torino", (-40.0, 695.0), ((-40.0, 695.0),), ())
    router = GpsRouter(net, depot, planned)
    pose = (0.0, 370.0, 0.0)
    assert router.locate(pose[0], pose[1]) == 18
    mask = minimap_with_route_along(straight[18], straight[19], right, right_on, pose=pose)
    router.update(pose, mask)
    rerouted = router.update(pose, mask)
    assert rerouted is not None
    assert rerouted.lanes[-2:] == (21, 23)


def test_a_packed_mask_comes_back_as_it_went() -> None:
    from core.planning.gps import unpack_mask

    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON)
    assert np.array_equal(unpack_mask(np.packbits(mask).tobytes()), mask)
    assert unpack_mask(b"") is None


def minimap_drawn(lines, camera, pose=POSE, thickness=9) -> np.ndarray:
    left, top, right, bottom = REGION
    mask = np.zeros((bottom - top, right - left), np.uint8)
    x, z, course = pose
    for line in lines:
        dx, dz = line[:, 0] - x, line[:, 2] - z
        forward = dx * math.sin(course) + dz * math.cos(course)
        rightward = -(dx * math.cos(course) - dz * math.sin(course))
        uv, ok = camera.project(rightward, forward)
        cv2.polylines(mask, [uv[ok].astype(np.int32)], False, 1, thickness)
    return mask.astype(bool)


@pytest.mark.parametrize("zoom", [1.0, 4.2])
def test_the_minimap_s_zoom_is_measured_from_its_roads(zoom: float) -> None:
    """In Milan at night the minimap was 4.2 times further out than in Turin."""
    from core.planning.gps import estimate_zoom

    roads = minimap_drawn([APPROACH, LEFT, RIGHT, LEFT_ON, RIGHT_ON], MINIMAP.zoomed(zoom))
    assert estimate_zoom(NET, POSE, roads) == pytest.approx(zoom, rel=0.1)


def test_zoomed_out_the_gps_turn_is_still_found() -> None:
    camera = MINIMAP.zoomed(4.2)
    mask = minimap_drawn([APPROACH, RIGHT, RIGHT_ON], camera, thickness=5)
    turn = gps_turn(NET, PLANNED_LEFT, 0, POSE, mask, camera)
    assert turn is not None and turn.lane == 2


# Both ways at the fork meet again, and the route goes on beyond.
MERGE = polyline((40, 400), (0, 440))
MERGE_RIGHT = polyline((-40, 400), (0, 440))
BEYOND = polyline((0, 440), (0, 1400))
MEETING = network(
    [APPROACH, LEFT, RIGHT, LEFT_ON, RIGHT_ON, MERGE, MERGE_RIGHT, BEYOND],
    successors=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)],
)
FAR_DEPOT = CompanyPlace("transinet", "torino", (0.0, 1395.0), ((0.0, 1395.0),), ())


def test_after_the_gps_s_turn_the_route_goes_back_onto_the_plan() -> None:
    """Planning afresh to a depot 80 km off took 4 s; meeting the planned
    route again takes milliseconds, and keeps the rest of it."""
    planned = Route(MEETING, (0, 1, 3, 5, 7), (ONWARD,) * 4)
    router = GpsRouter(MEETING, FAR_DEPOT, planned)
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON)
    router.update(POSE, mask)
    rerouted = router.update(POSE, mask)
    assert rerouted is not None
    assert rerouted.lanes == (0, 2, 4, 6, 7)
    assert rerouted.steps == (ONWARD,) * 4


def test_a_fork_the_truck_is_already_at_is_left_as_planned() -> None:
    """Changed there, the route came when the truck was in the junction."""
    depot = CompanyPlace("transinet", "torino", (-40.0, 395.0), ((-40.0, 395.0),), ())
    router = GpsRouter(NET, depot, PLANNED_LEFT)
    pose = (0.0, 92.0, 0.0)
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=pose)
    assert router.update(pose, mask) is None
    assert router.update(pose, mask) is None
    assert router.route.lanes == (0, 1, 3)


# A lane that becomes an exit: the right lane (x = -3.75) leaves for the
# exit at z = 300, the left goes on through.
THROUGH_LANE = polyline((0, 0), (0, 300))
EXIT_LANE = polyline((-3.75, 0), (-3.75, 300))
THROUGH_ON = polyline((0, 300), (0, 900))
EXIT_ON = polyline((-3.75, 300), (-120, 420), (-120, 900))
SPLIT = network(
    [THROUGH_LANE, EXIT_LANE, THROUGH_ON, EXIT_ON],
    successors=[(0, 2), (1, 3)],
    changes=[(0, 1)],
)
THROUGH_DEPOT = CompanyPlace("transinet", "torino", (0.0, 895.0), ((0.0, 895.0),), ())


def test_moving_into_a_lane_the_gps_does_not_take_is_seen() -> None:
    """A Danish motorway: the route moved into the right lane before it
    became the exit; with no fork in that lane the GPS was never asked. The
    minimap shows about 250 m ahead, so it is seen once the split is."""
    planned = Route(SPLIT, (0, 1, 3), (CHANGE, ONWARD))
    pose = (0.0, 150.0, 0.0)
    mask = minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose)
    turn = gps_turn(SPLIT, planned, 0, pose, mask)
    assert turn is not None
    assert (turn.index, turn.lane, turn.step) == (0, 2, ONWARD)


def test_in_a_lane_the_gps_does_not_take_the_route_moves_back_across() -> None:
    planned = Route(SPLIT, (1, 3), (ONWARD,))
    router = GpsRouter(SPLIT, THROUGH_DEPOT, planned)
    pose = (-3.75, 150.0, 0.0)
    mask = minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose)
    router.update(pose, mask)
    rerouted = router.update(pose, mask)
    assert rerouted is not None
    assert rerouted.lanes == (1, 0, 2)
    assert rerouted.steps == (CHANGE, ONWARD)


def test_lanes_the_gps_takes_both_of_are_left_alone() -> None:
    """Side by side the whole way, two lanes of one road are one red line."""
    planned = Route(SPLIT, (0, 1, 3), (CHANGE, ONWARD))
    pose = (0.0, 150.0, 0.0)
    mask = minimap_with_route_along(EXIT_LANE, EXIT_ON, pose=pose)
    assert gps_turn(SPLIT, planned, 0, pose, mask) is None


def test_where_the_lanes_part_too_close_the_plan_stands() -> None:
    planned = Route(SPLIT, (1, 3), (ONWARD,))
    pose = (-3.75, 270.0, 0.0)
    mask = minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose)
    assert gps_turn(SPLIT, planned, 0, pose, mask) is None


def test_a_lane_change_the_gps_asked_for_is_not_undone_at_once() -> None:
    """On a Danish motorway the route went back and forth between two lanes
    every second, the truck weaving under it."""
    router = GpsRouter(SPLIT, THROUGH_DEPOT, Route(SPLIT, (1, 3), (ONWARD,)))
    pose = (-3.75, 150.0, 0.0)
    through = minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose)
    router.update(pose, through, now=0.0)
    assert router.update(pose, through, now=0.5) is not None
    exit_way = minimap_with_route_along(EXIT_LANE, EXIT_ON, pose=pose)
    for now in (1.0, 1.5, 2.0, 2.5):
        assert router.update(pose, exit_way, now=now) is None
    assert router.route.lanes == (1, 0, 2)


def test_when_the_gps_s_way_finds_no_depot_off_the_refused_lanes_it_is_taken_anyway() -> None:
    """The exit lane's road comes back round to the one on; kept off all of it,
    the GPS's way found no depot, twelve times running, and the route stayed
    on the exit the GPS did not take."""
    exit_on = polyline((-3.75, 300), (-120, 420))
    after = polyline((-120, 420), (-120, 700))
    connector = polyline((0, 900), (-120, 420))
    net = network(
        [THROUGH_LANE, EXIT_LANE, THROUGH_ON, exit_on, connector, after],
        successors=[(0, 2), (1, 3), (2, 4), (4, 5), (3, 5)],
        changes=[(0, 1)],
    )
    depot = CompanyPlace("lkwlog", "dover", (-120.0, 695.0), ((-120.0, 695.0),), ())
    router = GpsRouter(net, depot, Route(net, (1, 3, 5), (ONWARD, ONWARD)))
    pose = (-3.75, 150.0, 0.0)
    mask = minimap_with_route_along(THROUGH_LANE, THROUGH_ON, pose=pose)
    router.update(pose, mask)
    rerouted = router.update(pose, mask)
    assert rerouted is not None
    assert rerouted.lanes == (1, 0, 2, 4, 5)
