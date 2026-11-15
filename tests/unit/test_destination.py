"""Where the game's GPS goes when nothing says: found from its distance,
and gone over to as the red line and the distance tell."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.map import CompanyPlace
from core.planning.destination import (
    GPS_COMPANY,
    DestinationSearch,
    candidates,
    gps_places,
    place_from,
    place_record,
)
from core.planning.game_route import CHECK_EVERY_S, GameRouter
from core.planning.nav_field import NavField
from tests.unit.lane_networks import network, polyline
from tests.unit.test_gps import minimap_with_route_along

# A fork: left to one depot 300 m on, right to another 700 m on.
APPROACH = polyline((0, 0), (0, 100))
LEFT = polyline((0, 100), (200, 140))
RIGHT = polyline((0, 100), (-200, 140))
LEFT_ON = polyline((200, 140), (200, 440))
RIGHT_ON = polyline((-200, 140), (-200, 840))
LANES = [APPROACH, LEFT, RIGHT, LEFT_ON, RIGHT_ON]
NEAR = CompanyPlace("near", "torino", (200.0, 440.0), ((200.0, 430.0),), ())
FAR = CompanyPlace("far", "torino", (-200.0, 840.0), ((-200.0, 830.0),), ())
ELSEWHERE = CompanyPlace("elsewhere", "milano", (900.0, 900.0), ((900.0, 900.0),), ())
POSE = (0.0, 40.0, 0.0)


@dataclass
class Places:
    companies: tuple

    def company(self, company: str, city: str):
        return next((c for c in self.companies if (c.company, c.city) == (company, city)), None)


def fork():
    return network(LANES, successors=[(0, 1), (0, 2), (1, 3), (2, 4)])


def game_distance(net, depot) -> float:
    from core.planning.game_route import depot_goals

    field = NavField.to_goals(net, depot_goals(net, depot))
    lane = 0
    return field.distance_at(lane, net.project(lane, POSE[0], POSE[1]).s)


def test_the_depot_as_far_as_the_game_says_comes_first() -> None:
    net = fork()
    places = Places((NEAR, FAR, ELSEWHERE))
    found = gps_places(net, places, POSE, game_distance(net, FAR))
    assert found[0] is FAR
    assert ELSEWHERE not in found


def test_a_point_that_fits_better_than_any_depot_comes_before_them() -> None:
    """In Bern the trailer waited 416 m from the nearest gate: the point fitted
    1.3 %, a depot 2.7 %, and the depot was driven to."""
    from core.planning.nav_field import nav_along

    net = fork()
    far = game_distance(net, FAR)
    # The right branch's end is 40 m short of FAR's gate; ask for 30 m past
    # the near depot's way, as far as no depot.
    asked = game_distance(net, NEAR) + nav_along(net, 3, 30.0)
    found = gps_places(net, Places((NEAR, FAR)), POSE, asked)
    assert found[0].company == GPS_COMPANY
    assert far != asked


def test_with_no_depot_that_far_the_points_on_the_roads_are_the_places() -> None:
    net = fork()
    places = Places((ELSEWHERE,))
    far = game_distance(net, FAR)
    found = gps_places(net, places, POSE, far)
    assert found and all(p.company == GPS_COMPANY for p in found)
    # On the right road's far end, not on the left one, which is shorter.
    assert any(abs(p.position[0] + 200.0) < 1.0 and p.position[1] > 700 for p in found)
    assert all(p.position[0] < 0 for p in found)


def test_candidates_narrow_as_the_truck_drives_one_way() -> None:
    from core.planning.nav_field import nav_along

    net = fork()
    near = game_distance(net, NEAR)
    asked = near - nav_along(net, 3, 100.0)  # 100 m short of the near depot's gate
    search = DestinationSearch(net, candidates(net, POSE, asked))
    assert len(search.found) == 2, "one on each branch"
    # Up the right branch 160 m: the game's distance is down by the way driven.
    on_right = (-200.0, 300.0, 0.0)
    driven = (
        nav_along(net, 0, 1e9)
        - nav_along(net, 0, POSE[1])
        + nav_along(net, 2, 1e9)
        + nav_along(net, 4, 160.0)
    )
    search.narrow(on_right, asked - driven)
    assert search.known is not None and search.known.x < 0


def test_a_place_round_trips_through_route_metadata() -> None:
    places = Places((NEAR,))
    assert place_from(place_record(NEAR), places) is NEAR
    point = CompanyPlace(GPS_COMPANY, "10,20", (10.0, 20.0), ((10.0, 20.0),), ())
    assert place_from(place_record(point), places) == point


def test_the_red_line_on_another_s_way_goes_over_to_it() -> None:
    net = fork()
    router = GameRouter.plan(net, NEAR, POSE, alternatives=(FAR,))
    assert router.depot is NEAR and router.route.lanes[-1] == 3
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    # Without the game's distance to go by: the red line alone.
    router.update(POSE, mask, now=0.0)
    assert router.depot is NEAR, "seen once is not enough"
    router.update(POSE, mask, now=0.5)
    assert router.depot is FAR
    assert router.route.lanes[-1] == 4
    assert NEAR in router.alternatives


def test_the_game_s_distance_fitting_another_goes_over_to_it() -> None:
    net = fork()
    far = game_distance(net, FAR)
    router = GameRouter.plan(net, NEAR, POSE, alternatives=(FAR,))
    assert router.route.lanes[-1] == 3
    route = router.update(POSE, now=0.0, nav_distance=far)
    assert router.depot is FAR
    assert route is not None and route.lanes[-1] == 4


def test_without_alternatives_nothing_changes_hands() -> None:
    net = fork()
    router = GameRouter.plan(net, NEAR, POSE)
    router.update(POSE, now=0.0, nav_distance=game_distance(net, FAR))
    router.update(POSE, now=CHECK_EVERY_S + 1.0, nav_distance=game_distance(net, FAR))
    assert router.depot is NEAR


def test_a_red_line_on_no_one_s_way_is_a_misread_said_once() -> None:
    """Driving in Bern, a red line on a way slow for every place flipped
    between being another's and a misread, and said so every update."""
    net = fork()
    elsewhere = CompanyPlace("x", "y", (0.0, 50.0), ((0.0, 50.0),), ())
    router = GameRouter.plan(net, NEAR, POSE, alternatives=(elsewhere,))
    router.network = net
    import numpy as np

    speeds = np.full(len(LANES), 72.0, dtype=np.float32)
    speeds[[2, 4]] = 2.0  # the right way minutes slower
    from dataclasses import replace

    router = GameRouter.plan(
        replace(net, lane_nav_speed=speeds), NEAR, POSE, alternatives=(elsewhere,)
    )
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    for k in range(6):
        router.update(POSE, mask, now=0.5 * k)
    said = [n for n in router.notes if "slower by the map: watched" in n]
    assert len(said) == 1
    assert router.depot is NEAR
    assert router.route.lanes[1] != 2, "a way to none of the places is never taken"


def test_near_the_end_the_red_line_goes_over_to_another_within_metres() -> None:
    """0.8 km from a trailer in Bern a share of the distance left 31 m, and
    the red line to the right place was turned away as a misread."""
    from core.planning.nav_field import nav_along

    net = fork()
    # Both places a few hundred metres up their branch: a short way to go.
    near_point = CompanyPlace("gps", "a", (200.0, 200.0), ((200.0, 200.0),), ())
    far_point = CompanyPlace("gps", "b", (-200.0, 220.0), ((-200.0, 220.0),), ())
    to_far = game_distance(net, far_point)
    router = GameRouter.plan(net, near_point, POSE, alternatives=(far_point,))
    router._nav = to_far + nav_along(net, 4, 3.0)  # a little off, as the map is
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    router.update(POSE, mask, now=0.0)
    router.update(POSE, mask, now=0.5)
    assert router.depot is far_point


def test_a_red_line_on_another_s_way_a_tenth_off_the_map_goes_over_to_it() -> None:
    """Off a job done in Bern the GPS went to a trailer at a depot whose way
    the map had 10.4 % short of the game's; at 4 % no place took the red
    line's way, and the truck turned back for the wrong one."""
    net = fork()
    to_far = game_distance(net, FAR)
    router = GameRouter.plan(net, NEAR, POSE, alternatives=(FAR,))
    router._nav = to_far * 1.10
    mask = minimap_with_route_along(APPROACH, RIGHT, RIGHT_ON, pose=POSE)
    router.update(POSE, mask, now=0.0)
    router.update(POSE, mask, now=0.5)
    assert router.depot is FAR


def test_the_game_s_distance_never_switches_to_a_place_behind_the_truck() -> None:
    """In Bern the point switched to by distance was 1 km back along the
    road just driven; the truck turned round for it."""
    net = fork()
    router = GameRouter.plan(net, NEAR, POSE, alternatives=(FAR,))
    fits = game_distance(net, FAR)
    facing_away = (POSE[0], POSE[1], math.pi)  # FAR now behind the truck
    assert router._switch_by_distance(facing_away, fits, 0) is None
    assert router.depot is NEAR
    assert router._switch_by_distance(POSE, fits, 0) is not None, "ahead: switched to"
    assert router.depot is FAR
