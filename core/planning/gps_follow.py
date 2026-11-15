"""Following the game's GPS as it is drawn, with no route of our own.

Re-planning to the depot whenever the GPS turned another way kept a route of
the planner's own beyond the turn, and where the planner and the game
disagreed -- which ferry to take to Sweden, which exit off a motorway -- the
truck was pulled back towards the planner's way, the game re-routed, and
the road grew longer. The driver wanted the GPS followed one to one.

So here the only route is the one the minimap draws. From the lane the truck
is on, the road ahead is taken the straightest way (`EXTEND_M` of it, for
the speed profile and the lights), and at each junction the minimap shows,
the branch the red line takes replaces it: onward from the lane the truck
is in, or from a lane beside it, moved into before the junction. Beyond what
the minimap shows the way stays provisional, and is put right as it comes
into view.

The route ends at a gate of the job's depot, or at a port -- short of its
booths (`Route.to_ferry`); the ship is taken from the game's menu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from core.map import CompanyPlace, LaneMatcher, RoadNetwork
from core.planning.gps import (
    BRANCH_M,
    LANE_LATE_M,
    LATE_M,
    LOCATE_AHEAD,
    LOCATE_BEHIND,
    LOCATE_RADIUS_M,
    LOOK_AHEAD_M,
    MINIMAP,
    OFF_ROUTE,
    ON_ROUTE,
    ZOOM_EVERY_S,
    MinimapCamera,
    _apart,
    _score,
    _turn,
    _way_on,
    estimate_zoom,
)
from core.planning.route import CHANGE, FERRY, ONWARD, Route, RouteError

__all__ = ["GpsFollower", "red_line_turn"]

EXTEND_M: Final = 3000.0
"""Road kept ahead of the truck, the straightest way where nothing says otherwise."""

EXTEND_WHEN_M: Final = 1500.0
"""The road ahead is extended again once less than this is left."""

WAY_M: Final = 200.0
"""How far each way on from a junction is followed to see where it parts."""

ARRIVE_M: Final = 60.0
"""A lane ending this near a gate of the job's depot is where the route ends."""


@dataclass
class GpsFollower:
    """Keeps a route that goes where the minimap's red line goes.

    Has the interface of `core.planning.gps.GpsRouter` -- `update` returns a
    new route or None -- so the drive's GPS worker takes either.
    """

    network: RoadNetwork
    depot: CompanyPlace | None
    route: Route
    camera: MinimapCamera = MINIMAP
    at: int = 0
    finished: bool = False
    _pending: tuple[int, tuple[int, ...], int] | None = None
    _zoom_at: float | None = None

    @classmethod
    def ahead_of(
        cls, network: RoadNetwork, depot: CompanyPlace | None, pose: tuple[float, float, float]
    ) -> GpsFollower:
        """Starting on the lane the truck is on, the straightest way on."""
        match = LaneMatcher(network).update(*pose)
        if match is None:
            raise RouteError(f"start {pose[:2]} is not on the network")
        follower = cls(network, depot, Route(network, (match.lane,), ()))
        follower._extend()
        return follower

    @classmethod
    def along(cls, network: RoadNetwork, depot: CompanyPlace | None, route: Route) -> GpsFollower:
        """Starting from a route already planned -- the road ahead -- extended."""
        follower = cls(network, depot, route)
        follower._extend()
        return follower

    # ------------------------------------------------------------------ #

    def locate(self, x: float, z: float) -> int:
        """Where along the route the truck is, as an index into its lanes."""
        lanes = self.route.lanes
        near = range(max(0, self.at - LOCATE_BEHIND), min(len(lanes), self.at + LOCATE_AHEAD))
        for candidates in (near, range(len(lanes))):
            best: tuple[int, float] | None = None
            for i in candidates:
                projection = self.network.project(lanes[i], x, z)
                if projection.overshoot < 1.0 and (best is None or projection.distance < best[1]):
                    best = (i, projection.distance)
            if best is not None and best[1] < LOCATE_RADIUS_M:
                self.at = best[0]
                break
        return self.at

    def update(
        self,
        pose: tuple[float, float, float],
        mask: NDArray[np.bool_],
        roads: NDArray[np.bool_] | None = None,
        now: float = 0.0,
    ) -> Route | None:
        """The route anew if the minimap has put it right, or it has been
        extended; None if it stands."""
        if roads is not None and (self._zoom_at is None or now - self._zoom_at >= ZOOM_EVERY_S):
            zoom = estimate_zoom(self.network, pose, roads)
            if zoom is not None:
                self.camera = MINIMAP.zoomed(zoom)
                self._zoom_at = now
        at = self.locate(pose[0], pose[1])
        changed = False
        choice = self._red_line(at, pose, mask)
        if choice is None:
            self._pending = None
        elif self._pending != choice:
            self._pending = choice  # seen once; twice running to be believed
        else:
            self._pending = None
            self._take(*choice)
            changed = True
        if not self.finished and self._ahead(at) < EXTEND_WHEN_M:
            self._extend()
            changed = True
        return self.route if changed else None

    # ------------------------------------------------------------------ #

    def _red_line(
        self, at: int, pose: tuple[float, float, float], mask: NDArray[np.bool_]
    ) -> tuple[int, tuple[int, ...], int] | None:
        return red_line_turn(self.network, self.route, at, pose, mask, self.camera)

    def _take(self, i: int, via: tuple[int, ...], onward: int) -> None:
        """The route up to lane `i`, then across through the lanes `via` (none
        to go on from lane `i` itself), then on to `onward`; then the
        straightest way on."""
        lanes, steps = self.route.lanes, self.route.steps
        # Moving across into lane i already, from a lane the new way goes
        # through: go back no further than that lane.
        start = i
        while via and start > 0 and steps[start - 1] == CHANGE and lanes[start - 1] == via[0]:
            start, via = start - 1, via[1:]
        head = (*lanes[: start + 1], *via, onward)
        head_steps = (*steps[:start], *([CHANGE] * len(via)), ONWARD)
        self.route = Route(self.network, head, head_steps)
        self.finished = False
        self._extend()

    def _ahead(self, at: int) -> float:
        lanes, steps = self.route.lanes, self.route.steps
        return sum(
            float(self.network.lane_length[lanes[k]])
            for k in range(at, len(lanes))
            if k == at or steps[k - 1] == ONWARD
        )

    def _extend(self) -> None:
        """The straightest way on from the route's end, to `EXTEND_M` ahead of
        the truck; ending at the depot, or at a port."""
        network = self.network
        lanes, steps = list(self.route.lanes), list(self.route.steps)
        covered = self._ahead(self.at)
        while covered < EXTEND_M:
            last = lanes[-1]
            if self._at_depot(last):
                self.finished = True
                break
            ferries = network.ferry_links.get(last, ())
            if ferries:
                lanes.append(ferries[0].land)
                steps.append(FERRY)
                self.finished = True
                break
            options = [int(o) for o in network.next_lanes(last)]
            if not options:
                beside = [
                    int(b) for b in network.lanes_alongside(last) if len(network.next_lanes(int(b)))
                ]
                if not beside:
                    self.finished = True  # the road ends
                    break
                lanes.append(beside[0])
                steps.append(CHANGE)
                continue
            points = network.lane_points(last)
            heading = math.atan2(
                float(points[-1][0] - points[-2][0]), float(points[-1][2] - points[-2][2])
            )
            shown = [o for o in options if not network.lane_hidden[o]] or options
            onward = min(shown, key=lambda o, h=heading: _turn(network, o, h))
            lanes.append(onward)
            steps.append(ONWARD)
            covered += float(network.lane_length[onward])
        self.route = Route(network, tuple(lanes), tuple(steps))

    def _at_depot(self, lane: int) -> bool:
        if self.depot is None:
            return False
        end = self.network.lane_points(lane)[-1]
        return any(
            math.hypot(float(end[0]) - gx, float(end[2]) - gz) < ARRIVE_M
            for gx, gz in self.depot.gates
        )


def red_line_turn(
    network: RoadNetwork,
    route: Route,
    at: int,
    pose: tuple[float, float, float],
    mask: NDArray[np.bool_],
    camera: MinimapCamera = MINIMAP,
) -> tuple[int, tuple[int, ...], int] | None:
    """The first junction ahead of `route.lanes[at]` where the minimap's red
    line leaves `route`, as (index of the lane before it, lanes to move
    across through, lane on); None."""
    lanes, steps = route.lanes, route.steps
    x, z = pose[0], pose[1]
    for i in range(at, len(lanes) - 1):
        lane = lanes[i]
        end = network.lane_points(lane)[-1]
        # From the truck, not from the start of the lane it is on: on a
        # long lane the junction at the end of the one beside it was
        # never looked at.
        to_fork = math.hypot(float(end[0]) - x, float(end[2]) - z)
        if to_fork > LOOK_AHEAD_M:
            break
        if steps[i] != ONWARD or to_fork < LATE_M:
            continue
        # Every way on from this lane or the lanes beside it -- and the
        # lanes beside those: on a Belgian motorway the red line took
        # the branch two lanes over, which one lane over never reached.
        ways: dict[int, tuple[int, ...]] = {int(o): () for o in network.next_lanes(lane)}
        for via in lanes_across(network, lane):
            for o in network.next_lanes(via[-1]):
                ways.setdefault(int(o), via)
        if len(ways) < 2:
            continue
        planned = lanes[i + 1]
        if planned not in ways:
            continue
        # Each way far enough to part from the planned one, and only the
        # parts apart compared: near Calais a left turn ran beside the
        # straight lane for 50 m before it turned, both lay on the red
        # line that far, and the turn was never taken.
        reach = max(WAY_M, BRANCH_M * max(1.0, camera.zoom))
        planned_way = _way_on(network, planned, reach)
        best: tuple[float, int] | None = None
        for option in ways:
            if option == planned:
                continue
            way = _way_on(network, option, reach)
            apart_way, apart_planned = _apart(way, planned_way), _apart(planned_way, way)
            if len(apart_way) == 0 or len(apart_planned) == 0:
                continue
            on_way = _score(apart_way, pose, mask, camera)
            on_planned = _score(apart_planned, pose, mask, camera)
            if on_way is None or on_planned is None:
                continue
            if (
                on_way >= ON_ROUTE
                and on_planned <= OFF_ROUTE
                and (best is None or on_way > best[0])
            ):
                best = (on_way, option)
        if best is None:
            continue
        option = best[1]
        if to_fork < LANE_LATE_M * len(ways[option]):
            continue  # too late to move across for it
        return i, ways[option], option
    return None


ACROSS_MAX: Final = 3
"""Lanes the route may move across for a junction the red line shows."""


def lanes_across(network: RoadNetwork, lane: int) -> list[tuple[int, ...]]:
    """The lanes reachable across from `lane`, each with the lanes moved
    through to get there, nearest first, up to `ACROSS_MAX` over."""
    found: list[tuple[int, ...]] = []
    seen = {lane}
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(ACROSS_MAX):
        nxt = []
        for via in frontier:
            last = via[-1] if via else lane
            for beside in network.lanes_alongside(last):
                beside = int(beside)
                if beside not in seen:
                    seen.add(beside)
                    nxt.append((*via, beside))
        found.extend(nxt)
        frontier = nxt
    return found
