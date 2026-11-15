"""Where the game's GPS is going, when nothing says.

A job whose trailer waits at a depot is not told to the SDK until the
trailer is hitched (6 October: nothing, not even after a reload), and a
place marked on the game's map never is. The navigation says how far the
route is and how long it takes, in the game's metres and seconds, and
nothing more.

That is enough to look for it. The game's route is the quickest as it
reckons time (core.planning.nav_field); searched forward from the truck by
the same reckoning, the points whose quickest way is as long as the game's
distance *and* takes its time are where the route can end
(`candidates`). From one place there are a few -- a ring of points the
right distance away, the time picking those on roads of the right speeds
-- and as the truck drives, each new reading of the distance keeps only
those still that far (`DestinationSearch.narrow`): the ways part, and the
wrong ones fall out.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from core.map import RoadNetwork
from core.map.ferries import GAME_KM_PER_MAP_KM
from core.planning.nav_field import (
    ACROSS_S,
    ACROSS_STEP,
    ONWARD_STEP,
    SEA_STEP,
    _cached,
    _weights,
    nav_along,
)

__all__ = [
    "Candidate",
    "DestinationSearch",
    "candidates",
    "forward_field",
]

DISTANCE_SHARE: Final = 0.02
DISTANCE_MIN_M: Final = 300.0
"""A point is as far as the game says if within this share of it, or this
many of the game's metres: the map's reckoning is off the game's by 1-3 %."""

TIME_SHARE: Final = 0.05
TIME_MIN_S: Final = 60.0
CLUSTER_M: Final = 150.0
"""Candidates closer than this, in world metres, are one place."""

START_RADIUS_M: Final = 8.0


@dataclass(frozen=True, slots=True)
class Candidate:
    """A place the game's route may end."""

    lane: int
    s: float
    """World metres along the lane."""

    x: float
    z: float
    distance_error: float
    """The game's metres between the way here and the game's distance."""

    time_error: float

    @property
    def score(self) -> float:
        """Lower is likelier: both errors, each against its tolerance."""
        return abs(self.distance_error) / DISTANCE_MIN_M + abs(self.time_error) / TIME_MIN_S


def start_lane(network: RoadNetwork, x: float, z: float, course: float) -> tuple[int, float] | None:
    """The lane the truck is on, facing its way, and how far along it."""
    best: tuple[float, int, float] | None = None
    for lane in network.lanes_near(x, z, START_RADIUS_M):
        lane = int(lane)
        projection = network.project(lane, x, z)
        points = network.lane_points(lane)
        k = min(max(int(np.searchsorted(_arc(points), projection.s)), 1), len(points) - 1)
        heading = math.atan2(
            float(points[k][0] - points[k - 1][0]), float(points[k][2] - points[k - 1][2])
        )
        off = abs(math.remainder(heading - course, 2.0 * math.pi))
        if off > math.radians(60.0):
            continue
        cost = projection.distance + 5.0 * off
        if best is None or cost < best[0]:
            best = (cost, lane, projection.s)
    return None if best is None else (best[1], best[2])


def _arc(points) -> np.ndarray:
    p = np.asarray(points, float)[:, [0, 2]]
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def forward_field(
    network: RoadNetwork, lane: int, s: float, limit_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """From `s` world metres into `lane`: the game's quickest time (s) and
    its distance (the game's metres) to the start of every lane within
    `limit_m` of the game's metres; inf beyond. The start lane's own start
    is behind the truck: it is reached only round again."""
    starts, ends, how, sea_s, sea_m = _forward(network)
    nav, per_metre = _weights(network)
    hidden = _cached("hidden", network, lambda: np.asarray(network.lane_hidden).tolist())
    n = len(network)
    time = [math.inf] * n
    distance = [math.inf] * n
    # The rest of the start lane, then on from its end.
    rest = nav[lane] - nav_along(network, lane, s)
    queue: list[tuple[float, float, int]] = []
    _push_next(queue, lane, rest * per_metre[lane], rest, starts, ends, how, sea_s, sea_m, nav)
    while queue:
        t, d, at = heapq.heappop(queue)
        if t >= time[at]:
            continue
        time[at], distance[at] = t, d
        if d > limit_m:
            continue
        for k in range(starts[at], starts[at + 1]):
            nxt = ends[k]
            if hidden[nxt] and not hidden[at]:
                continue  # a road the GPS shows goes on by roads it shows
            kind = how[k]
            if kind == ONWARD_STEP:
                step_t, step_d = nav[at] * per_metre[at], nav[at]
            elif kind == ACROSS_STEP:
                step_t, step_d = ACROSS_S, 0.0
            else:
                step_t = nav[at] * per_metre[at] + sea_s[k]
                step_d = nav[at] + sea_m[k]
            if t + step_t < time[nxt]:
                heapq.heappush(queue, (t + step_t, d + step_d, nxt))
    return np.asarray(time), np.asarray(distance)


def _push_next(queue, lane, t, d, starts, ends, how, sea_s, sea_m, nav) -> None:
    for k in range(starts[lane], starts[lane + 1]):
        if how[k] == ONWARD_STEP:
            heapq.heappush(queue, (t, d, ends[k]))
        elif how[k] == SEA_STEP:
            heapq.heappush(queue, (t + sea_s[k], d + sea_m[k], ends[k]))


def _forward(network: RoadNetwork):
    return _cached("forward", network, lambda: _build_forward(network))


def _build_forward(network: RoadNetwork):
    n = len(network)
    src_on = np.repeat(np.arange(n), np.diff(np.asarray(network.successor_start)))
    dst_on = np.asarray(network.successors, dtype=np.int64)
    src_ac = np.repeat(np.arange(n), np.diff(np.asarray(network.change_start)))
    dst_ac = np.asarray(network.changes, dtype=np.int64)
    sea = [link for links in network.ferry_links.values() for link in links]
    src = np.concatenate([src_on, src_ac, np.array([x.board for x in sea], dtype=np.int64)])
    dst = np.concatenate([dst_on, dst_ac, np.array([x.land for x in sea], dtype=np.int64)])
    how = np.concatenate(
        [
            np.full(len(src_on), ONWARD_STEP),
            np.full(len(src_ac), ACROSS_STEP),
            np.full(len(sea), SEA_STEP),
        ]
    )
    plain = np.zeros(len(src_on) + len(src_ac))
    sea_s = np.concatenate([plain, np.array([x.minutes * 60.0 for x in sea])])
    sea_m = np.concatenate(
        [plain, np.array([x.km * 1000.0 or x.cost * GAME_KM_PER_MAP_KM for x in sea])]
    )
    order = np.argsort(src, kind="stable")
    starts = np.searchsorted(src[order], np.arange(n + 1))
    return (
        starts.tolist(),
        dst[order].tolist(),
        how[order].tolist(),
        sea_s[order].tolist(),
        sea_m[order].tolist(),
    )


def candidates(
    network: RoadNetwork,
    pose: tuple[float, float, float],
    nav_distance: float,
    nav_time: float | None = None,
) -> list[Candidate]:
    """The places the game's route from `pose` (x, z, course) may end, by
    its distance (the game's metres) and time (the game's seconds), likeliest
    first; one per `CLUSTER_M`."""
    start = start_lane(network, *pose)
    if start is None:
        return []
    tolerance = max(DISTANCE_MIN_M, DISTANCE_SHARE * nav_distance)
    time, distance = forward_field(network, start[0], start[1], nav_distance + tolerance)
    nav, per_metre = _weights(network)
    lengths = np.asarray(nav)
    rates = np.asarray(per_metre)
    # On a lane its way enters at `distance`, the point at the game's distance.
    into = nav_distance - distance
    on = np.isfinite(distance) & (into >= -tolerance) & (into <= lengths + tolerance)
    found: list[Candidate] = []
    for lane in np.flatnonzero(on):
        lane = int(lane)
        along = min(max(float(into[lane]), 0.0), float(lengths[lane]))
        d_err = float(distance[lane]) + along - nav_distance
        t = float(time[lane]) + along * float(rates[lane])
        t_err = 0.0 if nav_time is None else t - nav_time
        if nav_time is not None and abs(t_err) > max(TIME_MIN_S, TIME_SHARE * nav_time):
            continue
        if abs(d_err) > tolerance:
            continue
        world = along / lengths[lane] * float(network.lane_length[lane]) if lengths[lane] else 0.0
        point = _point_at(network, lane, world)
        found.append(Candidate(lane, world, point[0], point[1], d_err, t_err))
    return _clustered(sorted(found, key=lambda c: c.score))


def _point_at(network: RoadNetwork, lane: int, s: float) -> tuple[float, float]:
    points = np.asarray(network.lane_points(lane), float)[:, [0, 2]]
    arc = _arc(network.lane_points(lane))
    return float(np.interp(s, arc, points[:, 0])), float(np.interp(s, arc, points[:, 1]))


def _clustered(found: list[Candidate]) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in found:
        if all(math.hypot(c.x - k.x, c.z - k.z) > CLUSTER_M for k in kept):
            kept.append(c)
    return kept


@dataclass
class DestinationSearch:
    """The places the route may end, narrowed as the truck drives."""

    network: RoadNetwork
    found: list[Candidate]

    @classmethod
    def start(
        cls,
        network: RoadNetwork,
        pose: tuple[float, float, float],
        nav_distance: float,
        nav_time: float | None = None,
    ) -> DestinationSearch:
        return cls(network, candidates(network, pose, nav_distance, nav_time))

    def narrow(self, pose: tuple[float, float, float], nav_distance: float) -> list[Candidate]:
        """Keep the places whose way from `pose` is as long as the game says now."""
        start = start_lane(self.network, *pose)
        if start is None or not self.found:
            return self.found
        tolerance = max(DISTANCE_MIN_M, DISTANCE_SHARE * nav_distance)
        _, distance = forward_field(self.network, start[0], start[1], nav_distance + tolerance)
        kept = []
        for c in self.found:
            if c.lane == start[0] and c.s >= start[1]:
                way = nav_along(self.network, c.lane, c.s) - nav_along(
                    self.network, c.lane, start[1]
                )
            else:
                way = float(distance[c.lane]) + nav_along(self.network, c.lane, c.s)
            if abs(way - nav_distance) <= tolerance:
                kept.append(c)
        self.found = kept
        return kept

    @property
    def known(self) -> Candidate | None:
        """The place, once only one is left."""
        return self.found[0] if len(self.found) == 1 else None


DEPOT_SHARE: Final = 0.12
"""A depot is a candidate within this share of the game's distance: the
map's plain reckoning was 1-3 % off the game's, 4-10 % on the Swiss roads
to Bern, where the game weighs local roads otherwise (game_route.MODELS
fits that once the destination is known)."""

FREE_POINTS: Final = 8
PLACES: Final = 10
NEAR_DEPOT_M: Final = 300.0
"""A point on the roads this near a depot's gate is that depot."""

GPS_COMPANY: Final = "gps"
"""The company of a place that is no depot: a point marked on the map, or a
trailer waiting where no gate of the map's is."""


def gps_places(
    network: RoadNetwork,
    places,
    pose: tuple[float, float, float],
    nav_distance: float,
    goals_of=None,
) -> list:
    """Where the game's GPS may be going from `pose`, as places to plan to
    (core.map.CompanyPlace), the best fit to the game's distance first: the
    depots whose way is as long as it says (within `DEPOT_SHARE`), and the
    points on the roads that far (`candidates`), as places of company
    `GPS_COMPANY` with the point for a gate -- a point near a depot's gate
    standing for the depot. In Bern a job's trailer waited 416 m from the
    nearest depot's gate: the depot that fitted 2.7 % was driven to, and the
    point that fitted 1.3 %, and kept fitting within 0.1 %, was not looked
    at. `goals_of` maps a depot to its gate lanes (game_route.depot_goals)."""
    from core.map import CompanyPlace
    from core.planning.game_route import depot_goals
    from core.planning.route import RouteError

    start = start_lane(network, *pose)
    if start is None:
        return []
    _, distance = forward_field(network, start[0], start[1], nav_distance * (1 + DEPOT_SHARE))
    ranked: list[tuple[float, object]] = []
    for depot in places.companies:
        try:
            goals = goals_of(depot) if goals_of is not None else depot_goals(network, depot)
        except RouteError:
            continue
        ways = [
            distance[lane] + along for lane, along in goals.items() if np.isfinite(distance[lane])
        ]
        if ways:
            miss = abs(min(ways) - nav_distance) / max(nav_distance, 1.0)
            if miss <= DEPOT_SHARE:
                ranked.append((miss, depot))
    gates = [g for _, depot in ranked for g in depot.gates]
    for c in candidates(network, pose, nav_distance)[:FREE_POINTS]:
        if any(math.hypot(c.x - gx, c.z - gz) < NEAR_DEPOT_M for gx, gz in gates):
            continue
        point = CompanyPlace(GPS_COMPANY, f"{c.x:.0f},{c.z:.0f}", (c.x, c.z), ((c.x, c.z),), ())
        ranked.append((abs(c.distance_error) / max(nav_distance, 1.0), point))
    ranked.sort(key=lambda found: found[0])
    return [place for _, place in ranked[:PLACES]]


def place_record(place) -> dict:
    """A place as route metadata keeps it."""
    return {
        "company": place.company,
        "city": place.city,
        "position": list(place.position),
        "gates": [list(g) for g in place.gates],
    }


def place_from(record: dict, places=None):
    """A place from `place_record`: the map's own where it is a depot."""
    from core.map import CompanyPlace

    if places is not None and record["company"] != GPS_COMPANY:
        found = places.company(record["company"], record["city"])
        if found is not None:
            return found
    return CompanyPlace(
        record["company"],
        record["city"],
        tuple(record["position"]),
        tuple(tuple(g) for g in record["gates"]),
        (),
    )
