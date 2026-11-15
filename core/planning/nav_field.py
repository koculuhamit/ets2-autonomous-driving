"""The game's route to a destination, from every lane at once.

The game's GPS draws the quickest route, as it reckons time: each road at a
share of its truck speed limit, a ferry at its crossing time
(core.map.nav_scale). The telemetry says how long that route is and how long
it takes (navigation distance and time). The network knows each lane the
same way (`RoadNetwork.lane_nav_length`, `lane_nav_speed`), so one search
backwards from the destination over every lane gives the game's route from
anywhere -- and its distance, which is checked against the game's: equal,
the same route.

Moving across into a lane alongside costs next to nothing: the game routes
along roads, not lanes. Roads the game's map leaves out (hidden) are only a
way onto the roads it shows, never a way between two of them: landed off
the Channel Tunnel's train at Folkestone, hidden roads are the way out of
the terminal. Roads the GPS is told to avoid cost `AVOID_FACTOR` times
their time.
"""

from __future__ import annotations

import heapq
import json
import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from core.map import RoadNetwork
from core.map.ferries import GAME_KM_PER_MAP_KM

__all__ = [
    "ACROSS_S",
    "ACROSS_STEP",
    "AVOID_FACTOR",
    "MODELS",
    "ONWARD_STEP",
    "SEA_STEP",
    "NavField",
    "RouteModel",
    "nav_along",
    "nav_length",
]

ACROSS_S: Final = 0.05
"""What moving across into a lane alongside costs: next to nothing, enough to
prefer staying in lane between two ways that take the same time."""

AVOID_FACTOR: Final = 10.0
"""How much longer a road the GPS is told to avoid is reckoned to take."""

DEFAULT_SPEED_KMH: Final = 60.0 * 0.8
"""For a cache built without GPS speeds."""

ONWARD_STEP: Final = 0
ACROSS_STEP: Final = 1
SEA_STEP: Final = 2


def nav_length(network: RoadNetwork, lane: int) -> float:
    """A lane's length in the game's metres."""
    if network.lane_nav_length is not None:
        return float(network.lane_nav_length[lane])
    return float(network.lane_length[lane]) * GAME_KM_PER_MAP_KM


def nav_along(network: RoadNetwork, lane: int, s: float) -> float:
    """The game's metres of `lane` up to `s` world metres along it."""
    length = float(network.lane_length[lane])
    if length <= 0.0:
        return 0.0
    return nav_length(network, lane) * min(max(s, 0.0), length) / length


@dataclass(frozen=True)
class RouteModel:
    """How the game's GPS weighs roads, beyond the time it reckons for them.

    Its time is measured (core.map.nav_scale), and on most jobs the quickest
    way by it is the game's route to within a few km of 1,500. Not on all:
    from Antwerp to Nynäshamn the game kept to motorways by Breda, 75 km and,
    by its own reckoning, 54 minutes longer than a way with 37 km of Dutch
    and German local roads; from Milan to Göteborg it drove 85 km of a
    French local road without a second look. The game plans in a way this
    does not reproduce, so a few models are tried, and the one whose route is
    as long as the game says its own is, is driven (core.planning.game_route):
    local roads at `local_factor` times their time, except within
    `free_radius` world metres of where the route was planned from and of
    the destination.
    """

    local_factor: float = 1.0
    free_radius: float = 2000.0


MODELS: Final = (RouteModel(1.0), RouteModel(3.0), RouteModel(10.0))
"""The models tried, the plain quickest first."""


@dataclass(frozen=True)
class NavField:
    """From the start of each lane: the game's time to the destination
    (seconds; inf where it cannot be reached), and the distance of that way
    (the game's metres)."""

    network: RoadNetwork
    time: np.ndarray
    distance: np.ndarray
    onward: np.ndarray
    """The lane the game's way goes on to from each lane; -1 at a goal and
    where there is none."""

    step: np.ndarray
    """How: `ONWARD_STEP`, `ACROSS_STEP` into the lane alongside, or `SEA_STEP`."""

    model: RouteModel = field(default_factory=lambda: RouteModel())
    per_metre: list[float] = field(default_factory=list, repr=False, compare=False)
    """What a metre of the game's costs on each lane, in the game's seconds
    as the model weighs them."""

    @classmethod
    def to_goals(
        cls,
        network: RoadNetwork,
        goals: Mapping[int, float],
        model: RouteModel | None = None,
        free_near: Sequence[tuple[float, float]] = (),
    ) -> NavField:
        """The field for lanes `goals`, each with the game's metres from its
        start to the destination; weighed by `model`, local roads within
        `model.free_radius` of `free_near` (and of the goals) as they are."""
        model = model or RouteModel()
        starts, froms, how, sea_s, sea_m = _reverse(network)
        n = len(network)
        nav, per_metre = _weights(network)
        hidden = _cached("hidden", network, lambda: np.asarray(network.lane_hidden).tolist())
        if model.local_factor != 1.0 and network.lane_speed_class is not None:
            per_metre = _local_weighted(network, per_metre, goals, free_near, model)
        time = [math.inf] * n
        distance = [math.inf] * n
        onward = [-1] * n
        step = [0] * n
        queue: list[tuple[float, int]] = []
        for lane, metres in goals.items():
            t = metres * per_metre[lane]
            if t < time[lane]:
                time[lane], distance[lane] = t, metres
                queue.append((t, lane))
        heapq.heapify(queue)
        while queue:
            t, lane = heapq.heappop(queue)
            if t > time[lane]:
                continue
            d = distance[lane]
            for k in range(starts[lane], starts[lane + 1]):
                before = froms[k]
                if hidden[lane] and not hidden[before]:
                    continue  # a road the GPS shows goes on by roads it shows
                kind = how[k]
                if kind == ONWARD_STEP:
                    total = t + nav[before] * per_metre[before]
                    metres = d + nav[before]
                elif kind == ACROSS_STEP:
                    total = t + ACROSS_S
                    metres = d
                else:
                    total = t + nav[before] * per_metre[before] + sea_s[k]
                    metres = d + nav[before] + sea_m[k]
                if total < time[before]:
                    time[before], distance[before] = total, metres
                    onward[before], step[before] = lane, kind
                    heapq.heappush(queue, (total, before))
        return cls(
            network,
            np.asarray(time, dtype=np.float64),
            np.asarray(distance, dtype=np.float64),
            np.asarray(onward, dtype=np.int64),
            np.asarray(step, dtype=np.int8),
            model,
            per_metre,
        )

    def time_at(self, lane: int, s: float = 0.0) -> float:
        """The game's seconds to the destination from `s` world metres into `lane`."""
        return float(self.time[lane]) - nav_along(self.network, lane, s) * self.per_metre[lane]

    def distance_at(self, lane: int, s: float = 0.0) -> float:
        """The game's metres to the destination from `s` world metres into `lane`."""
        return float(self.distance[lane]) - nav_along(self.network, lane, s)

    def way(self, lane: int, limit: int = 1_000_000) -> tuple[list[int], list[int]]:
        """The game's way from `lane`: its lanes, and the step from each to the next."""
        lanes, steps = [lane], []
        while self.onward[lanes[-1]] >= 0 and len(lanes) < limit:
            steps.append(int(self.step[lanes[-1]]))
            lanes.append(int(self.onward[lanes[-1]]))
        return lanes, steps


# --------------------------------------------------------------------- #

_CACHE: dict[tuple[str, int], tuple[weakref.ref, object]] = {}
"""Per network: what is built once and kept -- by id, checked by a weak
reference, since a network is not hashable and an id is reused once its
network is gone."""


def _cached(kind: str, network: RoadNetwork, build):
    key = (kind, id(network))
    found = _CACHE.get(key)
    if found is not None and found[0]() is network:
        return found[1]
    value = build()
    _CACHE[key] = (weakref.ref(network), value)
    return value


def _weights(network: RoadNetwork) -> tuple[list[float], list[float]]:
    """Each lane's length in the game's metres, and the game's seconds per
    such metre on it."""
    return _cached("weights", network, lambda: _build_weights(network))


def _local_weighted(
    network: RoadNetwork,
    per_metre: list[float],
    goals: Mapping[int, float],
    free_near: Sequence[tuple[float, float]],
    model: RouteModel,
) -> list[float]:
    """`per_metre` with local roads `model.local_factor` times dearer, but
    near `free_near` and the goals."""
    ends = _cached("ends", network, lambda: _lane_ends(network))
    goal_points = [tuple(ends[g]) for g in goals]
    free = np.zeros(len(ends), dtype=bool)
    for x, z in [*free_near, *goal_points]:
        free |= np.hypot(ends[:, 0] - x, ends[:, 1] - z) < model.free_radius
    local = np.asarray(network.lane_speed_class) == 0
    factor = np.where(local & ~free, model.local_factor, 1.0)
    return (np.asarray(per_metre) * factor).tolist()


def _lane_ends(network: RoadNetwork) -> np.ndarray:
    ends = np.asarray(network.points)[np.asarray(network.lane_start)[1:] - 1]
    return ends[:, [0, 2]].astype(np.float64)


def _build_weights(network: RoadNetwork) -> tuple[list[float], list[float]]:
    n = len(network)
    if network.lane_nav_length is not None:
        nav = np.asarray(network.lane_nav_length, dtype=np.float64)
    else:
        nav = np.asarray(network.lane_length, dtype=np.float64) * GAME_KM_PER_MAP_KM
    if network.lane_nav_speed is not None:
        speed = np.asarray(network.lane_nav_speed, dtype=np.float64)
    else:
        speed = np.full(n, DEFAULT_SPEED_KMH)
    per_metre = 3.6 / np.maximum(speed, 1.0)
    if network.lane_gps_avoid is not None:
        per_metre = np.where(
            np.asarray(network.lane_gps_avoid), per_metre * AVOID_FACTOR, per_metre
        )
    per_metre = np.where(_avoided(network), per_metre * AVOID_FACTOR, per_metre)
    source = getattr(network, "source", None)
    if source is not None:
        from core.map.fuel import load_station_lanes

        stations = load_station_lanes(source)
        stations = stations[stations < n]
        per_metre[stations] *= STATION_FACTOR
    return nav.tolist(), per_metre.tolist()


STATION_FACTOR: Final = 1000.0
"""A gas station's lanes, and its service road's (core.map.fuel.station_lanes),
cost this many times their time: the truck never goes through a station on
its own, only out of one it stands in. On a motorway near Innsbruck the way
went by the service road through a station beside it, and the truck hit the
pumps' island at 49 km/h; the driver fills up by hand."""


AVOID_FILE: Final = Path(__file__).resolve().parents[2] / "configs" / "avoid.json"
"""Places whose quiet roads -- no traffic of the game's on them -- cost as
the game's own avoided lanes do (`AVOID_FACTOR`), as a driver has said: by
a border in Denmark the way went through a station's service road, by the
main road, and the truck hit a tree at its side, twice."""


def load_avoided(file: Path = AVOID_FILE) -> list[tuple[float, float, float, bool]]:
    """(x, z, radius, every lane) of each place in `AVOID_FILE`; none if there
    is none. A place with `"all": true` has every lane avoided, not only its
    quiet roads: in England a motorway services' inner road, open to the
    game's traffic, was the way through a junction it shares a prefab with."""
    if not file.is_file():
        return []
    places = json.loads(file.read_text(encoding="utf-8"))["places"]
    return [
        (
            float(p["position"][0]),
            float(p["position"][1]),
            float(p["radius"]),
            bool(p.get("all", False)),
        )
        for p in places
    ]


def _avoided(network: RoadNetwork, places=None) -> np.ndarray:
    lanes = len(network)
    places = load_avoided() if places is None else places
    avoided = np.zeros(lanes, dtype=bool)
    ai = getattr(network, "lane_ai", None)
    if not places or ai is None:
        return avoided
    quiet = (np.asarray(network.lane_source) == 0) & ~np.asarray(ai, dtype=bool)
    starts = np.asarray(network.lane_start)
    points = np.asarray(network.points)[:, [0, 2]].astype(np.float64)
    middle = points[(starts[:-1] + starts[1:] - 1) // 2]
    for x, z, radius, *every in places:
        inside = np.hypot(middle[:, 0] - x, middle[:, 1] - z) <= radius
        avoided |= inside if every and every[0] else quiet & inside
    return avoided


def _reverse(
    network: RoadNetwork,
) -> tuple[list[int], list[int], list[int], list[float], list[float]]:
    """For each lane, the lanes that lead into it: (row starts, lanes, step,
    crossing seconds, crossing metres), rows by the lane led into."""
    return _cached("reverse", network, lambda: _build_reverse(network))


def _build_reverse(
    network: RoadNetwork,
) -> tuple[list[int], list[int], list[int], list[float], list[float]]:
    n = len(network)
    src_on = np.repeat(np.arange(n), np.diff(np.asarray(network.successor_start)))
    dst_on = np.asarray(network.successors, dtype=np.int64)
    src_ac = np.repeat(np.arange(n), np.diff(np.asarray(network.change_start)))
    dst_ac = np.asarray(network.changes, dtype=np.int64)
    sea = [link for links in network.ferry_links.values() for link in links]
    src = np.concatenate([src_on, src_ac, np.array([s.board for s in sea], dtype=np.int64)])
    dst = np.concatenate([dst_on, dst_ac, np.array([s.land for s in sea], dtype=np.int64)])
    how = np.concatenate(
        [
            np.full(len(src_on), ONWARD_STEP),
            np.full(len(src_ac), ACROSS_STEP),
            np.full(len(sea), SEA_STEP),
        ]
    )
    plain = np.zeros(len(src_on) + len(src_ac))
    sea_s = np.concatenate([plain, np.array([s.minutes * 60.0 for s in sea])])
    # The game counts a crossing as long as it states; `cost` is map metres,
    # kept no less than the straight line for the router's estimate.
    sea_m = np.concatenate(
        [plain, np.array([s.km * 1000.0 or s.cost * GAME_KM_PER_MAP_KM for s in sea])]
    )
    order = np.argsort(dst, kind="stable")
    starts = np.searchsorted(dst[order], np.arange(n + 1))
    result = (
        starts.tolist(),
        src[order].tolist(),
        how[order].tolist(),
        sea_s[order].tolist(),
        sea_m[order].tolist(),
    )
    return result
