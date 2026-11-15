"""Where a truck can fill up: gas station pumps, as places on lanes.

A gas station prefab's descriptor has a spawn point at each pump, where the
game puts a truck it refuels; the service areas' forecourt lanes run right
through them, 0.1-0.5 m off. So each pump becomes a place on the lane of its
station that passes nearest -- how far along it the truck stops. A pump is
only one to drive to if roads lead onto its lanes and off them again: the
small stations by the road are pump islands on open ground, their few lanes
joined to nothing, and of 1409 pumps on lanes only 374 can be driven to. Out
of Oslo every pump within 5 km was such, and the truck reached the depot with
1 km of fuel.

Kept beside the compiled network as `fuel.json`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from core.map.network import RoadNetwork
    from core.map.sector import Prefab

__all__ = [
    "FUEL_FILE",
    "PUMP_ON_LANE_M",
    "STATION_LANES_FILE",
    "STATION_NEAR_M",
    "Pump",
    "load_pumps",
    "load_station_lanes",
    "pumps_on",
    "save_pumps",
    "station_lanes",
]

FUEL_FILE: Final = "fuel.json"
STATION_LANES_FILE: Final = "station_lanes.npy"
STATION_NEAR_M: Final = 8.0
"""A station's own lanes, and the quiet roads -- no traffic of the game's on
them -- passing within this of a pump, whether or not the station can be
driven to, are kept out of by the planner
(core.planning.nav_field.STATION_FACTOR). Its forecourt lanes run 0.1-0.5 m
off the pumps; the main road's lanes, in the same prefab, further."""

PUMP_ON_LANE_M: Final = 1.5
"""How far off its station's nearest lane a pump may be: they measure
0.1-0.5 m in the service areas out of Calais."""


@dataclass(frozen=True, slots=True)
class Pump:
    lane: int
    s: float
    """How far along `lane` the truck stops, metres."""

    position: tuple[float, float]
    """(x, z)."""


def pumps_on(network: RoadNetwork, prefabs: Iterable[Prefab]) -> list[Pump]:
    """Each pump of each prefab, on the nearest of that prefab's lanes."""
    owners = np.asarray(network.lane_owner)
    pumps = []
    for prefab in prefabs:
        if not prefab.gas_pumps:
            continue
        lanes = np.flatnonzero(owners == int(prefab.uid, 16))
        for point in prefab.gas_pumps:
            x, z = float(point[0]), float(point[2])
            nearest = None
            for lane in lanes:
                projection = network.project(int(lane), x, z)
                if nearest is None or projection.distance < nearest.distance:
                    nearest = projection
            if nearest is None or nearest.distance > PUMP_ON_LANE_M:
                continue
            lane = int(nearest.lane)
            if _leads_out(network, lane, forward=True) and _leads_out(network, lane, forward=False):
                pumps.append(Pump(lane, float(nearest.s), (x, z)))
    return pumps


def _leads_out(network: RoadNetwork, lane: int, *, forward: bool, most: int = 200) -> bool:
    """Whether a lane joins, ahead (or behind), a shown lane of another item:
    whether a truck can leave its station that way (or get into it)."""
    owners = network.lane_owner
    if not forward:
        before: dict[int, list[int]] = {}
        starts, nexts = network.successor_start, network.successors
    seen, queue = {lane}, [lane]
    while queue and len(seen) <= most:
        current = queue.pop()
        if forward:
            following = [int(n) for n in network.next_lanes(current)]
        else:
            following = before.setdefault(current, _previous(starts, nexts, current))
        for other in following:
            if owners[other] != owners[lane] and not network.lane_hidden[other]:
                return True
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return False


def _previous(starts, nexts, lane: int) -> list[int]:
    found = np.flatnonzero(np.asarray(nexts) == lane)
    return [int(np.searchsorted(starts, k, side="right") - 1) for k in found]


def save_pumps(directory: str | Path, pumps: Iterable[Pump]) -> None:
    document = [{"lane": p.lane, "s": p.s, "position": list(p.position)} for p in pumps]
    (Path(directory) / FUEL_FILE).write_text(json.dumps(document), encoding="utf-8")


def load_pumps(directory: str | Path) -> list[Pump]:
    """The pumps of a cache; none for one built without them."""
    path = Path(directory) / FUEL_FILE
    if not path.is_file():
        return []
    return [
        Pump(int(p["lane"]), float(p["s"]), (float(p["position"][0]), float(p["position"][1])))
        for p in json.loads(path.read_text(encoding="utf-8"))
    ]


def station_lanes(network: RoadNetwork, prefabs: Iterable[Prefab]) -> np.ndarray:
    """The lanes passing a gas station's pumps -- within `STATION_NEAR_M` of
    one -- other than the game's traffic's roads: its forecourt, and a service
    area's road past them. A station's prefab often holds the main road's
    lanes too, a few metres further off: those are kept."""
    sources, ai = np.asarray(network.lane_source), np.asarray(network.lane_ai, dtype=bool)
    owners = np.asarray(network.lane_owner)
    found: set[int] = set()
    for prefab in prefabs:
        if not prefab.gas_pumps:
            continue
        mine = int(prefab.uid, 16)
        for point in prefab.gas_pumps:
            x, z = float(point[0]), float(point[2])
            for lane in network.lanes_near(x, z, STATION_NEAR_M):
                lane = int(lane)
                if network.project(lane, x, z).distance > STATION_NEAR_M:
                    continue
                if (sources[lane] == 1 and int(owners[lane]) == mine) or (
                    sources[lane] == 0 and not ai[lane]
                ):
                    found.add(lane)
    return np.array(sorted(found), dtype=np.int64)


def load_station_lanes(directory: str | Path) -> np.ndarray:
    """The lanes `station_lanes` found when the cache was built; none for a
    cache built before them."""
    path = Path(directory) / STATION_LANES_FILE
    return np.load(path) if path.is_file() else np.empty(0, dtype=np.int64)
