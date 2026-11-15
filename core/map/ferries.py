"""Crossings of the sea, as links from one lane to another.

A job from Göteborg to Dover cannot be driven: the game's GPS takes a ship
to Denmark and the Channel Tunnel's train to England, and a planner that
knows only roads found no way at all. The map has a ferry item in each port,
where the truck drives on, and the definitions say which ports are joined
and how far. So each line becomes a link from the lanes at one port's ferry
to the lanes where the other port puts the truck ashore -- one more way on
from a lane, for the planner, at a cost.

The cost is the line's distance as the game states it, in its own
kilometres, brought to map metres by `GAME_KM_PER_MAP_KM`, and never less
than the straight line between the ports, so the planner's straight-line
estimate stays a lower bound.

Kept beside the compiled network as `ferries.json`: a few hundred links.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from core.map.network import RoadNetwork
    from core.map.sector import Ferry, FerryLine

__all__ = ["FerryLink", "link_ferries", "load_links", "save_links"]

FERRIES_FILE: Final = "ferries.json"

SEARCH_RADII_M: Final = (120.0, 300.0)
"""A ferry item stands on the quay or in the water, not on a lane: 17 to 450 m
from the nearest one, measured over the installed map. Lanes are looked for
this near, and further only if there are none."""

DEAD_END_BONUS_M: Final = 20.0
"""The drive onto a ship ends where its lane does: a lane going nowhere near
the ferry is taken over a lane going on past it, unless 20 m further."""

CLOSE_CALL_M: Final = 10.0
"""Lanes within this of the best one are taken too: a quay has several."""

LEAVE_PORT_M: Final = 300.0
"""A lane the truck is put ashore on leads this far from the ferry item, on
roads the game's GPS uses. At Folkestone the lanes that start nearest the
train's item are the platform's own, going on 60 m to where trucks drive
onto the train, and beside them hidden roads; landed there, no road led
anywhere, and the planner never took the tunnel."""

LEAVE_SEARCH_LANES: Final = 400
"""How many lanes on a lane is followed to see whether it leaves the port."""

GAME_KM_PER_MAP_KM: Final = 19.0
"""The game's kilometres, as its GPS and its ferry lines count them, per
kilometre of map: the scale of the countryside, where most of a ferry's
approach is."""


@dataclass(frozen=True, slots=True)
class FerryLink:
    board: int
    """The lane the truck drives on by."""

    land: int
    """The lane it drives off by at the other port."""

    cost: float
    """Map metres, for the planner."""

    start: str
    end: str
    train: bool
    minutes: float = 0.0
    """How long the crossing takes, as the game states it: what its GPS
    weighs it by (core.planning.nav_field)."""

    km: float = 0.0
    """How long the crossing is, as the game states it: what its navigation
    distance counts. The Channel Tunnel's train is 60 km by the game, though
    its terminals are 125 of the game's kilometres apart as the crow flies."""


def link_ferries(
    network: RoadNetwork, ferries: Sequence[Ferry], lines: Sequence[FerryLine]
) -> list[FerryLink]:
    """Every line whose two ports are both on `network`, lane to lane."""
    ports = {ferry.port: ferry for ferry in ferries}
    links: list[FerryLink] = []
    for line in lines:
        start, end = ports.get(line.start), ports.get(line.end)
        if start is None or end is None:
            continue
        ashore = end.position + end.unload_offset
        board = _boarding(network, start.position[[0, 2]])
        land = _landing(network, ashore[[0, 2]])
        straight = math.hypot(
            float(ashore[0] - start.position[0]), float(ashore[2] - start.position[2])
        )
        cost = max(straight, line.distance_km * 1000.0 / GAME_KM_PER_MAP_KM)
        links.extend(
            FerryLink(
                int(b),
                int(a),
                cost,
                line.start,
                line.end,
                start.train or end.train,
                line.minutes,
                line.distance_km,
            )
            for b in board
            for a in land
        )
    return links


def _boarding(network: RoadNetwork, at: np.ndarray) -> list[int]:
    """The lanes that end nearest a ferry item, a dead end first."""

    def score(lane: int) -> float:
        end = network.lane_points(lane)[-1]
        gap = math.hypot(float(end[0] - at[0]), float(end[2] - at[1]))
        return gap - (DEAD_END_BONUS_M if len(network.next_lanes(lane)) == 0 else 0.0)

    return _best(network, at, score)


def _landing(network: RoadNetwork, at: np.ndarray) -> list[int]:
    """The lanes that start nearest where a ship puts the truck ashore, and
    lead away from it (`LEAVE_PORT_M`)."""

    def score(lane: int) -> float:
        if network.lane_hidden[lane] or not _leaves(network, lane, at):
            return math.inf
        first = network.lane_points(lane)[0]
        return math.hypot(float(first[0] - at[0]), float(first[2] - at[1]))

    return _best(network, at, score)


def _leaves(network: RoadNetwork, lane: int, at: np.ndarray) -> bool:
    """Whether some way on from `lane`, off hidden roads, gets `LEAVE_PORT_M`
    from `at`."""
    seen, stack = {lane}, [lane]
    while stack and len(seen) <= LEAVE_SEARCH_LANES:
        current = stack.pop()
        end = network.lane_points(current)[-1]
        if math.hypot(float(end[0] - at[0]), float(end[2] - at[1])) >= LEAVE_PORT_M:
            return True
        for following in network.next_lanes(current):
            following = int(following)
            if following not in seen and not network.lane_hidden[following]:
                seen.add(following)
                stack.append(following)
    return False


def _best(network: RoadNetwork, at: np.ndarray, score) -> list[int]:
    for radius in SEARCH_RADII_M:
        scored = [
            (score(int(lane)), int(lane))
            for lane in network.lanes_near(float(at[0]), float(at[1]), radius)
        ]
        scored = [(value, lane) for value, lane in scored if math.isfinite(value)]
        if scored:
            best = min(value for value, _ in scored)
            return sorted(lane for value, lane in scored if value <= best + CLOSE_CALL_M)
    return []


def save_links(directory: str | Path, links: Sequence[FerryLink]) -> None:
    document = [
        {
            "board": link.board,
            "land": link.land,
            "cost": link.cost,
            "start": link.start,
            "end": link.end,
            "train": link.train,
            "minutes": link.minutes,
            "km": link.km,
        }
        for link in links
    ]
    (Path(directory) / FERRIES_FILE).write_text(json.dumps(document), encoding="utf-8")


def load_links(directory: str | Path) -> dict[int, tuple[FerryLink, ...]]:
    """The links by the lane they are boarded from; none for a cache without."""
    path = Path(directory) / FERRIES_FILE
    if not path.is_file():
        return {}
    by_lane: dict[int, list[FerryLink]] = {}
    for body in json.loads(path.read_text(encoding="utf-8")):
        link = FerryLink(
            int(body["board"]),
            int(body["land"]),
            float(body["cost"]),
            body["start"],
            body["end"],
            bool(body["train"]),
            float(body.get("minutes") or 0.0),
            float(body.get("km") or 0.0),
        )
        by_lane.setdefault(link.board, []).append(link)
    return {lane: tuple(found) for lane, found in by_lane.items()}
