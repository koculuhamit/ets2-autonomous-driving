"""The road network as the driving stack uses it: arrays on disk, mapped in.

`LaneGraph` is how the network gets built -- one Python object per lane, easy
to reason about and to test, and far too heavy to start the stack with. Built
from the whole of Europe it takes 22 s and 1.9 GB, every time. So it is built
once, offline, and compiled into this: flat arrays in `.npy` files that are
memory-mapped on open, which costs nothing until a page is touched.

    network = RoadNetwork.compile(LaneGraph.build(Sector.load("europe.json")))
    network.save("data/map/europe")

    network = RoadNetwork.open("data/map/europe")
    lane, distance = network.nearest(x, z)

Lanes are rows. A lane's points are a slice of one shared point array, and its
successors a slice of one shared index array -- compressed sparse rows, the
same layout every graph library uses, because nothing else keeps a
half-million-lane graph both small and fast to walk.
"""

from __future__ import annotations

import copyreg
import heapq
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from core.map.ferries import FerryLink, load_links
from core.map.graph import LANE_CHANGE_COST_M, Lane, LaneGraph
from core.map.sector import Signal

__all__ = ["MapCacheError", "Projection", "RoadNetwork"]

FORMAT: Final = "ets2ad-map-cache"
FORMAT_VERSION: Final = 4
MANIFEST_FILE: Final = "manifest.json"

#: Grid cell for the point index, metres. Points are about 2 m apart, so a
#: cell holds tens of them on an open road and a few hundred in a junction.
CELL_SIZE_M: Final = 16.0

#: Upper bound on the spacing of a lane's points, as sampled by the extractor.
_POINT_SPACING_M: Final = 2.0

#: Cell coordinates are packed into one int64 key. Europe spans about +/-70 km,
#: which is under 5000 cells either way; this leaves room for a map ten times
#: the size.
_CELL_BIAS: Final = 1 << 20

SOURCES: Final = ("road", "prefab")
BLINKERS: Final = ("", "NoBlinker", "NoBlinkerForced", "LeftBlinker", "RightBlinker")

#: Stored node uid meaning "no node": the lane starts or ends inside a junction.
NO_NODE: Final = 0


@dataclass(frozen=True, slots=True)
class Projection:
    """A position seen from one lane."""

    lane: int
    s: float
    """Metres along the lane from its start."""

    lateral: float
    """Signed distance from the lane's line, positive to its left."""

    heading: float
    """The lane's direction there, radians, `atan2(dx, dz)`."""

    distance: float
    length: float
    overshoot: float
    """How far beyond the lane's start or end the position lies, or 0."""


class MapCacheError(RuntimeError):
    """A cache that cannot be read: missing, from another format, or damaged."""


@dataclass(frozen=True)
class RoadNetwork:
    """Directed lanes, the connections between them, and a spatial index."""

    points: np.ndarray
    """`(P, 3)` float32, every lane's points back to back."""

    lane_start: np.ndarray
    """`(L + 1,)` int64. Lane `i` is `points[lane_start[i]:lane_start[i + 1]]`."""

    lane_length: np.ndarray
    lane_source: np.ndarray
    lane_owner: np.ndarray
    """Uid of the road or prefab, as uint64."""

    lane_offset: np.ndarray
    lane_semaphore: np.ndarray
    lane_blinker: np.ndarray
    lane_rule: np.ndarray
    """Index into `rules`; 0 is no rule."""

    lane_ai: np.ndarray
    lane_hidden: np.ndarray
    """Lanes of roads the game's map and GPS do not show (`Lane.hidden`)."""
    lane_start_node: np.ndarray
    lane_end_node: np.ndarray

    successor_start: np.ndarray
    """`(L + 1,)` int64. Successors of lane `i` are `successors[s[i]:s[i + 1]]`."""

    successors: np.ndarray

    change_start: np.ndarray
    """`(L + 1,)` int64. Lanes alongside lane `i` are `changes[c[i]:c[i + 1]]`."""

    changes: np.ndarray

    cell_keys: np.ndarray
    """Sorted packed cell keys that contain at least one point."""

    cell_start: np.ndarray
    cell_points: np.ndarray
    """Point indices, grouped by cell."""

    rules: tuple[str, ...]

    lane_signal: np.ndarray | None = None
    """Index into `signals` of what stands at each lane's start; -1 for nothing."""

    signals: tuple[Signal, ...] = ()

    lane_left_hand: np.ndarray | None = None
    """Lanes of roads where traffic keeps left; None for a cache built without."""

    lane_nav_length: np.ndarray | None = None
    """Each lane's length as the game's navigation counts it (core.map.nav_scale);
    None for a cache built without."""

    lane_nav_speed: np.ndarray | None = None
    """The speed, km/h, the game's GPS reckons each lane at (core.map.nav_scale)."""

    lane_speed_class: np.ndarray | None = None
    """Each lane's speed class, an index into `core.map.nav_scale.SPEED_CLASSES`."""

    lane_gps_avoid: np.ndarray | None = None
    """Lanes of roads the game's GPS routes round."""

    ferry_links: dict[int, tuple[FerryLink, ...]] = field(default_factory=dict)
    """Crossings of the sea, by the lane they are boarded from (core.map.ferries)."""

    source: Path | None = field(default=None, compare=False, repr=False)
    """The cache it was opened from; None for one compiled here."""

    def __len__(self) -> int:
        return len(self.lane_length)

    def __reduce__(self):
        """Opened from a cache, it goes to another process as the cache's
        place, and is opened there from the same files: spawned on Windows,
        the GPS worker got a copy of the whole map, 279 MB, beside the one
        mapped in the loop's process; mapped in both, the pages are shared."""
        if self.source is None:
            return (copyreg.__newobj__, (RoadNetwork,), self.__dict__)
        return (RoadNetwork.open, (str(self.source),))

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    @classmethod
    def compile(cls, graph: LaneGraph) -> RoadNetwork:
        """Flatten a built graph into arrays."""
        lanes = graph.lanes
        counts = np.array([len(lane.points) for lane in lanes], dtype=np.int64)
        lane_start = np.zeros(len(lanes) + 1, dtype=np.int64)
        np.cumsum(counts, out=lane_start[1:])

        points = np.empty((int(lane_start[-1]), 3), dtype=np.float32)
        for index, lane in enumerate(lanes):
            points[lane_start[index] : lane_start[index + 1]] = lane.points

        rules = ("", *sorted({lane.rule for lane in lanes} - {""}))
        rule_index = {rule: i for i, rule in enumerate(rules)}

        successor_lists = [graph.successors.get(i, []) for i in range(len(lanes))]
        successor_start = np.zeros(len(lanes) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in successor_lists], out=successor_start[1:])
        successors = np.fromiter(
            (target for targets in successor_lists for target in targets),
            dtype=np.int32,
            count=int(successor_start[-1]),
        )

        change_start, changes = _rows([graph.lane_changes.get(i, []) for i in range(len(lanes))])

        cell_keys, cell_start, cell_points = _point_index(points)

        signals = tuple(
            sorted(
                {lane.signal for lane in lanes if lane.signal is not None},
                key=lambda s: (s.kind, s.profile, s.sleep or (-1, -1)),
            )
        )
        signal_index = {signal: i for i, signal in enumerate(signals)}

        return cls(
            points=points,
            lane_start=lane_start,
            lane_length=np.array([lane.length for lane in lanes], dtype=np.float32),
            lane_source=np.array([SOURCES.index(lane.source) for lane in lanes], dtype=np.uint8),
            lane_owner=np.array([int(lane.owner, 16) for lane in lanes], dtype=np.uint64),
            lane_offset=np.array([lane.offset for lane in lanes], dtype=np.float32),
            lane_semaphore=np.array([lane.semaphore for lane in lanes], dtype=np.int32),
            lane_blinker=np.array([_blinker(lane.blinker) for lane in lanes], dtype=np.uint8),
            lane_rule=np.array([rule_index[lane.rule] for lane in lanes], dtype=np.uint16),
            lane_ai=np.array([lane.ai_vehicles for lane in lanes], dtype=bool),
            lane_hidden=np.array([lane.hidden for lane in lanes], dtype=bool),
            lane_start_node=np.array([_node(lane.start_node) for lane in lanes], dtype=np.uint64),
            lane_end_node=np.array([_node(lane.end_node) for lane in lanes], dtype=np.uint64),
            successor_start=successor_start,
            successors=successors,
            change_start=change_start,
            changes=changes,
            cell_keys=cell_keys,
            cell_start=cell_start,
            cell_points=cell_points,
            rules=rules,
            lane_signal=np.array(
                [-1 if lane.signal is None else signal_index[lane.signal] for lane in lanes],
                dtype=np.int16,
            ),
            signals=signals,
            lane_left_hand=np.array([lane.left_hand for lane in lanes], dtype=bool),
        )

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path, *, source: dict | None = None) -> None:
        """Write the arrays and a manifest.

        The manifest goes last, so a directory without one is a write that did
        not finish rather than a cache that is quietly missing a file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MANIFEST_FILE).unlink(missing_ok=True)

        for name in _ARRAYS:
            np.save(directory / f"{name}.npy", getattr(self, name), allow_pickle=False)
        for name in _OPTIONAL_ARRAYS:
            if getattr(self, name) is not None:
                np.save(directory / f"{name}.npy", getattr(self, name), allow_pickle=False)

        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "cell_size": CELL_SIZE_M,
            "lanes": len(self),
            "points": len(self.points),
            "connections": len(self.successors),
            "lane_changes": len(self.changes),
            "rules": list(self.rules),
            "blinkers": list(BLINKERS),
            "sources": list(SOURCES),
            "signals": [
                {"kind": s.kind, "profile": s.profile, "sleep": list(s.sleep) if s.sleep else None}
                for s in self.signals
            ],
            "source": source or {},
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def open(cls, directory: str | Path) -> RoadNetwork:
        """Map a saved cache in. Nothing is read until it is used.

        Raises:
            MapCacheError: no finished cache there, a different format, or a
                version this code does not read.
        """
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise MapCacheError(f"no map cache at {directory} (no {MANIFEST_FILE})")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise MapCacheError(f"not a map cache: {manifest.get('format')!r}")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise MapCacheError(
                f"map cache version {manifest.get('format_version')}, "
                f"expected {FORMAT_VERSION}; rebuild it"
            )
        if manifest.get("cell_size") != CELL_SIZE_M:
            raise MapCacheError("map cache was indexed at a different cell size; rebuild it")

        arrays = {}
        for name in _ARRAYS:
            path = directory / f"{name}.npy"
            if not path.is_file():
                raise MapCacheError(f"map cache at {directory} is missing {path.name}")
            arrays[name] = _mapped(path)
        for name in _OPTIONAL_ARRAYS:
            path = directory / f"{name}.npy"
            if path.is_file():
                arrays[name] = _mapped(path)

        signals = tuple(
            Signal(s["kind"], s["profile"], tuple(s["sleep"]) if s["sleep"] else None)
            for s in manifest.get("signals", ())
        )
        network = cls(
            **arrays,
            rules=tuple(manifest["rules"]),
            signals=signals,
            ferry_links=load_links(directory),
            source=Path(directory),
        )
        if len(network.lane_start) != len(network) + 1 or network.lane_start[-1] != len(
            network.points
        ):
            raise MapCacheError(f"map cache at {directory} is inconsistent")
        return network

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def lane_points(self, lane: int) -> np.ndarray:
        return self.points[self.lane_start[lane] : self.lane_start[lane + 1]]

    def next_lanes(self, lane: int) -> np.ndarray:
        return self.successors[self.successor_start[lane] : self.successor_start[lane + 1]]

    def lanes_alongside(self, lane: int) -> np.ndarray:
        """Lanes a vehicle on this one can change into."""
        return self.changes[self.change_start[lane] : self.change_start[lane + 1]]

    def lane(self, index: int) -> Lane:
        """One lane as the builder's type, for code that wants the object."""
        return Lane(
            points=np.asarray(self.lane_points(index), dtype=float),
            source=SOURCES[int(self.lane_source[index])],
            owner=f"{int(self.lane_owner[index]):016X}",
            offset=float(self.lane_offset[index]),
            semaphore=int(self.lane_semaphore[index]),
            signal=self.signal(index),
            rule=self.rules[int(self.lane_rule[index])],
            blinker=BLINKERS[int(self.lane_blinker[index])],
            ai_vehicles=bool(self.lane_ai[index]),
            hidden=bool(self.lane_hidden[index]),
            start_node=_uid(int(self.lane_start_node[index])),
            end_node=_uid(int(self.lane_end_node[index])),
        )

    def signal(self, lane: int) -> Signal | None:
        """The traffic light or barrier at the start of a lane, if any."""
        if self.lane_signal is None:
            return None
        index = int(self.lane_signal[lane])
        return None if index < 0 else self.signals[index]

    def lanes_near(self, x: float, z: float, radius: float) -> np.ndarray:
        """Every lane with a point within `radius` of a position, unordered."""
        candidates = self._points_near(x, z, radius)
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        found = self.points[candidates]
        close = candidates[np.hypot(found[:, 0] - x, found[:, 2] - z) <= radius]
        return np.unique(np.searchsorted(self.lane_start, close, side="right") - 1)

    def project(self, lane: int, x: float, z: float) -> Projection:
        """Where a position falls on a lane: how far along, how far off, which way.

        Measured to the lane's line, not to its points, and clamped to its ends:
        `overshoot` says by how much the position lies beyond either end, which
        is how a matcher tells a lane it is on from one it has left.
        """
        points = np.asarray(self.lane_points(lane), dtype=float)
        if len(points) < 2:
            p = points[0]
            distance = math.hypot(float(p[0]) - x, float(p[2]) - z)
            return Projection(lane, 0.0, distance, 0.0, distance, 0.0, distance)

        ax, az = points[:-1, 0], points[:-1, 2]
        dx, dz = points[1:, 0] - ax, points[1:, 2] - az
        span = dx * dx + dz * dz
        span[span == 0.0] = 1e-12
        raw = ((x - ax) * dx + (z - az) * dz) / span
        t = np.clip(raw, 0.0, 1.0)
        px, pz = ax + t * dx, az + t * dz
        distances = np.hypot(px - x, pz - z)
        k = int(np.argmin(distances))

        lengths = np.sqrt(span)
        along = float(np.sum(lengths[:k]) + t[k] * lengths[k])
        total = float(np.sum(lengths))
        heading = math.atan2(float(dx[k]), float(dz[k]))
        # Left of a heading (sin h, cos h) is (cos h, -sin h).
        lateral = (x - float(px[k])) * math.cos(heading) - (z - float(pz[k])) * math.sin(heading)
        overshoot = 0.0
        if k == 0 and raw[0] < 0.0:
            overshoot = float(-raw[0] * lengths[0])
        elif k == len(t) - 1 and raw[-1] > 1.0:
            overshoot = float((raw[-1] - 1.0) * lengths[-1])
        return Projection(lane, along, lateral, heading, float(distances[k]), total, overshoot)

    def nearest(self, x: float, z: float, radius: float = 32.0) -> tuple[int, float] | None:
        """The closest lane to a position, and the distance to it.

        Searches the points of every cell within `radius`, then measures to the
        segments either side of them: points are 2 m apart, and a distance to
        the nearest point alone would be up to a metre long.

        Every lane with a point within one spacing of the closest one is
        measured that way, not only the closest point's lane. Picking the lane
        by its nearest point and only then measuring chose a worse lane in 3 of
        2000 queries across Europe -- one whose points were nearer but whose
        line was not.
        """
        candidates = self._points_near(x, z, radius)
        if candidates.size == 0:
            return None

        found = self.points[candidates]
        distances = np.hypot(found[:, 0] - x, found[:, 2] - z)
        shortlist = candidates[distances <= float(distances.min()) + _POINT_SPACING_M]

        best: tuple[int, float] | None = None
        for point in shortlist:
            point = int(point)
            lane = int(np.searchsorted(self.lane_start, point, side="right")) - 1
            distance = self._distance_to_lane(lane, point, x, z)
            if best is None or distance < best[1]:
                best = (lane, distance)

        if best is None or best[1] > radius:
            return None
        return best

    def route(self, start: int, goal: int) -> list[int] | None:
        """Shortest sequence of lanes from one to another, by A*."""
        if start == goal:
            return [start]
        target = self.points[self.lane_start[goal]]
        tx, tz = float(target[0]), float(target[2])

        open_set: list[tuple[float, int]] = [(0.0, start)]
        came_from: dict[int, int] = {}
        best = {start: 0.0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                return path[::-1]
            steps = [(int(f), float(self.lane_length[f])) for f in self.next_lanes(current)]
            steps += [(int(f), LANE_CHANGE_COST_M) for f in self.lanes_alongside(current)]
            for following, step in steps:
                cost = best[current] + step
                if cost >= best.get(following, math.inf):
                    continue
                best[following] = cost
                came_from[following] = current
                end = self.points[self.lane_start[following + 1] - 1]
                heuristic = math.hypot(float(end[0]) - tx, float(end[2]) - tz)
                heapq.heappush(open_set, (cost + heuristic, following))
        return None

    def _points_near(self, x: float, z: float, radius: float) -> np.ndarray:
        reach = math.ceil(radius / CELL_SIZE_M)
        cx, cz = math.floor(x / CELL_SIZE_M), math.floor(z / CELL_SIZE_M)
        wanted = np.array(
            [
                _pack(cx + dx, cz + dz)
                for dx in range(-reach, reach + 1)
                for dz in range(-reach, reach + 1)
            ],
            dtype=np.int64,
        )
        positions = np.searchsorted(self.cell_keys, wanted)
        slices = [
            self.cell_points[self.cell_start[p] : self.cell_start[p + 1]]
            for p, key in zip(positions, wanted, strict=True)
            if p < len(self.cell_keys) and self.cell_keys[p] == key
        ]
        if not slices:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(slices)

    def _distance_to_lane(self, lane: int, point: int, x: float, z: float) -> float:
        first, last = int(self.lane_start[lane]), int(self.lane_start[lane + 1]) - 1
        best = math.inf
        for a in (point - 1, point):
            if a < first or a + 1 > last:
                continue
            best = min(best, _segment_distance(self.points[a], self.points[a + 1], x, z))
        if best == math.inf:
            p = self.points[point]
            best = math.hypot(float(p[0]) - x, float(p[2]) - z)
        return best


_ARRAYS: Final = (
    "points",
    "lane_start",
    "lane_length",
    "lane_source",
    "lane_owner",
    "lane_offset",
    "lane_semaphore",
    "lane_signal",
    "lane_blinker",
    "lane_rule",
    "lane_ai",
    "lane_hidden",
    "lane_start_node",
    "lane_end_node",
    "successor_start",
    "successors",
    "change_start",
    "changes",
    "cell_keys",
    "cell_start",
    "cell_points",
)


_OPTIONAL_ARRAYS: Final = (
    "lane_left_hand",
    "lane_nav_length",
    "lane_nav_speed",
    "lane_speed_class",
    "lane_gps_avoid",
)
"""Arrays a cache may lack, having been built before they were added."""


def _mapped(path: Path) -> np.ndarray:
    """The array in `path`, read from the file as it is used -- as a plain
    array: through `np.memmap` every index went through Python, and a 30 km
    route's path took 6 s, of which 2.6 in the memmap's own code."""
    return np.load(path, mmap_mode="r", allow_pickle=False).view(np.ndarray)


def _rows(lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Lists of indices as compressed sparse rows."""
    start = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum([len(items) for items in lists], out=start[1:])
    flat = np.fromiter(
        (item for items in lists for item in items), dtype=np.int32, count=int(start[-1])
    )
    return start, flat


def _pack(cx: int, cz: int) -> int:
    return ((cx + _CELL_BIAS) << 22) | (cz + _CELL_BIAS)


def _point_index(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group point indices by grid cell, as sorted keys plus row offsets."""
    cx = np.floor(points[:, 0] / CELL_SIZE_M).astype(np.int64)
    cz = np.floor(points[:, 2] / CELL_SIZE_M).astype(np.int64)
    keys = ((cx + _CELL_BIAS) << 22) | (cz + _CELL_BIAS)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    cell_keys, first = np.unique(sorted_keys, return_index=True)
    cell_start = np.append(first, len(sorted_keys)).astype(np.int64)
    return cell_keys, cell_start, order.astype(np.int64)


def _segment_distance(a: np.ndarray, b: np.ndarray, x: float, z: float) -> float:
    ax, az, bx, bz = float(a[0]), float(a[2]), float(b[0]), float(b[2])
    dx, dz = bx - ax, bz - az
    span = dx * dx + dz * dz
    t = 0.0 if span == 0.0 else max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / span))
    return math.hypot(ax + t * dx - x, az + t * dz - z)


def _blinker(name: str) -> int:
    return BLINKERS.index(name) if name in BLINKERS else 0


def _node(uid: str | None) -> int:
    return NO_NODE if uid is None else int(uid, 16)


def _uid(value: int) -> str | None:
    return None if value == NO_NODE else f"{value:016X}"
