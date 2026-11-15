"""The road network as a directed graph of lanes.

A route is a sequence of lanes, so a lane is the unit: a directed polyline
with a length, a successor set, and whatever the game says about driving it.

Two kinds go in. Roads are stored as a single centreline plus a road look, so
their lanes are made by offsetting that centreline sideways and pointing each
one the way its traffic goes. Junction lanes come ready-made in the prefab
descriptors, already one per direction. They are joined where their ends
coincide, which works because the extracted geometry lands within millimetres
of the game's own -- see `docs/results/map_extraction.md`.

Nothing here knows about the truck. What it answers is "which lane is this
position on" and "where can that lane go next", which is what a planner
needs and all it needs.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from core.map.sector import Road, RoadLook, Sector, Signal

__all__ = ["Lane", "LaneGraph"]

#: Endpoints closer than this are the same place. The extractor's geometry
#: agrees with the game's to millimetres, so this is slack, not tolerance.
JOIN_TOLERANCE_M: Final = 0.5

#: How far apart two lanes at a shared map node may end and start and still
#: be the same lane continuing. Wider than one lane, narrower than two.
MAX_NODE_GAP_M: Final = 7.0

#: Cosine of the largest turn allowed between a lane and its continuation at a
#: shared node. Lanes meeting at a road-to-road node run nearly straight on.
MIN_ALIGNMENT: Final = 0.7

#: Widest offset step between two lanes that are still one carriageway: a lane
#: width with room to spare. The narrowest measured is 2.5 m, the common 4.5 m.
MAX_LANE_CHANGE_GAP_M: Final = 5.5

#: What a lane change costs a route, in metres of driving. Not the length of
#: the lane moved to -- that runs alongside the one already paid for -- but a
#: price on the manoeuvre, so a route does not weave for a metre's gain.
LANE_CHANGE_COST_M: Final = 50.0

#: Grid cell for the nearest-lane search, metres.
CELL_SIZE_M: Final = 32.0


@dataclass(frozen=True, slots=True)
class Lane:
    """One directed strip of drivable road."""

    points: np.ndarray
    """`(n, 3)` in the direction of travel."""

    source: str
    """`road` or `prefab`: where the geometry came from."""

    owner: str
    """Uid of the road or prefab it belongs to."""

    offset: float = 0.0
    """Lateral offset from the road centreline, positive left. Zero in junctions."""

    semaphore: int = -1
    signal: Signal | None = None
    """The traffic light or barrier at the lane's start, if it has one."""

    rule: str = ""
    blinker: str = ""
    ai_vehicles: bool = True
    hidden: bool = False
    """On a road the game's own map and GPS leave out: service roads, and
    roads closed off behind yellow crosses."""

    left_hand: bool = False
    """On a road where traffic keeps left: Britain. Its exits are on the left."""

    start_node: str | None = None
    """Uid of the map node the lane leaves from, if it leaves from one."""

    end_node: str | None = None

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    @property
    def length(self) -> float:
        return float(np.hypot(*np.diff(self.points[:, [0, 2]], axis=0).T).sum())


@dataclass
class LaneGraph:
    """Lanes and the connections between them."""

    lanes: list[Lane] = field(default_factory=list)
    successors: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    predecessors: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    lane_changes: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    """Lanes alongside, running the same way, that a vehicle can move across to.

    Kept apart from `successors` because it is a different act: a successor is
    where a lane leads, a lane change is a manoeuvre with a blinker and a gap
    to find. The network needs both. Without lane changes, a lane that the
    junction ahead does not feed is a dead end, and 27% of the continent's
    road lanes -- 9% of the base game's, every one of them a motorway or
    multi-lane road -- came out cut off from the rest.
    """

    @classmethod
    def build(cls, sector: Sector) -> LaneGraph:
        """Turn an extraction into a graph."""
        graph = cls()

        meeting = _roads_meeting(sector)
        for road in sector.roads:
            look = sector.looks.get(road.look)
            if look is None:
                continue
            first = len(graph.lanes)
            lanes = _road_lanes(
                road,
                look,
                _offsets_met(sector, road, road.start_node, meeting),
                _offsets_met(sector, road, road.end_node, meeting),
            )
            graph.lanes.extend(lanes)
            graph._link_alongside(road, first, lanes)

        for prefab in sector.prefabs:
            first = len(graph.lanes)
            for lane in prefab.lanes:
                graph.lanes.append(
                    Lane(
                        points=lane.points,
                        source="prefab",
                        owner=prefab.uid,
                        semaphore=lane.semaphore,
                        signal=sector.signal(prefab, lane.semaphore),
                        rule=lane.rule,
                        blinker=lane.blinker,
                        start_node=lane.from_node,
                        end_node=lane.to_node,
                    )
                )
            # Inside a junction the connections are stated, not inferred.
            for index, lane in enumerate(prefab.lanes):
                for following in lane.next_lanes:
                    if following < len(prefab.lanes):
                        graph.connect(first + index, first + following)

        graph._join_ends()
        graph._join_at_shared_nodes()
        graph._index()
        return graph

    def _link_alongside(self, road: Road, first: int, lanes: list[Lane]) -> None:
        """Join each lane to its neighbours on the same carriageway.

        Neighbours are lanes of the same road running the same way, next to
        each other in offset. Not across the centre line, and not across a gap
        wider than a lane: two carriageways a verge apart are not one road to
        change lanes on.
        """
        for forward in (True, False):
            side = sorted(
                (lane.offset, first + index)
                for index, lane in enumerate(lanes)
                if (lane.start_node == road.start_node) == forward
            )
            for (a_offset, a), (b_offset, b) in itertools.pairwise(side):
                if abs(b_offset - a_offset) <= MAX_LANE_CHANGE_GAP_M:
                    self.lane_changes[a].append(b)
                    self.lane_changes[b].append(a)

    def connect(self, source: int, target: int) -> None:
        if target not in self.successors[source]:
            self.successors[source].append(target)
            self.predecessors[target].append(source)

    def nearest(self, x: float, z: float, radius: float = CELL_SIZE_M) -> tuple[int, float] | None:
        """The closest lane to a position, and how far away it is.

        Returns None when nothing is within `radius`, which is the honest
        answer off-road and in the yards the extraction does not cover.
        """
        best: tuple[int, float] | None = None
        cell = (int(x // CELL_SIZE_M), int(z // CELL_SIZE_M))
        reach = int(radius // CELL_SIZE_M) + 1
        for dx in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                for index in self._cells.get((cell[0] + dx, cell[1] + dz), ()):
                    points = self.lanes[index].points
                    distance = float(np.hypot(points[:, 0] - x, points[:, 2] - z).min())
                    if distance <= radius and (best is None or distance < best[1]):
                        best = (index, distance)
        return best

    def route(self, start: int, goal: int) -> list[int] | None:
        """The shortest sequence of lanes from one to another, or None.

        A* over lane lengths, with straight-line distance to the goal as the
        heuristic -- admissible because no lane is shorter than the distance
        it covers.
        """
        if start == goal:
            return [start]

        target = self.lanes[goal].start
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

            steps = [(f, self.lanes[f].length) for f in self.successors.get(current, ())]
            steps += [(f, LANE_CHANGE_COST_M) for f in self.lane_changes.get(current, ())]
            for following, step in steps:
                cost = best[current] + step
                if cost >= best.get(following, math.inf):
                    continue
                best[following] = cost
                came_from[following] = current
                end = self.lanes[following].end
                heapq.heappush(
                    open_set,
                    (cost + float(np.hypot(end[0] - target[0], end[2] - target[2])), following),
                )
        return None

    def _join_ends(self) -> None:
        """Connect a lane to every lane that starts where it finishes."""
        starts: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, lane in enumerate(self.lanes):
            starts[_key(lane.start)].append(index)

        for index, lane in enumerate(self.lanes):
            end = lane.end
            for key in _neighbourhood(end):
                for candidate in starts.get(key, ()):
                    if candidate == index:
                        continue
                    if _distance(end, self.lanes[candidate].start) <= JOIN_TOLERANCE_M:
                        self.connect(index, candidate)

    def _join_at_shared_nodes(self) -> None:
        """Connect what geometry missed but the map says is connected.

        Where two roads of different looks meet, their lanes need not line up:
        a lane is added or dropped, or the carriageway gap changes, and the
        ends sit a lane width apart. Across Europe 1562 visible road lanes
        ended between 1 and 6 m from the nearest lane start. The map node they
        share says they continue into each other; which lane continues into
        which is the nearest one pointing the same way.

        Only lanes that geometry left with nowhere to go are considered, so
        this adds the missing link rather than second-guessing a measured one.
        """
        starts_at: dict[str, list[int]] = defaultdict(list)
        for index, lane in enumerate(self.lanes):
            if lane.start_node is not None:
                starts_at[lane.start_node].append(index)

        for index, lane in enumerate(self.lanes):
            if self.successors.get(index) or lane.end_node is None:
                continue
            heading = _direction(lane.points, at_end=True)
            best: tuple[float, int] | None = None
            for candidate in starts_at.get(lane.end_node, ()):
                other = self.lanes[candidate]
                if other.owner == lane.owner:
                    continue
                if float(np.dot(heading, _direction(other.points, at_end=False))) < MIN_ALIGNMENT:
                    continue
                gap = _distance(lane.end, other.start)
                if gap <= MAX_NODE_GAP_M and (best is None or gap < best[0]):
                    best = (gap, candidate)
            if best is not None:
                self.connect(index, best[1])

    def _index(self) -> None:
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, lane in enumerate(self.lanes):
            for point in lane.points:
                cell = (int(point[0] // CELL_SIZE_M), int(point[2] // CELL_SIZE_M))
                if not cells[cell] or cells[cell][-1] != index:
                    cells[cell].append(index)
        self._cells = cells

    _cells: dict[tuple[int, int], list[int]] = field(default_factory=dict, init=False)


def _road_lanes(
    road: Road,
    look: RoadLook,
    at_start: tuple[float, ...] | None = None,
    at_end: tuple[float, ...] | None = None,
) -> list[Lane]:
    """Offset a road's centreline into one directed lane per lane.

    `at_start` and `at_end` are where the road met at either end has its
    lanes, in this road's frame (`_offsets_met`): each lane leaves the one
    and joins the other, easing onto its own offset in between.
    """
    measurement = look.measurement(road.left_hand_traffic)
    if measurement is None or len(road.points) < 2:
        return []
    offsets = measurement.offsets

    centre = road.points
    tangents = _tangents(centre)
    # Left of a heading (sin, cos) is (cos, -sin).
    left = np.column_stack([tangents[:, 1], np.zeros(len(tangents)), -tangents[:, 0]])
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centre[:, [0, 2]], axis=0).T))])
    u = along / max(float(along[-1]), 1e-6)
    ease = u * u * (3.0 - 2.0 * u)

    lanes = []
    for index, offset in enumerate(offsets):
        lateral = np.full(len(centre), offset)
        if at_start is not None:
            lateral += (at_start[index] - offset) * (1.0 - ease)
        if at_end is not None:
            lateral += (at_end[index] - offset) * ease
        points = centre + left * lateral[:, None]
        # Which way traffic runs on this lane is measured at the junctions the
        # look meets, not inferred from which side of the centreline it is on:
        # a one-way road has every lane running the same way, and the side rule
        # gets six of the fifteen looks surveyed in a city sector wrong. The
        # fallback is only for looks no junction was seen for.
        if measurement.forward is not None and index < len(measurement.forward):
            forward = measurement.forward[index]
        else:
            forward = offset < 0.0 if not road.left_hand_traffic else offset > 0.0
        lanes.append(
            Lane(
                points=points if forward else points[::-1].copy(),
                source="road",
                owner=road.uid,
                offset=offset,
                ai_vehicles=road.ai_vehicles,
                hidden=road.hidden,
                left_hand=road.left_hand_traffic,
                start_node=road.start_node if forward else road.end_node,
                end_node=road.end_node if forward else road.start_node,
            )
        )
    return lanes


JOIN_STEP_MIN_M: Final = 0.2
"""Lanes of two roads meeting end to end that are further apart than this
are eased together along the road rather than left to meet a step apart."""


def _roads_meeting(sector: Sector) -> dict[str, list[Road]]:
    """The roads at each map node."""
    meeting: dict[str, list[Road]] = defaultdict(list)
    for road in sector.roads:
        meeting[road.start_node].append(road)
        meeting[road.end_node].append(road)
    return meeting


def _offsets_met(
    sector: Sector, road: Road, node: str, meeting: dict[str, list[Road]]
) -> tuple[float, ...] | None:
    """Where the one other road at `node` has its lanes, lane for lane with
    `road`'s and in its frame; None where there is no such road, where they
    already meet, or where the two do not have the same lanes each way.

    Two looks' lanes need not sit at the same offsets: a German motorway's
    median widens from 2 m to 4 m between two pieces, a British one's from
    4 m to 30 m, and the pieces share a node. The road is continuous in the
    game; drawn at each look's offsets it was a 1 m step in the line, or a
    13 m gap that cut Plymouth off from the rest of the map.

    In the game a road goes from the lanes of the road before it, at its
    start, to its own, at its end. Near Exeter a British motorway piece
    (look2, lanes 7.25 m out) runs on from one with lanes 4.25 m out, and
    60% along it the painted lane was 5.9 m out. Eased at both ends instead,
    onto the next piece's 17.25 m as well, the line was 12.8 m out there:
    the truck drove the hard shoulder, and hit something on it.
    """
    others = [other for other in meeting.get(node, ()) if other is not road]
    if len(others) != 1:
        return None
    other = others[0]
    if road.end_node == node and other.start_node == node:
        return None  # the road on eases from this one's lanes onto its own
    if not (road.start_node == node and other.end_node == node) and road.uid < other.uid:
        return None  # end to end or start to start: one of the two eases onto the other
    mine = _layout(sector, road)
    theirs = _layout(sector, other)
    if mine is None or theirs is None or len(mine[0]) != len(theirs[0]):
        return None
    offsets, forward = theirs
    # Seen from this road: the same way along if the other ends where this
    # starts (or starts where this ends); the other way round if both start
    # or both end here, left and right swapped.
    same_way = (other.end_node == node) != (road.end_node == node)
    if not same_way:
        offsets = tuple(-o for o in offsets)
        forward = tuple(not f for f in forward)
    own_offsets, own_forward = mine
    matched: list[float] = [0.0] * len(own_offsets)
    for way in (True, False):
        own = sorted(
            (o, i)
            for i, (o, f) in enumerate(zip(own_offsets, own_forward, strict=True))
            if f == way
        )
        met = sorted(o for o, f in zip(offsets, forward, strict=True) if f == way)
        if len(own) != len(met):
            return None
        for (_, i), o in zip(own, met, strict=True):
            matched[i] = o
    if max(abs(a - b) for a, b in zip(matched, own_offsets, strict=True)) <= JOIN_STEP_MIN_M:
        return None
    return tuple(matched)


def _layout(sector: Sector, road: Road) -> tuple[tuple[float, ...], tuple[bool, ...]] | None:
    """A road's lane offsets and which run forward along it."""
    look = sector.looks.get(road.look)
    measurement = look.measurement(road.left_hand_traffic) if look else None
    if measurement is None:
        return None
    offsets = measurement.offsets
    forward = tuple(
        measurement.forward[i]
        if measurement.forward is not None and i < len(measurement.forward)
        else (o < 0.0 if not road.left_hand_traffic else o > 0.0)
        for i, o in enumerate(offsets)
    )
    return offsets, forward


def _tangents(points: np.ndarray) -> np.ndarray:
    """Unit direction at each point, in the ground plane."""
    dx = np.gradient(points[:, 0])
    dz = np.gradient(points[:, 2])
    norm = np.hypot(dx, dz)
    norm[norm == 0.0] = 1.0
    return np.column_stack([dx / norm, dz / norm])


def _direction(points: np.ndarray, *, at_end: bool) -> np.ndarray:
    """Unit heading in the ground plane at one end of a polyline."""
    a, b = (points[-2], points[-1]) if at_end else (points[0], points[1])
    vector = np.array([b[0] - a[0], b[2] - a[2]])
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 0.0 else vector


def _key(point: np.ndarray) -> tuple[int, int]:
    return (int(point[0] // JOIN_TOLERANCE_M), int(point[2] // JOIN_TOLERANCE_M))


def _neighbourhood(point: np.ndarray) -> list[tuple[int, int]]:
    cx, cz = _key(point)
    return [(cx + dx, cz + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))
