"""From a position to somewhere else, along lanes the game says exist.

A route is a sequence of lanes and how each leads to the next: onwards along
the road, or across into a lane alongside. That is what a planner reasons in.
What a controller follows is one continuous line, so the route also knows how
to draw itself: lanes end to end, and each lane change drawn as a smooth move
across rather than a 4.5 m step, which is what the lane sequence would say
literally and what no controller should be asked to follow.

    route = plan(network, start=(x, z, course), through=[(x1, z1), (x2, z2)])
    path = ReferencePath(route.points()[:, [0, 2]])
"""

from __future__ import annotations

import functools
import heapq
import json
import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from core.map import CompanyPlace, LaneMatcher, RoadNetwork, Signal
from core.map.network import BLINKERS

__all__ = [
    "Crossing",
    "Route",
    "RouteError",
    "RouteHazards",
    "StopLine",
    "load_crossings",
    "load_points",
    "load_stop_lines",
    "plan",
    "plan_ahead",
    "plan_to_depot",
    "reroute_from",
    "save",
]

#: Distance over which a lane change is drawn. A smoothstep across a lane of
#: 4.5 m in time T peaks at 6 * 4.5 / T^2 of lateral acceleration, so length
#: sets how hard the move is: 50 m at 60 km/h is 3 s and 3 m/s^2 -- harder
#: than any bend on the autostrada it was planned for, 2.6 m/s^2 in
#: simulation. 120 m is about 7 s and 0.5 m/s^2 at 60, and gentler below it.
#: The move is confined to the lane it starts on, so a shorter lane makes it
#: sharper.
LANE_CHANGE_LENGTH_M: Final = 120.0

#: A move across finishes this far before the end of the lane it starts on,
#: rather than starting where that lane does: a road's lanes can run side by
#: side for a kilometre, and moved across at their start the truck changed
#: lanes long before it needed to -- into an exit lane 400 m ahead of the
#: exit, before the minimap could show which way the GPS went there.
LANE_CHANGE_FINISH_M: Final = 100.0

#: What the planner counts a lane change as, in metres of road. At 50 it
#: moved across to cut a corner and back again 3 km on; a driver keeps to
#: a lane unless the way on needs another.
JOIN_EASE_M: Final = 80.0
JOIN_STEP_MAX_M: Final = 4.0
"""Two lanes one after the other that meet up to this far apart are one lane
of a road whose pieces' looks differ -- a motorway's median 2 m wide, then 4
-- and the line eases across the step over `JOIN_EASE_M` rather than take
it: a 1 m step drawn as it is was braked for as a bend of 5 m radius."""

CHANGE_COST_M: Final = 300.0

#: A lane shorter than this is no place to move across -- the move is
#: confined to the lane, and on an 18 m one it was a swerve -- so a change
#: from one costs `SHORT_CHANGE_PENALTY_M` more.
SHORT_CHANGE_M: Final = 100.0
SHORT_CHANGE_PENALTY_M: Final = 2000.0

#: A stretch on roads without traffic mid-route -- a border's customs lanes
#: -- is one passed through if its mapped length is no more than this times
#: the distance across it, and it turns no more than this; one that loops
#: round is a yard or a rest area.
QUIET_STRAIGHT_RATIO: Final = 1.25

#: A taught crossing is used for a customs stretch whose ends are this close
#: to where it was taught, m, and is joined on over this, m.
CROSSING_MATCH_M: Final = 15.0

#: A taught passage is on a route that passes within this of both its ends,
#: going its way, in order.
PASSAGE_MATCH_M: Final = 6.0
CROSSING_BLEND_M: Final = 15.0
PASSAGE_JOIN_M: Final = 2.5
PASSAGE_JOIN_DEG: Final = 30.0
PASSAGE_JOINED_MIN_M: Final = 30.0
"""A route coming into a taught passage by another way than the driver's,
where it first runs within `PASSAGE_JOIN_M` of the taught line going its
way within `PASSAGE_JOIN_DEG`, takes the taught line on from there to its
end -- if at least `PASSAGE_JOINED_MIN_M` of it is left. Into France at
Basel, by a road joining the taught way half through the customs, the
truck kept to the map's lanes, and they ran through the island of the
'Douane France' sign."""

#: How much further from the road's centre a customs stretch's lanes are
#: drawn, m, easing in and out over the ramp at its ends.
CUSTOMS_SPREAD_M: Final = 1.5
CUSTOMS_SPREAD_RAMP_M: Final = 20.0
QUIET_MAX_TURN_DEG: Final = 45.0

#: How far from a waypoint the network may be and still be where it meant.
WAYPOINT_RADIUS_M: Final = 30.0

#: How far from a depot's gate the road may be. A gate is a node of the
#: yard; the yard's own driveway is not on the network, and in Turin
#: (transinet) the road was 31.7 m from the nearer gate.
GATE_RADIUS_M: Final = 60.0

#: What a metre of a hidden road costs, in metres. The game's GPS never uses
#: roads its map does not show, and one of them, closed off by yellow
#: crosses where the game has no road beyond, nearly put the truck into an
#: invisible wall in Milan. Not forbidden: a truck can start on one.
HIDDEN_COST: Final = 50.0

#: What a metre of a road without traffic costs. At a border on the way to
#: Gothenburg the planner's shortcut went through the customs yard -- roads
#: the game's traffic does not use, shown on the map -- and the truck hit a
#: parked truck and a booth. The GPS keeps to the road through the border.
NO_TRAFFIC_COST: Final = 10.0

#: What missing a waypoint by a metre is worth, in metres of driving.
MISS_WEIGHT: Final = 20.0

ROUTE_FORMAT: Final = "ets2ad-route-1"

PORT_BOOTHS_WITHIN_M: Final = 1500.0
"""A drive to a ship ends before the first booth this near it (`Route.to_ferry`)."""

PORT_QUIET_WITHIN_M: Final = 2000.0
"""A drive to a ship, or into a port, ends where the route first goes onto
roads without the game's traffic this near the ship or the port: the
port's own roads, passport booths and lanes to the quay, which the driver
takes by hand (the driver's word, 12 November). At Calais the booths have
no barrier the map knows; the truck drove into them at 30 km/h and hit an
island, and before them was side-swiped by a lorry changing lanes."""

PORT_SHORT_M: Final = 100.0
"""Or, with no booth that near, this far short of the ship. At 300 the truck
stopped 378 m out of Dover's port, further than the driver wanted to go."""

ONWARD: Final = "onward"
CHANGE: Final = "change"
FERRY: Final = "ferry"
"""Onto a ship, or the Channel Tunnel's train, and off at the next lane
(core.map.ferries). A drive ends where the truck boards: the crossing is
the game's, taken from its menu."""


@dataclass(frozen=True, slots=True)
class RouteHazards:
    junctions: int
    traffic_lights: int
    """Distinct lights governing a lane of the route."""

    first_traffic_light_m: float | None
    left_turns: int
    """Junction lanes that expect the left indicator."""

    right_turns: int
    lane_changes: int
    barriers: int = 0
    """Toll gates, level crossings and other barriers across the route."""


@dataclass(frozen=True, slots=True)
class StopLine:
    """Where the route enters a lane a traffic light governs."""

    position: np.ndarray
    """`(3,)` world position of the lane's start: the stop line."""

    signal: Signal
    """The light, as the map knows it; `Signal("", "")` for an older cache
    that does not say what the light is."""


class RouteError(RuntimeError):
    """No route: off the network, or nothing leads where it was asked to go."""


@dataclass(frozen=True)
class Route:
    """Lanes in order, and how each leads to the next."""

    network: RoadNetwork
    lanes: tuple[int, ...]
    steps: tuple[str, ...]
    """`steps[i]` is how `lanes[i]` leads to `lanes[i + 1]`: onward or change."""

    @property
    def length(self) -> float:
        """Metres of road, not counting a lane change as extra distance."""
        return float(
            sum(
                float(self.network.lane_length[lane])
                for lane, step in zip(self.lanes, (ONWARD, *self.steps), strict=True)
                if step == ONWARD
            )
        )

    def ahead(self, start: int, metres: float) -> Route:
        """The route up to the first lane that ends `metres` of road past the
        start of `lanes[start]`; the whole of it if it is no longer."""
        run = 0.0
        for index in range(start, len(self.lanes)):
            if index == start or self.steps[index - 1] == ONWARD:
                run += float(self.network.lane_length[self.lanes[index]])
            if run >= metres:
                return Route(self.network, self.lanes[: index + 1], self.steps[:index])
        return self

    @property
    def lane_changes(self) -> int:
        return self.steps.count(CHANGE)

    @property
    def ferry(self) -> int | None:
        """Index in `lanes` of the lane the first crossing is boarded from."""
        return self.steps.index(FERRY) if FERRY in self.steps else None

    def to_ferry(self) -> Route:
        """The route as far as the truck drives it: to the port of its first
        crossing, short of the port's booths (`PORT_BOOTHS_WITHIN_M`), or
        `PORT_SHORT_M` short of the ship where it has none. Past the booths
        a port is queues and ramps the map does not show -- at Calais the
        truck caught a booth's island -- and the driver takes the ship from
        the game's menu anyway.

        Nor does it go into the port's junction, the ship's or not: a route
        going in, crossing or not -- the minimap's red line followed the
        straightest way on -- ends where it goes in, or short of the booths.
        Into Kristiansand's port the truck drove out over the quay, among
        its bollards, and hit one."""
        at = self.ferry
        if at is None:
            entry = self._port_entry()
            if entry is None:
                return self
            cut = self._booth_before(entry, self.lanes[: entry + 1])
            cut = entry if cut is None else cut
            quiet = self._quiet_before(entry)
            return self._cut(cut if quiet is None else min(cut, quiet))
        lanes = self.lanes[: at + 1]
        cut = self._booth_before(at, lanes)
        if cut is None:
            back = 0.0
            for index in range(len(lanes) - 1, 0, -1):
                back += float(self.network.lane_length[lanes[index]])
                if back >= PORT_SHORT_M:
                    cut = index
                    break
        # Never into the port itself: Kristiansand's runs 143 m back from
        # the ship, and 100 m short of it the truck was out on the quay, among
        # its bollards.
        entry = self._port_entry()
        if entry is not None and entry <= at and (cut is None or entry < cut):
            cut = entry
        quiet = self._quiet_before(at)
        if quiet is not None and (cut is None or quiet < cut):
            cut = quiet
        if cut is None or cut < 1:
            return Route(self.network, lanes, self.steps[:at])
        return self._cut(cut)

    def _cut(self, cut: int) -> Route:
        """The route ending before its lane `cut`."""
        if cut < 1:
            cut = 1
        return Route(self.network, self.lanes[:cut], self.steps[: cut - 1])

    def _booth_before(self, at: int, lanes) -> int | None:
        """Back from `lanes[at]`, the first booth within `PORT_BOOTHS_WITHIN_M`:
        the index of its lane, the drive ending before it; None if none."""
        back = 0.0
        cut = None
        for index in range(at, 0, -1):
            lane = lanes[index]
            signal = self.network.signal(lane)
            if signal is not None and signal.kind and not signal.is_traffic_light:
                cut = index
            back += float(self.network.lane_length[lane])
            if back > PORT_BOOTHS_WITHIN_M:
                break
        return cut

    def _quiet_before(self, at: int) -> int | None:
        """Index of the first lane without the game's traffic within
        `PORT_QUIET_WITHIN_M` before lane `at`, after some with it: where the
        port's own roads begin (`PORT_QUIET_WITHIN_M`); None if none."""
        network = self.network
        ai = network.lane_ai
        lengths = network.lane_length
        before = [0.0] * (at + 1)
        for index in range(at - 1, -1, -1):
            before[index] = before[index + 1] + float(lengths[self.lanes[index]])
        with_traffic = False
        for index in range(at + 1):
            lane = self.lanes[index]
            if bool(ai[lane]):
                with_traffic = True
            elif with_traffic and before[index] <= PORT_QUIET_WITHIN_M:
                return index
        return None

    def _port_entry(self) -> int | None:
        """Index of the lane the route goes into a port by: the junction a
        ship is boarded from (core.map.ferries); None if it goes into none.
        A route starting in a port, off a ship, is not going into it."""
        links = getattr(self.network, "ferry_links", None)
        if not links:
            return None
        owners = self.network.lane_owner
        ports = {owners[int(lane)] for lane in links}
        outside = False
        for index, lane in enumerate(self.lanes):
            inside = owners[int(lane)] in ports
            if inside and outside:
                return index
            outside = outside or not inside
        return None

    def hazards(self) -> RouteHazards:
        """What along the route a driver has to watch for, as the map says.

        Counted before driving, because nothing yet sees any of it: no traffic
        light, no vehicle ahead. Until perception exists, a route to drive for
        real is one chosen for having little of this.
        """
        junctions: set[int] = set()
        lights: set[tuple[int, int]] = set()
        barriers: set[tuple[int, int]] = set()
        left = right = 0
        at = 0.0
        first_light: float | None = None
        for lane, step in zip(self.lanes, (ONWARD, *self.steps), strict=True):
            owner = int(self.network.lane_owner[lane])
            if self.network.lane_source[lane] == 1:
                junctions.add(owner)
            semaphore = int(self.network.lane_semaphore[lane])
            signal = self.network.signal(lane)
            if signal is None:
                pass
            elif signal.kind and not signal.is_traffic_light:
                barriers.add((owner, semaphore))
            else:
                lights.add((owner, semaphore))
                if first_light is None:
                    first_light = at
            blinker = int(self.network.lane_blinker[lane])
            left += blinker == BLINKERS.index("LeftBlinker")
            right += blinker == BLINKERS.index("RightBlinker")
            if step == ONWARD:
                at += float(self.network.lane_length[lane])
        return RouteHazards(
            junctions=len(junctions),
            traffic_lights=len(lights),
            first_traffic_light_m=first_light,
            left_turns=left,
            right_turns=right,
            lane_changes=self.lane_changes,
            barriers=len(barriers),
        )

    def stop_lines(self) -> list[StopLine]:
        """Every traffic light on the route, where the truck must stop for it.

        A light governs the lanes that start at its stop line, so the line
        is where the route first enters one. Barriers -- toll gates, level
        crossings, depot gates -- are not included: they are not lights; nor
        are the lines a driver has said have none (`load_no_lights`).
        """
        lines: list[StopLine] = []
        previous: tuple[int, int] | None = None
        none_there = load_no_lights()
        for lane in self.lanes:
            semaphore = int(self.network.lane_semaphore[lane])
            key = (int(self.network.lane_owner[lane]), semaphore) if semaphore >= 0 else None
            signal = self.network.signal(lane)
            if (
                key is not None
                and key != previous
                and signal is not None
                and (signal.is_traffic_light or not signal.kind)
            ):
                at = np.asarray(self.network.lane_points(lane)[0], float)
                if not (
                    len(none_there)
                    and np.hypot(*(none_there - at[[0, 2]]).T).min() < NO_LIGHT_WITHIN_M
                ):
                    lines.append(StopLine(at, signal))
            previous = key
        return lines

    def barriers(self) -> list[StopLine]:
        """Toll gates and other barriers on the route, where each begins.

        Not stopped at -- with the barriers made passable the truck drives
        through -- but slowed for: a toll lane is a booth's width.
        """
        found: list[StopLine] = []
        previous: tuple[int, int] | None = None
        for lane in self.lanes:
            semaphore = int(self.network.lane_semaphore[lane])
            key = (int(self.network.lane_owner[lane]), semaphore) if semaphore >= 0 else None
            signal = self.network.signal(lane)
            if (
                key is not None
                and key != previous
                and signal is not None
                and signal.kind
                and not signal.is_traffic_light
            ):
                found.append(StopLine(np.asarray(self.network.lane_points(lane)[0], float), signal))
            previous = key
        return found

    def points(self, lane_change_length: float = LANE_CHANGE_LENGTH_M) -> np.ndarray:
        """The route as one `(n, 3)` line, in the direction of travel.

        A lane followed by the lane alongside it becomes one stretch that
        starts on the first and eases across onto the second over
        `lane_change_length` metres, finishing `LANE_CHANGE_FINISH_M` before
        the lane ends (`change_begins`): what comes after a lane change
        usually needs the new lane, but not a kilometre before.
        """
        pieces: list[np.ndarray] = []
        spread = self._customs_spread()
        current = spread.get(0, np.asarray(self.network.lane_points(self.lanes[0]), dtype=float))
        for index, (step, lane) in enumerate(zip(self.steps, self.lanes[1:], strict=True), 1):
            following = spread.get(index, np.asarray(self.network.lane_points(lane), dtype=float))
            if step == CHANGE:
                current = _ease_across(current, following, lane_change_length)
                continue
            pieces.append(current)
            current = following
        pieces.append(current)

        joined = [pieces[0]]
        steps: list[tuple[int, np.ndarray]] = []
        at = len(pieces[0])
        for piece in pieces[1:]:
            gap = piece[0] - joined[-1][-1]
            # Consecutive lanes share their meeting point; drop the duplicate.
            if np.hypot(gap[0], gap[2]) < 0.5:
                piece = piece[1:]
            elif np.hypot(gap[0], gap[2]) <= JOIN_STEP_MAX_M:
                steps.append((at, gap))
            if len(piece):
                joined.append(piece)
                at += len(piece)
        line = np.vstack(joined)
        if steps:
            line = _ease_steps(line, steps)
        return self._through_taught_crossings(line)

    def taught_crossings(self) -> list[tuple[int, int, np.ndarray]]:
        """The customs stretches a driver has shown the way through
        (`load_crossings`), as (first, last, points) -- `points` `(n, 3)`."""
        found = []
        for crossing in load_crossings():
            if crossing.kind != PASSAGE:
                continue
            first, last = self._passage(crossing.points)
            if first is not None and last is not None:
                found.append((first, last, crossing.points))
                continue
            joined = self._joined_passage(crossing.points)
            if joined is not None:
                found.append(joined)
        for first, last in self.quiet_stretches():
            start = np.asarray(self.network.lane_points(self.lanes[first])[0], float)
            end = np.asarray(self.network.lane_points(self.lanes[last])[-1], float)
            for crossing in load_crossings():
                if crossing.kind != CUSTOMS:
                    continue
                if (
                    np.hypot(*(crossing.entry - start[[0, 2]])) < CROSSING_MATCH_M
                    and np.hypot(*(crossing.exit - end[[0, 2]])) < CROSSING_MATCH_M
                ):
                    found.append((first, last, crossing.points))
                    break
        return found

    def _passage(self, points: np.ndarray) -> tuple[int | None, int | None]:
        """Indices of the route's lanes where a taught passage begins and
        ends: the first lane going its way within `PASSAGE_MATCH_M` of its
        start, and the first from there of its end; None where the route
        does not pass it."""
        ends = []
        after = 0
        for at, onward in ((points[0], points[min(2, len(points) - 1)]), (points[-1], None)):
            if onward is None:
                onward = 2.0 * points[-1] - points[max(0, len(points) - 3)]
            heading = math.atan2(onward[0] - at[0], onward[2] - at[2])
            near = {
                int(lane)
                for lane in self.network.lanes_near(float(at[0]), float(at[2]), PASSAGE_MATCH_M)
            }
            index = None
            for i in range(after, len(self.lanes)):
                lane = self.lanes[i]
                if lane not in near:
                    continue
                projection = self.network.project(lane, float(at[0]), float(at[2]))
                turn = (projection.heading - heading + math.pi) % (2.0 * math.pi) - math.pi
                if projection.distance <= PASSAGE_MATCH_M and abs(turn) < math.radians(45.0):
                    index = i
                    break
            if index is None:
                return None, None
            ends.append(index)
            after = index
        return ends[0], ends[1]

    def _joined_passage(self, points: np.ndarray) -> tuple[int, int, np.ndarray] | None:
        """(first, last, points) of a taught passage the route comes into by
        another way, from where it joins the taught line on
        (`PASSAGE_JOIN_M`); None if it does not, or ends elsewhere."""
        end = self._on_route(
            points[-1], _heading(points[max(0, len(points) - 3)], points[-1]), PASSAGE_MATCH_M, 45.0
        )
        if end is None:
            return None
        step = np.hypot(*np.diff(points[:, [0, 2]], axis=0).T)
        left = np.concatenate([np.cumsum(step[::-1])[::-1], [0.0]])
        for k in range(len(points) - 1):
            if left[k] < PASSAGE_JOINED_MIN_M:
                return None
            heading = _heading(points[k], points[k + 1])
            first = self._on_route(points[k], heading, PASSAGE_JOIN_M, PASSAGE_JOIN_DEG)
            if first is not None and first <= end:
                return first, end, points[k:]
        return None

    @functools.cached_property
    def _lane_indices(self) -> dict[int, int]:
        """Each of the route's lanes -> where it first comes in `lanes`."""
        indices: dict[int, int] = {}
        for i, lane in enumerate(self.lanes):
            indices.setdefault(int(lane), i)
        return indices

    def _on_route(
        self, at: np.ndarray, heading: float, within: float, turn_deg: float
    ) -> int | None:
        """Index of the first of the route's lanes passing within `within` of
        `at`, going `heading` there within `turn_deg`."""
        indices = self._lane_indices
        near = sorted(
            (indices[int(lane)], int(lane))
            for lane in self.network.lanes_near(float(at[0]), float(at[2]), within)
            if int(lane) in indices
        )
        for i, lane in near:
            projection = self.network.project(lane, float(at[0]), float(at[2]))
            turn = (projection.heading - heading + math.pi) % (2.0 * math.pi) - math.pi
            if projection.distance <= within and abs(turn) < math.radians(turn_deg):
                return i
        return None

    def _through_taught_crossings(self, line: np.ndarray) -> np.ndarray:
        """`line` with each taught crossing's own line in place of the mapped
        one, joined on over `CROSSING_BLEND_M` at either end."""
        for _first, _last, taught in reversed(self.taught_crossings()):
            i0 = int(np.argmin(np.hypot(*(line[:, [0, 2]] - taught[0, [0, 2]]).T)))
            rest = np.hypot(*(line[i0:, [0, 2]] - taught[-1, [0, 2]]).T)
            i1 = i0 + int(np.argmin(rest))
            if i1 <= i0:
                continue
            line = np.vstack([line[:i0], taught, line[i1 + 1 :]])
            line = _blend_joins(line, i0, i0 + len(taught) - 1)
        return line

    def _customs_spread(self) -> dict[int, np.ndarray]:
        """The points of each lane in a customs stretch, moved `CUSTOMS_SPREAD_M`
        further from its road's centre line, easing in and out over
        `CUSTOMS_SPREAD_RAMP_M` at the stretch's ends.

        In the Italian-Swiss customs the booths stand on islands on the road's
        centre line, wider than the map's lanes -- 2.3 m either side of it --
        leave room for; in the middle of the left lane as mapped the truck's
        right side was over the island."""
        moved: dict[int, np.ndarray] = {}
        taught = {(first, last) for first, last, _ in self.taught_crossings()}
        for first, last in self.quiet_stretches():
            if (first, last) in taught:
                continue
            at = 0.0
            total = sum(
                float(self.network.lane_length[self.lanes[i]])
                for i in range(first, last + 1)
                if i == first or self.steps[i - 1] == ONWARD
            )
            for i in range(first, last + 1):
                lane = self.lanes[i]
                points = np.asarray(self.network.lane_points(lane), dtype=float)
                offset = float(self.network.lane_offset[lane])
                if i > first and self.steps[i - 1] != ONWARD:
                    at -= float(self.network.lane_length[self.lanes[i - 1]])
                if offset == 0.0 or len(points) < 2:
                    at += float(self.network.lane_length[lane])
                    continue
                step = np.hypot(*np.diff(points[:, [0, 2]], axis=0).T)
                s = at + np.concatenate([[0.0], np.cumsum(step)])
                ease = np.clip(np.minimum(s, total - s) / CUSTOMS_SPREAD_RAMP_M, 0.0, 1.0)
                heading = np.arctan2(np.gradient(points[:, 0]), np.gradient(points[:, 2]))
                # Left of travel is (cos c, -sin c); a positive offset is left.
                side = math.copysign(1.0, offset) * CUSTOMS_SPREAD_M * ease
                moved_points = points.copy()
                moved_points[:, 0] += side * np.cos(heading)
                moved_points[:, 2] -= side * np.sin(heading)
                moved[i] = moved_points
                at += float(self.network.lane_length[lane])
        return moved

    def quiet_stretches(self) -> list[tuple[int, int]]:
        """Runs of lanes on roads without traffic, between roads with it, as
        (first, last) indices into `lanes`, that the route passes through
        (`QUIET_STRAIGHT_RATIO`, `QUIET_MAX_TURN_DEG`): a border's customs
        lanes, driven slowly. One the route starts or ends in is a yard, not a
        way through, and is not counted.

        Driving them straight through instead, since the lanes between the
        booths are not in the map, the truck hit the plaza's furniture 319 m
        on; on the mapped customs road, a waiting truck and a booth."""
        stretches: list[tuple[int, int]] = []
        first: int | None = None
        for index, lane in enumerate(self.lanes):
            quiet = self.network.lane_source[lane] == 0 and not self.network.lane_ai[lane]
            prefab = self.network.lane_source[lane] == 1
            if quiet and first is None:
                first = index
            elif not quiet and not prefab and first is not None:
                last = index - 1
                while last > first and self.network.lane_source[self.lanes[last]] == 1:
                    last -= 1
                stretches.append((first, last))
                first = None
        kept = []
        for first, last in stretches:
            if first == 0:
                continue  # the route starts in it: a yard
            entry = np.asarray(self.network.lane_points(self.lanes[first]), float)
            leave = np.asarray(self.network.lane_points(self.lanes[last]), float)
            # A lane changed into runs alongside the one before; as in
            # `length`, only lanes driven onward add to the distance.
            mapped = sum(
                float(self.network.lane_length[self.lanes[i]])
                for i in range(first, last + 1)
                if i == first or self.steps[i - 1] == ONWARD
            )
            crow = float(np.hypot(*(leave[-1, [0, 2]] - entry[0, [0, 2]])))
            turn = _wrap(
                math.atan2(leave[-1, 0] - leave[-2, 0], leave[-1, 2] - leave[-2, 2])
                - math.atan2(entry[1, 0] - entry[0, 0], entry[1, 2] - entry[0, 2])
            )
            if (
                crow > 0
                and mapped / crow <= QUIET_STRAIGHT_RATIO
                and abs(turn) <= math.radians(QUIET_MAX_TURN_DEG)
            ):
                kept.append((first, last))
        return kept


def save(route: Route, file: str | Path, **metadata: object) -> None:
    """Write a route for `tools/follow.py`: its line, its lanes, and why.

    The line is what gets driven. The lanes and steps are kept so a run can be
    traced back to the network it was planned on.
    """
    lines = route.stop_lines()
    np.savez(
        file,
        format=np.array(ROUTE_FORMAT),
        points=route.points(),
        stop_lines=np.array([line.position for line in lines], dtype=float).reshape(-1, 3),
        stop_line_signals=np.array(
            json.dumps(
                [
                    {
                        "kind": line.signal.kind,
                        "profile": line.signal.profile,
                        "sleep": list(line.signal.sleep) if line.signal.sleep else None,
                    }
                    for line in lines
                ]
            )
        ),
        lanes=np.asarray(route.lanes, dtype=np.int64),
        steps=np.asarray(route.steps, dtype="U8"),
        metadata=np.array(json.dumps(metadata)),
    )


def load_points(file: str | Path) -> np.ndarray:
    """The `(n, 3)` line of a saved route.

    Raises:
        RouteError: not a saved route.
    """
    with np.load(file, allow_pickle=False) as data:
        if "format" not in data or str(data["format"]) != ROUTE_FORMAT:
            raise RouteError(f"{file} is not a saved route")
        return np.asarray(data["points"], dtype=float)


def load_stop_lines(file: str | Path) -> list[StopLine]:
    """The traffic lights of a saved route; none for a route saved before
    routes carried them.

    Raises:
        RouteError: not a saved route.
    """
    with np.load(file, allow_pickle=False) as data:
        if "format" not in data or str(data["format"]) != ROUTE_FORMAT:
            raise RouteError(f"{file} is not a saved route")
        if "stop_lines" not in data:
            return []
        positions = np.asarray(data["stop_lines"], dtype=float).reshape(-1, 3)
        signals = json.loads(str(data["stop_line_signals"]))
    return [
        StopLine(
            position,
            Signal(body["kind"], body["profile"], tuple(body["sleep"]) if body["sleep"] else None),
        )
        for position, body in zip(positions, signals, strict=True)
    ]


def plan(
    network: RoadNetwork,
    start: tuple[float, float, float],
    through: Sequence[tuple[float, float]],
) -> Route:
    """A route from a pose, through each position in turn, ending at the last.

    Args:
        start: `(x, z, course)` -- where the vehicle is and which way it is
            going, course in the `atan2(dx, dz)` frame.
        through: positions to pass, the last being the destination.

    Raises:
        RouteError: the start or a waypoint is off the network, or a leg has
            no route -- a destination beyond a road that ends, for instance.
    """
    if not through:
        raise RouteError("nowhere to go")

    match = LaneMatcher(network).update(*start)
    if match is None:
        raise RouteError(f"start {start[:2]} is not on the network")

    lanes = [match.lane]
    steps: list[str] = []
    for x, z in through:
        goals = [int(lane) for lane in network.lanes_near(x, z, WAYPOINT_RADIUS_M)]
        if not goals:
            raise RouteError(f"waypoint ({x:.0f}, {z:.0f}) is not on the network")
        leg = _shortest(network, lanes[-1], goals, (x, z))
        if leg is None:
            raise RouteError(f"nothing leads from lane {lanes[-1]} to ({x:.0f}, {z:.0f})")
        leg_lanes, leg_steps = leg
        lanes.extend(leg_lanes[1:])
        steps.extend(leg_steps)
    return Route(network, tuple(lanes), tuple(steps))


def plan_to_depot(
    network: RoadNetwork,
    start: tuple[float, float, float],
    depot: CompanyPlace,
    through: Sequence[tuple[float, float]] = (),
) -> Route:
    """A route from a pose to the gate of a company's yard.

    To the gate, not into the yard: a yard's lanes are mostly islands -- the
    paths to its parking spots are not joined to the road network -- and
    getting a trailer into a spot is a different job. A yard has two or more
    gates; the one chosen is the one with the shortest route.

    Raises:
        RouteError: none of the gates can be reached.
    """
    routes = []
    for gate in depot.gates:
        try:
            routes.append(plan(network, start, [*through, _onto_road(network, gate)]))
        except RouteError:
            continue
    if not routes:
        raise RouteError(f"no gate of {depot.company} in {depot.city} can be reached")
    return min(routes, key=lambda route: route.length)


def reroute_from(
    network: RoadNetwork,
    lanes: Sequence[int],
    steps: Sequence[str],
    depot: CompanyPlace,
    avoid: frozenset[int] = frozenset(),
) -> Route:
    """A route that starts with `lanes` as given, then takes the shortest way
    from the last of them to the depot's nearest gate.

    For turning where the game's GPS turns (core.planning.gps): the lanes up
    to and through the fork are fixed, and only what follows is planned.

    Raises:
        RouteError: no gate can be reached from there.
    """
    best: Route | None = None
    for gate in depot.gates:
        x, z = _onto_road(network, gate)
        goals = [int(lane) for lane in network.lanes_near(x, z, WAYPOINT_RADIUS_M)]
        if not goals:
            continue
        goals = [goal for goal in goals if goal not in avoid]
        if not goals:
            continue
        leg = _shortest(network, int(lanes[-1]), goals, (x, z), avoid)
        if leg is None:
            continue
        leg_lanes, leg_steps = leg
        route = Route(network, (*lanes, *leg_lanes[1:]), (*steps, *leg_steps))
        if best is None or route.length < best.length:
            best = route
    if best is None:
        raise RouteError(f"no gate of {depot.company} in {depot.city} can be reached that way")
    return best


REJOIN_SEARCH_M: Final = 5000.0
"""How far (in cost) from a fork a way back onto the planned route is looked for."""


def rejoin(
    route: Route,
    lanes: Sequence[int],
    steps: Sequence[str],
    after: int,
    avoid: frozenset[int] = frozenset(),
) -> Route | None:
    """`lanes` as given, then the cheapest way back onto `route` beyond its
    lane `after`, and the rest of `route` from there; None if no way back
    is found within `REJOIN_SEARCH_M`.

    For turning where the game's GPS turns: planning afresh to a depot 80 km
    off took 4 s, and at a junction the new route came after the truck had
    gone the planned way. The GPS's way nearly always meets the planned
    route again within a few junctions, which a search this small finds in
    milliseconds; forks further on are the GPS's to correct as they come.
    """
    network = route.network
    # What the planned route still costs from the start of each of its lanes.
    ahead: dict[int, tuple[int, float]] = {}
    remaining = 0.0
    for index in range(len(route.lanes) - 1, after, -1):
        lane = route.lanes[index]
        if index == len(route.lanes) - 1 or route.steps[index] == ONWARD:
            remaining += float(network.lane_length[lane]) * _cost_factor(network, lane)
        else:
            remaining += _change_cost(network, lane)
        ahead[lane] = (index, remaining)  # the earliest, if the route comes by twice
    # The way the GPS did not take, and the lanes of it in `avoid`: from the
    # lane beside an exit lane, changing into the exit lane's next lane was
    # the cheapest way back onto the plan, and the route went back and forth
    # between the two lanes every second.
    refused = {route.lanes[after + 1], *avoid}
    for lane in refused:
        ahead.pop(lane, None)

    start = int(lanes[-1])
    best_cost = {start: 0.0}
    came: dict[int, tuple[int, str]] = {}
    queue: list[tuple[float, int]] = [(0.0, start)]
    settled: set[int] = set()
    chosen: tuple[float, int] | None = None
    while queue:
        cost, lane = heapq.heappop(queue)
        if lane in settled:
            continue
        settled.add(lane)
        if cost > REJOIN_SEARCH_M or (chosen is not None and cost >= chosen[0]):
            break
        if lane in ahead and lane != start:
            score = cost + ahead[lane][1]
            if chosen is None or score < chosen[0]:
                chosen = (score, lane)
            continue
        for following, step_cost, kind in _options(network, lane):
            if following in refused:
                # Not back through it: from the lane beside an exit lane,
                # changing back into it is the cheapest way onto the exit.
                continue
            total = cost + step_cost
            if total < best_cost.get(following, math.inf):
                best_cost[following] = total
                came[following] = (lane, kind)
                heapq.heappush(queue, (total, following))
    if chosen is None:
        return None

    way, kinds = [chosen[1]], []
    while way[-1] != start:
        previous, kind = came[way[-1]]
        way.append(previous)
        kinds.append(kind)
    way, kinds = way[::-1], kinds[::-1]
    index = ahead[chosen[1]][0]
    return Route(
        network,
        (*lanes, *way[1:], *route.lanes[index + 1 :]),
        (*steps, *kinds, *route.steps[index:]),
    )


def _onto_road(network: RoadNetwork, gate: tuple[float, float]) -> tuple[float, float]:
    """The gate, or where the road nearest it is if it is beyond `WAYPOINT_RADIUS_M`."""
    near = network.nearest(gate[0], gate[1], radius=WAYPOINT_RADIUS_M)
    if near is not None:
        return gate
    near = network.nearest(gate[0], gate[1], radius=GATE_RADIUS_M)
    if near is None:
        return gate
    lane = near[0]
    projection = network.project(lane, gate[0], gate[1])
    points = np.asarray(network.lane_points(lane), dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points[:, [0, 2]], axis=0).T))])
    at = float(np.clip(projection.s, 0.0, s[-1]))
    return (float(np.interp(at, s, points[:, 0])), float(np.interp(at, s, points[:, 2])))


def plan_ahead(
    network: RoadNetwork,
    start: tuple[float, float, float],
    distance: float,
    *,
    stop_before_lights: bool = True,
) -> Route:
    """The road ahead for `distance` metres, taking the straightest way on.

    No destination: at each junction, the lane that turns least. Made for
    driving a planned route for the first time, where the question is whether
    the stack can follow the map, not where it goes.

    With `stop_before_lights` the route ends before the first lane a traffic
    light governs, since nothing in the stack can see one yet. It also ends
    where the road does.

    Raises:
        RouteError: the start is off the network.
    """
    match = LaneMatcher(network).update(*start)
    if match is None:
        raise RouteError(f"start {start[:2]} is not on the network")

    lanes = [match.lane]
    covered = float(network.lane_length[match.lane]) - match.s
    while covered < distance:
        options = [int(f) for f in network.next_lanes(lanes[-1])]
        if stop_before_lights:
            options = [f for f in options if int(network.lane_semaphore[f]) < 0]
        if not options:
            break
        heading = _heading_at_end(network, lanes[-1])
        following = min(options, key=lambda f: abs(_wrap(_heading_at_start(network, f) - heading)))
        lanes.append(following)
        covered += float(network.lane_length[following])
    return Route(network, tuple(lanes), (ONWARD,) * (len(lanes) - 1))


def _heading_at_end(network: RoadNetwork, lane: int) -> float:
    points = network.lane_points(lane)
    (x0, _, z0), (x1, _, z1) = points[-2], points[-1]
    return math.atan2(float(x1 - x0), float(z1 - z0))


def _heading_at_start(network: RoadNetwork, lane: int) -> float:
    points = network.lane_points(lane)
    (x0, _, z0), (x1, _, z1) = points[0], points[1]
    return math.atan2(float(x1 - x0), float(z1 - z0))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _shortest(
    network: RoadNetwork,
    start: int,
    goals: list[int],
    target: tuple[float, float],
    avoid: frozenset[int] = frozenset(),
) -> tuple[list[int], list[str]] | None:
    """The best way from `start` to a lane passing near `target`.

    Not simply to the nearest lane: inside a junction the lanes overlap, and
    the nearest one to a waypoint can belong to a branch going the other way.
    Planning the taught loop that way went round it twice -- 1004 m for a leg
    of 70 -- to arrive on a lane 1.1 m from the waypoint when one 1.5 m away
    was straight ahead. So every lane near the waypoint is a candidate, scored
    as the cost of reaching it plus `MISS_WEIGHT` metres per metre it misses the
    waypoint by.

    One search does it: an A* outwards from `start`, towards the waypoint, that
    scores each candidate as it is reached and stops once nothing still queued
    could beat the best score. Searching for each candidate separately took
    5.5 s for the taught loop, most of it proving unreachable candidates
    unreachable; this takes 21 ms.
    """
    miss = {goal: network.project(goal, *target).distance for goal in goals}
    # A lower bound on what is left to pay from the end of a lane: the goals
    # are within the waypoint radius and at most `reach` long, so no lane that
    # ends further than that from the waypoint can be nearer a goal than this.
    reach = WAYPOINT_RADIUS_M + max(float(network.lane_length[g]) for g in goals)

    def remaining(lane: int) -> float:
        if lane in miss:
            return 0.0
        end = network.points[network.lane_start[lane + 1] - 1]
        return max(0.0, math.hypot(float(end[0]) - target[0], float(end[2]) - target[1]) - reach)

    best_cost = {start: 0.0}
    came: dict[int, tuple[int, str]] = {}
    queue: list[tuple[float, float, int]] = [(remaining(start), 0.0, start)]
    chosen: tuple[float, int] | None = None
    settled: set[int] = set()
    while queue:
        bound, cost, lane = heapq.heappop(queue)
        if lane in settled:
            continue
        settled.add(lane)
        if chosen is not None and bound >= chosen[0]:
            break
        if lane in miss:
            score = cost + MISS_WEIGHT * miss[lane]
            if chosen is None or score < chosen[0]:
                chosen = (score, lane)
        for following, step_cost, kind in _options(network, lane):
            if following in avoid:
                continue
            total = cost + step_cost
            if total < best_cost.get(following, math.inf):
                best_cost[following] = total
                came[following] = (lane, kind)
                heapq.heappush(queue, (total + remaining(following), total, following))
    if chosen is None:
        return None

    path, kinds = [chosen[1]], []
    while path[-1] != start:
        previous, kind = came[path[-1]]
        path.append(previous)
        kinds.append(kind)
    return path[::-1], kinds[::-1]


def _options(network: RoadNetwork, lane: int) -> list[tuple[int, float, str]]:
    onward = [
        (
            int(f),
            float(network.lane_length[f]) * _cost_factor(network, int(f)),
            ONWARD,
        )
        for f in network.next_lanes(lane)
    ]
    across = [(int(f), _change_cost(network, lane), CHANGE) for f in network.lanes_alongside(lane)]
    by_sea = [(link.land, link.cost, FERRY) for link in network.ferry_links.get(lane, ())]
    return onward + across + by_sea


def _change_cost(network: RoadNetwork, lane: int) -> float:
    """What moving across from `lane` into the lane beside it costs."""
    short = float(network.lane_length[lane]) < SHORT_CHANGE_M
    return CHANGE_COST_M + (SHORT_CHANGE_PENALTY_M if short else 0.0)


def change_begins(lane_length: float, length: float = LANE_CHANGE_LENGTH_M) -> float:
    """Metres into a lane of `lane_length` where a move across it begins."""
    span = min(length, lane_length)
    return max(0.0, lane_length - LANE_CHANGE_FINISH_M - span)


def _cost_factor(network: RoadNetwork, lane: int) -> float:
    """What a metre of this lane costs against a metre of an ordinary one."""
    if network.lane_hidden[lane]:
        return HIDDEN_COST
    if network.lane_source[lane] == 0 and not network.lane_ai[lane]:
        return NO_TRAFFIC_COST
    if _outermost(network, lane):
        return OUTER_LANE_COST
    return 1.0


#: A metre of the outermost lane of a road with more than one, on the side
#: its exits are -- the right, or the left where traffic keeps left -- costs
#: this much: a route keeps to the middle lane of three, or the other lane of
#: two, unless it is about to leave. In the outer lane the way on is decided
#: where it ends, and an exit the GPS did not take was taken before the
#: minimap could say so; from another lane, leaving takes a lane change,
#: which the GPS is asked about (core.planning.gps) before it begins.
OUTER_LANE_COST: Final = 1.25


def _outermost(network: RoadNetwork, lane: int) -> bool:
    """Whether a road lane has lanes beside it, and none on its exit side."""
    found = _OUTERMOST.get(id(network))
    if found is None or found[0]() is not network:
        # An id is reused once its network is gone: a new network at the old
        # one's address read the old one's answers, and a test of a road where
        # traffic keeps left decided a right-hand road's lanes (Windows CI).
        found = (weakref.ref(network), {})
        _OUTERMOST[id(network)] = found
    known = found[1].get(lane)
    if known is not None:
        return known
    beside = network.lanes_alongside(lane)
    outer = False
    if network.lane_source[lane] == 0 and len(beside):
        left_hand = network.lane_left_hand is not None and bool(network.lane_left_hand[lane])
        points = network.lane_points(lane)
        heading = math.atan2(float(points[1][0] - points[0][0]), float(points[1][2] - points[0][2]))
        outer = True
        for other in beside:
            start = network.lane_points(int(other))[0]
            dx, dz = float(start[0] - points[0][0]), float(start[2] - points[0][2])
            right = -(dx * math.cos(heading) - dz * math.sin(heading))
            if (right > 0.0) != left_hand:
                outer = False  # a lane on the exit side of this one
                break
    found[1][lane] = outer
    return outer


_OUTERMOST: dict[int, tuple[weakref.ref, dict[int, bool]]] = {}
"""Per network, by id checked by a weak reference (as nav_field's cache):
whether each lane asked about is outermost."""


@dataclass(frozen=True)
class Crossing:
    """A driver's line through a border's customs, where the map's lanes are
    not the way through (tools/teach_crossing.py); or, a `PASSAGE`, through
    any place the map's lanes lead over what is not road -- out of a
    roundabout in Bern, the lane onto the main road ran over an island."""

    entry: np.ndarray
    """`(2,)` world (x, z) where the customs stretch begins, as mapped."""

    exit: np.ndarray
    points: np.ndarray
    """`(n, 3)` the driven line, from near `entry` to near `exit`."""

    kind: str = "customs"
    """`CUSTOMS`, matched to a route's customs stretch by its ends as mapped;
    or `PASSAGE`, to any route that passes its own ends going its way."""


CUSTOMS: Final = "customs"
PASSAGE: Final = "passage"


CROSSINGS_FILE: Final = Path(__file__).resolve().parents[2] / "configs" / "crossings.json"


@functools.cache
def load_crossings(file: Path = CROSSINGS_FILE) -> tuple[Crossing, ...]:
    """The taught crossings; none if there is no file."""
    if not file.is_file():
        return ()
    return tuple(
        Crossing(
            np.asarray(body["entry"], float),
            np.asarray(body["exit"], float),
            np.asarray(body["points"], float),
            body.get("kind", CUSTOMS),
        )
        for body in json.loads(file.read_text(encoding="utf-8"))["crossings"]
    )


NO_LIGHTS_FILE: Final = Path(__file__).resolve().parents[2] / "configs" / "no_lights.json"
NO_LIGHT_WITHIN_M: Final = 5.0
"""A stop line this near a place in `NO_LIGHTS_FILE` is not a light's."""


@functools.cache
def load_no_lights(file: Path = NO_LIGHTS_FILE) -> np.ndarray:
    """`(n, 2)` world (x, z) of stop lines the map gives a light the road
    does not have, as a driver has said; none if there is no file. At the
    south end of the bridge in Geneva one stands 52 m after a real light,
    too close behind it to be found dark (TrafficLights), and the truck
    stopped and waited at it every time."""
    if not file.is_file():
        return np.zeros((0, 2))
    lights = json.loads(file.read_text(encoding="utf-8"))["lights"]
    return np.asarray([light["position"] for light in lights], float).reshape(-1, 2)


def _blend_joins(line: np.ndarray, start: int, end: int) -> np.ndarray:
    """Smooth the two joins of a spliced-in piece, `line[start:end + 1]`, over
    `CROSSING_BLEND_M` either side, so the controller is not handed a step."""
    out = line.copy()
    step = np.hypot(*np.diff(line[:, [0, 2]], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(step)])
    for join in (start, end):
        near = np.abs(s - s[join]) <= CROSSING_BLEND_M
        index = np.flatnonzero(near)
        if len(index) < 3:
            continue
        weights = 1.0 - np.abs(s[index] - s[join]) / CROSSING_BLEND_M
        window = max(3, len(index) // 2)
        kernel = np.ones(window) / window
        for axis in (0, 2):
            smooth = np.convolve(
                np.pad(line[index, axis], window, mode="edge"), kernel, mode="same"
            )[window:-window]
            out[index, axis] = (1.0 - weights) * line[index, axis] + weights * smooth
    return out


def _ease_steps(line: np.ndarray, steps: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Where a lane starts off to one side of where the last ended -- the next
    road piece's look puts its lanes elsewhere -- the line from there on is
    moved back by the step, and eases onto its own over `JOIN_EASE_M`."""
    out = line.copy()
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line[:, [0, 2]], axis=0).T))])
    for index, gap in steps:
        t = np.clip((s[index:] - s[index]) / JOIN_EASE_M, 0.0, 1.0)
        weight = 1.0 - t * t * (3.0 - 2.0 * t)
        out[index:] -= weight[:, None] * gap[None, :]
    return out


def _ease_across(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    """Blend from line `a` onto line `b`, which runs alongside it.

    Both are the same road's lanes, sampled at the same arc lengths, so point i
    of one sits beside point i of the other. The blend is a smoothstep, so the
    line leaves `a` and meets `b` tangentially -- no kink for the controller.
    """
    if len(a) != len(b):
        b = _resample_like(b, a)
    step = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 2]))
    s = np.concatenate([[0.0], np.cumsum(step)])
    span = max(min(length, float(s[-1])), 1e-6)
    t = np.clip((s - change_begins(float(s[-1]), length)) / span, 0.0, 1.0)
    weight = (t * t * (3.0 - 2.0 * t))[:, None]
    return a * (1.0 - weight) + b * weight


def _resample_like(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """`b` at the fractional arc positions of `a`'s points."""

    def fractions(points: np.ndarray) -> np.ndarray:
        s = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 2])))]
        )
        return s / max(float(s[-1]), 1e-9)

    fa, fb = fractions(a), fractions(b)
    return np.column_stack([np.interp(fa, fb, b[:, k]) for k in range(3)])


def _heading(frm: np.ndarray, to: np.ndarray) -> float:
    """Heading from `frm` to `to`, `(3,)` world points, as lanes give it."""
    return math.atan2(float(to[0] - frm[0]), float(to[2] - frm[2]))
