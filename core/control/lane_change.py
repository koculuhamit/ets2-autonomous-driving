"""Changing lanes only into a lane that is clear.

A planned route changes lanes where the map says it must -- to take an exit,
say -- and `Route.points` eases the line across over 120 m, finishing 100 m
before the end of the lane it leaves (`change_begins`). Driven blind, that
eased the truck onto a lorry it was overtaking on a Danish motorway: the
lorry had been in plain view of the camera for seconds, but nothing looked
at the lane being moved into.

So each lane change is a gate (`LaneChangeGate`). Approaching it, the lane
being moved into is watched; with something in it at the change's start --
ahead within a gap, or beside the truck -- the line is held in the lane the
truck is in, and the change waits. Once the lane is clear the change begins
from where the truck is, over its whole length, as long as it can still be
finished before the lanes part (`LaneChange.latest`). Past that, on a fast
road whose lane goes on, the change is missed: the truck drives on in its
lane and the game plans anew. Slow, or in a lane that ends, it slows to wait
at the latest start instead.

The line was not always held. Out of Duisburg on the A1 at night, the red
line took a fork's other branch 187 m before the lane change it needed had
to be done; the change's start was behind the truck, so the line was eased
across in 80 m, onto a car in that lane, and the gate, holding only the
speed, braked for 14 km/h on the motorway: the truck hit the car.

What the camera sees of the lane beside it ends a few metres ahead of the
truck. A vehicle last seen there, close, is remembered (`BESIDE_MEMORY_S`)
and carried along at its closing speed: one the truck is overtaking falls
back past the trailer and is let go; one that keeps pace is assumed to still
be there.

Nothing behind is seen: the mirrors are not read. They were, until 7
October: in Bern, at dusk and in traffic, they showed the truck's own
trailer, oncoming buses and a tram at its stop as traffic coming up, and
the truck went across and back, again and again. A change is held short of
its start while the camera sees the lane taken; the route changes lanes
only where it must (core.planning.lanes), so changes are few.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from core.localization.path import ReferencePath
from core.perception.camera import GroundCamera
from core.perception.tracking import Path as LanePath
from core.planning.route import (
    CHANGE,
    LANE_CHANGE_LENGTH_M,
    ONWARD,
    Route,
    change_begins,
)

__all__ = ["GateDecision", "LaneChange", "LaneChangeGate", "lane_changes"]

APPROACH_M: Final = 250.0
"""The lane beside is watched from this far before a change: stopping from
90 km/h at `DECELERATION` takes 208 m."""

HOLD_M: Final = 20.0
"""A change begun is still given up for a lane seen taken this far into it:
the ease across is a smoothstep over 120 m; 20 m in it has moved 0.3 m."""

LATEST_FINISH_M: Final = 30.0
"""A change waited for is finished at the latest this far before the lanes
part."""

APART_M: Final = 1.5
"""Beyond the lanes' own distance apart, how much more has them parted."""

ONWARD_M: Final = 3000.0
"""How far the lane the truck is in is followed on, for a change missed."""

MISS_ROOM_M: Final = 300.0
"""A change can be missed only where the lane the truck is in goes on this
far after the lanes part."""

STOP_BELOW_MPS: Final = 50.0 / 3.6
MISS_SLOW_MPS: Final = 60.0 / 3.6
"""Up to this speed the truck slows to wait for a lane; faster, a change it
can miss is slowed for, down to `MISS_SLOW_MPS` and no more, for the lane to
clear, and missed if it does not -- on the A1 the gate once braked for
14 km/h on the motorway. Not slowed for at all, the change into
the lane on the right on an English motorway junction was missed at 75 km/h
twice running, a car beside, and the truck was left in the lane into a
services."""

WAIT_AT_MOST_S: Final = 30.0
WAITING_BELOW_MPS: Final = 0.5
"""Standing this long for a taken lane, a change that can be missed
(`MISS_ROOM_M`) is given up and the truck goes on in its lane: the game plans
anew from wherever it gets to. At a light in Aberdeen a lorry stood through two greens in the lane
the route wanted, and the truck stood beside it at the green, for good."""

RETURN_M: Final = 60.0
"""A change given up for a lane seen taken eases back over this far."""

LATE_CHANGE_MIN_S: Final = 4.8
LATE_CHANGE_MIN_M: Final = 60.0
"""A change held past its latest start -- a route taken with its start
already behind the truck -- is still made, the lane clear, over what room is
left before the lanes part (less `LATEST_FINISH_M`), if crossing it takes no
less than this -- the whole 120 m at 90 km/h -- nor in fewer metres. On a
delivery to Amsterdam the red line took another way 7 s after the last, its
change into the lane on the right just behind the truck at 50 km/h; the lane
was empty and the change was missed on the spot, 50 km more to drive."""

STILL_IN_LANE_M: Final = 0.5
COMMITTED_M: Final = 1.0
"""A change the truck has moved this far across from the lane it is leaving
is gone on with, whatever is seen in the lane it moves into: at Malmö, half
way into the right lane for a turn, a car 14 m ahead in it turned the truck
back, into a mixer that had come up beside it in the lane it had left. A
vehicle ahead in the new lane is followed there."""
"""A truck this near the middle of the lane a change leaves has not begun it."""

PARTING_SEARCH_M: Final = 1500.0
"""How far past a change the lanes are followed to see where they part."""

DECELERATION: Final = 1.5
"""m/s^2 the truck slows at for an occupied lane."""

TARGET_HALF_WIDTH_M: Final = 1.8
"""Of the lane being moved into, either side of its middle: a vehicle any of
whose width is within this is in it (`_in_lane`)."""

FRONT_GAP_S: Final = 1.2
FRONT_GAP_MIN_M: Final = 15.0
"""A vehicle ahead in the other lane nearer than this many seconds at the
truck's speed (and never less than the minimum) is too close to move in
front of; further on, following takes over once the truck is behind it."""

BESIDE_WITHIN_M: Final = 35.0
"""A vehicle in the other lane this near is one to remember when it goes
out of view beside the truck."""

BESIDE_MEMORY_S: Final = 8.0
BEHIND_CLEAR_M: Final = 22.0
"""A remembered vehicle is clear once carried this far behind the eye: the
tractor and trailer are about 16.5 m, and a margin."""

SIDE_OF_VIEW_PX: Final = 250.0
"""A track with no distance -- its foot hidden by the dashboard, so close --
counts as beside the truck when its box's middle is this far off the view's
centre towards the other lane."""


@dataclass(frozen=True)
class LaneChange:
    start: float
    """Metres along the path where the line begins to ease across."""

    side: int
    """+1 into the lane on the right, -1 into the lane on the left."""

    target: np.ndarray
    """`(n, 2)` world (x, z) of the middle of the lane moved into."""

    span: float = LANE_CHANGE_LENGTH_M
    """Metres the ease across takes."""

    latest: float | None = None
    """Metres along the path where the change can begin at the latest and be
    finished before the lanes part; None for at its start."""

    leaving: ReferencePath | None = None
    """The middle of the lane left, and on from it straight on: the line
    held to while the change waits."""

    can_miss: bool = False
    """Whether the lane left goes on after the lanes part."""

    @property
    def last_start(self) -> float:
        return self.start if self.latest is None else max(self.start, self.latest)


def lane_changes(route: Route, path: ReferencePath) -> list[LaneChange]:
    """Each run of lane changes on `route`, where it begins on `path`."""
    network = route.network
    found: list[LaneChange] = []
    near = 0.0
    index = 0
    while index < len(route.steps):
        if route.steps[index] != CHANGE:
            index += 1
            continue
        first = index
        while index < len(route.steps) and route.steps[index] == CHANGE:
            index += 1
        # lanes[first] is left for lanes[index] (a run of changes ends on it).
        leaving = np.asarray(network.lane_points(route.lanes[first]), float)
        pieces = [np.asarray(network.lane_points(route.lanes[index]), float)]
        covered = float(network.lane_length[route.lanes[index]])
        onward = index
        while covered < APPROACH_M and onward < len(route.steps):
            if route.steps[onward] != ONWARD:
                break
            onward += 1
            pieces.append(np.asarray(network.lane_points(route.lanes[onward]), float))
            covered += float(network.lane_length[route.lanes[onward]])
        target = np.vstack(pieces)[:, [0, 2]]

        start = leaving[0, [0, 2]]
        sample = path.nearest(float(start[0]), float(start[1]), near=near, window=1e9)
        near = sample.s
        length = float(network.lane_length[route.lanes[first]])
        begins = sample.s + change_begins(length)
        heading = math.atan2(leaving[1, 0] - leaving[0, 0], leaving[1, 2] - leaving[0, 2])
        dx, dz = target[0] - start
        right = -(dx * math.cos(heading) - dz * math.sin(heading))
        span = min(LANE_CHANGE_LENGTH_M, length)
        line = ReferencePath(
            np.vstack([leaving[:, [0, 2]], _onward(network, route.lanes[first])]), spacing=1.0
        )
        gap = float(math.hypot(dx, dz))
        parted, goes_on = _parting(path, line, begins + span, gap)
        found.append(
            LaneChange(
                begins,
                1 if right > 0 else -1,
                target,
                span=span,
                latest=parted - LATEST_FINISH_M - span,
                leaving=line,
                can_miss=goes_on >= MISS_ROOM_M,
            )
        )
    return found


def _onward(network, lane: int) -> np.ndarray:
    """`(n, 2)` world (x, z) on from the end of `lane`, `ONWARD_M` of the
    lanes straight on from it (none where it ends). A road hidden from the
    game's map ends it: near Nuremberg the lane left at a motorway exit, a
    change into the exit lane missed, ran on into a hidden road, closed in
    the game by the barrier at the fork, and the truck hit the barrier."""
    hidden = getattr(network, "lane_hidden", None)
    pieces, covered = [], 0.0
    points = np.asarray(network.lane_points(lane), float)
    for _ in range(64):
        if covered >= ONWARD_M or len(points) < 2:
            break
        end = points[-1] - points[-2]
        heading = math.atan2(end[0], end[2])
        best, turn = None, math.inf
        for following in network.next_lanes(lane):
            theirs = np.asarray(network.lane_points(int(following)), float)
            if len(theirs) < 2:
                continue
            start = theirs[1] - theirs[0]
            off = abs(
                (math.atan2(start[0], start[2]) - heading + math.pi) % (2 * math.pi) - math.pi
            )
            if off < turn:
                best, turn = int(following), off
        if best is None or (hidden is not None and bool(hidden[best])):
            break
        lane, points = best, np.asarray(network.lane_points(best), float)
        pieces.append(points[1:, [0, 2]])
        covered += float(network.lane_length[lane])
    return np.vstack(pieces) if pieces else np.empty((0, 2))


def _parting(
    path: ReferencePath, line: ReferencePath, across: float, gap: float
) -> tuple[float, float]:
    """Where along `path`, once across at `across`, it parts from `line`:
    further from it than the lanes' `gap` and `APART_M`, or past its end;
    and how far `line` goes on from there."""
    near = None
    s = across
    while s < min(path.length, across + PARTING_SEARCH_M):
        point = path.sample_at(s)
        on = line.nearest(point.x, point.z, near=near, window=100.0)
        near = on.s
        apart = math.hypot(on.x - point.x, on.z - point.z)
        if apart > gap + APART_M or on.s >= line.length - 1.0:
            return s, line.length - on.s
        s += 2.0
    return s, line.length - (near or 0.0)


@dataclass(frozen=True)
class GateDecision:
    target_speed: float | None
    """A cap on the speed, m/s; None when the way across is clear."""

    reason: str

    path: ReferencePath | None = None
    """The line to drive from now on, where the gate has changed it."""

    missed: bool = False
    """The change has been given up: the truck drives on in its lane."""


@dataclass
class _Beside:
    seen_at: float
    distance: float
    closing: float


AHEAD, HELD, BEGUN, DONE, MISSED = "ahead", "held", "begun", "done", "missed"


class LaneChangeGate:
    """Holds each lane change while the lane is taken: the line in the lane
    the truck is in, and on a slow road the truck short of the change.
    `path` is the line the changes were found on (`lane_changes`)."""

    def __init__(
        self, changes: Sequence[LaneChange], camera: GroundCamera, path: ReferencePath
    ) -> None:
        self.changes = list(changes)
        self.camera = camera
        self.path = path
        self._beside: dict[int, _Beside] = {}
        self._state = [AHEAD] * len(self.changes)
        self._begun: dict[int, float] = {}
        self._standing_since: float | None = None
        """When the truck came to stand for a taken lane (`WAIT_AT_MOST_S`)."""
        self._mirror: list[float] = []
        self._mirror_track: list[tuple[float, float]] = []
        """(when, how near) of the nearest vehicle in the mirror on the side
        of the change at hand, seen `MIRROR_FAR` or nearer: how fast it comes
        up (`_coming_up`)."""
        """When a vehicle was last seen near in the mirror on the side of the
        change at hand (`MIRROR_NEAR`)."""
        self._fast: set[int] = set()
        """Changes it can miss, first found taken faster than `STOP_BELOW_MPS`:
        slowed for, never stopped for, however slow the truck has become."""
        self._spans: dict[int, float] = {}
        """Changes begun late (`LATE_CHANGE_MIN_S`): the metres they cross in."""

    def adopt(self, progress: float, x: float, z: float) -> ReferencePath | None:
        """The line to drive on a route taken with the truck at `(x, z)`,
        `progress` along it, whose change has begun behind the truck while
        the truck is still in the lane left: held there, to begin from where
        the truck is once clear. None where there is none such.

        A change wholly behind the truck is done: a route taken anew keeps
        the start of the one before it, and its changes behind the truck are
        no business of the gate's."""
        for index, change in enumerate(self.changes):
            if self._passed(index, progress):
                continue
            if not change.start < progress < change.start + change.span:
                continue
            if change.leaving is not None:
                on = change.leaving.nearest(x, z)
                if math.hypot(on.x - x, on.z - z) < STILL_IN_LANE_M:
                    self._state[index] = HELD
                    return self._held(index)
            self._state[index], self._begun[index] = BEGUN, change.start
        return None

    def signalled(self) -> list[LaneChange]:
        """The changes as they will be made, for the indicators: those begun
        from where they began, none missed."""
        return [
            replace(change, start=self._begun.get(index, change.start))
            for index, change in enumerate(self.changes)
            if self._state[index] != MISSED
        ]

    def watching(self, progress: float) -> int:
        """The side of the lane change at hand -- +1 right, -1 left -- whose
        mirror is to be looked in (`MIRROR_NEAR`); 0 for none."""
        for index, change in enumerate(self.changes):
            state = self._state[index]
            if state in (DONE, MISSED) or self._passed(index, progress):
                continue
            if progress >= change.start - APPROACH_M:
                return change.side
            return 0
        return 0

    def _passed(self, index: int, progress: float) -> bool:
        """Whether change `index`, not begun, is wholly behind the truck at
        `progress`: then done. On a delivery to Amsterdam a route taken anew
        kept the old one's start, a change into the lane on the left 1.5 km
        behind the truck was found taken, and holding it put the truck on
        that lane's line: 669 m from the path at 80 km/h, and the drive
        stopped."""
        change = self.changes[index]
        if self._state[index] == AHEAD and progress >= change.start + change.span:
            self._state[index] = DONE
            return True
        return False

    def _active(self, progress: float) -> int | None:
        for index, change in enumerate(self.changes):
            state = self._state[index]
            if state in (DONE, MISSED) or self._passed(index, progress):
                continue
            if state == BEGUN and progress >= self._begun[index] + self._span(index):
                self._state[index] = DONE
                continue
            if progress >= change.start - APPROACH_M:
                return index
            return None
        return None

    def step(
        self,
        *,
        progress: float,
        speed: float,
        pose: tuple[float, float, float],
        tracks: Sequence,
        unseen: Sequence[tuple[int, float]] = (),
        now: float,
        mirror: tuple[int, float] | None = None,
        mirror_far: tuple[int, float] | None = None,
    ) -> GateDecision:
        """The cap for the next lane change, and the line where it changes.

        `tracks` are the perception report's measured tracks (distance,
        closing, lateral from the eye, width); `unseen` is (id, box middle's
        column in pixels) for confirmed tracks with no distance.
        """
        index = self._active(progress)
        if index is None:
            self._beside.clear()
            self._mirror.clear()
            self._mirror_track.clear()
            return GateDecision(None, "")
        change = self.changes[index]
        state = self._state[index]
        target = self._target_lane(change, pose)
        why = self._taken(change, target, speed, tracks, unseen, now)
        if mirror is not None and mirror[0] == change.side and mirror[1] >= MIRROR_NEAR:
            self._mirror.append(now)
        if mirror_far is not None and mirror_far[0] == change.side and mirror_far[1] >= MIRROR_FAR:
            self._mirror_track.append((now, mirror_far[1]))
        self._mirror = [t for t in self._mirror if now - t <= MIRROR_WITHIN_S]
        self._mirror_track = [(t, n) for t, n in self._mirror_track if now - t <= MIRROR_TRACK_S]
        begun_late = state == BEGUN and progress >= self._begun[index] + HOLD_M
        if begun_late:
            why = None  # past the hold point only a vehicle coming up fast turns it back
        if (
            why is None
            and not begun_late
            and speed >= MIRROR_FROM_MPS
            and len(self._mirror) >= MIRROR_SEEN
            and not (state == BEGUN and self._committed(change, pose))
        ):
            why = "a vehicle coming up in it, in the mirror"
        fast_behind = (
            speed >= MIRROR_FROM_MPS
            and self._coming_up(now)
            and not (state == BEGUN and self._across(change, pose) >= MIRROR_BACK_M)
        )
        if why is None and fast_behind:
            why = "a vehicle coming up fast in it, in the mirror"
        side = "right" if change.side > 0 else "left"
        if why is not None and change.can_miss and speed > STOP_BELOW_MPS:
            self._fast.add(index)
        waits = not change.can_miss or (speed <= STOP_BELOW_MPS and index not in self._fast)
        if why is None or abs(speed) >= WAITING_BELOW_MPS or state == BEGUN or not change.can_miss:
            self._standing_since = None
        elif self._standing_since is None:
            self._standing_since = now
        elif now - self._standing_since >= WAIT_AT_MOST_S:
            self._standing_since = None
            self._state[index] = MISSED
            return GateDecision(
                None,
                f"lane on the {side} never clear: change missed",
                path=self._held(index) if state == AHEAD else None,
                missed=True,
            )
        if state == AHEAD and progress < change.start:
            if why is None:
                return GateDecision(None, "")
            cap = self._cap(change, progress) if waits else self._slow(change, progress)
            return GateDecision(cap, f"lane on the {side} taken: {why}")
        if state == AHEAD:
            if why is None:
                self._state[index], self._begun[index] = BEGUN, change.start
                return GateDecision(None, "")
            self._state[index] = HELD
            return self._holding(change, progress, side, why, waits, self._held(index))
        if state == BEGUN:
            if why is None or (self._committed(change, pose) and not fast_behind):
                return GateDecision(None, "")
            self._state[index] = HELD
            back = self._held(index, returning_at=progress)
            del self._begun[index]
            return self._holding(change, progress, side, why, waits, back)
        # Held.
        if why is None and (progress <= change.last_start or waits):
            self._state[index], self._begun[index] = BEGUN, progress
            return GateDecision(None, "", path=self._begun_at(index, progress))
        if progress > change.last_start and not waits:
            span = self._late_span(change, progress, speed) if why is None else None
            if span is not None:
                self._state[index], self._begun[index] = BEGUN, progress
                self._spans[index] = span
                return GateDecision(None, "", path=self._begun_at(index, progress))
            self._state[index] = MISSED
            reason = (
                f"lane on the {side} taken past the change's latest start ({why}): change missed"
                if why is not None
                else f"lane on the {side} clear, but too late to cross before the lanes part: "
                "change missed"
            )
            return GateDecision(None, reason, missed=True)
        return self._holding(change, progress, side, why or "", waits, None)

    @staticmethod
    def _late_span(change: LaneChange, progress: float, speed: float) -> float | None:
        """The metres a change past its latest start is crossed in from
        `progress`, finished `LATEST_FINISH_M` before the lanes part; None
        where that is quicker than `LATE_CHANGE_MIN_S` at `speed`, or
        shorter than `LATE_CHANGE_MIN_M`."""
        if change.latest is None:
            return None
        parted = change.latest + LATEST_FINISH_M + change.span
        room = parted - LATEST_FINISH_M - progress
        if room < max(LATE_CHANGE_MIN_M, speed * LATE_CHANGE_MIN_S):
            return None
        return min(room, change.span)

    def _span(self, index: int) -> float:
        return self._spans.get(index, self.changes[index].span)

    @classmethod
    def _committed(cls, change: LaneChange, pose: tuple[float, float, float]) -> bool:
        """Whether the truck has moved `COMMITTED_M` across from the lane left."""
        return cls._across(change, pose) >= COMMITTED_M

    @staticmethod
    def _across(change: LaneChange, pose: tuple[float, float, float]) -> float:
        """Metres the truck has moved across from the lane left; 0 for none
        known."""
        if change.leaving is None:
            return 0.0
        x, z = pose[0], pose[1]
        on = change.leaving.nearest(x, z)
        return math.hypot(on.x - x, on.z - z)

    def _coming_up(self, now: float) -> bool:
        """Whether the vehicle in the mirror comes up fast: its box grows so
        that, at that rate, it would be on the truck within
        `MIRROR_CLOSING_S` -- its size over how fast that grows -- from
        `MIRROR_CLOSING_SEEN` sightings or more over `MIRROR_CLOSING_OVER_S`,
        growing steadily (`MIRROR_SHRINK`, `MIRROR_JUMP`)."""
        track = self._mirror_track
        if len(track) < MIRROR_CLOSING_SEEN or track[-1][0] - track[0][0] < MIRROR_CLOSING_OVER_S:
            return False
        times = np.array([t for t, _ in track]) - now
        sizes = np.array([n for _, n in track])
        if np.any(np.diff(sizes) < -MIRROR_SHRINK * sizes[:-1]):
            return False
        if sizes[-1] > MIRROR_JUMP * sizes[-2]:
            return False
        rate, size_now = np.polyfit(times, sizes, 1)
        return bool(rate > 0.0 and size_now > 0.0 and size_now / rate <= MIRROR_CLOSING_S)

    def _holding(self, change, progress, side, why, waits, path) -> GateDecision:
        cap = self._cap(change, progress) if waits else self._slow(change, progress)
        return GateDecision(cap, f"lane on the {side} taken, held: {why}", path=path)

    @classmethod
    def _slow(cls, change: LaneChange, progress: float) -> float:
        """For a change it can miss: slowing for it, no slower than
        `MISS_SLOW_MPS`."""
        return max(MISS_SLOW_MPS, cls._cap(change, progress))

    @staticmethod
    def _cap(change: LaneChange, progress: float) -> float:
        room = change.last_start - progress
        return math.sqrt(2.0 * DECELERATION * max(room, 0.0))

    def _held(self, index: int, returning_at: float | None = None) -> ReferencePath | None:
        """The line kept in the lane left from the change's start on (or, a
        change begun, eased back to it from `returning_at`)."""
        change = self.changes[index]
        if change.leaving is None:
            return None
        if returning_at is None:
            return self._shaped(change, lambda s: 0.0, change.start, onto=False)
        begun = self._begun.get(index, change.start)
        span = self._span(index)

        def weight(s: float) -> float:
            return _smoothstep((s - begun) / span) * (
                1.0 - _smoothstep((s - returning_at) / RETURN_M)
            )

        return self._shaped(change, weight, returning_at + RETURN_M, onto=False)

    def _begun_at(self, index: int, begin: float) -> ReferencePath | None:
        """The line moving across from `begin`, over the change's length (or
        the shorter one it was begun late with)."""
        change = self.changes[index]
        if change.leaving is None:
            return None
        span = self._span(index)
        return self._shaped(
            change,
            lambda s: _smoothstep((s - begin) / span),
            begin + span,
            onto=True,
        )

    def _shaped(self, change: LaneChange, weight, until: float, *, onto: bool) -> ReferencePath:
        """The route's line with `change` moved across by `weight` (0 in the
        lane left, 1 in the lane moved into) from its start to `until`; then
        on along the route (`onto`) or along the lane left."""
        assert change.leaving is not None
        path, leaving = self.path, change.leaving
        target = ReferencePath(change.target, spacing=1.0)
        points = path.points
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        moved = [points[arc < change.start]]
        near_left = near_target = None
        for point, s in zip(
            points[(arc >= change.start) & (arc <= until)],
            arc[(arc >= change.start) & (arc <= until)],
            strict=True,
        ):
            left = leaving.nearest(float(point[0]), float(point[1]), near=near_left, window=50.0)
            near_left = left.s
            if s >= change.start + change.span:
                across = point  # the route's own line is in the lane moved into
            else:
                on = target.nearest(float(point[0]), float(point[1]), near=near_target, window=50.0)
                near_target = on.s
                across = np.array([on.x, on.z])
            w = weight(float(s))
            moved.append([[left.x + (across[0] - left.x) * w, left.z + (across[1] - left.z) * w]])
        if onto:
            moved.append(points[arc > until])
        else:
            if near_left is None:
                here = path.sample_at(change.start)
                near_left = leaving.nearest(here.x, here.z).s
            rest = leaving.points
            rest_arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(rest, axis=0).T))])
            moved.append(rest[rest_arc > near_left + 0.5])
        return ReferencePath(
            np.vstack([np.asarray(m, float).reshape(-1, 2) for m in moved]), spacing=path.spacing
        )

    def _target_lane(self, change: LaneChange, pose: tuple[float, float, float]):
        x, z, course = pose
        dx, dz = change.target[:, 0] - x, change.target[:, 1] - z
        ahead = dx * math.sin(course) + dz * math.cos(course)
        right = -(dx * math.cos(course) - dz * math.sin(course))
        try:
            return LanePath.from_truck_frame(ahead, right, self.camera)
        except ValueError:
            return None

    def _taken(self, change, target, speed, tracks, unseen, now) -> str | None:
        front = max(FRONT_GAP_MIN_M, FRONT_GAP_S * speed)
        present: set[int] = set()
        why = None
        for track in tracks:
            if not track.confirmed:
                continue
            distance, closing, lateral, width = track.estimate()
            present.add(track.id)
            if target is None or not _in_lane(
                lateral, width, target.right_at(distance), change.side
            ):
                self._beside.pop(track.id, None)
                continue
            if distance < BESIDE_WITHIN_M:
                self._beside[track.id] = _Beside(now, distance, closing)
            if distance < front and why is None:
                why = f"a vehicle {distance:.0f} m ahead in it"
        for track_id, middle in unseen:
            present.add(track_id)
            if (middle - self.camera.centre_col) * change.side > SIDE_OF_VIEW_PX and why is None:
                why = "a vehicle close beside"
        for track_id, seen in list(self._beside.items()):
            if track_id in present:
                continue
            age = now - seen.seen_at
            where = seen.distance + seen.closing * age
            if age > BESIDE_MEMORY_S or where < -BEHIND_CLEAR_M or where > front:
                del self._beside[track_id]
            elif why is None:
                why = "a vehicle last seen beside"
        return why


MIRROR_FAR: Final = 0.02
MIRROR_TRACK_S: Final = 1.5
MIRROR_CLOSING_SEEN: Final = 4
MIRROR_CLOSING_OVER_S: Final = 0.4
MIRROR_CLOSING_S: Final = 1.5
MIRROR_SHRINK: Final = 0.1
MIRROR_JUMP: Final = 2.0
MIRROR_BACK_M: Final = 1.5
"""A vehicle in the mirror on the side of a change, however far, whose box
grows so fast that at that rate it would be on the truck within
`MIRROR_CLOSING_S` -- its size over how fast it grows, fitted over the last
`MIRROR_TRACK_S` from `MIRROR_CLOSING_SEEN` sightings `MIRROR_FAR` or nearer
spread over `MIRROR_CLOSING_OVER_S` at least, none smaller by
`MIRROR_SHRINK` than the one before nor the last `MIRROR_JUMP` times it,
another box -- holds the change, and turns
one begun back while the truck is under `MIRROR_BACK_M` across, past the
hold point and `COMMITTED_M` too. On a delivery to Amsterdam a car came up
in the lane on the left at speed, unseen in the mirror as the change began;
seen 1.8 s before the hit, it never grew to `MIRROR_NEAR`, the gate had
stopped looking 20 m into the change, and the truck moved across into it.
A vehicle keeping its distance behind does not grow, and holds nothing.
Over the frames kept at 350 recorded events (brakes, hits, lights; 4 s
each, both mirrors) this rule fired in 12, most of them a car coming up
fast, and on the hit 1.1 s before it; looser (3 sightings, 2.5 s) it fired
in 89, on boxes jumping from one vehicle to another and traffic closing on
a braking truck."""

MIRROR_FROM_MPS: Final = 25.0 / 3.6
MIRROR_NEAR: Final = 0.2
MIRROR_SEEN: Final = 2
MIRROR_WITHIN_S: Final = 1.0
"""From `MIRROR_FROM_MPS` on, a lane change waits while the mirror on its
side shows a vehicle near -- its box `MIRROR_NEAR` of the mirror's height
or more -- `MIRROR_SEEN` times within `MIRROR_WITHIN_S`: coming up in the
lane moved into, which the camera ahead never sees. On the E6 by
Helsingborg and in London a lorry coming up in it hit the truck as it
moved across. In plain driving a box so near is in one frame of 40 in the
left mirror; not read slower, where in Bern the mirrors took trams and
buses for traffic behind. By night the detector sees little in them."""


def _in_lane(lateral: float, width: float, middle: float, side: int) -> bool:
    """Whether a vehicle on the side of the change reaches into the lane at
    `middle`: any of its width, not only its middle. A lorry beside the truck
    on a Dutch motorway showed its side, 9.4 m of box whose middle came out
    7.2 m to the right -- 3.4 m from the lane's middle -- while its near edge
    was 2.5 m out, in the lane; the truck moved across into it."""
    if lateral * side <= 0.0:
        return False  # in the truck's own lane or beyond it on the other side
    near, far = lateral - width / 2.0, lateral + width / 2.0
    return far >= middle - TARGET_HALF_WIDTH_M and near <= middle + TARGET_HALF_WIDTH_M


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
