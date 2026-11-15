"""Stopping for traffic lights: where the map says they are, as the camera sees them.

The route knows every light's stop line (core.planning.route.StopLine), and
perception says what colour is lit (core.perception.lights). This decides,
each step, what that means for the speed, in the same terms as following a
vehicle does: a target speed for the `min()`, or an urgent deceleration.

- **Green**: nothing. Once the truck is closer to the line than it could
  stop from for amber, it is *committed*: if the light changes now, it goes
  through, as a driver would.
- **Amber**: stop if that takes no more than `AMBER_STOP_DECELERATION`,
  otherwise go through.
- **Red**: stop, braking past comfort if it has to, up to
  `RED_STOP_DECELERATION`. Seen later than that, stopping would leave the
  truck in the junction; it goes through.
- **Never seen**: no colour of it seen yet, it is not slowed for at all
  (since 6 October): the truck goes on at the road's speed, and a red or
  amber first seen on the way is stopped for as above, or gone through
  when too late. Slowed for as if red, and passed at 20 km/h when still
  unseen 30 m out, the truck dropped speed at every T junction the map
  gives a light the road does not have.
- **Unknown after a colour**: the lamps have left the view -- close under
  the head, or behind a lorry -- after one was seen. Approached as the
  colour last seen, and at the line waited at -- `UNSEEN_WAIT_S`, longer
  than these lights' whole 42 s cycle -- unless the traffic standing there
  moves off, and then taken for one that cannot be seen from there.
- **Red for ever**: no light in the game's profiles stays red longer than
  46 s. A red seen unchanged for `STUCK_RED_S` at the stop point is not a
  light's -- a no-parking sign's red ring held the truck at a light the map
  has and the road does not -- and is passed like a light never seen.
- **Asleep**: many lights are dark from 23:30 to 03:00, and the map says
  which. A dark light is a junction without one. A red seen there is the
  next light's: in city-01, one 64 m further on stopped the truck at a
  sleeping one when red was obeyed at sleeping lights too.

The truck stops `STOP_SHORT_M` before the line, measured from its position
in the telemetry: far enough back to see the lamp on the pole beside the
line, which is how it knows when to go again.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from core.localization.path import ReferencePath
from core.map import Signal
from core.perception.lights import HOLD_S, LightState
from core.planning.route import StopLine

__all__ = ["LightDecision", "LightLine", "TrafficLights", "lines_on_path"]

STOP_SHORT_M: Final = 10.0
"""Where to stop, before the line, measured from the truck's reference point.
The painted line can be a crossing's width before the lane the map says the
light governs. Stopped 8 m back in the game, with an 80 degree view, the
truck was at the zebra and its light's lamps were out of view -- the pole's
to the right, the overhead ones behind the cab roof -- so it never saw the
green; the pole's lamp left the view about 10 m back. At 13 m, with the
view widened to 110 degrees, the driver found the truck stopping too far
back. Overhead lamps 5.5 m up leave the view behind the roof at 11.4 m
whatever the width of the view; the timeout for an unseen light
(`UNSEEN_WAIT_S`) is what covers them."""

STOP_DECELERATION: Final = 1.5
"""The deceleration the approach is planned at, m/s^2."""

COMFORT_DECELERATION: Final = 2.5
AMBER_STOP_DECELERATION: Final = 3.0
RED_STOP_DECELERATION: Final = 5.0
MAX_DECELERATION: Final = 7.0

DARK_DECIDE_M: Final = 30.0
DARK_WATCHED_M: Final = 30.0
DARK_SPEED: Final = 20.0 / 3.6
"""A light none of whose lamps has been seen, watched over at least
`DARK_WATCHED_M` of the approach, by `DARK_DECIDE_M` from its line, is taken
for one that is not there or not lit: passed at `DARK_SPEED`, not stopped
at. Every working light in the drives so far was seen from 50-110 m, by day
and by night; the ones never seen were junctions the map gives a light and
the road does not."""

PASSED_CLEAR_M: Final = 40.0
"""How far past a light's line its lamps may still be in view: across the
junction it stands at."""

UNSEEN_WAIT_S: Final = 45.0
"""Stopped at a light none of whose lamps has been seen, how long to wait
before going on. Its green has come and gone unseen by then."""

QUEUE_STOOD_S: Final = 2.0
"""At a red that has gone out of view, standing at least this long before
the traffic moving off is taken for the green (`traffic_going`)."""

STARTED_AT_WAIT_S: Final = 10.0
"""The same wait at a light the drive began within `DARK_DECIDE_M` of, whose
approach was never watched to tell it is not there. In Bern a drive begun
8 m from a line whose lights stood out of view waited the whole 45 s at a
junction without one; the driver chose this, knowing a red out of view
would then be gone through after it."""

STUCK_RED_S: Final = 90.0
"""Twice the longest red of any profile (`3ph`: 44 s, and 2 s of amber)."""

STOPPED_MPS: Final = 0.3
MOVING_MPS: Final = 1.0

LOOK_AHEAD_M: Final = 150.0
"""Lights further than this are not yet considered."""

AT_STOP_M: Final = 0.5


@dataclass(frozen=True, slots=True)
class LightLine:
    """A stop line, as a distance along the path being driven."""

    s: float
    signal: Signal


@dataclass(frozen=True, slots=True)
class LightDecision:
    target_speed: float | None
    urgent_deceleration: float | None
    reason: str
    distance_m: float | None = None
    """From the truck to the stop line of the light being considered."""


@dataclass
class TrafficLights:
    """The lights along one path, and what to do about the next one."""

    lines: Sequence[LightLine]
    _committed: set[float] = field(default_factory=set)
    _stopping: set[float] = field(default_factory=set)
    _waiting_since: float | None = None
    _red_since: float | None = None
    _watched_from: dict[float, float] = field(default_factory=dict)
    _saw_colour: set[float] = field(default_factory=set)
    _dark: set[float] = field(default_factory=set)
    _unwatched: set[float] = field(default_factory=set)
    _shadowed: set[float] = field(default_factory=set)
    """Lines approached while the last light's colour stood (`_held_over`):
    a colour of theirs may have been taken for that light's."""
    _with_the_traffic: set[float] = field(default_factory=set)
    """Lines gone through, out of view, as the traffic at them moved off."""
    """Lines the drive began within `DARK_DECIDE_M` of (`STARTED_AT_WAIT_S`)."""
    _begun: bool = False
    """Lines a stop has been decided for. The actuator cannot brake between
    0.22 and 1.81 m/s^2, so on the way down the deceleration still needed
    wanders past the go-through thresholds; a stop once begun is finished."""

    _next: float | None = None
    _passed: float | None = None
    """The line last gone past, making `_next` the next."""
    _cleared_at: float | None = None

    def __post_init__(self) -> None:
        self.lines = sorted(self.lines, key=lambda line: line.s)

    def carry_over(self, previous: TrafficLights, tolerance: float = 1.0) -> None:
        """Keep what `previous` had decided about the lines both have.

        When the route is changed on the move (core.planning.gps), the lines
        up to the fork are the same lines; a light already being gone
        through, or stopped for, must stay so.
        """
        for line in self.lines:
            if any(abs(line.s - s) <= tolerance for s in previous._committed):
                self._committed.add(line.s)
            if any(abs(line.s - s) <= tolerance for s in previous._stopping):
                self._stopping.add(line.s)
            if any(abs(line.s - s) <= tolerance for s in previous._unwatched):
                self._unwatched.add(line.s)
            if any(abs(line.s - s) <= tolerance for s in previous._shadowed):
                self._shadowed.add(line.s)
            if any(abs(line.s - s) <= tolerance for s in previous._saw_colour):
                self._saw_colour.add(line.s)
        self._waiting_since = previous._waiting_since
        self._red_since = previous._red_since
        self._begun = previous._begun

    def _held_over(self, progress: float, now: float, following: float) -> bool:
        """Whether the colour read may still be the light just gone past's.

        Its lamps stay in view across the junction -- in Uppsala, a green
        one 25 m beyond its line, crossed slowly in traffic -- and the
        filter holds a state `HOLD_S` after them. Taken for the next light's,
        that one, which the map has and the road did not, counted as seen,
        and the truck waited at it for a lamp to show. So until the truck is
        `PASSED_CLEAR_M` past the line (or half way to the next), and for
        `HOLD_S` after, the next light is taken as not seen yet.
        """
        assert self._passed is not None
        clear = self._passed + min(PASSED_CLEAR_M, 0.5 * (following - self._passed))
        if progress < clear:
            return True
        if self._cleared_at is None:
            self._cleared_at = now
        return now - self._cleared_at < HOLD_S

    def next_line(self, progress: float) -> LightLine | None:
        return next((line for line in self.lines if line.s > progress), None)

    def step(
        self,
        *,
        progress: float,
        speed: float,
        state: LightState,
        minute_of_day: float,
        now: float = 0.0,
        traffic_going: bool = False,
    ) -> LightDecision:
        """What the next light means now. `traffic_going`: the vehicles
        standing with the truck are moving off (core.control.acc)."""
        line = self.next_line(progress)
        if not self._begun:
            self._begun = True
            if line is not None and line.s - progress <= DARK_DECIDE_M:
                self._unwatched.add(line.s)
        if line is None:
            return LightDecision(None, None, "no light ahead")
        if line.s != self._next:
            self._passed, self._next, self._cleared_at = self._next, line.s, None
        # Only a light whose lamps were seen can have them taken for the
        # next one's. In Bern a light none of whose lamps showed was passed
        # as dark, and the next, 68 m on, was not watched until 18 m short
        # of it: too late to tell it was dark too, and the truck stood at it.
        held_over = (
            self._passed is not None
            and self._passed in self._saw_colour
            and self._held_over(progress, now, line.s)
        )
        if held_over:
            state = LightState.UNKNOWN
        if line.s - progress > LOOK_AHEAD_M:
            return LightDecision(None, None, "no light ahead")
        distance = line.s - progress
        if line.s in self._with_the_traffic:
            return LightDecision(DARK_SPEED, None, "light out of view: with the traffic", distance)
        if line.s in self._committed:
            return LightDecision(None, None, "going through", distance)
        if line.signal.asleep(minute_of_day):
            return LightDecision(None, None, "light asleep", distance)
        self._watched_from.setdefault(line.s, distance)
        if held_over:
            # Not watched while the last light's colour stands: in Milan the
            # next light was 35 m on, its colours taken for the last one's,
            # and found dark it was driven through on red at 20 km/h.
            self._watched_from[line.s] = distance
            self._shadowed.add(line.s)
        if state is not LightState.UNKNOWN:
            self._saw_colour.add(line.s)
        elif not (
            line.s in self._saw_colour or line.s in self._unwatched or line.s in self._shadowed
        ):
            # No colour seen of it yet: not slowed for. Every working light
            # was seen from 50-110 m; a red first seen nearer is stopped for
            # as hard as `RED_STOP_DECELERATION`, or gone through. Slowed for
            # as if red, and passed at 20 km/h when never seen, the truck
            # dropped speed at every T junction the map gives a light the
            # road does not have (6 October). Still slowed for as before: a
            # light the drive began at, and one whose colours may have been
            # taken for the last one's.
            return LightDecision(None, None, "no lamp seen: on", distance)
        elif (
            line.s not in self._saw_colour
            and distance <= DARK_DECIDE_M
            and self._watched_from[line.s] - distance >= DARK_WATCHED_M
        ):
            self._dark.add(line.s)
        if line.s in self._dark and state is LightState.UNKNOWN:
            return LightDecision(DARK_SPEED, None, "no lamp seen: slowly through", distance)

        room = distance - STOP_SHORT_M
        speed = max(speed, 0.0)
        at_stop = room <= AT_STOP_M and speed < STOPPED_MPS
        if state is LightState.RED and at_stop:
            if self._red_since is None:
                self._red_since = now
            if now - self._red_since >= STUCK_RED_S:
                self._committed.add(line.s)
                self._red_since = None
                return LightDecision(None, None, "red too long to be a light: going on", distance)
        else:
            self._red_since = None
        if state is not LightState.UNKNOWN or not at_stop:
            self._waiting_since = None
        else:
            if self._waiting_since is None:
                self._waiting_since = now
            if (
                traffic_going
                and line.s in self._saw_colour
                and now - self._waiting_since >= QUEUE_STOOD_S
            ):
                # A red it was waiting at, hidden by a lorry drawn up beside:
                # the traffic standing at it moving off is its green.
                self._with_the_traffic.add(line.s)
                self._waiting_since = None
                return LightDecision(
                    DARK_SPEED, None, "light out of view: with the traffic", distance
                )
            # A light stopped short of only because it stood close after the
            # last one, no colour of it ever seen: at the bridge in Geneva,
            # 52 m after a green, the map has one the road does not, and the
            # truck waited the whole 45 s at it.
            short_wait = line.s in self._unwatched or (
                line.s in self._shadowed and line.s not in self._saw_colour
            )
            wait = STARTED_AT_WAIT_S if short_wait else UNSEEN_WAIT_S
            if now - self._waiting_since >= wait:
                self._committed.add(line.s)
                self._waiting_since = None
                return LightDecision(None, None, "never seen: going on", distance)
        needed = speed * speed / (2.0 * room) if room > AT_STOP_M else math.inf
        # Committed on green once amber could no longer be stopped for.
        # At the comfort limit instead, in city-01 the truck committed 0.2 s
        # before a red it could still stop for at 2.6 m/s^2 -- the amber
        # between them unseen.
        committed_within = speed * speed / (2.0 * AMBER_STOP_DECELERATION)

        if state is LightState.GREEN:
            # On the move and too close to stop in comfort: whatever the
            # light does next, the truck is on its way through. Not at rest:
            # held at a green by something else, it saw the light go red
            # again in the game, and must still stop for that.
            if speed > MOVING_MPS and room <= committed_within + AT_STOP_M:
                self._committed.add(line.s)
                return LightDecision(None, None, "green: going through", distance)
            return LightDecision(None, None, "green", distance)

        if line.s not in self._stopping:
            if state is LightState.AMBER and needed > AMBER_STOP_DECELERATION:
                self._committed.add(line.s)
                return LightDecision(None, None, "amber, too close to stop", distance)
            if state is LightState.RED and needed > RED_STOP_DECELERATION and speed > 3.0:
                self._committed.add(line.s)
                return LightDecision(None, None, "red, too late to stop", distance)
            if state is not LightState.UNKNOWN:
                self._stopping.add(line.s)

        if room <= AT_STOP_M:
            return LightDecision(0.0, None, f"stopped for {state.value}", distance)
        target = math.sqrt(2.0 * STOP_DECELERATION * room)
        urgent = None
        if state is not LightState.UNKNOWN and needed > COMFORT_DECELERATION:
            urgent = min(needed, MAX_DECELERATION)
        return LightDecision(target, urgent, f"stopping for {state.value}", distance)


ON_PATH_M: Final = 3.0
"""A stop line further than this from the path is not on it."""


def lines_on_path(lines: Sequence[StopLine], path: ReferencePath) -> list[LightLine]:
    """Where along `path` each stop line is, in route order.

    A route can pass the same place twice, so each line is looked for only
    beyond the one before it.
    """
    points = path.points
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    found: list[LightLine] = []
    after = -math.inf
    for line in lines:
        distances = np.hypot(points[:, 0] - line.position[0], points[:, 1] - line.position[2])
        candidates = np.flatnonzero((s > after) & (distances < ON_PATH_M))
        if len(candidates) == 0:
            continue
        # The first stretch of the path that passes the line, and its closest point.
        run_end = candidates[0]
        while run_end + 1 < len(points) and distances[run_end + 1] < ON_PATH_M:
            run_end += 1
        index = candidates[0] + int(np.argmin(distances[candidates[0] : run_end + 1]))
        found.append(LightLine(float(s[index]), line.signal))
        # The next line is on a later pass, if this place is passed again.
        after = float(s[run_end])
    return found
