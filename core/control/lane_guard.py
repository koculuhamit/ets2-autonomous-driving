"""Keeping the truck inside its painted lane where the map's line leaves it.

The map steers (ADR-0006), and where it is right the paint agrees. In the
container port at Bremen it was not: short of a left turn the map's lane
swung 2 m right of the painted one, the truck followed it over the right
line, and hit the barriers standing on it. The camera had both lines in
view, 4.4 m apart, all the way.

So the painted lane, where it is trusted (`trusted`), bounds the line on the
side the route is not going (`LaneGuard`): turning left 40 m on, the line
may leave the lane to the left -- the turn, the lane change, the exit -- but
not to the right, and the other way about; going straight, neither. Where
it would, the aim is moved back in, far enough for the truck's side to stay
`MARGIN_M` inside the line. The shift is taken up and let go gently, and
let go when the paint goes out of view.

The lines come from the classic finder (core.perception.lanes): straight
lines, wrong now and then on arrows and crossings. Only a pair as far apart
as a lane, agreeing with the last pair, is trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

__all__ = ["LaneGuard", "trusted"]

CHECK_M: Final = 12.0
"""Where the lines are read (core.perception.lanes.LANE_CHECK_M)."""

GOING_AT_M: Final = 40.0
GOING_M: Final = 1.0
TURNING_M: Final = 3.0
"""The route's line this far to one side `GOING_AT_M` ahead, of where it is
at `CHECK_M`, is going that way: off the lane there is allowed. This far
(`TURNING_M`), it is turning off the lane whose paint is seen: the paint no
longer tells how far the map is out, and a shift held is not lessened by
it -- at Bremen, the line into the left turn crossed the paint ahead from
right to left, and the shift went to nothing at the turn the map had as
far out as the straight."""

LEAVING_M: Final = 2.5
"""The route's line further than this from the painted lane's middle is
leaving the lane -- into a junction, an exit -- not the map out of place,
which it has been by 1.5 m at most: it is not shifted back for, and what
was held is let go. Into an S-bend off a Dutch dual carriageway the line
was 4.4 m right of the paint's middle; the guard took it for the map out
and held the truck 1.5 m left, onto the median's sign, twice."""

MIN_WIDTH_M: Final = 3.0
MAX_WIDTH_M: Final = 5.5
SAME_LANE_M: Final = 0.6
SAME_LANE_S: Final = 0.5
INSIDE_M: Final = 0.3
PAIRS: Final = 3
PAIRS_WITHIN_S: Final = 0.6
"""Lines this far apart, whose middle moved no more than `SAME_LANE_M`
since the last pair `SAME_LANE_S` ago, with the truck's middle `INSIDE_M`
or more inside each, are the truck's lane; `PAIRS` such pairs within
`PAIRS_WITHIN_S`, none other between, are trusted. Over the drives
recorded, a pair alone was another lane's as often as not -- the lines
found nearest were the next lane's -- and would have moved the truck 3 to
5 m. Pairs come between views with one line only, so not one after the
other."""

SEEN_OVER: Final = 2
"""The line over the paint in this many trusted pairs running before the
aim is moved: over the drives recorded, the line was over in one pair
alone about twice a kilometre, in two running half as often; at Bremen it
was over in every pair for 0.8 s."""

MAX_SPEED_MPS: Final = 20.0 / 3.6
"""Faster than this the paint is not acted on: on motorways out of Bremen,
at 55 and 80 km/h, the classic finder's lines moved the truck 1.5 m towards
the next lane three times in 20 minutes, the map's line right each time. At
39 km/h in England, the lane's lines opening to 4.8 m where a junction's lanes
begin, it moved the truck 1.4 m and let it go back, and the trailer hit a car
coming up in the next lane. Where the map is out -- ports, yards -- the truck
is slower."""

HALF_WIDTH_M: Final = 1.25
MARGIN_M: Final = 0.15
"""Half the truck's width, and what it keeps from a line."""

MAX_SHIFT_M: Final = 1.5
SHIFT_RATE_MPS: Final = 2.5
RELEASE_RATE_MPS: Final = 1.0
HOLD_M: Final = 50.0
HOLD_S: Final = 0.6
"""The shift is taken up at most this fast and let go at `RELEASE_RATE_MPS`;
it is held `HOLD_M` on along the path from the last trusted pair (`HOLD_S`
where the path is not told), and never exceeds `MAX_SHIFT_M`. In the port
at Bremen the map's line left the paint 1.3 s short of the barriers: at
1 m/s the shift came too late. Kept off them, the truck turned left 40 m
on, the paint out of view, the shift let go after 0.6 s -- and the turn,
as far off on the map as the straight before it, took it into the barriers
along its outside."""


def trusted(
    left: float,
    right: float,
    last: tuple[float, float] | None,
    now: float,
    truck: float = 0.0,
) -> bool:
    """Whether a pair of lines, metres right of the eye, is the truck's
    lane's: as wide as one, the truck's middle (`truck`, right of the eye)
    inside it, and where the last pair (time, middle) put it."""
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    if not MIN_WIDTH_M < right - left < MAX_WIDTH_M:
        return False
    if not left + INSIDE_M < truck < right - INSIDE_M:
        return False
    if last is None:
        return False
    then, middle = last
    return now - then <= SAME_LANE_S and abs((left + right) / 2.0 - middle) <= SAME_LANE_M


@dataclass
class LaneGuard:
    """The aim's shift, metres to the right, that keeps the truck in its lane.
    `truck` is where the truck's middle is, metres right of the eye (minus
    the camera's `seat_right_m`)."""

    truck: float = 0.0
    shift: float = 0.0
    _pairs: list[float] = field(default_factory=list)
    _over: int = 0
    _wanted: float = 0.0
    _wanted_at: float | None = None
    _wanted_from: float | None = None
    _last_pair: tuple[float, float] | None = None
    _stepped_at: float | None = None

    def wanted(self, left: float, right: float, at_check: float, ahead: float) -> float:
        """The shift for lines `left` and `right` and the route's line at
        `at_check`, all metres right of the eye `CHECK_M` ahead, the route's
        line at `ahead` `GOING_AT_M` ahead."""
        middle = (left + right) / 2.0
        room = (right - left) / 2.0 - HALF_WIDTH_M - MARGIN_M
        off = at_check - middle
        if abs(off) > LEAVING_M:
            return 0.0  # the route leaving this lane, not the map out
        going = ahead - at_check
        # Going right, the line may leave the lane to the right; left, left.
        over_right = off - room if going < GOING_M else -math.inf
        over_left = -off - room if going > -GOING_M else -math.inf
        if over_right > 0.0:
            return -min(over_right, MAX_SHIFT_M)
        if over_left > 0.0:
            return min(over_left, MAX_SHIFT_M)
        return 0.0

    def step(
        self,
        now: float,
        left: float | None = None,
        right: float | None = None,
        at_check: float | None = None,
        ahead: float | None = None,
        *,
        progress: float | None = None,
        speed: float = 0.0,
    ) -> float:
        """The shift now, `progress` metres along the path. Lines and the
        route's line are given when a new view has them (see `wanted`);
        without them the last wanted shift is held `HOLD_M` on, then let go."""
        if left is not None and right is not None and at_check is not None and ahead is not None:
            fresh = False
            if math.isfinite(left) and math.isfinite(right) and right - left > MIN_WIDTH_M:
                fresh = trusted(left, right, self._last_pair, now, self.truck)
                if fresh:
                    self._pairs.append(now)
                else:
                    self._pairs.clear()
            self._pairs = [t for t in self._pairs if now - t <= PAIRS_WITHIN_S]
            if fresh and len(self._pairs) >= PAIRS and speed <= MAX_SPEED_MPS:
                wanted = self.wanted(left, right, at_check, ahead)
                self._over = self._over + 1 if wanted else 0
                turning = abs(ahead - at_check) >= TURNING_M
                lessened = abs(wanted) < abs(self._wanted)
                if turning and lessened:
                    pass  # held on, `HOLD_M` from where it was last wanted
                elif self._over >= SEEN_OVER or not wanted:
                    self._wanted = wanted
                    self._wanted_at = now
                    self._wanted_from = progress
            if math.isfinite(left) and math.isfinite(right):
                self._last_pair = (now, (left + right) / 2.0)
        if self._wanted_at is None:
            self._wanted = 0.0
        elif progress is not None and self._wanted_from is not None:
            if progress - self._wanted_from > HOLD_M:
                self._wanted = 0.0
        elif now - self._wanted_at > HOLD_S:
            self._wanted = 0.0
        dt = 0.0 if self._stepped_at is None else max(now - self._stepped_at, 0.0)
        self._stepped_at = now
        taking_up = abs(self._wanted) > abs(self.shift)
        step = (SHIFT_RATE_MPS if taking_up else RELEASE_RATE_MPS) * dt
        self.shift += max(-step, min(step, self._wanted - self.shift))
        return self.shift
