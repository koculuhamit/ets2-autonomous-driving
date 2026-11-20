"""Following the vehicle ahead: how fast to go so the gap is right.

Not a separate controller. The speed controller (core.control.longitudinal)
already knows how to reach a speed with an actuator that cannot decelerate
between 0.22 and 1.81 m/s^2; what following needs is a *target* speed, and
one that folds into the same `min()` as the route's corner speeds and the
road's limit:

    target = min(route profile, road limit, following speed)

The following speed is the lead's own speed, corrected for how far the gap
is from where it should be:

    wanted gap = standstill gap + time gap x own speed
    following  = lead speed + (gap - wanted gap) / closing time

At the wanted gap the truck matches the lead; too close, it goes slower than
the lead until the gap has opened; too far, faster. `closing_time` sets how
hard: 4 s closes a 20 m error at 5 m/s of speed difference.

The comfort limits in the speed controller cap how hard it brakes for that,
which is right for a lead that slows gently and wrong for one that stops.
So each step also asks: at the current closing speed, what constant
deceleration stops the closing at the standstill gap -- or, once inside it,
at `min_gap_m`? If that is more than comfort allows, the answer is an
*urgent* deceleration, which the caller commands directly, past the comfort
limit. Aiming the hard stop at the minimum gap from the start, the first
version stopped 4 m behind a lead it meant to stop 8 m behind, and 3.9 m
behind one that braked at 7 m/s^2. That is the
difference between ACC and a collision; it is not a substitute for an
emergency brake that does not depend on the detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = ["FollowingDecision", "GapPolicy", "Lead", "follow", "stop_short"]

COMFORT_DECELERATION: Final = 2.5
"""What the speed controller will ask for on its own, m/s^2."""

ONCOMING_MPS: Final = 4.0
ONCOMING_SHARE: Final = 0.3
"""A 'lead' coming towards the truck faster than the larger of these -- 4 m/s,
or 30 % of the truck's speed -- is oncoming. Two of the hard brakes a city
drive would have made were for cars coming the other way through a junction.

Not tighter: a *stopped* car is closing at the truck's own speed, give or
take the error of that estimate, and at 2 m/s the first version declared a
stopped car 90 m ahead oncoming, ignored it, and drove into it in
simulation."""

MAX_DECELERATION: Final = 7.0
"""The most an urgent decision asks for. The truck can do 9.7; a load cannot
always take it, and the detector's distance is not good to the metre."""

STOP_DECELERATION: Final = 1.5
"""What a stop short of a place is planned at (`stop_short`), as for a red
light (core.control.traffic_light)."""

GAP_BAND_SHARE: Final = 0.2
BAND_SLOWER: Final = 4.0
"""Within this share of the time gap's metres either side of the wanted gap,
the gap is closed this many times more slowly. The detector's distance
wanders by metres at 50 m; corrected at full rate, following a car in Milan
the target swung by 5 km/h and throttle and brake took turns every two
seconds. A driver lets a gap that is about right be."""


@dataclass(frozen=True, slots=True)
class GapPolicy:
    time_gap_s: float = 1.5
    """Seconds of own speed between the truck and the lead: 37.5 m at 90 km/h.
    It was 2.0, ETS2's own AI's; the truck then braked for a car seen ahead
    sooner and harder than the driver wanted."""

    standstill_gap_m: float = 12.0
    """The gap when stopped behind the lead, eye to where it meets the road.

    Not less than the camera can see: nearer than about 9.5 m a vehicle's
    wheels are behind the dashboard. Asked for 8 m, the truck crept towards a
    trailer with its measured distance stuck at the edge of view, 10 m, and
    a distance budget stopped it 7.7 m out."""

    closing_time_s: float = 4.0
    """How quickly a gap error is closed."""

    min_gap_m: float = 6.0
    """What an urgent stop aims to keep, however the lead behaves."""

    def __post_init__(self) -> None:
        if self.time_gap_s <= 0 or self.closing_time_s <= 0:
            raise ValueError("the time gap and the closing time must be positive")
        if not 0 < self.min_gap_m < self.standstill_gap_m:
            raise ValueError("the minimum gap must be positive and below the standstill gap")

    def wanted_gap(self, speed: float) -> float:
        return self.standstill_gap_m + self.time_gap_s * max(speed, 0.0)


@dataclass(frozen=True, slots=True)
class Lead:
    distance_m: float
    closing_mps: float
    """Rate the gap changes; negative while it shrinks."""


@dataclass(frozen=True, slots=True)
class FollowingDecision:
    target_speed: float | None
    """None when there is nothing to follow: the road decides."""

    urgent_deceleration: float | None
    """m/s^2 to command directly, past the comfort limit; None when the
    speed controller can handle it."""

    reason: str


def follow(lead: Lead | None, speed: float, policy: GapPolicy | None = None) -> FollowingDecision:
    """The speed to follow `lead` at, from the truck's own `speed` (m/s)."""
    policy = policy or GapPolicy()
    if lead is None:
        return FollowingDecision(None, None, "clear road")

    if speed + lead.closing_mps < -max(ONCOMING_MPS, ONCOMING_SHARE * speed):
        # Coming towards the truck: not something to follow. A car in the
        # lane ahead driving the wrong way is for an emergency brake that
        # does not depend on the detector, not for adaptive cruise.
        return FollowingDecision(None, None, "oncoming: not a lead")
    lead_speed = max(speed + lead.closing_mps, 0.0)
    gap_error = lead.distance_m - policy.wanted_gap(speed)
    band = GAP_BAND_SHARE * policy.time_gap_s * max(speed, 0.0)
    inner = min(max(gap_error, -band), band)
    correction = inner / BAND_SLOWER + (gap_error - inner)
    target = max(lead_speed + correction / policy.closing_time_s, 0.0)

    urgent = None
    reason = f"following at {lead.distance_m:.0f} m (wants {policy.wanted_gap(speed):.0f})"
    if lead.closing_mps < 0.0:
        aim = (
            policy.standstill_gap_m
            if lead.distance_m > policy.standstill_gap_m + 1.0
            else (policy.min_gap_m)
        )
        room = lead.distance_m - aim
        # No more than stopping takes: a lead closing faster than the truck
        # goes is coming towards it, and braking past a stop does nothing for
        # that. A low loader turning across in Hamburg seemed to close at
        # 6.8 m/s on a truck doing 2.5, and was braked for at 7 m/s^2
        # (16 November); stopping took 0.4.
        closing = min(-lead.closing_mps, max(speed, 0.0))
        needed = MAX_DECELERATION if room <= 0.0 else closing**2 / (2.0 * room)
        if needed > COMFORT_DECELERATION:
            urgent = min(needed, MAX_DECELERATION)
            reason = f"closing {-lead.closing_mps:.1f} m/s at {lead.distance_m:.0f} m: braking hard"
    return FollowingDecision(target, urgent, reason)


def stop_short(room: float, speed: float, reason: str, slack: float = 0.0) -> FollowingDecision:
    """To stop `room` metres on from `speed` (m/s): at `STOP_DECELERATION`,
    braked for directly (an urgent deceleration) once the truck is over that
    curve -- the speed controller alone held it there, coasting, until 2.5
    were needed, and from 0 to 2 m short the stop took 7 m/s^2, five times
    on one drive to Leipzig (16 November). Never faster than now, but for a
    crawl (`APPROACH_CREEP_MPS`): giving way to a car in a queue out of
    Milan, the speed it could still stop from was a ceiling the truck sped up
    to, 12 to 16 km/h, and then braked at 7 m/s^2.

    `slack`: metres past the place the truck may still stop in when it is
    come on too late for the place itself -- what lies beyond it before what
    it stops for -- braked for no harder than that takes."""
    if room <= 0.0 and (speed <= NO_URGENCY_BELOW_MPS or room + slack <= 0.0):
        urgent = MAX_DECELERATION if speed > NO_URGENCY_BELOW_MPS else None
        return FollowingDecision(0.0, urgent, reason)
    if room <= 0.0:
        target, needed = 0.0, math.inf
    else:
        target = min(math.sqrt(2.0 * STOP_DECELERATION * room), max(speed, APPROACH_CREEP_MPS))
        needed = speed * speed / (2.0 * room)
    urgent = None
    # Slower than `NO_URGENCY_BELOW_MPS` the speed controller's own hold
    # stops the truck: what a stop needs, worked out over its last few
    # centimetres, would be a brake at full.
    if needed > STOP_DECELERATION and speed > NO_URGENCY_BELOW_MPS:
        within = speed * speed / (2.0 * (room + max(slack, 0.0)))
        urgent = min(max(within, STOP_DECELERATION), MAX_DECELERATION)
    return FollowingDecision(target, urgent, reason)


APPROACH_CREEP_MPS: Final = 3.0
"""How fast a stop ahead is crept up to from slower."""

NO_URGENCY_BELOW_MPS: Final = 3.0
"""At the place already, slower than this the speed controller's own hold
stops the truck (core.control.longitudinal.HOLD_BELOW)."""
