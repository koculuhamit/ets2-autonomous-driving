"""Adaptive cruise: perception, the route and the gap policy, in one step.

The control loop hands this the newest perception report, where the truck
is on its route, and its speed; it hands back what following wants -- a
speed cap, perhaps an urgent deceleration -- or stops the truck if it can
no longer see.

Seeing is a precondition, not an input. A report older than `MAX_AGE_S`,
none at all, or one taken while another window covered the game, means the
road ahead is unknown; driving on at speed would be driving blind, so that
raises `PerceptionLost`, and the caller brings the truck to rest as it does
for any other unsafe condition.

The lead must be in the lane the route takes, seen from the driver's seat
(core.perception.tracking.Path): straight ahead is wrong wherever the road
bends or splits.

A lead that disappears close up has not gone: nearer than about 9.5 m its
wheels are behind the dashboard, and the camera has no distance for it. So
a lead last seen within `REMEMBER_WITHIN_M` is remembered as standing where
it was, and the truck stops for it. Driving towards a dropped trailer in the
game, the first version lost it 10.5 m out, took the road for clear and
accelerated; the distance budget stopped the truck about 5 m short. The
memory goes only when a lead is seen again not far beyond it -- the same
vehicle having moved off -- and never for something seen further down the
road, which the remembered one would be hiding. Only a lead followed for
`REMEMBER_AFTER_S` is remembered: stopped at a red light in the game, a car
crossing the junction 9.8 m ahead was the lead for a moment, was remembered,
and held the truck through the whole green that followed.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from core.control.following import FollowingDecision, GapPolicy, Lead, follow, stop_short
from core.control.limits import UnsafeCondition
from core.localization.path import ReferencePath
from core.perception.camera import GroundCamera
from core.perception.tracking import (
    CROSSING_WITHIN_M,
    MAX_LEAD_WIDTH_M,
    MIN_ALONG_MPS,
    PATH_HALF_WIDTH_M,
    STALE_MEASUREMENT_S,
    CrossingFilter,
    CutInFilter,
    lead_vehicle,
)
from core.perception.tracking import Path as LanePath

__all__ = [
    "AdaptiveCruise",
    "OutOfCab",
    "PerceptionLost",
    "crossing_spans",
    "junction_spans",
    "lane_ahead",
    "out_of_a_ring",
    "prefab_spans",
    "ring_entry_spans",
    "ring_way_spans",
]

MAX_AGE_S: Final = 0.5
"""A report older than this is not the road now. At 90 km/h, 12.5 m."""

LANE_AHEAD_M: Final = 150.0

REMEMBER_WITHIN_M: Final = 15.0
REMEMBER_AFTER_S: Final = 1.0
"""How long a vehicle must have been the lead to be remembered."""

REMEMBER_BELOW_MPS: Final = 2.0
"""Only a lead standing -- moving slower than this -- is remembered as
standing where it was. At night on a motorway out of Torino a car
overtaking on the right was taken for the lead 10 m ahead, going as fast
as the truck; it drew away out of view, was remembered standing, and the
truck braked at 7 m/s^2 from 77 km/h to a stop in the fast lane. One
moving is lost for a moment (`LEAD_HOLD_S`), not stopped for."""

OUT_OF_VIEW_BEND_RAD: Final = math.radians(20.0)
OUT_OF_VIEW_BEND_PAST_M: Final = 10.0
"""A lead the path bends away to by this much, out of view, is not
remembered standing there, nor held stopped close up: it went round the
bend, out of the camera's view. Round a tight right turn in Leipzig a lorry
ahead at 7 km/h was remembered standing 14 m on, out of view, and the truck
stopped in the turn for it twice, set the parking brake and crept on
(16 November)."""

REMEMBER_AT_MOST_S: Final = 10.0
GREEN_FOR_S: Final = 3.0
"""A lead remembered, not seen again for `REMEMBER_AT_MOST_S`, with the light
green for `GREEN_FOR_S`, is let go and crept past (`CREEP_PAST_MPS`): a vehicle
standing at the light would have moved by then. At a light on the way out of
Scotland something read for 2 s, 9.5 m ahead, across an empty road at night was
remembered, and the truck stood through the green for good; at a light in
Aberdeen, a lorry in the next lane. Not without a green: stopped close behind a
car the detector cannot see, the truck would creep into it."""

IN_VIEW_FROM_M: Final = 10.5
IN_VIEW_GONE_S: Final = 6.0
"""A lead remembered further than this, where a vehicle's wheels are in view
(nothing nearer than about 9.5 m gets a distance: `CUT_MARGIN_PX`), not seen for
`IN_VIEW_GONE_S`, has gone, with or without a light: it is let go and crept past.
At a roundabout's entry out of Calais, at night, the car the truck stopped
behind, 11.6 m off, went on round the ring; remembered, it held the truck at the
entry for good."""

REMEMBER_STILL_M: Final = 3.0
"""And only one that has stayed within this of where it was along the road
for `REMEMBER_AFTER_S`. On a narrow road in Norway, the yellow middle line
close by the truck's left, cars coming the other way were the lead as they
came by, their speed read as standing -- the closing speed is the smaller
of two readings, so it lags -- and each was remembered 4 m ahead once past:
the truck stopped for no one, went on, and stopped again with the next. Its
speed alone does not tell: braking hard, a car standing was read as coming
at 5 m/s."""

FORGET_BEYOND_M: Final = 20.0
"""A lead seen again no more than this beyond the remembered one ends the
memory: it is the same vehicle, moving off."""

LEAD_SPEED_TAU_S: Final = 1.5
"""The lead's own speed is smoothed over about this long before following
uses it. Read raw, it wandered by 3 km/h (std) while following in the game;
through the speed controller's gain that is 1.7 m/s^2 of request, over the
brake's threshold, and the truck pulsed the brake 18 times a minute. A real
lead changes speed over seconds, not frames; a lead that really brakes hard
is the urgent path's business, which does not wait for this."""

URGENT_AFTER_S: Final = 0.1
"""An urgent deceleration is commanded only once following has asked for one
this long without a break: a lead that looks like it is closing fast for a
single report is more often a box than a car."""


ONCOMING_LOOK_M: Final = 100.0
ONCOMING_GAP_S: Final = 8.0
TURN_APPROACH_M: Final = 60.0
OVERTAKER_NEAR_M: Final = 25.0
OVERTAKER_ASIDE_M: Final = 2.5
OVERTAKER_IN_M: Final = 1.2
"""A vehicle first seen nearer than `OVERTAKER_NEAR_M`, `OVERTAKER_ASIDE_M`
or more to the side of the lane's middle, came up from beside or behind:
overtaking. It is followed only once its middle is within `OVERTAKER_IN_M`
of the lane's: in front, in the lane. Seen beside the cab's pillar at
10 m, coaches, tankers and caravans going by in the next lane were taken
for vehicles ahead about 11 times an hour, at speed, and braked the truck
for a moment each (the driver's word, 12 November: not smooth). A vehicle
the truck comes up to is first seen far off, and is not one; nor, through a
junction, one turning across close in front.

Nor one the truck goes by: seen in the next lane, `OVERTAKER_ASIDE_M` or
more aside, from `OVERTAKER_NEAR_M` or further, within `PASSING_SEEN_S`,
and nearer than `OVERTAKER_NEAR_M` now -- where, beside the cab, where it
meets the road is hard to see. Out of Dover, at 89 km/h, a car in the
next lane being overtaken was taken, at 11 m, for one ahead."""

PASSING_SEEN_S: Final = 3.0
MOVING_IN_M: Final = 1.0
"""But not one seen moving in from afar: `MOVING_IN_M` or more nearer the
lane's middle over `PASSING_SEEN_S`, while `OVERTAKER_NEAR_M` or further.
In Norway a lorry pulling out of a side road came from 5.8 m to 3.2 m off
while 36 to 21 m ahead, and had to be followed."""


TIGHT_LOOK_M: Final = 60.0
TIGHT_CLEAR_M: Final = 6.0
TIGHT_WAIT_AT_MOST_S: Final = 20.0
MOUTH_CLEAR_M: Final = 5.0
MOUTH_SEEN_M: Final = 60.0
"""The mouth of the road a tight turn goes into is clear when most of it is
in view, nearer than `MOUTH_SEEN_M`, and nothing seen -- confirmed or not --
is within `MOUTH_CLEAR_M` of it (`AdaptiveCruise.mouth_clear`): the truck
then goes round wide, over it (core.planning.tight_turns.wide_turn)."""

TIGHT_LATE_DECELERATION: Final = 2.0
"""A vehicle first seen in a tight turn when stopping short of it would take
more than this is not waited for: the truck is at the turn already, and
goes on, a vehicle in its way followed as any other. A Renault Magnum came
up to two tight turns in Sheffield at 10 km/h, saw a car in each 0 to 1 m
short of where to wait, and braked at 7 m/s^2 for it (16 November)."""

TIGHT_STANDING_MPS: Final = 0.5
TIGHT_WHY: Final = "waiting for the junction to clear for a tight turn"
"""Coming to a tight turn (core.planning.tight_turns), from `TIGHT_LOOK_M`
short of it, the truck waits short of the junction (`STOP_BEFORE_TURN_M`)
while a vehicle is within `TIGHT_CLEAR_M` of the turn's lanes, moving or
standing: round a tight turn the rig sweeps the junction, and goes on into
it deeper. On the way to Bergen it followed a tanker into such a junction
and stood there behind it, the tanker turning round in front; its trailer,
swinging by, hit the cab. Waited for `TIGHT_WAIT_AT_MOST_S`, standing, it
goes on as before: the vehicle may be waiting for the truck."""

STOP_BEFORE_TURN_M: Final = 2.0
"""Turning across the oncoming lanes (`turn_across_spans`), the truck waits
short of the turn while a vehicle coming the other way, nearer than
`ONCOMING_LOOK_M`, would reach it within `ONCOMING_GAP_S`: a turning
lorry takes about that to clear the lane. Out of Mannheim the truck turned
left in front of a car coming the other way, which it saw only 6.9 m off,
and hit it. Once in the turn it goes on: stopped in the oncoming lane is
worse."""

RING_SIDE_PX: Final = 250.0
"""A vehicle with no distance this far off the view's middle, on the side
the ring's traffic comes from, is on the ring close by (as
lane_change.SIDE_OF_VIEW_PX for the lane beside)."""

RING_WATCH_M: Final = 40.0
RING_LOOK_M: Final = 35.0
RING_LATERAL_M: Final = 3.0
RING_MEMORY_S: Final = 4.0
RING_MOVING_MPS: Final = 1.0
"""Only a vehicle moving -- across the view, or along the road, this fast --
is the ring's traffic: one parked by the entry would hold the truck there."""
"""Coming within `RING_WATCH_M` of a roundabout's entry, a vehicle seen
within `RING_LOOK_M`, `RING_LATERAL_M` or more off to the side the ring's
traffic comes from, is waited for at the entry until `RING_MEMORY_S` after
it was last seen: from the cab, the ring's last metres to the entry are
behind the pillar and the mirror. Near Duisburg a lorry coming round was
seen 19 m off and 13 m to the left, then not until it was across the
truck's way, and hit."""

RING_ENTRY_MPS: Final = 1.5
RING_APPROACH_DECELERATION: Final = 1.5
RING_CREEP_M: Final = 8.0
"""Into a roundabout no faster than this, about 5 km/h, and on for this far
into it, as a driver edges in: from it the truck stops within a metre of a
vehicle on the ring seen as it gets there; the clear way in is taken without
stopping. At 16 km/h, in Malmö, a police car coming round from the left was
in the windscreen for a moment as the truck came in, and was hit: the
ring's last metres to the entry are behind the cab's pillar."""

PREFAB_AHEAD_M: Final = 30.0
PREFAB_BESIDE_M: Final = 12.0
"""Coming within `PREFAB_AHEAD_M` of a junction, and through it, a vehicle
moving into the lane from as far as `PREFAB_BESIDE_M` to the side is followed
however it came, the other way too: at a junction in England, through on the
amber, a car coming the other way turned across the truck from 10 m to the
right and was hit at 31 km/h. On the open road, where the lanes coming the
other way swing across the view in a bend, one coming the other way is not
(core.perception.tracking.CUT_IN_COMING_MPS)."""

CROSSING_BELOW_MPS: Final = 50.0 / 3.6
"""Faster than this, a vehicle read as crossing is followed if in the lane,
like any other, and given way to only if it closes on the lane steadily
(`CrossingMotion.steady`). On a dual carriageway through a junction in
England, at 75 and 80 km/h, lorries and coaches beyond the median, 12 m off,
read as crossing, and the truck was braked at 6.8 m/s^2 for them; so from 9
November none was given way to this fast -- and by Metz, at 78 km/h, a
lorry pulling out of a side road across the truck's way, seen 31 m off and
1.2 s before, was not, and was hit."""

JUNCTION_THROUGH_MPS: Final = 60.0 / 3.6
JUNCTION_BEFORE_M: Final = 40.0
JUNCTION_APPROACH_DECELERATION: Final = 1.0
"""Through a junction where other ways cross the route's lane on the level
(`crossing_spans`), and from `JUNCTION_BEFORE_M` short of it, no faster than
this -- the driver's limit, 11 November -- slowing to it from far enough back
at `JUNCTION_APPROACH_DECELERATION`. Whatever pulls out is first seen from
behind the cab's pillar, near: by Metz a lorry out of a side road was seen
31 m off; at 78 km/h a stop takes about 33 m, from 60 about 25."""

GIVE_WAY_CLEAR_S: Final = 1.5
"""Once the truck gives way -- to the oncoming traffic short of a left turn,
or to a vehicle crossing -- it goes on only when nothing has called for it
for this long. Decided afresh on each view, the truck braked for a car
coming the other way, lost sight of it for a moment and went to full
throttle, saw it again and braked hard, three times in one left turn."""

LEAD_HOLD_S: Final = 0.6
"""A lead out of sight for no longer than this is followed where it would
be by now. Lost for a view in a bend, it was gone, and the truck went from
braking to full throttle and back within 0.2 s."""

CLOSE_HOLD_M: Final = 20.0
CLOSE_HOLD_S: Final = 3.0
"""A lead lost nearer than `CLOSE_HOLD_M` is held this long: beyond
`LEAD_HOLD_S` where it was by then, no further. And while it is held, a
vehicle seen more than `FORGET_BEYOND_M` beyond it is not taken for the
lead in its place. At a border a lorry's trailer, 10 m ahead in the queue
through customs, filled the windscreen and was lost; a car 70 m on became
the lead in 0.3 s, and the truck drove into the trailer.

Only a lead followed for `REMEMBER_AFTER_S` and not come `REMEMBER_STILL_M`
nearer along the road meanwhile is held so: not a car crossing a junction,
the lead for a moment, nor one coming the other way past the truck."""

CREEP_PAST_MPS: Final = 1.5
CREEP_PAST_M: Final = 15.0
"""A lead let go after `CLOSE_HOLD_S`, still unseen, is crept past at this
speed, about 5 km/h, until this far beyond where it was: gone, or too close
for the detector to see, it is not driven into at speed. No slower than a
stopped truck needs to be asked to move off again
(core.control.parking.RESUME_TARGET_MPS): at 1.4 m/s, stopped behind a car
at the Chiasso customs, the truck stood on after the car had gone."""

HIDDEN_FOOT_M: Final = 8.0
HIDDEN_ROOM_M: Final = 2.0
"""A vehicle in the truck's way whose wheels the dashboard hides -- seen with
no distance, or seen still but no longer measured -- is stopped for, within
`HIDDEN_ROOM_M`: it is closer than the road shows, about `HIDDEN_FOOT_M`. A
lorry turning in front at night, and a trailer queuing at a border, were
seen as their wheels went under the dashboard 10 m off, then measured no
more, and hit."""

QUEUE_AHEAD_M: Final = 40.0
QUEUE_LATERAL_M: Final = 7.5
MOVING_OFF_MPS: Final = 1.5
MOVING_OFF_S: Final = 0.6
"""Standing, a vehicle within `QUEUE_AHEAD_M` ahead and `QUEUE_LATERAL_M`
either side -- in the truck's lane or one beside it -- not crossing, going
away faster than `MOVING_OFF_MPS` for `MOVING_OFF_S`, is the traffic at the
light going (`traffic_going`): in Bern a lorry drew up in the lane on the
right, hid the red the truck was waiting at, and the truck could not see it
go green."""

ONCOMING_OWN_MPS: Final = 8.0
"""A vehicle seen coming towards the truck at this or more of its own is
coming the other way, wherever it seems to be: round a left-hand bend near
Salzburg (18 November) a car in the oncoming lane, 4 to 6 m to the left,
came from 115 m to 13 m in a second where the bend put the truck's lane;
taken for one standing in it, the truck braked at 8.6 m/s^2 to a stop.
Slower than this, one is taken to stand at most: on another truck's camera,
cars standing at a red seemed to come at up to 17 km/h (Duisburg)."""

OUT_OF_CAB_S: Final = 1.0
"""The game's camera out of the cab this long, the road is not seen as
perception reads it (core.perception.cab): the truck stops."""

JUNCTION_MARGIN_M: Final = 15.0
"""A crossing vehicle this far short of a junction is in it: waiting to pull out."""


def turn_across_spans(route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where `path` turns across a junction's lanes coming the other way --
    a left turn, where traffic keeps right -- as (start, end) metres along
    it: where the oncoming traffic is waited for (`AdaptiveCruise`). Only
    where the route turns: going straight on past a side road, the lane of
    the oncoming traffic turning into it crosses the route too, and the
    truck waited for traffic that was to wait for it."""
    return _spans(route, path, opposite=True)


def junction_spans(route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where `path` crosses another way through a junction -- the route's
    lanes inside prefabs that cross another lane of their prefab -- as
    (start, end) metres along it.

    Only these are given way in (`AdaptiveCruise`): turning left across the
    oncoming lanes, or over a side road. A slip road joining a motorway, or
    leaving it, meets its lanes end to end and crosses none, and vehicles
    swept sideways on its bends were given way to at 7 m/s^2 for nothing.

    And only where the route has no right of way (`_gives_way`): going
    straight on through a T junction near Duisburg, the side road's turning
    lanes crossed the route's, and the truck crept and stopped for parked
    cars by the junction. Into a roundabout, where the route's lane joins
    the ring end to end, it gives way to the ring: entering at 32 km/h it
    hit a lorry going round.
    """
    return _spans(route, path, opposite=False)


def crossing_spans(route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where `path` goes through a junction whose other ways cross the
    route's lane on the level, right of way or not -- not a roundabout, and
    not a slip road, which meets its lanes end to end -- as (start, end)
    metres along it (`JUNCTION_THROUGH_MPS`)."""
    network = route.network
    spans: list[tuple[float, float]] = []
    near = 0.0
    for lane in route.lanes:
        if int(network.lane_source[lane]) != 1 or _on_a_ring(network, lane):
            continue
        if not _crossed_on_the_level(network, lane):
            continue
        start = network.lane_points(lane)[0]
        s = path.nearest(float(start[0]), float(start[2]), near=near, window=1e9).s
        near = s
        end = s + float(network.lane_length[lane])
        if spans and s <= spans[-1][1] + 1.0:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((s, end))
    return spans


LEVEL_M: Final = 3.0
"""Lanes crossing with their heights this far apart do not meet: one goes
over the other."""


def _crossed_on_the_level(network, lane: int) -> bool:
    """Whether another lane of a junction lane's prefab, off a roundabout,
    crosses it away from the ends of both, at the same height."""
    mine3 = np.asarray(network.lane_points(lane))
    mine = mine3[:, [0, 2]]
    for other in _same_owner(network, lane):
        other = int(other)
        if other == lane:
            continue
        theirs3 = np.asarray(network.lane_points(other))
        theirs = theirs3[:, [0, 2]]
        for point in _crossings(mine, theirs):
            ends = (mine[0], mine[-1], theirs[0], theirs[-1])
            if min(float(np.hypot(*(point - e))) for e in ends) <= ENDS_MEET_M:
                continue
            here = mine3[int(np.argmin(np.hypot(*(mine - point).T))), 1]
            there = theirs3[int(np.argmin(np.hypot(*(theirs - point).T))), 1]
            # A ring last: walked for every lane of a junction, it took most
            # of the second a 3 km path's spans took.
            if abs(float(here) - float(there)) < LEVEL_M and not _on_a_ring(network, other):
                return True
    return False


def ring_entry_spans(route, path: ReferencePath) -> list[tuple[float, float, int]]:
    """Where `path` enters a roundabout, as (start, end, side) -- metres along
    it, and the side the ring's traffic comes from, -1 left, 1 right where
    traffic keeps left: where the ring's traffic is given way to however it
    moves (`AdaptiveCruise.rings`)."""
    network = route.network
    left_hand = network.lane_left_hand
    side = 1 if left_hand is not None and bool(left_hand[route.lanes[0]]) else -1
    return [(start, end, side) for start, end in _spans(route, path, opposite=False, rings=True)]


def prefab_spans(route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where `path` runs through junctions -- the route's prefab lanes -- as
    (start, end) metres along it, whether or not the route gives way there."""
    network = route.network
    spans: list[tuple[float, float]] = []
    near = 0.0
    for lane in route.lanes:
        if int(network.lane_source[lane]) != 1:
            continue
        start = network.lane_points(lane)[0]
        s = path.nearest(float(start[0]), float(start[2]), near=near, window=1e9).s
        near = s
        end = s + float(network.lane_length[lane])
        if spans and s <= spans[-1][1] + 1.0:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((s, end))
    return spans


def ring_way_spans(route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where `path` goes through a roundabout, as (start, end) metres along
    it: from the way into it to the end of the last of its lanes the route
    takes, the way out."""
    network = route.network
    owners = network.lane_owner
    ways: list[tuple[float, float]] = []
    near = 0.0
    lanes = route.lanes
    index = 0
    while index < len(lanes):
        lane = lanes[index]
        if int(network.lane_source[lane]) != 1 or not _enters_a_ring(network, lane):
            index += 1
            continue
        last = index
        while last + 1 < len(lanes) and owners[lanes[last + 1]] == owners[lane]:
            last += 1
        start = network.lane_points(lane)[0]
        s = path.nearest(float(start[0]), float(start[2]), near=near, window=1e9).s
        out = network.lane_points(lanes[last])[-1]
        end = path.nearest(float(out[0]), float(out[2]), near=s, window=1e9).s
        ways.append((s, max(end, s)))
        near = end
        index = last + 1
    return ways


def _spans(
    route, path: ReferencePath, *, opposite: bool, rings: bool = False
) -> list[tuple[float, float]]:
    network = route.network
    spans: list[tuple[float, float]] = []
    near = 0.0
    owners = network.lane_owner
    came_in: np.ndarray | None = None
    for before, lane in zip((None, *route.lanes[:-1]), route.lanes, strict=True):
        if before is None or owners[before] != owners[lane]:
            # Into a junction: the way the route came in, for how far it turns.
            came_in = None if before is None else _end_direction(network, before)
        if int(network.lane_source[lane]) != 1:
            continue
        if rings:
            if not _enters_a_ring(network, lane):
                continue
        else:
            turn = _turn_deg(network, lane, came_in)
            if opposite and turn < TURNING_DEG:
                continue
            crosses = _crosses_another(network, lane, opposite=opposite, turn=turn)
            if not (crosses or (not opposite and _enters_a_ring(network, lane))):
                continue
        start = network.lane_points(lane)[0]
        s = path.nearest(float(start[0]), float(start[2]), near=near, window=1e9).s
        near = s
        end = s + float(network.lane_length[lane])
        if spans and s <= spans[-1][1] + 1.0:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((s, end))
    return spans


ENDS_MEET_M: Final = 1.5
"""Lanes meeting within this of an end of either merge or part there: they
do not cross."""


def _crosses_another(
    network, lane: int, *, opposite: bool = False, turn: float | None = None
) -> bool:
    """Whether a prefab lane crosses another lane of its prefab away from the
    ends of both, one it has to give way to (`_gives_way`, the lane turning
    `turn` degrees); with `opposite`, one that comes the other way -- whose
    start heads against this lane's start."""
    mine = np.asarray(network.lane_points(lane))[:, [0, 2]]
    entry = _heading(mine)
    for other in _same_owner(network, lane):
        other = int(other)
        if other == lane:
            continue
        theirs = np.asarray(network.lane_points(other))[:, [0, 2]]
        if opposite and float(np.dot(entry, _heading(theirs))) > -OPPOSITE_COS:
            continue
        for point in _crossings(mine, theirs):
            ends = (mine[0], mine[-1], theirs[0], theirs[-1])
            if min(float(np.hypot(*(point - e))) for e in ends) > ENDS_MEET_M:
                if _gives_way(network, lane, other, turn):
                    return True
                break
    return False


STRAIGHT_DEG: Final = 15.0
TURNING_DEG: Final = 30.0
"""A junction lane that bends no more than `STRAIGHT_DEG` goes straight on;
one that bends `TURNING_DEG` or more turns. Going straight on, the route has
the right of way over a lane that turns across it."""


def _gives_way(network, lane: int, other: int, turn: float | None = None) -> bool:
    """Whether a route's junction lane gives way to `other`, a lane of the
    same junction meeting it. The map has no right of way, so it is read off
    the lanes: into a roundabout the ring has it; going straight on, the
    route has it over a lane that turns; otherwise -- turning, or two roads
    crossing straight -- the route gives way."""
    if _on_a_ring(network, lane):
        return False
    if _on_a_ring(network, other):
        return True
    mine = _turn_deg(network, lane) if turn is None else turn
    return not (mine <= STRAIGHT_DEG and _turn_deg(network, other) >= TURNING_DEG)


MERGE_DEG: Final = 20.0
"""Lanes ending together from more than this far apart join at an angle: a
roundabout's entry and its ring. A slip road joins a motorway nearly side
by side (0-14 degrees on the route out of Duisburg)."""


def _enters_a_ring(network, lane: int) -> bool:
    """Whether a junction lane ends where a roundabout's ring goes on, from
    off the ring: the entry, which gives way to the ring."""
    if _on_a_ring(network, lane):
        return False
    mine = np.asarray(network.lane_points(lane))[:, [0, 2]]
    chord = mine[-1] - mine[0]
    for other in _same_owner(network, lane):
        other = int(other)
        if other == lane:
            continue
        theirs = np.asarray(network.lane_points(other))[:, [0, 2]]
        if float(np.hypot(*(theirs[-1] - mine[-1]))) > ENDS_MEET_M:
            continue
        if float(np.hypot(*(theirs[0] - mine[0]))) <= ENDS_MEET_M:
            continue  # leaving together: a split, not a join
        if _angle_deg(chord, theirs[-1] - theirs[0]) >= MERGE_DEG and _on_a_ring(network, other):
            return True
    return False


def out_of_a_ring(network, lane: int) -> bool:
    """Whether a junction lane is a roundabout's but neither on its ring nor
    into it: a way out (or one by it), an island and its signs on the outside."""
    if int(network.lane_source[lane]) != 1:
        return False
    known = _WAYS_OUT.setdefault(id(network), {})
    if lane not in known:
        owner = network.lane_owner[lane]
        known[lane] = not (_on_a_ring(network, lane) or _enters_a_ring(network, lane)) and any(
            _on_a_ring(network, int(other))
            for other in np.flatnonzero(np.asarray(network.lane_owner) == owner)
        )
    return known[lane]


_WAYS_OUT: dict[int, dict[int, bool]] = {}
"""Per network, `out_of_a_ring` of each junction lane asked about."""


def _on_a_ring(network, lane: int, most: int = 64) -> bool:
    """Whether a junction lane goes round a roundabout: following its
    junction's lanes from it comes back to it."""
    owners = network.lane_owner
    seen, queue = set(), [lane]
    while queue and len(seen) < most:
        for following in network.next_lanes(queue.pop()):
            following = int(following)
            if owners[following] != owners[lane]:
                continue
            if following == lane:
                return True
            if following not in seen:
                seen.add(following)
                queue.append(following)
    return False


def _turn_deg(network, lane: int, came_in: np.ndarray | None = None) -> float:
    """How far a junction lane has turned by its end, degrees, from the way
    into the junction: `came_in`, or the start of the junction's lanes that
    lead to it alone. A junction's turn is drawn in pieces; the last piece of
    a turn across the route near Duisburg bent only 9-28 degrees of its 38-98."""
    if came_in is None:
        owners = network.lane_owner
        first = lane
        for _ in range(8):
            before = [b for b in _predecessors(network, first) if owners[b] == owners[lane]]
            if len(before) != 1:
                break
            first = before[0]
        points = np.asarray(network.lane_points(first))[:, [0, 2]]
        came_in = points[min(3, len(points) - 1)] - points[0]
    return _angle_deg(came_in, _end_direction(network, lane))


def _end_direction(network, lane: int) -> np.ndarray:
    points = np.asarray(network.lane_points(lane))[:, [0, 2]]
    return points[-1] - points[max(0, len(points) - 4)]


def _predecessors(network, lane: int) -> list[int]:
    """The lanes leading into `lane`."""
    found = _PREDECESSORS.get(id(network))
    if found is None or found[0]() is not network:
        starts = np.asarray(network.successor_start)
        nexts = np.asarray(network.successors)
        sources = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        order = np.argsort(nexts, kind="stable")
        found = (weakref.ref(network), nexts[order], sources[order])
        _PREDECESSORS[id(network)] = found
    _, targets, sources = found
    low, high = np.searchsorted(targets, lane), np.searchsorted(targets, lane, side="right")
    return [int(v) for v in sources[low:high]]


_PREDECESSORS: dict[int, tuple[weakref.ref, np.ndarray, np.ndarray]] = {}
"""Per network, by id checked by a weak reference (as route's `_OUTERMOST`):
every lane's successors, sorted by successor."""


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    norms = float(np.hypot(*a)) * float(np.hypot(*b))
    if norms <= 0.0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / norms, -1.0, 1.0))))


OPPOSITE_COS: Final = 0.5
"""A lane whose start heads more than 120 degrees from another's comes the other way."""


def _heading(points: np.ndarray) -> np.ndarray:
    """The unit direction a polyline sets off in."""
    step = points[min(3, len(points) - 1)] - points[0]
    length = float(np.hypot(*step))
    return step / length if length > 0.0 else np.zeros(2)


def _same_owner(network, lane: int) -> list[int]:
    """The lanes of `lane`'s prefab or road, `lane` too; by owner, built once
    per network. Looked for over the whole map's lanes for each lane of a
    route, the junctions of a 3 km path took 2 s, and the drive was slow to
    start."""
    found = _BY_OWNER.get(id(network))
    if found is None or found[0]() is not network:
        owners = np.asarray(network.lane_owner)
        order = np.argsort(owners, kind="stable")
        keys, starts = np.unique(owners[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        table = {
            key: order[start:end].tolist()
            for key, start, end in zip(keys.tolist(), starts, ends, strict=True)
        }
        found = (weakref.ref(network), table)
        _BY_OWNER[id(network)] = found
    return found[1].get(np.asarray(network.lane_owner)[lane].item(), [lane])


_BY_OWNER: dict[int, tuple] = {}
"""Per network, checked by a weak reference: its lanes by owner (`_same_owner`)."""


def _crossings(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Where polylines `a` and `b`, `(n, 2)` each, cross."""
    if (
        a[:, 0].max() < b[:, 0].min()
        or b[:, 0].max() < a[:, 0].min()
        or a[:, 1].max() < b[:, 1].min()
        or b[:, 1].max() < a[:, 1].min()
    ):
        return []  # apart: most lanes of a big junction, and what kept a drive from starting
    p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
    q, s = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]
    cross = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / cross
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / cross
    hit = (np.abs(cross) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    i, j = np.nonzero(hit)
    return [a[k] + t[k, m] * (a[k + 1] - a[k]) for k, m in zip(i, j, strict=True)]


class PerceptionLost(UnsafeCondition):
    """The stack cannot see the road ahead."""


class OutOfCab(PerceptionLost):
    """The game's camera is not where perception is calibrated for: the
    drive stops and says so, a view in time or not. Taken for a view a
    moment late, it slowed the truck to a stop on an open road near Bergen
    and held it there, saying nothing: the camera had been moved up after a
    reload, and the dashboard no longer matched."""


def lane_ahead(
    path: ReferencePath,
    progress: float,
    x: float,
    z: float,
    course: float,
    camera: GroundCamera,
    length: float = LANE_AHEAD_M,
) -> LanePath | None:
    """The route from `progress` on, as the driver's eye sees it; None near its end."""
    stations = np.arange(progress, min(progress + length, path.length), 2.0)
    if len(stations) < 3:
        return None
    samples = [path.sample_at(float(s)) for s in stations]
    dx = np.array([p.x for p in samples]) - x
    dz = np.array([p.z for p in samples]) - z
    # Forward is (sin c, cos c) and left is (cos c, -sin c) for a course in
    # the atan2(dx, dz) frame; see core.control.lateral._to_vehicle_frame.
    ahead = dx * np.sin(course) + dz * np.cos(course)
    right = -(dx * np.cos(course) - dz * np.sin(course))
    try:
        return LanePath.from_truck_frame(ahead, right, camera)
    except ValueError:
        return None


@dataclass(slots=True)
class _GiveWay:
    """Giving way: to stop at `stop_s` along the path until nothing has
    called for it for `GIVE_WAY_CLEAR_S`, or the truck is past `until_s`."""

    stop_s: float
    until_s: float
    track_id: int
    seen_at: float
    why: str


@dataclass(frozen=True, slots=True)
class CruiseStep:
    decision: FollowingDecision
    lead: Lead | None
    lead_id: int | None
    report_age_s: float


class AdaptiveCruise:
    def __init__(self, camera: GroundCamera, policy: GapPolicy | None = None) -> None:
        self.camera = camera
        self.policy = policy or GapPolicy()
        self._urgent_since: float | None = None
        self._crossing = CrossingFilter()
        self._cut_in = CutInFilter()
        self._remembered: tuple[float, float] | None = None
        self._remembered_at: float = 0.0
        self.green_since: float | None = None
        """Since when the light ahead has shown green, as the drive tells it."""
        """(distance, progress) of a close lead when last seen."""
        self._standing_since: tuple[float, float] | None = None
        """Since when the lead has been standing, and where along the path."""
        """Since when a lead, whichever track it is, has been standing ahead:
        timed per track, a lorry re-tracked as the truck drew up behind it
        was never long enough the lead to be remembered."""
        self._lead_speed: tuple[int, float, float] | None = None
        """(track id, smoothed own speed, time) of the lead being followed."""
        self._last_lead: tuple[int, float, float, float] | None = None
        self._lead_since: tuple[int, float, float] | None = None
        self._creep_until: float | None = None
        """Up to where along the path a lead let go close up is crept past."""
        """The lead's track, since when it has been the lead, and where along
        the path it was then."""
        """(track id, where along the path, own speed, time) the lead was last seen."""
        self._give_way: _GiveWay | None = None
        self._out_of_cab_since: float | None = None
        self.out_of_cab_s = 0.0
        """How long the game's camera has been out of the cab, seconds."""
        self._moving_off: dict[int, float] = {}
        """Track id -> since when it has been going away, while standing."""
        self.traffic_going = False
        """Whether the traffic alongside and ahead is moving off
        (`MOVING_OFF_S`), the truck standing: for core.control.traffic_light."""
        self.junctions: list[tuple[float, float]] = []
        """Stretches of the path through junctions, as (start, end) metres
        along it (`junction_spans`): the only places a crossing vehicle is
        given way to."""
        self.turns_across: list[tuple[float, float]] = []
        self.tight_turns: list[tuple[float, float, np.ndarray]] = []
        """Tight turns along the path (core.planning.tight_turns.tight_turn_spans):
        (start, end, the turn's lanes as world (x, z)), waited short of while
        something is in them (`TIGHT_CLEAR_M`)."""
        self._first_seen: dict[int, tuple[float, float]] = {}
        """Each track's distance and offset from the lane's middle when first
        seen (`OVERTAKER_NEAR_M`)."""
        self._seen_aside: dict[int, float] = {}
        """When each track was last seen in the next lane from afar
        (`PASSING_SEEN_S`)."""
        self._afar: dict[int, list[tuple[float, float]]] = {}
        """Each track's (time, distance from the lane's middle) while
        `OVERTAKER_NEAR_M` or further, over `PASSING_SEEN_S` (`MOVING_IN_M`)."""
        self._tight_wait: tuple[float, float] | None = None
        """(the turn's start, since when) the truck has stood short of it."""
        """Stretches of the path that turn across the oncoming lanes
        (`turn_across_spans`): where the oncoming traffic is waited for."""
        self._ring_side_first: dict[int, tuple[float, float]] = {}
        """Of those, when and how far to the side each was first seen."""
        self._ring_side_seen: dict[int, float] = {}
        """Vehicles seen on the side the ring's traffic comes from, coming to
        a roundabout's entry: track id -> when last seen (`RING_MEMORY_S`)."""
        self.rings: list[tuple[float, float, int]] = []
        """Stretches of the path into roundabouts, and the side the ring's
        traffic comes from (`ring_entry_spans`): where a vehicle crossing is
        given way to even moving partly the truck's way, and one close on that
        side, too close to measure, is waited for."""
        self.ring_ways: list[tuple[float, float]] = []
        self.crossings: list[tuple[float, float]] = []
        """Stretches of the path through junctions crossed on the level
        (`crossing_spans`): taken no faster than `JUNCTION_THROUGH_MPS`."""
        self.prefabs: list[tuple[float, float]] = []
        """Stretches of the path through junctions (`prefab_spans`)."""
        """Stretches of the path through roundabouts, into them and out
        (`ring_way_spans`)."""

    def step(
        self,
        report,
        now: float,
        *,
        path: ReferencePath,
        progress: float,
        x: float,
        z: float,
        course: float,
        speed: float,
    ) -> CruiseStep:
        """What following wants now. Raises PerceptionLost if the road is unseen."""
        result = self._step(
            report, now, path=path, progress=progress, x=x, z=z, course=course, speed=speed
        )
        # No crawl short of a turn across the oncoming lanes, nor through a
        # tight turn or round a roundabout: the driver's word (16 November),
        # the bend's own speed (core.planning.speed_profile) and no lower --
        # crawling round at 8 to 12 km/h, braking and stopping in the turn,
        # looked bad. What is waited for short of a turn is still waited for.
        cap, why = self._ring_cap(progress), "a roundabout"
        junction = self._junction_cap(progress)
        if junction is not None and (cap is None or junction < cap):
            cap, why = junction, "a junction"
        target = result.decision.target_speed
        if cap is None or (target is not None and target <= cap):
            return result
        decision = replace(
            result.decision,
            target_speed=cap,
            reason=f"{result.decision.reason}; slowing for {why}",
        )
        return replace(result, decision=decision)

    def _ring_cap(self, progress: float) -> float | None:
        """Short of a roundabout, the speed the truck can stop at its entry
        from in comfort, no lower than `RING_ENTRY_MPS`; through it and out,
        None elsewhere. It
        came at one near Duisburg at 32 km/h, and the lorry coming round,
        seen late, was hit."""
        for start, _, _ in self.rings:
            room = start - progress - STOP_BEFORE_TURN_M
            if -(STOP_BEFORE_TURN_M + RING_CREEP_M) <= room <= TURN_APPROACH_M:
                return max(
                    math.sqrt(2.0 * RING_APPROACH_DECELERATION * max(room, 0.0)), RING_ENTRY_MPS
                )
        return None

    def _junction_cap(self, progress: float) -> float | None:
        """Short of a junction crossed on the level, the speed from which the
        truck slows to `JUNCTION_THROUGH_MPS` by `JUNCTION_BEFORE_M` short of
        it; that speed through it; None elsewhere."""
        caps = []
        for start, end in self.crossings:
            room = start - JUNCTION_BEFORE_M - progress
            if progress > end or room > 1000.0:
                continue
            caps.append(
                math.sqrt(
                    JUNCTION_THROUGH_MPS**2 + 2.0 * JUNCTION_APPROACH_DECELERATION * max(room, 0.0)
                )
            )
        return min(caps) if caps else None

    def _step(
        self,
        report,
        now: float,
        *,
        path: ReferencePath,
        progress: float,
        x: float,
        z: float,
        course: float,
        speed: float,
    ) -> CruiseStep:
        if report is None:
            raise PerceptionLost("no perception report yet")
        age = now - report.grabbed_at
        if report.covered:
            raise PerceptionLost("another window is in front of the game")
        if age > MAX_AGE_S:
            raise PerceptionLost(f"the newest view of the road is {age:.2f} s old")
        if getattr(report, "in_cab", True):
            self._out_of_cab_since = None
            self.out_of_cab_s = 0.0
        elif self._out_of_cab_since is None:
            self._out_of_cab_since = now
        elif now - self._out_of_cab_since >= OUT_OF_CAB_S:
            self.out_of_cab_s = now - self._out_of_cab_since
            raise OutOfCab(
                "the game's camera is not in the cab as calibrated: switch to the cab view (1)"
                " and put the seat and head back where they were"
            )

        lane = lane_ahead(path, progress, x, z, course, self.camera)
        crossing = self._crossing.crossing(report.tracks, lane, report.grabbed_at)
        # One joining the traffic is followed once in the lane, like any other;
        # one coming the other way too, but where the route crosses its lanes.
        oncoming = not self._in_a_junction(progress, ahead=CROSSING_WITHIN_M)
        crossing -= self._crossing.joining(speed, oncoming=oncoming)
        if speed >= CROSSING_BELOW_MPS:
            crossing = frozenset()
        self._watch_the_queue(report.tracks, crossing, speed, now)
        in_a_prefab = any(start - PREFAB_AHEAD_M <= progress <= end for start, end in self.prefabs)
        if in_a_prefab:
            cutting_in = self._cut_in.cutting_in(
                report.tracks, lane, report.grabbed_at, beside=PREFAB_BESIDE_M
            )
        else:
            cutting_in = self._cut_in.cutting_in(report.tracks, lane, report.grabbed_at, speed)
        if self._in_a_junction(progress, ahead=CROSSING_WITHIN_M) or self._near_a_ring(progress):
            # Where the route crosses other ways and joins rings, what moves
            # into its way is given way to, not followed (_watch_for_give_way).
            cutting_in = frozenset()
        # Pulling into the lane, seen from its corner however wide: followed,
        # in a junction too (CrossingFilter.pulling_in); into a roundabout,
        # what comes round is given way to instead.
        pulling_in = self._crossing.pulling_in(speed)
        if not self._near_a_ring(progress):
            cutting_in |= pulling_in
        overtaking = self._overtaking(report.tracks, lane, report.grabbed_at)
        if in_a_prefab:
            # Through a junction what comes up close from the side is turning
            # across, and followed (PREFAB_BESIDE_M), not going by.
            overtaking = frozenset()
        found = lead_vehicle(
            report.tracks, lane, crossing | overtaking, cutting_in - crossing - overtaking
        )
        remembered = None
        if self._remembered is not None:
            distance_then, progress_then = self._remembered
            place = distance_then - (progress - progress_then)
            unseen_for = now - self._remembered_at
            at_a_green = (
                unseen_for >= REMEMBER_AT_MOST_S
                and self.green_since is not None
                and now - self.green_since >= GREEN_FOR_S
            )
            if at_a_green or (place >= IN_VIEW_FROM_M and unseen_for >= IN_VIEW_GONE_S):
                self._creep_until = progress + max(place, 0.0) + CREEP_PAST_M
                self._remembered = None
                self._last_lead = None
        if self._remembered is not None:
            distance_then, progress_then = self._remembered
            remembered = distance_then - (progress - progress_then)
        if found is not None and remembered is not None:
            if found[1][0] > remembered + FORGET_BEYOND_M:
                found = None  # further down the road, behind the remembered one
            else:
                remembered = None  # seen again: followed, the memory kept fresh
                self._remembered_at = now
        self._watch_for_give_way(report.tracks, progress, speed, now, oncoming, report.unseen)
        self._watch_tight_turn(report.tracks, progress, x, z, course, speed, now)
        self._seen = (report.tracks, x, z, course)
        self._path = path
        giving = self._giving_way(progress, speed, now)
        step = self._following(found, remembered, now, progress, speed, age)
        hidden = self._foot_hidden(report, lane)
        if hidden is not None:
            decision = stop_short(
                HIDDEN_ROOM_M, speed, "stopping for a vehicle close ahead, its wheels hidden"
            )
            target = step.decision.target_speed
            if target is None or decision.target_speed <= target:
                step = CruiseStep(decision, Lead(HIDDEN_FOOT_M, -speed), hidden, age)
        if giving is not None:
            decision, stop_id = giving
            target = step.decision.target_speed
            if target is None or decision.target_speed <= target:
                urgent = max(
                    decision.urgent_deceleration or 0.0, step.decision.urgent_deceleration or 0.0
                )
                decision = replace(decision, urgent_deceleration=urgent or None)
                lead = Lead(self._give_way.stop_s - progress + self.policy.standstill_gap_m, -speed)
                step = CruiseStep(decision, lead, stop_id, age)
        return replace(step, decision=self._debounce(step.decision, now))

    def _foot_hidden(self, report, lane) -> int | None:
        """A vehicle in the truck's way whose wheels the dashboard hides
        (`HIDDEN_FOOT_M`): its track, or None."""

        def right_at(ahead: float) -> float:
            return 0.0 if lane is None else lane.right_at(ahead)

        camera = self.camera
        middle = camera.centre_col + camera.focal_px * right_at(HIDDEN_FOOT_M) / HIDDEN_FOOT_M
        reach = camera.focal_px * PATH_HALF_WIDTH_M / HIDDEN_FOOT_M
        for track_id, column in getattr(report, "unseen", ()):
            if abs(column - middle) <= reach:
                return track_id
        for track in report.tracks:
            if not track.confirmed or track.measurement_age <= STALE_MEASUREMENT_S:
                continue
            distance, _, lateral, width = track.estimate()
            if distance > CLOSE_HOLD_M or width > MAX_LEAD_WIDTH_M:
                continue
            if abs(lateral - right_at(distance)) <= PATH_HALF_WIDTH_M:
                return track.id
        return None

    def _watch_the_queue(self, tracks, crossing, speed: float, now: float) -> None:
        """Whether the vehicles standing with the truck are moving off."""
        if speed >= MIN_ALONG_MPS:
            self._moving_off.clear()
            self.traffic_going = False
            return
        going = {}
        for track in tracks:
            if not track.confirmed or track.id in crossing:
                continue
            if track.measurement_age > STALE_MEASUREMENT_S:
                continue
            distance, closing, lateral, _ = track.estimate()
            near = distance < QUEUE_AHEAD_M and abs(lateral) < QUEUE_LATERAL_M
            if near and closing - speed > MOVING_OFF_MPS:
                going[track.id] = self._moving_off.get(track.id, now)
        self._moving_off = going
        self.traffic_going = any(now - since >= MOVING_OFF_S for since in going.values())

    def _overtaking(self, tracks, lane, now: float) -> frozenset[int]:
        """Tracks going by the truck, or gone by by it, that are not in front
        of it (`OVERTAKER_NEAR_M`)."""
        alive = set()
        found = set()
        for track in tracks:
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, _, lateral, _ = estimate
            offset = lateral - (0.0 if lane is None else lane.right_at(distance))
            alive.add(track.id)
            first = self._first_seen.setdefault(track.id, (distance, offset))
            if distance >= OVERTAKER_NEAR_M:
                afar = self._afar.setdefault(track.id, [])
                afar.append((now, abs(offset)))
                while afar and now - afar[0][0] > PASSING_SEEN_S:
                    afar.pop(0)
                if abs(offset) >= OVERTAKER_ASIDE_M:
                    self._seen_aside[track.id] = now
            if abs(offset) <= OVERTAKER_IN_M:
                continue
            overtaker = first[0] < OVERTAKER_NEAR_M and abs(first[1]) >= OVERTAKER_ASIDE_M
            aside = self._seen_aside.get(track.id)
            afar = self._afar.get(track.id, [])
            moving_in = len(afar) >= 2 and afar[0][1] - afar[-1][1] >= MOVING_IN_M
            passed = (
                aside is not None
                and now - aside <= PASSING_SEEN_S
                and distance < OVERTAKER_NEAR_M
                and not moving_in
            )
            if overtaker or passed:
                found.add(track.id)
        for gone in set(self._first_seen) - alive:
            del self._first_seen[gone]
        for gone in set(self._seen_aside) - alive:
            del self._seen_aside[gone]
        for gone in set(self._afar) - alive:
            del self._afar[gone]
        return frozenset(found)

    def mouth_clear(self, mouth) -> bool:
        """Whether nothing is seen within `MOUTH_CLEAR_M` of the points
        `mouth` (`(n, 2)` world x, z), most of them in view
        (`MOUTH_SEEN_M`, the camera's width): unseen is not clear."""
        seen = getattr(self, "_seen", None)
        mouth = np.asarray(mouth, float)
        if seen is None or len(mouth) == 0:
            return False
        tracks, x, z, course = seen
        ahead = np.array([math.sin(course), math.cos(course)])
        right = np.array([-math.cos(course), math.sin(course)])
        relative = mouth - np.array([x, z])
        forward = relative @ ahead
        aside = relative @ right
        half_view = math.atan2(self.camera.centre_col, self.camera.focal_px)
        in_view = (
            (forward > 1.0)
            & (np.hypot(forward, aside) <= MOUTH_SEEN_M)
            & (np.abs(np.arctan2(aside, np.maximum(forward, 1e-6))) <= half_view)
        )
        if in_view.mean() < 0.5:
            return False
        for track in tracks:
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, _, lateral, _ = estimate
            where = (
                np.array([x, z]) + distance * ahead + (lateral + self.camera.seat_right_m) * right
            )
            if float(np.min(np.hypot(*(mouth - where).T))) <= MOUTH_CLEAR_M:
                return False
        return True

    def _watch_tight_turn(
        self, tracks, progress: float, x: float, z: float, course: float, speed: float, now: float
    ) -> None:
        """Wait short of a tight turn while a vehicle is in it (`TIGHT_CLEAR_M`)."""
        turn = next((t for t in self.tight_turns if t[0] - TIGHT_LOOK_M <= progress <= t[0]), None)
        if turn is None:
            self._tight_wait = None
            return
        start, _, points = turn
        stop_s = start - STOP_BEFORE_TURN_M
        if progress > stop_s + 0.5:
            return  # in it: on, as stopped in a junction is worse
        if self._tight_wait is not None and self._tight_wait[0] != start:
            self._tight_wait = None
        if speed < TIGHT_STANDING_MPS and progress >= stop_s - 3.0:
            if self._tight_wait is None:
                self._tight_wait = (start, now)
            elif now - self._tight_wait[1] >= TIGHT_WAIT_AT_MOST_S:
                return  # waited long enough: it may be waiting for the truck
        ahead = np.array([math.sin(course), math.cos(course)])
        right = np.array([-math.cos(course), math.sin(course)])
        nearest: tuple[int, float] | None = None
        for track in tracks:
            if not track.confirmed:
                continue
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, _, lateral, _ = estimate
            aside = lateral + self.camera.seat_right_m  # from the truck's middle
            where = np.array([x, z]) + distance * ahead + aside * right
            off = float(np.min(np.hypot(*(points - where).T)))
            if off <= TIGHT_CLEAR_M and (nearest is None or distance < nearest[1]):
                nearest = (track.id, distance)
        if nearest is None:
            return
        held = self._give_way
        if held is not None and held.stop_s <= stop_s and held.why != TIGHT_WHY:
            return  # stopping sooner for something else already
        waiting = held is not None and held.why == TIGHT_WHY and held.stop_s == stop_s
        room = stop_s - progress
        if not waiting and (room <= 0.0 or speed * speed / (2.0 * room) > TIGHT_LATE_DECELERATION):
            return  # seen too late to wait short of the turn in comfort
        self._give_way = _GiveWay(stop_s, start, nearest[0], now, TIGHT_WHY)

    def _watch_for_give_way(
        self, tracks, progress: float, speed: float, now: float, oncoming: bool, unseen=()
    ) -> None:
        """Start, or keep, giving way to what calls for it now."""
        watched = next((r for r in self.rings if r[0] - RING_WATCH_M <= progress <= r[0]), None)
        if watched is None:
            self._ring_side_seen.clear()
            self._ring_side_first.clear()
        else:
            start, _, side = watched
            for track in tracks:
                if not track.confirmed:
                    continue
                distance, closing, lateral, _ = track.estimate()
                if distance > RING_LOOK_M or lateral * side < RING_LATERAL_M:
                    continue
                first_at, first_lateral = self._ring_side_first.setdefault(track.id, (now, lateral))
                across = abs(lateral - first_lateral) / max(now - first_at, 1e-6)
                along = abs(speed + closing)
                if (now - first_at > 0.0 and across >= RING_MOVING_MPS) or along >= RING_MOVING_MPS:
                    self._ring_side_seen[track.id] = now
            for track_id, seen in list(self._ring_side_seen.items()):
                if now - seen > RING_MEMORY_S:
                    del self._ring_side_seen[track_id]
            if self._ring_side_seen:
                track_id = max(self._ring_side_seen, key=self._ring_side_seen.__getitem__)
                self._give_way = _GiveWay(
                    start - STOP_BEFORE_TURN_M,
                    start,
                    track_id,
                    now,
                    "giving way to a vehicle coming round the ring",
                )
                return
        ring = next((r for r in self.rings if r[0] - JUNCTION_MARGIN_M <= progress <= r[0]), None)
        if ring is not None:
            start, _, side = ring
            for track_id, middle in unseen:
                if (middle - self.camera.centre_col) * side > RING_SIDE_PX:
                    # On the ring, close, its wheels under the dashboard: no
                    # distance, so no crossing to judge. Near Duisburg a lorry
                    # coming round was seen like this for 3 s, and hit.
                    self._give_way = _GiveWay(
                        start - STOP_BEFORE_TURN_M,
                        start,
                        track_id,
                        now,
                        "giving way to a vehicle on the ring close by",
                    )
                    return
        waiting = self._oncoming_gap(tracks, progress, speed)
        if waiting is not None:
            turn, track_id = waiting
            self._give_way = _GiveWay(
                turn - STOP_BEFORE_TURN_M, turn, track_id, now, "waiting for the oncoming traffic"
            )
            return
        entering = any(start - JUNCTION_MARGIN_M <= progress <= end for start, end, _ in self.rings)
        blocking = self._crossing.in_the_way(
            speed,
            oncoming=oncoming,
            joining_too=entering,
            any_width=not entering,
            steady_only=speed >= CROSSING_BELOW_MPS,
        )
        if blocking is None or not self._in_a_junction(progress + blocking[1]):
            # On an open road a vehicle "crossing" is one in the next lane,
            # swept across by a bend; in France one braked the truck hard,
            # again and again. One that cuts in becomes the lead anyway.
            return
        # Crossing, so not a lead; but across the lane when the truck gets
        # there, so a vehicle to stop short of.
        track_id, distance = blocking
        stop_s = progress + distance - self.policy.standstill_gap_m
        if self._give_way is not None and self._give_way.why.startswith("waiting"):
            stop_s = min(stop_s, self._give_way.stop_s)
        self._give_way = _GiveWay(
            stop_s, progress + distance, track_id, now, "giving way to a vehicle crossing"
        )

    def _giving_way(
        self, progress: float, speed: float, now: float
    ) -> tuple[FollowingDecision, int] | None:
        """The stop the truck is giving way with, and for which vehicle;
        None once the way is clear."""
        held = self._give_way
        if held is None:
            return None
        if now - held.seen_at >= GIVE_WAY_CLEAR_S or progress > held.until_s:
            self._give_way = None
            return None
        room = held.stop_s - progress
        # Come on late, the truck may stop past the place: short of the
        # vehicle crossing by the least gap, not the standstill gap; at the
        # turn, not 2 m short of it.
        slack = (
            self.policy.standstill_gap_m - self.policy.min_gap_m
            if held.why == "giving way to a vehicle crossing"
            else STOP_BEFORE_TURN_M
        )
        decision = stop_short(room, speed, f"{held.why}, {max(room, 0.0):.0f} m on", slack)
        return decision, held.track_id

    def forget_lead(self) -> None:
        """The vehicle followed or remembered is not in the way any more: a
        lane change given up, it was in the lane the truck will not go into.
        Remembered, it held the truck where it stood for good."""
        self._remembered = None
        self._last_lead = None
        self._lead_since = None
        self._standing_since = None
        self._creep_until = None

    def rebase(self, shift: float) -> None:
        """The path changed under the truck, its place on it moved by `shift`
        metres: where along it vehicles were seen moves with it. Kept as it
        was, a lead 50 m ahead was taken for one 18 km off when the route
        to a pump was taken, and lost."""
        if self._remembered is not None:
            distance, then = self._remembered
            self._remembered = (distance, then + shift)
        if self._last_lead is not None:
            track_id, at, own_speed, then = self._last_lead
            self._last_lead = (track_id, at + shift, own_speed, then)
        if self._lead_since is not None:
            track_id, since, place = self._lead_since
            self._lead_since = (track_id, since, place + shift)
        if self._standing_since is not None:
            since, place = self._standing_since
            self._standing_since = (since, place + shift)
        if self._creep_until is not None:
            self._creep_until += shift
        if self._tight_wait is not None:
            self._tight_wait = (self._tight_wait[0] + shift, self._tight_wait[1])
        if self._give_way is not None:
            self._give_way = replace(
                self._give_way,
                stop_s=self._give_way.stop_s + shift,
                until_s=self._give_way.until_s + shift,
            )

    def _held_lead(self, now: float, progress: float) -> tuple[int, float, float] | None:
        """The lead last seen, out of sight now, while it is held
        (`LEAD_HOLD_S`, `CLOSE_HOLD_S`): (its track, how far ahead it would
        be, its speed); None once let go."""
        if self._last_lead is None:
            return None
        track_id, at, own_speed, then = self._last_lead
        lost = now - then
        if lost <= LEAD_HOLD_S:
            return track_id, at + own_speed * lost - progress, own_speed
        place = at + own_speed * LEAD_HOLD_S - progress
        since = self._lead_since
        held_long = (
            since is not None
            and since[0] == track_id
            and then - since[1] >= REMEMBER_AFTER_S
            and at >= since[2] - REMEMBER_STILL_M
        )
        if held_long and place < CLOSE_HOLD_M and not self._bends_before(progress, place):
            if lost <= CLOSE_HOLD_S:
                return track_id, place, 0.0
            self._creep_until = progress + max(place, 0.0) + CREEP_PAST_M
            self._last_lead = None
        return None

    def _following(
        self, found, remembered: float | None, now: float, progress: float, speed: float, age
    ) -> CruiseStep:
        """Following the lead -- seen, remembered close up, or lost a moment ago."""
        if found is None and remembered is not None:
            # Out of view, not gone: stop for it where it was.
            lead = Lead(max(remembered, 0.0), -speed)
            decision = replace(
                follow(lead, speed, self.policy),
                reason=f"stopping for a vehicle out of view, {remembered:.0f} m ahead",
            )
            return CruiseStep(decision, lead, None, age)
        held = self._held_lead(now, progress)
        if held is not None and (found is None or found[1][0] > held[1] + FORGET_BEYOND_M):
            # Lost for a moment, or lost close up: where it would be by now.
            # Not braked hard for: that takes seeing it.
            track_id, distance, own_speed = held
            lead = Lead(distance, own_speed - speed)
            decision = replace(
                follow(lead, speed, self.policy),
                urgent_deceleration=None,
                reason=f"following a vehicle out of sight a moment, {lead.distance_m:.0f} m",
            )
            return CruiseStep(decision, lead, track_id, age)
        if found is None and self._creep_until is not None and progress < self._creep_until:
            decision = FollowingDecision(
                CREEP_PAST_MPS, None, "creeping on where a vehicle was lost close up"
            )
            return CruiseStep(decision, None, None, age)
        self._creep_until = None
        if found is None:
            self._last_lead = None
            self._standing_since = None
            return CruiseStep(follow(None, speed, self.policy), None, None, age)
        track, (distance, closing, _, _) = found
        towards = -(speed + closing)  # its own speed towards the truck
        if towards >= ONCOMING_OWN_MPS:
            # Coming the other way: not followed, nor remembered standing,
            # nor held when it goes by out of view.
            self._last_lead = None
            self._remembered = None
            self._standing_since = None
            return CruiseStep(
                FollowingDecision(None, None, "oncoming: not a lead"), None, None, age
            )
        if 0.0 < towards < ONCOMING_OWN_MPS and not self._in_a_junction(progress, ahead=distance):
            # In the truck's lane on the open road nothing comes towards it:
            # at most it stands. With another truck's camera a DAF misjudged
            # distances by half again, cars stopping at red lights seemed to
            # come at 17 km/h, were let go as oncoming (core.control.following),
            # and were braked for at 6 to 9 m/s^2 from 18 to 30 m (Duisburg,
            # 15 November). Through a junction one coming the other way is
            # still not followed.
            closing = -speed
        standing = speed + closing < REMEMBER_BELOW_MPS
        if not standing:
            self._remembered = None
            self._standing_since = None
        else:
            place = progress + distance
            if (
                self._standing_since is None
                or abs(place - self._standing_since[1]) > REMEMBER_STILL_M
            ):
                self._standing_since = (now, place)
            if (
                distance < REMEMBER_WITHIN_M
                and now - self._standing_since[0] >= REMEMBER_AFTER_S
                and not self._bends_before(progress, distance)
            ):
                self._remembered = (distance, progress)
                self._remembered_at = now
        own_speed = self._smoothed_speed(track.id, speed + closing, now)
        self._last_lead = (track.id, progress + distance, own_speed, now)
        if self._lead_since is None or self._lead_since[0] != track.id:
            self._lead_since = (track.id, now, progress + distance)
        decision = follow(Lead(distance, own_speed - speed), speed, self.policy)
        # The speed to aim for comes from the smoothed estimate; a hard stop
        # comes from the raw one alone. Taking it from the smoothed one too,
        # a filter still carrying a bad reading asked for 3.9 m/s^2 in the
        # game while the raw estimate asked for nothing.
        urgent = follow(Lead(distance, closing), speed, self.policy).urgent_deceleration
        decision = replace(decision, urgent_deceleration=urgent)
        return CruiseStep(decision, Lead(distance, closing), track.id, age)

    def _oncoming_gap(self, tracks, progress: float, speed: float) -> tuple[float, int] | None:
        """Short of a turn across the oncoming lanes, where along the path
        it begins and for which vehicle to wait, while one coming the other
        way would be at the turn within `ONCOMING_GAP_S`; None if the way is
        clear, or the truck is in the turn or not near one."""
        turn = next(
            (start for start, _ in self.turns_across if 0.0 <= start - progress <= TURN_APPROACH_M),
            None,
        )
        if turn is None:
            return None
        for track in tracks:
            if not track.confirmed or track.measurement_age > STALE_MEASUREMENT_S:
                continue
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, closing, _, _ = estimate
            towards = -(speed + closing)
            if distance > ONCOMING_LOOK_M or towards < MIN_ALONG_MPS:
                continue
            if distance / max(-closing, 0.1) <= ONCOMING_GAP_S:
                return turn, track.id
        return None

    def in_a_junction(self, s: float) -> bool:
        """Whether `s` is in a junction, or in a turn across the oncoming lanes."""
        return self._in_a_junction(s) or any(
            start - STOP_BEFORE_TURN_M <= s <= end for start, end in self.turns_across
        )

    def _near_a_ring(self, s: float) -> bool:
        """Whether `s` is in a roundabout, or within `CROSSING_WITHIN_M` of one."""
        return any(start - CROSSING_WITHIN_M <= s <= end for start, end, _ in self.rings)

    def _bends_before(self, progress: float, distance: float) -> bool:
        """Whether the path turns by `OUT_OF_VIEW_BEND_RAD` or more between
        the truck and `distance` metres on (and `OUT_OF_VIEW_BEND_PAST_M`
        past): a vehicle there goes out of the camera's view round the bend,
        not out of the way."""
        path = getattr(self, "_path", None)
        if path is None:
            return False
        here = path.sample_at(max(progress, 0.0)).heading
        for along in np.arange(2.0, max(distance, 0.0) + OUT_OF_VIEW_BEND_PAST_M, 2.0):
            there = path.sample_at(min(progress + float(along), path.length)).heading
            if abs(math.remainder(there - here, 2.0 * math.pi)) >= OUT_OF_VIEW_BEND_RAD:
                return True
        return False

    def _in_a_junction(self, s: float, ahead: float = 0.0) -> bool:
        """Whether `s`, or `ahead` metres past it, is where the route crosses
        another way (`junction_spans`)."""
        return any(
            start - JUNCTION_MARGIN_M <= s + ahead and s <= end for start, end in self.junctions
        )

    def _smoothed_speed(self, track_id: int, raw: float, now: float) -> float:
        """The lead's own speed, first-order filtered while it stays the same lead."""
        if self._lead_speed is None or self._lead_speed[0] != track_id:
            self._lead_speed = (track_id, raw, now)
            return raw
        _, smooth, then = self._lead_speed
        smooth += (raw - smooth) * min(1.0, max(now - then, 0.0) / LEAD_SPEED_TAU_S)
        self._lead_speed = (track_id, smooth, now)
        return smooth

    def _debounce(self, decision, now: float):
        if decision.urgent_deceleration is None:
            self._urgent_since = None
        else:
            if self._urgent_since is None:
                self._urgent_since = now
            if now - self._urgent_since < URGENT_AFTER_S:
                decision = replace(decision, urgent_deceleration=None, reason="urgent? waiting")
        return decision
