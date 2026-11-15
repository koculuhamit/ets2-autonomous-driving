"""Junction turns too tight for the rig to keep to its lane: driven deeper.

Into the depot road at Duisburg the route turns right on a 9 m radius. A
trailer with 8.85 m from its kingpin to its axles cannot follow its tractor
round that: on the lane's middle it ran 1.4 m over the kerb on the inside;
swung wide, as `widen_for_trailer` swings every turn, the cab's outside
corner went 1.2 m over the splitter island at the mouth of the road, and hit
its sign -- twice. A driver takes such a turn by going on into the junction
first and turning late and tight, over the junction's own ground, which the
lane's middle line keeps out of.

So a turn whose radius would have the trailer cut inside by `TIGHT_CUT_M` or
more (`tight_turns`) is driven along another line (`shaped_turn`): straight
on into the junction for some metres, a Dubins turn at some radius, onto the
exit lane some metres in. Each line of that family is driven in simulation
-- the rig of the 8x4 driven, a `DESIGN_TRAILER_M` trailer, the pure pursuit
the truck steers by -- over the ground lanes cover (each lane's middle
+-`HALF_LANE_M`, which leaves the islands between them out): the junction's
own, whichever way they go, and beyond it only those going the way the turn
goes. The line on which the cab and the trailer stray least off it is taken,
the least deep of those as good. A line no better than the lane's own is
not, nor is a roundabout's.

What the map knows of the ground is its lanes, not its kerbs: at Duisburg's
depot road, with oncoming lanes no ground, the best line strays 0.55 m to
the widened line's 0.75 -- not enough to be taken, and that turn is still
the driver's to watch. Over the 9 tight turns between Duisburg and Hamburg 3
were shaped: 1.5 m to 0.75, 0.7 to 0.25, 0.75 to 0.25.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core.control.acc import _on_a_ring
from core.control.lateral import ACTROS_GEOMETRY, PurePursuit
from core.control.steering import STEERING_CALIBRATION, SteeringPositionController
from core.localization.path import ReferencePath
from core.planning.dubins import dubins_path
from core.planning.route import CHANGE, ONWARD

__all__ = [
    "DESIGN_TRAILER_M",
    "TIGHT_CUT_M",
    "Stretch",
    "Turn",
    "between",
    "kept_in",
    "shaped_turn",
    "splice",
    "stretches",
    "tight_turn_spans",
    "tight_turns",
]

DESIGN_TRAILER_M: Final = 9.0
"""Kingpin to axles of the trailer a turn is shaped for, before the trailer
is known: 8.85 m on the car carrier into Duisburg's depot road, 6.2 m on the
tanker; a line good for the longer was good for the shorter."""

TIGHT_CUT_M: Final = 1.0
"""A turn on which the trailer would cut inside by this much is tight."""

HALF_LANE_M: Final = 2.0
CELL_M: Final = 0.25
GROUND_MARGIN_M: Final = 30.0
"""The ground is mapped this far about the stretch shaped."""

SAME_WAY_SLACK_RAD: Final = math.radians(45.0)
"""Ground is the lanes going the way the turn goes, from the way in round to
the way out, this much either side: never a lane coming the other way, nor
the gap between it and the route's -- out of Duisburg the search, with
every lane for ground, cut a corner over both to the road beyond."""

AGAINST_EXIT_RAD: Final = math.radians(45.0)
AGAINST_CLEAR_M: Final = 0.0
AGAINST_NEAR_EXIT_M: Final = 12.0
"""Lanes coming out of the road turned into are those within this of the
way out: by heading alone, near Verona, on a route that bent left before
it turned right, the way in's own oncoming lane beside the truck counted
among them, every line went over it in simulation before the turn began,
the line was pulled in its furthest for it, and the trailer cut over the
pavement into a bollard."""
"""Lanes going within `AGAINST_EXIT_RAD` of the way opposite the way out
come out of the road turned into, and their traffic waits at its mouth:
they are not ground, even the junction's, and a line that takes the cab or
the trailer over them, nearer than `AGAINST_CLEAR_M` short of their side,
is not driven. At a right turn in Norway, at night, the line went on into
the junction for its whole 9 m first and swung over the mouth of the
oncoming lane, where a lorry stood waiting, and hit it. Off the ground by
1.25 m, the cab over that lane counted for no more than the trailer over a
kerb. It was 0.5 m into their lane, let be, until on the way to Bergen,
twice at one right turn as tight as a hairpin, the cab came up to the mouth
where a coach, then a lorry, stood waiting: hit by a tanker squeezing by
as it stood, then hitting the lorry. Not into their lane at all now (the
driver's rule, 11 November); see `kept_in`."""

INWARD_M: Final = (0.0, 0.25, 0.5, 0.75, 1.0)
INWARD_MORE_M: Final = 0.5
INWARD_EASE_M: Final = 25.0
"""A turn to the exit side whose line takes the rig over the lanes coming
out of the road turned into, and no shaped line keeping off them, is driven
on the lane's own line pulled towards the inside of the turn
(`kept_in`): by the least of `INWARD_M` that keeps off them in simulation,
and `INWARD_MORE_M` more for the truck running wider than simulated (0.5 to
0.9 m at Bergen), eased in and out over `INWARD_EASE_M`. The trailer cuts
the inside corner more for it: the driver's choice, over the other way's
lane. Out of 40 such turns on recorded routes, 22 had their line over it."""

DEEPER_CLEAR_M: Final = 0.5
"""A line into the junction deeper and round keeps this much further off
the lanes coming out of the road turned into than in simulation: as
`INWARD_MORE_M`, for the truck running wider than simulated. Near Verona
the line kept in strayed 2.5 m over the inside kerb in simulation, the
trailer 2.9 m in the game, into a bollard on the pavement; 6 m deeper, it
strays 1.1 m."""

OVER_AGAINST_M: Final = 1000.0
"""How far off the ground the lanes coming out of the road turned into are
taken to be: no line straying so far is driven."""

APPROACH_M: Final = 60.0
EXIT_M: Final = 45.0
"""Of the way in and out, how much is shaped and driven in simulation."""

DEEPER_M: Final = (0.0, 3.0, 6.0, 9.0)
RADII_M: Final = (7.0, 9.0, 11.0, 13.0)
ONTO_EXIT_M: Final = (8.0, 14.0)
"""The family of lines: how far on into the junction before turning, at
what radius, how far into the exit lane to be on it."""

GOOD_ENOUGH_M: Final = 0.3
"""Lines straying no more than this off the ground are as good as each
other: of those, the least deep, at the widest radius."""

BETTER_BY_M: Final = 0.2
"""A shaped line is driven only if it strays this much less than the lane's."""

SPEED_MPS: Final = 4.5

MAX_APPLIED_STEERING: Final = 0.77
"""The most of the wheel the game turns, asked for full lock: 0.771 at
15-20 km/h, in every drive recorded on Windows. Simulated to full lock, a
line of 7 m radius was found good north of Zurich, and the truck, its
wheel at 0.77 on a 7.7 m circle, ran 2-3 m wide of it and over the
junction."""

# The 8x4 driven: driven axles 2.87 m behind its origin, 3.34 m from the
# steered ones; the fifth wheel 0.17 m ahead of them.
AXLE_BEHIND_M: Final = 2.87
WHEELBASE_M: Final = 3.34
HITCH_AHEAD_M: Final = 0.17
CAB_AHEAD_OF_ORIGIN_M: Final = 1.5
CAB_BEHIND_AXLE_M: Final = 1.0
CAB_HALF_M: Final = 1.25
TRAILER_AHEAD_M: Final = 1.3
TRAILER_BEYOND_AXLES_M: Final = 3.5
TRAILER_HALF_M: Final = 1.3


@dataclass(frozen=True, slots=True)
class Turn:
    """Junction lanes of a route a turn is made on."""

    first: int
    last: int
    """Indices into `route.lanes`, both included."""


def tight_turns(route, trailer_length: float = DESIGN_TRAILER_M) -> list[Turn]:
    """The route's junction turns tight for a trailer this long."""
    network = route.network
    turns: list[Turn] = []
    k = 0
    lanes = route.lanes
    while k < len(lanes):
        if int(network.lane_source[lanes[k]]) != 1:
            k += 1
            continue
        first = k
        while k + 1 < len(lanes) and int(network.lane_source[lanes[k + 1]]) == 1:
            k += 1
        points = np.vstack(
            [np.asarray(network.lane_points(lanes[j]))[:, [0, 2]] for j in range(first, k + 1)]
        )
        radius = _tightest_radius(points)
        cut = radius - math.sqrt(max(radius * radius - trailer_length * trailer_length, 0.0))
        if radius < trailer_length or cut >= TIGHT_CUT_M:
            turns.append(Turn(first, k))
        k += 1
    return turns


@dataclass(frozen=True, slots=True)
class Stretch:
    """The route about a tight turn: the way in, the way out, and the lanes
    they run on, by which a line shaped for it is known again on another
    route."""

    key: tuple[int, ...]
    begins: int
    """Index into `route.lanes` of the first lane of the way in."""
    approach: NDArray[np.float64]
    exit_: NDArray[np.float64]
    junction: tuple[int, ...] = ()
    """The lanes of the turn itself."""


def stretches(route, trailer_length: float = DESIGN_TRAILER_M) -> list[Stretch]:
    """The route's tight turns, each with `APPROACH_M` and `EXIT_M` (and some)
    of the lanes leading in and out; a turn without both, or with a lane
    change on them, is left as it is."""
    network = route.network
    lanes, steps = route.lanes, route.steps
    lengths = network.lane_length

    def line(first: int, end: int) -> NDArray[np.float64]:
        return np.vstack(
            [np.asarray(network.lane_points(lanes[j]), float)[:, [0, 2]] for j in range(first, end)]
        )

    found: list[Stretch] = []
    for turn in tight_turns(route, trailer_length):
        if any(steps[j] != ONWARD for j in range(turn.first, turn.last)):
            continue
        if any(_on_a_ring(network, int(lanes[j])) for j in range(turn.first, turn.last + 1)):
            continue  # a roundabout: driven round, as its lanes go
        begins, run = turn.first, 0.0
        while begins > 0 and steps[begins - 1] == ONWARD and run < APPROACH_M + 20.0:
            begins -= 1
            run += float(lengths[lanes[begins]])
        ends, run = turn.last, 0.0
        while ends + 1 < len(lanes) and steps[ends] == ONWARD and run < EXIT_M + 20.0:
            ends += 1
            run += float(lengths[lanes[ends]])
        if begins == turn.first or ends == turn.last:
            continue
        found.append(
            Stretch(
                key=tuple(int(lane) for lane in lanes[begins : ends + 1]),
                begins=begins,
                approach=_tail(line(begins, turn.first), APPROACH_M + 20.0),
                exit_=_head(line(turn.last + 1, ends + 1), EXIT_M + 20.0),
                junction=tuple(int(lane) for lane in lanes[turn.first : turn.last + 1]),
            )
        )
    return found


def tight_turn_spans(
    route, path, trailer_length: float = DESIGN_TRAILER_M
) -> list[tuple[float, float, NDArray[np.float64]]]:
    """The route's tight turns along `path`: (start, end) metres along it,
    and the `(n, 2)` world (x, z) of the turn's lanes, for the truck to wait
    short of a turn while something is in it
    (core.control.acc.AdaptiveCruise.tight_turns)."""
    network = route.network
    spans = []
    # Where each lane begins along the route, to look for its turn near:
    # looked for from the last turn within 3 km, 17 of 70 turns on recorded
    # routes were placed on another part of the path, km off.
    along = [0.0]
    for index, lane in enumerate(route.lanes[:-1]):
        onward = route.steps[index] != CHANGE
        along.append(along[-1] + (float(network.lane_length[lane]) if onward else 0.0))
    for turn in tight_turns(route, trailer_length):
        lanes = route.lanes[turn.first : turn.last + 1]
        if any(_on_a_ring(network, int(lane)) for lane in lanes):
            continue
        points = np.vstack(
            [np.asarray(network.lane_points(lane), float)[:, [0, 2]] for lane in lanes]
        )
        whole = points
        points = _tight_part(points, SLOW_RADIUS_M)
        if points is None:
            continue
        into = int(np.argmin(np.hypot(*(whole - points[0]).T)))
        ahead = float(np.sum(np.hypot(*np.diff(whole[: into + 1], axis=0).T)))
        first = path.nearest(
            float(points[0][0]), float(points[0][1]), near=along[turn.first] + ahead, window=150.0
        )
        start = first.s
        # As long as the bend itself: its end looked for on the path was, by
        # Dover, where the route came back past it 500 m on, and the truck
        # crept at 8 km/h all that way.
        end = start + float(np.sum(np.hypot(*np.diff(points, axis=0).T)))
        spans.append((start, end, points))
    return spans


SLOW_RADIUS_M: Final = 12.0
"""Only a bend on a radius under this -- a hairpin, as by Bergen at 9.5 m
-- is a turn to wait short of and creep round; a slip road's loop at 16 m
the speed profile slows for as it bends."""

TIGHT_PART_MARGIN_M: Final = 10.0
"""Of a tight turn's junction lanes, only the part bending tighter than the
trailer keeps to, and this much either side, is the turn: a junction's lanes
can run on for hundreds of metres. Near Calais a 644 m run of them, with one
tight bend, was all of it slowed to 8 km/h, on the straight."""


def _tight_part(points: NDArray[np.float64], radius: float) -> NDArray[np.float64] | None:
    """The points of a polyline about where it bends on a radius under
    `radius` (over 5 m), `TIGHT_PART_MARGIN_M` either side; None if nowhere."""
    steps = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] < 5.0:
        return points
    samples = np.arange(0.0, arc[-1], 1.0)
    xs = np.interp(samples, arc, points[:, 0])
    zs = np.interp(samples, arc, points[:, 1])
    headings = np.unwrap(np.arctan2(np.diff(xs), np.diff(zs)))
    if len(headings) < 6:
        return points
    turn = np.abs(headings[5:] - headings[:-5])
    tight = np.flatnonzero(turn >= 5.0 / radius)
    if len(tight) == 0:
        return None
    begins = max(float(samples[tight[0]]) - TIGHT_PART_MARGIN_M, 0.0)
    ends = min(float(samples[tight[-1]]) + 5.0 + TIGHT_PART_MARGIN_M, float(arc[-1]))
    keep = (arc >= begins) & (arc <= ends)
    return points[keep] if keep.sum() >= 2 else points


def between(points: NDArray[np.floating], start, end) -> tuple[int, int]:
    """Indices of the points of a line nearest `start`, and after it `end`."""
    points = np.asarray(points, float)
    first = int(np.argmin(np.hypot(*(points - np.asarray(start, float)).T)))
    after = points[first:]
    return first, first + int(np.argmin(np.hypot(*(after - np.asarray(end, float)).T)))


def splice(points: NDArray[np.floating], line: NDArray[np.floating]) -> NDArray[np.float64]:
    """`points`, a line driven, with `line` in place of its stretch."""
    first, last = between(points, line[0], line[-1])
    points = np.asarray(points, float)
    return np.vstack([points[:first], np.asarray(line, float), points[last + 1 :]])


def shaped_turn(
    network,
    approach: NDArray[np.floating],
    exit_: NDArray[np.floating],
    lane_line: NDArray[np.floating],
    junction: tuple[int, ...] = (),
    *,
    trailer_length: float = DESIGN_TRAILER_M,
) -> NDArray[np.float64] | None:
    """`(n, 2)` world (x, z) of a line from `APPROACH_M` before the junction
    to `EXIT_M` after it, driven instead of `lane_line` -- the route's own
    line over the same stretch; None where none does better. `approach`
    ends where the junction begins, `exit_` begins where it ends; `junction`
    is the turn's lanes, whose junction is ground all over."""
    approach = _tail(np.asarray(approach, float), APPROACH_M + 20.0)
    exit_ = _head(np.asarray(exit_, float), EXIT_M + 20.0)
    if len(approach) < 3 or len(exit_) < 3:
        return None
    outside, origin = _ground(
        network, np.vstack([approach, exit_]), lane_line, junction, exit_=exit_
    )
    own = _stray(
        ReferencePath(np.asarray(lane_line, float), spacing=1.0),
        approach,
        outside,
        origin,
        trailer_length,
    )

    def clear():
        """The ground for a line into the junction deeper: their lanes
        `DEEPER_CLEAR_M` wider."""
        return _ground(
            network,
            np.vstack([approach, exit_]),
            lane_line,
            junction,
            exit_=exit_,
            clear=DEEPER_CLEAR_M,
        )[0]

    if own >= OVER_AGAINST_M:
        # Over the lanes out of the road turned into: on a turn across them
        # that is the way, and it is left as it is; on one to the exit side,
        # pulled in to keep off them.
        if not _to_exit_side(network, lane_line, junction):
            return None
        kept = kept_in(network, approach, exit_, junction, outside, origin, trailer_length)
        # Deeper into the junction may keep the trailer further off the
        # kerb: as every line into it deeper, off their lanes by
        # `DEEPER_CLEAR_M` more, as `kept_in` keeps off them by `INWARD_MORE_M`.
        kept_stray = _stray(
            ReferencePath(kept, spacing=1.0), approach, outside, origin, trailer_length
        )
        if kept_stray - BETTER_BY_M < GOOD_ENOUGH_M:
            return kept
        deeper = _deepest_better(approach, exit_, clear(), origin, trailer_length, kept_stray)
        return kept if deeper is None else deeper
    if own - BETTER_BY_M < GOOD_ENOUGH_M:
        return None  # good enough
    return _deepest_better(approach, exit_, clear(), origin, trailer_length, own)


def _deepest_better(approach, exit_, outside, origin, trailer_length: float, than: float):
    """The best of the lines into the junction deeper and round
    (`DEEPER_M`, `RADII_M`, `ONTO_EXIT_M`): the one straying least, and of
    those as good the least deep, if it strays `BETTER_BY_M` less than
    `than`; None if none does."""
    best: tuple[float, float, float, NDArray[np.float64]] | None = None
    for deeper, radius, onto in itertools.product(DEEPER_M, RADII_M, ONTO_EXIT_M):
        line = _line(approach, exit_, deeper, radius, onto)
        if line is None:
            continue
        # A line straying further than the best so far, or than would beat
        # the lane's own, is given up on as soon as it does.
        bound = min(than - BETTER_BY_M, math.inf if best is None else best[0])
        stray = _stray(
            ReferencePath(line, spacing=1.0),
            approach,
            outside,
            origin,
            trailer_length,
            give_up=bound,
        )
        rank = (max(stray, GOOD_ENOUGH_M), deeper, -radius)
        if stray <= bound and (best is None or rank < best[:3]):
            best = (*rank, line)
    return None if best is None else best[3]


def _to_exit_side(network, line, junction: tuple[int, ...]) -> bool:
    """Whether `line` turns to the exit side: right, or left where traffic
    keeps left -- across no lane coming the other way."""
    turning = _headings(np.asarray(line, float))
    total = float(turning[-1] - turning[0])  # positive is to the left
    return total > 0.0 if _keeps_left(network, line) else total < 0.0


def _keeps_left(network, line) -> bool:
    """Whether traffic keeps left where `line` begins: by the roads' lanes
    there -- a junction's are never marked. Asked of a junction's lane, every
    turn in Britain was one to the right-hand side: a right turn across the
    oncoming lanes by Birmingham was pulled in (`kept_in`), and its line
    ran over the pavement."""
    if network.lane_left_hand is None:
        return False
    x, z = (float(v) for v in np.asarray(line, float)[0])
    roads = [
        int(lane)
        for lane in network.lanes_near(x, z, 30.0)
        if int(network.lane_source[int(lane)]) == 0
    ]
    if not roads:
        return False
    return float(np.mean([bool(network.lane_left_hand[lane]) for lane in roads])) > 0.5


def kept_in(
    network,
    approach: NDArray[np.floating],
    exit_: NDArray[np.floating],
    junction: tuple[int, ...],
    outside,
    origin,
    trailer_length: float = DESIGN_TRAILER_M,
) -> NDArray[np.float64]:
    """The lane's own line through a turn to the exit side, pulled towards
    the inside of the turn as far as keeps the rig off the lanes coming out
    of the road turned into (`INWARD_M`), and `INWARD_MORE_M` more; pulled
    the furthest if none does."""
    lanes = [np.asarray(network.lane_points(lane), float)[:, [0, 2]] for lane in junction]
    own = np.vstack([approach, *lanes, exit_])
    begins = float(np.sum(np.hypot(*np.diff(approach, axis=0).T)))
    through = float(sum(np.sum(np.hypot(*np.diff(lane, axis=0).T)) for lane in lanes))
    chosen = INWARD_M[-1]
    for inward in INWARD_M:
        line = _pulled_in(own, inward, begins, begins + through)
        stray = _stray(ReferencePath(line, spacing=1.0), approach, outside, origin, trailer_length)
        if stray < OVER_AGAINST_M:
            chosen = min(inward + INWARD_MORE_M, INWARD_M[-1])
            break
    return _pulled_in(own, chosen, begins, begins + through)


def _pulled_in(line, metres: float, begins: float, ends: float) -> NDArray[np.float64]:
    """`line` moved `metres` towards the inside of its turn between `begins`
    and `ends` metres along it, eased in and out over `INWARD_EASE_M`."""
    line = _even(np.asarray(line, float))
    if metres <= 0.0:
        return line
    # Each point's way from the line PULL_HEADING_M either side: from its
    # neighbours alone, where lanes join end to end and points lie on top of
    # one another, the way swung, the line pulled in hooked back on itself,
    # and following it the truck put full lock on and went over a pavement.
    count = len(line)
    reach = max(1, round(PULL_HEADING_M))
    ahead = line[np.minimum(np.arange(count) + reach, count - 1)]
    behind = line[np.maximum(np.arange(count) - reach, 0)]
    way = ahead - behind
    heading = np.arctan2(way[:, 0], way[:, 1])
    turning = np.unwrap(heading)
    inside = 1.0 if turning[-1] - turning[0] > 0.0 else -1.0  # left, or right
    arc = np.arange(count, dtype=float)
    weight = np.clip(
        np.minimum((arc - begins + INWARD_EASE_M), (ends + INWARD_EASE_M - arc)) / INWARD_EASE_M,
        0.0,
        1.0,
    )
    weight = weight * weight * (3.0 - 2.0 * weight)
    left = np.column_stack([np.cos(heading), -np.sin(heading)])
    return line + inside * metres * weight[:, None] * left


PULL_HEADING_M: Final = 4.0


def _even(line: NDArray[np.float64]) -> NDArray[np.float64]:
    """`line` as points 1 m apart along it."""
    keep = np.concatenate([[True], np.hypot(*np.diff(line, axis=0).T) > 1e-6])
    line = line[keep]
    steps = np.hypot(*np.diff(line, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    at = np.append(np.arange(0.0, arc[-1], 1.0), arc[-1])
    return np.column_stack([np.interp(at, arc, line[:, 0]), np.interp(at, arc, line[:, 1])])


def _line(approach, exit_, deeper: float, radius: float, onto: float):
    """Approach, on `deeper` metres, a Dubins turn at `radius` onto the exit
    `onto` metres in, and on along the exit."""
    join = approach[-1]
    heading = math.atan2(*(approach[-1] - approach[-3]))
    ahead = np.array([math.sin(heading), math.cos(heading)])
    turn_at = join + deeper * ahead
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(exit_, axis=0).T))])
    k = min(int(np.searchsorted(arc, onto)), len(exit_) - 2)
    end = exit_[k]
    end_heading = math.atan2(*(exit_[k + 1] - exit_[k]))
    way = dubins_path(
        (float(turn_at[0]), float(turn_at[1]), heading),
        (float(end[0]), float(end[1]), end_heading),
        radius,
        step=0.5,
    )
    if not way:
        return None
    straight = [join + t * ahead for t in np.arange(1.0, deeper, 1.0)]
    pieces = [
        approach,
        *([np.array(straight)] if straight else []),
        np.array([w[:2] for w in way]),
        exit_[k + 1 :],
    ]
    return np.vstack(pieces)


def _stray(
    path: ReferencePath,
    approach,
    outside,
    origin,
    trailer_length: float,
    *,
    give_up: float = math.inf,
) -> float:
    """How far, at worst, the cab or the trailer driven along `path` goes off
    the ground, metres; once more than `give_up`, that much."""
    geometry = replace(ACTROS_GEOMETRY, wheelbase=WHEELBASE_M, axle_behind=AXLE_BEHIND_M)
    controller = PurePursuit(geometry, lookahead_time=1.2)
    inner = SteeringPositionController()
    start = path.nearest(float(approach[0][0]), float(approach[0][1]), window=1e9)
    x, z, heading = start.x, start.z, start.heading
    beta = applied = 0.0
    progress = None
    dt = 1.0 / 30.0
    worst = 0.0
    for _ in range(int(60.0 / dt)):
        ox, oz = x + AXLE_BEHIND_M * math.sin(heading), z + AXLE_BEHIND_M * math.cos(heading)
        command = controller.step(
            x=ox, z=oz, heading=heading, speed=SPEED_MPS, path=path, progress=progress
        )
        progress = command.progress
        if command.at_end:
            break
        rate = inner.command(target=command.steering, measured=applied, speed=SPEED_MPS)
        applied += STEERING_CALIBRATION.sign * STEERING_CALIBRATION.rate_gain * rate * dt
        applied = max(-MAX_APPLIED_STEERING, min(MAX_APPLIED_STEERING, applied))
        wheel = geometry.angle_for_command(applied)
        turn = SPEED_MPS / WHEELBASE_M * math.tan(wheel) * dt
        beta += turn - SPEED_MPS * math.sin(beta) / trailer_length * dt
        heading += turn
        x += SPEED_MPS * math.sin(heading) * dt
        z += SPEED_MPS * math.cos(heading) * dt
        hx, hz = x + HITCH_AHEAD_M * math.sin(heading), z + HITCH_AHEAD_M * math.cos(heading)
        cab = _box(
            x, z, heading, AXLE_BEHIND_M + CAB_AHEAD_OF_ORIGIN_M, CAB_BEHIND_AXLE_M, CAB_HALF_M
        )
        trailer = _box(
            hx,
            hz,
            heading - beta,
            TRAILER_AHEAD_M,
            trailer_length + TRAILER_BEYOND_AXLES_M,
            TRAILER_HALF_M,
        )
        worst = max(worst, _off(outside, origin, np.vstack([cab, trailer])))
        if worst > give_up:
            break
    return worst


def _box(x: float, z: float, heading: float, ahead: float, behind: float, half: float):
    """Points along both sides of a body, world (x, z)."""
    along, sides = _BOX_ALONG * (ahead + behind) - behind, _BOX_SIDES * half
    sin, cos = math.sin(heading), math.cos(heading)
    return np.column_stack([x + along * sin + sides * cos, z + along * cos - sides * sin])


_BOX_ALONG = np.repeat(np.linspace(0.0, 1.0, 7), 2)
_BOX_SIDES = np.tile([-1.0, 1.0], 7)


def _ground(
    network, about, line, junction: tuple[int, ...] = (), *, exit_=None, clear: float = 0.0
) -> tuple[NDArray[np.float32], tuple[float, float]]:
    """How far each cell of the ground about the points `about` is from the
    lanes': those of the junction of the lanes `junction`, but for the ones
    coming out of the road turned into (`AGAINST_EXIT_RAD`), and elsewhere
    those going the ways `line` goes, the route's own through the turn.
    Those coming out of the road turned into are `OVER_AGAINST_M` off: with
    `exit_`, the way out, only within `AGAINST_NEAR_EXIT_M` of it; and
    `clear` metres wider than their lane."""
    near_exit = None
    if exit_ is not None and len(exit_):
        near_exit = np.asarray(exit_, float)
    low = np.asarray(about, float).min(axis=0) - GROUND_MARGIN_M
    high = np.asarray(about, float).max(axis=0) + GROUND_MARGIN_M
    origin = (float(low[0]), float(low[1]))
    width, height = (np.ceil((high - low) / CELL_M)).astype(int)
    free = np.zeros((height, width), np.uint8)
    barred = np.zeros((height, width), np.uint8)
    narrow = max(1, round(2.0 * (HALF_LANE_M - AGAINST_CLEAR_M + clear) / CELL_M))
    hidden = network.lane_hidden
    centre = (low + high) / 2.0
    reach = float(np.hypot(*(high - low))) / 2.0 + 30.0
    turning = _headings(np.asarray(line, float))
    least = float(turning.min()) - SAME_WAY_SLACK_RAD
    sweep = float(turning.max()) + SAME_WAY_SLACK_RAD - least
    against = float(turning[-1]) + math.pi
    thickness = max(1, round(2.0 * HALF_LANE_M / CELL_M))
    owners = network.lane_owner
    box = {owners[int(lane)] for lane in junction}
    for lane in network.lanes_near(float(centre[0]), float(centre[1]), reach):
        lane = int(lane)
        if hidden is not None and bool(hidden[lane]):
            continue
        points = np.asarray(network.lane_points(lane))[:, [0, 2]]
        if len(points) < 2:
            continue
        steps = np.diff(points, axis=0)
        going = np.arctan2(steps[:, 0], steps[:, 1])
        coming_out = np.abs(np.mod(going - against + np.pi, 2.0 * np.pi) - np.pi)
        out_of_exit = coming_out <= AGAINST_EXIT_RAD
        if near_exit is not None and out_of_exit.any():
            middles = (points[:-1] + points[1:]) / 2.0
            gaps = np.hypot(
                middles[:, None, 0] - near_exit[None, :, 0],
                middles[:, None, 1] - near_exit[None, :, 1],
            ).min(axis=1)
            out_of_exit &= gaps <= AGAINST_NEAR_EXIT_M
        same_way = (np.mod(going - least, 2.0 * np.pi) <= sweep) | (owners[lane] in box)
        same_way &= ~out_of_exit
        cells = np.round((points - origin) / CELL_M).astype(np.int32)
        for run in np.split(np.arange(len(steps)), np.flatnonzero(np.diff(same_way)) + 1):
            if same_way[run[0]]:
                cv2.polylines(free, [cells[run[0] : run[-1] + 2]], False, 1, thickness)
        for run in np.split(np.arange(len(steps)), np.flatnonzero(np.diff(out_of_exit)) + 1):
            if out_of_exit[run[0]]:
                cv2.polylines(barred, [cells[run[0] : run[-1] + 2]], False, 1, narrow)
    outside = cv2.distanceTransform((1 - free).astype(np.uint8), cv2.DIST_L2, 5) * CELL_M
    outside[barred > 0] = OVER_AGAINST_M
    return outside, origin


def _headings(line) -> NDArray[np.float64]:
    """Headings along a line, unwrapped: radians, as far round as it turns."""
    steps = np.diff(line, axis=0)
    steps = steps[np.hypot(*steps.T) > 0.05]
    return np.unwrap(np.arctan2(steps[:, 0], steps[:, 1]))


def _off(outside, origin, points) -> float:
    cells = np.round((points - origin) / CELL_M).astype(int)
    height, width = outside.shape
    cells[:, 0] = cells[:, 0].clip(0, width - 1)
    cells[:, 1] = cells[:, 1].clip(0, height - 1)
    return float(outside[cells[:, 1], cells[:, 0]].max())


def _tightest_radius(points) -> float:
    """The smallest radius over 5 m of a polyline, metres."""
    steps = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] < 5.0:
        return math.inf
    samples = np.arange(0.0, arc[-1], 1.0)
    xs = np.interp(samples, arc, points[:, 0])
    zs = np.interp(samples, arc, points[:, 1])
    headings = np.unwrap(np.arctan2(np.diff(xs), np.diff(zs)))
    if len(headings) < 6:
        return math.inf
    turn = np.abs(headings[5:] - headings[:-5])
    largest = float(turn.max()) if len(turn) else 0.0
    return math.inf if largest <= 1e-6 else 5.0 / largest


def _tail(points, length: float):
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    return points[arc >= arc[-1] - length]


def _head(points, length: float):
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    return points[arc <= length]
