"""Which lane to be in, along the road the game's GPS takes.

The GPS picks roads, not lanes (core.planning.nav_field). Driving it takes a
lane at every metre, and the old planner chose lanes and roads together: it
kept out of the exit-side lane (it cost a quarter more) so the minimap could
be asked about each exit before the truck was committed to it, and changed
lanes wherever a metre of cost said so. Both showed on the road: lane
changes for nothing, each slowing the truck behind the gate, and an exit
seen too late to reach from a lane in the middle.

Here the road is fixed first -- the corridor: every lane on a way as quick
as the game's, give or take `SLACK_S`, which is the game's road in all its
lanes and nothing else -- and a lane plan is chosen within it:

* the lane the truck is in, for as long as it stays on the game's road --
  never a change only to be in another lane (the driver's rule, 11
  November: no lane change unless the way calls for one);
* a lane change only where the lane is about to leave the corridor, begun
  `LEAD_PER_LANE_M` per lane still to cross ahead of where it must be done;
* one lane change per stretch of lane, unless the road leaves no other way.

Nothing behind the truck is seen (core.control.lane_change): every lane
change is made blind to a vehicle coming up in the lane moved into, and by
Helsingborg one coming up from behind hit the trailer as the truck moved
across. So the fewer the better.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final

from core.map import RoadNetwork
from core.planning.nav_field import ACROSS_S, SEA_STEP, NavField, _weights
from core.planning.route import CHANGE, FERRY, ONWARD, Route, RouteError

__all__ = [
    "CHANGE_COST_M",
    "LEAD_PER_LANE_M",
    "SLACK_S",
    "Corridor",
    "corridor",
    "plan_lanes",
]

SLACK_S: Final = 10.0
"""How much slower than the game's way a lane's best way may be and still be
the game's road: lanes side by side differ by a few metres on a bend, a road
the GPS did not take by minutes."""

BRANCH_SLACK_S: Final = 3.0
"""Where a lane leads on to more than one, a lane whose way is slower than
the quickest of them by more than this is another road, whatever
`SLACK_S` allows: in Bern the way once round a roundabout and back, 10 s
slower, was in the corridor, and the plan took it to spare a lane change
on the game's way, and turned the truck back."""

CHANGE_COST_M: Final = 300.0
"""What a lane change costs, in metres of road."""

SHORT_CHANGE_M: Final = 100.0
SHORT_CHANGE_PENALTY_M: Final = 2000.0
"""A change out of a lane shorter than `SHORT_CHANGE_M` is a swerve."""

TWICE_PENALTY_M: Final = 3000.0
"""Moving across a second lane in the stretch just moved into: only where
the road leaves no other way, as where a lane ends beside a junction."""

LEAD_PER_LANE_M: Final = 500.0
"""A lane that leaves the road within this per lane still to cross there is
late to be in: about 20 s at 90 km/h to find a gap and move across.

Lanes to cross there, not every change still ahead: by Helsingborg the
right lane forked off 1.4 km on, and a lane ending 1.5 km past the fork
called for a second change; counted as two lanes to cross before the fork,
the first change began 1.2 km early, and in it a vehicle coming up from
behind hit the trailer. Until 11 November a lane away from the exit side
also cost more per metre, and the truck moved back to the exit side after
each junction: changes for nothing, each one blind behind."""

LATE_COST: Final = 5.0
"""Per metre of a lane driven late (see `LEAD_PER_LANE_M`)."""


@dataclass(frozen=True)
class Corridor:
    """The game's road ahead, in all its lanes."""

    lanes: frozenset[int]
    goals: frozenset[int]
    """Where the plan ends: the destination, the ship, or the horizon."""

    by_sea: bool
    """Whether the goals are where a ship is boarded."""

    distance: dict[int, float]
    """World metres from the start to the start of each lane."""


def corridor(network: RoadNetwork, field: NavField, start: int, horizon: float) -> Corridor:
    """Every lane on a way from `start` within `SLACK_S` of the game's
    quickest, to `horizon` world metres ahead, or to the destination or the
    ship if nearer."""
    nav, _ = _weights(network)
    per_metre = field.per_metre
    best = float(field.time[start])
    if not math.isfinite(best):
        raise RouteError(f"lane {start} does not lead to the destination")
    elapsed = {start: 0.0}
    covered = {start: 0.0}
    queue = [(0.0, start)]
    kept: set[int] = set()
    goals: set[int] = set()
    by_sea = False
    while queue:
        t, lane = heapq.heappop(queue)
        if t > elapsed.get(lane, math.inf):
            continue
        if t + float(field.time[lane]) > best + SLACK_S:
            continue
        kept.add(lane)
        here = covered[lane]
        length = float(network.lane_length[lane])
        if int(field.onward[lane]) < 0:
            goals.add(lane)  # the destination
            continue
        if int(field.step[lane]) == SEA_STEP:
            goals.add(lane)
            by_sea = True
            continue
        if here + length >= horizon:
            goals.add(lane)
            continue
        onward = t + nav[lane] * per_metre[lane]
        following_lanes = [int(n) for n in network.next_lanes(lane)]
        quickest = min((float(field.time[n]) for n in following_lanes), default=math.inf)
        for following in following_lanes:
            if float(field.time[following]) > quickest + BRANCH_SLACK_S:
                continue  # another road
            if onward < elapsed.get(following, math.inf):
                elapsed[following] = onward
                covered[following] = here + length
                heapq.heappush(queue, (onward, following))
        for beside in network.lanes_alongside(lane):
            beside = int(beside)
            if t + ACROSS_S < elapsed.get(beside, math.inf):
                elapsed[beside] = t + ACROSS_S
                covered[beside] = here
                heapq.heappush(queue, (t + ACROSS_S, beside))
    if not goals:
        raise RouteError(f"no way on from lane {start}")
    return Corridor(frozenset(kept), frozenset(goals), by_sea, covered)


def plan_lanes(
    network: RoadNetwork,
    field: NavField,
    start: int,
    horizon: float,
    *,
    changed_into_start: bool = False,
) -> Route:
    """The lanes to drive from `start`, along the game's road (`corridor`),
    to `horizon` world metres ahead or the destination; with a ferry step
    at the end where the road goes on by sea.

    `changed_into_start`: the truck has just moved across into `start`, and
    does not move across again before the next stretch.
    """
    road = corridor(network, field, start, horizon)
    lanes = road.lanes
    successors = {
        lane: [int(n) for n in network.next_lanes(lane) if int(n) in lanes] for lane in lanes
    }
    beside = {
        lane: [int(b) for b in network.lanes_alongside(lane) if int(b) in lanes] for lane in lanes
    }
    changes_left = _changes_left(lanes, road.goals, successors, beside)
    late = _late(network, lanes, road.goals, successors, beside, changes_left, road.distance)

    def extra(lane: int) -> float:
        """What a metre of `lane` costs over a metre of road: late."""
        return LATE_COST if late[lane] else 0.0

    def driven(lane: int) -> float:
        return float(network.lane_length[lane]) * (1.0 + extra(lane))

    def change(lane: int) -> float:
        # The truck drives most of a lane it moves out of before it is
        # across, so a late lane is as late to change out of as to keep to.
        length = float(network.lane_length[lane])
        short = length < SHORT_CHANGE_M
        return CHANGE_COST_M + (SHORT_CHANGE_PENALTY_M if short else 0.0) + length * extra(lane)

    # Cost to go, backwards from the goals: from the start of a lane, free
    # to move across (`free`), or having just moved across into it and so
    # going on along it first (`on`).
    predecessors = _predecessors(lanes, successors)
    into = _predecessors(lanes, beside)
    on = {lane: math.inf for lane in lanes}
    free = {lane: math.inf for lane in lanes}
    queue: list[tuple[float, int, bool]] = []
    for goal in road.goals:
        on[goal] = driven(goal)
        queue.append((on[goal], goal, True))
    heapq.heapify(queue)
    settled: set[tuple[int, bool]] = set()
    while queue:
        cost, lane, going_on = heapq.heappop(queue)
        if (lane, going_on) in settled:
            continue
        settled.add((lane, going_on))
        if going_on:
            # Free at `lane` can go on too; the lanes beside can move into it,
            # and at a price, so can a lane just moved into.
            for other, total in [(lane, cost)] + [
                (other, change(other) + cost) for other in into[lane]
            ]:
                if total < free[other]:
                    free[other] = total
                    heapq.heappush(queue, (total, other, False))
            for other in into[lane]:
                total = change(other) + TWICE_PENALTY_M + cost
                if total < on[other]:
                    on[other] = total
                    heapq.heappush(queue, (total, other, True))
        else:
            for before in predecessors[lane]:
                total = driven(before) + cost
                if total < on[before]:
                    on[before] = total
                    heapq.heappush(queue, (total, before, True))
    if not math.isfinite(free[start]):
        raise RouteError(f"lane {start} cannot keep to the game's road")

    route_lanes, steps = [start], []
    lane, just_changed = start, changed_into_start
    while lane not in road.goals:
        options: list[tuple[float, int, str]] = [
            (driven(lane) + free[n], n, ONWARD) for n in successors[lane]
        ]
        twice = TWICE_PENALTY_M if just_changed else 0.0
        options += [(change(lane) + twice + on[b], b, CHANGE) for b in beside[lane]]
        options = [o for o in options if math.isfinite(o[0])]
        if not options:
            raise RouteError(f"the lane plan ends at lane {lane}")
        _, lane, step = min(options, key=lambda o: (o[0], o[2] != ONWARD))
        route_lanes.append(lane)
        steps.append(step)
        just_changed = step == CHANGE
        if len(route_lanes) > 200_000:
            raise RouteError("the lane plan does not end")
    if road.by_sea:
        route_lanes.append(network.ferry_links[lane][0].land)
        steps.append(FERRY)
    return Route(network, tuple(route_lanes), tuple(steps))


def _changes_left(
    lanes: frozenset[int],
    goals: frozenset[int],
    successors: dict[int, list[int]],
    beside: dict[int, list[int]],
) -> dict[int, int]:
    """How many lane changes each lane is from the goals, at least."""
    predecessors = _predecessors(lanes, successors)
    into = _predecessors(lanes, beside)
    count = {lane: math.inf for lane in lanes}
    queue: list[tuple[int, int]] = [(0, goal) for goal in goals]
    for goal in goals:
        count[goal] = 0
    heapq.heapify(queue)
    while queue:
        c, lane = heapq.heappop(queue)
        if c > count[lane]:
            continue
        for before in predecessors[lane]:
            if c < count[before]:
                count[before] = c
                heapq.heappush(queue, (c, before))
        for other in into[lane]:
            if c + 1 < count[other]:
                count[other] = c + 1
                heapq.heappush(queue, (c + 1, other))
    return {lane: int(c) if math.isfinite(c) else 99 for lane, c in count.items()}


def _late(
    network: RoadNetwork,
    lanes: frozenset[int],
    goals: frozenset[int],
    successors: dict[int, list[int]],
    beside: dict[int, list[int]],
    changes_left: dict[int, int],
    distance: dict[int, float],
) -> dict[int, bool]:
    """Lanes that leave the road -- need a change to stay on it -- too soon
    to be driven in: within `LEAD_PER_LANE_M` per lane to cross before they
    do (`_to_cross`)."""
    # How far each lane goes on before it needs another change: through the
    # following lanes that need as many.
    reach: dict[int, float] = {}
    for lane in _backwards(lanes, successors):
        length = float(network.lane_length[lane])
        if lane in goals:
            reach[lane] = math.inf
            continue
        same = [
            reach.get(n, 0.0) for n in successors[lane] if changes_left[n] == changes_left[lane]
        ]
        reach[lane] = length + (max(same) if same else 0.0)
    cross = _to_cross(lanes, beside, changes_left, reach, distance)
    # Late: when this stretch of lane ends, less is left before the change
    # has to be done than `LEAD_PER_LANE_M` for each lane to cross by then.
    return {
        lane: changes_left[lane] > 0
        and reach[lane] - float(network.lane_length[lane]) < LEAD_PER_LANE_M * cross[lane]
        for lane in lanes
    }


def _to_cross(
    lanes: frozenset[int],
    beside: dict[int, list[int]],
    changes_left: dict[int, int],
    reach: dict[int, float],
    distance: dict[int, float],
) -> dict[int, int]:
    """For each lane that needs a change, how many lanes it has to cross
    before its stretch ends: one, and one more for each lane on the way
    across whose own stretch ends there too, within `LEAD_PER_LANE_M` -- the
    middle lane of three where the road forks off the far side -- but none
    for a change the lane moved into needs further on."""
    ends = {lane: distance.get(lane, 0.0) + reach[lane] for lane in lanes}
    cross: dict[int, int] = {}
    for lane in sorted(lanes, key=lambda lane: changes_left[lane]):
        c = changes_left[lane]
        if c == 0:
            cross[lane] = 0
            continue
        more = [
            cross[other] if ends[other] <= ends[lane] + LEAD_PER_LANE_M else 0
            for other in beside[lane]
            if changes_left[other] == c - 1
        ]
        cross[lane] = 1 + (min(more) if more else 0)
    return cross


def _predecessors(lanes: frozenset[int], links: dict[int, list[int]]) -> dict[int, list[int]]:
    """For each lane, the lanes whose `links` lead to it."""
    back: dict[int, list[int]] = {lane: [] for lane in lanes}
    for lane, following in links.items():
        for n in following:
            back[n].append(lane)
    return back


def _backwards(lanes: frozenset[int], successors: dict[int, list[int]]) -> list[int]:
    """The lanes, each after all it leads to (where there is no loop)."""
    order: list[int] = []
    state: dict[int, int] = {}
    for root in lanes:
        if root in state:
            continue
        stack = [(root, 0)]
        while stack:
            lane, k = stack.pop()
            if k == 0:
                if state.get(lane) is not None:
                    continue
                state[lane] = 1
            following = successors[lane]
            if k < len(following):
                stack.append((lane, k + 1))
                n = following[k]
                if n not in state:
                    stack.append((n, 0))
            else:
                state[lane] = 2
                order.append(lane)
    return order
