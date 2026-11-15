"""Driving the game's own route: planned the way its GPS plans, and checked.

Following the minimap's red line junction by junction saw each junction
150-250 m ahead: an exit off the far lane of a motorway came into view too
late to reach, a road that ended was driven to its end, and a branch the
truck was not in the lane for was swerved into at speed. The telemetry has
no route either. But it has the route's length and time, and the map has
what the game weighs roads by (core.map.nav_scale): the route the game's GPS
draws is the quickest one on the map (core.planning.nav_field). Planned
from the map, it is known kilometres ahead, and the lanes can be chosen for
it in good time (core.planning.lanes).

Where the plan and the game part anyway, it shows, and the plan follows the
game:

* the game plans anew when the truck leaves its route: its navigation
  distance grows (`REROUTED_M`), and the route is planned again from where
  the truck is;
* a junction the minimap shows the red line leaving the route at is
  taken the red line's way, if the map has it as close to as quick
  (`CLOSE_CALL_S`) -- two ways a few seconds apart are the only ones the
  map and the game can disagree on; a way minutes slower, seen once or
  twice, is a minimap misread, but seen leaving at that junction for
  `DOUBTED_SEEN_S` it is the game's, and taken: out of Duisburg, where the
  map's reckoning was 36 km off the game's, the truck went its own way
  twice and the game planned anew each time;
* where the game's distance fits no model's way by far, the game goes a way
  of its own, through points marked on its map, and a red line on a slower
  way is taken as a close call is (`OWN_WAY_SHARE`).

Where the destination itself is not known -- a job's trailer, which the
game does not tell the SDK of until it is hitched, or a place marked on the
map -- the router drives to the likeliest of `alternatives`
(core.planning.destination), and goes over to another when the red line
leaves the route for that one's way, or the game's distance fits that one
and not the one driven to.

The route is kept `HORIZON_M` ahead, and extended as it is driven, the
next `KEEP_M` of it left as it was so a lane change about to begin is not
undone.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from core.map import CompanyPlace, LaneMatcher, RoadNetwork
from core.planning.gps import MINIMAP, ZOOM_EVERY_S, MinimapCamera, estimate_zoom
from core.planning.gps_follow import lanes_across, red_line_turn
from core.planning.lanes import plan_lanes
from core.planning.nav_field import MODELS, NavField, nav_along, nav_length
from core.planning.route import (
    CHANGE,
    FERRY,
    ONWARD,
    WAYPOINT_RADIUS_M,
    Route,
    RouteError,
    _onto_road,
)

__all__ = [
    "CLOSE_CALL_S",
    "CLOSE_CALL_SHARE",
    "HORIZON_M",
    "REROUTED_M",
    "GameRouter",
    "depot_goals",
]

HORIZON_M: Final = 30_000.0
"""World metres of route kept ahead: 20 minutes at 90 km/h."""

EXTEND_WHEN_M: Final = 15_000.0
KEEP_M: Final = 2_000.0
"""Of a route extended, what is left as it was ahead of the truck."""

REROUTED_M: Final = 200.0
"""The game's navigation distance growing by more than this between two
updates is the game planning anew; driving, it only ever shrinks."""

CLOSE_CALL_S: Final = 120.0
CLOSE_CALL_SHARE: Final = 0.005
"""A way the minimap shows the red line on is taken if the map has it within
`CLOSE_CALL_S`, or this share of the time still to go, of the planned way's
time; slower than that, the minimap is misread. The map's reckoning is off
the game's by 1-3% over a route: in Dover, 26 hours from Oslo, the ferry and
the Channel Tunnel were 4 minutes apart by the map, the game took the ferry,
and its red line, turned away as a misread and then taken, sent the truck
round a roundabout again and again."""

TURNED_BACK_RAD: Final = math.radians(135.0)
"""A way leaving this far round from the truck's heading goes back: never
the slower way the red line is taken to show."""

LOOP_LOOK_LANES: Final = 60
"""How many lanes of a red line's way are followed on to see whether it
comes back where the truck has been (`GameRouter._loops_back`)."""

DOUBTED_SEEN_S: Final = 1.5
DOUBTED_SIGHTINGS: Final = 4
"""A red line leaving the route for a way the map has as minutes slower is
taken once seen there this long, in this many minimaps running: a misread
comes and goes; the game's way stays drawn."""

OWN_WAY_SHARE: Final = 0.02
"""The game's distance off every model's way by more than `MISMATCH_M` and
this share of it: the game goes a way of its own -- through points the
driver has placed on the map, which the telemetry does not tell of -- and a
red line on a way the map has as minutes slower is believed as a close call
is. To Luxembourg by two borders marked on the map, the game's route 21.8 km
off the nearest model's, the truck watched such a way at a red light, went
on its own, and the game planned anew twice: round a block and back."""

CHECK_EVERY_S: Final = 10.0
MISMATCH_M: Final = 5_000.0
MISMATCHED_CHECKS: Final = 3
"""The route and the game's differing in length by more than `MISMATCH_M`
of the game's metres at this many checks running, `CHECK_EVERY_S` apart:
they part somewhere ahead, and the models are weighed again."""

GATE_WITHIN_M: Final = 3_000.0
GATE_FIT_M: Final = 200.0
GATE_MARGIN_M: Final = 150.0
"""Within this of the game's metres of a depot with several gates, the gate
driven to is the one whose way fits the game's navigation distance within
`GATE_FIT_M`, if it fits it better than the way driven by `GATE_MARGIN_M`. The game does not go
to the nearest gate: at ika_bohag in Milan its distance fitted the far gate
within 30 m and the near one, 470 m short, not at all; the route went to
the near one, and where the red line took the far one's road, turned round
in a junction's loop to come back. Further off, the city's reckoning is out
by more than the gates are apart: 1.2 km at 6.5 km to go."""

FIT_MARGIN_M: Final = 2_000.0
"""A model is preferred to a plainer one only if its route fits the game's
length better by this."""

SWITCH_MISS_SHARE: Final = 0.04
SWITCH_MISS_M: Final = 300.0
SWITCH_FIT_SHARE: Final = 0.02
SWITCH_FIT_M: Final = 150.0
"""At a check, the destination driven to is left for an alternative when its
way misses the game's distance by this share and these metres, and the
alternative's way fits it within `SWITCH_FIT_SHARE` or `SWITCH_FIT_M`. A
red line on an alternative's way is taken where that one's way fits within
`SWITCH_MISS_SHARE` or `SWITCH_MISS_M`: 0.8 km from a trailer in Bern, a
share alone left 31 m, and the red line to the right one was turned away."""

SWITCH_WAY_SHARE: Final = 0.12
"""A red line on an alternative's way is taken for that one where its way
misses the game's distance by no more than this share (or `SWITCH_MISS_M`):
as `destination.DEPOT_SHARE`, by which the places were found. Off a job
done in Bern the GPS went to a trailer at a depot whose way the map had
10.4 % short; at 4 % the red line's way was taken for no place, the route
on from it went to the wrong one, and the truck turned back for it."""

BEHIND_RAD: Final = math.radians(120.0)
"""A place this far round from the truck's heading is behind it: never one
the game's distance is switched to (`_switch_by_distance`) -- in Bern the
point switched to was 1 km back along the road just driven, and the truck
turned round for it, the trailer ahead."""

LOCATE_RADIUS_M: Final = 6.0
LOST_UPDATES: Final = 3
"""Updates running the truck is on none of the route's lanes before the
route is planned again from where it is: a moment between two lanes
overlapping in a junction is not being lost."""


def _behind(pose: tuple[float, float, float], place: tuple[float, float]) -> bool:
    """Whether `place` is behind the truck at `pose` (`BEHIND_RAD`)."""
    x, z, heading = pose
    bearing = math.atan2(place[0] - x, place[1] - z)
    return abs(math.remainder(bearing - heading, 2.0 * math.pi)) > BEHIND_RAD


def depot_goals(network: RoadNetwork, depot: CompanyPlace) -> dict[int, float]:
    """The lanes by a depot's gates, each with the game's metres from its
    start to the gate. Kept for the network: looking where the GPS goes
    (core.planning.destination.gps_places) asks it of all 517 depots, 1.5 s
    of the 10 s a drive to a place marked on the map took to begin."""
    key = (depot.company, depot.city, tuple(map(tuple, depot.gates)))
    if not _GOALS or _GOALS[0][0]() is not network:
        _GOALS[:] = [(weakref.ref(network), {})]
    kept = _GOALS[0][1]
    if key not in kept:
        goals: dict[int, float] = {}
        for one in _gate_goals(network, depot):
            for lane, along in one.items():
                goals[lane] = min(goals.get(lane, math.inf), along)
        kept[key] = goals
    if not kept[key]:
        raise RouteError(f"no gate of {depot.company} in {depot.city} is on the network")
    return dict(kept[key])


_GOALS: list = []
"""A weak reference to the network `depot_goals` was asked of, and its
answers by depot."""


def _gate_goals(network: RoadNetwork, depot: CompanyPlace) -> list[dict[int, float]]:
    """`depot_goals` for each of a depot's gates alone."""
    found = []
    for gate in depot.gates:
        goals: dict[int, float] = {}
        x, z = _onto_road(network, gate)
        for lane in network.lanes_near(x, z, WAYPOINT_RADIUS_M):
            lane = int(lane)
            along = nav_along(network, lane, network.project(lane, x, z).s)
            goals[lane] = min(goals.get(lane, math.inf), along)
        found.append(goals)
    return found


_KEPT: list = []
"""The models' fields to the last depot weighed for (`depot_fields`), with
a weak reference to their network and what they were weighed for."""


def depot_fields(
    network: RoadNetwork, goals: dict[int, float], near: Sequence[tuple[float, float]]
) -> list[NavField]:
    """A field for each of `MODELS` to `goals`, local roads near `near` as
    they are: each a search back over the whole map, a second apiece. Kept
    for the last depot asked for, so that a drive about to start has them
    ready (tools/follow.py --standby) and plans in a fraction of a second
    once the beacon is on."""
    key = (tuple(sorted(goals.items())), tuple(map(tuple, near)))
    if _KEPT and _KEPT[0][0]() is network and _KEPT[0][1] == key:
        return _KEPT[0][2]
    fields = [NavField.to_goals(network, goals, model, free_near=near) for model in MODELS]
    _KEPT[:] = [(weakref.ref(network), key, fields)]
    return fields


@dataclass
class GameRouter:
    """Keeps the route the game's GPS takes, `HORIZON_M` ahead.

    Has the interface of `core.planning.gps.GpsRouter` -- `update` returns a
    new route or None -- with the game's navigation distance as well.
    """

    network: RoadNetwork
    depot: CompanyPlace
    route: Route
    origin: tuple[float, float] | None = None
    """Where the game planned its route from: the job's source, or where it
    last planned anew."""

    field: NavField | None = None
    """The game's way to the depot, by the model that fits the game best."""

    camera: MinimapCamera = MINIMAP
    at: int = 0
    _fields: list[NavField] | None = None
    _nav: float | None = None
    _lost: int = 0
    _replan: bool = False
    _mismatched: int = 0
    _checked_at: float | None = None
    _pending: tuple | None = None
    _doubted: tuple | None = None
    """((the junction's index on the route, the road the red line takes from
    it) the map has as minutes slower, when first seen, how many minimaps
    running): `DOUBTED_SEEN_S`."""
    _zoom_at: float | None = None
    _gate_fields: list[NavField] | None = None
    _taught_for: Route | None = None
    _taught_lines: list = dataclass_field(default_factory=list)
    """Ways to each of the depot's gates alone, by the model driven."""
    alternatives: tuple = ()
    """Other places the game's route may end (core.map.CompanyPlace), where
    the destination is not known: likeliest first."""

    _alt_fields: dict = dataclass_field(default_factory=dict)
    own_way: bool = False
    """Whether the game's distance, at the last fit, says the game goes a way
    of its own (`OWN_WAY_SHARE`)."""

    notes: list[str] = dataclass_field(default_factory=list)
    """What it has done and why, for the drive's log; the caller empties it."""

    @classmethod
    def plan(
        cls,
        network: RoadNetwork,
        depot: CompanyPlace,
        pose: tuple[float, float, float],
        *,
        origin: tuple[float, float] | None = None,
        nav_distance: float | None = None,
        alternatives: Sequence = (),
    ) -> GameRouter:
        """The game's route from a pose to the depot: by the model whose
        route is as long as `nav_distance`, the game's own, if given."""
        router = cls(network, depot, Route(network, (0,), ()), origin)
        router.alternatives = tuple(alternatives)
        router._fit(pose, nav_distance)
        if nav_distance is not None:
            router._choose_gate(pose, nav_distance)
        router.route = router._planned_from(pose)
        return router

    @classmethod
    def along(
        cls,
        network: RoadNetwork,
        depot: CompanyPlace,
        route: Route,
        *,
        origin: tuple[float, float] | None = None,
        alternatives: Sequence = (),
    ) -> GameRouter:
        """Driving `route` -- one planned with `plan` and saved -- on to the
        depot. The models are weighed on the first update, where it runs."""
        router = cls(network, depot, route, origin)
        router.alternatives = tuple(alternatives)
        return router

    def replan_here(self) -> None:
        """Plan afresh from where the truck is on the next update: the route
        driven runs out short of the depot (tools/follow.py, `REPLAN_LEFT_M`)."""
        self._replan = True

    def nav_distance(self, pose: tuple[float, float, float]) -> float:
        """The game's metres from `pose` to the depot along the route and on
        the game's way from its end: what the game's navigation should say."""
        assert self.field is not None
        at = self.locate(pose[0], pose[1])
        at = 0 if at is None else at
        lanes, steps = self.route.lanes, self.route.steps
        here = self.network.project(lanes[at], pose[0], pose[1]).s
        total = -nav_along(self.network, lanes[at], here)
        for k in range(at, len(lanes)):
            if k == at or steps[k - 1] == ONWARD:
                total += nav_length(self.network, lanes[k])
        last = lanes[-1]
        if self.route.steps and self.route.steps[-1] == FERRY:
            last = lanes[-2]
            total -= nav_length(self.network, lanes[-1])
        return total - nav_length(self.network, last) + float(self.field.distance[last])

    # ------------------------------------------------------------------ #

    def locate(self, x: float, z: float) -> int | None:
        """Index in the route's lanes of the lane the truck is on; None if
        it is on none of them."""
        lanes = self.route.lanes
        near = range(max(0, self.at - 2), min(len(lanes), self.at + 12))
        for candidates in (near, range(len(lanes))):
            best: tuple[int, float] | None = None
            for i in candidates:
                projection = self.network.project(lanes[i], x, z)
                if projection.overshoot < 1.0 and (best is None or projection.distance < best[1]):
                    best = (i, projection.distance)
            if best is not None and best[1] < LOCATE_RADIUS_M:
                self.at = best[0]
                return self.at
        # Through a border's customs on a driven line the map has no lanes
        # for, the truck is on the route however far it is from them: out of
        # the Italian-Swiss customs, 6 m off the mapped lanes in a booth's
        # lane, it was taken to be lost, the route was planned again over
        # the booths' islands, and the truck was steered into one.
        for first, last, points in self._taught():
            if np.hypot(*(points[:, [0, 2]] - (x, z)).T).min() < LOCATE_RADIUS_M:
                self.at = min(
                    range(first, last + 1),
                    key=lambda i: self.network.project(lanes[i], x, z).distance,
                )
                return self.at
        return None

    def _taught(self) -> list[tuple[int, int, np.ndarray]]:
        """The route's taught crossings (`Route.taught_crossings`), for the
        route as it is now."""
        if self._taught_for is not self.route:
            self._taught_for = self.route
            self._taught_lines = self.route.taught_crossings()
        return self._taught_lines

    def update(
        self,
        pose: tuple[float, float, float],
        mask: NDArray[np.bool_] | None = None,
        roads: NDArray[np.bool_] | None = None,
        now: float = 0.0,
        nav_distance: float | None = None,
    ) -> Route | None:
        """The route anew if the game has planned anew, a model fits the game
        better, the minimap's red line has been seen to leave it, or it has
        been extended; else None."""
        if roads is not None and (self._zoom_at is None or now - self._zoom_at >= ZOOM_EVERY_S):
            zoom = estimate_zoom(self.network, pose, roads)
            if zoom is not None:
                self.camera = MINIMAP.zoomed(zoom)
                self._zoom_at = now

        if self._fields is None:
            # First update, in the process that drives: weigh the models.
            self._fit(pose, nav_distance)
            if nav_distance is not None:
                self._choose_gate(pose, nav_distance)
            self._nav = nav_distance
            self._checked_at = now
            return self._adopt(self._planned_from(pose, self.locate(pose[0], pose[1])), pose)

        rerouted = (
            nav_distance is not None
            and self._nav is not None
            and nav_distance > self._nav + REROUTED_M
        )
        if nav_distance is not None:
            self._nav = nav_distance
        at = self.locate(pose[0], pose[1])
        self._lost = 0 if at is not None else self._lost + 1
        asked, self._replan = self._replan, False
        if rerouted or self._lost >= LOST_UPDATES or asked:
            # The game has a new route from here, and the truck is where it
            # is: planned afresh, from here.
            self._pending = None
            self._lost = 0
            self.notes.append(
                "the game planned anew: so does the route"
                if rerouted
                else "the route runs out short of the depot: planned again from here"
                if asked
                else "off the route: planned again from here"
            )
            if rerouted:
                self.origin = (pose[0], pose[1])
                self._fields = self._gate_fields = None
                self._fit(pose, nav_distance)
                self._choose_gate(pose, nav_distance)
            self.route = self._planned_from(pose, at)
            self.at = 0
            return self.route
        if at is None:
            return None

        if nav_distance is not None and (
            self._checked_at is None or now - self._checked_at >= CHECK_EVERY_S
        ):
            self._checked_at = now
            switched = self._switch_by_distance(pose, nav_distance, at)
            if switched is not None:
                return switched
            refitted = self._check(pose, nav_distance)
            if refitted is not None:
                return refitted
            if self._choose_gate(pose, nav_distance):
                return self._adopt(self._planned_from(pose, at), pose)

        if mask is not None:
            choice = red_line_turn(self.network, self.route, at, pose, mask, self.camera)
            if choice is not None and self._loops_back(*choice):
                # Off a pump near Bremen the station's way in runs beside its
                # way out, and the red line out was read onto the way back
                # in, 43 s slower by the map, within a close call: taken, the
                # truck went round the station back to the pumps and hit
                # something there. The game's way on never comes back.
                choice = None
            if (
                choice is not None
                and self.alternatives
                and self._pending != ("misread", choice)
                and not self._believable(pose, *choice)
            ):
                # Another's way, seen twice running, is gone over to; one
                # that is no one's is a misread, and stays one.
                if self._pending != ("other", choice):
                    self._pending = ("other", choice)
                    return self._extended_if_due(at)
                self._pending = None
                if self._switch_by_red_line(pose, choice):
                    return self.route
            if choice is not None and not self._believable(pose, *choice):
                # The same junction and road, whichever of its lanes the
                # minimap is read onto: a lane at a time, the count began
                # anew at each reading, and the junction was passed.
                road = (choice[0], int(self.network.lane_owner[choice[2]]))
                if self._doubted is None or self._doubted[0] != road:
                    self._doubted = (road, now, 1)
                    self.notes.append("the red line on a way minutes slower by the map: watched")
                elif self._reaches(choice[2]) and self._still_to_take(pose, *choice):
                    # A way that leads to the destination at all, ahead and
                    # not back: one that does not is a misread, or another
                    # place's way.
                    self._doubted = (road, self._doubted[1], self._doubted[2] + 1)
                    _, first, count = self._doubted
                    if count >= DOUBTED_SIGHTINGS and now - first >= DOUBTED_SEEN_S:
                        # Drawn there, and staying drawn: the game's way, whatever
                        # the map makes of it.
                        self._doubted = self._pending = None
                        self._turn(*choice)
                        self.notes.append(
                            "the red line stays on a way slower by the map: the game's, taken"
                        )
                        return self.route
                self._pending = ("misread", choice)
                return self._extended_if_due(at)
            self._doubted = None
            if choice is None:
                self._pending = None
            elif self._pending != choice:
                self._pending = choice  # seen once; twice running to be believed
            else:
                self._pending = None
                self._turn(*choice)
                self.notes.append("the red line leaves the route at a junction: its way taken")
                return self.route

        return self._extended_if_due(at)

    # ------------------------------------------------------------------ #
    # A destination not known: going over to another

    def _alternative_field(self, k: int) -> NavField | None:
        if k not in self._alt_fields:
            try:
                goals = depot_goals(self.network, self.alternatives[k])
            except RouteError:
                self._alt_fields[k] = None
            else:
                near = [self.origin] if self.origin is not None else []
                self._alt_fields[k] = NavField.to_goals(self.network, goals, free_near=near)
        return self._alt_fields[k]

    def _go_over(self, k: int, pose: tuple[float, float, float], why: str) -> None:
        """Drive to alternative `k` from now; the place left becomes one."""
        place = self.alternatives[k]
        rest = [a for j, a in enumerate(self.alternatives) if j != k]
        self.alternatives = (*rest, self.depot)
        self._alt_fields = {}
        self.depot = place
        self._fields = self._gate_fields = None
        self._fit(pose, self._nav)
        self.notes.append(f"{why}: the GPS goes to {place.company} {place.city}")

    def _switch_by_red_line(
        self, pose: tuple[float, float, float], choice: tuple[int, tuple[int, ...], int]
    ) -> bool:
        """Whether the red line's way, not the destination driven to's, is
        the quickest to an alternative whose way fits the game's distance;
        if so, that one is driven to, by the red line's way."""
        if not self.alternatives:
            return False
        i, _, onward = choice
        junction = self.route.lanes[i]
        ways = self.network.successors[
            int(self.network.successor_start[junction]) : int(
                self.network.successor_start[junction + 1]
            )
        ]
        best: tuple[float, int] | None = None
        for k in range(len(self.alternatives)):
            field = self._alternative_field(k)
            if field is None or not math.isfinite(float(field.time[onward])):
                continue
            quickest = min(float(field.time[int(w)]) for w in ways)
            if float(field.time[onward]) - quickest > CLOSE_CALL_S:
                continue
            miss = 0.0
            if self._nav is not None:
                rest = self._nav - self._nav_to(pose, i + 1)
                miss = abs(rest - float(field.distance[onward]))
                if miss > max(SWITCH_MISS_M, SWITCH_WAY_SHARE * self._nav):
                    continue
            if best is None or miss < best[0]:
                best = (miss, k)
        if best is None:
            return False
        self._go_over(best[1], pose, "the red line leaves the route for another's way")
        self._turn(*choice)
        return True

    def _switch_by_distance(
        self, pose: tuple[float, float, float], nav_distance: float, at: int
    ) -> Route | None:
        """The route to an alternative, if the game's distance has left the
        one driven to's and fits that one's."""
        if not self.alternatives or self.field is None:
            return None
        miss = abs(nav_distance - self._way_length(self.field, pose))
        if miss <= max(SWITCH_MISS_M, SWITCH_MISS_SHARE * nav_distance):
            return None
        fits = []
        for k in range(len(self.alternatives)):
            field = self._alternative_field(k)
            if field is None or _behind(pose, self.alternatives[k].position):
                continue
            theirs = abs(nav_distance - self._way_length(field, pose))
            if theirs <= max(SWITCH_FIT_M, SWITCH_FIT_SHARE * nav_distance):
                fits.append((theirs, k))
        if not fits:
            return None
        self._go_over(min(fits)[1], pose, f"{miss / 1000:.1f} km off the game's distance")
        return self._adopt(self._planned_from(pose, at), pose)

    def _extended_if_due(self, at: int) -> Route | None:
        if not self._ends() and self._ahead(at) < EXTEND_WHEN_M:
            self._extend(at)
            return self.route
        return None

    # ------------------------------------------------------------------ #

    def _fit(self, pose: tuple[float, float, float], nav_distance: float | None) -> None:
        """Weigh the models (`MODELS`) and keep the one whose way from `pose`
        is as long as the game's navigation distance -- the plainest where
        the game's is not known, or two fit as well."""
        goals = depot_goals(self.network, self.depot)
        near = [self.origin] if self.origin is not None else []
        self._fields = depot_fields(self.network, goals, near)
        self.field = self._fields[0]
        self.own_way = False
        if nav_distance is None:
            return
        misses = [abs(nav_distance - self._way_length(f, pose)) for f in self._fields]
        better = [k for k in range(len(misses)) if misses[k] <= misses[0] - FIT_MARGIN_M]
        best = min(better, key=misses.__getitem__) if better else 0
        self.field = self._fields[best]
        self.own_way = min(misses) > max(MISMATCH_M, OWN_WAY_SHARE * nav_distance)
        self.notes.append(
            f"the game's route is {nav_distance / 1000:.1f} km; by local roads "
            + ", ".join(
                f"x{m.local_factor:g} off by {miss / 1000:.1f}"
                for m, miss in zip(MODELS, misses, strict=True)
            )
            + f" km: driving x{MODELS[best].local_factor:g}"
            + ("; a way of the game's own, its red line believed" if self.own_way else "")
        )

    def _choose_gate(self, pose: tuple[float, float, float], nav_distance: float) -> bool:
        """Whether the way to one gate of the depot fits the game's distance
        better than the way driven (`GATE_WITHIN_M`); if so, it is driven."""
        if len(self.depot.gates) < 2 or nav_distance > GATE_WITHIN_M or self.field is None:
            return False
        model = self.field.model
        if self._gate_fields is None or self._gate_fields[0].model is not model:
            near = [self.origin] if self.origin is not None else []
            self._gate_fields = [
                NavField.to_goals(self.network, goals, model, free_near=near)
                for goals in _gate_goals(self.network, self.depot)
            ]
        misses = [abs(nav_distance - self._way_length(f, pose)) for f in self._gate_fields]
        best = min(range(len(misses)), key=misses.__getitem__)
        driven = abs(nav_distance - self._way_length(self.field, pose))
        if (
            self._gate_fields[best] is self.field
            or misses[best] > GATE_FIT_M
            or misses[best] + GATE_MARGIN_M > driven
        ):
            return False
        self.field = self._gate_fields[best]
        gate = self.depot.gates[best]
        self.notes.append(
            f"the game's distance fits the gate at ({gate[0]:.0f}, {gate[1]:.0f}) within "
            f"{misses[best]:.0f} m: driving to it"
        )
        return True

    def _way_length(self, field: NavField, pose: tuple[float, float, float]) -> float:
        """The game's metres of `field`'s way from `pose`; inf where none."""
        try:
            lane = _start_lane(self.network, field, pose)
        except RouteError:
            return math.inf
        return field.distance_at(lane, self.network.project(lane, pose[0], pose[1]).s)

    def _planned_from(self, pose: tuple[float, float, float], at: int | None = None) -> Route:
        """Planned from the lane the truck is on: the route's lane `at` if it
        is on the route and the depot can be reached from there, else the
        lane found under the pose. Out of Malmö the game planned anew in a
        junction, and the lane found afresh under the truck was another of
        the junction's, 9 m over; the line jumped, and the drive stopped."""
        assert self.field is not None
        start = None
        if at is not None:
            lane = self.route.lanes[at]
            if math.isfinite(float(self.field.time[lane])):
                start = lane
        if start is None:
            start = _start_lane(self.network, self.field, pose)
        return plan_lanes(self.network, self.field, start, HORIZON_M)

    def _adopt(self, planned: Route, pose: tuple[float, float, float]) -> Route | None:
        """`planned` from here on, if it goes otherwise than the route."""
        at = self.locate(pose[0], pose[1])
        if at is not None and planned.lanes[0] == self.route.lanes[at]:
            ahead = self.route.lanes[at : at + len(planned.lanes)]
            if tuple(planned.lanes[: len(ahead)]) == tuple(ahead):
                return None
        self.at = 0
        self.route = planned
        return self.route

    def _check(self, pose: tuple[float, float, float], nav_distance: float) -> Route | None:
        """Whether the route is still as long as the game's; if it has not
        been for `MISMATCHED_CHECKS` running, whether another model fits
        better, and if so the route by it."""
        miss = abs(nav_distance - self.nav_distance(pose))
        self._mismatched = self._mismatched + 1 if miss > MISMATCH_M else 0
        if self._mismatched < MISMATCHED_CHECKS or self._fields is None:
            return None
        self._mismatched = 0
        misses = [abs(nav_distance - self._way_length(f, pose)) for f in self._fields]
        best = min(range(len(misses)), key=misses.__getitem__)
        if self._fields[best] is self.field or misses[best] > miss - FIT_MARGIN_M:
            return None
        self.field = self._fields[best]
        self.notes.append(
            f"{miss / 1000:.1f} km off the game's distance; local roads "
            f"x{MODELS[best].local_factor:g} fit it within {misses[best] / 1000:.1f}"
        )
        return self._adopt(self._planned_from(pose, self.locate(pose[0], pose[1])), pose)

    def _believable(
        self, pose: tuple[float, float, float], i: int, via: tuple[int, ...], onward: int
    ) -> bool:
        """Whether the red line's way can be the game's: one the map has as
        nearly as quick as the planned one, or one whose length on from the
        junction fits the game's navigation distance better than the planned
        way's does.

        Time alone turned away a red line out of Helsingborg that was the
        game's way, minutes slower by the map's reckoning: the truck went
        on, and the game planned anew, twice. Its distance would have told.
        """
        assert self.field is not None
        planned = self.route.lanes[i + 1]
        theirs = float(self.field.time[onward])
        if not math.isfinite(theirs):
            return False
        mine = float(self.field.time[planned])
        if theirs - mine <= max(CLOSE_CALL_S, CLOSE_CALL_SHARE * mine):
            return True
        if self.own_way:
            return self._still_to_take(pose, i, via, onward)
        if self._nav is None:
            return False
        rest = self._nav - self._nav_to(pose, i + 1)
        miss_theirs = abs(rest - float(self.field.distance[onward]))
        miss_planned = abs(rest - float(self.field.distance[planned]))
        return miss_theirs + FIT_MARGIN_M < miss_planned

    def _still_to_take(
        self, pose: tuple[float, float, float], i: int, via: tuple[int, ...], onward: int
    ) -> bool:
        """Whether the red line's way, the map's slower one, can be taken
        now: the junction not entered yet, and the way not back the way the
        truck came. Out of Duisburg the ring of a roundabout drawn red was
        read as a way round and back; taken while in it, the truck went
        round and back to the depot it had left."""
        if self.at > i:
            return False  # in the junction already: too late to turn
        points = np.asarray(self.network.lane_points(onward))[:, [0, 2]]
        heading = math.atan2(*(points[-1] - points[max(0, len(points) - 4)]))
        return abs(math.remainder(heading - pose[2], 2.0 * math.pi)) < TURNED_BACK_RAD

    def _loops_back(self, i: int, via: tuple[int, ...], onward: int) -> bool:
        """Whether the red line's way, the quickest on from it to the
        destination, comes back onto the route where the truck has been:
        round and back, never the game's way on."""
        assert self.field is not None
        been = set(self.route.lanes[: i + 1])
        lane = onward
        for _ in range(LOOP_LOOK_LANES):
            if lane in been:
                return True
            onward_lanes = [
                int(k) for k in self.network.next_lanes(lane) if math.isfinite(self.field.time[k])
            ]
            if not onward_lanes:
                return False
            lane = min(onward_lanes, key=lambda k: float(self.field.time[k]))
        return False

    def _reaches(self, lane: int) -> bool:
        """Whether the destination can be reached from `lane` at all."""
        assert self.field is not None
        return math.isfinite(float(self.field.time[lane]))

    def _nav_to(self, pose: tuple[float, float, float], index: int) -> float:
        """The game's metres from `pose` along the route to the start of its
        lane `index`."""
        lanes, steps = self.route.lanes, self.route.steps
        at = self.at
        here = self.network.project(lanes[at], pose[0], pose[1]).s
        total = -nav_along(self.network, lanes[at], here)
        for k in range(at, index):
            if k == at or steps[k - 1] == ONWARD:
                total += nav_length(self.network, lanes[k])
        return total

    def _turn(self, i: int, via: tuple[int, ...], onward: int) -> None:
        """The route up to lane `i`, across through `via`, on to `onward`,
        and the game's way from there."""
        lanes, steps = self.route.lanes, self.route.steps
        # Moving across into lane i already, from a lane the new way goes
        # through: go back no further than that lane.
        start = i
        while via and start > 0 and steps[start - 1] == CHANGE and lanes[start - 1] == via[0]:
            start, via = start - 1, via[1:]
        later = self._across_later(start, via, onward)
        if later is not None:
            # Too short a lane to move across in: across in the next one,
            # which runs beside the lane on, and straight into that.
            start, via = later
            head = (*lanes[: start + 1], *via)
            head_steps = (*steps[:start], *([CHANGE] * len(via)))
            rest = plan_lanes(self.network, self.field, onward, HORIZON_M - self._behind(start))
            self.route = Route(
                self.network, (*head, *rest.lanes), (*head_steps, CHANGE, *rest.steps)
            )
            return
        head = (*lanes[: start + 1], *via)
        head_steps = (*steps[:start], *([CHANGE] * len(via)))
        rest = plan_lanes(self.network, self.field, onward, HORIZON_M - self._behind(start))
        self.route = Route(self.network, (*head, *rest.lanes), (*head_steps, ONWARD, *rest.steps))

    def _across_later(
        self, start: int, via: tuple[int, ...], onward: int
    ) -> tuple[int, tuple[int, ...]] | None:
        """Where to move across for the way on through `via` instead, as
        (index of the lane to move across from, lanes moved through before
        `onward`), when lane `start` is under `ACROSS_LANE_M` a lane moved
        across and the lane after it on the route runs beside `onward`, no
        more lanes over and longer; None to move across where it is."""
        lanes, steps = self.route.lanes, self.route.steps
        if not via or start + 1 >= len(lanes) or steps[start] != ONWARD:
            return None
        length = float(self.network.lane_length[lanes[start]])
        if length >= ACROSS_LANE_M * len(via):
            return None
        after = lanes[start + 1]
        if float(self.network.lane_length[after]) <= length:
            return None
        for across in lanes_across(self.network, after):
            if across[-1] == onward and len(across) <= len(via):
                return start + 1, across[:-1]
        return None

    def _extend(self, at: int) -> None:
        """Planned again from `KEEP_M` ahead, to `HORIZON_M`."""
        lanes, steps = self.route.lanes, self.route.steps
        keep, covered = at, 0.0
        while keep < len(lanes) - 1 and covered < KEEP_M:
            keep += 1
            if steps[keep - 1] == ONWARD:
                covered += float(self.network.lane_length[lanes[keep]])
        changed = keep > 0 and steps[keep - 1] == CHANGE
        rest = plan_lanes(
            self.network, self.field, lanes[keep], HORIZON_M, changed_into_start=changed
        )
        self.route = Route(self.network, (*lanes[:keep], *rest.lanes), (*steps[:keep], *rest.steps))

    def _ends(self) -> bool:
        """Whether the route already goes to the depot or the ship."""
        last = self.route.lanes[-1]
        return (bool(self.route.steps) and self.route.steps[-1] == FERRY) or int(
            self.field.onward[last]
        ) < 0

    def _ahead(self, at: int) -> float:
        lanes, steps = self.route.lanes, self.route.steps
        return sum(
            float(self.network.lane_length[lanes[k]])
            for k in range(at, len(lanes))
            if k == at or steps[k - 1] == ONWARD
        )

    def _behind(self, i: int) -> float:
        """World metres of route from the truck to the start of lane `i`."""
        lanes, steps = self.route.lanes, self.route.steps
        return sum(
            float(self.network.lane_length[lanes[k]])
            for k in range(self.at, i)
            if k == self.at or steps[k - 1] == ONWARD
        )


ACROSS_LANE_M: Final = 40.0
"""Metres of lane a move across for the red line's way takes at the least,
for each lane moved over; in a shorter lane it is put off to the next, beside
the way on (`GameRouter._across_later`). By Uppsala the way on was two lanes
over from a 15 m lane just past a junction: the truck swerved across both in
it, slowed to 11 km/h on the motorway for the bends, and was run into from
behind."""


START_SEARCH_M: Final = 15.0
START_HEADING_RAD: Final = math.radians(45.0)


def _start_lane(network: RoadNetwork, field: NavField, pose: tuple[float, float, float]) -> int:
    """The lane the truck is on, if the depot can be reached from it; else
    the nearest lane within `START_SEARCH_M` going the same way it can be
    reached from. Near Brussels the truck was on a road the game's map
    hides, beside the motorway, which leads nowhere the GPS goes. A lane the
    map shows is taken over a hidden one; a hidden one only when there is
    no other, as in the Channel Tunnel's terminal at Folkestone."""
    x, z, course = pose
    match = LaneMatcher(network).update(*pose)
    matched = match is not None and math.isfinite(float(field.time[match.lane]))
    if matched and not network.lane_hidden[match.lane]:
        return match.lane
    near: list[tuple[bool, float, int]] = []
    for lane in network.lanes_near(x, z, START_SEARCH_M):
        lane = int(lane)
        if not math.isfinite(float(field.time[lane])):
            continue
        projection = network.project(lane, x, z)
        turn = abs((projection.heading - course + math.pi) % (2.0 * math.pi) - math.pi)
        if projection.overshoot > 1.0 or turn > START_HEADING_RAD:
            continue
        near.append((bool(network.lane_hidden[lane]), projection.distance, lane))
    shown = [found for found in near if not found[0]]
    if shown:
        return min(shown)[2]
    if matched:
        return match.lane
    if near:
        return min(near)[2]
    raise RouteError(f"the depot cannot be reached from {pose[:2]}")
