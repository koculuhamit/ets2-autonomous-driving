"""A delivery, from engaging to the depot: which drive comes next, and when.

One drive (tools/follow.py) goes from where the truck is to the end of a
route: the depot, or the ship or train that crosses the sea. A delivery from
Uppsala to Plymouth was eight of them, each planned and started by hand,
with the driver boarding the ferry and the train in between. This is the
state of the whole delivery, stepped by what the telemetry shows and by how
each drive ended; the caller runs the drives it asks for.

The driver engages it with the truck's beacon: switched on, the truck drives
on; switched off, the driver has it (follow.py --beacon lets go at once).
After a drive has ended the beacon must have been off before it counts
again, so a beacon left on from the last drive does not start the next one:
across the sea, off the train at Folkestone, the driver drives off the
platform first and then switches it off and on.

- `NO_JOB`: nothing to deliver, and no route on the game's GPS.
- `READY`: a job -- or a route on the GPS with no job told (`GPS_JOB`: a
  job's trailer not hitched yet, a place marked on the map; where it ends is
  found from its distance, core.planning.destination) -- waiting for the
  beacon.
- `DRIVING`: a drive is running.
- `CROSSING`: at a port or a train terminal; the driver takes the crossing
  from the game's menu, and the truck is `CROSSED_M` away before it is
  `READY` again.
- `ARRIVED`: at the depot's gate; the driver parks and delivers, and the
  next job makes it `READY`.
- `STOPPED`: a drive stopped short -- a safety stop, no route; `READY` for
  the beacon again once the driver has seen why.

A paused game changes nothing: engaging is not heard while it is paused.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import Final

__all__ = [
    "ARRIVED_NAV_M",
    "CROSSED_M",
    "GPS_JOB",
    "DriveEnd",
    "Mission",
    "Observation",
    "State",
    "how_it_ended",
]

CROSSED_M: Final = 5000.0
"""How far from the port the truck must be to be across the sea. The game
moves it there at once: Calais to Folkestone is 7 km of map, and the
nearest ports on a line are further apart than that."""


ARRIVED_NAV_M: Final = 2000.0
"""At a route's end with the game's navigation this near the depot, the
truck is there: 86 m at Stokes in Plymouth. At a port it is hundreds of km."""

NEW_PLACE_NAV_M: Final = 1000.0
"""Arrived where the GPS went, its distance this far again: a new place has
been marked on the map, and it is driven to."""

AT_DEPOT_NAV_M: Final = 200.0
"""Engaged with the game's navigation this near the depot, the truck is at
it: 0-90 m in the depots reached. Further off it drives on: reloaded 738 m
short of Cesare Smar in Milan, the last two lights still to come, the
beacon did nothing when this was `ARRIVED_NAV_M`."""

AT_PORT_M: Final = 600.0
"""At a route's end this near where a ship or train is boarded, the truck is
at the crossing."""


GPS_JOB: Final = ("", "where the GPS goes")
"""The job of a route on the game's GPS that no job is told for."""


class State(enum.StrEnum):
    NO_JOB = "no_job"
    READY = "ready"
    DRIVING = "driving"
    CROSSING = "crossing"
    ARRIVED = "arrived"
    STOPPED = "stopped"


class DriveEnd(enum.StrEnum):
    """How a drive ended, as the caller makes out."""

    AT_CROSSING = "at_crossing"
    AT_DEPOT = "at_depot"
    DISENGAGED = "disengaged"
    UNSAFE = "unsafe"
    NO_ROUTE = "no_route"
    LOST_GAME = "lost_game"


@dataclass(frozen=True, slots=True)
class Observation:
    """What the telemetry shows, each time it is looked at."""

    paused: bool
    job: tuple[str, str] | None
    """(company, city) of the job's destination; None without a job."""

    beacon: bool
    position: tuple[float, float]
    """(x, z), metres."""

    nav_distance: float = math.inf
    """The game's navigation distance to the job's destination, metres."""


@dataclass
class Mission:
    state: State = State.NO_JOB
    message: str = "no job"
    """What the driver is told: the state, and why."""

    _job: tuple[str, str] | None = None
    _armed: bool = False
    """Whether the beacon has been off since the last drive ended."""

    _crossing_from: tuple[float, float] | None = None

    def observe(self, seen: Observation) -> bool:
        """Step on what the telemetry shows; True when a drive is to start now."""
        if not seen.beacon:
            self._armed = True
        if self.state is State.DRIVING:
            return False
        if seen.job is None and 0.0 < seen.nav_distance < math.inf:
            seen = replace(seen, job=GPS_JOB)
        if seen.job is None:
            self._job = None
            self._set(State.NO_JOB, "no job")
            return False
        if seen.job != self._job:
            self._job = seen.job
            where = (
                "to where the GPS goes"
                if seen.job == GPS_JOB
                else f"to {seen.job[0]} in {seen.job[1]}"
            )
            self._set(State.READY, f"{where}: switch the beacon on")
        if self.state is State.CROSSING:
            assert self._crossing_from is not None
            away = math.dist(seen.position, self._crossing_from)
            if away < CROSSED_M:
                return False
            # Only a switch made across counts: off the train, the driver
            # first drives off the platform, which has no lanes to drive.
            self._armed = not seen.beacon
            self._set(State.READY, "across: switch the beacon off and on to drive on")
        if seen.paused:
            return False
        if self.state is State.ARRIVED and seen.nav_distance >= NEW_PLACE_NAV_M:
            # A place marked on the map anew once the last one was reached:
            # the job is the same, and near Geneva the mission stood arrived
            # at the old place for good.
            self._set(State.READY, "the GPS goes on: switch the beacon off and on")
        if self.state is State.ARRIVED:
            return False
        if seen.beacon and self._armed:
            self._armed = False
            if seen.nav_distance < AT_DEPOT_NAV_M:
                # At Plymouth with the job not yet delivered, a drive would be
                # planned into the depot's yard, which is the driver's.
                self._set(State.ARRIVED, "already at the depot: park and deliver")
                return False
            self._set(State.DRIVING, "driving")
            return True
        return False

    @property
    def to_gps(self) -> bool:
        """Whether the drive is to where the game's GPS goes, no job told."""
        return self._job == GPS_JOB

    def drive_ended(self, how: DriveEnd, where: tuple[float, float], detail: str = "") -> None:
        """A drive has ended, `how`, with the truck at `where`."""
        self._armed = False
        if how is DriveEnd.AT_CROSSING:
            self._crossing_from = where
            self._set(State.CROSSING, "at the crossing: take it from the game's menu")
        elif how is DriveEnd.AT_DEPOT:
            self._set(State.ARRIVED, "at the depot: park and deliver")
        elif how is DriveEnd.DISENGAGED:
            self._set(State.READY, "the driver has the truck: switch the beacon on to drive on")
        else:
            why = {
                DriveEnd.UNSAFE: "stopped",
                DriveEnd.NO_ROUTE: "no route",
                DriveEnd.LOST_GAME: "lost the game",
            }[how]
            self._set(State.STOPPED, f"{why}{': ' + detail if detail else ''}")

    def _set(self, state: State, message: str) -> None:
        self.state, self.message = state, message


def how_it_ended(
    reason: str | None,
    detail: str,
    nav_distance: float,
    to_port: float,
) -> tuple[DriveEnd, str]:
    """How a drive ended, from what tools/follow.py wrote (`reason`, None if
    it wrote nothing), the game's navigation distance left, and how far the
    truck is from the nearest place a ship or train is boarded."""
    if reason == "end_of_route":
        if nav_distance < ARRIVED_NAV_M:
            return DriveEnd.AT_DEPOT, ""
        if to_port < AT_PORT_M:
            return DriveEnd.AT_CROSSING, ""
        return DriveEnd.UNSAFE, "the route ended short of the depot and of any port"
    if reason in ("disengaged", "interrupted"):
        return DriveEnd.DISENGAGED, ""
    if reason == "lost_game":
        return DriveEnd.LOST_GAME, detail
    if reason == "unsafe":
        return DriveEnd.UNSAFE, detail
    if reason == "duration":
        return DriveEnd.UNSAFE, "the drive ran out of time"
    return DriveEnd.UNSAFE, detail or "the drive did not start"
