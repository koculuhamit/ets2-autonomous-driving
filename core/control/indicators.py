"""Indicating before turns and lane changes, as a driver does.

The map says which indicator each junction lane expects -- a lane turning off
right is `RightBlinker` -- so a turn is indicated from `TURN_LEAD_M` before
the truck enters that lane until it leaves it. A lane change is indicated
from `CHANGE_LEAD_M` before the move across begins until it is done. Where
the two overlap, the one that ends first wins.

The indicators are the game's own switches, toggled by a button each: this
presses one when the switch the telemetry reports differs from the one
wanted -- the other side's to switch over, the same side's to cancel -- and
then waits for the telemetry to follow, as the parking brake does
(core.control.parking).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from core.common.controls import Button
from core.control.lane_change import LaneChange
from core.localization.path import ReferencePath
from core.map.network import BLINKERS
from core.planning.route import LANE_CHANGE_LENGTH_M, ONWARD, Route

__all__ = ["Indicators", "Signal", "route_signals"]

TURN_LEAD_M: Final = 60.0
"""A turn is indicated from this far before its lane: 4 s at 50 km/h."""

CHANGE_LEAD_M: Final = 40.0
"""A lane change is indicated from this far before the move across begins."""

PRESS_S: Final = 0.2
SETTLE_S: Final = 0.8
"""A press is held this long, and the telemetry given this long to follow."""

LEFT: Final = -1
RIGHT: Final = 1


@dataclass(frozen=True, slots=True)
class Signal:
    start: float
    end: float
    """Metres along the path the indicator is on between."""

    side: int
    """`LEFT` or `RIGHT`."""


def route_signals(
    route: Route, path: ReferencePath, changes: Sequence[LaneChange] = ()
) -> list[Signal]:
    """Where along `path` to indicate: each turn the map expects an indicator
    for, and each lane change in `changes` (core.control.lane_change)."""
    network = route.network
    left, right = BLINKERS.index("LeftBlinker"), BLINKERS.index("RightBlinker")
    found: list[Signal] = []
    near = 0.0
    for index, lane in enumerate(route.lanes):
        if index and route.steps[index - 1] != ONWARD:
            continue  # moved across into, not driven into: not a turn
        blinker = int(network.lane_blinker[lane])
        if blinker not in (left, right):
            continue
        start = network.lane_points(lane)[0]
        s = path.nearest(float(start[0]), float(start[2]), near=near, window=1e9).s
        near = s
        found.append(
            Signal(
                s - TURN_LEAD_M,
                s + float(network.lane_length[lane]),
                LEFT if blinker == left else RIGHT,
            )
        )
    found.extend(
        Signal(
            change.start - CHANGE_LEAD_M,
            change.start + LANE_CHANGE_LENGTH_M,
            LEFT if change.side < 0 else RIGHT,
        )
        for change in changes
    )
    return found


def wanted(signals: Sequence[Signal], progress: float) -> int:
    """The side to indicate at `progress`, 0 for none. Of two at once, the
    one that ends first: a lane change is finished before the turn after it
    is indicated."""
    active = [s for s in signals if s.start <= progress <= s.end]
    if not active:
        return 0
    return min(active, key=lambda s: s.end).side


@dataclass
class Indicators:
    """Presses the indicator buttons so the switches follow `route_signals`."""

    signals: list[Signal]
    _pressing: Button | None = None
    _pressed_at: float | None = None

    def buttons(
        self, *, progress: float, left_on: bool, right_on: bool, now: float
    ) -> frozenset[Button]:
        if self._pressed_at is not None:
            since = now - self._pressed_at
            if since < PRESS_S and self._pressing is not None:
                return frozenset({self._pressing})
            if since < PRESS_S + SETTLE_S:
                return frozenset()
            self._pressed_at = self._pressing = None
        want = wanted(self.signals, progress)
        on = LEFT if left_on else RIGHT if right_on else 0
        if want == on:
            return frozenset()
        side = want if want else on  # switch over, or cancel the one that is on
        self._pressing = Button.INDICATOR_LEFT if side == LEFT else Button.INDICATOR_RIGHT
        self._pressed_at = now
        return frozenset({self._pressing})
