"""Handing the truck back to the driver: hazard lights and two blasts of the horn.

The game's own messages -- "parking brake engaged" -- cannot be written to
from a plugin, and a window drawn over the game was not wanted. So when the
autonomous drive ends -- at a port, at the depot, on a safety stop -- the
truck says so the way a truck can: once it is at rest on its parking brake,
the hazard lights come on and the horn sounds twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.common.controls import Button

__all__ = ["HANDOVER_S", "HazardsOff", "handover_buttons"]

PRESS_S: Final = 0.2
HORN_S: Final = 0.3
GAP_S: Final = 0.3

HANDOVER_S: Final = PRESS_S + GAP_S + HORN_S + GAP_S + HORN_S + GAP_S
"""How long the whole signal takes."""


def handover_buttons(t: float) -> frozenset[Button]:
    """The buttons held `t` seconds into the hand-over: the hazard switch
    pressed once, then the horn, a pause, the horn again."""
    if 0.0 <= t < PRESS_S:
        return frozenset({Button.HAZARD})
    first = PRESS_S + GAP_S
    second = first + HORN_S + GAP_S
    if first <= t < first + HORN_S or second <= t < second + HORN_S:
        return frozenset({Button.HORN})
    return frozenset()


OFF_WITHIN_S: Final = 5.0
"""Only so soon after a drive begins are the hazard lights switched off:
later on, the driver switched them on, in a jam say."""

PRESS_AGAIN_S: Final = 1.0
"""The switch is pressed again only once the telemetry has had this long to
show the lights off: pressed again too soon, it would switch them back on."""


@dataclass
class HazardsOff:
    """Switches off, as a drive begins, the hazard lights the last hand-over
    switched on (the game's switch toggles them)."""

    _pressed_at: float | None = None

    def buttons(self, *, on: bool, since_start: float) -> frozenset[Button]:
        """What to press `since_start` seconds into the drive, with the
        lights `on` as the telemetry shows them."""
        if not on or since_start > OFF_WITHIN_S:
            return frozenset()
        if self._pressed_at is None or since_start - self._pressed_at >= PRESS_AGAIN_S:
            self._pressed_at = since_start
        if since_start - self._pressed_at < PRESS_S:
            return frozenset({Button.HAZARD})
        return frozenset()
