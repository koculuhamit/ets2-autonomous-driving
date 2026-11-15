"""Coming to rest exactly at a route's end: a depot's gate, a port.

The lateral controller says it is at the end once its aim point -- metres
ahead -- is: at the depot in Linz, 8 m short of the gate, still gaining speed
at 13 km/h.

So towards every route's end the speed is capped by what stops the truck
there at a comfortable deceleration (`speed_to_stop_in`), and the drive ends
only with the truck itself there (`reached`).
"""

from __future__ import annotations

import math
from typing import Final

__all__ = [
    "END_STOP_DECELERATION",
    "PAST_M",
    "RESTED_M",
    "reached",
    "speed_to_stop_in",
]

END_STOP_DECELERATION: Final = 1.0
"""m/s^2: a gentle stop, under the profile's planning 1.5."""

RESTED_M: Final = 1.5
"""At rest this near the end, the truck is there: the cap brings it to rest
within centimetres of it, and the last are not worth creeping for."""

PAST_M: Final = 0.5
"""Gone this far past the end, it is there too."""

STILL_MPS: Final = 0.3


def speed_to_stop_in(remaining: float, deceleration: float = END_STOP_DECELERATION) -> float:
    """The fastest from which `remaining` metres stop the truck at
    `deceleration`."""
    return math.sqrt(2.0 * deceleration * max(remaining, 0.0))


def reached(remaining: float, speed: float) -> bool:
    """Whether the truck, `remaining` metres from the end and going at
    `speed`, is at the end: past it, or at rest within `RESTED_M` of it."""
    return remaining <= -PAST_M or (remaining <= RESTED_M and abs(speed) < STILL_MPS)
