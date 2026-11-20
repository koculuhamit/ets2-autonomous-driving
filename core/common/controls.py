"""Control input types.

`Axis` and `Button` describe what the stack commands, not how it reaches the
game. They used to live in the PAL, back when input was expected to need an
OS-specific virtual joystick. ADR-0002 settled that differently: the plugin
registers the input device itself, so there is no platform-specific input code
on either side and nothing here is about the operating system any more.

`clamp_axis` is the one behaviour worth centralising. Every path that can put a
number on a pedal goes through it, so out-of-range values and NaN are handled
once, the same way, instead of at each call site that remembered to.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

__all__ = ["AXIS_LIMITS", "Axis", "Button", "clamp_axis"]


class Axis(enum.StrEnum):
    """Continuous control inputs.

    `STEER` is bipolar in [-1, 1], negative left. The others are unipolar in
    [0, 1].

    Note that steering is commanded as a *rate*, not a position: the game
    integrates it and holds the result. See ADR-0002. The range here is the
    range of the command, not of the resulting angle.
    """

    STEER = "steer"
    THROTTLE = "throttle"
    BRAKE = "brake"
    CLUTCH = "clutch"


class Button(enum.StrEnum):
    """Discrete control inputs."""

    HANDBRAKE = "handbrake"
    GEAR_UP = "gear_up"
    GEAR_DOWN = "gear_down"
    ENGINE = "engine"
    HORN = "horn"
    LIGHTS = "lights"
    INDICATOR_LEFT = "indicator_left"
    INDICATOR_RIGHT = "indicator_right"

    #: The game's "attach" (T): hitches the trailer the fifth wheel is under,
    #: and unhitches the one hitched.
    TRAILER_ATTACH = "trailer_attach"

    #: Engine brake. The service brake cannot decelerate gently -- it does
    #: nothing below about 0.055 of command and then gives 1.8 m/s^2 -- and
    #: this truck has no retarder, so this is the remaining candidate for
    #: smooth deceleration. See docs/results/sysid.md.
    MOTOR_BRAKE = "motor_brake"
    ENGINE_BRAKE_UP = "engine_brake_up"
    ENGINE_BRAKE_DOWN = "engine_brake_down"
    HAZARD = "hazard"

    #: The retarder lever, a step at a time (core.control.retarder).
    RETARDER_UP = "retarder_up"
    RETARDER_DOWN = "retarder_down"

    #: The game's "activate" (Enter): held at a gas pump, it fills the tank.
    ACTIVATE = "activate"


#: Valid range per axis.
AXIS_LIMITS: Final[Mapping[Axis, tuple[float, float]]] = MappingProxyType(
    {
        Axis.STEER: (-1.0, 1.0),
        Axis.THROTTLE: (0.0, 1.0),
        Axis.BRAKE: (0.0, 1.0),
        Axis.CLUTCH: (0.0, 1.0),
    }
)


def clamp_axis(axis: Axis, value: float) -> float:
    """Clamp `value` into `axis`'s range, mapping NaN to the neutral end.

    NaN fails every comparison, so it would slip through a plain min/max and
    reach the game as garbage. A controller emitting NaN is a bug to fix
    upstream, but the vehicle must not act it out while that happens.
    """
    low, high = AXIS_LIMITS[axis]
    if value != value:  # NaN
        return 0.0 if low <= 0.0 <= high else low
    return min(max(float(value), low), high)
