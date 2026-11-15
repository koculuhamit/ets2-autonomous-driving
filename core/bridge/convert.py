"""Turning raw telemetry into quantities with agreed units and signs.

The block stores what the game reports, in the game's conventions. Those are
not the ones the rest of the stack wants, and two of them are actively
misleading: `linear_acceleration.z` runs *opposite* to forward acceleration,
and `heading` is in rotations with a direction that has to be derived rather
than assumed. Code that reads those fields directly gets a plausible number
with the wrong sign, which is the failure this project keeps designing against.

So the conversions live here, once. Every constant below was measured against
recorded runs rather than taken from documentation, and the measurement is
named so it can be repeated:

* **Forward acceleration** is `-linear_acceleration.z`. Checked against the
  derivative of speed over three independent manoeuvres -- coasting,
  accelerating and braking -- giving slopes of -0.98, -0.99 and -0.80 with
  correlations of -0.996, -0.986 and -0.944. The weaker braking figure is
  suspension pitch, not a different convention.
* **Forward direction** is `(-sin(heading), -cos(heading))` in the world
  `(x, z)` plane. Checked against the actual direction of travel across five
  runs covering headings near 0, 91, 179, 270 and 360 degrees: mean alignment
  1.00000, worst sample 0.999.
* **Negative pitch is downhill.** Fitting lift-off acceleration against
  `sin(pitch)` gives a slope of -8.33 m/s^2 with R^2 = 0.92, against g =
  9.81 -- the right sign, and 85 % of the magnitude.

See `docs/results/sysid.md` for the runs these came from.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = [
    "GRAVITY",
    "course_rad",
    "forward_acceleration",
    "forward_unit",
    "grade_acceleration",
    "heading_rad",
    "hitch_angle_rad",
    "job_target",
    "pitch_rad",
    "roll_rad",
    "speed_kmh",
    "wrap_angle",
]

GRAVITY: Final = 9.80665

#: The game reports orientation in rotations: 1.0 is a full turn.
_ROTATIONS_TO_RADIANS: Final = 2.0 * math.pi

_MS_TO_KMH: Final = 3.6


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return -((-angle + math.pi) % (2.0 * math.pi) - math.pi)


def speed_kmh(block: np.void) -> float:
    """Forward speed in km/h. Negative when reversing."""
    return float(block["truck"]["speed"]) * _MS_TO_KMH


def forward_acceleration(block: np.void) -> float:
    """Acceleration along the direction of travel, m/s^2.

    Positive speeds the truck up. This is the *negation* of the raw channel:
    the vehicle-frame z axis reported by the game runs backwards, so reading
    `linear_acceleration.z` directly gives a correct magnitude with the wrong
    sign -- a controller built on it would accelerate to brake.
    """
    return -float(block["truck"]["linear_acceleration"]["z"])


def heading_rad(block: np.void) -> float:
    """Heading in radians, as the game measures it.

    Converted from rotations, nothing else. The useful form is usually
    `forward_unit`, because the mapping from this angle to a direction in the
    world is not the obvious one.
    """
    return float(block["truck"]["placement"]["orientation"]["heading"]) * _ROTATIONS_TO_RADIANS


def course_rad(block: np.void) -> float:
    """Direction of travel as an angle, radians, in the `atan2(dx, dz)` frame.

    **This is not `heading_rad`, and mixing them up costs 180 degrees.** The
    game's heading is a value whose forward direction is `(-sin h, -cos h)`;
    an angle measured the ordinary way from a pair of position differences is
    `atan2(forward_x, forward_z)`, which works out as `heading + pi`.

    Anything that compares the vehicle against geometry derived from positions
    -- a reference path, a map centreline, a recorded trace -- wants this one.
    `heading_rad` is for reading the channel and little else.

    It is a separate function rather than a corrected `heading_rad` because
    both quantities are real and the difference is not a bug to be hidden: one
    is what the game reports, the other is where the vehicle is going.
    """
    forward_x, forward_z = forward_unit(block)
    return math.atan2(forward_x, forward_z)


def pitch_rad(block: np.void) -> float:
    """Pitch in radians. Negative is nose-down, i.e. travelling downhill."""
    return float(block["truck"]["placement"]["orientation"]["pitch"]) * _ROTATIONS_TO_RADIANS


def roll_rad(block: np.void) -> float:
    """Roll in radians."""
    return float(block["truck"]["placement"]["orientation"]["roll"]) * _ROTATIONS_TO_RADIANS


def forward_unit(block: np.void) -> tuple[float, float]:
    """Unit vector the truck points along, as world `(x, z)`.

    Not `(sin, cos)` of the heading, which is exactly backwards -- measured
    across all four quadrants, not inferred.
    """
    heading = heading_rad(block)
    return (-math.sin(heading), -math.cos(heading))


def grade_acceleration(block: np.void) -> float:
    """Along-track acceleration from gravity, m/s^2.

    Positive downhill. Subtract it from measured acceleration to get what the
    powertrain and drag are doing, or add it as a feedforward term when asking
    for an acceleration on a slope.

    On the roads measured so far this term dominates: a 5 degree grade
    contributes about 0.9 m/s^2, against roughly 0.2 m/s^2 of combined drag and
    engine braking at 45 km/h. Gradient feedforward matters far more than an
    aerodynamic one at these speeds.
    """
    return -GRAVITY * math.sin(pitch_rad(block))


def hitch_angle_rad(block: np.void, trailer: int = 0) -> float:
    """Angle between the truck and a trailer, wrapped to (-pi, pi].

    Zero when the combination is straight. Returns 0.0 when no trailer is
    attached.

    **The sign convention is not yet established.** The magnitude tracks
    manoeuvres correctly and the wrap behaves, but which sign corresponds to
    which way the trailer is folded has not been checked against a known
    manoeuvre. Phase 8 depends on this and should verify it before relying on
    it -- a reversed sign there is a controller that jackknifes deliberately.
    """
    trailer_block = block["trailers"][trailer]
    if not int(trailer_block["connected"]):
        return 0.0
    truck_heading = heading_rad(block)
    trailer_heading = (
        float(trailer_block["placement"]["orientation"]["heading"]) * _ROTATIONS_TO_RADIANS
    )
    return wrap_angle(truck_heading - trailer_heading)


def road_speed_limit(block: np.void) -> float | None:
    """The speed limit where the truck is, m/s, or None when the game has none.

    From the navigation channel, which is the limit of the road the truck is
    on now -- not the one ahead. A drop from 130 to 50 at the edge of a town is
    seen when the truck reaches it, so this can hold a speed down but not plan
    a deceleration into a lower limit; that needs the limits from the map.
    """
    limit = float(block["navigation"]["speed_limit"])
    return limit if limit > 0.0 else None


def job_target(block: np.void) -> tuple[str, str] | None:
    """Where the job's GPS route goes, as (company, city): the depot its
    trailer waits at while the trailer is not hitched, the destination once
    it is; None with no job. In Milan a job's trailer waited at its source,
    the game's route went there, and a drive set off for the destination."""
    job = block["job"]
    if not int(job["job_active"]):
        return None
    end = "destination" if int(block["trailers"][0]["connected"]) else "source"

    def text(field: str) -> str:
        return bytes(job[f"{end}_{field}"]).split(b"\0", 1)[0].decode()

    return text("company"), text("city")
