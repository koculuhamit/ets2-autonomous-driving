"""The retarder: the gentle deceleration the service brake cannot give.

The service brake does nothing below about 0.055 of command and then gives
1.8 m/s^2 at once; lifting off gives 0.22. Between them there was nothing,
and every slowing-down that wanted more than coasting was a step of 1.6 m/s^2
(docs/results/sysid.md). Most trucks have a retarder -- three to five steps
on the trucks driven so far (`retarder_step_count`) -- and it was never used.

Two parts:

- `RetarderEstimate` learns, while driving, what a step of the retarder
  slows the truck by: its strength is set per truck by the game and was never
  measured, so it starts from a cautious guess and is corrected from the
  telemetry whenever the retarder is on and the pedals are not.
- `RetarderLever` moves the game's lever to the step wanted, one press of
  "retarder up" or "retarder down" at a time, watching the telemetry's
  `retarder_level` for each to land.

`core.control.longitudinal.ActuatorMap` decides the step wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from core.common.controls import Button

__all__ = [
    "PER_STEP_GUESS",
    "RETARDER_FROM_MPS",
    "RetarderEstimate",
    "RetarderLever",
]

PER_STEP_GUESS: Final = 0.25
"""What one step is taken to slow the calibrated truck by, m/s^2, until it
has been seen: low on purpose -- a retarder weaker than thought leaves the
speed controller's integrator to ask for another step; a stronger one would
slow the truck by more than was asked for."""

RETARDER_FROM_MPS: Final = 15.0 / 3.6
"""Below this a retarder fades, and a stop is the service brake's job."""

LEARN_AFTER_S: Final = 1.0
"""A step is judged only once it has been on this long: the game takes a
moment to bring the retarder's torque up."""

LEARN_TIME_S: Final = 4.0
"""How quickly the estimate follows what is seen: the time constant."""

THROTTLE_OFF: Final = 0.02
"""The game's pedals, let go, read a little over nothing: 1e-9 to 0.004."""

LEAST_PER_STEP: Final = 0.05
MOST_PER_STEP: Final = 1.5
"""Bounds on a step's strength, m/s^2: what is seen outside them is a bump,
a hill the grade channel lagged on, or a trailer swinging."""


@dataclass
class RetarderEstimate:
    """What one step of the retarder slows the calibrated truck by, m/s^2,
    learned from what the truck does with it on (`observe`)."""

    per_step: float = PER_STEP_GUESS
    seen_s: float = 0.0
    """How long the retarder has been seen at work, seconds."""
    _level: int = 0
    _level_since: float = 0.0

    def observe(
        self,
        *,
        level: int,
        throttle: float,
        brake: float,
        acceleration: float,
        grade: float,
        speed: float,
        coast_deceleration: float,
        mass_ratio: float,
        dt: float,
    ) -> None:
        """One frame of telemetry: the retarder's `level`, the pedals the
        game applied, the truck's `acceleration` along its way and the
        `grade` (both m/s^2, the grade positive downhill), its `speed`; and
        what coasting gives the calibrated truck, and the mass against it."""
        if level != self._level:
            self._level, self._level_since = level, 0.0
            return
        self._level_since += dt
        if (
            level <= 0
            or throttle > THROTTLE_OFF
            or brake > THROTTLE_OFF
            or speed < RETARDER_FROM_MPS
            or self._level_since < LEARN_AFTER_S
            or dt <= 0.0
        ):
            return
        # The powertrain's own deceleration, as the calibrated truck would
        # show it: gravity taken out, the mass taken back.
        slowing = (grade - acceleration) * mass_ratio
        seen = (slowing - coast_deceleration) / level
        if not LEAST_PER_STEP <= seen <= MOST_PER_STEP:
            return
        weight = min(dt / LEARN_TIME_S, 1.0)
        self.per_step += weight * (seen - self.per_step)
        self.seen_s += dt


LEVER_PRESS_S: Final = 0.2
"""How long a lever button is held: as the indicators' (core.control.indicators)."""

LEVER_SETTLE_S: Final = 0.4
"""After a press, how long the telemetry has to show the lever moved before
it is pressed again: pressed again too soon, it would go a step too far."""

UNANSWERED_PRESSES: Final = 3
"""Presses in a row the lever did not move for, and it is taken not to work:
a plugin older than the retarder's buttons, or a game that renamed them. The
gap is then left to the brake, as before, not coasted through."""


@dataclass
class RetarderLever:
    """Presses the game's retarder up and down until its lever is where it
    is wanted."""

    _pressing: Button | None = None
    _pressed_at: float = 0.0
    _level_then: int = 0
    presses: int = field(default=0)
    _unanswered: int = 0
    works: bool = True
    """False once `UNANSWERED_PRESSES` presses in a row moved nothing."""

    def buttons(self, *, wanted: int, level: int, now: float) -> frozenset[Button]:
        """The buttons to hold now, to bring the lever from `level` (the
        telemetry's) to `wanted`."""
        if self._pressing is not None:
            since = now - self._pressed_at
            if since < LEVER_PRESS_S:
                return frozenset({self._pressing})
            if level == self._level_then and since < LEVER_PRESS_S + LEVER_SETTLE_S:
                return frozenset()
            self._pressing = None
            self._unanswered = self._unanswered + 1 if level == self._level_then else 0
            if self._unanswered >= UNANSWERED_PRESSES:
                self.works = False
        if not self.works or wanted == level:
            return frozenset()
        self._pressing = Button.RETARDER_UP if wanted > level else Button.RETARDER_DOWN
        self._pressed_at, self._level_then = now, level
        self.presses += 1
        return frozenset({self._pressing})
