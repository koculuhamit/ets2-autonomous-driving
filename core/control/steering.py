"""Steering position control.

The game does not accept a steering angle. It accepts a steering *rate*: the
commanded value sets how fast the wheel turns, the applied angle integrates it,
and the wheel stays where it was left rather than self-centring. ADR-0002 has
the measurements.

Every layer above this one wants to think in angles, so this is where the
conversion lives. A proportional loop on the reported angle turns a position
request into a rate command, and because the plant is a pure integrator the
closed loop is first-order: no overshoot, no oscillation, and a time constant
set by one gain.

That structure is why the calibration does not need to be accurate for the
loop to *settle*: it was measured settling to within 0.03 % of full travel
with its rate gain set to half the true value. It does need to be accurate for
the loop to be *fast*. The time constant is `1 / gain` only when the
calibration is right; with the gain overstated four times, it is four times
longer, and a steering wheel half a second behind the controller runs wide in
every bend at 60 km/h. That is what the first calibration did -- see
`STEERING_RATE_BY_SPEED`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

__all__ = [
    "STEERING_CALIBRATION",
    "STEERING_CENTRING_BY_SPEED",
    "STEERING_RATE_BY_SPEED",
    "SteeringCalibration",
    "SteeringPositionController",
    "centring_at",
    "rate_gain_at",
]


@dataclass(frozen=True, slots=True)
class SteeringCalibration:
    """How a steering command maps to a rate of change of the applied angle."""

    rate_gain: float
    """`d(effective_steering)/dt` at full command, per second.

    Measured near centre; it falls off towards the end stops, which the loop
    absorbs. Treating it as constant is a deliberate simplification, not an
    oversight.
    """

    sign: float
    """+1 or -1. The game's `steering` mix *subtracts* the semantical input, so
    a positive command turns the wheel negative."""

    def __post_init__(self) -> None:
        if self.rate_gain <= 0.0:
            raise ValueError("rate_gain must be positive")
        if self.sign not in (1.0, -1.0):
            raise ValueError("sign must be +1.0 or -1.0")


#: Measured on ETS2 1.61 with a Mercedes Actros 2014, stationary and at full
#: command; see docs/results/input_path.md. It is right for that and wrong for
#: driving: the game treats the stack's steering like a keyboard's, and a
#: keyboard steers more slowly the faster the truck goes.
STEERING_CALIBRATION: Final = SteeringCalibration(rate_gain=0.75, sign=-1.0)

#: `d(effective_steering)/dt` per unit command, for the small commands a
#: controller actually sends, against speed in m/s. From steering doublets of
#: known size added while driving (`tools/follow.py --excite`), 2026-09-23:
#: 0.25 at 8 km/h, 0.23 at 25-35, 0.21 at 43, 0.19 at 50-60, the same for
#: every amplitude from 0.05 to 0.3. Not extrapolated past 60 km/h: above
#: that the last value holds until it is measured.
STEERING_RATE_BY_SPEED: Final = ((0.0, 0.25), (2.2, 0.25), (8.3, 0.235), (12.0, 0.21), (16.7, 0.19))


STEERING_CENTRING_BY_SPEED: Final = (
    (0.0, 0.0),
    (8.5, 0.5),
    (15.0, 1.11),
    (19.6, 1.17),
    (23.0, 1.21),
)
"""The game pulls the steering back to centre on its own, at this rate per
unit of steering, per second, by speed (m/s):

    d(effective_steering)/dt = sign x gain(v) x command - centring(v) x steering

Fitted to the three adaptive-cruise drives of 2026-09-24, 5000 samples at
10 Hz, R^2 0.79-0.87 above 43 km/h (docs/results/steering_centring.md). Below
18 km/h there was no data; the table assumes it fades to nothing at rest.

Above 40 km/h the time constant is under a second. A controller that treats
the angle as a pure integrator reaches only about 63 % of the angle it asks
for in a steady bend, and the truck runs wide -- 1.5 m at 85 km/h."""


def _lookup(table: tuple[tuple[float, float], ...], speed: float) -> float:
    speed = abs(speed)
    for (s0, g0), (s1, g1) in itertools.pairwise(table):
        if speed <= s1:
            return g0 if speed <= s0 else g0 + (g1 - g0) * (speed - s0) / (s1 - s0)
    return table[-1][1]


def rate_gain_at(speed: float) -> float:
    """The steering actuator's gain at a speed, m/s, interpolated from the table."""
    return _lookup(STEERING_RATE_BY_SPEED, speed)


def centring_at(speed: float) -> float:
    """How hard the game pulls the steering to centre at a speed, per second."""
    return _lookup(STEERING_CENTRING_BY_SPEED, speed)


class SteeringPositionController:
    """Holds `effective_steering` at a requested position.

    Stateless between calls by design. The plant carries all the state -- the
    game's steering angle is the integrator -- so adding any here would mean
    two integrators in series and the overshoot that comes with them.

    A leaky integrator, it turned out: the game pulls the angle back to
    centre (`STEERING_CENTRING_BY_SPEED`). Given the speed, the command adds
    what that pull takes away, rather than an integral term to fight it.

        controller = SteeringPositionController()
        command = controller.command(target=0.1, measured=snapshot_angle)
    """

    def __init__(
        self,
        calibration: SteeringCalibration = STEERING_CALIBRATION,
        *,
        gain: float = 8.0,
        max_command: float = 1.0,
        tolerance: float = 0.0,
    ) -> None:
        """
        Args:
            calibration: the rate mapping.
            gain: proportional gain, in inverse seconds. The closed loop is
                first-order with a time constant of roughly `1 / gain`, so 8
                gives about 125 ms. Raising it far past the frame rate buys
                nothing: the loop cannot react faster than the 60 Hz telemetry
                it reads.
            max_command: clamp on the emitted rate command.
            tolerance: angle error below which the command is zero. Zero by
                default -- the loop converges on its own, and a deadband only
                trades a settled error for a quieter output.
        """
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        if not 0.0 < max_command <= 1.0:
            raise ValueError("max_command must be in (0, 1]")
        if tolerance < 0.0:
            raise ValueError("tolerance must not be negative")
        self._calibration = calibration
        self._gain = gain
        self._max_command = max_command
        self._tolerance = tolerance

    @property
    def calibration(self) -> SteeringCalibration:
        return self._calibration

    def command(self, target: float, measured: float, speed: float | None = None) -> float:
        """Return the steering command that drives `measured` towards `target`.

        Args:
            target: wanted `effective_steering`, in [-1, 1].
            measured: current `effective_steering`, from telemetry.
            speed: m/s. With it, the actuator's gain comes from the measured
                table for that speed, and the game's pull back to centre is
                compensated; without it, the gain is the calibration's, which
                is right only for a stationary truck, and nothing is.

        NaN in either input produces a zero command. A controller feeding NaN
        into this is a bug to fix upstream, but the steering should not move
        because of it -- and note that zero here means "stop turning", not
        "straighten", since the wheel holds its angle.
        """
        if target != target or measured != measured:  # NaN
            return 0.0

        error = target - measured
        if abs(error) <= self._tolerance:
            return 0.0

        if speed is None:
            command = self._calibration.sign * self._gain * error / self._calibration.rate_gain
        else:
            # Ask for the rate the loop wants, plus whatever the game's pull
            # back to centre will take away: without the second term the
            # angle settles where the two balance, well short of the target.
            wanted = self._gain * error + centring_at(speed) * measured
            command = self._calibration.sign * wanted / rate_gain_at(speed)
        return max(-self._max_command, min(self._max_command, command))
