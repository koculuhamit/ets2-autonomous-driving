"""Noticing when a human takes over.

The plugin's input device *adds* to the game's control mixes rather than
replacing them, so a driver can steer or brake while the stack is driving
without disengaging anything. That is the emergency path, and it was confirmed
in real conditions -- the operator steered away from the verge during the first
autonomous speed hold and it worked.

It also revealed the problem this module solves. The steering loop was holding
the wheel at zero, so it undid every correction the operator made. Eight times,
each pulled back within a fraction of a second. Someone steering to avoid a
collision should be handed the wheel, not argued with.

Detection differs by control, because the telemetry does:

* **Pedals are exact.** Commanding 0.600 of brake reads back as 0.600 in
  `input_brake`. Any sustained difference is another pair of feet.
* **Steering is not.** `input_steering` saturates to +/-1 for any non-zero
  command -- the game reports a direction, not a magnitude. So steering is
  caught through its *rate*: the commanded value sets how fast the wheel turns,
  we know what we asked for, and a wheel moving differently is being moved by
  someone else.

Everything is judged over a dwell time rather than instantaneously. The command
takes 15-28 ms to arrive, so a fresh command always disagrees with telemetry
for a frame or two, and a detector without dwell would fire on every change of
its own making.

**Validation status, honestly.** The pedal test rests on a direct measurement:
commanding 0.600 of brake reads back as exactly 0.600. The steering test does
not. Checked against the one recording that contains real manual steering, it
separates poorly -- the residual during intervention averaged 0.22 against a
99th percentile of 0.39 during normal driving -- because a position loop reacts
to the disturbance within a fraction of a second and folds it into the expected
rate, and because the plant gain is only roughly known.

Two things follow. The tolerance scales with the size of our own command, since
that is where the model error lives; and the detector is trustworthy mainly
when we are *not* steering hard, where the expected rate is small and the
residual is unambiguous. Establishing it properly needs a purpose-built run:
hold a fixed angle, intervene at recorded times, and compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.control.steering import STEERING_CALIBRATION, SteeringCalibration

__all__ = ["OverrideDetector", "OverrideEvent"]

#: Pedal difference that counts as someone else pressing, in command units.
DEFAULT_PEDAL_TOLERANCE: Final = 0.08

#: Unexplained steering rate that counts as someone else steering, in applied
#: angle per second, when we are commanding nothing. This is the case the test
#: is reliable in: with a zero command the expected rate is zero and any
#: movement is external, with no model in the way.
DEFAULT_STEERING_TOLERANCE: Final = 0.08

#: Additional tolerance proportional to our own commanded rate. The plant gain
#: is known to about 0.75 near centre and falls off towards the stops, so the
#: harder we are steering the less the residual means.
DEFAULT_STEERING_TOLERANCE_FRACTION: Final = 0.6

#: How long a disagreement must persist. Longer than the command latency
#: (15-28 ms) and than a single dropped frame, short enough to be out of the
#: way before a human has finished their input.
DEFAULT_DWELL: Final = 0.12


@dataclass(frozen=True, slots=True)
class OverrideEvent:
    """A control the stack did not move."""

    control: str
    """`"steering"`, `"throttle"` or `"brake"`."""

    magnitude: float
    """How far off the observation was: command units for pedals, applied
    angle per second for steering."""

    def __str__(self) -> str:
        return f"manual {self.control} ({self.magnitude:+.3f})"


class OverrideDetector:
    """Watches for a human on the controls.

    One instance per engagement. `reset()` on engage, and on disengage, so a
    handover does not leave a stale dwell timer armed.
    """

    def __init__(
        self,
        calibration: SteeringCalibration = STEERING_CALIBRATION,
        *,
        pedal_tolerance: float = DEFAULT_PEDAL_TOLERANCE,
        steering_tolerance: float = DEFAULT_STEERING_TOLERANCE,
        steering_tolerance_fraction: float = DEFAULT_STEERING_TOLERANCE_FRACTION,
        dwell: float = DEFAULT_DWELL,
    ) -> None:
        if pedal_tolerance <= 0.0 or steering_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        if steering_tolerance_fraction < 0.0:
            raise ValueError("steering_tolerance_fraction must not be negative")
        if dwell < 0.0:
            raise ValueError("dwell must not be negative")
        self._calibration = calibration
        self._pedal_tolerance = pedal_tolerance
        self._steering_tolerance = steering_tolerance
        self._steering_tolerance_fraction = steering_tolerance_fraction
        self._dwell = dwell

        self._previous_angle: float | None = None
        self._elapsed: dict[str, float] = {}

    def reset(self) -> None:
        """Forget the dwell timers and the last steering angle."""
        self._previous_angle = None
        self._elapsed.clear()

    def update(
        self,
        *,
        commanded_steer: float,
        commanded_throttle: float,
        commanded_brake: float,
        measured_angle: float,
        input_throttle: float,
        input_brake: float,
        dt: float,
    ) -> OverrideEvent | None:
        """Check one frame.

        Args:
            commanded_steer: what we sent on the steering axis, a *rate*.
            commanded_throttle: what we sent on the throttle.
            commanded_brake: what we sent on the brake.
            measured_angle: `effective_steering` from telemetry.
            input_throttle: `input_throttle` from telemetry.
            input_brake: `input_brake` from telemetry.
            dt: seconds since the previous call.

        Returns:
            The largest disagreement that has persisted for the dwell time, or
            None.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        candidates = self._pedal_candidates(
            commanded_throttle, commanded_brake, input_throttle, input_brake, dt
        )

        steering = self._steering_residual(commanded_steer, measured_angle, dt)
        expected_rate = self._calibration.sign * self._calibration.rate_gain * commanded_steer
        allowance = self._steering_tolerance + self._steering_tolerance_fraction * abs(
            expected_rate
        )
        if steering is not None and abs(steering) > allowance:
            candidates.append(OverrideEvent("steering", steering))
            self._elapsed["steering"] = self._elapsed.get("steering", 0.0) + dt
        else:
            self._elapsed["steering"] = 0.0

        sustained = [c for c in candidates if self._elapsed.get(c.control, 0.0) >= self._dwell]
        if not sustained:
            return None
        return max(sustained, key=lambda event: abs(event.magnitude))

    def update_pedals(
        self,
        *,
        commanded_throttle: float,
        commanded_brake: float,
        input_throttle: float,
        input_brake: float,
        dt: float,
    ) -> OverrideEvent | None:
        """`update` for the pedals alone, which are exact: the steering test
        is not trusted enough to hand the truck over on. Only a pedal pressed
        further than sent counts -- a foot adds to the stack's inputs -- not
        one applied less: the game took no inputs for a moment as a drive
        began, and the gearbox lets the throttle go for each shift."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        candidates = self._pedal_candidates(
            commanded_throttle, commanded_brake, input_throttle, input_brake, dt, pressed=True
        )
        sustained = [c for c in candidates if self._elapsed.get(c.control, 0.0) >= self._dwell]
        return max(sustained, key=lambda event: abs(event.magnitude)) if sustained else None

    def _pedal_candidates(
        self,
        commanded_throttle: float,
        commanded_brake: float,
        input_throttle: float,
        input_brake: float,
        dt: float,
        *,
        pressed: bool = False,
    ) -> list[OverrideEvent]:
        candidates: list[OverrideEvent] = []
        for control, commanded, observed in (
            ("throttle", commanded_throttle, input_throttle),
            ("brake", commanded_brake, input_brake),
        ):
            difference = observed - commanded
            if (difference if pressed else abs(difference)) > self._pedal_tolerance:
                candidates.append(OverrideEvent(control, difference))
                self._elapsed[control] = self._elapsed.get(control, 0.0) + dt
            else:
                self._elapsed[control] = 0.0
        return candidates

    def _steering_residual(
        self, commanded: float, measured_angle: float, dt: float
    ) -> float | None:
        """How much of the wheel's movement we did not ask for, per second.

        None on the first call, which has no previous angle to difference
        against.
        """
        previous, self._previous_angle = self._previous_angle, measured_angle
        if previous is None:
            return None
        observed_rate = (measured_angle - previous) / dt
        expected_rate = self._calibration.sign * self._calibration.rate_gain * commanded
        return observed_rate - expected_rate
