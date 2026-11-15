"""Longitudinal control: holding a target speed.

The shape of this module is dictated by what the vehicle turned out to be able
to do, measured in `docs/results/sysid.md`:

* **Deceleration is quantised.** Releasing the throttle gives 0.22 m/s^2. The
  smallest brake command that does anything gives 1.81. Between those there is
  nothing -- not on the service brake, not on the engine brake, and not on
  partial throttle, which cannot make tractive force negative and so tops out
  at the lift-off figure. Four mechanisms, none of them in the gap.
* **The brake has a dead zone** below roughly 0.055 of command, so a controller
  that ramps the brake up from zero does nothing and then decelerates at
  0.18 g.
* **Throttle response depends on gear**, and the automatic shifts without
  notice, so an open-loop throttle map is a statement about the gearbox.
* **Gradient dominates.** A 5 degree grade contributes 0.86 m/s^2 against
  roughly 0.24 of combined drag and engine braking at 45 km/h.

Two consequences run through everything below.

**Feedforward carries the model, feedback carries the error.** The measured
maps set the operating point; a PI loop absorbs the difference, including gear
changes, load and whatever the map got wrong. Nothing here depends on the maps
being accurate, only on their being roughly right and monotone.

**The controller commits rather than modulates.** Asked for 0.5 m/s^2 of
deceleration it cannot comply, so it chooses: coast, and accept converging
slowly, or brake, and accept overshooting. Switching on the instantaneous
error would chatter between the two, so the choice is made with hysteresis and
held. That is what "predictive rather than proportional" means in practice.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Final

from core.bridge import convert

__all__ = [
    "ACTROS_EMPTY",
    "ActuatorMap",
    "LongitudinalCalibration",
    "LongitudinalCommand",
    "SpeedController",
    "SpeedControllerGains",
    "held_still",
    "mass_ratio",
]


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    """Piecewise-linear lookup, clamped at both ends.

    Clamped rather than extrapolated: beyond the measured range the honest
    answer is "the most I know how to ask for", and extrapolating a saturating
    curve would invent authority the vehicle does not have.
    """
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    index = bisect.bisect_left([p[0] for p in points], x)
    (x0, y0), (x1, y1) = points[index - 1], points[index]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _invert(points: tuple[tuple[float, float], ...], y: float) -> float:
    """Piecewise-linear lookup the other way round. Requires a monotone table."""
    if y <= points[0][1]:
        return points[0][0]
    if y >= points[-1][1]:
        return points[-1][0]
    index = bisect.bisect_left([p[1] for p in points], y)
    (x0, y0), (x1, y1) = points[index - 1], points[index]
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


@dataclass(frozen=True, slots=True)
class LongitudinalCalibration:
    """Measured actuator behaviour for one vehicle and load.

    Every default here came off the recordings in `data/sysid/`, with an empty
    box trailer on flat ground. A loaded trailer will differ and has not been
    measured.
    """

    brake_dead_zone: float = 0.055
    """Brake commands at or below this do nothing at all. Measured: 0.05 gave
    0.02 m/s^2 -- indistinguishable from coasting -- and 0.06 gave 1.81."""

    brake_curve: tuple[tuple[float, float], ...] = (
        (0.06, 1.81),
        (0.08, 2.24),
        (0.10, 2.84),
        (0.12, 3.30),
        (0.14, 3.70),
        (0.15, 4.06),
        (0.25, 5.27),
        (0.40, 6.83),
        (0.60, 8.89),
        (0.85, 9.71),
    )
    """Brake command to deceleration, m/s^2. Monotone, and noticeably steeper
    at the bottom: the first 0.09 of command buys as much as the next 0.7."""

    throttle_curve: tuple[tuple[float, float], ...] = (
        (0.13, 0.0),
        (0.35, 1.01),
        (0.50, 1.21),
        (0.70, 1.37),
        (1.00, 1.50),
    )
    """Throttle command to acceleration, m/s^2, in cruising gears. The zero
    crossing is the equilibrium throttle measured at 50 km/h in tenth gear;
    the rest come from the per-gear sweep at lower speeds. Rough, deliberately
    -- it is a feedforward term, not a plant model."""

    coast_deceleration: float = 0.22
    """Deceleration with the throttle released and the gearbox engaged, on the
    flat. Drag, rolling resistance and engine braking together."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.brake_dead_zone < self.brake_curve[0][0]:
            raise ValueError("brake_dead_zone must sit below the first braking command")
        for name, curve in (("brake", self.brake_curve), ("throttle", self.throttle_curve)):
            if len(curve) < 2:
                raise ValueError(f"{name}_curve needs at least two points")
            commands = [c for c, _ in curve]
            values = [v for _, v in curve]
            if commands != sorted(commands) or values != sorted(values):
                raise ValueError(f"{name}_curve must be monotone in both columns")
        if self.coast_deceleration <= 0.0:
            raise ValueError("coast_deceleration must be positive")

    @property
    def minimum_braking(self) -> float:
        """The least deceleration the service brake can produce, m/s^2.

        Not zero, and that is the whole problem: it is 1.81, eight times what
        coasting gives.
        """
        return self.brake_curve[0][1]

    @property
    def maximum_braking(self) -> float:
        return self.brake_curve[-1][1]

    @property
    def maximum_acceleration(self) -> float:
        return self.throttle_curve[-1][1]


#: Mercedes Actros 2014 with an empty box trailer, ETS2 1.55.0.0.
ACTROS_EMPTY: Final = LongitudinalCalibration()

#: What the calibration was measured at, and what a tractor and an empty box
#: trailer weigh, kg. Estimates: the telemetry gives the cargo, not the rest.
CALIBRATION_MASS_KG: Final = 15000.0
TRACTOR_MASS_KG: Final = 8000.0
EMPTY_TRAILER_MASS_KG: Final = 7000.0


def mass_ratio(*, trailer: bool, cargo_kg: float) -> float:
    """How much heavier the vehicle is than it was when calibrated.

    Brake and engine give forces, not accelerations: the same brake command
    that stopped the empty truck at 1.81 m/s^2 gave 0.87 with 12.4 t of garlic
    on, and the truck ran 7 km/h too fast into a junction turn. The pedals are
    asked for this much more to make up for it.
    """
    mass = TRACTOR_MASS_KG + (EMPTY_TRAILER_MASS_KG + max(cargo_kg, 0.0) if trailer else 0.0)
    return mass / CALIBRATION_MASS_KG


@dataclass(frozen=True, slots=True)
class LongitudinalCommand:
    """What to send to the pedals, and why.

    `reason` exists for the dashboard and for debugging: "why is it not
    slowing down" is the question a quantised actuator provokes, and the
    answer is usually "because coasting was judged sufficient".
    """

    throttle: float
    brake: float
    reason: str

    @property
    def is_braking(self) -> bool:
        return self.brake > 0.0


class ActuatorMap:
    """Turns a desired acceleration into pedal commands.

    Holds the coast-or-brake decision, because that decision is a property of
    the actuators rather than of the controller above them.
    """

    def __init__(
        self,
        calibration: LongitudinalCalibration = ACTROS_EMPTY,
        *,
        brake_engage: float = 0.90,
        brake_release: float = 0.35,
    ) -> None:
        """
        Args:
            calibration: measured actuator behaviour.
            brake_engage: start braking when more deceleration than this is
                wanted, m/s^2. Sits between what coasting gives (0.22) and the
                brake's floor (1.81), so neither choice is ever far wrong.
            brake_release: stop braking when the request softens to less than
                this. Below `brake_engage` on purpose -- without the gap the
                controller would chatter across the boundary every time the
                request wandered, and each crossing is a step of 1.6 m/s^2.
        """
        if not 0.0 < brake_release < brake_engage:
            raise ValueError("brake_release must be positive and below brake_engage")
        self._calibration = calibration
        self._brake_engage = brake_engage
        self._brake_release = brake_release
        self._braking = False
        self.mass_ratio = 1.0
        """Mass against the calibration's (`mass_ratio`); requests are scaled by it."""

    @property
    def calibration(self) -> LongitudinalCalibration:
        return self._calibration

    @property
    def is_braking(self) -> bool:
        """Whether the brake is currently latched on."""
        return self._braking

    def reset(self) -> None:
        """Forget the latched state. Call on engage, not between steps."""
        self._braking = False

    def command_for(self, acceleration: float) -> LongitudinalCommand:
        """Pedal commands for a desired acceleration in m/s^2.

        Positive accelerates. The result will not match the request whenever
        the request falls in the gap between coasting and the brake's floor;
        `reason` says which way it went.
        """
        if acceleration != acceleration:  # NaN
            return LongitudinalCommand(0.0, 0.0, "NaN request; coasting")
        # What the calibrated truck would need to be asked for to do the same.
        acceleration = acceleration * self.mass_ratio

        if acceleration >= 0.0:
            self._braking = False
            throttle = _invert(self._calibration.throttle_curve, acceleration)
            return LongitudinalCommand(throttle, 0.0, "accelerating")

        wanted = -acceleration
        if wanted <= self._calibration.coast_deceleration:
            self._braking = False
            return LongitudinalCommand(0.0, 0.0, "coasting is enough")

        # In the gap. Latch, so the choice is held rather than remade on every
        # frame: each switch is a 1.6 m/s^2 step and the passengers feel it.
        threshold = self._brake_release if self._braking else self._brake_engage
        if wanted < threshold:
            self._braking = False
            return LongitudinalCommand(
                0.0,
                0.0,
                f"coasting: {wanted:.2f} m/s2 wanted, brake floor is "
                f"{self._calibration.minimum_braking:.2f}",
            )

        self._braking = True
        brake = _invert(self._calibration.brake_curve, wanted)
        brake = max(brake, self._calibration.brake_curve[0][0])
        return LongitudinalCommand(0.0, min(brake, 1.0), "braking")

    def acceleration_for(self, throttle: float, brake: float) -> float:
        """What the vehicle will actually do, given pedal commands.

        The forward direction of the same maps, used by the simulated plant in
        the tests and by anything that wants to predict rather than command.
        """
        if brake > self._calibration.brake_dead_zone:
            calibrated = -_interpolate(self._calibration.brake_curve, brake)
        elif throttle > 0.0:
            calibrated = _interpolate(self._calibration.throttle_curve, throttle)
        else:
            calibrated = -self._calibration.coast_deceleration
        return calibrated / self.mass_ratio


@dataclass(frozen=True, slots=True)
class SpeedControllerGains:
    """PI gains, in units of acceleration per unit of speed error."""

    proportional: float = 2.0
    """m/s^2 per m/s of error.

    Stiffer than intuition suggests, and the reason is the quantised actuator.
    A gentle loop lets the speed drift before it commits to braking, so when
    braking finally engages it arrives late and the excursion is large -- the
    same ripple argument as a bang-bang controller, where a higher switching
    rate means a smaller amplitude. Simulated on a downgrade inside the gap,
    a proportional gain of 0.55 gives a 1.08 m/s limit cycle and 2.0 gives
    0.30.

    **This is a starting point, not a tuned value.** The simulation the figure
    came from has no engine lag, no gearbox and no measurement noise, so it
    cannot find the stiffness at which the real vehicle would start to hunt --
    it prefers ever-stiffer gains without limit, which is exactly the sign of a
    model that omits the dynamics that would punish them. Tune against the
    truck."""

    integral: float = 0.25
    """m/s^2 per m/s per second. Absorbs steady error the feedforward misses --
    a gradient, a gear the throttle map does not describe, a load."""

    integral_limit: float = 1.2
    """Clamp on the integral's contribution, m/s^2. Without it a long climb
    that the vehicle simply cannot hold winds the term up, and the truck then
    accelerates away from the target when the hill ends."""


STOP_TARGET: Final = 0.3
"""A target at or below this, m/s, means stop."""

HOLD_BELOW: Final = 3.0
"""Below this speed, asked to stop, the controller holds rather than
modulates, m/s."""

MAX_JERK: Final = 2.5
"""How fast what is felt may change, m/s^3, in ordinary driving. Without a
bound the speed controller passed every wobble of its target straight to
the pedals: following a car in Milan, full throttle and the brake took turns
every second or two, and between two bends the truck went from full throttle
to braking in 0.1 s. At this rate the brake cannot come on until about 0.4 s
after the throttle went off, nor the throttle within a second of a brake."""

URGENT_JERK: Final = 10.0
"""How fast an urgent deceleration builds, m/s^3: firm, from nothing to
5 m/s^2 in half a second, where the pedal had gone from full throttle to
full brake between two frames for a car read as stopped 61 m ahead. A stop
that is really needed loses some 2 m to it at 70 km/h."""

HOLD_BRAKE: Final = 0.2
"""Brake command that finishes a stop and holds it: about 4.7 m/s^2 from
3 m/s, well under a second. Modulating instead, the truck sat at 5 km/h for
three seconds behind a trailer it meant to stop for, the brake flicking
between 0.06 and 0.10 against the engine's idle push in fourth gear."""

STILL_MPS: Final = 0.3


def held_still(
    command: LongitudinalCommand, speed: float, brake_down: bool = True
) -> LongitudinalCommand:
    """`command` as the game's automatic gearbox needs it at a standstill
    with no throttle: it puts a stopped truck into reverse for the brake
    pressed while the truck stands with it up, and backs it on that pedal.

    The brake already down (`brake_down`: the brake last sent) is held at no
    less than `HOLD_BRAKE`: outside Bern a small urgent deceleration for a
    car ahead, under the brake's dead zone, let it go for one frame at a
    stop, it was pressed again, and the truck rolled back. The brake up is
    left up: in Linz a drive begun at rest, the driver's foot off the brake,
    pressed it for one frame before pulling away, and the truck stood in
    reverse with the throttle braking it."""
    if abs(speed) >= STILL_MPS or command.throttle > 0.0:
        return command
    if not brake_down:
        if command.brake <= 0.0:
            return command
        return LongitudinalCommand(0.0, 0.0, command.reason + " (brake left up)")
    if command.brake < HOLD_BRAKE:
        return LongitudinalCommand(0.0, HOLD_BRAKE, command.reason + " (held still)")
    return command


class SpeedController:
    """Holds a target speed.

    Feedforward from the measured maps plus a PI loop on speed error, mapped
    through `ActuatorMap` so that the quantised actuators are handled in one
    place.

        controller = SpeedController()
        command = controller.step(target_speed=12.5, block=snapshot.block, dt=0.016)

    Gradient is fed forward from the pitch channel rather than left to the
    integrator, because it is the largest term at these speeds and the
    integrator is deliberately slow.
    """

    def __init__(
        self,
        calibration: LongitudinalCalibration = ACTROS_EMPTY,
        gains: SpeedControllerGains | None = None,
        *,
        actuators: ActuatorMap | None = None,
        max_acceleration: float = 1.2,
        max_deceleration: float = 2.5,
    ) -> None:
        """
        Args:
            calibration: measured actuator behaviour.
            gains: PI gains.
            actuators: the actuator map, if one is shared or pre-configured.
            max_acceleration: comfort limit on *felt* acceleration, m/s^2.
            max_deceleration: comfort limit on *felt* deceleration, m/s^2.
                2.5 is the top of the comfortable band; the vehicle can reach
                9.7, which is for emergencies and not for this controller.

        Both limits bound the feedback term only. The gradient feedforward is
        whatever the hill requires: holding speed up a slope is not
        acceleration, and capping it there means declining to climb.
        """
        if max_acceleration <= 0.0 or max_deceleration <= 0.0:
            raise ValueError("comfort limits must be positive")
        self._calibration = calibration
        self._gains = gains or SpeedControllerGains()
        self._actuators = actuators or ActuatorMap(calibration)
        self._max_acceleration = max_acceleration
        self._max_deceleration = max_deceleration
        self._integral = 0.0
        self._felt = 0.0
        """The felt acceleration last asked for, m/s^2: where the next may
        change from at `MAX_JERK`."""
        self.last_request = 0.0

    @property
    def actuators(self) -> ActuatorMap:
        return self._actuators

    @property
    def integral(self) -> float:
        return self._integral

    def reset(self) -> None:
        """Clear the loop state. Call when engaging, or after a handover."""
        self._integral = 0.0
        self._felt = 0.0
        self.last_request = 0.0
        self._actuators.reset()

    def step(
        self,
        *,
        target_speed: float,
        speed: float,
        grade: float,
        dt: float,
        coast_above: float | None = None,
    ) -> LongitudinalCommand:
        """One control step.

        Args:
            target_speed: wanted speed, m/s.
            speed: current speed, m/s.
            grade: along-track acceleration from gravity, m/s^2, positive
                downhill -- `convert.grade_acceleration` of the same frame.
            dt: seconds since the last step.
            coast_above: a speed not to accelerate past, m/s, but to coast
                from (`SpeedProfile.coasting_at`): with a bend ahead the
                truck would have to brake for, it lifts off instead of
                gaining speed it would then brake away.

        Returns:
            The pedal commands to publish.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if target_speed <= STOP_TARGET and speed < HOLD_BELOW:
            self._integral = 0.0
            self._felt = 0.0
            self.last_request = -HOLD_BRAKE
            return LongitudinalCommand(0.0, HOLD_BRAKE, "stopping: holding the brake")

        error = target_speed - speed

        proportional = self._gains.proportional * error
        self._integral = _clamp(
            self._integral + self._gains.integral * error * dt,
            -self._gains.integral_limit,
            self._gains.integral_limit,
        )

        # The comfort limits bound what a passenger feels, which is the *net*
        # acceleration -- so they apply to the feedback term alone. Applying
        # them to the total would cap the gradient term too, and a truck
        # holding speed up a hill is not accelerating: clamping there simply
        # refuses to climb. An earlier version did exactly that and came to a
        # standstill on a 1.5 m/s^2 grade.
        desired = proportional + self._integral
        comfortable = _clamp(desired, -self._max_deceleration, self._max_acceleration)
        if coast_above is not None:
            # Up to it, gaining speed ever more gently; past it, lifting off.
            lift_off = self._gains.proportional * (coast_above - speed)
            comfortable = min(comfortable, max(lift_off, -self._calibration.coast_deceleration))
        step = MAX_JERK * dt
        comfortable = _clamp(comfortable, self._felt - step, self._felt + step)
        self._felt = comfortable

        # What the powertrain must produce: the felt acceleration plus whatever
        # gravity is doing. `grade` is positive downhill, hence the sign.
        request = comfortable - grade
        self.last_request = request

        command = self._actuators.command_for(request)
        self._unwind_if_saturated(command, error, dt, clamped=comfortable != desired)
        return command

    def urgent(self, deceleration: float, dt: float) -> LongitudinalCommand:
        """Brake at `deceleration`, m/s^2, past the comfort limit, reached
        at `URGENT_JERK` from what is felt now; ordinary steps after it ease
        out of it at `MAX_JERK`."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._integral = 0.0
        self._felt = max(-deceleration, min(self._felt, 0.0) - URGENT_JERK * dt)
        self.last_request = self._felt
        return self._actuators.command_for(self._felt)

    def step_from_block(
        self, *, target_speed: float, block: object, dt: float
    ) -> LongitudinalCommand:
        """Convenience wrapper reading speed and gradient from a telemetry block."""
        return self.step(
            target_speed=target_speed,
            speed=float(block["truck"]["speed"]),  # type: ignore[index]
            grade=convert.grade_acceleration(block),  # type: ignore[arg-type]
            dt=dt,
        )

    def _unwind_if_saturated(
        self, command: LongitudinalCommand, error: float, dt: float, *, clamped: bool
    ) -> None:
        """Back the integral off whenever the request cannot be delivered.

        Three ways that happens here, and only the first is ordinary:

        * the pedal is already at its stop;
        * the comfort limit clipped the feedback term, which is the common
          case during a long acceleration and, left unhandled, shows up as
          overshoot when the target is finally reached;
        * the request landed in the gap between coasting and the brake's floor
          -- a saturation the vehicle cannot escape, so an unchecked integral
          grows until braking engages and then arrives as a lurch.
        """
        at_pedal_stop = (command.throttle >= 1.0 and error > 0.0) or (
            command.brake >= 1.0 and error < 0.0
        )
        stuck_in_the_gap = (
            not command.is_braking
            and self.last_request < -self._calibration.coast_deceleration
            and error < 0.0
        )
        if at_pedal_stop or clamped or stuck_in_the_gap:
            self._integral -= self._gains.integral * error * dt


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)
