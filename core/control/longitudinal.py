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
from core.control.retarder import PER_STEP_GUESS, RETARDER_FROM_MPS

__all__ = [
    "ACTROS_EMPTY",
    "ActuatorMap",
    "BrakeEstimate",
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
    retarder: int = 0
    """The retarder's step wanted (core.control.retarder); 0 is off."""

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
        self._retarding = 0
        self._retarder_held_s = 0.0
        self.mass_ratio = 1.0
        """Mass against the calibration's (`mass_ratio`); requests are scaled by it."""
        self.retarder_steps = 0
        """The truck's retarder steps (`retarder_step_count`); 0: none."""
        self.retarder_per_step = PER_STEP_GUESS
        """What a step slows the calibrated truck by, m/s^2 (`RetarderEstimate`)."""
        self.brake_ratio: float | None = None
        """The brake's own: how much more the calibrated truck's brake must
        be asked for than it gives this one (`BrakeEstimate`); None: the
        mass's (`mass_ratio`)."""
        self.speed: float | None = None
        """The truck's speed, m/s, if known: the retarder is left off below
        `RETARDER_FROM_MPS`."""

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
        self._retarding = 0
        self._retarder_held_s = 0.0

    def _retarder_reach(self) -> float:
        """How much more than coasting the retarder can slow the calibrated
        truck by now, m/s^2: 0 without one, or too slow for it."""
        if self.retarder_steps <= 0 or (self.speed is not None and self.speed < RETARDER_FROM_MPS):
            return 0.0
        return self.retarder_steps * self.retarder_per_step

    def tick(self, dt: float) -> None:
        """Time goes by: `dt` seconds since the last command."""
        self._retarder_held_s += max(dt, 0.0)

    def _set_retarder(self, step: int) -> int:
        """To `step`, but not down within `RETARDER_HOLD_S` of the last
        change: each change is a press of the lever."""
        if step < self._retarding and self._retarder_held_s < RETARDER_HOLD_S:
            return self._retarding
        if step != self._retarding:
            self._retarding, self._retarder_held_s = step, 0.0
        return self._retarding

    def _retarder_step(self, beyond_coasting: float) -> int:
        """The step that gives `beyond_coasting` m/s^2, held unless another
        is a good deal nearer."""
        exact = beyond_coasting / self.retarder_per_step
        if abs(exact - self._retarding) > RETARDER_STEP_HYSTERESIS:
            return self._set_retarder(max(0, min(round(exact), self.retarder_steps)))
        return self._retarding

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
            if acceleration < RETARDER_OVER_THROTTLE and self._set_retarder(0) > 0:
                # A step just taken is held, and not pulled against: on the
                # way to Eindhoven the lever went 1, 0, 1, 0 every 0.2 s, the
                # throttle full between (18 November).
                return LongitudinalCommand(
                    0.0, 0.0, "retarder held: coasting", retarder=self._retarding
                )
            self._retarding, self._retarder_held_s = 0, 0.0
            throttle = _invert(self._calibration.throttle_curve, acceleration)
            return LongitudinalCommand(throttle, 0.0, "accelerating")

        wanted = -acceleration
        coast = self._calibration.coast_deceleration
        if wanted <= coast:
            self._braking = False
            step = self._set_retarder(0)
            reason = "retarder held" if step > 0 else "coasting is enough"
            return LongitudinalCommand(0.0, 0.0, reason, retarder=step)

        # In the gap. Latch, so the choice is held rather than remade on every
        # frame: each switch is a 1.6 m/s^2 step and the passengers feel it.
        # A retarder fills the gap as far as it reaches, and the brake is
        # only for past that.
        reach = self._retarder_reach()
        engage = max(self._brake_engage, min(coast + reach, self._calibration.minimum_braking))
        release = engage - (self._brake_engage - self._brake_release)
        threshold = release if self._braking else engage
        if wanted < threshold:
            self._braking = False
            if reach > 0.0:
                step = self._retarder_step(wanted - coast)
                if step > 0:
                    return LongitudinalCommand(
                        0.0,
                        0.0,
                        f"retarder {step} of {self.retarder_steps}: {wanted:.2f} m/s2 wanted",
                        retarder=step,
                    )
            else:
                self._retarding = 0
            return LongitudinalCommand(
                0.0,
                0.0,
                f"coasting: {wanted:.2f} m/s2 wanted, brake floor is "
                f"{self._calibration.minimum_braking:.2f}",
            )

        self._braking = True
        # The retarder stays where it is, and the brake gives the rest: let
        # off at each touch of the brake, it was worked to and fro.
        step = self._retarding if self._retarder_reach() > 0.0 else 0
        wanted = max(wanted - step * self.retarder_per_step, 0.0)
        ratio = self.brake_ratio if self.brake_ratio is not None else brake_prior(self.mass_ratio)
        if self.mass_ratio > 0.0:
            # The brake as this truck has been seen to brake, or a guess
            # short of the load's: not as the load is taken to make it.
            wanted = wanted / self.mass_ratio * ratio
        brake = _invert(self._calibration.brake_curve, wanted)
        brake = max(brake, self._calibration.brake_curve[0][0])
        return LongitudinalCommand(0.0, min(brake, 1.0), "braking", retarder=step)

    def acceleration_for(self, throttle: float, brake: float, retarder: int = 0) -> float:
        """What the vehicle will actually do, given pedal commands and the
        retarder's step.

        The forward direction of the same maps, used by the simulated plant in
        the tests and by anything that wants to predict rather than command.
        """
        if brake > self._calibration.brake_dead_zone:
            calibrated = -_interpolate(self._calibration.brake_curve, brake)
            retarding = retarder * self.retarder_per_step / self.mass_ratio
            return calibrated / (self.brake_ratio or self.mass_ratio) - retarding
        elif throttle > 0.0:
            calibrated = _interpolate(self._calibration.throttle_curve, throttle)
        else:
            calibrated = -self._calibration.coast_deceleration - retarder * self.retarder_per_step
        return calibrated / self.mass_ratio


@dataclass
class BrakeEstimate:
    """How much harder the calibrated truck's brake must be asked for than
    it gives this one, learned from what the truck does braking: the load
    guessed from the cargo alone (`mass_ratio`) was 1.42 on a Scania R near
    Eindhoven, and its brake gave 25 to 40 % more than that allowed for --
    2.5 m/s^2 asked for, 3.5 to 4.9 given (18 November)."""

    ratio: float | None = None
    seen_s: float = 0.0
    _brake: float = 0.0
    _steady_s: float = 0.0

    def observe(
        self,
        *,
        brake: float,
        throttle: float,
        acceleration: float,
        grade: float,
        speed: float,
        calibration: LongitudinalCalibration,
        dt: float,
    ) -> None:
        """One frame of telemetry: the brake and throttle the game applied,
        the acceleration along the way and the grade (positive downhill)."""
        if abs(brake - self._brake) > BRAKE_STEADY_BAND:
            self._brake, self._steady_s = brake, 0.0
            return
        self._steady_s += dt
        if (
            brake <= calibration.brake_dead_zone
            or throttle > THROTTLE_OFF
            or speed < BRAKE_LEARN_FROM_MPS
            or self._steady_s < BRAKE_STEADY_S
            or dt <= 0.0
        ):
            return
        slowing = grade - acceleration
        if slowing <= 0.1:
            return
        seen = _interpolate(calibration.brake_curve, brake) / slowing
        if not BRAKE_LEAST_RATIO <= seen <= BRAKE_MOST_RATIO:
            return
        if self.ratio is None:
            self.ratio = seen
        else:
            self.ratio += min(dt / BRAKE_LEARN_TIME_S, 1.0) * (seen - self.ratio)
        self.seen_s += dt


THROTTLE_OFF: Final = 0.02
"""The game's throttle, let go, reads a little over nothing: 1e-9 to 0.004."""

BRAKE_STEADY_BAND: Final = 0.01
BRAKE_STEADY_S: Final = 0.3
"""A brake held within this of one command this long is judged: the game
takes a moment to bring the brake on."""
BRAKE_LEARN_FROM_MPS: Final = 3.0
BRAKE_LEARN_TIME_S: Final = 3.0
BRAKE_LEAST_RATIO: Final = 0.4
BRAKE_MOST_RATIO: Final = 4.0


def brake_prior(mass: float) -> float:
    """What the brake is taken to need, against the calibrated truck, until
    seen (`BrakeEstimate`): half the load's guess beyond one. Seen, it was
    1.05 for a guessed 1.42 (a Scania R, 6.3 t), 1.81 for 3.14, 2.08 for
    1.83 (12.4 t of garlic): taken at the load's, the first brake of a drive
    was 4.3 m/s^2 for 2.5 asked for. Short of it, the speed controller and a
    stop's own working out ask for more."""
    return 1.0 + 0.5 * max(mass - 1.0, 0.0)


RETARDER_HOLD_S: Final = 1.5
"""A retarder step is held this long before the lever goes back down."""

RETARDER_OVER_THROTTLE: Final = 0.5
"""Asked to speed up by less than this, m/s^2 for the calibrated truck,
with a step just taken, the truck coasts with it on instead."""

RETARDER_STEP_HYSTERESIS: Final = 0.75
"""How much nearer, in steps, another retarder step must be to be moved to:
a request wandering about half-way between two steps is not to work the
lever back and forth."""


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
    command: LongitudinalCommand, speed: float, brake_down: bool = True, reversed: bool = False
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
    reverse with the throttle braking it.

    `reversed`: the gearbox has gone into reverse at rest all the same --
    near Linz within half a second of stopping on the brake held -- and the
    brake now drives the truck back: let go of, the truck held on the
    parking brake instead (tools/follow.py)."""
    if reversed and command.throttle <= 0.0:
        return LongitudinalCommand(0.0, 0.0, command.reason + " (in reverse: brake let go)")
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
        self._actuators.speed = speed
        self._actuators.tick(dt)

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
        self._actuators.tick(dt)
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
