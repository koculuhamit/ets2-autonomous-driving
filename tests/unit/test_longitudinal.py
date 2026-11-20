"""Tests for longitudinal control.

Most of these run a simulated vehicle built from the same measured maps the
controller uses. That makes them a test of the control logic rather than of the
model -- the plant cannot disagree with the controller about physics, so what
is left to fail is the loop, the anti-windup and the coast-or-brake decision.

The plant's limitations are the point of several tests. Deceleration between
0.22 and 1.81 m/s^2 is not available, so a gradient needing 0.5 produces a
limit cycle; the tests pin how large it is rather than pretending it is absent.
"""

from __future__ import annotations

import itertools
from collections import deque

import pytest

from core.control.longitudinal import (
    ACTROS_EMPTY,
    ActuatorMap,
    LongitudinalCalibration,
    SpeedController,
    SpeedControllerGains,
)

#: Command-to-effect delay, in 60 Hz frames. Measured at 15-28 ms median
#: across the four axes; see docs/results/input_path.md.
LATENCY_FRAMES = 2


class Vehicle:
    """A truck that obeys the measured maps, after the measured delay.

    `grade` is along-track acceleration from gravity, positive downhill, the
    same convention as `convert.grade_acceleration`.

    The delay matters: without it a stiffer loop looks unconditionally better,
    and the tuning argument below would be drawn from a plant that cannot
    punish stiffness. It still cannot fully -- there is no engine lag, no
    gearbox and no noise here -- which is why the gains are called a starting
    point rather than a result.
    """

    def __init__(self, speed: float = 0.0, grade: float = 0.0) -> None:
        self.speed = speed
        self.grade = grade
        self._map = ActuatorMap()
        self._pending: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * LATENCY_FRAMES, maxlen=LATENCY_FRAMES
        )

    def apply(self, throttle: float, brake: float, dt: float) -> None:
        self._pending.append((throttle, brake))
        applied_throttle, applied_brake = self._pending[0]
        acceleration = self._map.acceleration_for(applied_throttle, applied_brake) + self.grade
        self.speed = max(0.0, self.speed + acceleration * dt)


def run(
    controller: SpeedController,
    vehicle: Vehicle,
    target: float,
    seconds: float,
    dt: float = 1.0 / 60.0,
) -> list[float]:
    """Drive the simulated vehicle and return its speed history."""
    history = []
    for _ in range(int(seconds / dt)):
        command = controller.step(
            target_speed=target, speed=vehicle.speed, grade=vehicle.grade, dt=dt
        )
        vehicle.apply(command.throttle, command.brake, dt)
        history.append(vehicle.speed)
    return history


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_minimum_braking_is_far_above_coasting() -> None:
    """The fact the whole module is shaped around: the brake's floor is eight
    times what coasting gives, with nothing in between."""
    assert ACTROS_EMPTY.minimum_braking == pytest.approx(1.81)
    assert ACTROS_EMPTY.coast_deceleration == pytest.approx(0.22)
    assert ACTROS_EMPTY.minimum_braking > 5 * ACTROS_EMPTY.coast_deceleration


def test_curves_must_be_monotone() -> None:
    with pytest.raises(ValueError, match="monotone"):
        LongitudinalCalibration(brake_curve=((0.1, 2.0), (0.2, 1.0)))


def test_dead_zone_must_sit_below_the_first_braking_command() -> None:
    with pytest.raises(ValueError, match="dead_zone"):
        LongitudinalCalibration(brake_dead_zone=0.5)


# --------------------------------------------------------------------------- #
# ActuatorMap
# --------------------------------------------------------------------------- #


def test_positive_request_uses_throttle_only() -> None:
    command = ActuatorMap().command_for(1.0)
    assert command.throttle > 0.0
    assert command.brake == 0.0


def test_small_deceleration_coasts() -> None:
    command = ActuatorMap().command_for(-0.1)
    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_brake_is_never_commanded_inside_the_dead_zone() -> None:
    """A command in the dead zone does nothing, so asking for one is asking for
    no braking while believing otherwise."""
    actuators = ActuatorMap()
    for request in [-x / 10 for x in range(1, 120)]:
        brake = actuators.command_for(request).brake
        assert brake == 0.0 or brake >= ACTROS_EMPTY.brake_curve[0][0]
        assert not (0.0 < brake <= ACTROS_EMPTY.brake_dead_zone)


def test_large_deceleration_brakes() -> None:
    command = ActuatorMap().command_for(-5.0)
    assert command.brake > 0.0
    assert command.throttle == 0.0


def test_brake_command_increases_with_demand() -> None:
    actuators = ActuatorMap()
    gentle = actuators.command_for(-2.5).brake
    firm = actuators.command_for(-6.0).brake
    assert firm > gentle


def test_brake_saturates_at_measured_authority() -> None:
    """Beyond the measured range the honest answer is the most we know how to
    ask for; extrapolating invents authority the vehicle does not have."""
    command = ActuatorMap().command_for(-50.0)
    assert command.brake == pytest.approx(ACTROS_EMPTY.brake_curve[-1][0])


def test_nan_request_coasts() -> None:
    command = ActuatorMap().command_for(float("nan"))
    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_the_gap_is_decided_with_hysteresis() -> None:
    """Each crossing of the boundary is a 1.6 m/s^2 step, so the decision is
    latched: without the gap between engage and release, a request wandering
    across the threshold would chatter."""
    actuators = ActuatorMap(brake_engage=0.9, brake_release=0.35)

    assert actuators.command_for(-0.5).brake == 0.0  # below engage: coast
    assert actuators.command_for(-1.0).brake > 0.0  # above engage: brake
    assert actuators.command_for(-0.5).brake > 0.0  # latched: still braking
    assert actuators.command_for(-0.2).brake == 0.0  # below release: let go
    assert actuators.command_for(-0.5).brake == 0.0  # latched off again


def test_reset_clears_the_latch() -> None:
    actuators = ActuatorMap()
    actuators.command_for(-3.0)
    assert actuators.is_braking is True
    actuators.reset()
    assert actuators.is_braking is False


def test_release_must_be_below_engage() -> None:
    with pytest.raises(ValueError, match="brake_release"):
        ActuatorMap(brake_engage=0.5, brake_release=0.9)


def test_acceleration_for_round_trips_through_the_map() -> None:
    actuators = ActuatorMap()
    for request in (1.2, 0.5, -3.0, -6.0):
        command = actuators.command_for(request)
        produced = actuators.acceleration_for(command.throttle, command.brake)
        assert (produced >= 0) == (request >= 0)


# --------------------------------------------------------------------------- #
# Closed loop
# --------------------------------------------------------------------------- #


def test_reaches_the_target_from_rest() -> None:
    controller, vehicle = SpeedController(), Vehicle(speed=0.0)
    history = run(controller, vehicle, target=12.5, seconds=40.0)
    assert history[-1] == pytest.approx(12.5, abs=0.3)


def test_settles_without_sustained_overshoot() -> None:
    controller, vehicle = SpeedController(), Vehicle(speed=0.0)
    history = run(controller, vehicle, target=12.5, seconds=60.0)
    settled = history[-600:]  # last ten seconds
    assert max(settled) <= 12.5 + 0.5
    assert min(settled) >= 12.5 - 0.5


def test_slows_to_a_lower_target() -> None:
    controller, vehicle = SpeedController(), Vehicle(speed=20.0)
    history = run(controller, vehicle, target=10.0, seconds=40.0)
    assert history[-1] == pytest.approx(10.0, abs=0.4)


def test_holds_speed_up_a_gradient() -> None:
    """The gradient feedforward carries this; the integrator is too slow to."""
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=-0.8)  # uphill
    history = run(controller, vehicle, target=12.5, seconds=30.0)
    assert history[-1] == pytest.approx(12.5, abs=0.5)


def test_holds_speed_down_a_steep_gradient() -> None:
    """Steep enough that braking is genuinely required, so the actuator can
    comply and the speed holds."""
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=2.5)
    history = run(controller, vehicle, target=12.5, seconds=30.0)
    assert history[-1] == pytest.approx(12.5, abs=1.0)


def test_a_gradient_inside_the_gap_produces_a_bounded_limit_cycle() -> None:
    """Honest about the one case the vehicle cannot serve.

    A downgrade needing 0.5 m/s^2 to hold speed falls between coasting (0.22)
    and the brake's floor (1.81). The controller alternates, and the speed
    oscillates. What matters is that the oscillation is bounded and modest --
    not that it is absent, because no command exists that would make it so.
    """
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=0.5)
    history = run(controller, vehicle, target=12.5, seconds=90.0)
    settled = history[-1800:]  # last thirty seconds
    excursion = max(settled) - min(settled)
    assert excursion < 4.0, f"limit cycle of {excursion:.1f} m/s is too wide"
    assert abs(sum(settled) / len(settled) - 12.5) < 1.5


def test_integral_does_not_wind_up_on_an_impossible_climb() -> None:
    """A hill the truck cannot hold would otherwise charge the integrator, and
    the stored error arrives as acceleration the moment the hill ends."""
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=-5.0)  # steeper than full throttle
    run(controller, vehicle, target=12.5, seconds=60.0)
    assert abs(controller.integral) <= controller._gains.integral_limit + 1e-9


def test_integral_does_not_wind_up_while_stuck_in_the_gap() -> None:
    """The gap is a saturation the vehicle cannot escape. Without unwinding,
    the request keeps growing and arrives as a lurch when braking engages."""
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=0.5)
    run(controller, vehicle, target=12.5, seconds=60.0)
    assert abs(controller.integral) < controller._gains.integral_limit


def test_felt_acceleration_stays_inside_the_comfort_limits() -> None:
    """On the flat, where the request and the felt acceleration are the same
    thing, the limits bound what the passenger gets."""
    controller = SpeedController(max_acceleration=1.0, max_deceleration=2.0)
    vehicle = Vehicle(speed=0.0)
    for _ in range(1200):
        command = controller.step(target_speed=25.0, speed=vehicle.speed, grade=0.0, dt=1 / 60)
        assert -2.0 - 1e-9 <= controller.last_request <= 1.0 + 1e-9
        vehicle.apply(command.throttle, command.brake, 1 / 60)


def test_comfort_limit_does_not_cap_the_gradient_term() -> None:
    """Holding speed up a hill is not acceleration.

    An earlier version clamped the total request, gradient feedforward
    included, and so refused to produce more than the comfort limit however
    steep the road. It came to a standstill on a 1.5 m/s^2 grade while
    reporting that it was obeying a comfort constraint.
    """
    controller = SpeedController(max_acceleration=1.2)
    vehicle = Vehicle(speed=12.5, grade=-1.5)  # needs 1.5 just to hold
    history = run(controller, vehicle, target=12.5, seconds=40.0)
    assert history[-1] == pytest.approx(12.5, abs=0.2)
    assert controller.last_request > 1.2, "the gradient term was capped"


def test_a_climb_beyond_the_vehicle_is_reported_by_slowing_not_by_pretending() -> None:
    """Full throttle produces 1.5 m/s^2; a 5 m/s^2 grade is simply beyond it.
    The truck slows, which is the truthful outcome."""
    controller = SpeedController()
    vehicle = Vehicle(speed=12.5, grade=-5.0)
    history = run(controller, vehicle, target=12.5, seconds=40.0)
    assert history[-1] < history[0]


def test_the_integral_absorbs_a_wrong_throttle_map() -> None:
    """The point of the feedforward/feedback split.

    The throttle map came off a handful of runs in low gears and is a rough
    guide, not a plant model. Here the controller is given one that is wrong by
    a third and still holds the target: the error the map leaves behind is what
    the integral is for.
    """
    optimistic = LongitudinalCalibration(
        throttle_curve=((0.13, 0.0), (0.35, 1.5), (0.50, 1.8), (0.70, 2.0), (1.00, 2.2)),
    )
    controller = SpeedController(optimistic)  # believes the truck is stronger
    vehicle = Vehicle(speed=12.0)  # obeys the real map
    history = run(controller, vehicle, target=12.5, seconds=60.0)
    assert history[-1] == pytest.approx(12.5, abs=0.2)
    assert controller.integral != 0.0, "the integral should be carrying the model error"


def test_reset_clears_the_loop_state() -> None:
    controller = SpeedController()
    vehicle = Vehicle(speed=20.0)
    run(controller, vehicle, target=8.0, seconds=2.0)  # mid-deceleration, latch set
    assert controller.actuators.is_braking is True
    controller.reset()
    assert controller.integral == 0.0
    assert controller.last_request == 0.0
    assert controller.actuators.is_braking is False


def test_zero_dt_is_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        SpeedController().step(target_speed=10.0, speed=5.0, grade=0.0, dt=0.0)


def test_stiffer_gains_shrink_the_limit_cycle() -> None:
    """The opposite of the intuition this module started with.

    A gentle loop seemed right for a quantised actuator -- why ask for
    corrections it cannot make? -- but it lets the speed drift before
    committing, so braking arrives late and the excursion is large. Faster
    commitment means a smaller amplitude, the same argument as switching
    frequency in a bang-bang controller. Measured, not assumed: 0.55 gives a
    1.08 m/s limit cycle here and 2.0 gives 0.33.
    """
    gentle = SpeedController(gains=SpeedControllerGains(proportional=0.55, integral=0.06))
    stiff = SpeedController(gains=SpeedControllerGains(proportional=2.0, integral=0.25))
    gentle_history = run(gentle, Vehicle(speed=12.0, grade=0.5), target=12.5, seconds=90.0)
    stiff_history = run(stiff, Vehicle(speed=12.0, grade=0.5), target=12.5, seconds=90.0)
    gentle_spread = max(gentle_history[-1800:]) - min(gentle_history[-1800:])
    stiff_spread = max(stiff_history[-1800:]) - min(stiff_history[-1800:])
    assert stiff_spread < gentle_spread / 2


def test_asked_to_stop_at_walking_pace_it_holds_the_brake() -> None:
    """What happened: modulating near zero, the truck sat at 5 km/h for three
    seconds behind a trailer, the brake flicking between 0.06 and 0.10."""
    from core.control.longitudinal import HOLD_BRAKE, SpeedController

    controller = SpeedController()
    for speed in (1.4, 0.5, 0.0):
        command = controller.step(target_speed=0.0, speed=speed, grade=0.0, dt=1 / 60)
        assert command.brake == HOLD_BRAKE and command.throttle == 0.0
    moving = controller.step(target_speed=0.0, speed=10.0, grade=0.0, dt=1 / 60)
    assert moving.brake < HOLD_BRAKE or moving.reason != "stopping: holding the brake"


def test_a_loaded_truck_is_braked_harder_for_the_same_deceleration() -> None:
    """In the game: 12.4 t of cargo, the brake's floor gave 0.87 m/s^2 for the
    1.81 it gives empty, and the truck ran 7 km/h too fast into a turn."""
    from core.control.longitudinal import mass_ratio

    loaded = mass_ratio(trailer=True, cargo_kg=12400.0)
    assert loaded == pytest.approx(1.83, abs=0.01)
    assert mass_ratio(trailer=False, cargo_kg=12400.0) < 1.0, "bobtail: the cargo is not on"

    empty, heavy = ActuatorMap(), ActuatorMap()
    heavy.mass_ratio = loaded
    assert heavy.command_for(-1.5).brake > empty.command_for(-1.5).brake
    # And once its brake is known (core.control.longitudinal.BrakeEstimate),
    # what the heavy truck then does is what was asked for.
    heavy.brake_ratio = loaded
    command = heavy.command_for(-2.0)
    assert heavy.acceleration_for(command.throttle, command.brake) == pytest.approx(-2.0, abs=0.1)


def test_loaded_the_truck_slows_for_a_corner_as_it_does_empty() -> None:
    """40 to 20 km/h ahead of a junction turn, planned at 1.5 m/s^2."""
    from core.control.longitudinal import mass_ratio

    def slow_down(ratio_known: float, ratio_real: float) -> float:
        controller = SpeedController()
        controller.actuators.mass_ratio = ratio_known
        plant = ActuatorMap()
        plant.mass_ratio = ratio_real
        dt, speed, worst = 1 / 60, 40 / 3.6, 0.0
        for k in range(int(12 / dt)):
            target = max(20 / 3.6, 40 / 3.6 - 1.5 * k * dt)
            command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
            speed += plant.acceleration_for(command.throttle, command.brake) * dt
            worst = max(worst, speed - target)
        return worst * 3.6

    loaded = mass_ratio(trailer=True, cargo_kg=12400.0)
    assert slow_down(loaded, loaded) < 3.0
    assert slow_down(1.0, loaded) > slow_down(loaded, loaded) + 1.0, "unscaled, it lags"


def test_a_target_that_wobbles_does_not_take_the_pedals_from_throttle_to_brake() -> None:
    """What happened in Milan, following a car: the target wobbled by a few
    km/h, and full throttle and the brake took turns every second or two."""
    from core.control.longitudinal import MAX_JERK

    controller, vehicle = SpeedController(), Vehicle(speed=19.0)
    dt, felt, switches, was = 1 / 60, [], 0, None
    for k in range(int(20 / dt)):
        wobble = 1.0 if (k * dt) % 1.0 < 0.5 else -1.0  # +-3.6 km/h every half second
        command = controller.step(target_speed=19.0 + wobble, speed=vehicle.speed, grade=0.0, dt=dt)
        felt.append(controller.last_request)
        pedal = "brake" if command.brake > 0 else "throttle" if command.throttle > 0.13 else None
        if pedal is not None and was is not None and pedal != was:
            switches += 1
        was = pedal or was
        vehicle.apply(command.throttle, command.brake, dt)
    assert max(abs(b - a) for a, b in itertools.pairwise(felt)) <= MAX_JERK * dt + 1e-9
    assert switches == 0, f"{switches} switches between throttle and brake"


def test_short_of_a_bend_it_lifts_off_rather_than_gain_speed_to_brake_away() -> None:
    """Between two bends out of Milan the profile rose from 48 to 60 km/h for
    a moment: full throttle, then the brake at 0.14."""
    controller = SpeedController()
    command = controller.step(target_speed=60 / 3.6, speed=45 / 3.6, grade=0.0, dt=1.0)
    assert command.throttle > 0.13
    controller.reset()
    command = controller.step(
        target_speed=60 / 3.6, speed=45 / 3.6, grade=0.0, dt=1.0, coast_above=44 / 3.6
    )
    assert command.throttle == 0.0 and command.brake == 0.0, "coasting"
    # Up a hill, lifting off means holding about the speed, not losing it fast.
    controller.reset()
    command = controller.step(
        target_speed=60 / 3.6, speed=45 / 3.6, grade=-1.0, dt=1.0, coast_above=44 / 3.6
    )
    assert command.throttle > 0.13


def test_an_urgent_stop_builds_up_instead_of_slamming_the_brake() -> None:
    """For a car read as stopped 61 m ahead the pedal went from full throttle
    to full brake between two frames."""
    from core.control.longitudinal import URGENT_JERK

    controller = SpeedController()
    first = controller.urgent(5.0, dt=1 / 60)
    assert controller.last_request == pytest.approx(-URGENT_JERK / 60)
    assert first.brake == 0.0, "under the brake's floor yet"
    for _ in range(60):
        command = controller.urgent(5.0, dt=1 / 60)
    assert controller.last_request == pytest.approx(-5.0)
    assert command.brake > 0.2
    # Over, it eases out rather than letting go at once.
    after = controller.step(target_speed=20.0, speed=15.0, grade=0.0, dt=1 / 60)
    assert after.brake > 0.0


def test_at_a_stop_the_brake_is_never_let_go_and_pressed_again() -> None:
    """Outside Bern a small urgent deceleration, under the brake's dead zone,
    let the brake go for one frame at a stop; pressed again, the game's
    gearbox took it for reverse, and the truck rolled back."""
    from core.control.longitudinal import HOLD_BRAKE, LongitudinalCommand, held_still

    let_go = LongitudinalCommand(0.0, 0.0, "urgent")
    assert held_still(let_go, speed=0.0).brake == HOLD_BRAKE
    assert held_still(let_go, speed=5.0) is let_go, "moving: as asked"
    pulling = LongitudinalCommand(0.3, 0.0, "moving off")
    assert held_still(pulling, speed=0.0) is pulling
    firm = LongitudinalCommand(0.0, 0.5, "stopping hard")
    assert held_still(firm, speed=0.0) is firm


def test_at_a_stop_the_brake_left_up_is_not_pressed() -> None:
    """In Linz a drive begun at rest, the brake up, pressed it for one frame
    before pulling away: the gearbox went into reverse, and the throttle
    braked the truck where it stood."""
    from core.control.longitudinal import LongitudinalCommand, held_still

    coasting = LongitudinalCommand(0.0, 0.0, "coasting is enough")
    assert held_still(coasting, speed=0.0, brake_down=False) is coasting
    stopping = LongitudinalCommand(0.0, 0.4, "red light")
    assert held_still(stopping, speed=0.0, brake_down=False).brake == 0.0
    assert held_still(stopping, speed=2.0, brake_down=False) is stopping, "moving: as asked"
    pulling = LongitudinalCommand(0.3, 0.0, "moving off")
    assert held_still(pulling, speed=0.0, brake_down=False) is pulling


def test_the_brake_s_own_strength_is_learned_and_braked_by() -> None:
    """Near Eindhoven a Scania R's load was taken as 1.42 times the
    calibrated truck's; its brake gave as the calibrated truck's does, and
    2.5 m/s^2 asked for gave 3.5 to 4.9."""
    from core.control.longitudinal import ACTROS_EMPTY, ActuatorMap, BrakeEstimate

    estimate = BrakeEstimate()
    actual = ActuatorMap()  # the truck: brakes as calibrated
    for _ in range(int(5 / (1 / 60))):
        estimate.observe(
            brake=0.10,
            throttle=0.001,
            acceleration=actual.acceleration_for(0.0, 0.10),
            grade=0.0,
            speed=12.0,
            calibration=ACTROS_EMPTY,
            dt=1 / 60,
        )
    assert estimate.ratio == pytest.approx(1.0, abs=0.02)
    guessed = ActuatorMap()
    guessed.mass_ratio = 1.42
    heavy = guessed.command_for(-2.5).brake
    guessed.reset()
    guessed.brake_ratio = estimate.ratio
    learned = guessed.command_for(-2.5).brake
    assert learned < heavy
    assert -actual.acceleration_for(0.0, learned) == pytest.approx(2.5, abs=0.15)


def test_nothing_is_learned_of_the_brake_with_the_throttle_on_or_slow() -> None:
    from core.control.longitudinal import ACTROS_EMPTY, BrakeEstimate

    for frame in ({"throttle": 0.3, "speed": 12.0}, {"throttle": 0.0, "speed": 1.0}):
        estimate = BrakeEstimate()
        for _ in range(120):
            estimate.observe(
                brake=0.10,
                acceleration=-2.8,
                grade=0.0,
                calibration=ACTROS_EMPTY,
                dt=1 / 60,
                **frame,
            )
        assert estimate.ratio is None


def test_in_reverse_at_rest_the_brake_is_let_go() -> None:
    """Stopped at a red near Linz (20 November) on the brake held, the game
    put the truck into reverse within half a second, and the brake -- its
    pedal backing the truck in reverse -- drove it back."""
    from core.control.longitudinal import HOLD_BRAKE, LongitudinalCommand, held_still

    holding = LongitudinalCommand(0.0, HOLD_BRAKE, "stopping")
    assert held_still(holding, 0.0, brake_down=True).brake == HOLD_BRAKE
    let_go = held_still(holding, 0.0, brake_down=True, reversed=True)
    assert (let_go.throttle, let_go.brake) == (0.0, 0.0)
    # Pulling away, the throttle takes it out of reverse: left as it is.
    pulling = LongitudinalCommand(0.4, 0.0, "accelerating")
    assert held_still(pulling, 0.0, reversed=True).throttle == 0.4
