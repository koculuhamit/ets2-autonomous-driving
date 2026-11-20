"""The retarder: the step chosen, the lever moved, its strength learned."""

from __future__ import annotations

import pytest

from core.common.controls import Button
from core.control.longitudinal import ActuatorMap, SpeedController
from core.control.retarder import (
    LEVER_PRESS_S,
    LEVER_SETTLE_S,
    PER_STEP_GUESS,
    RetarderEstimate,
    RetarderLever,
)


def _actuators(steps: int = 4, per_step: float = 0.3, speed: float = 20.0) -> ActuatorMap:
    actuators = ActuatorMap()
    actuators.retarder_steps = steps
    actuators.retarder_per_step = per_step
    actuators.speed = speed
    return actuators


def test_without_a_retarder_the_gap_is_coasted_as_before() -> None:
    actuators = ActuatorMap()
    command = actuators.command_for(-0.6)
    assert (command.throttle, command.brake, command.retarder) == (0.0, 0.0, 0)
    assert actuators.command_for(-1.0).brake > 0.0


def test_the_gap_is_filled_by_the_retarder_not_the_brake() -> None:
    """0.8 m/s^2 wanted: coasting gives 0.22, the brake's least is 1.81."""
    command = _actuators().command_for(-0.8)
    assert command.brake == 0.0
    assert command.retarder == 2, command.reason
    # Past what the retarder reaches, the brake.
    assert _actuators().command_for(-2.5).brake > 0.0
    assert _actuators().command_for(-2.5).retarder == 0


def test_the_brake_waits_until_the_retarder_is_not_enough() -> None:
    """Four steps of 0.3 reach 1.42: 1.2 is still the retarder's, where
    without one the brake came on from 0.9."""
    command = _actuators().command_for(-1.2)
    assert command.brake == 0.0
    assert command.retarder == 3


def test_the_retarder_is_left_off_slow() -> None:
    command = _actuators(speed=2.0).command_for(-0.8)
    assert command.retarder == 0


def test_a_step_is_held_while_the_request_wanders_near_it() -> None:
    actuators = _actuators()
    assert actuators.command_for(-0.22 - 0.6).retarder == 2
    assert actuators.command_for(-0.22 - 0.75).retarder == 2
    assert actuators.command_for(-0.22 - 0.42).retarder == 2
    assert actuators.command_for(-0.22 - 1.1).retarder == 4


def test_accelerating_lets_the_retarder_off_once_it_has_been_on_a_while() -> None:
    """On the way to Eindhoven the lever went 1, 0, 1, 0 every 0.2 s, the
    throttle full between: a step just taken is held, and coasted with."""
    from core.control.longitudinal import RETARDER_HOLD_S

    actuators = _actuators()
    actuators.command_for(-0.8)
    held = actuators.command_for(0.3)
    assert held.retarder == 2 and held.throttle == 0.0
    assert actuators.command_for(1.0).retarder == 0, "a real pull lets it off"
    actuators.command_for(-0.8)
    actuators.tick(RETARDER_HOLD_S + 0.1)
    assert actuators.command_for(0.3).retarder == 0


def test_the_retarder_stays_on_while_the_brake_gives_the_rest() -> None:
    actuators = _actuators()
    actuators.command_for(-0.8)
    braking = actuators.command_for(-2.5)
    assert braking.brake > 0.0 and braking.retarder == 2
    alone = _actuators().command_for(-2.5)
    assert braking.brake < alone.brake


def test_the_plant_slows_by_the_steps_on() -> None:
    actuators = _actuators()
    assert actuators.acceleration_for(0.0, 0.0, 2) == pytest.approx(-0.22 - 0.6)


def test_slowing_from_70_to_50_never_touches_the_brake() -> None:
    """A speed limit 20 km/h lower, slowed for at 1 m/s^2 as the speed
    profile has it: with a retarder, without the brake's 1.8 m/s^2 step."""
    controller = SpeedController(actuators=_actuators())
    speed, target, dt, braked, stepped = 70 / 3.6, 70 / 3.6, 1 / 60, 0, 0
    for _ in range(int(40 / dt)):
        target = max(target - 1.0 * dt, 50 / 3.6)
        command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        braked += command.brake > 0.0
        stepped = max(stepped, command.retarder)
        speed += (
            controller.actuators.acceleration_for(command.throttle, command.brake, command.retarder)
            * dt
        )
    assert braked == 0
    assert stepped > 0
    assert speed == pytest.approx(50 / 3.6, abs=0.5)
    # Without one, the same slowing down is braked for.
    controller = SpeedController()
    speed, target, braked = 70 / 3.6, 70 / 3.6, 0
    for _ in range(int(40 / dt)):
        target = max(target - 1.0 * dt, 50 / 3.6)
        command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        braked += command.brake > 0.0
        speed += controller.actuators.acceleration_for(command.throttle, command.brake) * dt
    assert braked > 0


def test_the_lever_is_pressed_a_step_at_a_time() -> None:
    lever = RetarderLever()
    assert lever.buttons(wanted=2, level=0, now=0.0) == {Button.RETARDER_UP}
    assert lever.buttons(wanted=2, level=0, now=LEVER_PRESS_S / 2) == {Button.RETARDER_UP}
    # Let go, and waiting for the telemetry to show it.
    assert lever.buttons(wanted=2, level=0, now=LEVER_PRESS_S + 0.05) == frozenset()
    # It moved: the next press.
    assert lever.buttons(wanted=2, level=1, now=LEVER_PRESS_S + 0.1) == {Button.RETARDER_UP}
    later = LEVER_PRESS_S + 0.1 + LEVER_PRESS_S + 0.01
    assert lever.buttons(wanted=2, level=2, now=later) == frozenset()
    assert lever.buttons(wanted=0, level=2, now=later + 0.01) == {Button.RETARDER_DOWN}
    assert lever.presses == 3


def test_a_press_the_game_missed_is_pressed_again() -> None:
    lever = RetarderLever()
    lever.buttons(wanted=1, level=0, now=0.0)
    again = LEVER_PRESS_S + LEVER_SETTLE_S + 0.01
    assert lever.buttons(wanted=1, level=0, now=again) == {Button.RETARDER_UP}


def _observe(estimate: RetarderEstimate, seconds: float, **frame) -> None:
    values = {
        "level": 2,
        "throttle": 0.0,
        "brake": 0.0,
        "acceleration": -(0.22 + 2 * 0.5),
        "grade": 0.0,
        "speed": 20.0,
        "coast_deceleration": 0.22,
        "mass_ratio": 1.0,
    }
    values.update(frame)
    for _ in range(int(seconds * 60)):
        estimate.observe(dt=1 / 60, **values)


def test_a_step_is_learned_from_the_truck_slowing() -> None:
    estimate = RetarderEstimate()
    _observe(estimate, 20.0)
    assert estimate.per_step == pytest.approx(0.5, abs=0.02)
    assert estimate.seen_s > 15.0


def test_a_heavier_truck_teaches_the_same_step() -> None:
    """Twice the mass slows half as much: the step, for the calibrated
    truck, is the same."""
    estimate = RetarderEstimate()
    _observe(estimate, 20.0, acceleration=-(0.22 + 2 * 0.5) / 2.0, mass_ratio=2.0)
    assert estimate.per_step == pytest.approx(0.5, abs=0.02)


def test_downhill_the_hill_is_taken_out() -> None:
    estimate = RetarderEstimate()
    _observe(estimate, 20.0, acceleration=-(0.22 + 2 * 0.5) + 0.4, grade=0.4)
    assert estimate.per_step == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    "frame",
    [{"throttle": 0.2}, {"brake": 0.1}, {"speed": 2.0}, {"level": 0}],
)
def test_nothing_is_learned_with_a_pedal_down_slow_or_the_retarder_off(frame) -> None:
    estimate = RetarderEstimate()
    _observe(estimate, 10.0, **frame)
    assert estimate.per_step == PER_STEP_GUESS
    assert estimate.seen_s == 0.0


def test_a_step_just_changed_is_not_judged_yet() -> None:
    estimate = RetarderEstimate()
    _observe(estimate, 0.9)
    assert estimate.per_step == PER_STEP_GUESS


def test_a_lever_that_never_moves_is_given_up_on() -> None:
    """An older plugin has no retarder buttons: the lever never moves, and
    the gap goes back to the brake rather than being coasted through."""
    from core.control.retarder import UNANSWERED_PRESSES

    lever = RetarderLever()
    now = 0.0
    while lever.works and now < 30.0:
        lever.buttons(wanted=2, level=0, now=now)
        now += 1 / 60
    assert not lever.works
    assert lever.presses == UNANSWERED_PRESSES
    assert lever.buttons(wanted=2, level=0, now=now) == frozenset()


def test_a_lever_that_moves_slowly_is_not_given_up_on() -> None:
    lever = RetarderLever()
    level, now = 0, 0.0
    while now < 10.0:
        pressed = lever.buttons(wanted=4, level=level, now=now)
        if pressed and now - lever._pressed_at > LEVER_PRESS_S * 0.9:
            level += 1
        now += 1 / 60
    assert lever.works
    assert level == 4
