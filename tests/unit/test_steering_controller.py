"""Tests for the steering position controller.

The plant is a pure integrator that holds its output, so the failure modes are
specific: a sign error drives the wheel the wrong way and keeps going, and an
unclamped command saturates the actuator. Neither shows up as an exception.
"""

from __future__ import annotations

import itertools

import pytest

from core.control.steering import (
    STEERING_CALIBRATION,
    SteeringCalibration,
    SteeringPositionController,
)


@pytest.fixture
def controller() -> SteeringPositionController:
    return SteeringPositionController()


def test_at_target_commands_nothing(controller: SteeringPositionController) -> None:
    assert controller.command(target=0.3, measured=0.3) == 0.0


def test_command_sign_opposes_the_error(controller: SteeringPositionController) -> None:
    """The game's mix subtracts our input, so a positive angle needs a negative
    command. Getting this backwards turns a controller into a diverging one --
    the plant integrates, so the wheel runs to the stop rather than settling."""
    assert controller.command(target=0.5, measured=0.0) < 0.0
    assert controller.command(target=-0.5, measured=0.0) > 0.0


def test_command_is_proportional_to_the_error(controller: SteeringPositionController) -> None:
    small = controller.command(target=0.01, measured=0.0)
    large = controller.command(target=0.02, measured=0.0)
    assert large == pytest.approx(2 * small)


@pytest.mark.parametrize(
    ("target", "measured"),
    [(1.0, -1.0), (-1.0, 1.0), (0.9, 0.0), (0.0, 0.9)],
)
def test_command_is_clamped(
    controller: SteeringPositionController, target: float, measured: float
) -> None:
    assert abs(controller.command(target=target, measured=measured)) <= 1.0


def test_max_command_is_respected() -> None:
    controller = SteeringPositionController(max_command=0.3)
    assert abs(controller.command(target=1.0, measured=-1.0)) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("target", "measured"),
    [(float("nan"), 0.0), (0.0, float("nan")), (float("nan"), float("nan"))],
)
def test_nan_commands_nothing(
    controller: SteeringPositionController, target: float, measured: float
) -> None:
    """Zero here means 'stop turning', not 'straighten': the wheel holds."""
    assert controller.command(target=target, measured=measured) == 0.0


def test_tolerance_suppresses_small_commands() -> None:
    controller = SteeringPositionController(tolerance=0.01)
    assert controller.command(target=0.005, measured=0.0) == 0.0
    assert controller.command(target=0.05, measured=0.0) != 0.0


def test_closed_loop_converges_without_overshoot() -> None:
    """Simulate the plant: an integrator driven by the command.

    A first-order loop should approach the target monotonically. Overshoot here
    would mean the gain is too high for the frame rate, which on the real truck
    is a wheel that hunts.
    """
    controller = SteeringPositionController()
    calibration = controller.calibration
    dt = 1.0 / 60.0
    angle = 0.0
    target = 0.3
    history = []
    for _ in range(120):  # two seconds
        command = controller.command(target=target, measured=angle)
        angle += calibration.sign * command * calibration.rate_gain * dt
        history.append(angle)

    assert history[-1] == pytest.approx(target, abs=1e-3)
    assert max(history) <= target + 1e-9, "overshoot: the loop is under-damped"
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(history))


def test_converges_when_the_calibration_is_wrong() -> None:
    """The measured rate gain was half the true value and the loop still
    settled cleanly. That tolerance is the reason this structure was chosen,
    so it is worth a test rather than a note."""
    controller = SteeringPositionController(SteeringCalibration(rate_gain=0.375, sign=-1.0))
    true_gain = 0.75
    dt = 1.0 / 60.0
    angle = 0.0
    for _ in range(120):
        command = controller.command(target=-0.2, measured=angle)
        angle += -1.0 * command * true_gain * dt
    assert angle == pytest.approx(-0.2, abs=1e-3)


@pytest.mark.parametrize("gain", [0.0, -1.0])
def test_invalid_gain_is_rejected(gain: float) -> None:
    with pytest.raises(ValueError, match="gain must be positive"):
        SteeringPositionController(gain=gain)


@pytest.mark.parametrize("max_command", [0.0, 1.5, -0.5])
def test_invalid_max_command_is_rejected(max_command: float) -> None:
    with pytest.raises(ValueError, match="max_command"):
        SteeringPositionController(max_command=max_command)


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="rate_gain"):
        SteeringCalibration(rate_gain=0.0, sign=-1.0)
    with pytest.raises(ValueError, match="sign"):
        SteeringCalibration(rate_gain=0.75, sign=0.0)


def test_default_calibration_matches_the_measured_sign() -> None:
    """The mix subtracts semantical.steering; see ADR-0002."""
    assert STEERING_CALIBRATION.sign == -1.0
    assert STEERING_CALIBRATION.rate_gain > 0.0


# --------------------------------------------------------------------------- #
# The actuator as measured while driving
# --------------------------------------------------------------------------- #


def test_the_measured_gain_falls_with_speed() -> None:
    """Keyboard-style steering: the faster the truck, the slower the wheel turns."""
    from core.control.steering import rate_gain_at

    assert rate_gain_at(0.0) == pytest.approx(0.25)
    assert rate_gain_at(16.7) == pytest.approx(0.19)
    speeds = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 16.7]
    gains = [rate_gain_at(s) for s in speeds]
    assert all(a >= b for a, b in itertools.pairwise(gains))


def test_the_gain_is_not_extrapolated_past_what_was_measured() -> None:
    from core.control.steering import rate_gain_at

    assert rate_gain_at(30.0) == pytest.approx(rate_gain_at(16.7))
    assert rate_gain_at(-10.0) == pytest.approx(rate_gain_at(10.0))


def test_given_a_speed_the_loop_uses_the_measured_gain() -> None:
    from core.control.steering import rate_gain_at

    controller = SteeringPositionController()
    error = 0.01
    stationary = controller.command(target=error, measured=0.0)
    driving = controller.command(target=error, measured=0.0, speed=16.7)
    assert stationary == pytest.approx(STEERING_CALIBRATION.sign * 8.0 * error / 0.75)
    assert driving == pytest.approx(STEERING_CALIBRATION.sign * 8.0 * error / rate_gain_at(16.7))
    # Four times the command for the same error: the calibration overstated the
    # actuator that much at road speed.
    assert abs(driving / stationary) == pytest.approx(0.75 / 0.19, rel=1e-6)


def _hold(controller, target: float, speed: float, compensate: bool, seconds: float = 8.0) -> float:
    """The angle a bend's steering settles at, with the game's pull to centre."""
    from core.control.steering import STEERING_CALIBRATION, centring_at, rate_gain_at

    angle, dt = 0.0, 1 / 60
    for _ in range(int(seconds / dt)):
        command = controller.command(
            target=target, measured=angle, speed=speed if compensate else None
        )
        rate = (
            STEERING_CALIBRATION.sign * rate_gain_at(speed) * command - centring_at(speed) * angle
        )
        angle += rate * dt
    return angle


def test_the_pull_to_centre_leaves_a_bend_under_steered_unless_compensated() -> None:
    """What happened: at 85 km/h the truck ran 1.5 m wide in bends. Fitted to
    the drives, the game pulls the angle to centre with a time constant under
    a second, and the uncompensated loop settles well short of its target."""
    from core.control.steering import SteeringPositionController

    controller = SteeringPositionController()
    short = _hold(controller, 0.03, 23.0, compensate=False)
    held = _hold(controller, 0.03, 23.0, compensate=True)
    assert short < 0.7 * 0.03
    assert held == pytest.approx(0.03, rel=0.02)


def test_centring_is_nothing_at_rest_and_strong_at_speed() -> None:
    from core.control.steering import centring_at

    assert centring_at(0.0) == 0.0
    assert centring_at(23.0) == pytest.approx(1.21)
    assert centring_at(40.0) == pytest.approx(1.21), "held beyond the last measurement"
