"""Tests for manual-override detection.

The pedal path rests on a direct measurement -- a commanded 0.600 of brake
reads back as exactly 0.600 -- so it is tested as a contract. The steering path
does not, and the tests say so: they cover the mechanics and the case the
detector is actually reliable in, which is when the stack is not steering.
"""

from __future__ import annotations

import pytest

from core.control.override import OverrideDetector
from core.control.steering import STEERING_CALIBRATION


def detector(**kwargs: float) -> OverrideDetector:
    return OverrideDetector(**kwargs)  # type: ignore[arg-type]


def feed(
    det: OverrideDetector,
    *,
    frames: int = 20,
    steer: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    angle: float = 0.0,
    angle_rate: float = 0.0,
    input_throttle: float | None = None,
    input_brake: float | None = None,
    dt: float = 1 / 60,
):
    """Run several identical frames and return the last result."""
    result = None
    for index in range(frames):
        result = det.update(
            commanded_steer=steer,
            commanded_throttle=throttle,
            commanded_brake=brake,
            measured_angle=angle + angle_rate * index * dt,
            input_throttle=throttle if input_throttle is None else input_throttle,
            input_brake=brake if input_brake is None else input_brake,
            dt=dt,
        )
    return result


# --------------------------------------------------------------------------- #
# Pedals -- exact, because the telemetry is
# --------------------------------------------------------------------------- #


def test_nothing_reported_when_the_pedals_match() -> None:
    assert feed(detector(), throttle=0.35, brake=0.0) is None


def test_extra_throttle_is_reported() -> None:
    event = feed(detector(), throttle=0.2, input_throttle=0.7)
    assert event is not None
    assert event.control == "throttle"
    assert event.magnitude == pytest.approx(0.5)


def test_extra_brake_is_reported() -> None:
    event = feed(detector(), brake=0.0, input_brake=0.6)
    assert event is not None
    assert event.control == "brake"


def test_a_pedal_difference_within_tolerance_is_ignored() -> None:
    """The command takes 15-28 ms to arrive, so telemetry always lags a little
    behind a changing command."""
    assert feed(detector(), throttle=0.30, input_throttle=0.34) is None


def test_the_largest_disagreement_wins() -> None:
    event = feed(detector(), throttle=0.1, brake=0.0, input_throttle=0.3, input_brake=0.9)
    assert event is not None
    assert event.control == "brake"


# --------------------------------------------------------------------------- #
# Dwell
# --------------------------------------------------------------------------- #


def test_a_brief_disagreement_is_ignored() -> None:
    """Without dwell the detector fires on its own command changes."""
    assert feed(detector(), frames=2, throttle=0.1, input_throttle=0.9) is None


def test_a_sustained_disagreement_fires() -> None:
    assert feed(detector(), frames=20, throttle=0.1, input_throttle=0.9) is not None


def test_the_dwell_timer_resets_when_the_controls_agree() -> None:
    det = detector()
    feed(det, frames=5, throttle=0.1, input_throttle=0.9)  # not yet long enough
    feed(det, frames=5, throttle=0.1, input_throttle=0.1)  # agreement resets it
    assert feed(det, frames=5, throttle=0.1, input_throttle=0.9) is None


def test_reset_clears_the_timers() -> None:
    det = detector()
    feed(det, frames=5, throttle=0.1, input_throttle=0.9)
    det.reset()
    assert feed(det, frames=5, throttle=0.1, input_throttle=0.9) is None


# --------------------------------------------------------------------------- #
# Steering -- reliable when the stack is not steering
# --------------------------------------------------------------------------- #


def test_a_wheel_that_moves_while_we_command_nothing_is_someone_else() -> None:
    """The unambiguous case, and the one that matters for a speed-only
    controller: with a zero command the expected rate is zero, so any movement
    is external with no model in the way."""
    event = feed(detector(), steer=0.0, angle_rate=0.3)
    assert event is not None
    assert event.control == "steering"


def test_a_stationary_wheel_under_a_zero_command_is_not_reported() -> None:
    assert feed(detector(), steer=0.0, angle_rate=0.0) is None


def test_a_wheel_moving_as_commanded_is_not_reported() -> None:
    commanded = 0.4
    expected_rate = STEERING_CALIBRATION.sign * STEERING_CALIBRATION.rate_gain * commanded
    assert feed(detector(), steer=commanded, angle_rate=expected_rate) is None


def test_the_tolerance_widens_with_our_own_command() -> None:
    """The plant gain is known to roughly 0.75 near centre and falls off
    towards the stops, so the harder we steer the less a residual means. A
    fixed tolerance would make the detector fire on model error alone."""
    det = detector()
    expected = STEERING_CALIBRATION.sign * STEERING_CALIBRATION.rate_gain * 1.0
    # A 30 % model error at full command should not read as intervention.
    assert feed(det, steer=1.0, angle_rate=expected * 0.7) is None
    # The same absolute residual with no command of our own should.
    det.reset()
    assert feed(det, steer=0.0, angle_rate=abs(expected) * 0.3) is not None


def test_the_first_frame_cannot_judge_steering() -> None:
    """There is no previous angle to difference against."""
    det = detector()
    assert (
        det.update(
            commanded_steer=0.0,
            commanded_throttle=0.0,
            commanded_brake=0.0,
            measured_angle=0.5,
            input_throttle=0.0,
            input_brake=0.0,
            dt=1 / 60,
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kwargs", [{"pedal_tolerance": 0.0}, {"steering_tolerance": -1.0}])
def test_invalid_tolerances_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        detector(**kwargs)


def test_negative_dwell_is_rejected() -> None:
    with pytest.raises(ValueError, match="dwell"):
        detector(dwell=-0.1)


def test_zero_dt_is_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        OverrideDetector().update(
            commanded_steer=0.0,
            commanded_throttle=0.0,
            commanded_brake=0.0,
            measured_angle=0.0,
            input_throttle=0.0,
            input_brake=0.0,
            dt=0.0,
        )


def _pedals(detector: OverrideDetector, *, sent, read, frames: int, dt: float = 0.02):
    event = None
    for _ in range(frames):
        event = detector.update_pedals(
            commanded_throttle=sent[0],
            commanded_brake=sent[1],
            input_throttle=read[0],
            input_brake=read[1],
            dt=dt,
        )
    return event


def test_a_foot_on_the_brake_is_the_driver_taking_the_truck() -> None:
    """At an automatic gate out of Luxembourg the driver braked, and the
    stack pressed full throttle against the brake until it gave up."""
    detector = OverrideDetector(pedal_tolerance=0.2, dwell=0.2)
    event = _pedals(detector, sent=(0.5, 0.0), read=(0.5, 0.6), frames=12)
    assert event is not None and event.control == "brake"


def test_a_pedal_applied_less_than_sent_is_not_a_foot() -> None:
    """At the start of a drive the game took no inputs for a moment, and the
    gearbox lets the throttle go for every shift: no foot on anything."""
    detector = OverrideDetector(pedal_tolerance=0.2, dwell=0.2)
    assert _pedals(detector, sent=(1.0, 0.0), read=(0.0, 0.0), frames=100) is None
