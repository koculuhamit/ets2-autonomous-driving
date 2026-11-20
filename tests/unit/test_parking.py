"""Pressing a toggle only when the telemetry says it needs flipping."""

from __future__ import annotations

from core.control.parking import PRESS_S, SET_AFTER_S, SETTLE_S, ParkingBrake


def presses(brake: ParkingBrake, want_set: bool, states, dt: float = 1 / 60) -> list[bool]:
    """Step through `states` (the telemetry's parking_brake per step)."""
    return [brake.press(want_set=want_set, is_set=s, now=k * dt) for k, s in enumerate(states)]


def test_a_press_is_one_pulse_even_while_the_state_lags() -> None:
    steps = int((PRESS_S + SETTLE_S) * 60) + 10
    pulse = presses(ParkingBrake(), True, [False] * steps)
    held = int(PRESS_S * 60)
    assert all(pulse[:held]) and not any(pulse[held + 1 : int((PRESS_S + SETTLE_S) * 60)])


def test_no_press_when_the_brake_is_already_as_wanted() -> None:
    assert not any(presses(ParkingBrake(), True, [True] * 120))
    assert not any(presses(ParkingBrake(), False, [False] * 120))


def test_a_press_that_did_not_take_is_tried_again() -> None:
    steps = int((PRESS_S + SETTLE_S) * 60) + 30
    pulse = presses(ParkingBrake(), True, [False] * steps)
    assert pulse[-20], "still unset after settling: press again"


def test_a_slower_toggle_is_given_longer_before_another_press() -> None:
    """The engine takes its time to start: pressed again while it cranks,
    it would be switched off."""
    steps = int((PRESS_S + 3.0) * 60) - 5
    pulse = presses(ParkingBrake(settle_s=3.0), True, [False] * steps)
    held = int(PRESS_S * 60)
    assert not any(pulse[held + 1 :])


def test_once_set_it_stops_pressing() -> None:
    held = int(PRESS_S * 60)
    states = [False] * (held + 5) + [True] * 100
    pulse = presses(ParkingBrake(), True, states)
    assert not any(pulse[held + 5 :])


def test_the_brake_is_set_only_after_a_real_stop_has_lasted() -> None:
    brake = ParkingBrake()
    assert not brake.should_hold(speed=0.0, stopping=True, now=0.0)
    assert not brake.should_hold(speed=0.0, stopping=True, now=SET_AFTER_S / 2)
    assert brake.should_hold(speed=0.0, stopping=True, now=SET_AFTER_S + 0.01)
    assert not brake.should_hold(speed=2.0, stopping=True, now=SET_AFTER_S + 0.02), "moving"
    assert not brake.should_hold(speed=0.0, stopping=False, now=SET_AFTER_S + 0.03), "told to go"


def test_a_stop_is_not_undone_by_the_target_wandering_over_the_threshold() -> None:
    """What happened: 12.3 m became 13.1 m measured, the target 0.31 m/s, and
    the truck let go of its stop on a slope."""
    from core.control.parking import RESUME_TARGET_MPS, StopLatch

    latch = StopLatch()
    kw = dict(stop_target=0.3, hold_below=3.0)
    assert latch.stopping(target=0.0, speed=0.5, **kw)
    assert latch.stopping(target=0.31, speed=0.0, **kw), "a wander is not a reason to go"
    assert latch.stopping(target=1.0, speed=0.0, **kw)
    assert not latch.stopping(target=RESUME_TARGET_MPS, speed=0.0, **kw), "the lead has moved off"
    assert not latch.stopping(target=1.0, speed=1.2, **kw), "moving again, no latch"


def test_on_the_flat_the_parking_brake_waits_for_a_long_stop() -> None:
    """Pulled at every stop, in the middle of turns too, it looked bad (the
    driver's word, 16 November): on the flat the service brake holds."""
    from core.control.parking import FLAT_SET_AFTER_S, SLOPE_MPS2

    brake = ParkingBrake()
    assert not brake.should_hold(speed=0.0, stopping=True, now=0.0, grade=0.05)
    assert not brake.should_hold(speed=0.0, stopping=True, now=SET_AFTER_S + 1.0, grade=0.05)
    assert brake.should_hold(speed=0.0, stopping=True, now=FLAT_SET_AFTER_S + 0.1, grade=0.05)
    sloped = ParkingBrake()
    assert not sloped.should_hold(speed=0.0, stopping=True, now=0.0, grade=-SLOPE_MPS2)
    assert sloped.should_hold(speed=0.0, stopping=True, now=SET_AFTER_S + 0.1, grade=-SLOPE_MPS2)


def test_on_the_flat_the_parking_brake_is_set_well_before_the_gearbox_reverses() -> None:
    """Near Eindhoven, the brake held at a red for 8.5 s, the gearbox went
    into reverse and the truck rolled back."""
    from core.control.parking import FLAT_SET_AFTER_S, REVERSE_AFTER_S

    assert FLAT_SET_AFTER_S + 2.0 * SETTLE_S < REVERSE_AFTER_S
