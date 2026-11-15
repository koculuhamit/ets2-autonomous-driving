"""Tests for the vehicle safety limits.

Written after a bug that these would have caught: the distance budget fired
during the emergency stop it had itself triggered, so the abort aborted itself
and the truck was left rolling with no commands. The guard made the situation
worse than no guard.

So the tests below care as much about when a limit must *not* fire as when it
must.
"""

from __future__ import annotations

import pytest

from core.control.limits import (
    DistanceBudgetExceeded,
    InputsNotApplied,
    SafetyMonitor,
    SpeedLimitExceeded,
    Stuck,
    UnsafeCondition,
    WrongDirection,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def monitor(clock: FakeClock, **kwargs: object) -> SafetyMonitor:
    defaults: dict[str, object] = {"speed_limit_ms": 20.0, "clock": clock}
    defaults.update(kwargs)
    return SafetyMonitor(**defaults)  # type: ignore[arg-type]


def step(
    guard: SafetyMonitor,
    *,
    speed: float = 0.0,
    throttle: float = 0.0,
    position: tuple[float, float] = (0.0, 0.0),
    expect_forward: bool = False,
) -> None:
    guard.check(speed=speed, throttle=throttle, position=position, expect_forward=expect_forward)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def test_speed_within_the_limit_passes(clock: FakeClock) -> None:
    step(monitor(clock), speed=19.0)


def test_speed_above_the_limit_raises(clock: FakeClock) -> None:
    with pytest.raises(SpeedLimitExceeded, match="km/h"):
        step(monitor(clock), speed=25.0)


def test_speed_limit_applies_in_reverse(clock: FakeClock) -> None:
    with pytest.raises(SpeedLimitExceeded):
        step(monitor(clock), speed=-25.0)


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #


def test_distance_within_budget_passes(clock: FakeClock) -> None:
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(0.0, 0.0))
    step(guard, position=(0.0, 50.0))
    assert guard.distance_travelled == pytest.approx(50.0)


def test_distance_beyond_budget_raises(clock: FakeClock) -> None:
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(0.0, 0.0))
    with pytest.raises(DistanceBudgetExceeded, match="120 m of a 100 m"):
        step(guard, position=(0.0, 120.0))


def test_distance_is_measured_from_the_first_sample(clock: FakeClock) -> None:
    """The run starts where the truck is, not at the world origin."""
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(-6500.0, 29000.0))
    step(guard, position=(-6500.0, 29030.0))
    assert guard.distance_travelled == pytest.approx(30.0)


def test_distance_is_measured_in_the_ground_plane(clock: FakeClock) -> None:
    guard = monitor(clock)
    step(guard, position=(0.0, 0.0))
    step(guard, position=(3.0, 4.0))
    assert guard.distance_travelled == pytest.approx(5.0)


def test_distance_follows_the_road_and_not_the_crow(clock: FakeClock) -> None:
    """The budget is road, so a corner costs what it costs.

    Measured as displacement from the start, a 542 m drive round a bend
    reported 311 m and the budget was never close to firing.
    """
    guard = monitor(clock)
    for z in (0.0, 30.0, 60.0):
        step(guard, position=(0.0, z))
    for x in (30.0, 60.0):
        step(guard, position=(x, 60.0))
    assert guard.distance_travelled == pytest.approx(120.0)


def test_a_loop_does_not_refund_the_budget(clock: FakeClock) -> None:
    """Back where it started is not the same as having gone nowhere."""
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(0.0, 0.0))
    step(guard, position=(0.0, 40.0))
    step(guard, position=(40.0, 40.0))
    with pytest.raises(DistanceBudgetExceeded):
        step(guard, position=(40.0, 0.0))


def test_no_budget_means_no_distance_limit(clock: FakeClock) -> None:
    guard = monitor(clock, distance_budget_m=None)
    step(guard, position=(0.0, 0.0))
    step(guard, position=(0.0, 100_000.0))


# --------------------------------------------------------------------------- #
# Stuck -- the limit that fires at zero speed
# --------------------------------------------------------------------------- #


def test_stuck_fires_after_the_timeout(clock: FakeClock) -> None:
    guard = monitor(clock, stuck_seconds=2.5)
    step(guard, speed=0.0, throttle=0.7)
    clock.advance(2.6)
    with pytest.raises(Stuck, match="in the way"):
        step(guard, speed=0.0, throttle=0.7)


def test_stuck_does_not_fire_before_the_timeout(clock: FakeClock) -> None:
    """Long enough for a clutch to take up and an automatic to pick a gear."""
    guard = monitor(clock, stuck_seconds=2.5)
    step(guard, speed=0.0, throttle=0.7)
    clock.advance(2.0)
    step(guard, speed=0.0, throttle=0.7)


def test_movement_resets_the_stuck_timer(clock: FakeClock) -> None:
    guard = monitor(clock, stuck_seconds=2.5)
    step(guard, speed=0.0, throttle=0.7)
    clock.advance(2.0)
    step(guard, speed=5.0, throttle=0.7)  # moving again
    clock.advance(2.0)
    step(guard, speed=0.0, throttle=0.7)  # only 2 s since it stopped


def test_stuck_ignores_a_standstill_with_no_throttle(clock: FakeClock) -> None:
    """Parked is not stuck. Without this the guard fires every time the truck
    waits between measurements."""
    guard = monitor(clock, stuck_seconds=2.5)
    for _ in range(5):
        clock.advance(10.0)
        step(guard, speed=0.0, throttle=0.0)


def test_stuck_fires_at_a_speed_every_other_limit_calls_safe(clock: FakeClock) -> None:
    """Zero speed passes the speed limit, the direction check and, at the
    origin, the distance budget. This is the only limit that sees it."""
    guard = monitor(clock, speed_limit_ms=20.0, distance_budget_m=100.0, stuck_seconds=1.0)
    step(guard, speed=0.0, throttle=1.0, expect_forward=True)
    clock.advance(1.5)
    with pytest.raises(Stuck):
        step(guard, speed=0.0, throttle=1.0, expect_forward=True)


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #


def test_reversing_while_commanded_forward_raises(clock: FakeClock) -> None:
    with pytest.raises(WrongDirection, match="commanded forward"):
        step(monitor(clock), speed=-2.0, throttle=0.5, expect_forward=True)


def test_small_rollback_is_tolerated(clock: FakeClock) -> None:
    """A truck creeping backwards on a slope before the clutch takes up is not
    driving the wrong way."""
    step(monitor(clock), speed=-0.2, throttle=0.5, expect_forward=True)


def test_reversing_is_fine_when_not_commanded_forward(clock: FakeClock) -> None:
    step(monitor(clock), speed=-5.0, expect_forward=False)


# --------------------------------------------------------------------------- #
# Suspension -- the bug this module was extracted for
# --------------------------------------------------------------------------- #


def test_suspension_lets_the_stop_run(clock: FakeClock) -> None:
    """The bug: the distance budget fired inside the emergency stop it had
    triggered, so the stop never ran and the truck kept rolling with no
    commands. A guard that prevents stopping is worse than no guard."""
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(0.0, 0.0))
    with pytest.raises(DistanceBudgetExceeded):
        step(guard, position=(0.0, 150.0))

    with guard.suspend():
        # Braking to rest: still over budget, and about to be stationary with
        # the brake on. Both limits would otherwise fire here.
        step(guard, position=(0.0, 160.0), speed=5.0, throttle=0.0)
        clock.advance(10.0)
        step(guard, position=(0.0, 165.0), speed=0.0, throttle=0.0)


def test_suspension_still_tracks_distance(clock: FakeClock) -> None:
    """The reported figure has to stay true, or the run's summary lies."""
    guard = monitor(clock, distance_budget_m=100.0)
    with guard.suspend():
        step(guard, position=(0.0, 0.0))
        step(guard, position=(0.0, 140.0))
    assert guard.distance_travelled == pytest.approx(140.0)


def test_suspension_is_restored_after_the_block(clock: FakeClock) -> None:
    guard = monitor(clock, distance_budget_m=100.0)
    step(guard, position=(0.0, 0.0))
    with guard.suspend():
        step(guard, position=(0.0, 150.0))
    assert guard.suspended is False
    with pytest.raises(DistanceBudgetExceeded):
        step(guard, position=(0.0, 150.0))


def test_suspension_is_restored_even_after_an_exception(clock: FakeClock) -> None:
    guard = monitor(clock)
    with pytest.raises(RuntimeError), guard.suspend():
        raise RuntimeError("something else went wrong")
    assert guard.suspended is False


def test_suspension_nests(clock: FakeClock) -> None:
    guard = monitor(clock)
    with guard.suspend():
        with guard.suspend():
            assert guard.suspended is True
        assert guard.suspended is True
    assert guard.suspended is False


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("speed_limit", [0.0, -1.0])
def test_invalid_speed_limit_is_rejected(clock: FakeClock, speed_limit: float) -> None:
    with pytest.raises(ValueError, match="speed_limit_ms"):
        monitor(clock, speed_limit_ms=speed_limit)


@pytest.mark.parametrize("budget", [0.0, -50.0])
def test_invalid_distance_budget_is_rejected(clock: FakeClock, budget: float) -> None:
    with pytest.raises(ValueError, match="distance_budget_m"):
        monitor(clock, distance_budget_m=budget)


def test_every_limit_is_an_unsafe_condition() -> None:
    """Callers catch the base class on the abort path; a limit outside the
    hierarchy would escape that handler and skip the stop entirely."""
    for limit in (SpeedLimitExceeded, DistanceBudgetExceeded, Stuck, WrongDirection):
        assert issubclass(limit, UnsafeCondition)


# --------------------------------------------------------------------------- #
# Inputs that never arrive
# --------------------------------------------------------------------------- #


def feed(
    guard: SafetyMonitor,
    clock: FakeClock,
    seconds: float,
    *,
    throttle: float = 0.0,
    brake: float = 0.0,
    applied: float = 0.0,
    speed: float = 10.0,
) -> None:
    """Sixty checks a second for `seconds`, as the control loop makes them."""
    for _ in range(round(seconds * 60)):
        guard.check(
            speed=speed, throttle=throttle, brake=brake, position=(0.0, 0.0), applied=applied
        )
        clock.advance(1 / 60)


def test_a_pedal_commanded_and_never_applied_stops_the_run(clock: FakeClock) -> None:
    """What happened when the game window lost focus: telemetry flowing, inputs ignored."""
    guard = monitor(clock)
    with pytest.raises(InputsNotApplied, match="window focused"):
        feed(guard, clock, 2.0, throttle=0.43, applied=0.0)


def test_it_fires_while_moving_where_the_stuck_check_cannot(clock: FakeClock) -> None:
    """Moving, a truck whose inputs are ignored coasts on; nothing else notices."""
    guard = monitor(clock)
    with pytest.raises(InputsNotApplied):
        feed(guard, clock, 2.0, throttle=0.5, applied=0.0, speed=16.0)


def test_a_gear_shift_is_not_mistaken_for_it(clock: FakeClock) -> None:
    """The gearbox cuts the throttle for up to 0.78 s a shift in every recorded run."""
    guard = monitor(clock)
    for _ in range(5):
        feed(guard, clock, 0.8, throttle=0.6, applied=0.0)
        feed(guard, clock, 2.0, throttle=0.6, applied=0.6)


def test_the_game_standing_still_in_a_shift_is_not_mistaken_for_it(clock: FakeClock) -> None:
    """In a roundabout in Torino a shift cut the throttle for 0.7 s and the
    game froze for 0.83 s in it: no new frame, the inputs not ignored."""
    guard = monitor(clock)
    frame = 0

    def frames(seconds: float, applied: float) -> None:
        nonlocal frame
        for _ in range(round(seconds * 60)):
            frame += 1
            guard.check(speed=5.0, throttle=1.0, position=(0.0, 0.0), applied=applied, frame=frame)
            clock.advance(1 / 60)

    frames(0.3, 0.0)
    for _ in range(round(0.83 * 240)):  # the loop goes round, the game stands
        guard.check(speed=5.0, throttle=1.0, position=(0.0, 0.0), applied=0.0, frame=frame)
        clock.advance(1 / 240)
    frames(0.4, 0.0)
    frames(1.0, 1.0)


def test_frames_still_coming_with_nothing_applied_stop_the_run(clock: FakeClock) -> None:
    guard = monitor(clock)
    with pytest.raises(InputsNotApplied):
        for frame in range(120):
            guard.check(speed=5.0, throttle=1.0, position=(0.0, 0.0), applied=0.0, frame=frame)
            clock.advance(1 / 60)


def test_braking_counts_as_a_pedal(clock: FakeClock) -> None:
    guard = monitor(clock)
    with pytest.raises(InputsNotApplied):
        feed(guard, clock, 2.0, brake=0.3, applied=0.0)


def test_a_light_touch_is_not_a_command(clock: FakeClock) -> None:
    """Below the throttle threshold there is nothing to expect the game to apply."""
    guard = monitor(clock)
    feed(guard, clock, 5.0, throttle=0.05, applied=0.0)


def test_without_the_applied_value_the_check_is_skipped(clock: FakeClock) -> None:
    guard = monitor(clock)
    for _ in range(300):
        guard.check(speed=10.0, throttle=0.8, position=(0.0, 0.0))
        clock.advance(1 / 60)


def _damaged(cabin: float = 0.0, trailer: float = 0.0, cargo: float = 0.0) -> dict:
    return {
        "truck": {"wear_cabin": cabin, "wear_chassis": 0.02},
        "trailers": [{"wear_chassis": trailer, "cargo_damage": cargo}],
    }


def test_damage_since_the_drive_began_is_a_hit() -> None:
    """Out of Milan, clipping a kerb: 0.0049 of damage, then 2.5 s at full
    throttle against it before anything stopped the truck."""
    from core.control.limits import Collided, DamageWatch

    watch = DamageWatch()
    watch.check(_damaged(cabin=0.05))  # what it set off with is not a hit
    watch.check(_damaged(cabin=0.0505))
    with pytest.raises(Collided, match="hit something"):
        watch.check(_damaged(cabin=0.05, trailer=0.0049))


def test_damaged_cargo_is_a_hit_too() -> None:
    from core.control.limits import Collided, DamageWatch

    watch = DamageWatch()
    watch.check(_damaged())
    with pytest.raises(Collided):
        watch.check(_damaged(cargo=0.002))


def test_at_rest_for_longer_than_any_light_holds_a_truck_it_hands_over() -> None:
    from core.control.limits import STANDSTILL_S, Standstill, StoodTooLong

    watch = Standstill()
    for _ in range(int(STANDSTILL_S) - 1):
        watch.update(0.0, 1.0)
    watch.update(5.0, 1.0)  # moved off: counted afresh
    for _ in range(int(STANDSTILL_S) - 1):
        watch.update(0.1, 1.0)
    with pytest.raises(StoodTooLong, match="5 min"):
        watch.update(0.0, 1.0)


def test_speed_lost_faster_than_the_brakes_can_is_a_hit_at_once() -> None:
    """Out of the Italian-Swiss customs: 3.1 m/s gone in 0.1 s against a
    barrier, and the game's damage 12 s later; the brakes give 9.7 m/s^2."""
    from core.control.limits import Collided, Impacts

    braking = Impacts()
    for k in range(100):
        braking.check(k / 60, 20.0 - 9.7 * k / 60)
    hit = Impacts()
    for k in range(10):
        hit.check(k / 60, 8.7)
    with pytest.raises(Collided, match="km/h lost at once"):
        for k in range(10, 20):
            hit.check(k / 60, 8.7 - 3.1 * min(1.0, (k - 9) / 6))
