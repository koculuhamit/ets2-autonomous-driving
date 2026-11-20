"""Following a vehicle: the target-speed rule alone, and closed around a truck.

The closed-loop cases use the actuator as measured (core.control.longitudinal:
nothing between 0.22 and 1.81 m/s^2 of deceleration), a 0.3 s lag on what it
delivers, and a perception that reports the gap 0.1 s late, 20 times a
second, 3 % off -- roughly what the camera gave against a trailer at 40 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from core.control.following import (
    BAND_SLOWER,
    GAP_BAND_SHARE,
    MAX_DECELERATION,
    FollowingDecision,
    GapPolicy,
    Lead,
    follow,
)
from core.control.longitudinal import SpeedController

KMH = 1 / 3.6


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def test_with_nothing_ahead_the_road_decides() -> None:
    assert follow(None, 20.0) == FollowingDecision(None, None, "clear road")


def test_at_the_wanted_gap_the_truck_matches_the_lead() -> None:
    policy = GapPolicy()
    speed = 25.0
    decision = follow(Lead(policy.wanted_gap(speed), -3.0), speed, policy)
    assert decision.target_speed == pytest.approx(22.0)
    assert decision.urgent_deceleration is None


def test_too_close_it_drops_back_and_too_far_it_catches_up() -> None:
    policy = GapPolicy()
    band = GAP_BAND_SHARE * policy.time_gap_s * 20.0
    close = follow(Lead(policy.wanted_gap(20.0) - 20.0, 0.0), 20.0, policy)
    far = follow(Lead(policy.wanted_gap(20.0) + 20.0, 0.0), 20.0, policy)
    correction = (band / BAND_SLOWER + 20.0 - band) / policy.closing_time_s
    assert close.target_speed == pytest.approx(20.0 - correction)
    assert far.target_speed == pytest.approx(20.0 + correction)


def test_a_gap_about_right_is_let_be() -> None:
    """Following a car in Milan, the detector's distance wandering by metres
    swung the target by 5 km/h, and throttle and brake took turns."""
    policy = GapPolicy()
    wanted = policy.wanted_gap(20.0)
    wobble = [follow(Lead(wanted + d, 0.0), 20.0, policy).target_speed for d in (-4.0, 4.0)]
    assert max(wobble) - min(wobble) < 2 * 4.0 / policy.closing_time_s / 2


def test_a_stopped_car_close_ahead_at_speed_is_urgent() -> None:
    policy = GapPolicy()
    decision = follow(Lead(60.0, -22.0), 22.0, policy)
    assert decision.target_speed < 3.0
    # Aimed at the standstill gap while there is room for it.
    room = 60.0 - policy.standstill_gap_m
    assert decision.urgent_deceleration == pytest.approx(22.0**2 / (2 * room))
    assert follow(Lead(30.0, -22.0), 22.0).urgent_deceleration == MAX_DECELERATION


def test_a_gentle_slowdown_is_left_to_the_speed_controller() -> None:
    assert follow(Lead(50.0, -2.0), 22.0).urgent_deceleration is None


def test_a_policy_that_makes_no_sense_is_refused() -> None:
    with pytest.raises(ValueError):
        GapPolicy(min_gap_m=10.0, standstill_gap_m=8.0)


# --------------------------------------------------------------------------- #
# Closed around a simulated truck
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    gaps: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray

    @property
    def closest(self) -> float:
        return float(self.gaps.min())


def drive(
    lead_speed,
    *,
    speed: float,
    gap: float | None,
    set_speed: float,
    seconds: float,
    appears_at: float = 0.0,
    vanishes_at: float = float("inf"),
    seed: int = 0,
) -> Run:
    """Truck behind a lead whose speed is `lead_speed(t)`; the lead is seen
    between `appears_at` and `vanishes_at`, starting `gap` metres ahead."""
    rng = np.random.default_rng(seed)
    dt, perception_dt, delay, lag = 1 / 60, 1 / 20, 0.1, 0.3
    controller = SpeedController()
    actuators = controller.actuators
    policy = GapPolicy()
    truck_s, lead_s = 0.0, gap if gap is not None else 1e9
    acceleration = 0.0
    history: list[tuple[float, float, float]] = []  # t, gap, closing
    seen: Lead | None = None
    next_perception = 0.0
    gaps, speeds, accelerations, times = [], [], [], []
    for k in range(int(seconds / dt)):
        t = k * dt
        v_lead = lead_speed(t)
        history.append((t, lead_s - truck_s, v_lead - speed))
        if t >= next_perception:
            next_perception += perception_dt
            late = [h for h in history if h[0] <= t - delay]
            if late and appears_at <= t < vanishes_at:
                _, true_gap, true_closing = late[-1]
                seen = Lead(true_gap * (1 + rng.normal(0, 0.03)), true_closing + rng.normal(0, 0.3))
            else:
                seen = None

        decision = follow(seen, speed, policy)
        if decision.urgent_deceleration is not None:
            command = controller.urgent(decision.urgent_deceleration, dt)
        else:
            target = set_speed
            if decision.target_speed is not None:
                target = min(target, decision.target_speed)
            command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        delivered = actuators.acceleration_for(command.throttle, command.brake)
        acceleration += (delivered - acceleration) * dt / lag
        speed = max(0.0, speed + acceleration * dt)
        truck_s += speed * dt
        lead_s += v_lead * dt
        if appears_at <= t < vanishes_at:
            gaps.append(lead_s - truck_s)
        speeds.append(speed)
        accelerations.append(acceleration)
        times.append(t)
    return Run(np.array(gaps), np.array(speeds), np.array(accelerations), np.array(times))


def test_catching_up_settles_at_the_wanted_gap() -> None:
    run = drive(lambda t: 70 * KMH, speed=90 * KMH, gap=120.0, set_speed=90 * KMH, seconds=90)
    wanted = GapPolicy().wanted_gap(70 * KMH)
    last = run.gaps[-600:]
    assert np.median(last) == pytest.approx(wanted, rel=0.1)
    assert run.closest > 0.8 * wanted, "no dive inside the gap on the way in"
    assert run.speeds[-1] == pytest.approx(70 * KMH, abs=1.0)


def test_a_lead_braking_to_a_stop_is_followed_to_a_stop_without_touching() -> None:
    def lead(t: float) -> float:
        return max(0.0, 80 * KMH - 4.0 * max(0.0, t - 5.0))

    policy = GapPolicy()
    run = drive(
        lead, speed=80 * KMH, gap=policy.wanted_gap(80 * KMH), set_speed=90 * KMH, seconds=40
    )
    assert run.speeds[-1] < 0.5
    # At a 1.5 s gap, a lead braking this hard is stopped behind at about
    # the minimum gap, 5.9 m; at 2.0 s it was 11.3 m.
    assert run.closest > policy.min_gap_m - 0.5
    assert -run.accelerations.min() <= MAX_DECELERATION + 0.5


def test_a_car_cutting_in_close_is_not_hit() -> None:
    run = drive(lambda t: 75 * KMH, speed=90 * KMH, gap=25.0, set_speed=90 * KMH, seconds=40)
    assert run.closest > GapPolicy().min_gap_m
    assert np.median(run.gaps[-300:]) == pytest.approx(GapPolicy().wanted_gap(75 * KMH), rel=0.15)


def test_a_stopped_car_seen_at_90_m_is_stopped_for() -> None:
    run = drive(lambda t: 0.0, speed=80 * KMH, gap=90.0, set_speed=80 * KMH, seconds=30)
    assert run.speeds[-1] < 0.5
    assert run.closest > GapPolicy().min_gap_m


def test_when_the_lead_is_gone_the_truck_returns_to_its_speed_gently() -> None:
    run = drive(
        lambda t: 60 * KMH,
        speed=60 * KMH,
        gap=GapPolicy().wanted_gap(60 * KMH),
        set_speed=80 * KMH,
        seconds=60,
        vanishes_at=10.0,
    )
    after = run.times > 10.5
    assert run.speeds[-1] == pytest.approx(80 * KMH, abs=0.5)
    assert run.accelerations[after].max() < 1.6


def test_an_oncoming_car_is_not_followed() -> None:
    """At 14 m/s, a car closing at 25 m/s is driving towards the truck."""
    decision = follow(Lead(30.0, -25.0), 14.0)
    assert decision.target_speed is None and decision.urgent_deceleration is None
    assert "oncoming" in decision.reason


def test_a_stopped_car_is_not_mistaken_for_an_oncoming_one() -> None:
    """Its closing speed is the truck's own, read 15 % high."""
    decision = follow(Lead(80.0, -22.0 * 1.15), 22.0)
    assert decision.target_speed is not None


def test_giving_way_the_truck_does_not_speed_up_towards_where_it_stops() -> None:
    """Out of Milan, giving way to a car in a queue 9 m short of where to
    stop, at 12 km/h: it sped up to 16 km/h, and braked at 7 m/s^2."""
    from core.control.following import APPROACH_CREEP_MPS, COMFORT_DECELERATION, stop_short

    controller = SpeedController()
    dt, speed, room, top, hardest = 1 / 60, 12 / 3.6, 9.4, 0.0, 0.0
    while room > 0.0 and speed > 0.05:
        decision = stop_short(room, speed, "giving way")
        if decision.urgent_deceleration is not None:
            hardest = max(hardest, decision.urgent_deceleration)
            command = controller.urgent(decision.urgent_deceleration, dt)
        else:
            command = controller.step(
                target_speed=decision.target_speed, speed=speed, grade=0.0, dt=dt
            )
        speed = max(
            0.0, speed + controller.actuators.acceleration_for(command.throttle, command.brake) * dt
        )
        room -= speed * dt
        top = max(top, speed)
    assert top <= 12 / 3.6 + 0.1
    # At the place, give or take the last centimetres: braked for directly
    # on the planned curve, the brake's least is a little more than it.
    assert room > -0.5, "stopped at the place"
    assert hardest <= COMFORT_DECELERATION + 1.0
    # From a crawl far off, it still comes on.
    assert stop_short(40.0, 0.0, "giving way").target_speed == APPROACH_CREEP_MPS


def test_a_lead_closing_faster_than_the_truck_goes_is_braked_for_no_more_than_stopping() -> None:
    """Hamburg: a low loader turning across seemed to close at 6.8 m/s on a
    truck doing 2.5 m/s 12 m off, and was braked for at 7 m/s^2; stopping
    took 0.4."""
    from core.control.following import COMFORT_DECELERATION, Lead, follow

    decision = follow(Lead(12.2, -6.8), 2.5)
    assert decision.urgent_deceleration is None or decision.urgent_deceleration <= (
        COMFORT_DECELERATION
    )
    # A stopped car near ahead at speed is still braked for hard.
    assert follow(Lead(14.0, -13.9), 13.9).urgent_deceleration is not None


def _stop(room: float, speed: float, slack: float = 0.0) -> tuple[float, float]:
    """`stop_short` closed round the measured actuators, as the loop drives
    it: the hardest deceleration asked for, and the metres left at the stop."""
    from core.control.following import stop_short

    controller = SpeedController()
    dt, hardest = 1 / 60, 0.0
    while speed > 0.05 and room > -slack - 5.0:
        decision = stop_short(room, speed, "giving way", slack)
        if decision.urgent_deceleration is not None:
            hardest = max(hardest, decision.urgent_deceleration)
            command = controller.urgent(decision.urgent_deceleration, dt)
        else:
            command = controller.step(
                target_speed=decision.target_speed, speed=speed, grade=0.0, dt=dt
            )
        speed = max(
            0.0, speed + controller.actuators.acceleration_for(command.throttle, command.brake) * dt
        )
        room -= speed * dt
    return hardest, room


def test_a_stop_in_good_time_is_braked_for_gently_not_left_to_the_last_metres() -> None:
    """Leipzig (16 November): giving way at 25 km/h from 21 m, the truck
    coasted on at 25 until 0 to 2 m short, then braked at 7 m/s^2."""
    from core.control.following import COMFORT_DECELERATION

    for room, kmh in ((21.0, 25.0), (40.0, 30.0), (10.0, 14.0)):
        hardest, left = _stop(room, kmh / 3.6)
        assert hardest <= COMFORT_DECELERATION, (room, kmh, hardest)
        assert left > -0.5, (room, kmh, left)


def test_a_stop_come_on_late_uses_the_room_beyond_it() -> None:
    """A vehicle crossing first seen 12 m ahead at 16 km/h: stopped short of
    it by the least gap, gently, not short of the standstill gap at 7 m/s^2."""
    from core.control.following import COMFORT_DECELERATION, MAX_DECELERATION, stop_short

    assert stop_short(0.0, 16 / 3.6, "crossing").urgent_deceleration == MAX_DECELERATION
    late = stop_short(0.0, 16 / 3.6, "crossing", slack=6.0).urgent_deceleration
    assert late is not None and late <= COMFORT_DECELERATION
    hardest, left = _stop(0.5, 16 / 3.6, slack=6.0)
    assert hardest <= COMFORT_DECELERATION
    assert left > -6.0, "short of the vehicle"
