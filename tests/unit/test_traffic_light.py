"""Stopping for traffic lights: the rule alone, and closed around a truck.

The truck is the one of test_following: the measured actuator, which cannot
decelerate between 0.22 and 1.81 m/s^2, and a 0.3 s lag on what it delivers.
The stop latch is left out; it has tests of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from core.control.longitudinal import HOLD_BELOW, SpeedController
from core.control.traffic_light import (
    DARK_SPEED,
    STOP_SHORT_M,
    STUCK_RED_S,
    UNSEEN_WAIT_S,
    LightDecision,
    LightLine,
    TrafficLights,
    lines_on_path,
)
from core.localization.path import ReferencePath
from core.map import Signal
from core.perception.lights import LightState
from core.planning.route import StopLine

KMH = 1 / 3.6
LIGHT = Signal("traffic_light_major", "cr_2x1")
SLEEPY = Signal("traffic_light_minor", "cr_1x1", (1410, 180))
LINE_AT = 300.0

RED, AMBER, GREEN, UNKNOWN = LightState.RED, LightState.AMBER, LightState.GREEN, LightState.UNKNOWN


@dataclass
class Run:
    positions: np.ndarray
    speeds: np.ndarray
    decelerations: np.ndarray
    times: np.ndarray

    def position_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.positions))

    def hardest_braking(self) -> float:
        """On the way down. Below `HOLD_BELOW` the speed controller's stop mode
        holds a fixed brake, the same for any stop (core.control.longitudinal)."""
        return float(self.decelerations[self.speeds > HOLD_BELOW].max())

    def stopped_at(self) -> float | None:
        """Where it first came to rest, if it did."""
        rest = np.flatnonzero(self.speeds < 0.05)
        return None if len(rest) == 0 else float(self.positions[rest[0]])


def drive(
    light: Callable[[float, float], LightState],
    *,
    speed: float = 50 * KMH,
    seconds: float = 60.0,
    signal: Signal = LIGHT,
    minute_of_day: float = 720.0,
    start: float = 0.0,
) -> Run:
    """A truck on a straight road towards one light at `LINE_AT`, showing
    `light(t, position)`, with a road limit of `speed`."""
    dt, lag = 1 / 60, 0.3
    lights = TrafficLights([LightLine(LINE_AT, signal)])
    controller = SpeedController()
    actuators = controller.actuators
    position, acceleration, road_limit = start, 0.0, speed
    positions, speeds, decelerations, times = [], [], [], []
    for k in range(int(seconds / dt)):
        t = k * dt
        decision = lights.step(
            progress=position,
            speed=speed,
            state=light(t, position),
            minute_of_day=minute_of_day,
            now=t,
        )
        target = (
            road_limit if decision.target_speed is None else min(road_limit, decision.target_speed)
        )
        if decision.urgent_deceleration is not None:
            command = actuators.command_for(-decision.urgent_deceleration)
        else:
            command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        delivered = actuators.acceleration_for(command.throttle, command.brake)
        acceleration += (delivered - acceleration) * dt / lag
        speed = max(0.0, speed + acceleration * dt)
        position += speed * dt
        positions.append(position)
        speeds.append(speed)
        decelerations.append(-acceleration)
        times.append(t)
    return Run(np.array(positions), np.array(speeds), np.array(decelerations), np.array(times))


def always(state: LightState) -> Callable[[float, float], LightState]:
    return lambda t, s: state


def test_a_red_light_seen_from_afar_is_stopped_at_before_the_line() -> None:
    run = drive(always(RED))
    stop = run.stopped_at()
    assert stop is not None
    assert LINE_AT - STOP_SHORT_M - 3.0 < stop < LINE_AT - STOP_SHORT_M + 1.0
    assert run.hardest_braking() < 2.6, "seen in time, a comfortable stop"


def test_it_goes_again_when_the_light_turns_green() -> None:
    run = drive(lambda t, s: RED if t < 40.0 else GREEN, seconds=70.0)
    assert run.position_at(39.0) < LINE_AT - STOP_SHORT_M + 1.0
    assert run.positions[-1] > LINE_AT + 50.0


def test_a_green_light_is_driven_through_without_slowing() -> None:
    run = drive(always(GREEN), seconds=30.0)
    assert run.speeds.min() > 50 * KMH - 0.5


def test_a_light_not_seen_yet_does_not_slow_a_truck_that_will_see_it() -> None:
    """Lamps are seen from about 110 m: unknown until then, green after."""
    run = drive(lambda t, s: UNKNOWN if s < LINE_AT - 110.0 else GREEN, seconds=30.0)
    assert run.speeds.min() > 50 * KMH - 1.0


def test_a_light_never_seen_is_not_slowed_for() -> None:
    """In the game: a T junction the map gives a light and the road does not.
    Slowed for as if red, the truck dropped speed at every one of them."""
    run = drive(always(UNKNOWN), seconds=40.0)
    assert run.positions[-1] > LINE_AT + 50.0
    assert run.speeds.min() > 50 * KMH - 0.5


def test_a_light_seen_on_the_way_in_and_lost_at_the_line_is_waited_for() -> None:
    """Its lamps have left the view; it is there, and was red."""
    run = drive(lambda t, s: RED if s < LINE_AT - 40.0 else UNKNOWN, seconds=UNSEEN_WAIT_S)
    assert run.stopped_at() is not None
    assert run.positions[-1] < LINE_AT


def test_a_light_seen_and_lost_is_given_up_on_after_a_whole_cycle() -> None:
    run = drive(lambda t, s: RED if s < LINE_AT - 40.0 else UNKNOWN, seconds=UNSEEN_WAIT_S + 60.0)
    rested = run.times[run.speeds < 0.05]
    assert rested[0] + UNSEEN_WAIT_S - 1.0 < run.times[np.argmax(run.positions > LINE_AT)]
    assert run.positions[-1] > LINE_AT + 50.0


def test_a_light_first_seen_red_close_in_is_still_stopped_for() -> None:
    """Nothing until 50 m out, the nearest any working light was first seen
    from, then red: approached at the road's speed, it is still stopped for."""
    run = drive(lambda t, s: UNKNOWN if s < LINE_AT - 50.0 else RED)
    assert run.positions[-1] < LINE_AT


def test_a_light_that_shows_while_waiting_is_waited_for() -> None:
    """Red now and then from where the truck stands: a light it can see."""

    def light(t: float, s: float) -> LightState:
        seen_coming = LINE_AT - 150.0 < s < LINE_AT - 100.0
        return RED if seen_coming or int(t) % 30 < 5 else UNKNOWN

    run = drive(light, seconds=UNSEEN_WAIT_S + 60.0)
    assert run.positions[-1] < LINE_AT


def test_a_colour_seen_on_the_way_in_does_not_keep_the_truck_for_good() -> None:
    """In the game: a red blob once, 90 m out, at a light that was not there."""
    run = drive(
        lambda t, s: RED if 100.0 < LINE_AT - s < 110.0 else UNKNOWN, seconds=UNSEEN_WAIT_S + 90.0
    )
    assert run.positions[-1] > LINE_AT + 20.0


def test_amber_far_enough_away_is_stopped_for() -> None:
    run = drive(lambda t, s: GREEN if s < LINE_AT - 60.0 else AMBER)
    assert run.positions[-1] < LINE_AT


def test_amber_too_close_to_stop_is_driven_through() -> None:
    """At 50 km/h, 20 m short of the stop point asks for 4.8 m/s^2."""
    run = drive(lambda t, s: GREEN if s < LINE_AT - STOP_SHORT_M - 20.0 else AMBER, seconds=40)
    assert run.positions[-1] > LINE_AT + 50.0
    assert run.speeds.min() > 50 * KMH - 1.0


def test_red_seen_late_is_stopped_for_hard_but_before_the_line() -> None:
    """At 50 km/h, 40 m from the stop point: 2.4 m/s^2 -- past comfort in the
    quantised actuator, so it is commanded directly."""
    run = drive(lambda t, s: UNKNOWN if s < LINE_AT - STOP_SHORT_M - 30.0 else RED)
    assert run.positions[-1] < LINE_AT


def test_red_seen_too_late_to_stop_is_driven_through() -> None:
    """Stopping from 50 km/h in 12 m would take 8 m/s^2 and leave the truck
    in the junction. (Driving, this needs the amber to have been missed: an
    unknown light has already slowed the truck, and a green one committed it.)"""
    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    decision = lights.step(
        progress=LINE_AT - STOP_SHORT_M - 12.0, speed=50 * KMH, state=RED, minute_of_day=720.0
    )
    assert decision.target_speed is None and decision.urgent_deceleration is None
    assert "too late" in decision.reason


def test_a_stop_once_begun_is_finished() -> None:
    """Braking for red, the deceleration still needed can wander high."""
    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    first = lights.step(progress=200.0, speed=50 * KMH, state=RED, minute_of_day=720.0)
    assert first.target_speed is not None
    later = lights.step(
        progress=LINE_AT - STOP_SHORT_M - 5.0, speed=8.0, state=RED, minute_of_day=720.0
    )
    assert later.urgent_deceleration == pytest.approx(6.4)


def test_once_through_on_green_a_change_to_red_is_ignored() -> None:
    """30 m short of the stop point at 50 km/h, red could still be stopped
    for at 3.2 m/s^2 -- but green was too close to stop in comfort: committed."""
    run = drive(lambda t, s: GREEN if s < LINE_AT - STOP_SHORT_M - 30.0 else RED, seconds=40)
    assert run.positions[-1] > LINE_AT + 50.0
    assert run.hardest_braking() < 0.5


def test_a_sleeping_light_is_a_junction_without_one() -> None:
    run = drive(always(UNKNOWN), signal=SLEEPY, minute_of_day=10.0, seconds=40)
    assert run.speeds.min() > 50 * KMH - 0.5


def test_red_seen_at_a_sleeping_light_is_the_next_light_s() -> None:
    """city-01: the red of a light 64 m beyond a sleeping one."""
    run = drive(always(RED), signal=SLEEPY, minute_of_day=10.0, seconds=40)
    assert run.speeds.min() > 50 * KMH - 0.5


def test_the_same_light_awake_is_stopped_at() -> None:
    run = drive(always(RED), signal=SLEEPY, minute_of_day=720.0)
    assert run.positions[-1] < LINE_AT


def test_a_light_far_ahead_is_not_yet_considered() -> None:
    lights = TrafficLights([LightLine(1000.0, LIGHT)])
    decision = lights.step(progress=0.0, speed=20.0, state=RED, minute_of_day=720.0)
    assert decision.target_speed is None and decision.distance_m is None


def test_the_next_light_is_the_first_one_not_yet_passed() -> None:
    lights = TrafficLights([LightLine(300.0, LIGHT), LightLine(100.0, LIGHT)])
    assert lights.next_line(50.0).s == 100.0
    assert lights.next_line(150.0).s == 300.0
    assert lights.next_line(350.0) is None


# --------------------------------------------------------------------------- #
# Where the lines are on the path
# --------------------------------------------------------------------------- #


def test_stop_lines_are_placed_along_the_path() -> None:
    path = ReferencePath(np.column_stack([np.zeros(101), np.linspace(0, 500, 101)]))
    lines = [
        StopLine(np.array([0.5, 0.0, 120.0]), LIGHT),
        StopLine(np.array([0.0, 0.0, 400.0]), SLEEPY),
        StopLine(np.array([50.0, 0.0, 200.0]), LIGHT),  # on another road
    ]
    placed = lines_on_path(lines, path)
    assert [line.s for line in placed] == pytest.approx([120.0, 400.0], abs=1.0)
    assert [line.signal for line in placed] == [LIGHT, SLEEPY]


def test_a_path_through_the_same_junction_twice_gets_both_stops() -> None:
    """Out along x = 0 and back along x = 0: the second line is the second pass."""
    out_and_back = np.array([[0.0, 0.0], [0.0, 300.0], [0.1, 300.0], [0.1, 0.0]])
    path = ReferencePath(out_and_back)
    same_place = np.array([0.0, 0.0, 100.0])
    placed = lines_on_path([StopLine(same_place, LIGHT), StopLine(same_place, LIGHT)], path)
    assert [line.s for line in placed] == pytest.approx([100.0, 500.0], abs=2.0)


def test_held_at_a_green_by_something_else_it_still_stops_when_red_comes_back() -> None:
    """In the game: stopped at the line, green came, a remembered car held the
    truck, the light went red -- and once released, the truck went on red."""
    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    at_line = LINE_AT - STOP_SHORT_M
    green = lights.step(progress=at_line, speed=0.0, state=GREEN, minute_of_day=720.0)
    assert green.target_speed is None
    red = lights.step(progress=at_line, speed=0.0, state=RED, minute_of_day=720.0)
    assert red.target_speed == 0.0


def test_a_red_that_never_changes_is_not_a_light() -> None:
    """In the game: a no-parking sign's red ring, at a light the map has and
    the road does not. No profile's red lasts longer than 46 s."""
    run = drive(always(RED), seconds=STUCK_RED_S + 60.0)
    rested = run.times[run.speeds < 0.05][0]
    left = run.times[np.argmax(run.positions > LINE_AT)]
    assert left > rested + STUCK_RED_S - 1.0
    assert run.positions[-1] > LINE_AT + 50.0


def test_a_light_that_changes_is_waited_for_however_long() -> None:
    """Red, a moment of nothing, red again: a real light, not a stuck sign."""

    def light(t: float, s: float) -> LightState:
        return UNKNOWN if 60.0 < t < 61.0 else RED

    run = drive(light, seconds=STUCK_RED_S + 60.0)
    assert run.positions[-1] < LINE_AT


def test_a_colour_held_over_from_the_light_just_passed_is_not_the_next_one_s() -> None:
    """Near Nynäshamn a red glimpsed at one line's end was held by the filter
    for two seconds onto the next line, a junction with no light, and the
    truck stopped there and waited for one to show."""
    second = LINE_AT + 100.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    speed, dt = 8.0, 1 / 30
    position, t, said = LINE_AT - 150.0, 0.0, []
    while position < second:
        red = LINE_AT - 1.0 < position < LINE_AT + 1.5 * speed
        decision = lights.step(
            progress=position,
            speed=speed,
            state=RED if red else UNKNOWN,
            minute_of_day=720.0,
            now=t,
        )
        if position > LINE_AT:
            said.append(decision.reason)
        position += speed * dt
        t += dt
    assert said[-1] == "no lamp seen: slowly through"


def test_a_light_still_in_view_across_its_junction_is_not_the_next_one_s() -> None:
    """In Uppsala the green of a light just crossed, slowly in traffic,
    stayed in view 25 m beyond its line; the next light, 126 m on, the map
    has and the road did not, and the truck waited at it for a lamp."""
    second = LINE_AT + 126.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    speed, dt = 4.0, 1 / 30
    position, t, said = LINE_AT - 5.0, 0.0, []
    while position < second:
        green = position < LINE_AT + 25.0
        decision = lights.step(
            progress=position,
            speed=speed,
            state=GREEN if green else UNKNOWN,
            minute_of_day=720.0,
            now=t,
        )
        if position > LINE_AT:
            said.append(decision.reason)
        position += speed * dt
        t += dt
    assert said[-1] == "no lamp seen: slowly through"


def test_a_red_at_a_light_close_behind_another_is_still_stopped_for() -> None:
    """Two lights 50 m apart: what is seen of the second once half way to it
    is its own."""
    second = LINE_AT + 50.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    speed, dt = 6.0, 1 / 30
    position, t, stopping = LINE_AT + 1.0, 0.0, False
    while position < second - STOP_SHORT_M:
        decision = lights.step(
            progress=position, speed=speed, state=RED, minute_of_day=720.0, now=t
        )
        stopping |= decision.reason.startswith("stop")
        position += speed * dt
        t += dt
    assert stopping


def test_a_light_close_after_one_crossed_is_not_found_dark_unwatched() -> None:
    """Milan: the next light 41 m on, its colours taken for the last one's
    while that one's stood; found dark 12 m short, it was driven through on
    red at 20 km/h. Not watched, it is stopped short of instead."""
    second = LINE_AT + 45.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    speed, dt = 6.0, 1 / 30
    position, t, said = LINE_AT - 60.0, 0.0, []
    while position < second - STOP_SHORT_M:
        decision = lights.step(
            progress=position,
            speed=speed,
            state=GREEN if position < LINE_AT + 1.0 else UNKNOWN,
            minute_of_day=720.0,
            now=t,
        )
        if position > LINE_AT + 5.0:
            said.append(decision.reason)
        position += speed * dt
        t += dt
    assert "no lamp seen: slowly through" not in said
    assert said[-1].startswith("stopping") or said[-1].startswith("stopped")


def _waited_at(start: float) -> float:
    """Seconds a truck at rest at the stop point waits for a light it never
    sees, its drive having begun `start` metres along."""
    from core.control.traffic_light import AT_STOP_M

    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    at_stop = LINE_AT - STOP_SHORT_M - AT_STOP_M / 2
    kwargs = dict(speed=0.0, state=UNKNOWN, minute_of_day=600.0)
    lights.step(progress=start, now=0.0, **kwargs)
    for k in range(1, 1000):
        decision = lights.step(progress=at_stop, now=k * 0.1, **kwargs)
        if decision.target_speed is None:
            return k * 0.1
    return math.inf


def test_a_drive_begun_at_a_light_never_seen_waits_less_before_going_on() -> None:
    """In Bern a drive begun 8 m from a line whose lights were out of view
    waited 45 s at a junction without one."""
    from core.control.traffic_light import STARTED_AT_WAIT_S

    assert _waited_at(LINE_AT - 8.0) == pytest.approx(STARTED_AT_WAIT_S, abs=0.2)
    assert STARTED_AT_WAIT_S < UNSEEN_WAIT_S - 30.0
    assert _waited_at(LINE_AT - 200.0) < 0.2, "approached and never seen: not a light"


def test_at_a_light_close_after_one_crossed_never_seen_the_wait_is_short() -> None:
    """At the bridge in Geneva, 52 m after a green, the map has a light the
    road does not: stopped short of it, as a light close behind another is
    (Milan), the truck waited the whole 45 s for a lamp to show."""
    from core.control.traffic_light import AT_STOP_M, STARTED_AT_WAIT_S

    second = LINE_AT + 52.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    at_stop = second - STOP_SHORT_M - AT_STOP_M / 2
    speed, dt = 6.0, 1 / 30
    position, t = LINE_AT - 20.0, 0.0
    while position < at_stop:
        state = GREEN if position < LINE_AT + 1.0 else UNKNOWN
        lights.step(progress=position, speed=speed, state=state, minute_of_day=720.0, now=t)
        position += speed * dt
        t += dt
    stood = t
    while t < stood + UNSEEN_WAIT_S:
        decision = lights.step(
            progress=at_stop, speed=0.0, state=UNKNOWN, minute_of_day=720.0, now=t
        )
        if decision.target_speed is None:
            break
        t += 0.1
    assert t - stood == pytest.approx(STARTED_AT_WAIT_S, abs=0.5)


def test_a_red_hidden_by_a_lorry_alongside_is_left_with_the_traffic() -> None:
    """In Bern a lorry drew up in the lane on the right and hid the red the
    truck was waiting at; it could not see the green."""
    from core.control.traffic_light import AT_STOP_M, QUEUE_STOOD_S

    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    at_stop = LINE_AT - STOP_SHORT_M - AT_STOP_M / 2
    kwargs = dict(progress=at_stop, speed=0.0, minute_of_day=600.0)
    assert lights.step(state=RED, now=0.0, **kwargs).target_speed == 0.0
    hidden = [
        lights.step(state=UNKNOWN, now=1.0 + k * 0.1, traffic_going=k * 0.1 >= 1.0, **kwargs)
        for k in range(int(QUEUE_STOOD_S * 10) + 5)
    ]
    assert all(d.target_speed == 0.0 for d in hidden[: int(QUEUE_STOOD_S * 10)])
    assert hidden[-1].target_speed == DARK_SPEED and "traffic" in hidden[-1].reason


def test_with_the_red_in_view_the_traffic_moving_off_is_not_its_green() -> None:
    """A lane beside may have a light of its own: a red seen stays a red."""
    from core.control.traffic_light import AT_STOP_M

    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    at_stop = LINE_AT - STOP_SHORT_M - AT_STOP_M / 2
    for k in range(60):
        decision = lights.step(
            progress=at_stop,
            speed=0.0,
            state=RED,
            minute_of_day=600.0,
            now=k * 0.1,
            traffic_going=True,
        )
    assert decision.target_speed == 0.0


def test_after_a_dark_light_the_next_one_close_by_is_watched_at_once() -> None:
    """In Bern two lights 68 m apart showed no lamp; the first was passed as
    dark, the second not watched until 18 m short of it, and stood at."""
    lights = TrafficLights([LightLine(200.0, LIGHT), LightLine(268.0, LIGHT)])
    decisions = []
    for k in range(400):
        progress = 50.0 + k * 0.5
        decision = lights.step(
            progress=progress, speed=5.0, state=UNKNOWN, minute_of_day=600.0, now=k * 0.1
        )
        if progress > 210.0 and progress < 268.0 - STOP_SHORT_M:
            decisions.append(decision)
    assert all(d.target_speed is None for d in decisions)


def _see_camera():
    from core.perception.camera import GroundCamera

    return GroundCamera(focal_px=819.0, horizon_row=438.3, centre_col=960.1, height_m=2.881)


def _row_of(height_m: float, from_eye_m: float, camera) -> float:
    return camera.horizon_row - (height_m - camera.height_m) / from_eye_m * camera.focal_px


def _view(top: float = 250.0, covered=()):
    from core.control.traffic_light import ViewBounds

    return ViewBounds(330.0, top, 1920.0, 700.0, covered=tuple(covered))


def test_an_overhead_light_is_stopped_for_where_its_lamp_stays_in_view() -> None:
    """Lamps 5.5 m up go behind the roof some 13 m from the line: stopped at
    9.2 m, a DAF stood at a green it could not see."""
    from core.control.traffic_light import STOP_SHORT_MAX_M, stop_short_to_see

    camera, eye_ahead, top = _see_camera(), 1.81, 250.0
    distance = 30.0
    row = _row_of(5.5, distance - eye_ahead, camera)
    stop = stop_short_to_see(distance, [(camera.centre_col, row)], camera, eye_ahead, _view(top))
    rise_at_stop = (5.5 - camera.height_m) / (stop - eye_ahead) * camera.focal_px
    assert camera.horizon_row - rise_at_stop > top or stop == STOP_SHORT_MAX_M
    assert 12.0 < stop <= STOP_SHORT_MAX_M


def test_a_light_on_a_pole_keeps_the_usual_stop() -> None:
    from core.control.traffic_light import stop_short_to_see

    camera = _see_camera()
    row = _row_of(4.0, 20.0 - 1.81, camera)
    assert stop_short_to_see(20.0, [(960.0, row)], camera, 1.81, _view()) == STOP_SHORT_M


def test_the_lowest_lamp_decides_and_nothing_seen_changes_nothing() -> None:
    """A lamp of the pole beside the line and one overhead: the pole's goes
    out of view last."""
    from core.control.traffic_light import stop_short_to_see

    camera = _see_camera()
    high, low = _row_of(5.5, 18.19, camera), _row_of(4.0, 18.19, camera)
    lamps = [(960.0, high), (960.0, low)]
    assert stop_short_to_see(20.0, lamps, camera, 1.81, _view()) == STOP_SHORT_M
    assert stop_short_to_see(20.0, [], camera, 1.81, _view()) == STOP_SHORT_M
    assert stop_short_to_see(20.0, [(960.0, 500.0)], camera, 1.81, _view()) == STOP_SHORT_M, (
        "below the horizon"
    )


def test_a_further_stop_is_kept_to_by_the_light() -> None:
    lights = TrafficLights([LightLine(100.0, LIGHT)])
    near = lights.step(progress=86.0, speed=0.0, state=LightState.RED, minute_of_day=720.0)
    far = TrafficLights([LightLine(100.0, LIGHT)]).step(
        progress=86.0, speed=0.0, state=LightState.RED, minute_of_day=720.0, stop_short=14.0
    )
    assert (near.target_speed or 0.0) > 0.0, "14 m short of the line: 4 m to go at the usual stop"
    assert far.target_speed == 0.0, "already at the light's own stop"


def _approach(lights, seen: dict[float, LightState], until: float) -> LightDecision:
    """Drive up to the line at 100 m in 1 m steps at 8 m/s, the light read as
    `seen` gives by distance (UNKNOWN where not given)."""
    decision = None
    for progress in np.arange(0.0, until, 1.0):
        distance = round(100.0 - progress)
        decision = lights.step(
            progress=float(progress),
            speed=8.0,
            state=seen.get(distance, LightState.UNKNOWN),
            minute_of_day=720.0,
            now=float(progress) / 8.0,
        )
    return decision


def test_a_light_seen_green_only_from_afar_is_gone_through_slowly() -> None:
    """Near Duisburg: green once at 105 m, never again, no light there."""
    lights = TrafficLights([LightLine(100.0, LIGHT)])
    near = _approach(lights, {75: LightState.GREEN}, 80.0)
    assert near.target_speed is not None and near.target_speed > 0.0
    assert "slowly through" in near.reason


def test_a_light_seen_green_near_and_lost_is_still_stood_at() -> None:
    lights = TrafficLights([LightLine(100.0, LIGHT)])
    near = _approach(lights, {32: LightState.GREEN}, 92.0)
    assert "slowly through" not in near.reason


def test_a_light_seen_red_from_afar_and_lost_is_not_taken_for_dark() -> None:
    lights = TrafficLights([LightLine(100.0, LIGHT)])
    near = _approach(lights, {75: LightState.RED}, 80.0)
    assert "slowly through" not in near.reason


def test_creeping_on_from_the_stop_on_green_does_not_commit_short_of_the_line() -> None:
    """Copenhagen: 8 m short of the line at 1 m/s on green, taken for on its
    way through, held 10 s by crossing traffic, over the line on red."""
    lights = TrafficLights([LightLine(100.0, LIGHT)])
    kw = dict(minute_of_day=720.0)
    lights.step(progress=91.0, speed=0.0, state=LightState.RED, now=0.0, **kw)
    lights.step(progress=92.0, speed=1.05, state=LightState.GREEN, now=1.0, **kw)
    red = lights.step(progress=92.0, speed=0.0, state=LightState.RED, now=5.0, **kw)
    assert red.target_speed == 0.0, "8 m short of the line, the red is stopped for"


def test_committed_on_green_then_held_short_of_the_line_the_light_counts_again() -> None:
    from core.control.traffic_light import THROUGH_STANDING_S

    lights = TrafficLights([LightLine(100.0, LIGHT)])
    kw = dict(minute_of_day=720.0)
    going = lights.step(progress=62.0, speed=13.9, state=LightState.GREEN, now=0.0, **kw)
    assert "going through" in going.reason
    # Held by something else 6 m short of the line, the light gone red.
    lights.step(progress=94.0, speed=0.0, state=LightState.RED, now=10.0, **kw)
    later = lights.step(
        progress=94.0, speed=0.0, state=LightState.RED, now=10.0 + THROUGH_STANDING_S + 0.1, **kw
    )
    assert later.target_speed == 0.0
    # Past the line it goes on whatever the light shows.
    past = TrafficLights([LightLine(100.0, LIGHT)])
    past.step(progress=62.0, speed=13.9, state=LightState.GREEN, now=0.0, **kw)
    over = past.step(progress=98.5, speed=0.0, state=LightState.RED, now=20.0, **kw)
    assert "going through" in over.reason


def test_a_line_close_after_a_green_with_no_lamp_in_view_is_not_stopped_at() -> None:
    """Leipzig (16 November): 33 m after a green, a line the map has and the
    road does not; no lamp in view the whole way, only the filter still
    holding the green. Stopped at, the truck waited 10 s for nothing."""
    second = LINE_AT + 33.0
    lights = TrafficLights([LightLine(LINE_AT, LIGHT), LightLine(second, LIGHT)])
    speed, dt = 7.0, 1 / 30
    position, t, slowest = LINE_AT - 60.0, 0.0, math.inf
    while position < second + 5.0:
        held = position < LINE_AT + 3.0
        decision = lights.step(
            progress=position,
            speed=speed,
            state=GREEN if held else UNKNOWN,
            lamp=GREEN if position < LINE_AT else UNKNOWN,
            minute_of_day=720.0,
            now=t,
        )
        if position > LINE_AT and decision.target_speed is not None:
            slowest = min(slowest, decision.target_speed)
        position += speed * dt
        t += dt
    assert slowest == math.inf, "not slowed for"


def test_a_lamp_going_out_of_view_is_stopped_for_where_it_was_seen_going() -> None:
    """Leipzig (16 November): a Volvo on the Actros's camera, told its near
    lamp would stay in view to 10 m, lost it at 14.5 and stood 16 s at a
    green it could not see. Measured from the lamp's own way across the view
    -- the camera another truck's, the pole 3 m short of the line -- the
    truck stops where the lamp is still in view."""
    from core.control.traffic_light import STOP_SHORT_MAX_M, stop_short_from_sightings

    view = _view(178.0, [(1655.0, 80.0, 1920.0, 405.0)])
    focal, horizon, centre, height = 900.0, 400.0, 955.0, 2.6  # the truck's own
    eye_ahead, setback, up, right = 1.8, 3.0, 5.0, 4.5

    def lamp(to_line: float) -> tuple[float, float]:
        z = to_line - setback - eye_ahead
        return centre + focal * right / z, horizon - focal * (up - height) / z

    sightings = [(d, *lamp(d)) for d in np.arange(40.0, 24.0, -0.5)]
    # Taken about the Actros's middle of the view, not the truck's own.
    stop = stop_short_from_sightings(sightings, (960.0, 438.0), view)
    assert stop is not None
    assert stop > STOP_SHORT_M
    assert view.shows(*lamp(stop)) or stop == STOP_SHORT_MAX_M
    lost = max(d for d in np.arange(10.0, 40.0, 0.25) if not view.shows(*lamp(d)))
    assert stop <= lost + 3.0, "not much further back than needed"


def test_a_lamp_seen_too_little_is_not_measured() -> None:
    from core.control.traffic_light import stop_short_from_sightings

    few = [(30.0, 1200.0, 300.0), (29.5, 1205.0, 298.0)]
    assert stop_short_from_sightings(few, (960.0, 438.0), _view()) is None


def test_short_of_a_red_the_truck_is_not_sped_up_towards_it() -> None:
    """A red seen 65 m off at 34 km/h: the truck sped up to 38 towards it,
    and braked the harder for it later (18 November)."""
    lights = TrafficLights([LightLine(LINE_AT, LIGHT)])
    decision = lights.step(
        progress=LINE_AT - 65.0, speed=9.3, state=RED, minute_of_day=720.0, now=0.0
    )
    assert decision.target_speed is not None and decision.target_speed <= 9.3 + 1e-6
    # From slower, a crawl up to the line still.
    slow = TrafficLights([LightLine(LINE_AT, LIGHT)])
    crawl = slow.step(progress=LINE_AT - 40.0, speed=0.5, state=RED, minute_of_day=720.0, now=0.0)
    assert crawl.target_speed > 0.5


def test_a_light_s_stop_is_moved_back_only_as_far_as_is_stopped_short_of_gently() -> None:
    """Moved 10 m back 32 m short of the line at 38 km/h, a red was braked
    for at 4 to 5.4 m/s^2, three times on one drive (18 November)."""
    from tools.follow import LIGHT_STOP_MOVE_DECELERATION, moved_stop

    speed = 38 / 3.6
    assert moved_stop(10.0, 20.0, 32.0, speed) == 10.0, "too late to move it back at all"
    stop = moved_stop(10.0, 20.0, 55.0, speed)
    assert 10.0 < stop < 20.0, "back as far as is gentle"
    assert speed**2 / (2.0 * (55.0 - stop)) == pytest.approx(LIGHT_STOP_MOVE_DECELERATION)
    assert moved_stop(10.0, 20.0, 70.0, speed) == 20.0, "in good time: all the way"
    assert moved_stop(16.0, 12.0, 20.0, speed) == 12.0, "towards the line: at once"
    assert moved_stop(15.0, 20.0, 18.0, speed) == 15.0, "never nearer than it was"
