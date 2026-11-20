"""Adaptive cruise: perception and the route together, and what blindness does."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from core.control.acc import (
    CLOSE_HOLD_S,
    CREEP_PAST_MPS,
    LEAD_HOLD_S,
    AdaptiveCruise,
    PerceptionLost,
    lane_ahead,
)
from core.localization.path import ReferencePath
from core.perception.camera import GroundCamera
from core.perception.service import PerceptionReport, TrackReport

CAMERA = GroundCamera(820.0, 438.0, 960.0, 2.9, seat_right_m=-0.86)
# Northwards along +z from the origin, then bending right: +x is the left at
# heading zero (core.control.lateral), so a right turn goes towards -x.
STRAIGHT = ReferencePath(np.column_stack([np.zeros(200), np.arange(200.0)]))
ARC = np.radians(np.linspace(0.0, 90.0, 100))
BEND = ReferencePath(np.column_stack([-(60.0 - 60.0 * np.cos(ARC)), 60.0 * np.sin(ARC)]))


def report(*tracks: TrackReport, grabbed_at: float = 100.0, covered: bool = False):
    return PerceptionReport(grabbed_at, 1, tracks, 20.0, covered=covered)


def car(distance: float, lateral: float, closing: float = 0.0, track_id: int = 1) -> TrackReport:
    return TrackReport(track_id, "car", True, True, distance, closing, lateral, 1.9)


def test_the_route_ahead_is_seen_from_the_drivers_seat() -> None:
    lane = lane_ahead(STRAIGHT, 0.0, 0.0, 0.0, 0.0, CAMERA)
    assert lane.right_at(50.0) == pytest.approx(0.86), "the lane's middle, right of the driver"
    bend = lane_ahead(BEND, 0.0, 0.0, 0.0, 0.0, CAMERA)
    assert bend.right_at(40.0) > 10.0, "a right bend runs off to the right"


def test_a_car_in_the_lane_is_followed_and_one_beside_it_is_not() -> None:
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=20.0)
    beside = cruise.step(report(car(30.0, 5.0)), 100.1, **kwargs)
    assert beside.lead is None and beside.decision.target_speed is None
    ahead = cruise.step(report(car(30.0, 5.0), car(45.0, 1.0, -3.0, 2)), 100.1, **kwargs)
    assert ahead.lead_id == 2
    assert ahead.decision.target_speed < 20.0


def test_in_a_bend_the_car_on_the_road_is_the_lead_not_the_one_straight_ahead() -> None:
    cruise = AdaptiveCruise(CAMERA)
    lane = lane_ahead(BEND, 0.0, 0.0, 0.0, 0.0, CAMERA)
    on_the_road = car(40.0, lane.right_at(40.0), track_id=7)
    straight_on = car(30.0, 0.0, track_id=8)
    step = cruise.step(
        report(on_the_road, straight_on),
        100.1,
        path=BEND,
        progress=0.0,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=15.0,
    )
    assert step.lead_id == 7


@pytest.mark.parametrize(
    ("seen", "why"),
    [
        (None, "no perception"),
        (report(grabbed_at=99.0), "s old"),
        (report(covered=True), "in front of the game"),
    ],
)
def test_without_a_fresh_view_of_the_road_the_truck_stops(seen, why) -> None:
    with pytest.raises(PerceptionLost, match=why):
        AdaptiveCruise(CAMERA).step(
            seen, 100.1, path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=20.0
        )


def _closed_loop(
    lead_speed, gap: float, speed: float, seconds: float, seed: int = 0, set_speed: float = 25.0
):
    """The truck on a straight route, a lead `gap` ahead: boxes from the camera
    model with a pixel of noise, 20 Hz perception 0.1 s late, the real tracker,
    the real cruise, the speed controller and the measured actuator."""
    from core.control.longitudinal import SpeedController
    from core.perception.detection import Detection, Kind
    from core.perception.tracking import Tracker

    rng = np.random.default_rng(seed)
    road = ReferencePath(np.column_stack([np.zeros(1500), np.arange(1500.0)]))
    cruise = AdaptiveCruise(CAMERA)
    tracker = Tracker(CAMERA)
    controller = SpeedController()
    actuators = controller.actuators
    dt, delay, lag = 1 / 60, 0.1, 0.3
    truck, lead, acceleration = 0.0, gap, 0.0
    reports: list[PerceptionReport] = []
    closest = gap
    for k in range(int(seconds / dt)):
        now = k * dt
        if lead - truck <= 0.0:
            return lead - truck, speed  # hit it
        ahead = lead - truck + 1.74  # the eye sits over the front axle
        if k % 3 == 0 and ahead < 9.5:
            # Its foot is behind the dashboard: the detector's box is cut and
            # the tracker measures nothing, as in the game below about 9.5 m.
            tracker.update([], now)
            reports.append(
                PerceptionReport(
                    now,
                    k,
                    tuple(
                        TrackReport(t.id, "car", t.confirmed, True, *t.estimate())
                        for t in tracker.tracks
                        if t.estimate() is not None
                    ),
                    20.0,
                )
            )
        elif k % 3 == 0:  # 20 Hz
            scale = CAMERA.focal_px / ahead
            bottom = CAMERA.horizon_row + CAMERA.focal_px * CAMERA.height_m / ahead
            centre = CAMERA.centre_col + 0.86 * scale + rng.normal(0, 0.5)
            box = (
                centre - 0.95 * scale + rng.normal(0, 1.0),
                bottom - 1.5 * scale,
                centre + 0.95 * scale + rng.normal(0, 1.0),
                bottom + rng.normal(0, 1.0),
            )
            tracker.update([Detection(Kind.CAR, 0.9, box)], now)
            reports.append(
                PerceptionReport(
                    now,
                    k,
                    tuple(
                        TrackReport(t.id, "car", t.confirmed, True, *t.estimate())
                        for t in tracker.tracks
                        if t.estimate() is not None
                    ),
                    20.0,
                )
            )
        late = [r for r in reports if r.grabbed_at <= now - delay] or reports[:1]
        step = cruise.step(
            late[-1],
            late[-1].grabbed_at + delay,
            path=road,
            progress=truck,
            x=0.0,
            z=truck,
            course=0.0,
            speed=speed,
        )
        target = (
            25.0 if step.decision.target_speed is None else min(25.0, step.decision.target_speed)
        )
        if step.decision.urgent_deceleration is not None:
            command = controller.urgent(step.decision.urgent_deceleration, dt)
        else:
            command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        delivered = actuators.acceleration_for(command.throttle, command.brake)
        acceleration += (delivered - acceleration) * dt / lag
        speed = max(0.0, speed + acceleration * dt)
        truck += speed * dt
        lead += lead_speed(now) * dt
        closest = min(closest, lead - truck)
    return closest, speed


def test_a_stopped_car_first_seen_at_90_m_is_stopped_for_through_the_real_tracker() -> None:
    """Closing speed waits for a settled track and hard braking for 0.1 s of
    asking. At 80 km/h a stopped car must be seen by about 75 m: at 70 m the
    truck stops 2-3 m short, at 60 m it does not stop in time."""
    for seed in range(3):
        closest, speed = _closed_loop(
            lambda t: 0.0, gap=90.0, speed=80 / 3.6, seconds=25, seed=seed
        )
        assert speed < 0.5
        assert closest > 4.0, f"seed {seed}: stopped {closest:.1f} m short"


def test_following_a_steady_car_settles_through_the_real_tracker() -> None:
    closest, speed = _closed_loop(lambda t: 70 / 3.6, gap=80.0, speed=80 / 3.6, seconds=60)
    assert speed == pytest.approx(70 / 3.6, abs=1.0)
    assert closest > 30.0


def test_a_stopped_vehicle_that_drops_out_of_view_close_up_is_still_stopped_for() -> None:
    """What happened in the game: approaching a dropped trailer at 25 km/h the
    truck slowed to 5 km/h, lost sight of it 10.5 m out -- its wheels behind
    the dashboard -- took the road for clear, and accelerated. The distance
    budget stopped it about 5 m short."""
    closest, speed = _closed_loop(
        lambda t: 0.0, gap=80.0, speed=0.0, seconds=40, set_speed=25 / 3.6
    )
    assert speed < 0.1, "stopped, and stays stopped"
    assert closest > 3.0


def test_following_a_slowing_lead_does_not_pulse_the_brake(monkeypatch) -> None:
    """What happened in the game: following a lead from 84 to 43 km/h, 18 brake
    applications a minute, driven by 3 km/h of noise in the lead's speed. Here,
    through the real tracker: 41 a minute unsmoothed, 8 smoothed."""
    import core.control.longitudinal as longitudinal

    applications = {"count": 0, "braking": False}
    step = longitudinal.SpeedController.step

    def counting(self, **kwargs):
        command = step(self, **kwargs)
        braking = command.brake > 0
        applications["count"] += braking and not applications["braking"]
        applications["braking"] = braking
        return command

    monkeypatch.setattr(longitudinal.SpeedController, "step", counting)
    kmh = 1 / 3.6

    def lead(t: float) -> float:
        return max(43 * kmh, 70 * kmh - 0.45 * max(0.0, t - 5.0))

    closest, speed = _closed_loop(lead, gap=45.0, speed=70 * kmh, seconds=60, set_speed=90 * kmh)
    assert applications["count"] <= 15, f"{applications['count']} brake applications in a minute"
    assert closest > 25.0
    assert speed == pytest.approx(43 * kmh, abs=1.5)


def test_a_hard_stop_comes_from_the_raw_estimate_not_the_smoothed_one() -> None:
    """What happened: a lead's smoothed speed still held a bad reading; it asked
    for 3.9 m/s^2 in the game while the raw estimate asked for nothing."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=15.0)
    # A bad first reading: the lead closing at 12 m/s, 30 m out.
    cruise.step(report(car(30.0, 0.9, -12.0), grabbed_at=100.0), 100.05, **kwargs)
    # Then the same lead, sensibly: keeping its distance.
    for k in range(1, 8):
        now = 100.0 + 0.05 * k
        step = cruise.step(report(car(30.0, 0.9, 0.0), grabbed_at=now), now + 0.05, **kwargs)
        assert step.decision.urgent_deceleration is None, f"after {0.05 * k:.2f} s"


def test_a_car_that_is_the_lead_only_for_a_moment_is_not_remembered() -> None:
    """What happened in the game: stopped at a red light, a car crossing the
    junction 9.8 m ahead was the lead for a moment, was remembered, and held
    the truck at rest through the whole green that followed."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    for k in range(4):
        now = 100.0 + 0.05 * k
        cruise.step(report(car(9.8, 0.9), grabbed_at=now), now + 0.01, **kwargs)
    gone = cruise.step(report(grabbed_at=101.0), 101.01, **kwargs)
    assert gone.lead is None and gone.decision.target_speed is None


def test_a_lead_followed_for_a_while_is_remembered_when_it_drops_out_of_view() -> None:
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    for k in range(30):
        now = 100.0 + 0.05 * k
        cruise.step(report(car(9.8, 0.9), grabbed_at=now), now + 0.01, **kwargs)
    gone = cruise.step(report(grabbed_at=102.0), 102.01, **kwargs)
    assert gone.lead is not None
    assert gone.decision.target_speed == pytest.approx(0.0, abs=0.3)


def test_a_lorry_turning_across_the_lane_is_stopped_for() -> None:
    """Dover: the lorry was the lead, then crossing, then ignored, and hit."""
    step = lorry_turning_across(junctions=[(15.0, 40.0)])
    assert step.lead_id == 5
    assert "giving way" in step.decision.reason
    assert step.decision.target_speed < 8.0


def test_at_speed_through_a_junction_a_vehicle_seeming_to_cross_is_not_given_way_to() -> None:
    """England, 75 and 80 km/h: coaches beyond the median read as crossing, and
    the truck was braked at 6.8 m/s^2 for them."""
    from core.control.acc import CROSSING_BELOW_MPS

    step = lorry_turning_across(junctions=[(0.0, 100.0)], speed=CROSSING_BELOW_MPS + 5.0)
    assert "giving way" not in step.decision.reason
    assert (step.decision.urgent_deceleration or 0.0) == 0.0


def test_on_an_open_road_a_vehicle_seeming_to_cross_is_not() -> None:
    """France: a lorry in the next lane, swept across by a bend, read as
    crossing and braked the truck hard again and again."""
    step = lorry_turning_across(junctions=[])
    assert step.lead_id is None
    assert step.decision.target_speed is None


def test_a_car_joining_from_a_slip_road_is_followed_not_given_way_to() -> None:
    """Out of Folkestone: cars coming down a slip road, as fast as the truck,
    were given way to as if standing still, at 7 m/s^2."""
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = [(0.0, 100.0)]
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=20.0)
    steps = []
    for k in range(40):
        t = 100.0 + k / 20
        joining = TrackReport(7, "car", True, True, 25.0, 0.0, max(0.9, 5.0 - 1.8 * (k / 20)), 1.9)
        steps.append(cruise.step(report(joining, grabbed_at=t), t + 0.05, **kwargs))
    assert all("giving way" not in step.decision.reason for step in steps)
    assert all(not step.decision.urgent_deceleration for step in steps)
    assert steps[-1].lead_id == 7, "in the lane, it is followed"


def lorry_turning_across(junctions, speed: float = 8.0):
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = junctions
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=speed)
    step = None
    for k in range(12):
        t = 100.0 + k / 20
        lorry = TrackReport(5, "truck", True, True, 20.0, -8.0, 6.0 - 2.0 * (k / 20), 12.0)
        step = cruise.step(report(lorry, grabbed_at=t), t + 0.05, **kwargs)
    return step


def junction(crossed: bool):
    """300 m north, then a junction lane turning left; with `crossed`, the
    oncoming lane of the same junction runs south across it."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from tests.unit.lane_networks import polyline

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(Lane(points=polyline((0, 300), (25, 330)), source="prefab", owner="2"))
    if crossed:
        graph.lanes.append(Lane(points=polyline((15, 345), (15, 290)), source="prefab", owner="2"))
    else:  # a slip road joining at the junction's end
        graph.lanes.append(Lane(points=polyline((40, 300), (25, 330)), source="prefab", owner="2"))
    graph.connect(0, 1)
    return RoadNetwork.compile(graph)


def test_a_route_gives_way_where_it_crosses_another_lane_of_a_junction() -> None:
    """Out of Mannheim the truck turned left across the oncoming lane and hit
    a car coming the other way."""
    from core.control.acc import junction_spans
    from core.planning.route import ONWARD, Route

    net = junction(crossed=True)
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    ((start, end),) = junction_spans(route, path)
    assert start == pytest.approx(300.0, abs=2.0)
    assert end == pytest.approx(300.0 + float(net.lane_length[1]), abs=2.0)


def test_a_junction_whose_lanes_only_meet_end_to_end_is_not_given_way_in() -> None:
    """A slip road joining a motorway: 7 m/s^2 of braking for cars going along."""
    from core.control.acc import junction_spans
    from core.planning.route import ONWARD, Route

    net = junction(crossed=False)
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    assert junction_spans(route, path) == []


def oncoming_car(distance: float, closing: float) -> TrackReport:
    return TrackReport(9, "car", True, True, distance, closing, 2.5, 1.9)


def turning_cruise(progress: float, speed: float, track: TrackReport):
    cruise = AdaptiveCruise(CAMERA)
    cruise.turns_across = [(30.0, 60.0)]
    kwargs = dict(path=STRAIGHT, progress=progress, x=0.0, z=progress, course=0.0, speed=speed)
    step = None
    for k in range(6):
        t = 100.0 + k / 20
        step = cruise.step(report(track, grabbed_at=t), t + 0.05, **kwargs)
    return step


def test_short_of_a_left_turn_the_oncoming_car_is_waited_for() -> None:
    """Out of Mannheim the truck turned left in front of a car coming the
    other way at 15 m/s, 50 m off, and hit it."""
    from core.control.following import STOP_DECELERATION

    cruise = AdaptiveCruise(CAMERA)
    cruise.turns_across = [(30.0, 60.0)]
    kwargs = dict(path=STRAIGHT, progress=10.0, x=0.0, z=10.0, course=0.0, speed=8.0)
    for k in range(6):
        t = 100.0 + k / 20
        step = cruise.step(report(oncoming_car(50.0, -23.0), grabbed_at=t), t + 0.05, **kwargs)
    assert cruise._give_way is not None
    assert cruise._give_way.why == "waiting for the oncoming traffic"
    room = 30.0 - 2.0 - 10.0
    assert step.decision.target_speed <= math.sqrt(2 * STOP_DECELERATION * room) + 1e-6
    from core.control.following import COMFORT_DECELERATION

    urgent = step.decision.urgent_deceleration
    assert urgent is None or urgent <= COMFORT_DECELERATION, "stopped for in comfort"


def test_once_waiting_for_the_oncoming_traffic_it_waits_until_the_way_is_clear() -> None:
    """Out of Milan, in one left turn: braked for an oncoming car, lost sight
    of it, full throttle, saw it again, braked hard -- three times."""
    from core.control.acc import GIVE_WAY_CLEAR_S

    cruise = AdaptiveCruise(CAMERA)
    cruise.turns_across = [(30.0, 60.0)]
    kwargs = dict(path=STRAIGHT, progress=20.0, x=0.0, z=20.0, course=0.0, speed=3.0)
    for k in range(6):
        t = 100.0 + k / 20
        cruise.step(report(oncoming_car(50.0, -18.0), grabbed_at=t), t + 0.05, **kwargs)
    seen_last = 100.0 + 5 / 20
    steps = {}
    for k in range(1, 50):
        t = seen_last + k / 20
        steps[round(k / 20, 2)] = cruise.step(report(grabbed_at=t), t + 0.05, **kwargs)
    still = [s for wait, s in steps.items() if wait + 0.05 < GIVE_WAY_CLEAR_S]
    assert all("waiting for the oncoming" in s.decision.reason for s in still)
    assert all(s.decision.target_speed == pytest.approx(3.0) for s in still)
    assert "waiting for the oncoming" not in steps[2.0].decision.reason


def test_a_lead_lost_for_a_moment_is_still_followed() -> None:
    """Lost for a view in a bend, the lead was gone, and the truck went from
    braking to full throttle and back within 0.2 s."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=15.0)
    for k in range(10):
        now = 100.0 + 0.05 * k
        seen = cruise.step(report(car(30.0, 0.9, -3.0), grabbed_at=now), now + 0.05, **kwargs)
    lost = cruise.step(report(grabbed_at=100.7), 100.75, **kwargs)
    assert lost.decision.target_speed == pytest.approx(seen.decision.target_speed, abs=1.0)
    assert lost.decision.urgent_deceleration is None
    gone = cruise.step(report(grabbed_at=101.5), 101.55, **kwargs)
    assert gone.decision.target_speed is None


def test_waiting_to_turn_a_lead_nearer_than_the_turn_is_still_followed() -> None:
    cruise = AdaptiveCruise(CAMERA)
    cruise.turns_across = [(60.0, 90.0)]
    kwargs = dict(path=STRAIGHT, progress=10.0, x=0.0, z=10.0, course=0.0, speed=8.0)
    for k in range(10):
        t = 100.0 + k / 20
        step = cruise.step(
            report(oncoming_car(80.0, -20.0), car(14.0, 0.9, -8.0, track_id=3), grabbed_at=t),
            t + 0.05,
            **kwargs,
        )
    assert step.lead_id == 3


def test_in_the_turn_it_goes_on() -> None:
    step = turning_cruise(progress=35.0, speed=6.0, track=oncoming_car(40.0, -21.0))
    assert "waiting for the oncoming" not in step.decision.reason


def test_an_oncoming_car_far_enough_off_leaves_a_gap_to_turn_in() -> None:
    step = turning_cruise(progress=10.0, speed=8.0, track=oncoming_car(95.0, -10.0))
    assert "waiting for the oncoming" not in step.decision.reason


def test_going_straight_over_a_junction_the_oncoming_traffic_is_not_waited_for() -> None:
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = [(30.0, 60.0)]
    kwargs = dict(path=STRAIGHT, progress=10.0, x=0.0, z=10.0, course=0.0, speed=8.0)
    for k in range(6):
        t = 100.0 + k / 20
        step = cruise.step(report(oncoming_car(50.0, -23.0), grabbed_at=t), t + 0.05, **kwargs)
    assert "waiting for the oncoming" not in step.decision.reason


def test_a_left_turn_is_where_the_route_crosses_a_lane_coming_the_other_way() -> None:
    from core.control.acc import turn_across_spans
    from core.planning.route import ONWARD, Route

    net = junction(crossed=True)  # its crossed lane runs south, against the truck
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    assert len(turn_across_spans(route, path)) == 1
    assert turn_across_spans(Route(junction(False), (0, 1), (ONWARD,)), path) == []


def test_coming_to_a_left_turn_with_nothing_coming_the_truck_does_not_crawl() -> None:
    """The driver's word (16 November): no crawl short of a turn, the bend's
    own speed; what is coming is still waited for."""
    clear = TrackReport(9, "car", True, True, 200.0, 0.0, 30.0, 1.9)
    for progress, speed in ((20.0, 10.0), (29.0, 3.0), (-300.0, 20.0)):
        step = turning_cruise(progress=progress, speed=speed, track=clear)
        assert step.decision.target_speed is None, step.decision.reason


def test_with_the_game_s_camera_out_of_the_cab_the_truck_stops() -> None:
    """In Bern, from the chase camera, the truck's own trailer was taken for
    a vehicle 9.5 m ahead and the truck never moved again."""
    from dataclasses import replace

    from core.control.acc import OUT_OF_CAB_S, OutOfCab

    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=10.0)
    outside = replace(report(grabbed_at=100.0), in_cab=False)
    cruise.step(outside, 100.05, **kwargs)
    # Its own kind of lost: never taken for a view a moment late (tools/follow.py),
    # which near Bergen slowed the truck to a stop and held it, saying nothing.
    with pytest.raises(OutOfCab, match="camera"):
        later = replace(report(grabbed_at=100.0 + OUT_OF_CAB_S), in_cab=False)
        cruise.step(later, 100.05 + OUT_OF_CAB_S, **kwargs)
    # How long it has been out: tools/follow.py slows for a look about, and
    # stops the drive only for a camera out for good (OUT_OF_CAB_GIVE_UP_S).
    assert cruise.out_of_cab_s == pytest.approx(OUT_OF_CAB_S)
    back = replace(report(grabbed_at=102.0), in_cab=True)
    cruise.step(back, 102.05, **kwargs)
    assert cruise.out_of_cab_s == 0.0


def test_standing_the_traffic_alongside_moving_off_is_seen() -> None:
    from core.control.acc import MOVING_OFF_S

    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    lorry = lambda closing: TrackReport(4, "truck", True, True, 14.0, closing, -3.5, 2.5)  # noqa: E731
    cruise.step(report(lorry(0.0), grabbed_at=100.0), 100.05, **kwargs)
    assert not cruise.traffic_going
    for k in range(1, 20):
        t = 100.0 + k * 0.05
        cruise.step(report(lorry(3.0), grabbed_at=t), t + 0.05, **kwargs)
        if k * 0.05 < MOVING_OFF_S - 0.06:
            assert not cruise.traffic_going
    assert cruise.traffic_going
    moving = dict(kwargs, speed=5.0)
    cruise.step(report(lorry(0.0), grabbed_at=102.0), 102.05, **moving)
    assert not cruise.traffic_going, "only while the truck stands"


def lorry_coming_round_the_ring(rings):
    """A lorry crossing from the left 18 m ahead, side on, moving partly the
    truck's way: one coming round a roundabout as the truck enters it."""
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = [(15.0, 30.0)]
    cruise.rings = rings
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=8.0)
    step = None
    for k in range(12):
        t = 100.0 + k / 20
        lorry = TrackReport(6, "truck", True, True, 18.0, -2.0, -6.0 + 3.0 * (k / 20), 9.0)
        step = cruise.step(report(lorry, grabbed_at=t), t + 0.05, **kwargs)
    return step


def test_into_a_roundabout_the_ring_s_traffic_is_given_way_to_however_it_moves() -> None:
    """Near Duisburg the truck entered a roundabout at 32 km/h and hit a
    lorry coming round: moving partly the truck's way, it read as joining."""
    entering = lorry_coming_round_the_ring(rings=[(15.0, 30.0, -1)])
    assert "giving way" in entering.decision.reason
    elsewhere = lorry_coming_round_the_ring(rings=[])
    assert "giving way" not in elsewhere.decision.reason, "joining the traffic, as before"


def t_junction(route_turns: bool):
    """North through a junction; its other lane comes in from the east and
    turns south across the route -- or, `route_turns`, the route turns left
    across a lane going straight on."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from tests.unit.lane_networks import polyline

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    if route_turns:
        graph.lanes.append(
            Lane(
                points=polyline((0, 300), (0, 306), (-6, 316), (-20, 320)),
                source="prefab",
                owner="2",
            )
        )
        graph.lanes.append(Lane(points=polyline((-4, 330), (-4, 290)), source="prefab", owner="2"))
    else:
        graph.lanes.append(Lane(points=polyline((0, 300), (0, 330)), source="prefab", owner="2"))
        graph.lanes.append(
            Lane(
                points=polyline((20, 312), (6, 312), (-2, 306), (-4, 296)),
                source="prefab",
                owner="2",
            )
        )
    graph.connect(0, 1)
    return RoadNetwork.compile(graph)


def test_going_straight_on_the_route_has_the_right_of_way_over_lanes_turning_across() -> None:
    """Near Duisburg, straight on past side roads, the lanes turning into and
    out of them crossed the route: the truck crept, and stopped for parked
    cars by the junction."""
    from core.control.acc import junction_spans, turn_across_spans
    from core.planning.route import ONWARD, Route

    straight = Route(t_junction(route_turns=False), (0, 1), (ONWARD,))
    path = ReferencePath(straight.points()[:, [0, 2]], spacing=2.0)
    assert junction_spans(straight, path) == []
    assert turn_across_spans(straight, path) == []
    turning = Route(t_junction(route_turns=True), (0, 1), (ONWARD,))
    path = ReferencePath(turning.points()[:, [0, 2]], spacing=2.0)
    assert len(junction_spans(turning, path)) == 1, "turning across, it gives way"


def roundabout(way_out: bool = False):
    """A ring of four quarter arcs, radius 20, round (0, 340); a road north
    to its southern point, and an entry lane joining the ring there; with
    `way_out`, a lane out of the ring at the end of its first quarter."""
    from core.map import Lane, LaneGraph, RoadNetwork
    from tests.unit.lane_networks import polyline

    centre, radius = (0.0, 340.0), 20.0

    def arc(a0: float, a1: float) -> np.ndarray:
        # Anticlockwise seen from above, as traffic keeping right goes round.
        angles = np.linspace(a0, a1, 12)
        return np.column_stack(
            [centre[0] + radius * np.cos(angles), np.zeros(12), centre[1] + radius * np.sin(angles)]
        )

    south = -math.pi / 2
    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(
        Lane(
            points=polyline(
                (0, 300),
                (6, 312),
                (radius * math.cos(south + 0.4), 340 + radius * math.sin(south + 0.4)),
            ),
            source="prefab",
            owner="2",
        )
    )
    quarters = [
        arc(south + 0.4 + q * math.pi / 2, south + 0.4 + (q + 1) * math.pi / 2) for q in range(4)
    ]
    for points in quarters:
        graph.lanes.append(Lane(points=points, source="prefab", owner="2"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    for q in range(4):
        graph.connect(2 + q, 2 + (q + 1) % 4)
    if way_out:
        angle = south + 0.4 + math.pi / 2
        start = (radius * math.cos(angle), 340 + radius * math.sin(angle))
        end = (start[0] + 15 * math.cos(angle), start[1] + 15 * math.sin(angle))
        graph.lanes.append(Lane(points=polyline(start, end), source="prefab", owner="2"))
        graph.connect(2, 6)
    return RoadNetwork.compile(graph)


def test_a_roundabouts_way_out_is_told_from_its_ring_and_its_way_in() -> None:
    from core.control.acc import out_of_a_ring

    network = roundabout(way_out=True)
    assert out_of_a_ring(network, 6)
    assert not out_of_a_ring(network, 1), "the way in"
    assert not out_of_a_ring(network, 2), "the ring"
    assert not out_of_a_ring(network, 0), "a road"


def test_the_way_into_a_roundabout_gives_way_to_the_ring() -> None:
    from core.control.acc import junction_spans, ring_entry_spans
    from core.planning.route import ONWARD, Route

    route = Route(roundabout(), (0, 1, 2), (ONWARD, ONWARD))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    (entry,) = ring_entry_spans(route, path)
    assert entry[0] == pytest.approx(300.0, abs=2.0)
    assert entry[2] == -1, "traffic keeping right comes round from the left"
    assert any(start <= entry[0] <= end for start, end in junction_spans(route, path))


def test_the_way_through_a_roundabout_runs_from_its_entry_to_its_way_out() -> None:
    from core.control.acc import ring_entry_spans, ring_way_spans
    from core.planning.route import ONWARD, Route

    route = Route(roundabout(), (0, 1, 2, 3), (ONWARD, ONWARD, ONWARD))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    (entry,) = ring_entry_spans(route, path)
    (way,) = ring_way_spans(route, path)
    assert way[0] == pytest.approx(entry[0])
    assert way[1] == pytest.approx(path.length, abs=2.0), "to the end of the ring's last lane"


def test_round_a_roundabout_and_out_no_crawl_once_in() -> None:
    """Edged into, as a driver would; round it and out at the bend's own
    speed (the driver's word, 16 November: not 12 km/h)."""
    from core.control.acc import RING_ENTRY_MPS

    cruise = AdaptiveCruise(CAMERA)
    cruise.rings = [(60.0, 70.0, -1)]
    cruise.ring_ways = [(60.0, 100.0)]
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=4.0)
    edging = cruise.step(report(grabbed_at=100.0), 100.05, progress=62.0, **kwargs)
    assert edging.decision.target_speed == pytest.approx(RING_ENTRY_MPS)
    round_ = cruise.step(report(grabbed_at=100.1), 100.15, progress=80.0, **kwargs)
    assert round_.decision.target_speed is None
    out = cruise.step(report(grabbed_at=100.2), 100.25, progress=105.0, **kwargs)
    assert out.decision.target_speed is None


def test_coming_to_a_roundabout_the_truck_slows_to_take_it_as_a_driver_would() -> None:
    from core.control.acc import RING_ENTRY_MPS

    cruise = AdaptiveCruise(CAMERA)
    cruise.rings = [(60.0, 70.0, -1)]
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=9.0)
    far = cruise.step(report(grabbed_at=100.0), 100.05, progress=0.0, **kwargs)
    assert far.decision.target_speed is None or far.decision.target_speed >= 9.0
    near = cruise.step(report(grabbed_at=100.1), 100.15, progress=50.0, **kwargs)
    assert near.decision.target_speed == pytest.approx(
        max(math.sqrt(2 * 1.5 * 8.0), RING_ENTRY_MPS), abs=0.05
    )
    assert "roundabout" in near.decision.reason
    at = cruise.step(report(grabbed_at=100.2), 100.25, progress=59.0, **kwargs)
    assert at.decision.target_speed == pytest.approx(RING_ENTRY_MPS), "not stopped: rolled in"
    edging = cruise.step(report(grabbed_at=100.3), 100.35, progress=65.0, **kwargs)
    assert edging.decision.target_speed == pytest.approx(RING_ENTRY_MPS), "edged in, a car length"
    round_ = cruise.step(report(grabbed_at=100.4), 100.45, progress=72.0, **kwargs)
    assert round_.decision.target_speed is None or round_.decision.target_speed > RING_ENTRY_MPS


def test_into_a_roundabout_a_vehicle_close_by_on_the_ring_s_side_is_waited_for() -> None:
    """Near Duisburg a lorry coming round was in the left door window for
    3 s, its wheels under the dashboard: no distance, so no crossing, and the
    truck drove into it."""
    cruise = AdaptiveCruise(CAMERA)
    cruise.rings = [(40.0, 50.0, -1)]
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=6.0)
    close_left = PerceptionReport(100.0, 1, (), 20.0, unseen=((7, 200.0),))
    step = cruise.step(close_left, 100.05, progress=32.0, **kwargs)
    assert step.lead_id == 7, "given way to"
    assert step.decision.target_speed < 6.0, "stopping short of the entry"
    right = PerceptionReport(100.1, 2, (), 20.0, unseen=((8, 1700.0),))
    other = AdaptiveCruise(CAMERA)
    other.rings = [(40.0, 50.0, -1)]
    assert other.step(right, 100.15, progress=32.0, **kwargs).lead_id is None
    away = AdaptiveCruise(CAMERA)
    away.rings = [(40.0, 50.0, -1)]
    far = away.step(close_left, 100.05, progress=5.0, **kwargs)
    assert far.lead_id is None, "not yet at the entry"


def test_into_a_roundabout_a_vehicle_seen_coming_round_is_waited_for_out_of_sight() -> None:
    """Near Duisburg a lorry coming round was seen 19 m off, 13 m to the left,
    then hidden by the pillar and the mirror until it was across the truck's
    way."""
    from core.control.acc import RING_MEMORY_S

    cruise = AdaptiveCruise(CAMERA)
    cruise.rings = [(40.0, 50.0, -1)]
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=5.0)
    for k, lateral in enumerate((-14.0, -13.0)):  # coming across, 2 m/s
        lorry = TrackReport(9, "truck", True, True, 19.0, -5.0, lateral, 9.0)
        cruise.step(report(lorry, grabbed_at=99.5 + k / 2), 99.55 + k / 2, progress=25.0, **kwargs)
    hidden = cruise.step(report(grabbed_at=102.0), 102.05, progress=33.0, **kwargs)
    assert hidden.lead_id == 9 and hidden.decision.target_speed < 5.0, "out of sight, waited for"
    later = 100.0 + RING_MEMORY_S + 0.5
    gone = cruise.step(report(grabbed_at=later), later + 0.05, progress=36.0, **kwargs)
    assert gone.lead_id is None, "long enough gone by"
    right = AdaptiveCruise(CAMERA)
    right.rings = [(40.0, 50.0, -1)]
    parked = TrackReport(4, "car", True, True, 19.0, -5.0, 8.0, 1.9)
    step = right.step(report(parked, grabbed_at=100.0), 100.05, progress=25.0, **kwargs)
    assert step.lead_id is None, "on the other side: not the ring's traffic"
    standing = AdaptiveCruise(CAMERA)
    standing.rings = [(40.0, 50.0, -1)]
    for k in range(10):  # parked by the entry: still, closing at the truck's speed
        parked = TrackReport(5, "car", True, True, 25.0 - k, -5.0, -6.0, 1.9)
        step = standing.step(
            report(parked, grabbed_at=100.0 + k / 5), 100.05 + k / 5, progress=25.0 + k, **kwargs
        )
    assert step.lead_id is None, "a parked car is not the ring's traffic"


TORINO_CUT_IN = (
    (-3.50, 49.5, 0.0, 5.50, 4.10),
    (-3.27, 47.8, 0.0, 5.17, 4.20),
    (-3.04, 40.8, -12.8, 4.97, 4.30),
    (-2.82, 38.4, -11.8, 4.81, 4.61),
    (-2.58, 35.9, -11.2, 4.62, 4.96),
    (-2.37, 33.6, -11.1, 4.43, 5.36),
    (-2.15, 30.9, -11.6, 4.23, 5.41),
    (-1.91, 27.7, -12.0, 4.00, 5.68),
)
"""(time, distance, closing, lateral, width) of the lorry that moved across from
a slip road out of Torino into the truck's lane at 80 km/h, as the tracker
had it from the black box of drive-08 (2026-10-22): hit, never followed."""


def test_a_lorry_moving_across_into_the_lane_on_the_motorway_is_braked_for() -> None:
    cruise = AdaptiveCruise(CAMERA)
    speed = 22.0
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=speed)
    braking_from = None
    for t, distance, closing, lateral, width in TORINO_CUT_IN:
        lorry = TrackReport(1, "truck", True, True, distance, closing, lateral, width)
        step = cruise.step(report(lorry, grabbed_at=100.0 + t), 100.05 + t, **kwargs)
        target = step.decision.target_speed
        if braking_from is None and step.lead_id == 1 and target is not None and target < speed:
            braking_from = (t, distance)
    assert braking_from is not None, "never braked for"
    assert braking_from[0] <= -2.8 and braking_from[1] >= 35.0


def test_a_lead_going_as_fast_as_the_truck_lost_from_view_is_not_stopped_for() -> None:
    """At night out of Torino a car overtaking on the right was the lead 10 m
    ahead at the truck's speed; it drew away out of view, was remembered as
    standing there, and the truck braked at 7 m/s^2 from 77 km/h to a stop."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=21.0)
    for k in range(40):
        now = 100.0 + 0.05 * k
        cruise.step(
            report(car(10.5, 0.9, closing=0.0), grabbed_at=now), now + 0.01, progress=0.0, **kwargs
        )
    gone = cruise.step(report(grabbed_at=102.0), 102.01, progress=1.0, **kwargs)
    assert gone.decision.urgent_deceleration is None
    assert "stopping for a vehicle out of view" not in gone.decision.reason
    assert gone.decision.target_speed is None or gone.decision.target_speed > 10.0


def test_a_car_coming_the_other_way_lost_from_view_is_not_stopped_for() -> None:
    """In Norway cars coming by on a narrow road were read as standing, the
    closing speed lagging, and remembered once past: the truck stopped for
    each one, 4 m short of no one."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    for k in range(40):
        now = 100.0 + 0.05 * k
        coming = car(14.0 - 6.0 * 0.05 * k, 0.9, closing=-1.0)
        cruise.step(report(coming, grabbed_at=now), now + 0.01, **kwargs)
    cruise.step(report(grabbed_at=102.0), 102.01, **kwargs)
    later = 102.0 + LEAD_HOLD_S + 0.1
    gone = cruise.step(report(grabbed_at=later), later + 0.01, **kwargs)
    assert "stopping for a vehicle out of view" not in gone.decision.reason
    assert gone.decision.target_speed is None


def test_a_lead_standing_close_lost_from_view_is_still_stopped_for() -> None:
    """Its wheels behind the dashboard, a standing lead is still there."""
    cruise = AdaptiveCruise(CAMERA)
    for k in range(40):
        now = 100.0 + 0.05 * k
        cruise.step(
            report(car(12.0, 0.9, closing=-3.0), grabbed_at=now),
            now + 0.01,
            path=STRAIGHT,
            progress=0.0,
            x=0.0,
            z=0.0,
            course=0.0,
            speed=3.0,
        )
    gone = cruise.step(
        report(grabbed_at=102.0),
        102.01,
        path=STRAIGHT,
        progress=1.0,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=3.0,
    )
    assert gone.lead is not None
    assert gone.decision.target_speed == pytest.approx(0.0, abs=0.5)


def test_a_standing_lead_tracked_under_more_than_one_id_is_still_remembered() -> None:
    """Drawing up behind a standing lorry re-tracked as it nears, the truck
    stops for it once its wheels go under the dashboard."""
    cruise = AdaptiveCruise(CAMERA)
    progress, now = 0.0, 100.0
    for k in range(28):
        now = 100.0 + 0.05 * k
        speed = 7.0 - 0.2 * k
        progress += speed * 0.05
        lorry = TrackReport(
            242 if (k // 3) % 2 else 245, "truck", True, True, 12.7 - 0.19 * k, -speed, 0.9, 2.5
        )
        cruise.step(
            report(lorry, grabbed_at=now),
            now + 0.02,
            path=STRAIGHT,
            progress=progress,
            x=0.0,
            z=0.0,
            course=0.0,
            speed=max(speed, 0.0),
        )
    gone = cruise.step(
        report(grabbed_at=now + 0.05),
        now + 0.07,
        path=STRAIGHT,
        progress=progress,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=1.4,
    )
    assert gone.lead is not None, "it is still there"
    assert gone.decision.target_speed == pytest.approx(0.0, abs=0.3)


def test_a_lead_lost_close_up_is_not_given_up_for_a_vehicle_far_beyond() -> None:
    """At a border a lorry's trailer 10 m ahead in the queue filled the
    windscreen and was lost; a car 70 m on became the lead in 0.3 s, and
    the truck drove into the trailer."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=3.3)
    for k in range(30):
        now = 100.0 + 0.05 * k
        cruise.step(
            report(car(10.0, 0.9, closing=-1.0), grabbed_at=now), now + 0.01, progress=0.0, **kwargs
        )
    for k in range(1, 40):
        now = 101.5 + 0.05 * k
        step = cruise.step(
            report(car(70.0, 0.9, closing=-1.0, track_id=2), grabbed_at=now),
            now + 0.01,
            progress=0.0,
            **kwargs,
        )
        assert step.lead is not None and step.lead.distance_m < 15.0, f"{0.05 * k:.2f} s on"
        assert step.decision.target_speed is not None and step.decision.target_speed < 3.3


def test_a_lead_lost_close_up_is_let_go_after_a_while() -> None:
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=3.0)
    for k in range(30):
        now = 100.0 + 0.05 * k
        cruise.step(
            report(car(12.0, 0.9, closing=-1.0), grabbed_at=now), now + 0.01, progress=0.0, **kwargs
        )
    later = 101.5 + CLOSE_HOLD_S + 0.2
    cruise.step(report(grabbed_at=later), later + 0.01, progress=0.0, **kwargs)
    past = cruise.step(report(grabbed_at=later + 1.0), later + 1.01, progress=40.0, **kwargs)
    assert past.lead is None and past.decision.target_speed is None


def test_a_vehicle_in_the_way_with_its_wheels_under_the_dashboard_is_stopped_for() -> None:
    """A lorry turning in front at night was seen as its wheels went under the
    dashboard 10 m off, and then with no distance; it was hit."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=4.0)
    ahead = 960.0 + 820.0 * 0.86 / 8.0
    seen = PerceptionReport(100.0, 1, (), 20.0, unseen=((7, ahead),))
    step = cruise.step(seen, 100.01, **kwargs)
    assert step.decision.target_speed is not None and step.decision.target_speed < 4.0
    assert step.lead_id == 7
    beside = PerceptionReport(100.05, 2, (), 20.0, unseen=((8, 300.0),))
    clear = AdaptiveCruise(CAMERA).step(beside, 100.06, **kwargs)
    assert clear.decision.target_speed is None, "one beside the truck's way"


def test_a_vehicle_seen_close_ahead_but_no_longer_measured_is_stopped_for() -> None:
    """At a border a trailer 10 m ahead in the queue was still seen, its
    wheels under the dashboard, but no longer measured: it was hit."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=3.0)
    trailer = TrackReport(3, "truck", True, True, 10.0, 0.0, 0.86, 2.5, measurement_age=0.5)
    step = cruise.step(report(trailer, grabbed_at=100.0), 100.01, **kwargs)
    assert step.decision.target_speed is not None and step.decision.target_speed < 3.0


def test_a_lead_lost_close_up_and_let_go_is_crept_past() -> None:
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=3.0)
    for k in range(30):
        now = 100.0 + 0.05 * k
        cruise.step(report(car(12.0, 0.9, closing=-1.0), grabbed_at=now), now + 0.01, **kwargs)
    later = 101.5 + CLOSE_HOLD_S + 0.2
    step = cruise.step(report(grabbed_at=later), later + 0.01, **kwargs)
    assert step.decision.target_speed == pytest.approx(CREEP_PAST_MPS)


def test_a_lead_is_kept_where_it_is_when_the_path_changes_under_the_truck() -> None:
    """Taken to a pump, the route new, a lead 50 m ahead was taken for one
    18 km off, and lost."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, x=0.0, z=0.0, course=0.0, speed=20.0)
    for k in range(20):
        now = 100.0 + 0.05 * k
        cruise.step(
            report(car(50.0, 0.9, closing=0.0), grabbed_at=now), now + 0.01, progress=0.0, **kwargs
        )
    cruise.rebase(-18000.0)
    step = cruise.step(report(grabbed_at=101.1), 101.11, progress=-18000.0, **kwargs)
    assert step.lead is not None and 45.0 < step.lead.distance_m < 80.0


def test_every_creep_is_enough_to_move_off_from_a_stop() -> None:
    """At the Chiasso customs the truck stopped behind a car, lost it close
    up, and was to creep on at 1.4 m/s; a stop is let go of at 1.5, and it
    stood there until the driver took it."""
    from core.control import acc
    from core.control.parking import RESUME_TARGET_MPS

    for name in ("CREEP_PAST_MPS", "RING_ENTRY_MPS"):
        assert getattr(acc, name) >= RESUME_TARGET_MPS, name


def test_a_lead_forgotten_no_longer_holds_the_truck() -> None:
    cruise = AdaptiveCruise(CAMERA)
    cruise._remembered = (10.0, 50.0)
    cruise._last_lead = (7, 60.0, 0.0, 99.0)
    cruise.forget_lead()
    step = cruise.step(
        report(grabbed_at=100.0),
        100.05,
        path=STRAIGHT,
        progress=50.0,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=0.0,
    )
    assert step.decision.target_speed is None, step.decision.reason


def test_at_a_green_a_lead_remembered_and_never_seen_again_is_let_go() -> None:
    """Something read for 2 s, 9.5 m ahead, across an empty road at night was
    remembered, and the truck stood at the green for good."""
    from core.control.acc import CREEP_PAST_MPS, GREEN_FOR_S, REMEMBER_AT_MOST_S

    cruise = AdaptiveCruise(CAMERA)
    cruise._remembered, cruise._remembered_at = (9.5, 50.0), 100.0
    kwargs = dict(path=STRAIGHT, progress=50.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    t = 100.0 + REMEMBER_AT_MOST_S + 1.0
    no_green = cruise.step(report(grabbed_at=t), t + 0.05, **kwargs)
    assert no_green.decision.target_speed == pytest.approx(0.0), "with no green it stays"
    cruise.green_since = t
    just_green = cruise.step(report(grabbed_at=t + 1.0), t + 1.05, **kwargs)
    assert just_green.decision.target_speed == pytest.approx(0.0), "it has not had time to go"
    later = t + GREEN_FOR_S + 1.0
    let_go = cruise.step(report(grabbed_at=later), later + 0.05, **kwargs)
    assert let_go.decision.target_speed == pytest.approx(CREEP_PAST_MPS), let_go.decision.reason


def test_a_lead_remembered_where_its_wheels_would_be_in_view_has_gone() -> None:
    """The car the truck stopped behind at a roundabout's entry, 11.6 m off,
    went on round the ring; remembered, it held the truck there for good."""
    from core.control.acc import CREEP_PAST_MPS, IN_VIEW_GONE_S

    cruise = AdaptiveCruise(CAMERA)
    cruise._remembered, cruise._remembered_at = (11.6, 50.0), 100.0
    kwargs = dict(path=STRAIGHT, progress=50.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    soon = 100.0 + IN_VIEW_GONE_S - 1.0
    held = cruise.step(report(grabbed_at=soon), soon + 0.05, **kwargs)
    assert held.decision.target_speed == pytest.approx(0.0)
    late = 100.0 + IN_VIEW_GONE_S + 1.0
    gone = cruise.step(report(grabbed_at=late), late + 0.05, **kwargs)
    assert gone.decision.target_speed == pytest.approx(CREEP_PAST_MPS), gone.decision.reason


def test_a_lead_remembered_too_close_for_its_wheels_to_show_is_kept_with_no_green() -> None:
    from core.control.acc import IN_VIEW_GONE_S

    cruise = AdaptiveCruise(CAMERA)
    cruise._remembered, cruise._remembered_at = (9.0, 50.0), 100.0
    kwargs = dict(path=STRAIGHT, progress=50.0, x=0.0, z=0.0, course=0.0, speed=0.0)
    late = 100.0 + 10 * IN_VIEW_GONE_S
    kept = cruise.step(report(grabbed_at=late), late + 0.05, **kwargs)
    assert kept.decision.target_speed == pytest.approx(0.0), kept.decision.reason


def test_through_a_junction_a_car_coming_the_other_way_turning_across_is_followed() -> None:
    """England, through on the amber: a car coming the other way turned across
    the truck from 10 m to the right and was hit at 31 km/h."""
    from core.control.acc import PREFAB_AHEAD_M

    def run(prefabs):
        cruise = AdaptiveCruise(CAMERA)
        cruise.prefabs = prefabs
        step = None
        for k in range(16):
            t = 100.0 + k / 20
            car_across = TrackReport(
                3, "car", True, True, 20.0 - 0.6 * k, -13.0, 8.0 - 0.25 * k, 2.6
            )
            step = cruise.step(
                report(car_across, grabbed_at=t),
                t + 0.05,
                path=STRAIGHT,
                progress=0.0,
                x=0.0,
                z=0.0,
                course=0.0,
                speed=8.6,
            )
        return step

    assert run([(PREFAB_AHEAD_M - 5.0, 80.0)]).lead_id == 3
    assert run([]).lead_id is None, "on the open road, one coming the other way is not"


def lorry_pulling_out(speed: float, lateral_at, junctions=((0.0, 100.0),)):
    """A lorry seen at an angle, 5 m wide, standing in the truck's direction
    31 m ahead, coming across from the left at `lateral_at(seconds)`."""
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = list(junctions)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=speed)
    steps = []
    for k in range(12):
        t = 100.0 + k / 20
        lorry = TrackReport(
            5, "truck", True, True, 31.0 - speed * k / 20, -speed, lateral_at(k, k / 20), 5.0
        )
        steps.append(cruise.step(report(lorry, grabbed_at=t), t + 0.05, **kwargs))
    return steps


def test_at_speed_a_lorry_pulling_out_across_the_way_is_given_way_to() -> None:
    """By Metz, at 78 km/h: a lorry out of a side road came from 10.6 m to
    3.6 m off in 0.7 s, every sighting nearer. Seen at an angle, not wide
    enough to show its side, and too fast for any crossing to be given way
    to, it was hit."""
    steps = lorry_pulling_out(21.6, lambda k, t: -10.6 + 9.0 * t)
    assert any("giving way" in step.decision.reason for step in steps)
    assert steps[-1].decision.urgent_deceleration


def test_at_speed_a_box_jumping_towards_the_lane_is_not_given_way_to() -> None:
    """England: coaches beyond the median read as crossing at 80 km/h. A box
    that jumps about is no vehicle closing in; slower, it is still given way to."""
    jumpy = lambda k, t: -12.0 + 8.0 * t - (1.0 if k % 3 == 0 else 0.0)  # noqa: E731
    fast = lorry_pulling_out(21.6, jumpy)
    assert all("giving way" not in step.decision.reason for step in fast)
    slow = lorry_pulling_out(8.0, jumpy)
    assert any("giving way" in step.decision.reason for step in slow)


def test_a_junction_crossed_on_the_level_is_slowed_for() -> None:
    from core.control.acc import crossing_spans
    from core.planning.route import ONWARD, Route

    net = junction(crossed=True)
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    ((start, _),) = crossing_spans(route, path)
    assert start == pytest.approx(300.0, abs=2.0)

    joined = junction(crossed=False)
    route = Route(joined, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    assert crossing_spans(route, path) == [], "a slip road meets its lanes end to end"


def test_a_way_going_over_the_junction_is_not_a_crossing() -> None:
    from core.control.acc import crossing_spans
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning.route import ONWARD, Route
    from tests.unit.lane_networks import polyline

    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(Lane(points=polyline((0, 300), (0, 360)), source="prefab", owner="2"))
    bridge = polyline((-30, 330), (30, 330))
    bridge[:, 1] = 7.0
    graph.lanes.append(Lane(points=bridge, source="prefab", owner="2"))
    graph.connect(0, 1)
    net = RoadNetwork.compile(graph)
    route = Route(net, (0, 1), (ONWARD,))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=2.0)
    assert crossing_spans(route, path) == []


def test_through_a_junction_crossed_on_the_level_no_faster_than_60() -> None:
    """The driver's limit (11 November), slowed to in good time."""
    from core.control.acc import JUNCTION_BEFORE_M, JUNCTION_THROUGH_MPS

    cruise = AdaptiveCruise(CAMERA)
    cruise.crossings = [(300.0, 330.0)]

    def target(progress: float) -> float | None:
        kwargs = dict(path=STRAIGHT, progress=progress, x=0.0, z=progress, course=0.0, speed=25.0)
        return cruise.step(report(grabbed_at=100.0), 100.05, **kwargs).decision.target_speed

    assert target(0.0) is None or target(0.0) >= 25.0, "far off, the road's own speed"
    assert target(150.0) < 25.0
    for progress in (300.0 - JUNCTION_BEFORE_M, 300.0, 330.0):
        assert target(progress) == pytest.approx(JUNCTION_THROUGH_MPS)
    assert target(340.0) is None or target(340.0) > JUNCTION_THROUGH_MPS


def test_short_of_a_tight_turn_the_truck_waits_while_a_vehicle_is_in_it() -> None:
    """On the way to Bergen the truck followed a tanker into a turn as tight
    as a hairpin and stood in it; turning round, the tanker's trailer hit it."""
    from core.control.acc import STOP_BEFORE_TURN_M, TIGHT_WAIT_AT_MOST_S, TIGHT_WHY

    turn = np.column_stack([np.zeros(20), 100.0 + np.arange(20.0)])
    cruise = AdaptiveCruise(CAMERA)
    cruise.tight_turns = [(100.0, 120.0, turn)]

    def step(progress: float, speed: float, t: float, *tracks):
        kwargs = dict(path=STRAIGHT, progress=progress, x=0.0, z=progress, course=0.0, speed=speed)
        return cruise.step(report(*tracks, grabbed_at=t), t + 0.05, **kwargs)

    in_it = TrackReport(8, "truck", True, True, 25.0, -5.0, -3.0, 2.5)  # beside the line
    waiting = step(80.0, 8.0, 100.0, in_it)
    assert cruise._give_way is not None and cruise._give_way.why == TIGHT_WHY
    assert waiting.decision.target_speed < 8.0
    clear = AdaptiveCruise(CAMERA)
    clear.tight_turns = [(100.0, 120.0, turn)]
    kwargs = dict(path=STRAIGHT, progress=80.0, x=0.0, z=80.0, course=0.0, speed=8.0)
    assert TIGHT_WHY not in clear.step(report(grabbed_at=100.0), 100.05, **kwargs).decision.reason
    # Standing short of it long enough, it goes on: the vehicle may be waiting for it.
    stop = 100.0 - STOP_BEFORE_TURN_M
    t = 101.0
    while t < 101.0 + TIGHT_WAIT_AT_MOST_S + 2.0:
        last = step(stop, 0.0, t, dataclasses.replace(in_it, distance=100.0 - stop + 5.0))
        t += 0.5
    assert TIGHT_WHY not in last.decision.reason


def test_a_turn_too_tight_for_the_trailer_is_found_along_the_path() -> None:
    from core.map import Lane, LaneGraph, RoadNetwork
    from core.planning.route import ONWARD, Route
    from core.planning.tight_turns import tight_turn_spans
    from tests.unit.lane_networks import polyline

    angles = np.radians(np.linspace(180.0, 90.0, 30))
    arc = np.column_stack([-6.0 + 6.0 * np.cos(angles), np.zeros(30), 300.0 + 6.0 * np.sin(angles)])
    arc[0] = (0.0, 0.0, 300.0)
    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 300)), source="road", owner="1"))
    graph.lanes.append(Lane(points=arc, source="prefab", owner="2"))
    graph.lanes.append(Lane(points=polyline((-6, 306), (-200, 306)), source="road", owner="3"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    net = RoadNetwork.compile(graph)
    route = Route(net, (0, 1, 2), (ONWARD, ONWARD))
    path = ReferencePath(route.points()[:, [0, 2]], spacing=1.0)
    ((start, end, points),) = tight_turn_spans(route, path)
    assert start == pytest.approx(300.0, abs=2.0)
    assert end > start
    assert len(points) >= 2


def test_through_a_tight_turn_no_slower_than_its_bend_takes() -> None:
    """The driver's word (16 November): 8 km/h round tight turns, slowed to
    from 1 km off, looked bad; the bend's own speed is the speed profile's."""
    cruise = AdaptiveCruise(CAMERA)
    cruise.tight_turns = [(300.0, 320.0, np.column_stack([np.zeros(20), 300.0 + np.arange(20.0)]))]

    def target(progress: float) -> float | None:
        kwargs = dict(path=STRAIGHT, progress=progress, x=0.0, z=progress, course=0.0, speed=20.0)
        return cruise.step(report(grabbed_at=100.0), 100.05, **kwargs).decision.target_speed

    for progress in (0.0, 200.0, 300.0, 310.0, 330.0):
        assert target(progress) is None


def lorry_from_its_corner(lateral_at, width: float = 12.0):
    """A lorry seen from its corner, `width` wide, 30 m ahead and going the
    truck's way 3.5 m/s slower, at `lateral_at(seconds)` -- right of the seat."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=13.3)
    steps = []
    for k in range(30):
        t = 100.0 + k / 20
        lorry = TrackReport(
            3, "truck", True, True, 30.0 - 3.5 * k / 20, -3.5, lateral_at(t - 100.0), width
        )
        steps.append(cruise.step(report(lorry, grabbed_at=t), t + 0.05, **kwargs))
    return steps


def test_a_lorry_pulling_into_the_lane_seen_from_its_corner_is_followed() -> None:
    """In Norway a lorry pulled out of a side road into the truck's lane 30 m
    ahead, seen 9 to 15 m wide: too wide for a vehicle ahead, and joining the
    traffic, not crossing it. Not followed, it was hit from behind at 41 km/h."""
    steps = lorry_from_its_corner(lambda t: 0.86 + 5.0 - 2.0 * min(t, 1.2))
    assert steps[-1].lead_id == 3
    assert steps[-1].decision.target_speed < 13.3


def test_a_wide_lorry_keeping_to_the_next_lane_is_not_followed() -> None:
    steps = lorry_from_its_corner(lambda t: 0.86 + 9.0, width=8.0)
    assert all(step.lead_id is None for step in steps)


def test_a_vehicle_going_by_beside_the_cab_is_followed_only_once_in_front() -> None:
    """Coaches, tankers and caravans overtaking in the next lane, first seen
    beside the cab's pillar at 10 m, were taken for vehicles ahead about 11
    times an hour, and braked the truck at speed for a moment each."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=22.0)

    def step(t: float, lateral: float):
        coach = TrackReport(9, "bus", True, True, 10.0, 0.0, 0.86 + lateral, 2.5)
        return cruise.step(report(coach, grabbed_at=t), t + 0.05, **kwargs)

    step(100.0, -3.2)  # first seen beside the cab
    assert step(100.1, -1.8).lead_id is None, "going by, its box towards the lane"
    assert step(100.2, -0.4).lead_id == 9, "in front, in the lane: followed"


def test_a_vehicle_first_seen_far_ahead_beside_the_lane_is_followed_as_it_comes_in() -> None:
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=22.0)
    car = TrackReport(4, "car", True, True, 40.0, -3.0, 0.86 - 3.0, 1.9)
    cruise.step(report(car, grabbed_at=100.0), 100.05, **kwargs)
    nearer = dataclasses.replace(car, distance=35.0, lateral=0.86 - 1.8)
    assert cruise.step(report(nearer, grabbed_at=100.1), 100.15, **kwargs).lead_id == 4


def test_a_car_being_passed_in_the_next_lane_is_not_followed_beside_the_cab() -> None:
    """Out of Dover, at 89 km/h, a car in the next lane being overtaken was
    taken, at 11 m, for one ahead: where it meets the road beside the cab is
    hard to see. In the next lane from afar, it is not followed close by."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=24.7)
    t = 100.0
    for distance in (45.0, 40.0, 35.0, 30.0, 26.0):
        car = TrackReport(5, "car", True, True, distance, -5.0, 0.86 - 3.5, 1.9)
        cruise.step(report(car, grabbed_at=t), t + 0.05, **kwargs)
        t += 0.2
    close = TrackReport(5, "car", True, True, 11.0, -5.0, 0.86 - 1.8, 1.9)
    assert cruise.step(report(close, grabbed_at=t), t + 0.05, **kwargs).lead_id is None


def test_a_lorry_moving_in_from_the_next_lane_afar_is_still_followed_close_by() -> None:
    """In Norway a lorry pulling out of a side road came from 5.8 m to 3.2 m
    off while 36 to 21 m ahead."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=13.3)
    t = 100.0
    for distance, lateral in ((36.0, 5.8), (33.0, 5.2), (30.0, 4.6), (27.0, 3.9), (25.0, 3.4)):
        lorry = TrackReport(1, "truck", True, True, distance, -3.8, 0.86 + lateral, 2.5)
        cruise.step(report(lorry, grabbed_at=t), t + 0.05, **kwargs)
        t += 0.3
    nearer = TrackReport(1, "truck", True, True, 21.0, -3.8, 0.86 + 1.6, 2.5)
    assert cruise.step(report(nearer, grabbed_at=t), t + 0.05, **kwargs).lead_id == 1


def test_a_car_in_the_lane_on_the_open_road_is_never_let_go_as_oncoming() -> None:
    """Duisburg, with another truck's camera: a car stopping at a red light
    seemed to come at 17 km/h, was let go as oncoming, and was braked for at
    8.8 m/s^2 from 30 m. In the lane on the open road it is followed as
    standing at worst, and slowed for from afar."""
    cruise = AdaptiveCruise(CAMERA)
    speed = 8.6
    step = cruise.step(
        report(car(60.0, 0.86, closing=-(speed + 4.7))),
        100.05,
        path=STRAIGHT,
        progress=0.0,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=speed,
    )
    assert "oncoming" not in step.decision.reason
    assert step.decision.target_speed is not None and step.decision.target_speed < speed
    assert step.lead is not None and step.lead.closing_mps == pytest.approx(-speed)


def test_through_a_junction_a_car_coming_the_other_way_is_still_not_followed() -> None:
    cruise = AdaptiveCruise(CAMERA)
    cruise.junctions = [(40.0, 80.0)]
    speed = 8.6
    step = cruise.step(
        report(car(60.0, 0.86, closing=-(speed + 8.0))),
        100.05,
        path=STRAIGHT,
        progress=0.0,
        x=0.0,
        z=0.0,
        course=0.0,
        speed=speed,
    )
    assert step.lead is None or step.lead.closing_mps < -speed


def test_a_vehicle_seen_in_a_tight_turn_too_late_to_stop_short_of_it_is_not_waited_for() -> None:
    """Sheffield: at 10 km/h, a car seen in the turn 1 m short of where to
    wait, braked for at 7 m/s^2. Seen in good time, it is waited for."""
    from core.control.acc import STOP_BEFORE_TURN_M, TIGHT_WHY

    turn = np.column_stack([np.zeros(20), 100.0 + np.arange(20.0)])
    in_it = TrackReport(8, "car", True, True, 13.0, -3.0, -3.0, 1.8)

    def first_look(progress: float, speed: float):
        cruise = AdaptiveCruise(CAMERA)
        cruise.tight_turns = [(100.0, 120.0, turn)]
        seen = dataclasses.replace(in_it, distance=100.0 - progress + 3.0)
        kwargs = dict(path=STRAIGHT, progress=progress, x=0.0, z=progress, course=0.0, speed=speed)
        return cruise.step(report(seen, grabbed_at=100.0), 100.05, **kwargs).decision

    late = first_look(100.0 - STOP_BEFORE_TURN_M - 1.0, 3.0)
    assert TIGHT_WHY not in late.reason
    assert late.urgent_deceleration is None
    early = first_look(70.0, 3.0)
    assert early.reason.startswith(TIGHT_WHY)


def test_a_lead_gone_round_a_bend_out_of_view_is_not_stopped_for() -> None:
    """Leipzig: round a tight right turn a lorry ahead at 7 km/h went out of
    view, was remembered standing 14 m on, and the truck stopped in the turn
    for it twice. On the straight a lead lost close up still is."""
    bend = np.vstack(
        [
            np.column_stack([np.zeros(20), np.arange(20.0)]),
            np.column_stack([np.arange(1.0, 40.0), np.full(39, 19.0)]),
        ]
    )
    bent = ReferencePath(bend, spacing=1.0)

    def lost(path) -> str:
        cruise = AdaptiveCruise(CAMERA)
        kwargs = dict(path=path, progress=5.0, x=0.0, z=5.0, course=0.0, speed=2.0)
        for k in range(40):
            now = 100.0 + 0.05 * k
            cruise.step(report(car(13.0, 0.9, closing=-1.0), grabbed_at=now), now + 0.01, **kwargs)
        later = 102.0 + 0.5
        return cruise.step(report(grabbed_at=later), later + 0.01, **kwargs).decision.reason

    assert "out of view" in lost(STRAIGHT)
    assert "out of view" not in lost(bent)


def test_a_car_coming_fast_the_other_way_round_a_bend_is_not_stopped_for() -> None:
    """Near Salzburg (18 November) a car in the oncoming lane came from 115 m
    to 13 m in a second where a left-hand bend put the truck's lane; taken
    for one standing in it, the truck braked at 8.6 m/s^2 to a stop."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=5.6)
    for k in range(10):
        now = 100.0 + 0.05 * k
        coming = car(18.0 - 0.9 * k, 0.9, closing=-5.6 - 14.0, track_id=131)
        step = cruise.step(report(coming, grabbed_at=now), now + 0.01, **kwargs)
        assert step.decision.urgent_deceleration is None, step.decision.reason
        assert step.decision.target_speed is None or step.decision.target_speed >= 5.0
    # Gone by out of view: not remembered standing there.
    later = 100.6
    gone = cruise.step(report(grabbed_at=later), later + 0.01, **kwargs)
    assert "out of view" not in gone.decision.reason


def test_a_car_seeming_to_come_slowly_in_the_lane_is_still_taken_to_stand() -> None:
    """On another truck's camera cars standing at a red seemed to come at up
    to 17 km/h (Duisburg): still stopped for, not let go as oncoming."""
    cruise = AdaptiveCruise(CAMERA)
    kwargs = dict(path=STRAIGHT, progress=0.0, x=0.0, z=0.0, course=0.0, speed=6.0)
    for k in range(10):
        now = 100.0 + 0.05 * k
        step = cruise.step(
            report(car(25.0, 0.9, closing=-6.0 - 4.7), grabbed_at=now), now + 0.01, **kwargs
        )
    assert step.decision.target_speed is not None and step.decision.target_speed < 6.0
