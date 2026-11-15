"""Following vehicles from frame to frame, and picking the one to follow."""

from __future__ import annotations

import numpy as np
import pytest

from core.perception.camera import GroundCamera
from core.perception.detection import Detection, Kind
from core.perception.tracking import (
    CONFIRM_HITS,
    CutInFilter,
    Path,
    Tracker,
    lead_vehicle,
    plausible,
)

CAMERA = GroundCamera(focal_px=820.0, horizon_row=438.0, centre_col=960.0, height_m=2.9)


def seen(distance: float, lateral: float, width: float = 2.5, kind: Kind = Kind.TRUCK) -> Detection:
    """The box a vehicle `width` wide and 3 m tall makes at a place on the road."""
    scale = CAMERA.focal_px / distance
    bottom = CAMERA.horizon_row + CAMERA.focal_px * CAMERA.height_m / distance
    centre = CAMERA.centre_col + lateral * scale
    box = (centre - width / 2 * scale, bottom - 3.0 * scale, centre + width / 2 * scale, bottom)
    return Detection(kind, 0.9, box)


def drive(tracker: Tracker, frames: int, rate: float = 20.0, **scene) -> list:
    """Feed frames of vehicles at distance(t); `scene` maps a name to (d(t), lateral, width)."""
    for k in range(frames):
        t = k / rate
        tracker.update([seen(d(t), lat, w) for d, lat, w in scene.values()], t)
    return tracker.tracks


def test_one_vehicle_stays_one_track_and_its_closing_speed_is_measured() -> None:
    tracker = Tracker(CAMERA)
    tracks = drive(tracker, 30, lead=(lambda t: 60.0 - 5.0 * t, 0.0, 2.5))
    assert len(tracks) == 1
    distance, closing, lateral, width = tracks[0].estimate()
    # Now, not half a second ago: a car closing at 5 m/s is 2.5 m nearer.
    assert distance == pytest.approx(60.0 - 5.0 * 29 / 20, abs=0.3)
    assert closing == pytest.approx(-5.0, rel=0.05)
    assert (lateral, width) == pytest.approx((0.0, 2.5), abs=0.05)


def test_a_one_frame_detection_is_never_the_lead() -> None:
    tracker = Tracker(CAMERA)
    tracker.update([seen(40.0, 0.0)], 0.0)
    assert lead_vehicle(tracker.tracks) is None
    for k in range(1, CONFIRM_HITS):
        tracker.update([seen(40.0, 0.0)], k * 0.05)
    assert lead_vehicle(tracker.tracks) is not None


def test_the_lead_is_the_nearest_in_the_lane_not_the_nearest_overall() -> None:
    tracker = Tracker(CAMERA)
    drive(
        tracker,
        10,
        beside=(lambda t: 20.0, -4.5, 2.5),
        far_ahead=(lambda t: 70.0, 0.3, 2.5),
        near_ahead=(lambda t: 35.0, -0.4, 1.9),
    )
    _, (distance, *_) = lead_vehicle(tracker.tracks)
    assert distance == pytest.approx(35.0, abs=0.1)


def test_a_portal_the_detector_calls_a_truck_is_not_a_vehicle() -> None:
    """What happened: a tunnel portal 150 m ahead, detected as a truck 38 m wide."""
    tracker = Tracker(CAMERA)
    drive(tracker, 10, portal=(lambda t: 150.0, 0.0, 38.0))
    assert lead_vehicle(tracker.tracks) is None
    assert not plausible(tracker.tracks[0])


def test_a_track_unseen_for_a_while_is_forgotten() -> None:
    tracker = Tracker(CAMERA)
    drive(tracker, 5, lead=(lambda t: 40.0, 0.0, 2.5))
    tracker.update([], 0.3)
    assert len(tracker.tracks) == 1, "a missed frame or two is not the end"
    tracker.update([], 0.9)
    assert tracker.tracks == []


def test_a_pitching_cab_does_not_make_a_far_car_drive_backwards() -> None:
    """What happened: 168 m ahead, box edges jumping a pixel or two with the cab
    made a car in front of a truck at 90 km/h go backwards at 152 km/h."""
    rng = np.random.default_rng(0)
    tracker = Tracker(CAMERA)
    for k in range(20):
        t = k / 20
        found = seen(150.0 - 3.0 * t, 0.0)
        left, top, right, bottom = found.box
        pitch = rng.normal(0.0, 1.5)  # the whole box moves up and down
        tracker.update([Detection(found.kind, 0.9, (left, top + pitch, right, bottom + pitch))], t)
    distance, closing, *_ = tracker.tracks[0].estimate()
    assert distance == pytest.approx(148.0, rel=0.1)
    # Never a phantom: at worst the approach is under-read, 150 m out.
    assert -4.5 <= closing <= 0.0


def test_where_the_road_bends_the_lead_is_on_the_lane_not_straight_ahead() -> None:
    """What happened: at a motorway split the gore sign stood straight ahead
    and became a stopped car in front, while the lane bent away from it."""
    from core.perception.tracking import Path

    tracker = Tracker(CAMERA)
    drive(
        tracker,
        10,
        gore_sign=(lambda t: 40.0, 0.0, 1.6),
        car_on_the_lane=(lambda t: 60.0, 6.0, 1.8),
    )
    assert lead_vehicle(tracker.tracks)[1][0] == pytest.approx(40.0, abs=0.5)
    bend = Path(np.array([1.0, 30.0, 60.0, 90.0]), np.array([0.0, 1.5, 6.0, 13.5]))
    assert lead_vehicle(tracker.tracks, bend)[1][0] == pytest.approx(60.0, abs=0.5)


def test_a_path_from_the_map_is_seen_from_the_drivers_seat() -> None:
    from core.perception.tracking import Path

    seated = GroundCamera(820.0, 438.0, 960.0, 2.9, seat_right_m=-0.86)
    path = Path.from_truck_frame([-5.0, 0.0, 10.0, 50.0], [0.0, 0.0, 0.0, 0.0], seated)
    assert path.right_at(30.0) == pytest.approx(0.86), "the lane's middle is right of the driver"
    assert path.ahead[0] > 0, "nothing behind the eye"


def test_nothing_far_down_the_road_is_the_lead() -> None:
    tracker = Tracker(CAMERA)
    drive(tracker, 10, far=(lambda t: 160.0, 0.0, 2.5))
    assert lead_vehicle(tracker.tracks) is None


def test_a_car_coming_out_from_behind_another_is_not_closing() -> None:
    """What happened: a box widening from 49 to 91 px in 0.3 s at a steady
    22.7 m -- more of the car coming into view -- read as closing at 46 m/s."""
    tracker = Tracker(CAMERA)
    for k in range(12):
        full = seen(22.7, 0.0)
        left, top, right, bottom = full.box
        revealed = min(1.0, 0.4 + 0.1 * k)
        box = (right - (right - left) * revealed, top, right, bottom)
        tracker.update([Detection(Kind.CAR, 0.9, box)], k / 20)
    _, closing, *_ = tracker.tracks[0].estimate()
    assert abs(closing) < 1.0


def test_a_young_track_reports_no_closing_yet() -> None:
    tracker = Tracker(CAMERA)
    for k in range(16):
        t = k / 20
        tracker.update([seen(40.0 - 20.0 * t, 0.0)], t)
        if k == 3:
            assert tracker.tracks[0].estimate()[1] == 0.0
    assert tracker.tracks[0].estimate()[1] == pytest.approx(-20.0, rel=0.1)


def test_a_car_crossing_in_front_is_not_the_lead() -> None:
    """What happened: cars crossing a junction in front, read as leads, made
    three of the four hard brakes a city driver never made."""
    from core.perception.tracking import CrossingFilter

    tracker, crossing = Tracker(CAMERA), CrossingFilter()
    for k in range(12):
        t = k / 20
        tracker.update([seen(30.0, -1.5 + 6.0 * t, width=4.5, kind=Kind.CAR)], t)
        excluded = crossing.crossing(tracker.tracks, None, t)
    assert excluded == {tracker.tracks[0].id}
    assert lead_vehicle(tracker.tracks, None, excluded) is None


def test_in_a_bend_a_lead_sweeping_across_the_image_is_not_crossing() -> None:
    """The truck turning moves the lead and the lane alike across the image."""
    from core.perception.tracking import CrossingFilter, Path

    tracker, crossing = Tracker(CAMERA), CrossingFilter()
    for k in range(12):
        t = k / 20
        sweep = 4.0 * t  # the whole scene, as the truck yaws
        tracker.update([seen(60.0, 1.0 + sweep)], t)
        lane = Path(np.array([1.0, 30.0, 60.0, 90.0]), np.array([0.0, 0.5, 1.0, 1.5]) + sweep)
        excluded = crossing.crossing(tracker.tracks, lane, t)
    assert excluded == frozenset()
    assert lead_vehicle(tracker.tracks, lane, excluded) is not None


def test_a_box_cut_off_at_the_bottom_gives_no_distance() -> None:
    tracker = Tracker(CAMERA)
    near = seen(12.0, 0.0)
    left, top, right, _ = near.box
    cut = Detection(Kind.CAR, 0.9, (left - 40, top, right + 40, 695.0))
    for k in range(8):
        tracker.update([near if k < 4 else cut], k / 20, cut_row=700.0)
    assert tracker.tracks[0].hits == 8
    assert len(tracker.tracks[0].sightings) == 4, "the cut boxes measured nothing"


def test_a_track_no_longer_measured_is_not_followed() -> None:
    """Its foot behind the dashboard, a crossing car kept its last distance."""
    tracker = Tracker(CAMERA)
    near = seen(14.0, 0.0)
    left, top, right, _ = near.box
    hidden = Detection(Kind.CAR, 0.9, (left, top, right, 695.0))
    for k in range(12):
        tracker.update([near if k < 5 else hidden], k / 20, cut_row=700.0)
    assert tracker.tracks[0].measurement_age == pytest.approx(0.35)
    assert lead_vehicle(tracker.tracks) is None


def test_far_away_nothing_is_judged_to_be_crossing() -> None:
    from core.perception.tracking import CrossingFilter

    tracker, crossing = Tracker(CAMERA), CrossingFilter()
    for k in range(12):
        t = k / 20
        tracker.update([seen(80.0, -1.5 + 6.0 * t)], t)
        excluded = crossing.crossing(tracker.tracks, None, t)
    assert excluded == frozenset()


def crossing_scene(
    distance: float,
    lateral_at,
    width: float,
    frames: int = 12,
    closing: float = 0.0,
    kind: Kind = Kind.TRUCK,
):
    """A vehicle `distance` ahead, as the truck closes on it at `closing`
    m/s (negative): minus the truck's speed for one moving across, not along."""
    from core.perception.tracking import CrossingFilter

    tracker, crossing = Tracker(CAMERA), CrossingFilter()
    for k in range(frames):
        t = k / 20
        tracker.update([seen(distance + closing * t, lateral_at(t), width, kind)], t)
        crossing.crossing(tracker.tracks, None, t)
    return tracker, crossing


def test_a_lorry_turning_across_the_lane_ahead_is_given_way_to() -> None:
    """Dover: a lorry turning across a junction was the lead, then found
    crossing and dropped, and the truck drove on into it."""
    tracker, crossing = crossing_scene(20.0, lambda t: 6.0 - 2.0 * t, width=12.0, closing=-8.0)
    found = crossing.in_the_way(speed=8.0)
    assert found is not None
    assert found == (tracker.tracks[0].id, pytest.approx(20.0 - 8.0 * 0.55, abs=1.5))


def test_a_car_that_will_be_across_before_the_truck_gets_there_is_not() -> None:
    """Crossing 30 m ahead at 6 m/s: gone a second before the truck arrives."""
    _, crossing = crossing_scene(
        30.0, lambda t: -1.5 + 6.0 * t, width=4.5, closing=-10.0, kind=Kind.CAR
    )
    assert crossing.in_the_way(speed=10.0) is None


def test_a_car_coming_down_a_slip_road_as_fast_as_the_truck_is_not_in_its_way() -> None:
    """Out of Folkestone: cars joining the main road from a slip road moved
    across the lane and were given way to as if standing still -- 7 m/s^2 of
    braking, three times. It shows its back: it is going along the road."""
    tracker, crossing = crossing_scene(22.0, lambda t: 4.5 - 1.8 * t, width=1.9, kind=Kind.CAR)
    assert crossing.crossing(tracker.tracks, None, 0.55) == frozenset()
    assert crossing.in_the_way(speed=20.0) is None


def test_a_vehicle_going_the_same_way_in_the_next_lane_is_not_crossing() -> None:
    """The A38 into Plymouth: a car in the next lane, swept across by a
    bend and read as closing at the truck's own speed, was given way to."""
    tracker, crossing = crossing_scene(
        28.0, lambda t: 3.5 - 2.0 * t, width=2.1, closing=-22.8, kind=Kind.CAR
    )
    assert crossing.crossing(tracker.tracks, None, 0.55) == frozenset()
    assert crossing.in_the_way(speed=22.8) is None


def test_a_broadside_car_moving_along_as_fast_as_the_truck_is_joining() -> None:
    tracker, crossing = crossing_scene(22.0, lambda t: 4.5 - 1.8 * t, width=4.0, kind=Kind.CAR)
    assert tracker.tracks[0].id in crossing.crossing(tracker.tracks, None, 0.55)
    assert crossing.joining(speed=20.0) == {tracker.tracks[0].id}
    assert crossing.in_the_way(speed=20.0) is None


def test_a_lorry_coming_the_other_way_broadside_as_it_passes_is_not_crossing() -> None:
    """Past Folkestone: a lorry on the other carriageway, seen side-on as it
    passed 18 m ahead and sweeping across the view, was given way to at
    7 m/s^2. It came along the road towards the truck, at 23 m/s."""
    tracker, crossing = crossing_scene(
        30.0, lambda t: 9.0 - 3.0 * t, width=12.0, closing=-46.0, kind=Kind.TRUCK
    )
    assert tracker.tracks[0].id in crossing.joining(speed=23.0)
    assert crossing.in_the_way(speed=23.0) is None


def test_a_car_pulling_out_slowly_across_the_lane_is_given_way_to() -> None:
    """Out of a side road, broadside, moving along at 3 m/s while the truck does 20."""
    _, crossing = crossing_scene(
        22.0, lambda t: 4.5 - 1.8 * t, width=4.5, closing=-17.0, kind=Kind.CAR
    )
    assert crossing.in_the_way(speed=20.0) is not None


def test_a_car_already_past_the_lane_and_going_away_is_not() -> None:
    _, crossing = crossing_scene(15.0, lambda t: 5.0 + 3.0 * t, width=4.5)
    assert crossing.in_the_way(speed=0.0) is None


TORINO_CAMERA = GroundCamera(
    focal_px=819.0, horizon_row=438.3, centre_col=960.1, height_m=2.881, seat_right_m=-0.86
)
TORINO_CUT_IN = (
    (-3.98, (1018, 422, 1082, 483)),
    (-3.72, (1017, 421, 1085, 486)),
    (-3.50, (1014, 420, 1090, 489)),
    (-3.27, (1010, 417, 1097, 492)),
    (-3.04, (1006, 416, 1105, 496)),
    (-2.82, (1001, 414, 1116, 500)),
    (-2.58, (1000, 412, 1123, 504)),
    (-2.37, (994, 409, 1133, 509)),
    (-2.15, (990, 406, 1144, 515)),
    (-1.91, (982, 403, 1160, 524)),
    (-1.69, (975, 396, 1178, 534)),
    (-1.47, (969, 391, 1195, 543)),
    (-1.23, (959, 380, 1221, 558)),
    (-1.02, (950, 371, 1254, 574)),
)
"""The detector's boxes of the lorry that moved across from a slip road out of
Torino into the truck's lane, seconds before it was hit at 80 km/h (from the
black box of drive-08, 2026-10-22)."""


def straight_lane(camera: GroundCamera) -> Path:
    ahead = np.arange(2.0, 150.0, 2.0)
    return Path.from_truck_frame(ahead, np.zeros_like(ahead), camera)


def leads_through(
    boxes, camera: GroundCamera, *, cut_ins: bool = True, speed: float | None = None
) -> list:
    """The lead after each frame of `boxes`, as (time, distance) or None."""
    tracker, cutter, lane = Tracker(camera), CutInFilter(), straight_lane(camera)
    leads = []
    for t, box in boxes:
        tracks = tracker.update([Detection(Kind.TRUCK, 0.95, box)], t)
        cutting = cutter.cutting_in(tracks, lane, t, speed) if cut_ins else frozenset()
        lead = lead_vehicle(tracks, lane, cutting_in=cutting)
        leads.append(None if lead is None else (t, lead[1][0]))
    return leads


def test_a_lorry_moving_across_into_the_lane_is_followed() -> None:
    """Its middle stayed 3-6 m off the lane's and angled it was over 6 m wide:
    by the rule for a vehicle ahead it was never the lead. Moving into the
    lane, it is, 2.8 s and 38 m before the truck got to it."""
    assert all(lead is None for lead in leads_through(TORINO_CUT_IN, TORINO_CAMERA, cut_ins=False))
    followed = [lead for lead in leads_through(TORINO_CUT_IN, TORINO_CAMERA) if lead is not None]
    assert followed, "the lorry was never followed"
    first_at, first_distance = followed[0]
    assert first_at <= -2.8
    assert first_distance == pytest.approx(38.0, abs=4.0)
    assert len(followed) == sum(1 for t, _ in TORINO_CUT_IN if t >= first_at), "and on to the end"


def lorry_beside(distance: float, middle: float, camera: GroundCamera) -> Detection:
    """The box of a lorry 2.55 m wide, 16.5 m long and 4 m tall, going the
    same way with its back `distance` ahead and its middle `middle` right of
    the eye: its back, and the left of its side running on to the front,
    nearer the middle of the image."""
    near = distance
    far = distance + 16.5
    left_side = middle - 2.55 / 2.0
    right_side = middle + 2.55 / 2.0
    edge = left_side if middle > 0.0 else right_side
    columns = (
        camera.centre_col + camera.focal_px * edge / far,
        camera.centre_col + camera.focal_px * left_side / near,
        camera.centre_col + camera.focal_px * right_side / near,
    )
    bottom = camera.horizon_row + camera.focal_px * camera.height_m / near
    top = camera.horizon_row - camera.focal_px * (4.0 - camera.height_m) / near
    return Detection(Kind.TRUCK, 0.9, (min(columns), top, max(columns), bottom))


@pytest.mark.parametrize("closing", [5.0, 12.0, 25.0])
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_lorry_going_straight_on_in_the_next_lane_is_not_cutting_in(
    closing: float, side: float
) -> None:
    """Overtaken, its side's far end creeps towards the middle of the image
    as the truck draws up; it is not moving across."""
    camera = TORINO_CAMERA
    middle = side * 3.75 - camera.seat_right_m
    tracker, cutter, lane = Tracker(camera), CutInFilter(), straight_lane(camera)
    t = 0.0
    while (distance := 55.0 - closing * t) > 8.0:
        tracks = tracker.update([lorry_beside(distance, middle, camera)], t)
        assert not cutter.cutting_in(tracks, lane, t), f"cutting in at {distance:.0f} m"
        assert lead_vehicle(tracks, lane) is None
        t += 0.1


def test_a_lorry_parked_off_the_road_whose_box_jumps_is_not_cutting_in() -> None:
    """Off a motorway at dusk a lorry stood 27-30 m to the right; its box went
    from its side to its back in a frame, 3 m towards the lane, and the truck
    braked from 70 to 41 km/h for it (drive-09, 2026-10-22)."""
    from core.perception.service import TrackReport

    cutter, lane = CutInFilter(), straight_lane(TORINO_CAMERA)
    sightings = [(30.4, 17.2)] * 4 + [(27.2, 8.3)] * 3
    for k, (lateral, width) in enumerate(sightings):
        lorry = TrackReport(2, "truck", True, True, 45.4, -10.0, lateral, width)
        assert not cutter.cutting_in([lorry], lane, k * 0.22)


def test_a_car_overtaking_alongside_as_fast_as_the_truck_is_not_cutting_in() -> None:
    """At night out of Torino: 10 m ahead in the next lane, its feet just above
    the dashboard, it crept towards the middle by 0.3 m/s (drive-01, 2026-10-24)."""
    from core.perception.service import TrackReport

    cutter, lane = CutInFilter(), straight_lane(TORINO_CAMERA)
    for k in range(30):
        lateral = 5.9 - 0.3 * k / 10.0
        car = TrackReport(8, "car", True, True, 9.8, 0.0, lateral, 3.05 + 0.02 * k)
        assert not cutter.cutting_in([car], lane, k / 10.0), f"cutting in after {k / 10:.1f} s"


class _Seen:
    """A track as `CutInFilter` and `lead_vehicle` read it."""

    def __init__(self, track_id: int, distance: float, closing: float, lateral: float) -> None:
        self.id, self.confirmed, self.measurement_age = track_id, True, 0.0
        self._estimate = (distance, closing, lateral, 2.0)

    def estimate(self):
        return self._estimate


def _moving_across(closings, speed):
    """Each 0.1 s a track 4.5 m off the lane's middle and closing on it at
    5 m/s, `closings` its closing speed; what the filter makes of it."""
    camera = TORINO_CAMERA
    cutter, lane = CutInFilter(), straight_lane(camera)
    middle = lane.right_at(30.0)
    said = []
    for k, closing in enumerate(closings):
        t = 0.1 * k
        seen = _Seen(7, 30.0, closing, middle - 4.5 + 5.0 * t)
        said.append(7 in cutter.cutting_in([seen], lane, t, speed))
    return said


def test_a_car_coming_the_other_way_round_a_bend_is_not_cutting_in() -> None:
    """In a bend the oncoming lane closes on the lane's middle. Its closing
    speed lagging down to the truck's own, it was then read as standing, and
    the truck braked at 4.8 m/s^2 at 51 km/h for a car going by."""
    speed = 14.3
    fading = [-20.0, -20.0, -20.0, -19.0, -18.0, -16.0, -14.0, -14.0, -14.0, -14.0]
    assert not any(_moving_across(fading, speed))
    assert any(_moving_across(fading, speed=None)), "unaware of the truck's speed, it counts"


@pytest.mark.parametrize("speed", [10.0, 13.5, 15.0])
def test_the_lorry_moving_across_at_torino_is_followed_at_the_truck_s_speed(speed: float) -> None:
    """It read as all but standing: taken for one not going the truck's way,
    it was never the lead, and would have been driven into."""
    followed = [
        lead
        for lead in leads_through(TORINO_CUT_IN, TORINO_CAMERA, speed=speed)
        if lead is not None
    ]
    assert followed, "the lorry was never followed"
    assert followed[0][0] <= -2.8


def test_a_car_going_the_truck_s_way_moving_across_is_still_cutting_in() -> None:
    assert any(_moving_across([-5.0] * 10, speed=14.3))


def test_a_vehicle_standing_in_the_lane_is_followed_whatever_it_was_read_as() -> None:
    camera = TORINO_CAMERA
    cutter, lane = CutInFilter(), straight_lane(camera)
    seen = _Seen(7, 30.0, -20.0, lane.right_at(30.0))
    assert 7 not in cutter.cutting_in([seen], lane, 0.0, 14.3)
    lead = lead_vehicle([seen], lane, cutting_in=frozenset())
    assert lead is not None and lead[0] is seen
