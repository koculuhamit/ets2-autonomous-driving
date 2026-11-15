"""The flat-road camera, and measuring it from what the game shows."""

from __future__ import annotations

import numpy as np
import pytest

from core.perception.camera import (
    GroundCamera,
    Line,
    focal_from_yaw,
    fov_deg,
    height_from_lane,
    vanishing_point,
)

CAMERA = GroundCamera(focal_px=830.0, horizon_row=437.0, centre_col=960.0, height_m=2.7)


def image_of(camera: GroundCamera, lateral: float, distance: float) -> tuple[float, float]:
    """(column, row) of a road point, by the pinhole model written out."""
    return (
        camera.centre_col + camera.focal_px * lateral / distance,
        camera.horizon_row + camera.focal_px * camera.height_m / distance,
    )


def lane_line(camera: GroundCamera, lateral: float) -> Line:
    return Line.through(*image_of(camera, lateral, 10.0), *image_of(camera, lateral, 40.0))


def test_a_road_point_comes_back_where_it_was() -> None:
    column, row = image_of(CAMERA, lateral=-1.8, distance=35.0)
    assert CAMERA.distance(row) == pytest.approx(35.0)
    assert CAMERA.lateral(column, 35.0) == pytest.approx(-1.8)


def test_nothing_on_the_road_is_at_or_above_the_horizon() -> None:
    assert CAMERA.distance(437.0) is None
    assert CAMERA.distance(300.0) is None
    assert CAMERA.locate((900, 380, 1000, 430)) is None


def test_a_box_on_the_road_gives_distance_offset_and_width() -> None:
    left, bottom = image_of(CAMERA, -1.25 + 0.5, 50.0)
    right, _ = image_of(CAMERA, 1.25 + 0.5, 50.0)
    distance, lateral, width = CAMERA.locate((left, bottom - 60, right, bottom))
    assert (distance, lateral, width) == pytest.approx((50.0, 0.5, 2.5))


def test_lane_lines_meet_at_the_horizon_straight_ahead() -> None:
    lines = [lane_line(CAMERA, x) for x in (-5.8, -1.3, 3.2)]
    assert vanishing_point(lines) == pytest.approx((960.0, 437.0))


def test_parallel_image_lines_have_no_vanishing_point() -> None:
    with pytest.raises(ValueError, match="do not meet"):
        vanishing_point([Line(0.0, 0.5), Line(100.0, 0.5)])


def test_the_lane_widening_gives_the_eye_height() -> None:
    """Whatever the focal length: the same lane seen by two lenses."""
    for focal in (600.0, 1400.0):
        camera = GroundCamera(focal, 437.0, 960.0, 2.7)
        assert height_from_lane(lane_line(camera, -1.5), lane_line(camera, 3.0), 4.5) == (
            pytest.approx(2.7)
        )


def test_lines_the_wrong_way_round_are_refused() -> None:
    with pytest.raises(ValueError, match="right line"):
        height_from_lane(lane_line(CAMERA, 3.0), lane_line(CAMERA, -1.5), 4.5)


def test_the_focal_length_comes_from_yaw_despite_bad_pairs() -> None:
    rng = np.random.default_rng(0)
    yaws = np.radians(rng.uniform(-1.0, 1.0, 200))
    shifts = 830.0 * np.tan(yaws) + rng.normal(0.0, 0.5, 200)
    shifts[:30] = rng.uniform(-20.0, 20.0, 30)  # the player looking around
    focal, used = focal_from_yaw(shifts, yaws)
    assert focal == pytest.approx(830.0, rel=0.01)
    assert 160 <= used <= 185


def test_a_focal_length_is_a_field_of_view() -> None:
    assert fov_deg(960.0, 1920.0) == pytest.approx(90.0)


def test_a_camera_survives_its_file() -> None:
    from core.perception.camera import GroundCamera as Camera

    assert Camera.from_json(CAMERA.to_json()) == CAMERA
    with pytest.raises(ValueError, match="not a camera"):
        Camera.from_json({"format": "other"})


def test_seen_from_above_the_road_moves_as_far_as_the_truck_did() -> None:
    """And a camera with the wrong scale says it moved further, in proportion."""
    import cv2

    from core.perception.camera import bird_eye_maps

    # Marks at irregular spacing: evenly spaced dashes would make the shift
    # ambiguous by a whole period.
    marks = np.cumsum(np.random.default_rng(1).uniform(2.5, 9.0, 40)) + 4.0

    def road_after(travelled: float) -> np.ndarray:
        image = np.full((1080, 1920), 60, dtype=np.uint8)
        for start in marks - travelled:
            for lateral in (-1.5, 0.0, 1.5):
                a = image_of(CAMERA, lateral, start)
                b = image_of(CAMERA, lateral, start + 1.5)
                cv2.line(image, tuple(map(round, a)), tuple(map(round, b)), 230, 5)
        return image.astype(np.float32)

    def measured(camera: GroundCamera) -> float:
        maps = bird_eye_maps(camera, (12.0, 44.0), (-3.0, 3.0), 0.05)
        before, after = (cv2.remap(road_after(d), *maps, cv2.INTER_LINEAR) for d in (0.0, 1.5))
        (_, shift), _ = cv2.phaseCorrelate(before, after)
        return -shift * 0.05

    assert measured(CAMERA) == pytest.approx(1.5, abs=0.1)
    too_long = GroundCamera(CAMERA.focal_px * 1.2, CAMERA.horizon_row, 960.0, CAMERA.height_m)
    assert measured(too_long) == pytest.approx(1.8, abs=0.12)
