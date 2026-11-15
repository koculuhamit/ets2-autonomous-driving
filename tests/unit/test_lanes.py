"""Finding the truck's lane lines in a drawn road."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.perception.camera import GroundCamera, height_from_lane, vanishing_point
from core.perception.lanes import lane_lines

CAMERA = GroundCamera(focal_px=830.0, horizon_row=437.0, centre_col=960.0, height_m=2.9)


def point(lateral: float, distance: float) -> tuple[int, int]:
    return (
        round(CAMERA.centre_col + CAMERA.focal_px * lateral / distance),
        round(CAMERA.horizon_row + CAMERA.focal_px * CAMERA.height_m / distance),
    )


def road(lines: dict[float, bool]) -> np.ndarray:
    """A dark road with paint at each lateral offset; True for dashed."""
    image = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    image[:437] = 170  # bright sky, which the top-hat must ignore
    for lateral, dashed in lines.items():
        step = 12.0 if dashed else 1.0
        for near in np.arange(4.0, 120.0, step):
            far = near + (4.5 if dashed else 1.0)
            cv2.line(image, point(lateral, near), point(lateral, far), (235, 235, 235), 4)
    return image


def test_the_lines_of_the_own_lane_are_found_among_others() -> None:
    image = road({-6.0: True, -1.5: True, 3.0: False, 7.5: False})
    left, right = lane_lines(image)
    assert left is not None and right is not None
    column, row = vanishing_point([left, right])
    assert (column, row) == pytest.approx((960.0, 437.0), abs=4.0)
    assert height_from_lane(left, right, 4.5) == pytest.approx(2.9, rel=0.05)


def test_a_road_without_paint_has_no_lines() -> None:
    blank = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    assert lane_lines(blank) == (None, None)


def test_the_lines_are_read_off_as_metres_either_side_of_the_eye() -> None:
    from core.perception.lanes import lateral_at

    left, right = lane_lines(road({-1.4: False, 3.1: True}))
    assert lateral_at(left, CAMERA) == pytest.approx(-1.4, abs=0.25)
    assert lateral_at(right, CAMERA) == pytest.approx(3.1, abs=0.25)
    assert np.isnan(lateral_at(None, CAMERA))
