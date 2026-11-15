"""The parts of the detector training tools that could be silently wrong.

A box shifted by the crop offset, or not flipped with its image, trains a
model that looks fine by its loss and is wrong on the road.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.perception.detection import Detection, Kind
from tools.pseudo_label import HORIZON, cut_by
from tools.train_detector import mirrored, targets_in_crop

CROP = (330, 250, 1920, 700)  # 1590 x 450
COCO_OF_KIND = [2, 7, 5, 3, 1, 0]  # car, truck, bus, motorcycle, bicycle, person


def test_a_label_becomes_a_normalised_box_in_the_crop() -> None:
    [box], classes = targets_in_crop(
        [{"bbox": [330 + 795 - 50, 250 + 225 - 20, 100, 40], "category_id": 1}], CROP, COCO_OF_KIND
    )
    assert box == pytest.approx([0.5, 0.5, 100 / 1590, 40 / 450])
    assert classes == [7], "a truck, as COCO's truck"


def test_a_label_partly_outside_the_crop_is_cut_and_one_wholly_outside_dropped() -> None:
    boxes, _ = targets_in_crop(
        [
            {"bbox": [300, 600, 100, 200], "category_id": 0},  # crosses the left and bottom edges
            {"bbox": [0, 0, 100, 100], "category_id": 0},  # in the sky above the windscreen
        ],
        CROP,
        COCO_OF_KIND,
    )
    assert len(boxes) == 1
    cx, cy, w, h = boxes[0]
    assert (cx - w / 2, cy + h / 2) == pytest.approx((0.0, 1.0))
    assert w == pytest.approx(70 / 1590)


def test_flipping_moves_a_box_to_the_other_side() -> None:
    boxes = np.array([[0.2, 0.5, 0.1, 0.2]], dtype=np.float32)
    assert mirrored(boxes)[0] == pytest.approx([0.8, 0.5, 0.1, 0.2])
    assert boxes[0, 0] == pytest.approx(0.2), "the original is left alone"


def test_a_box_the_zoomed_look_cut_off_is_left_to_the_wide_one() -> None:
    left, top, _, bottom = HORIZON.pixels(1920, 1080)
    inside = Detection(Kind.CAR, 0.9, (left + 100, top + 50, left + 140, top + 80))
    at_bottom = Detection(Kind.TRUCK, 0.9, (left + 100, bottom - 60, left + 300, bottom - 1))
    assert not cut_by(inside, HORIZON, 1920, 1080)
    assert cut_by(at_bottom, HORIZON, 1920, 1080)
