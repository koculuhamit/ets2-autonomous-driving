"""The test that would have caught a 180 degree error.

`convert.heading_rad` reports the game's heading channel; the direction of
travel in the `atan2(dx, dz)` frame that positions live in is `heading + pi`.
Feeding the first into a controller whose geometry comes from positions makes
every heading error 180 degrees out, and the symptom -- a controller steering
hard the wrong way -- says nothing about which of the two angles is wrong.

So the relationship gets asserted directly, against a vehicle moving along a
path built from its own positions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bridge import convert
from core.bridge.layout import BLOCK_DTYPE
from core.control.lateral import PurePursuit
from core.localization.path import ReferencePath


def block_at(heading_rotations: float) -> np.void:
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    block["truck"]["placement"]["orientation"]["heading"] = heading_rotations
    return block[0]


@pytest.mark.parametrize("rotations", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
def test_course_is_the_angle_of_the_forward_vector(rotations: float) -> None:
    block = block_at(rotations)
    forward_x, forward_z = convert.forward_unit(block)
    course = convert.course_rad(block)
    assert math.sin(course) == pytest.approx(forward_x, abs=1e-9)
    assert math.cos(course) == pytest.approx(forward_z, abs=1e-9)


@pytest.mark.parametrize("rotations", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
def test_course_and_heading_differ_by_half_a_turn(rotations: float) -> None:
    block = block_at(rotations)
    difference = convert.wrap_angle(convert.course_rad(block) - convert.heading_rad(block))
    assert abs(abs(difference) - math.pi) < 1e-9


def test_a_vehicle_on_its_own_path_has_no_heading_error() -> None:
    """The integration check. Build a path from a set of positions, then put a
    vehicle at one of them facing the way it was travelling, and the lateral
    controller should see essentially no heading error.

    With `heading_rad` in place of `course_rad` this reads 180 degrees.
    """
    z = np.arange(0.0, 400.0, 1.0)
    points = np.column_stack([np.sin(z / 60.0) * 15.0, z])
    path = ReferencePath(points, spacing=2.0)
    controller = PurePursuit()

    progress = None
    errors = []
    for index in range(20, len(points) - 40, 5):
        x, z_value = points[index]
        ahead = points[index + 1] - points[index]
        course = math.atan2(ahead[0], ahead[1])
        command = controller.step(
            x=float(x),
            z=float(z_value),
            heading=course,
            speed=11.0,
            path=path,
            progress=progress,
        )
        progress = command.progress
        errors.append(abs(command.heading_error))

    assert max(errors) < math.radians(2.0)
