"""Tests for the telemetry conversions.

These encode conventions that were *measured*, not read off documentation, so
the tests are written as statements about the vehicle: pointing north means
travelling north, nose-down means accelerating, forward acceleration is the
negation of the raw channel. A change that quietly flips one of those would
otherwise surface as a controller that brakes to speed up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bridge import convert
from core.bridge.layout import BLOCK_DTYPE


def make_block(
    *,
    speed: float = 0.0,
    accel_z: float = 0.0,
    heading: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    trailer_connected: bool = False,
    trailer_heading: float = 0.0,
) -> np.void:
    """Build a block with the fields these conversions read.

    `heading`, `pitch` and `roll` are in rotations, as the game reports them.
    """
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    truck = block["truck"]
    truck["speed"] = speed
    truck["linear_acceleration"]["z"] = accel_z
    truck["placement"]["orientation"]["heading"] = heading
    truck["placement"]["orientation"]["pitch"] = pitch
    truck["placement"]["orientation"]["roll"] = roll
    block["trailers"][0, 0]["connected"] = 1 if trailer_connected else 0
    block["trailers"][0, 0]["placement"]["orientation"]["heading"] = trailer_heading
    return block[0]


# --------------------------------------------------------------------------- #
# Speed and acceleration
# --------------------------------------------------------------------------- #


def test_speed_is_converted_to_kmh() -> None:
    assert convert.speed_kmh(make_block(speed=10.0)) == pytest.approx(36.0)


def test_reversing_speed_stays_negative() -> None:
    assert convert.speed_kmh(make_block(speed=-2.0)) == pytest.approx(-7.2)


def test_forward_acceleration_negates_the_raw_channel() -> None:
    """The vehicle-frame z axis runs backwards. Measured over coasting,
    accelerating and braking runs; see core/bridge/convert.py."""
    assert convert.forward_acceleration(make_block(accel_z=-1.5)) == pytest.approx(1.5)
    assert convert.forward_acceleration(make_block(accel_z=2.0)) == pytest.approx(-2.0)


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("heading_rotations", "expected_x", "expected_z"),
    [
        (0.0, 0.0, -1.0),  # heading 0
        (0.25, -1.0, 0.0),  # 90 degrees
        (0.5, 0.0, 1.0),  # 180
        (0.75, 1.0, 0.0),  # 270 -- the coastdown runs, travelling +x
    ],
)
def test_forward_unit_covers_every_quadrant(
    heading_rotations: float, expected_x: float, expected_z: float
) -> None:
    x, z = convert.forward_unit(make_block(heading=heading_rotations))
    assert x == pytest.approx(expected_x, abs=1e-9)
    assert z == pytest.approx(expected_z, abs=1e-9)


def test_forward_unit_is_not_sin_cos_of_heading() -> None:
    """Stated explicitly because it is the obvious guess and it is wrong: the
    direction is the negation, verified against real motion in all four
    quadrants."""
    heading = 0.3
    x, z = convert.forward_unit(make_block(heading=heading))
    radians = heading * 2 * math.pi
    assert x == pytest.approx(-math.sin(radians))
    assert z == pytest.approx(-math.cos(radians))


def test_forward_unit_is_a_unit_vector() -> None:
    for heading in (0.0, 0.1, 0.37, 0.5, 0.9):
        x, z = convert.forward_unit(make_block(heading=heading))
        assert math.hypot(x, z) == pytest.approx(1.0)


def test_orientation_is_converted_from_rotations_to_radians() -> None:
    assert convert.heading_rad(make_block(heading=0.5)) == pytest.approx(math.pi)
    assert convert.pitch_rad(make_block(pitch=0.25)) == pytest.approx(math.pi / 2)
    assert convert.roll_rad(make_block(roll=-0.25)) == pytest.approx(-math.pi / 2)


# --------------------------------------------------------------------------- #
# Gradient
# --------------------------------------------------------------------------- #


def test_downhill_gives_positive_grade_acceleration() -> None:
    """Negative pitch is nose-down. Measured: fitting lift-off acceleration
    against sin(pitch) gives -8.33 m/s^2 against g = 9.81, R^2 = 0.92."""
    downhill = make_block(pitch=-5.0 / 360.0)  # 5 degrees nose-down
    assert convert.grade_acceleration(downhill) > 0.0


def test_uphill_gives_negative_grade_acceleration() -> None:
    assert convert.grade_acceleration(make_block(pitch=5.0 / 360.0)) < 0.0


def test_flat_ground_gives_no_grade_acceleration() -> None:
    assert convert.grade_acceleration(make_block(pitch=0.0)) == pytest.approx(0.0)


def test_grade_acceleration_magnitude_matches_gravity() -> None:
    """A 5 degree grade is about 0.86 m/s^2, which is several times the drag
    measured at 45 km/h -- which is why gradient feedforward comes first."""
    block = make_block(pitch=-5.0 / 360.0)
    expected = convert.GRAVITY * math.sin(math.radians(5.0))
    assert convert.grade_acceleration(block) == pytest.approx(expected)
    assert expected == pytest.approx(0.855, abs=0.01)


# --------------------------------------------------------------------------- #
# Angles and the hitch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (1.5 * math.pi, -0.5 * math.pi),
        (-1.5 * math.pi, 0.5 * math.pi),
        (3.0 * math.pi, math.pi),
    ],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert convert.wrap_angle(angle) == pytest.approx(expected)


def test_hitch_angle_is_zero_without_a_trailer() -> None:
    block = make_block(heading=0.3, trailer_connected=False, trailer_heading=0.1)
    assert convert.hitch_angle_rad(block) == 0.0


def test_hitch_angle_is_zero_when_aligned() -> None:
    block = make_block(heading=0.3, trailer_connected=True, trailer_heading=0.3)
    assert convert.hitch_angle_rad(block) == pytest.approx(0.0)


def test_hitch_angle_wraps_across_north() -> None:
    """The combination is 2 degrees apart, not 358. Getting this wrong puts a
    full lock of phantom error into the parking controller every time the
    truck passes north."""
    block = make_block(heading=0.005, trailer_connected=True, trailer_heading=0.995)
    assert convert.hitch_angle_rad(block) == pytest.approx(math.radians(3.6), abs=1e-6)


def test_hitch_angle_stays_within_half_a_turn() -> None:
    for truck_heading in np.linspace(0.0, 1.0, 21):
        for trailer_heading in np.linspace(0.0, 1.0, 21):
            block = make_block(
                heading=float(truck_heading),
                trailer_connected=True,
                trailer_heading=float(trailer_heading),
            )
            assert -math.pi < convert.hitch_angle_rad(block) <= math.pi + 1e-12


def test_the_road_s_speed_limit_is_read_in_metres_per_second() -> None:
    block = make_block()
    block["navigation"]["speed_limit"] = 80 / 3.6
    assert convert.road_speed_limit(block) == pytest.approx(80 / 3.6)


def test_no_speed_limit_is_none_not_zero() -> None:
    """Zero is what the game sends when it has no limit, not a limit of zero:
    taken literally it would stop the truck on every road without a sign."""
    assert convert.road_speed_limit(make_block()) is None


def job_block(*, trailer_connected: bool, active: bool = True) -> np.void:
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    job = block["job"]
    job["job_active"] = 1 if active else 0
    job["source_company"] = b"ika_bohag"
    job["source_city"] = b"milano"
    job["destination_company"] = b"cargotras"
    job["destination_city"] = b"milano"
    block["trailers"][0, 0]["connected"] = 1 if trailer_connected else 0
    return block[0]


def test_the_job_s_route_goes_to_its_trailer_first_then_to_its_destination() -> None:
    """Milan: the trailer waited at its source, and the game's route went there."""
    assert convert.job_target(job_block(trailer_connected=False)) == ("ika_bohag", "milano")
    assert convert.job_target(job_block(trailer_connected=True)) == ("cargotras", "milano")
    assert convert.job_target(job_block(trailer_connected=True, active=False)) is None
