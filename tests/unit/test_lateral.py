"""Tests for lateral control.

The closed-loop tests drive a bicycle model through the *measured* steering
actuator -- a rate command the game integrates, taking about 1.3 s to reach
full lock. Leaving that out would make the controller look far better than it
is, because the slew limit is what bounds how fast it can answer a curve.

Several tests pin known properties of pure pursuit rather than thresholds
somebody chose: that it tracks a constant-curvature path essentially exactly,
and that its error entering a curve grows with the square of the lookahead.
Those are the reasons it is the right first controller and the reasons it will
eventually be replaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE
from core.control.lateral import (
    ACTROS_GEOMETRY,
    PurePursuit,
    TrailerGeometry,
    VehicleGeometry,
)
from core.control.steering import STEERING_CALIBRATION, SteeringPositionController
from core.localization.path import ReferencePath


def straight_path(length: float = 800.0) -> ReferencePath:
    z = np.arange(0.0, length, 1.0)
    return ReferencePath(np.column_stack([np.zeros_like(z), z]), spacing=2.0)


def arc_path(curvature: float, length: float = 900.0) -> ReferencePath:
    s = np.arange(0.0, length, 1.0)
    return ReferencePath(
        np.column_stack(
            [(1 - np.cos(curvature * s)) / curvature, np.sin(curvature * s) / curvature]
        ),
        spacing=2.0,
    )


def straight_then_arc(radius: float, straight: float = 250.0, arc: float = 500.0):
    """A step change in curvature -- no transition spiral, the hardest case."""
    z = np.arange(0.0, straight, 1.0)
    first = np.column_stack([np.zeros_like(z), z])
    curvature = -1.0 / radius
    s = np.arange(0.0, arc, 1.0)
    second = np.column_stack(
        [
            (1 - np.cos(curvature * s)) / curvature,
            straight + np.sin(curvature * s) / curvature,
        ]
    )
    return ReferencePath(np.vstack([first, second]), spacing=2.0)


def simulate(
    path: ReferencePath,
    controller: PurePursuit,
    *,
    speed: float,
    x: float = 0.0,
    z: float = 0.0,
    heading: float = 0.0,
    seconds: float = 60.0,
    dt: float = 1 / 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Bicycle model behind the measured steering actuator.

    Returns lateral error and path progress per step.
    """
    inner = SteeringPositionController()
    geometry = controller.geometry
    applied = 0.0
    progress = None
    errors, positions = [], []

    for _ in range(int(seconds / dt)):
        command = controller.step(
            x=x, z=z, heading=heading, speed=speed, path=path, progress=progress
        )
        progress = command.progress
        errors.append(command.lateral_error)
        positions.append(command.progress)

        rate = inner.command(target=command.steering, measured=applied)
        applied += STEERING_CALIBRATION.sign * STEERING_CALIBRATION.rate_gain * rate * dt
        applied = max(-1.0, min(1.0, applied))

        wheel_angle = geometry.angle_for_command(applied)
        heading += (speed / geometry.wheelbase) * math.tan(wheel_angle) * dt
        x += speed * math.sin(heading) * dt
        z += speed * math.cos(heading) * dt
        if command.at_end:
            break

    return np.array(errors), np.array(positions)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", [0.05, 0.2, 0.5, 0.9, 0.9488])
def test_the_steering_map_round_trips(command: float) -> None:
    angle = ACTROS_GEOMETRY.angle_for_command(command)
    assert ACTROS_GEOMETRY.command_for_angle(angle) == pytest.approx(command, abs=1e-6)


def test_the_steering_map_is_progressive_but_only_slightly() -> None:
    """The mean wheel angle is close to linear in the command -- 40.1 to 40.9
    degrees per unit across the range.

    The test exists because an earlier version of this map was indexed by the
    value the sweep *asked* for rather than the value it *applied*, and those
    differ at the top: the position loop reached 0.9488 when asked for 1.00.
    That made the map appear to lose sensitivity at lock when it gains it.
    Monotone ratio is the property that catches it.
    """
    ratios = [
        ACTROS_GEOMETRY.angle_for_command(command) / command
        for command, _ in ACTROS_GEOMETRY.steering_map[1:]
    ]
    assert ratios == sorted(ratios) or max(ratios) / min(ratios) < 1.05
    assert ratios[-1] > ratios[0]


def test_the_steering_map_is_odd() -> None:
    assert ACTROS_GEOMETRY.command_for_angle(
        -ACTROS_GEOMETRY.angle_for_command(0.4)
    ) == pytest.approx(-0.4, abs=1e-6)


def test_wheel_angle_is_clamped_to_the_measured_range() -> None:
    """Beyond the measurement the honest answer is the most we know how to ask
    for. The sweep never reached full lock, so the map stops at 0.9488 and the
    vehicle has a little more steering than this describes."""
    largest = ACTROS_GEOMETRY.steering_map[-1][0]
    assert ACTROS_GEOMETRY.command_for_angle(math.radians(80.0)) == pytest.approx(largest)
    assert largest < 1.0


def test_a_non_monotone_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="monotone"):
        VehicleGeometry(wheelbase=3.0, steering_map=((0.0, 0.0), (0.5, 0.3), (1.0, 0.2)))


def test_a_map_not_starting_at_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(0, 0\)"):
        VehicleGeometry(wheelbase=3.0, steering_map=((0.1, 0.05), (1.0, 0.6)))


def test_a_non_positive_wheelbase_is_rejected() -> None:
    with pytest.raises(ValueError, match="wheelbase"):
        VehicleGeometry(wheelbase=0.0, steering_map=((0.0, 0.0), (1.0, 0.6)))


# --------------------------------------------------------------------------- #
# Direction -- an inverted sign steers off the road
# --------------------------------------------------------------------------- #


def test_left_of_the_path_steers_right() -> None:
    path = straight_path()
    command = PurePursuit().step(x=3.0, z=100.0, heading=0.0, speed=11.0, path=path)
    assert command.lateral_error > 0.0
    assert command.steering < 0.0
    assert command.wheel_angle < 0.0


def test_right_of_the_path_steers_left() -> None:
    path = straight_path()
    command = PurePursuit().step(x=-3.0, z=100.0, heading=0.0, speed=11.0, path=path)
    assert command.lateral_error < 0.0
    assert command.steering > 0.0


def test_on_the_path_and_aligned_steers_straight() -> None:
    path = straight_path()
    command = PurePursuit().step(x=0.0, z=100.0, heading=0.0, speed=11.0, path=path)
    assert command.steering == pytest.approx(0.0, abs=1e-9)


def test_heading_error_is_reported_wrapped() -> None:
    path = straight_path()
    command = PurePursuit().step(x=0.0, z=100.0, heading=math.radians(200.0), speed=11.0, path=path)
    assert -math.pi < command.heading_error <= math.pi


# --------------------------------------------------------------------------- #
# Lookahead
# --------------------------------------------------------------------------- #


def test_lookahead_is_proportional_to_speed() -> None:
    """A fixed time ahead, so the same tuning behaves the same at every speed
    -- which is what stops a controller tuned at 30 km/h hunting at 80."""
    controller = PurePursuit(lookahead_time=2.0, min_lookahead=1.0, max_lookahead=1000.0)
    assert controller.lookahead_for(10.0) == pytest.approx(20.0)
    assert controller.lookahead_for(20.0) == pytest.approx(40.0)


def test_lookahead_is_clamped_at_both_ends() -> None:
    controller = PurePursuit(min_lookahead=8.0, max_lookahead=40.0)
    assert controller.lookahead_for(0.0) == pytest.approx(8.0)
    assert controller.lookahead_for(200.0) == pytest.approx(40.0)


def test_a_floor_below_the_vehicle_is_rejected_by_convention_not_code() -> None:
    """The floor exists because an aim point inside the truck degenerates the
    geometry, and a tractor-trailer is much longer than its wheelbase."""
    assert PurePursuit().lookahead_for(0.0) >= ACTROS_GEOMETRY.wheelbase


@pytest.mark.parametrize(
    "kwargs", [{"lookahead_time": 0.0}, {"min_lookahead": 0.0}, {"max_lookahead": 1.0}]
)
def test_invalid_lookahead_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="lookahead"):
        PurePursuit(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Closed loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("speed_kmh", [30.0, 50.0, 70.0])
def test_converges_onto_a_straight_path_from_an_offset(speed_kmh: float) -> None:
    errors, _ = simulate(straight_path(), PurePursuit(), speed=speed_kmh / 3.6, x=3.0, seconds=40.0)
    assert abs(errors[-1]) < 0.05
    assert abs(errors.min()) < 0.3, "overshoot past the path should stay small"


def test_convergence_time_is_the_same_at_every_speed() -> None:
    """The point of a speed-proportional lookahead."""
    times = []
    for speed_kmh in (30.0, 50.0, 70.0):
        errors, _ = simulate(
            straight_path(), PurePursuit(), speed=speed_kmh / 3.6, x=3.0, seconds=40.0
        )
        settled = next(
            (i / 60 for i in range(len(errors)) if np.all(np.abs(errors[i:]) < 0.15)),
            math.inf,
        )
        times.append(settled)
    assert max(times) - min(times) < 0.5


def test_a_constant_curvature_path_is_tracked_almost_exactly() -> None:
    """The property that makes pure pursuit the right first choice: the arc
    through the vehicle and the aim point *is* the path, so there is nothing
    to cut. Corner cutting is a transition effect, not a steady-state one."""
    errors, _ = simulate(arc_path(-1 / 250), PurePursuit(), speed=60 / 3.6, seconds=45.0)
    assert abs(errors[-300:].mean()) < 0.05


def test_entering_a_curve_costs_error_that_grows_with_the_lookahead() -> None:
    """The known tradeoff, and the reason a second controller is on the plan.

    The peak error on a step change in curvature scales with the square of the
    lookahead, so it is four times worse at twice the speed: 0.17 m at 30 km/h
    on a 120 m radius, 0.69 m at 60.
    """
    path = straight_then_arc(120.0)
    peaks = {}
    for speed_kmh in (30.0, 60.0):
        errors, progress = simulate(path, PurePursuit(), speed=speed_kmh / 3.6, seconds=70.0)
        in_curve = progress > 250.0
        peaks[speed_kmh] = np.abs(errors[in_curve]).max()

    assert peaks[30.0] < 0.3
    assert peaks[60.0] > 2.5 * peaks[30.0], "error should grow with the lookahead squared"
    assert peaks[60.0] < 1.0, "still under a metre on a tight curve at speed"


def test_recovers_after_the_curve_settles() -> None:
    errors, _progress = simulate(
        straight_then_arc(250.0), PurePursuit(), speed=50 / 3.6, seconds=70.0
    )
    assert abs(errors[-300:].mean()) < 0.05


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #


def test_the_end_of_the_path_is_reported() -> None:
    path = straight_path(length=100.0)
    command = PurePursuit().step(x=0.0, z=90.0, heading=0.0, speed=11.0, path=path)
    assert command.at_end is True


def test_the_wheel_does_not_swing_in_the_last_metres_of_the_path() -> None:
    """At a pump near Linz the aim point, pinned to the path's end, came under
    the truck: the wheel went to +0.72 and -1.0 in the last metre."""
    path = straight_path(length=100.0)
    controller = PurePursuit()
    mid = controller.step(x=0.3, z=50.0, heading=0.0, speed=2.0, path=path)
    for short in (5.0, 2.0, 0.5):
        end = controller.step(x=0.3, z=path.length - short, heading=0.0, speed=2.0, path=path)
        assert end.at_end
        assert abs(end.steering) <= abs(mid.steering) + 1e-6


def test_standing_on_the_aim_point_holds_straight() -> None:
    """Degenerate rather than impossible: a very short path and a stationary
    vehicle put the aim point under the truck."""
    z = np.arange(0.0, 6.0, 1.0)
    path = ReferencePath(np.column_stack([np.zeros_like(z), z]), spacing=2.0)
    command = PurePursuit().step(x=0.0, z=path.length, heading=0.0, speed=0.0, path=path)
    assert command.steering == pytest.approx(0.0)


def test_progress_is_returned_for_the_next_call() -> None:
    """Carried by the caller so the controller stays stateless and a replay is
    reproducible."""
    path = straight_path()
    controller = PurePursuit()
    first = controller.step(x=0.0, z=100.0, heading=0.0, speed=11.0, path=path)
    assert first.progress == pytest.approx(100.0, abs=1.0)
    second = controller.step(
        x=0.0, z=140.0, heading=0.0, speed=11.0, path=path, progress=first.progress
    )
    assert second.progress > first.progress


# --------------------------------------------------------------------------- #
# Curvature-aware lookahead
# --------------------------------------------------------------------------- #


def test_the_lookahead_shortens_where_the_path_bends() -> None:
    """A fixed time ahead is too far into a tight corner: 20 m of lookahead on
    a 10 m radius puts the aim point more than half way round, and the
    controller drives the chord."""
    controller = PurePursuit(curvature_gain=0.6)
    speed = 40 / 3.6

    on_a_straight = controller._lookahead_on(straight_path(), 100.0, speed)
    in_a_corner = controller._lookahead_on(arc_path(-1 / 15, length=400.0), 100.0, speed)

    assert on_a_straight == pytest.approx(controller.lookahead_for(speed))
    assert in_a_corner < on_a_straight / 2


def test_the_lookahead_never_goes_below_the_floor() -> None:
    """Below roughly a vehicle length the aim point is inside the truck."""
    controller = PurePursuit(curvature_gain=0.6, min_lookahead=8.0)
    assert controller._lookahead_on(arc_path(-1 / 5, length=200.0), 50.0, 30 / 3.6) >= 8.0


def test_the_scan_sees_a_corner_before_entering_it() -> None:
    """Reading the curvature underfoot would shorten the lookahead only once
    the corner had been entered, which is too late to have aimed anywhere
    useful. The window is scanned instead."""
    path = straight_then_arc(20.0, straight=250.0)
    controller = PurePursuit(curvature_gain=0.6)
    speed = 40 / 3.6

    well_before = controller._lookahead_on(path, 150.0, speed)
    approaching = controller._lookahead_on(path, 245.0, speed)
    assert approaching < well_before


def test_the_curvature_cap_can_be_disabled() -> None:
    controller = PurePursuit(curvature_gain=0.0)
    speed = 40 / 3.6
    assert controller._lookahead_on(arc_path(-1 / 15, length=400.0), 100.0, speed) == pytest.approx(
        controller.lookahead_for(speed)
    )


def test_curvature_scanning_improves_tracking_through_a_tight_corner() -> None:
    """Measured on the shape that motivated it: 1.57 m of error without the
    scan, 0.86 with, over a route whose tightest corner is a 10 m radius."""
    path = straight_then_arc(15.0, straight=150.0, arc=200.0)
    without, _ = simulate(path, PurePursuit(curvature_gain=0.0), speed=30 / 3.6, seconds=60.0)
    with_scan, _ = simulate(path, PurePursuit(curvature_gain=0.6), speed=30 / 3.6, seconds=60.0)
    assert np.abs(with_scan).max() < np.abs(without).max()


@pytest.mark.parametrize("kwargs", [{"curvature_gain": -1.0}, {"curvature_samples": 1}])
def test_invalid_curvature_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="curvature"):
        PurePursuit(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Trailer off-tracking
# --------------------------------------------------------------------------- #


def test_a_trailer_cuts_inside_a_right_turn() -> None:
    """Negative curvature is a right turn, and the trailer cuts to the inside,
    which is the negative side. Measured: the tractor held 0.70 m while the
    trailer was 2.12 m off the path."""
    trailer = TrailerGeometry(length=8.138)
    assert trailer.off_tracking(-1 / 20) < 0.0
    assert trailer.off_tracking(+1 / 20) > 0.0


def test_off_tracking_grows_as_the_corner_tightens() -> None:
    trailer = TrailerGeometry(length=8.138)
    wide = abs(trailer.off_tracking(-1 / 200))
    tight = abs(trailer.off_tracking(-1 / 20))
    assert tight > 5 * wide


def test_off_tracking_is_negligible_on_a_straight() -> None:
    assert TrailerGeometry(length=8.138).off_tracking(0.0) == 0.0


def test_off_tracking_matches_the_closed_form() -> None:
    """`R - sqrt(R^2 - L^2)`, not the `L^2 / 2R` approximation, which is 20 %
    out once the radius approaches the trailer's length."""
    trailer = TrailerGeometry(length=8.0)
    radius = 20.0
    expected = radius - math.sqrt(radius**2 - 8.0**2)
    assert abs(trailer.off_tracking(-1 / radius)) == pytest.approx(expected)


def test_a_corner_tighter_than_the_trailer_is_handled() -> None:
    """The geometry has no solution there and the real vehicle would be
    jackknifing; returning a NaN would propagate silently into a steering
    command."""
    value = TrailerGeometry(length=8.138).off_tracking(-1 / 5.0)
    assert math.isfinite(value)
    assert value < 0.0


def test_trailer_geometry_comes_from_the_telemetry() -> None:
    """Hook position and wheel positions are both reported, and both change
    with the trailer, so nothing here is configured."""
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    trailer = block["trailers"][0, 0]
    trailer["connected"] = 1
    trailer["hook_position"][2] = -5.0
    trailer["wheels"]["count"] = 2
    trailer["wheels"]["position"][0]["z"] = 2.0
    trailer["wheels"]["position"][1]["z"] = 4.0

    geometry = TrailerGeometry.from_block(block[0])
    assert geometry is not None
    assert geometry.length == pytest.approx(8.0)


def test_no_trailer_gives_no_geometry() -> None:
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    assert TrailerGeometry.from_block(block[0]) is None


def test_a_zero_length_trailer_is_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        TrailerGeometry(length=0.0)


def _through_a_corner(geometry: VehicleGeometry, ahead: float, radius: float = 15.0):
    """The driven axle's worst error, metres, through the steady part of a
    90-degree corner at 15 km/h, the controller told the position of a point
    `ahead` of it (the game's origin, by the front axle)."""
    straight, arc = 60.0, radius * math.pi / 2
    s = np.arange(0.0, straight, 0.5)
    first = np.column_stack([np.zeros_like(s), s])
    k = -1.0 / radius
    a = np.arange(0.0, arc, 0.5)
    second = np.column_stack([(1 - np.cos(k * a)) / k, straight + np.sin(k * a) / k])
    end, heading = second[-1], -math.pi / 2
    d = np.arange(0.5, 60.0, 0.5)
    third = np.column_stack([end[0] + d * math.sin(heading), end[1] + d * math.cos(heading)])
    path = ReferencePath(np.vstack([first, second, third]), spacing=0.5)
    controller = PurePursuit(geometry, lookahead_time=1.2)
    inner = SteeringPositionController()
    x = z = heading = applied = 0.0
    speed, dt, progress, worst = 15.0 / 3.6, 1 / 60, None, 0.0
    for _ in range(int(60.0 / dt)):
        fx, fz = x + ahead * math.sin(heading), z + ahead * math.cos(heading)
        command = controller.step(
            x=fx, z=fz, heading=heading, speed=speed, path=path, progress=progress
        )
        progress = command.progress
        rate = inner.command(target=command.steering, measured=applied)
        applied += STEERING_CALIBRATION.sign * STEERING_CALIBRATION.rate_gain * rate * dt
        applied = max(-1.0, min(1.0, applied))
        wheel = geometry.angle_for_command(applied)
        heading += speed / geometry.wheelbase * math.tan(wheel) * dt
        x += speed * math.sin(heading) * dt
        z += speed * math.cos(heading) * dt
        axle = path.nearest(x, z, near=progress - ahead)
        if straight + 0.4 * arc < axle.s < straight + 0.9 * arc:
            worst = max(worst, abs(path.lateral_error(x, z, axle)))
        if command.at_end:
            break
    return worst


def test_the_driven_axle_is_steered_round_a_corner_not_the_front() -> None:
    """The game gives the truck's origin, by the front axle, 2.87 m ahead of
    the driven axles of the 8x4 in Bern; steered as if it were the axle, the
    truck turned about its front and the trailer came round inside."""
    from dataclasses import replace

    ahead = 2.87
    blind = _through_a_corner(ACTROS_GEOMETRY, ahead)
    told = _through_a_corner(replace(ACTROS_GEOMETRY, axle_behind=ahead), ahead)
    assert told < 0.35, "as if given the axle itself"
    assert blind > told + 1.0, "1.7 m in on a 15 m radius, as in the game"
