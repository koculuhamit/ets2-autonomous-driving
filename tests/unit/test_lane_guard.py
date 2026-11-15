"""Keeping the truck inside its painted lane where the map's line leaves it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.control.lane_guard import (
    HALF_WIDTH_M,
    HOLD_S,
    MARGIN_M,
    MAX_SHIFT_M,
    SHIFT_RATE_MPS,
    LaneGuard,
    trusted,
)
from core.control.lateral import PurePursuit
from core.localization.path import ReferencePath

LEFT, RIGHT = -2.2, 2.2
"""A 4.4 m lane, the eye in its middle (metres right of the eye)."""

ROOM = 2.2 - HALF_WIDTH_M - MARGIN_M


def test_a_pair_is_trusted_only_as_wide_as_a_lane_and_where_the_last_was() -> None:
    assert trusted(LEFT, RIGHT, (0.0, 0.0), 0.1)
    assert not trusted(LEFT, RIGHT, None, 0.1), "one pair alone is not enough"
    assert not trusted(-1.0, 1.0, (0.0, 0.0), 0.1), "2 m: an arrow, not a lane"
    assert not trusted(LEFT, RIGHT, (0.0, 1.5), 0.1), "jumped across"
    assert not trusted(LEFT, RIGHT, (0.0, 0.0), 1.0), "the last too long ago"
    assert not trusted(float("nan"), RIGHT, (0.0, 0.0), 0.1)
    assert not trusted(-6.6, -2.2, (0.0, -4.4), 0.1), "the next lane's: the truck not in it"


def test_a_line_over_the_right_paint_short_of_a_left_turn_is_brought_back_in() -> None:
    """Bremen's port: 2.3 m right of the painted middle, a left turn ahead."""
    guard = LaneGuard()
    assert guard.wanted(LEFT, RIGHT, at_check=2.3, ahead=-3.0) == pytest.approx(-(2.3 - ROOM))
    assert guard.wanted(LEFT, RIGHT, at_check=2.4, ahead=-3.0) == pytest.approx(-MAX_SHIFT_M)


def test_a_line_far_off_the_paint_is_leaving_the_lane_and_left_alone() -> None:
    """Into an S-bend off a Dutch dual carriageway the line was 4.4 m right of
    the paint's middle; held 1.5 m left, the truck hit the median's sign."""
    guard = LaneGuard()
    assert guard.wanted(LEFT, RIGHT, at_check=4.4, ahead=-3.0) == 0.0
    assert guard.wanted(LEFT, RIGHT, at_check=-4.4, ahead=3.0) == 0.0


def test_the_line_may_leave_the_lane_the_way_the_route_goes() -> None:
    guard = LaneGuard()
    assert guard.wanted(LEFT, RIGHT, at_check=-2.3, ahead=-6.0) == 0.0, "turning left"
    assert guard.wanted(LEFT, RIGHT, at_check=2.3, ahead=6.0) == 0.0, "an exit to the right"
    assert guard.wanted(LEFT, RIGHT, at_check=-2.3, ahead=-2.0) == pytest.approx(2.3 - ROOM)


def test_inside_the_lane_nothing_is_shifted() -> None:
    guard = LaneGuard()
    assert guard.wanted(LEFT, RIGHT, at_check=0.6, ahead=0.6) == 0.0, "swung wide within it"
    assert guard.wanted(LEFT, RIGHT, at_check=-ROOM, ahead=0.0) == 0.0


def test_the_shift_is_taken_up_gently_held_a_moment_and_let_go() -> None:
    guard = LaneGuard()
    dt = 1 / 20
    t = 0.0
    for _ in range(5):  # the first, three more agreeing with it, over twice
        guard.step(t, LEFT, RIGHT, at_check=2.3, ahead=-3.0)
        t += dt
    for _ in range(40):
        guard.step(t, LEFT, RIGHT, at_check=2.3, ahead=-3.0)
        assert abs(guard.shift) <= SHIFT_RATE_MPS * t + 1e-9
        t += dt
    assert guard.shift == pytest.approx(-min(2.3 - ROOM, MAX_SHIFT_M), abs=1e-6)
    held = guard.shift
    guard.step(t + HOLD_S / 2)
    assert guard.shift == pytest.approx(held), "paint out of view a moment: held"
    t += HOLD_S + 0.1
    for _ in range(80):
        guard.step(t)
        t += dt
    assert guard.shift == 0.0, "out of view: let go"


def test_pairs_between_views_of_one_line_are_trusted_and_another_lane_s_resets() -> None:
    guard = LaneGuard()
    nan = float("nan")
    for k, (left, right) in enumerate([(LEFT, RIGHT), (nan, RIGHT), (LEFT, RIGHT), (LEFT, nan)]):
        guard.step(k * 0.05, left, right, at_check=2.3, ahead=-3.0)
    guard.step(0.2, LEFT, RIGHT, at_check=2.3, ahead=-3.0)
    assert guard._wanted == 0.0, "the first pair has none before it: two only"
    guard.step(0.25, LEFT, RIGHT, at_check=2.3, ahead=-3.0)
    assert guard._wanted == 0.0, "a third: trusted, over once"
    guard.step(0.3, LEFT, RIGHT, at_check=2.3, ahead=-3.0)
    assert guard._wanted < 0.0, "over twice running: moved"
    guard = LaneGuard()
    for k, (left, right) in enumerate([(LEFT, RIGHT), (-6.6, -2.2), (LEFT, RIGHT), (LEFT, RIGHT)]):
        guard.step(k * 0.05, left, right, at_check=2.3, ahead=-3.0)
    assert guard._wanted == 0.0, "the next lane's pair between: counted again"


def test_an_untrusted_pair_changes_nothing() -> None:
    guard = LaneGuard()
    for k in range(20):
        guard.step(k / 20, -0.9, 0.9, at_check=2.3, ahead=-3.0)  # 1.8 m: not a lane
    assert guard.shift == 0.0


def test_the_aim_shifted_right_steers_right_of_the_path() -> None:
    path = ReferencePath(np.column_stack([np.zeros(200), np.arange(0.0, 400.0, 2.0)]), spacing=2.0)
    pursuit = PurePursuit()
    straight = pursuit.step(x=0.0, z=50.0, heading=0.0, speed=10.0, path=path)
    right = pursuit.step(x=0.0, z=50.0, heading=0.0, speed=10.0, path=path, aim_shift=1.0)
    left = pursuit.step(x=0.0, z=50.0, heading=0.0, speed=10.0, path=path, aim_shift=-1.0)
    assert straight.wheel_angle == pytest.approx(0.0, abs=1e-9)
    assert math.copysign(1.0, right.wheel_angle) == -math.copysign(1.0, left.wheel_angle)
    assert right.lateral_error == straight.lateral_error, "errors stay the path's own"
    # Steering right is a negative wheel angle: left of the vehicle is +x at heading 0.
    assert right.wheel_angle < 0.0


def test_off_the_paint_the_shift_is_held_on_along_the_path_then_let_go() -> None:
    """Bremen: the turn 40 m on was as far off on the map as the straight."""
    from core.control.lane_guard import HOLD_M, RELEASE_RATE_MPS

    guard = LaneGuard()
    t, s = 0.0, 0.0
    for _ in range(30):
        guard.step(t, LEFT, RIGHT, at_check=2.3, ahead=-3.0, progress=s)
        t, s = t + 0.05, s + 0.5
    kept = guard.shift
    assert kept < -1.0
    while s < 0.5 * 30 + HOLD_M - 1.0:  # the paint out of view, 10 m/s
        guard.step(t, progress=s)
        t, s = t + 0.05, s + 0.5
    assert guard.shift == pytest.approx(kept), "held through the turn"
    for _ in range(4):
        guard.step(t, progress=s)
        t, s = t + 0.05, s + 0.5
    assert kept < guard.shift < 0.0, "then let go"
    assert guard.shift - kept <= RELEASE_RATE_MPS * 0.2 + 1e-9, "gently"


def test_into_a_turn_off_the_lane_the_shift_is_not_lessened_by_the_paint() -> None:
    """The line into the turn crosses the straight lane's paint: no measure
    of how far out the map is."""
    guard = LaneGuard()
    t, s = 0.0, 0.0
    for _ in range(30):
        guard.step(t, LEFT, RIGHT, at_check=2.3, ahead=-3.0 + 2.3 - 2.3, progress=s)
        t, s = t + 0.05, s + 0.5
    kept = guard.shift
    for at_check in (1.3, 0.2, -0.6):  # the line turning left across the paint
        guard.step(t, LEFT, RIGHT, at_check=at_check, ahead=at_check - 20.0, progress=s)
        t, s = t + 0.05, s + 0.5
    assert guard.shift == pytest.approx(kept)
    for _ in range(10):  # back on a straight road, the paint agreeing
        guard.step(t, LEFT, RIGHT, at_check=0.0, ahead=0.0, progress=s)
        t, s = t + 0.05, s + 0.5
    assert guard.shift > kept, "let go"


def test_at_motorway_speed_the_paint_is_not_acted_on() -> None:
    """Out of Bremen the classic finder moved the truck 1.5 m towards the
    next lane at 80 km/h, the map right."""
    from core.control.lane_guard import MAX_SPEED_MPS

    fast, slow = LaneGuard(), LaneGuard()
    for k in range(10):
        fast.step(k * 0.05, LEFT, RIGHT, at_check=2.3, ahead=-3.0, speed=MAX_SPEED_MPS + 1.0)
        slow.step(k * 0.05, LEFT, RIGHT, at_check=2.3, ahead=-3.0, speed=MAX_SPEED_MPS - 1.0)
    assert fast.shift == 0.0
    assert slow.shift < 0.0


def test_at_town_speed_the_paint_does_not_move_the_truck() -> None:
    """At 39 km/h in England, the lane's lines opening where a junction's lanes
    begin, the truck was moved 1.4 m and let go back into a car coming up."""
    guard = LaneGuard()
    for k in range(40):
        guard.step(k * 0.05, LEFT, RIGHT, at_check=2.3, ahead=2.3, speed=39.0 / 3.6)
    assert guard.shift == 0.0
