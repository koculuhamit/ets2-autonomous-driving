"""A new route leaving the one driven too soon to drive onto it."""

from __future__ import annotations

import numpy as np

from core.localization.path import ReferencePath
from tools.follow import parts_ahead


def road(leave_at: float | None, offset: float = 0.0) -> ReferencePath:
    """North along x = `offset` for 400 m, or bearing off to the right from
    `leave_at` m on: an exit."""
    z = np.arange(0.0, 400.0, 1.0)
    x = np.full_like(z, offset)
    if leave_at is not None:
        x = x - np.clip(z - leave_at, 0.0, None) * 0.3
    return ReferencePath(np.column_stack([x, z]), spacing=1.0)


def test_an_exit_taken_60_m_on_is_found_parting_soon() -> None:
    """Turned to a pump at 77 km/h, the exit 60 m on: over the gore, and its sign hit."""
    parts = parts_ahead(road(None), road(60.0), 0.0, 0.0, 0.0)
    assert parts is not None and 60.0 <= parts <= 75.0


def test_a_route_going_the_same_way_does_not_part() -> None:
    assert parts_ahead(road(None), road(None), 0.0, 0.0, 0.0) is None
    assert parts_ahead(road(None), road(300.0), 0.0, 0.0, 0.0) is None, "beyond the look"


def test_a_route_a_lane_over_already_is_left_to_easing() -> None:
    assert parts_ahead(road(None), road(None, offset=3.5), 0.0, 0.0, 0.0) is None


def lane_then_exit(over_from: float, fork_at: float) -> ReferencePath:
    """Into the lane on the right from `over_from` m on, 3.75 m over by 60 m
    later, and off on an exit from `fork_at`."""
    z = np.arange(0.0, 400.0, 1.0)
    x = -np.clip((z - over_from) / 60.0, 0.0, 1.0) * 3.75
    x = x - np.clip(z - fork_at, 0.0, None) * 0.3
    return ReferencePath(np.column_stack([x, z]), spacing=1.0)


def test_the_fork_is_found_past_the_lane_change_before_it() -> None:
    """An exit 100 m on at 78 km/h, its lane change begun 20 m on."""
    from tools.follow import FORK_M

    new = lane_then_exit(20.0, 100.0)
    assert parts_ahead(road(None), new, 0.0, 0.0, 0.0) < 60.0, "the lane change"
    fork = parts_ahead(road(None), new, 0.0, 0.0, 0.0, apart=FORK_M)
    assert fork is not None and 95.0 <= fork <= 115.0


def _two_lanes_parting(fork_at: float) -> tuple[ReferencePath, ReferencePath]:
    """The lane driven, turning off to the right from `fork_at`; and the lane
    beside it, 3.5 m to the left, going straight on."""
    z = np.arange(0.0, 400.0, 1.0)
    driven = -np.clip(z - fork_at, 0.0, None) * 0.3
    return (
        ReferencePath(np.column_stack([driven, z]), spacing=1.0),
        ReferencePath(np.column_stack([np.full_like(z, 3.5), z]), spacing=1.0),
    )


def test_a_lane_over_forking_before_the_easing_is_done_is_refused() -> None:
    """Out of an exit lane 2.2 s from its fork, at 59 km/h, the line drawn
    between the two ran straight on between them, into the barrier."""
    from types import SimpleNamespace

    from tools.follow import FORK_EASE_M, a_lane_over, refused_as_parting

    marks = SimpleNamespace(changes=[])
    old, new = _two_lanes_parting(30.0)
    assert a_lane_over(old, new, 0.0, 0.0, 0.0)
    assert refused_as_parting(old, new, marks, 0.0, 0.0, 0.0, 16.0) is not None
    old, new = _two_lanes_parting(FORK_EASE_M + 60.0)
    assert refused_as_parting(old, new, marks, 0.0, 0.0, 0.0, 16.0) is None, "room to ease over"


def test_a_lane_over_going_the_same_way_is_still_taken() -> None:
    from types import SimpleNamespace

    from tools.follow import refused_as_parting

    marks = SimpleNamespace(changes=[])
    assert (
        refused_as_parting(road(None), road(None, offset=3.5), marks, 0.0, 0.0, 0.0, 16.0) is None
    )
