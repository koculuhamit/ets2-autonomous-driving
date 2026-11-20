"""Round a tight turn wide while its mouth is clear, kept in otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from core.control.turn_line import SHARE_RATE, WideTurnChoice
from core.planning.tight_turns import WideSpan

SPAN = WideSpan(100.0, 130.0, np.full(51, -3.0), np.array([[10.0, 140.0]]))
DT = 0.05


def _drive(choice, progress_from, progress_to, clear, speed=2.0):
    """Steps from one place to another; the last shift function."""
    shift, progress = None, progress_from
    while progress <= progress_to:
        shift = choice.shift(spans=[SPAN], progress=progress, mouth_clear=clear, dt=DT)
        progress += speed * DT
    return shift


def test_away_from_a_wide_turn_nothing_is_shifted() -> None:
    choice = WideTurnChoice()
    assert choice.shift(spans=[SPAN], progress=50.0, mouth_clear=lambda _m: True, dt=DT) is None


def test_the_mouth_clear_the_aim_goes_wide_eased_in() -> None:
    choice = WideTurnChoice()
    first = choice.shift(spans=[SPAN], progress=100.0, mouth_clear=lambda _m: True, dt=DT)
    assert first is not None
    assert first(110.0) == pytest.approx(-3.0 * SHARE_RATE * DT)
    shift = _drive(choice, 100.0, 125.0, lambda _m: True)
    assert shift(110.0) == pytest.approx(-3.0)


def test_the_mouth_not_clear_or_unseen_the_line_is_kept() -> None:
    for clear in (lambda _m: False, None):
        choice = WideTurnChoice()
        assert _drive(choice, 96.0, 140.0, clear) is None
        assert choice.chosen == "kept"


def test_once_into_the_junction_the_choice_is_held() -> None:
    """Something coming up to the mouth after the turn has begun does not
    swing the truck back in halfway round, nor does a mouth clearing then
    swing it out."""
    choice = WideTurnChoice()
    _drive(choice, 96.0, 129.9, lambda _m: True)
    shift = _drive(choice, 130.0, 145.0, lambda _m: False)
    assert choice.chosen == "wide"
    assert shift(140.0) == pytest.approx(-3.0)
    choice = WideTurnChoice()
    _drive(choice, 96.0, 129.9, lambda _m: False)
    assert _drive(choice, 130.0, 145.0, lambda _m: True) is None
    assert choice.chosen == "kept"


def test_a_mouth_filling_before_the_junction_eases_the_aim_back_in() -> None:
    choice = WideTurnChoice()
    _drive(choice, 96.0, 115.0, lambda _m: True)
    _drive(choice, 115.0, 129.0, lambda _m: False)
    assert choice.shift(spans=[SPAN], progress=129.5, mouth_clear=lambda _m: False, dt=DT) is None


def test_the_cruise_sees_the_mouth_clear_only_in_view_and_empty() -> None:
    from core.control.acc import MOUTH_CLEAR_M, AdaptiveCruise
    from tests.unit.test_acc import CAMERA, STRAIGHT, TrackReport, report

    cruise = AdaptiveCruise(CAMERA)
    mouth = np.array([[3.0, 40.0], [4.0, 45.0]])
    assert not cruise.mouth_clear(mouth), "nothing seen yet"

    def look(*tracks, at: float = 10.0) -> bool:
        kwargs = dict(path=STRAIGHT, progress=at, x=0.0, z=at, course=0.0, speed=2.0)
        cruise.step(report(*tracks, grabbed_at=100.0), 100.05, **kwargs)
        return cruise.mouth_clear(mouth)

    assert look()
    # Right of heading +z is -x: a car 3 m left of the line ahead stands at x = +3.
    there = TrackReport(4, "car", True, True, 32.0, 0.0, -3.0 - CAMERA.seat_right_m, 1.8)
    assert not look(there)
    far_aside = TrackReport(4, "car", True, True, 32.0, 0.0, MOUTH_CLEAR_M + 10.0, 1.8)
    assert look(far_aside)
    assert not look(at=60.0), "behind the truck: not in view"
