"""A view of the road a moment late slows the truck; a lost one stops it."""

from __future__ import annotations

from core.perception.service import PerceptionReport
from tools.follow import LATE_VIEW_S, late_but_not_lost


def report(grabbed_at: float, covered: bool = False) -> PerceptionReport:
    return PerceptionReport(grabbed_at, 1, (), 10.0, covered=covered)


def test_a_view_half_a_second_old_is_late_not_lost() -> None:
    """Three drives in a row ended on views 0.50 to 0.51 s old, on open road."""
    assert late_but_not_lost(report(100.0), 100.51)


def test_a_view_older_than_the_grace_is_lost() -> None:
    assert not late_but_not_lost(report(100.0), 100.0 + LATE_VIEW_S + 0.01)


def test_a_covered_view_or_none_at_all_is_lost() -> None:
    assert not late_but_not_lost(report(100.0, covered=True), 100.2)
    assert not late_but_not_lost(None, 100.2)
