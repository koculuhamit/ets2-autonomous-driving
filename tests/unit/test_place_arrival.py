"""Following the red line to a place marked on the map: the game's distance
says when the place is come up to."""

from __future__ import annotations

import pytest

from tools.follow import PLACE_ARRIVED_NAV_M, PLACE_NAV_SETTLE_S, PlaceArrival


def test_far_off_the_place_says_nothing() -> None:
    assert PlaceArrival().to_go(1700.0, 0.0, 3.0) is None


def test_near_the_place_for_a_moment_it_is_how_far_on_the_path() -> None:
    arrival = PlaceArrival()
    assert arrival.to_go(120.0, 10.0, 3.0) is None, "not at the first reading"
    to_go = arrival.to_go(90.0, 10.0 + PLACE_NAV_SETTLE_S, 3.0)
    assert to_go == pytest.approx(30.0), "the game's 90 m, a third of them on the path"


def test_the_game_s_distance_gone_to_0_while_it_plans_anew_is_not_arriving() -> None:
    """Across every drive recorded, it read 0 for 0.06 s at most each time
    the game planned anew."""
    arrival = PlaceArrival()
    assert arrival.to_go(0.0, 0.0, 19.0) is None
    assert arrival.to_go(0.0, 0.06, 19.0) is None
    assert arrival.to_go(48_000.0, 0.08, 19.0) is None
    assert arrival.to_go(0.0, 0.5, 19.0) is None, "a new moment begins anew"
    assert PLACE_ARRIVED_NAV_M > 0.0
