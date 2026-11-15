"""Slowing through toll gates, as tools/follow.py does."""

from __future__ import annotations

import math

import pytest

from tools.follow import (
    BARRIER_AFTER_M,
    BARRIER_SPEED_KMH,
    MS_TO_KMH,
    barrier_speed,
)

SLOW = BARRIER_SPEED_KMH / MS_TO_KMH


def test_through_the_gate_the_truck_goes_slowly() -> None:
    assert barrier_speed([500.0], 500.0) == pytest.approx(SLOW)
    assert barrier_speed([500.0], 500.0 + BARRIER_AFTER_M - 1.0) == pytest.approx(SLOW)


def test_the_slowing_down_begins_in_time_and_gently() -> None:
    """From 80 km/h at 1 m/s^2 takes 220 m; the cap is above 80 until then."""
    assert barrier_speed([500.0], 500.0 - 250.0) > 80 / MS_TO_KMH
    assert barrier_speed([500.0], 500.0 - 100.0) == pytest.approx(math.sqrt(SLOW**2 + 2 * 100.0))


def test_away_from_gates_nothing_is_capped() -> None:
    assert barrier_speed([500.0], 100.0) == math.inf
    assert barrier_speed([500.0], 600.0) == math.inf
    assert barrier_speed([], 0.0) == math.inf
