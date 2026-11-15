"""Where a slow turn of tools/follow.py's loop went."""

from __future__ import annotations

import time

from tools.follow import SLOW_TURN_S, TurnClock


def test_a_slow_turn_is_told_by_the_part_that_took_it() -> None:
    clock = TurnClock()
    began = time.perf_counter()
    clock.mark("reading the telemetry")
    time.sleep(SLOW_TURN_S + 0.05)
    clock.mark("the GPS")
    now = time.perf_counter()
    said = clock.turn(now, now - began)
    assert said is not None
    assert "of it the GPS" in said
    assert "its thread on the CPU" in said


def test_a_quick_turn_is_not() -> None:
    clock = TurnClock()
    began = time.perf_counter()
    clock.mark("reading the telemetry")
    now = time.perf_counter()
    assert clock.turn(now, now - began) is None
