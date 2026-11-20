"""When recordings are enough to take a camera from."""

from __future__ import annotations

from tools.calibrate_camera import enough


def _result(straight: int, shifts: int, spread: float = 10.0) -> dict:
    return {"straight_frames": straight, "shifts": [0.0] * shifts, "horizon_spread_px": spread}


def test_a_daf_xd_s_thirteen_minutes_are_enough() -> None:
    assert enough([_result(69, 154, 11.6)], 7) == ""


def test_what_is_missing_is_said() -> None:
    why = enough([_result(10, 30)], 2)
    assert "straight" in why and "bends" in why and "moving" in why
    assert "looked around" in enough([_result(69, 154, 60.0)], 7)
    assert enough([_result(42, 154, 36.0)], 5) == "", "a rocking cab, many frames: known"
    scania = [_result(14, 145, 33.0), _result(28, 107, 38.8), _result(33, 74, 12.5)]
    assert enough(scania, 5) == "", "known over the three together"
    assert enough([_result(30, 100), _result(30, 0)], 5) == "", "summed over the recordings"
