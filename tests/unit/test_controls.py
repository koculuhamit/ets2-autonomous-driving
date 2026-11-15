"""Tests for the control value types.

`clamp_axis` is the single place out-of-range values and NaN are dealt with, so
every path that can put a number on a pedal depends on it behaving.
"""

from __future__ import annotations

import pytest

from core.common.controls import AXIS_LIMITS, Axis, Button, clamp_axis


def test_axis_limits_cover_every_axis() -> None:
    """An axis without a limit would raise a KeyError inside clamp_axis, on the
    path that was supposed to make bad values safe."""
    assert set(AXIS_LIMITS) == set(Axis)


@pytest.mark.parametrize(
    ("axis", "value", "expected"),
    [
        (Axis.STEER, 2.0, 1.0),
        (Axis.STEER, -2.0, -1.0),
        (Axis.STEER, 0.5, 0.5),
        (Axis.STEER, float("nan"), 0.0),
        (Axis.STEER, float("inf"), 1.0),
        (Axis.STEER, float("-inf"), -1.0),
        (Axis.THROTTLE, -0.5, 0.0),
        (Axis.THROTTLE, 1.5, 1.0),
        (Axis.THROTTLE, float("nan"), 0.0),
        (Axis.BRAKE, float("nan"), 0.0),
        (Axis.CLUTCH, 0.25, 0.25),
    ],
)
def test_clamp_axis(axis: Axis, value: float, expected: float) -> None:
    assert clamp_axis(axis, value) == pytest.approx(expected)


def test_clamp_axis_result_is_always_in_range() -> None:
    for axis in Axis:
        low, high = AXIS_LIMITS[axis]
        for value in (-1e9, -1.0, 0.0, 0.5, 1e9, float("nan"), float("inf")):
            assert low <= clamp_axis(axis, value) <= high


def test_axis_limits_are_immutable() -> None:
    """These are a contract shared with the plugin, not a default to tweak."""
    with pytest.raises(TypeError):
        AXIS_LIMITS[Axis.STEER] = (0.0, 2.0)  # type: ignore[index]


def test_enums_have_no_duplicate_values() -> None:
    """A duplicated StrEnum value silently aliases two members into one."""
    for enum_cls in (Axis, Button):
        values = [member.value for member in enum_cls]
        assert len(values) == len(set(values)), f"{enum_cls.__name__} has duplicate values"
