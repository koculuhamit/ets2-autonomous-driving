"""Finding where the map's lane parts from the painted one."""

from __future__ import annotations

import numpy as np
import pytest

from tools.analysis.lane_check import offsets, stretches


def test_the_painted_middle_is_measured_against_the_path_s() -> None:
    lanes = np.array(
        [
            [0.0, -1.0, 2.6, 0.8],  # painted middle 0.8: on it
            [0.1, -1.0, np.nan, 0.8],  # one line: not a lane
            [0.2, -1.0, 9.0, 0.8],  # 10 m wide: a misread
            [0.3, 0.2, 3.8, 0.8],  # painted middle 2.0: 1.2 m off
        ]
    )
    t, off = offsets(lanes)
    assert t.tolist() == [0.0, 0.3]
    assert off.tolist() == pytest.approx([0.0, 1.2])


def test_only_a_stretch_that_stays_off_is_listed() -> None:
    t = np.arange(0.0, 10.0, 0.1)
    off = np.where((t > 3.0) & (t < 6.0), 1.1, 0.1)
    off[20] = 3.0  # one odd view
    (found,) = stretches(t, off, over=0.6)
    assert found[0] == pytest.approx(3.0, abs=0.3) and found[1] == pytest.approx(6.0, abs=0.3)
    assert found[2] == pytest.approx(1.1)
