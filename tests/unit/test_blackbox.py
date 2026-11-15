"""The last seconds of the view, written out when something goes wrong."""

from __future__ import annotations

import numpy as np

from core.perception.blackbox import BlackBox


def frame(value: int) -> np.ndarray:
    return np.full((108, 192, 3), value, dtype=np.uint8)


def test_the_frames_about_a_moment_are_written_once_the_after_ones_are_in(tmp_path) -> None:
    box = BlackBox(keep_s=1.0, every_s=0.1, after_s=0.5)
    for k in range(40):
        box.add(frame(k), k * 0.05)
        if k == 20:
            box.request("brake-1", 1.0)
        if k == 29:
            assert box.due(k * 0.05) == [], "not before the half second after is in"
    ((label, frames),) = box.due(1.95)
    assert label == "brake-1"
    offsets = [offset for offset, _ in frames]
    assert min(offsets) >= -1.0 and max(offsets) <= 0.5
    assert len(frames) == 16, "every other frame, from a second before to half a second after"
    folder = BlackBox.write(tmp_path, label, frames)
    assert len(list(folder.glob("*.jpg"))) == 16


def test_stopping_writes_what_was_asked_for_at_once() -> None:
    box = BlackBox(keep_s=1.0, every_s=0.1, after_s=2.0)
    for k in range(10):
        box.add(frame(k), k * 0.1)
    box.request("off-path", 0.9)
    assert box.due(1.0) == []
    ((label, frames),) = box.due(1.0, all_of_them=True)
    assert label == "off-path" and len(frames) == 10


def test_old_frames_are_let_go() -> None:
    box = BlackBox(keep_s=1.0, every_s=0.1, after_s=0.5)
    for k in range(100):
        box.add(frame(k % 255), k * 0.1)
    assert len(box._frames) <= 21
