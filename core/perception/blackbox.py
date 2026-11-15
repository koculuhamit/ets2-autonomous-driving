"""The last few seconds of the view, kept to be looked at when something goes wrong.

A drive is recorded as telemetry and commands, not pictures: an hour of
frames is gigabytes. But when the truck brakes hard, leaves its path or is
hit, what it saw is the one thing that says why -- out of Folkestone a car
was hit at 60 km/h with no vehicle ever seen ahead, and nothing was left to
look at. So the perception process keeps the last `KEEP_S` of frames,
shrunk and JPEG-encoded, and on request writes them out with `AFTER_S` more.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["BlackBox"]

KEEP_S: Final = 4.0
EVERY_S: Final = 0.2
AFTER_S: Final = 1.5
SCALE: Final = 0.5
QUALITY: Final = 80


@dataclass
class BlackBox:
    """Frames of the last `keep_s`, one every `every_s`, written out on request."""

    keep_s: float = KEEP_S
    every_s: float = EVERY_S
    after_s: float = AFTER_S
    scale: float = SCALE
    _frames: deque = field(default_factory=deque)
    """(time, JPEG bytes)."""

    _pending: list[tuple[str, float, float]] = field(default_factory=list)
    """(label, asked at, write at)."""

    def add(self, image: NDArray[np.uint8], now: float) -> None:
        if self._frames and now - self._frames[-1][0] < self.every_s - 1e-3:
            return
        small = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if ok:
            self._frames.append((now, jpeg.tobytes()))
        # Kept a little past what a request needs, in case it is written late.
        while self._frames and now - self._frames[0][0] > self.keep_s + 2.0 * self.after_s:
            self._frames.popleft()

    def request(self, label: str, now: float) -> None:
        """Write out the frames about `now`, once `after_s` more are in."""
        self._pending.append((label, now, now + self.after_s))

    def due(self, now: float, *, all_of_them: bool = False) -> list[tuple[str, list]]:
        """The requests ready to be written -- all of them, when stopping --
        each with its frames: from `keep_s` before it was asked to `after_s` after."""
        ready, waiting = [], []
        for label, asked, write_at in self._pending:
            if all_of_them or now >= write_at:
                frames = [
                    (t, jpeg)
                    for t, jpeg in self._frames
                    if asked - self.keep_s <= t <= asked + self.after_s
                ]
                ready.append((label, [(t - asked, jpeg) for t, jpeg in frames]))
            else:
                waiting.append((label, asked, write_at))
        self._pending = waiting
        return ready

    @staticmethod
    def write(directory: Path, label: str, frames: list[tuple[float, bytes]]) -> Path:
        """Each frame as `<label>/<+seconds from the request>.jpg`."""
        folder = Path(directory) / label
        folder.mkdir(parents=True, exist_ok=True)
        for offset, jpeg in frames:
            (folder / f"{offset:+07.2f}.jpg").write_bytes(jpeg)
        return folder
