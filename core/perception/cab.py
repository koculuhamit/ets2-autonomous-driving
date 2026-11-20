"""Whether the game's camera is in the cab: what perception is calibrated for.

Everything perception measures assumes the interior camera: distances from
where a vehicle meets the road, lights by where they stand in the frame, the
mirrors by their insets. Switched to the chase camera in Bern, the truck's own
trailer filled the view 9.5 m ahead at the truck's speed; it was braked for
at 5 m/s^2, remembered once out of view, and the truck never moved again.

From the cab the lower middle of the frame is the dashboard and the wheel:
the same edges whatever the light (`DASHBOARD`), or, at night, dark. From
outside it is road and trailer. A frame is the cab's if the dashboard's edges
match those of the cab frames recorded (`EDGE_MATCH_MIN`), or most of it is
dark (`DARK_SHARE_MIN`): every one of 911 recorded cab frames was, and no
frame of the chase camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CAB_REFERENCE_FILE",
    "DashboardLearner",
    "RoofLearner",
    "dashboard_edges",
    "in_cab",
    "load_reference",
    "roof_row",
]

CAB_REFERENCE_FILE: Final = Path(__file__).resolve().parents[2] / "configs" / "cab_dashboard.npy"

DASHBOARD: Final = (0.62, 0.98, 0.30, 0.78)
"""(top, bottom, left, right) of the frame, as shares: the dashboard and wheel."""

DARK_BAND: Final = (0.78, 0.95, 0.30, 0.72)
SIZE: Final = (120, 48)
"""(width, height) the dashboard is compared at."""

EDGE_MATCH_MIN: Final = 0.06
"""Cab frames matched the recorded dashboard by 0.11 or more, but for two at
night; the chase camera's by 0.02 at most."""

DARK_V: Final = 60
DARK_SHARE_MIN: Final = 0.5
"""Of the dashboard below the wheel's hub, the share darker than `DARK_V`
that is the cab's at night: 0.79 and more where the edges were faint; 0.31
at most from outside."""


def dashboard_edges(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """The dashboard's edge strength, normalised, `SIZE` (as rows, columns)."""
    height, width = image.shape[:2]
    top, bottom, left, right = DASHBOARD
    band = image[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
    grey = cv2.resize(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA)
    grey = grey.astype(np.float32)
    magnitude = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    return (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)


def in_cab(image: NDArray[np.uint8], reference: NDArray[np.float32]) -> bool:
    """Whether a BGR frame of the game window is seen from the cab."""
    if float((dashboard_edges(image) * reference).mean()) >= EDGE_MATCH_MIN:
        return True
    height, width = image.shape[:2]
    top, bottom, left, right = DARK_BAND
    band = image[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
    value = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[..., 2]
    return float((value < DARK_V).mean()) >= DARK_SHARE_MIN


LEARN_FRAMES: Final = 60
LEARN_EVERY_S: Final = 0.3
"""A truck without a dashboard of its own learns it from this many frames,
this far apart: some 18 s of the drive's start (`DashboardLearner`)."""

LEARN_MATCH: Final = 0.12
LEARN_AGREE: Final = 0.8
LEARN_HALVES: Final = 0.2
"""The frames learned from agree when this share of them match their mean
by `LEARN_MATCH`, twice what a cab frame needs, and the mean of their first
half matches that of their second by `LEARN_HALVES`: a view that changed
half way -- the camera switched -- does not. Halves of the cab frames of
three drives (Scania at night, two Volvos) matched by 0.27 and more; the
Volvo's cab and its chase camera by -0.16."""


class DashboardLearner:
    """The dashboard of a truck that has none recorded, from what the
    drive's first frames have in common.

    Another truck's dashboard does not do: a Scania R at night matched the
    recorded one too faintly and was lit too brightly for the dark test, and
    five drives stopped as if the camera were out of the cab while it was in.
    The frames are taken to be the cab's -- the driver is told to drive from
    it -- and the check is off until they agree."""

    def __init__(self, frames: int = LEARN_FRAMES, every_s: float = LEARN_EVERY_S) -> None:
        self._frames = frames
        self._every_s = every_s
        self._last = -np.inf
        self._edges: list[NDArray[np.float32]] = []

    def offer(self, image: NDArray[np.uint8], now: float) -> NDArray[np.float32] | None:
        """The dashboard's edges, normalised, once learned; else None."""
        if now - self._last < self._every_s * 0.999:  # a frame on the mark counts
            return None
        self._last = now
        self._edges.append(dashboard_edges(image))
        if len(self._edges) < self._frames:
            return None
        mean = _normalised(np.mean(self._edges, axis=0))
        half = len(self._edges) // 2
        halves = float(
            (
                _normalised(np.mean(self._edges[:half], axis=0))
                * _normalised(np.mean(self._edges[half:], axis=0))
            ).mean()
        )
        scores = np.array([float((edges * mean).mean()) for edges in self._edges])
        if halves >= LEARN_HALVES and float((scores >= LEARN_MATCH).mean()) >= LEARN_AGREE:
            return mean.astype(np.float32)
        # Not alike: the older half dropped, and on.
        self._edges = self._edges[self._frames // 2 :]
        return None


ROOF_BAND: Final = (0, 400, 700 / 1920, 1500 / 1920)
"""(first row, last row, left, right) of a 1080-row frame looked in for the
cab's roof: the top of the frame, its middle columns."""

ROOF_CHANGING: Final = 25.0
ROOF_STILL: Final = 15.0
"""A row of the windscreen changes over a drive by `ROOF_CHANGING` (the
median over its columns of each pixel's spread, grey levels) or more; the
roof's, still, by less than `ROOF_STILL`."""

ROOF_FRAMES: Final = 120
ROOF_EVERY_S: Final = 0.5
"""A minute of the drive, a frame every half second (`RoofLearner`)."""

ROOF_MARGIN_PX: Final = 8
"""The windscreen taken from this far below the roof's edge."""


def roof_row(frames: NDArray[np.uint8]) -> int | None:
    """The first row of a 1080-row frame below the cab's roof: where the
    view through the windscreen begins, from `frames` (n, rows, cols) grey
    crops of `ROOF_BAND`. The roof stays still over a drive, the view moves.
    None when no clean edge shows: a minute standing, or all one road.

    From recorded drives: an Actros 267, a Volvo 248 -- the default view's
    250 -- and a DAF XD 162: its overhead lamps went behind the default
    view's top at 19 m and it stood at green lights it did not see."""
    spread = np.median(frames.astype(np.float32).std(axis=0), axis=1)
    rows = len(spread)
    for row in range(40, rows - 40):
        if (
            spread[row] >= ROOF_CHANGING
            and np.median(spread[row : row + 40]) >= ROOF_CHANGING
            and np.median(spread[row - 40 : row]) < ROOF_STILL
        ):
            return ROOF_BAND[0] + row
    return None


class RoofLearner:
    """Where a truck's windscreen begins, from a minute of its drive."""

    def __init__(self, frames: int = ROOF_FRAMES, every_s: float = ROOF_EVERY_S) -> None:
        self._frames = frames
        self._every_s = every_s
        self._last = -np.inf
        self._kept: list[NDArray[np.uint8]] = []

    def offer(self, image: NDArray[np.uint8], now: float) -> int | None:
        """The windscreen's first row, once learned; else None."""
        if now - self._last < self._every_s * 0.999:
            return None
        self._last = now
        first, last, left, right = ROOF_BAND
        width = image.shape[1]
        band = image[first:last, int(left * width) : int(right * width)]
        self._kept.append(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)[:, ::4])
        if len(self._kept) < self._frames:
            return None
        row = roof_row(np.stack(self._kept))
        if row is None:
            self._kept = self._kept[self._frames // 2 :]
            return None
        return row + ROOF_MARGIN_PX


def _normalised(edges: NDArray[np.floating]) -> NDArray[np.float32]:
    return ((edges - edges.mean()) / (edges.std() + 1e-6)).astype(np.float32)


def load_reference(path: Path = CAB_REFERENCE_FILE) -> NDArray[np.float32] | None:
    """The recorded dashboard's edges (tools/calibration/cab_reference.py);
    None where there is none, and no check is made."""
    return np.load(path).astype(np.float32) if path.is_file() else None
