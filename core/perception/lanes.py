"""The painted lines of the lane the truck is in, found the classic way.

Not for steering -- the map does that better (ADR-0006) -- but for measuring
the camera: on a straight road the two lines bounding the lane meet at the
horizon, and how fast they spread apart below it gives the eye's height
(core.perception.camera).

Paint is a thin bright stripe on darker asphalt. A white top-hat keeps
exactly that and drops broad bright areas -- the sky, a lit wall -- so the
same threshold works at dusk as in daylight. Hough segments through what is
left are sorted by slope: the lane's left line runs down to the left, the
right one down to the right, and of each kind the one whose foot is nearest
the middle of the image bounds the truck's own lane.

It gets arrows, zebra crossings and the edges of cars wrong now and then.
Callers take the median over many frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.perception.camera import GroundCamera, Line

__all__ = ["DEFAULT_ROAD", "LANE_CHECK_M", "RoadRegion", "lane_lines", "lateral_at"]

LANE_CHECK_M = 12.0
"""Where the lane's lines are read off for the truck's place in its lane:
near enough for the paint to be many pixels, far enough to be past the
bonnet in every cab."""


@dataclass(frozen=True, slots=True)
class RoadRegion:
    """Where to look for paint, in pixels of a 1920x1080 frame."""

    top: int
    bottom: int
    left: int
    right: int
    centre: float
    """The column straight ahead: lines are sorted to either side of it."""


DEFAULT_ROAD = RoadRegion(top=440, bottom=690, left=560, right=1450, centre=960.0)

MIN_SLOPE = 0.2
"""|column change per row| below this is too flat to be a lane line ahead."""

MAX_SLOPE = 3.0
"""And above this, nearly horizontal in the image: a stop line or a shadow."""


def lane_lines(
    image: NDArray[np.uint8], road: RoadRegion = DEFAULT_ROAD
) -> tuple[Line | None, Line | None]:
    """(left, right) lines of the truck's lane in a BGR frame; either may be None."""
    import cv2

    grey = cv2.cvtColor(image[road.top : road.bottom, road.left : road.right], cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    hat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    threshold = max(25.0, float(np.percentile(hat, 97)))
    paint = (hat > threshold).astype(np.uint8) * 255
    segments = cv2.HoughLinesP(paint, 1, np.pi / 360, 25, minLineLength=30, maxLineGap=40)
    if segments is None:
        return None, None

    lefts: list[tuple[float, Line]] = []
    rights: list[tuple[float, Line]] = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        if abs(int(y2) - int(y1)) < 10:
            continue
        line = Line.through(x1 + road.left, y1 + road.top, x2 + road.left, y2 + road.top)
        foot = line.column(road.bottom)
        if -MAX_SLOPE < line.b < -MIN_SLOPE and foot < road.centre:
            lefts.append((road.centre - foot, line))
        elif MIN_SLOPE < line.b < MAX_SLOPE and foot > road.centre:
            rights.append((foot - road.centre, line))

    def nearest(found: list[tuple[float, Line]]) -> Line | None:
        return min(found, key=lambda pair: pair[0])[1] if found else None

    return nearest(lefts), nearest(rights)


def lateral_at(line: Line | None, camera: GroundCamera, distance: float = LANE_CHECK_M) -> float:
    """Metres right of the driver's eye, of a lane line `distance` ahead; NaN
    without a line."""
    if line is None:
        return float("nan")
    row = camera.horizon_row + camera.focal_px * camera.height_m / distance
    return camera.lateral(line.column(row), distance)
